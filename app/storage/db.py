# app/storage/db.py
"""
SQLite persistence for meals (from Engine 3) and glucose readings (CGM/manual).
Everything the analysis layer reads comes from here.
"""
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "health.db"


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("_conn(): opening SQLite connection to %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
        logger.debug("_conn(): commit succeeded")
    except Exception:
        logger.exception("_conn(): operation failed — rolling back")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.debug("_conn(): connection closed")


def init_db() -> None:
    logger.info("init_db(): ensuring tables exist at %s", DB_PATH)
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS meals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            food_name    TEXT NOT NULL,
            portion_grams REAL,
            calories_kcal REAL,
            carbs_g      REAL,
            sugar_g      REAL,
            protein_g    REAL,
            fat_g        REAL,
            fiber_g      REAL,
            cholesterol_mg REAL,
            calcium_mg   REAL,
            source       TEXT,       -- 'database' | 'llm_estimate'
            confidence   REAL
        );

        CREATE TABLE IF NOT EXISTS glucose (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            glucose_mgdl REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_meals_ts   ON meals(timestamp);
        CREATE INDEX IF NOT EXISTS idx_glucose_ts ON glucose(timestamp);
        """)
    logger.info("init_db(): schema ready")


# ----------------------------------------------------------------
# Write: called by Engine 3 after nutrition is calculated
# ----------------------------------------------------------------
def save_meal_analysis(nutrition_result: dict,
                       timestamp: str | pd.Timestamp) -> int:
    """
    Persist the per‑item nutrition from calculate_nutrition().
    The `timestamp` is when the meal was eaten (user supplied / photo time).
    """
    ts = str(pd.Timestamp(timestamp))
    logger.info("save_meal_analysis(): timestamp=%s, items=%d",
                ts, len(nutrition_result.get("items", []) or []))

    rows = []
    skipped = 0
    for item in nutrition_result.get("items", []):
        n = item.get("nutrition") or {}
        food = item.get("requested_food")
        if not food:
            skipped += 1
            logger.debug("save_meal_analysis(): skipping item with no food "
                         "name (source=%s)", item.get("source"))
            continue

        rows.append((
            ts,
            food,
            item.get("portion_grams"),
            n.get("calories_kcal"),
            n.get("carbs_g"),
            n.get("sugar_g"),
            n.get("protein_g"),
            n.get("fat_g"),
            n.get("fiber_g"),
            n.get("cholesterol_mg"),
            n.get("calcium_mg"),
            item.get("source"),
            item.get("confidence"),
        ))

    if not rows:
        logger.warning("save_meal_analysis(): nothing to insert for ts=%s "
                       "(%d item(s) skipped) — returning 0", ts, skipped)
        return 0

    try:
        with _conn() as c:
            c.executemany("""
                INSERT INTO meals (timestamp, food_name, portion_grams,
                    calories_kcal, carbs_g, sugar_g, protein_g, fat_g,
                    fiber_g, cholesterol_mg, calcium_mg, source, confidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
    except Exception:
        logger.exception("save_meal_analysis(): insert failed for ts=%s "
                         "(%d row(s))", ts, len(rows))
        raise

    logger.info("save_meal_analysis(): inserted %d row(s) for ts=%s%s",
                len(rows), ts,
                f" (skipped {skipped} without a food name)" if skipped else "")
    return len(rows)


# ----------------------------------------------------------------
# Write: glucose readings (CGM or manual)
# ----------------------------------------------------------------
def save_glucose_readings(readings: list[dict]) -> int:
    """
    readings = [{'timestamp': '2025-10-10T08:00', 'glucose_mgdl': 92.0}, ...]
    """
    logger.info("save_glucose_readings(): %d input reading(s)",
                len(readings or []))

    rows = []
    skipped_invalid = 0
    skipped_unparseable_ts = 0
    for r in readings or []:
        if r.get("glucose_mgdl") is None:
            skipped_invalid += 1
            continue
        try:
            ts_str = str(pd.Timestamp(r["timestamp"]))
        except Exception:
            skipped_unparseable_ts += 1
            logger.debug("save_glucose_readings(): unparseable timestamp %r "
                         "— skipping", r.get("timestamp"))
            continue
        rows.append((ts_str, float(r["glucose_mgdl"])))

    if not rows:
        logger.warning("save_glucose_readings(): no valid readings to insert "
                       "(%d skipped for missing glucose, %d for bad timestamp)",
                       skipped_invalid, skipped_unparseable_ts)
        return 0

    if skipped_invalid or skipped_unparseable_ts:
        logger.warning("save_glucose_readings(): inserting %d/%d reading(s); "
                       "skipped %d missing-value, %d bad-timestamp",
                       len(rows), len(readings or []),
                       skipped_invalid, skipped_unparseable_ts)

    try:
        with _conn() as c:
            c.executemany(
                "INSERT INTO glucose (timestamp, glucose_mgdl) VALUES (?,?)",
                rows,
            )
    except Exception:
        logger.exception("save_glucose_readings(): insert failed (%d row(s))",
                         len(rows))
        raise

    logger.info("save_glucose_readings(): inserted %d row(s)", len(rows))
    return len(rows)


def import_glucose_csv(csv_path: str | Path,
                       ts_col: str = "timestamp",
                       val_col: str = "glucose_mgdl") -> int:
    """Import a CGM export (Libre / Dexcom / generic CSV)."""
    logger.info("import_glucose_csv(): path=%s, ts_col=%r, val_col=%r",
                csv_path, ts_col, val_col)

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("import_glucose_csv(): failed to read %s", csv_path)
        raise

    logger.debug("import_glucose_csv(): read %d row(s), columns=%s",
                 len(df), list(df.columns))

    if ts_col not in df or val_col not in df:
        logger.error("import_glucose_csv(): expected columns %r and %r not "
                     "both present in %s (got %s)",
                     ts_col, val_col, csv_path, list(df.columns))
        raise ValueError(f"CSV must contain columns '{ts_col}' and '{val_col}'")

    readings = df[[ts_col, val_col]].rename(
        columns={ts_col: "timestamp", val_col: "glucose_mgdl"}
    ).to_dict(orient="records")

    inserted = save_glucose_readings(readings)
    logger.info("import_glucose_csv(): imported %d/%d reading(s) from %s",
                inserted, len(df), csv_path)
    return inserted


# ----------------------------------------------------------------
# Read: everything the analysis layer needs comes from these
# ----------------------------------------------------------------
def load_glucose(start=None, end=None) -> pd.DataFrame:
    bounded = start is not None and end is not None
    logger.debug("load_glucose(): start=%s end=%s (bounded=%s)",
                 start, end, bounded)

    q = "SELECT timestamp, glucose_mgdl FROM glucose"
    params: list = []
    if bounded:
        q += " WHERE timestamp BETWEEN ? AND ?"
        params = [str(pd.Timestamp(start)), str(pd.Timestamp(end))]
    q += " ORDER BY timestamp"

    try:
        with _conn() as c:
            df = pd.read_sql_query(q, c, params=params)
    except Exception:
        logger.exception("load_glucose(): query failed (start=%s, end=%s)",
                         start, end)
        raise

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df.empty:
        logger.warning("load_glucose(): no readings%s",
                       f" between {start} and {end}" if bounded else "")
    else:
        logger.debug("load_glucose(): loaded %d reading(s)%s",
                     len(df), " (bounded)" if bounded else "")
    return df


def load_meals(start=None, end=None) -> pd.DataFrame:
    bounded = start is not None and end is not None
    logger.debug("load_meals(): start=%s end=%s (bounded=%s)",
                 start, end, bounded)

    q = "SELECT * FROM meals"
    params: list = []
    if bounded:
        q += " WHERE timestamp BETWEEN ? AND ?"
        params = [str(pd.Timestamp(start)), str(pd.Timestamp(end))]
    q += " ORDER BY timestamp"

    try:
        with _conn() as c:
            df = pd.read_sql_query(q, c, params=params)
    except Exception:
        logger.exception("load_meals(): query failed (start=%s, end=%s)",
                         start, end)
        raise

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df.empty:
        logger.warning("load_meals(): no meals%s",
                       f" between {start} and {end}" if bounded else "")
    else:
        logger.debug("load_meals(): loaded %d meal row(s)%s",
                     len(df), " (bounded)" if bounded else "")
    return df


def available_days() -> list[str]:
    """Which past dates actually have data — used to build a day selector."""
    try:
        with _conn() as c:
            df = pd.read_sql_query("""
                SELECT DISTINCT date(timestamp) AS day FROM glucose
                UNION
                SELECT DISTINCT date(timestamp) AS day FROM meals
                ORDER BY day
            """, c)
    except Exception:
        logger.exception("available_days(): query failed")
        raise

    days = df["day"].tolist()
    logger.info("available_days(): %d day(s) with data%s",
                len(days), f" ({days[0]} → {days[-1]})" if days else "")
    if not days:
        logger.warning("available_days(): no data in DB — day selector will be "
                       "empty")
    return days


def clear_all() -> None:
    logger.warning("clear_all(): deleting all meals and glucose rows from %s",
                   DB_PATH)
    try:
        with _conn() as c:
            c.executescript("DELETE FROM meals; DELETE FROM glucose;")
    except Exception:
        logger.exception("clear_all(): delete failed")
        raise
    logger.info("clear_all(): database cleared")