# app/storage/db.py
"""
SQLite persistence for meals (from Engine 3) and glucose readings (CGM/manual).
Everything the analysis layer reads comes from here.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "health.db"


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
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
    rows = []
    for item in nutrition_result.get("items", []):
        n = item.get("nutrition") or {}
        rows.append((
            ts,
            item.get("requested_food"),
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
        return 0

    with _conn() as c:
        c.executemany("""
            INSERT INTO meals (timestamp, food_name, portion_grams,
                calories_kcal, carbs_g, sugar_g, protein_g, fat_g,
                fiber_g, cholesterol_mg, calcium_mg, source, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


# ----------------------------------------------------------------
# Write: glucose readings (CGM or manual)
# ----------------------------------------------------------------
def save_glucose_readings(readings: list[dict]) -> int:
    """
    readings = [{'timestamp': '2025-10-10T08:00', 'glucose_mgdl': 92.0}, ...]
    """
    rows = [(str(pd.Timestamp(r["timestamp"])), float(r["glucose_mgdl"]))
            for r in readings
            if r.get("glucose_mgdl") is not None]
    if not rows:
        return 0
    with _conn() as c:
        c.executemany(
            "INSERT INTO glucose (timestamp, glucose_mgdl) VALUES (?,?)",
            rows,
        )
    return len(rows)


def import_glucose_csv(csv_path: str | Path,
                       ts_col: str = "timestamp",
                       val_col: str = "glucose_mgdl") -> int:
    """Import a CGM export (Libre / Dexcom / generic CSV)."""
    df = pd.read_csv(csv_path)
    if ts_col not in df or val_col not in df:
        raise ValueError(f"CSV must contain columns '{ts_col}' and '{val_col}'")
    readings = df[[ts_col, val_col]].rename(
        columns={ts_col: "timestamp", val_col: "glucose_mgdl"}
    ).to_dict(orient="records")
    return save_glucose_readings(readings)


# ----------------------------------------------------------------
# Read: everything the analysis layer needs comes from these
# ----------------------------------------------------------------
def load_glucose(start=None, end=None) -> pd.DataFrame:
    q = "SELECT timestamp, glucose_mgdl FROM glucose"
    params: list = []
    if start is not None and end is not None:
        q += " WHERE timestamp BETWEEN ? AND ?"
        params = [str(pd.Timestamp(start)), str(pd.Timestamp(end))]
    q += " ORDER BY timestamp"
    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_meals(start=None, end=None) -> pd.DataFrame:
    q = "SELECT * FROM meals"
    params: list = []
    if start is not None and end is not None:
        q += " WHERE timestamp BETWEEN ? AND ?"
        params = [str(pd.Timestamp(start)), str(pd.Timestamp(end))]
    q += " ORDER BY timestamp"
    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def available_days() -> list[str]:
    """Which past dates actually have data — used to build a day selector."""
    with _conn() as c:
        df = pd.read_sql_query("""
            SELECT DISTINCT date(timestamp) AS day FROM glucose
            UNION
            SELECT DISTINCT date(timestamp) AS day FROM meals
            ORDER BY day
        """, c)
    return df["day"].tolist()


def clear_all() -> None:
    with _conn() as c:
        c.executescript("DELETE FROM meals; DELETE FROM glucose;")