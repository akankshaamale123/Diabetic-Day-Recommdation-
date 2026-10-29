# app/storage/user_db.py
"""
Real users: login, journey start date, daily meals (from photos).
CGMacros stays separate — it is only TRAINING data for the response model.

Uses the same SQLite file as db.py (db.DB_PATH) but its own tables.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import db

logger = logging.getLogger(__name__)

NUTRIENT_COLS = [
    "calories_kcal", "carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g", 
    "cholesterol_mg", "calcium_mg"
] 


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session():
    """Commit on success, always close (sqlite's own `with` never closes)."""
    conn = _conn()
    try:
        with conn:
            yield conn
    except Exception:
        # `with conn:` already rolled back — just record the failure
        logger.exception("_session(): transaction failed — rolled back")
        raise
    finally:
        conn.close()


def init_user_tables() -> None:
    logger.info("init_user_tables(): ensuring user tables exist at %s",
                db.DB_PATH)
    try:
        with _session() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                created_at    TEXT NOT NULL,          -- journey start = first login
                diabetes_type TEXT DEFAULT 'type2',
                baseline_mgdl REAL DEFAULT 130        -- typical fasting glucose
            );
            CREATE TABLE IF NOT EXISTS user_meals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                timestamp     TEXT NOT NULL,
                meal_type     TEXT NOT NULL,
                foods_json    TEXT,
                calories_kcal REAL, carbs_g REAL, sugar_g REAL,
                fiber_g REAL, protein_g REAL, fat_g REAL,
                cholesterol_mg REAL, calcium_mg REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_user_meals ON user_meals(user_id, timestamp);
            CREATE TABLE IF NOT EXISTS user_glucose (      -- optional: real CGM/finger-prick
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                glucose_mgdl REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_glucose ON user_glucose(user_id, timestamp);
            """)
    except Exception:
        logger.exception("init_user_tables(): schema creation failed")
        raise
    logger.info("init_user_tables(): user schema ready")


# ---------------- users ----------------
def get_or_create_user(username: str, diabetes_type: str = "type2",
                       baseline_mgdl: float = 130.0,
                       created_at: str | None = None) -> dict:
    """First time = account created; `created_at` (default now) is day 1 of the journey."""
    username = username.strip().lower()
    if not username:
        logger.error("get_or_create_user(): empty username rejected")
        raise ValueError("Username is required.")

    logger.info("get_or_create_user(): username=%r, diabetes_type=%s, "
                "baseline_mgdl=%.1f", username, diabetes_type,
                float(baseline_mgdl))

    init_user_tables()

    try:
        with _session() as c:
            row = c.execute(
                "SELECT * FROM users WHERE username=?", (username,)
            ).fetchone()

            if row is None:
                ts = created_at or datetime.now().isoformat(timespec="seconds")
                c.execute(
                    "INSERT INTO users(username, created_at, diabetes_type, baseline_mgdl) "
                    "VALUES (?,?,?,?)",
                    (username, ts, diabetes_type, float(baseline_mgdl)))
                row = c.execute(
                    "SELECT * FROM users WHERE username=?", (username,)
                ).fetchone()
                logger.info("get_or_create_user(): created user id=%s "
                            "username=%r created_at=%s",
                            row["id"], row["username"], row["created_at"])
            else:
                logger.debug("get_or_create_user(): existing user id=%s "
                             "username=%r created_at=%s",
                             row["id"], row["username"], row["created_at"])
    except Exception:
        logger.exception("get_or_create_user(): DB operation failed for "
                         "username=%r", username)
        raise

    return dict(row)


def update_profile(user_id: int, diabetes_type: str, baseline_mgdl: float) -> None:
    logger.info("update_profile(): user_id=%s, diabetes_type=%s, "
                "baseline_mgdl=%.1f", user_id, diabetes_type,
                float(baseline_mgdl))
    try:
        with _session() as c:
            cur = c.execute(
                "UPDATE users SET diabetes_type=?, baseline_mgdl=? WHERE id=?",
                (diabetes_type, float(baseline_mgdl), user_id))
            if cur.rowcount == 0:
                logger.warning("update_profile(): no user with id=%s — "
                               "nothing updated", user_id)
            else:
                logger.info("update_profile(): updated user_id=%s", user_id)
    except Exception:
        logger.exception("update_profile(): DB update failed for user_id=%s",
                         user_id)
        raise


def login_date(user: dict) -> pd.Timestamp:
    created_at = user.get("created_at")

    if not created_at:
        logger.warning(
            "login_date(): created_at is missing for user_id=%s",
            user.get("id")
        )
        return pd.Timestamp.today().normalize()

    try:
        dt = pd.Timestamp(created_at).normalize()

        if pd.isna(dt):
            raise ValueError("created_at contains an invalid date")

    except (ValueError, TypeError, OverflowError):
        logger.exception(
            "login_date(): invalid created_at=%r for user_id=%s",
            created_at,
            user.get("id")
        )
        return pd.Timestamp.today().normalize()

    logger.debug(
        "login_date(): user_id=%s → %s",
        user.get("id"),
        dt.date()
    )

    return dt


# ---------------- meals ----------------
def add_meal(user_id: int, timestamp, meal_type: str,
             totals: dict, foods: list | None = None) -> int:
    try:
        ts = pd.Timestamp(timestamp).isoformat()
    except Exception:
        logger.exception("add_meal(): invalid timestamp=%r for user_id=%s",
                         timestamp, user_id)
        raise

    vals = [float(totals.get(k) or 0.0) for k in NUTRIENT_COLS]
    logger.info("add_meal(): user_id=%s, timestamp=%s, meal_type=%s, "
                "foods=%d, kcal=%.0f, carbs=%.1fg",
                user_id, ts, meal_type, len(foods or []),
                vals[0], vals[1])

    try:
        with _session() as c:
            cur = c.execute(
                "INSERT INTO user_meals(user_id,timestamp,meal_type,foods_json,"
                "calories_kcal,carbs_g,sugar_g,fiber_g,protein_g,fat_g,"
                "cholesterol_mg,calcium_mg) " 
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, ts, meal_type,
                 json.dumps(foods or [], ensure_ascii=False), *vals))
            meal_id = cur.lastrowid
    except Exception:
        logger.exception("add_meal(): insert failed for user_id=%s, ts=%s",
                         user_id, ts)
        raise

    logger.info("add_meal(): inserted meal id=%s for user_id=%s",
                meal_id, user_id)
    return meal_id


def load_meals(user_id: int, start=None, end=None) -> pd.DataFrame:
    bounded_start = start is not None
    bounded_end = end is not None
    logger.debug("load_meals(): user_id=%s, start=%s, end=%s", user_id, start, end)

    q, p = "SELECT * FROM user_meals WHERE user_id=?", [user_id]
    
    # FIX: Use datetime() to ignore the difference between 'T' and 'space'
    if bounded_start:
        q += " AND datetime(timestamp) >= datetime(?)"
        p.append(pd.Timestamp(start).isoformat())
    if bounded_end:
        q += " AND datetime(timestamp) <= datetime(?)"
        p.append(pd.Timestamp(end).isoformat())
        
    q += " ORDER BY timestamp"

    try:
        with _session() as c:
            df = pd.read_sql_query(q, c, params=p)
    except Exception:
        logger.exception("load_meals(): query failed for user_id=%s", user_id)
        raise

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")

    if df.empty:
        logger.warning("load_meals(): no meals for user_id=%s between %s and %s", user_id, start, end)
    else:
        logger.debug("load_meals(): user_id=%s → %d meal row(s)", user_id, len(df))
    return df


def delete_meal(meal_id: int) -> None:
    logger.info("delete_meal(): meal_id=%s", meal_id)
    try:
        with _session() as c:
            cur = c.execute("DELETE FROM user_meals WHERE id=?", (meal_id,))
            if cur.rowcount == 0:
                logger.warning("delete_meal(): no meal with id=%s — nothing "
                               "deleted", meal_id)
            else:
                logger.info("delete_meal(): deleted meal_id=%s", meal_id)
    except Exception:
        logger.exception("delete_meal(): delete failed for meal_id=%s",
                         meal_id)
        raise


# ---------------- optional measured glucose ----------------
def add_glucose(user_id: int, timestamp, value: float) -> None:
    try:
        ts = pd.Timestamp(timestamp).isoformat()
    except Exception:
        logger.exception("add_glucose(): invalid timestamp=%r for user_id=%s",
                         timestamp, user_id)
        raise

    logger.info("add_glucose(): user_id=%s, timestamp=%s, value=%.1f mg/dL",
                user_id, ts, float(value))
    try:
        with _session() as c:
            c.execute(
                "INSERT INTO user_glucose(user_id,timestamp,glucose_mgdl) "
                "VALUES (?,?,?)",
                (user_id, ts, float(value)))
    except Exception:
        logger.exception("add_glucose(): insert failed for user_id=%s",
                         user_id)
        raise


def load_glucose(user_id: int, start=None, end=None) -> pd.DataFrame:
    logger.debug("load_glucose(): user_id=%s, start=%s, end=%s", user_id, start, end)

    q, p = ("SELECT timestamp, glucose_mgdl FROM user_glucose WHERE user_id=?", [user_id])
    
    # FIX: Use datetime() here as well
    if start is not None:
        q += " AND datetime(timestamp) >= datetime(?)"
        p.append(pd.Timestamp(start).isoformat())
    if end is not None:
        q += " AND datetime(timestamp) <= datetime(?)"
        p.append(pd.Timestamp(end).isoformat())
        
    q += " ORDER BY timestamp"

    try:
        with _session() as c:
            df = pd.read_sql_query(q, c, params=p)
    except Exception:
        logger.exception("load_glucose(): query failed for user_id=%s", user_id)
        raise

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")

    if df.empty:
        logger.warning("load_glucose(): no readings for user_id=%s between %s and %s", user_id, start, end)
    else:
        logger.debug("load_glucose(): user_id=%s → %d reading(s)", user_id, len(df))
    return df