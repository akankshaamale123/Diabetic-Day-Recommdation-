# app/storage/user_db.py
"""
Real users: login, journey start date, daily meals (from photos).
CGMacros stays separate — it is only TRAINING data for the response model.

Uses the same SQLite file as db.py (db.DB_PATH) but its own tables.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import db

NUTRIENT_COLS = ["calories_kcal", "carbs_g", "sugar_g", "fiber_g",
                 "protein_g", "fat_g"]


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
    finally:
        conn.close()


def init_user_tables() -> None:
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


# ---------------- users ----------------
def get_or_create_user(username: str, diabetes_type: str = "type2",
                       baseline_mgdl: float = 130.0,
                       created_at: str | None = None) -> dict:
    """First time = account created; `created_at` (default now) is day 1 of the journey."""
    username = username.strip().lower()
    if not username:
        raise ValueError("Username is required.")
    init_user_tables()
    with _session() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO users(username, created_at, diabetes_type, baseline_mgdl) "
                "VALUES (?,?,?,?)",
                (username,
                 created_at or datetime.now().isoformat(timespec="seconds"),
                 diabetes_type, float(baseline_mgdl)))
            row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row)


def update_profile(user_id: int, diabetes_type: str, baseline_mgdl: float) -> None:
    with _session() as c:
        c.execute("UPDATE users SET diabetes_type=?, baseline_mgdl=? WHERE id=?",
                  (diabetes_type, float(baseline_mgdl), user_id))


def login_date(user: dict) -> pd.Timestamp:
    return pd.Timestamp(user["created_at"]).normalize()


# ---------------- meals ----------------
def add_meal(user_id: int, timestamp, meal_type: str,
             totals: dict, foods: list | None = None) -> int:
    ts = pd.Timestamp(timestamp).isoformat()
    vals = [float(totals.get(k) or 0.0) for k in NUTRIENT_COLS]
    with _session() as c:
        cur = c.execute(
            "INSERT INTO user_meals(user_id,timestamp,meal_type,foods_json,"
            "calories_kcal,carbs_g,sugar_g,fiber_g,protein_g,fat_g) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user_id, ts, meal_type, json.dumps(foods or [], ensure_ascii=False), *vals))
        return cur.lastrowid


def load_meals(user_id: int, start=None, end=None) -> pd.DataFrame:
    q, p = "SELECT * FROM user_meals WHERE user_id=?", [user_id]
    if start is not None:
        q += " AND timestamp >= ?"; p.append(pd.Timestamp(start).isoformat())
    if end is not None:
        q += " AND timestamp <= ?"; p.append(pd.Timestamp(end).isoformat())
    q += " ORDER BY timestamp"
    with _session() as c:
        df = pd.read_sql_query(q, c, params=p)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def delete_meal(meal_id: int) -> None:
    with _session() as c:
        c.execute("DELETE FROM user_meals WHERE id=?", (meal_id,))


# ---------------- optional measured glucose ----------------
def add_glucose(user_id: int, timestamp, value: float) -> None:
    with _session() as c:
        c.execute("INSERT INTO user_glucose(user_id,timestamp,glucose_mgdl) VALUES (?,?,?)",
                  (user_id, pd.Timestamp(timestamp).isoformat(), float(value)))


def load_glucose(user_id: int, start=None, end=None) -> pd.DataFrame:
    q, p = "SELECT timestamp, glucose_mgdl FROM user_glucose WHERE user_id=?", [user_id]
    if start is not None:
        q += " AND timestamp >= ?"; p.append(pd.Timestamp(start).isoformat())
    if end is not None:
        q += " AND timestamp <= ?"; p.append(pd.Timestamp(end).isoformat())
    q += " ORDER BY timestamp"
    with _session() as c:
        df = pd.read_sql_query(q, c, params=p)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df