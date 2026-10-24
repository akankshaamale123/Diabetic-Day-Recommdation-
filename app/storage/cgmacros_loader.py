# app/storage/cgmacros_loader.py
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from . import db

logger = logging.getLogger(__name__)


def _to_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.to_datetime(s, unit="s", errors="coerce")
    if out.isna().all():
        logger.debug("_to_datetime(): numeric epoch conversion produced all-"
                     "NaT — retrying as string timestamps")
        out = pd.to_datetime(series, errors="coerce")
    n_valid = int(out.notna().sum())
    logger.debug("_to_datetime(): parsed %d/%d timestamp(s)",
                 n_valid, len(out))
    return out


def discover_subjects(root: str | Path) -> list[str]:
    root = Path(root)
    logger.info("discover_subjects(): root=%s", root)

    if not root.is_dir():
        logger.error("discover_subjects(): root not found: %s", root)
        raise FileNotFoundError(f"CGMacros root not found: {root}")

    subjects = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("CGMacros-")
    )
    logger.info("discover_subjects(): found %d subject folder(s) under %s",
                len(subjects), root)
    if not subjects:
        logger.warning("discover_subjects(): no folders matching 'CGMacros-*' "
                       "under %s", root)
    else:
        logger.debug("discover_subjects(): subjects=%s", subjects)
    return subjects


def import_subject(subject_dir: Path) -> dict:
    logger.info("import_subject(): starting %s", subject_dir.name)
    zero = {"subject": subject_dir.name, "glucose_rows": 0, "meal_rows": 0}

    # 1. Find the single CSV in the folder
    csv_files = list(subject_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("import_subject(%s): no CSV file found — skipping",
                       subject_dir.name)
        return zero
    if len(csv_files) > 1:
        logger.warning("import_subject(%s): %d CSV files found — using only "
                       "the first (%s)", subject_dir.name, len(csv_files),
                       csv_files[0].name)

    csv_path = csv_files[0]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("import_subject(%s): failed to read %s",
                         subject_dir.name, csv_path)
        return zero

    df.columns = [c.strip() for c in df.columns]
    logger.debug("import_subject(%s): read %s — %d row(s), %d column(s)",
                 subject_dir.name, csv_path.name, len(df), len(df.columns))

    if "Timestamp" not in df.columns:
        logger.warning("import_subject(%s): 'Timestamp' column missing in %s "
                       "(columns=%s) — skipping",
                       subject_dir.name, csv_path.name, list(df.columns))
        return zero

    # Convert timestamp
    df["timestamp"] = _to_datetime(df["Timestamp"])
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    dropped_ts = before - len(df)
    if dropped_ts:
        logger.warning("import_subject(%s): dropped %d/%d row(s) with "
                       "unparseable timestamps",
                       subject_dir.name, dropped_ts, before)

    # 2. Extract Glucose (Look for Dexcom or Libre columns)
    gluc_col = None
    for col in ["Dexcom GL", "Libre GL", "glucose_value"]:
        if col in df.columns:
            gluc_col = col
            break

    g_df = pd.DataFrame()
    if gluc_col:
        g_df = (df[["timestamp", gluc_col]]
                .rename(columns={gluc_col: "glucose_mgdl"})
                .dropna())
        logger.info("import_subject(%s): glucose column=%r → %d reading(s)",
                    subject_dir.name, gluc_col, len(g_df))
    else:
        logger.warning("import_subject(%s): no glucose column found "
                       "(looked for Dexcom GL / Libre GL / glucose_value)",
                       subject_dir.name)

    # 3. Extract Meals (Rows where 'Carbs' or 'Calories' is not empty)
    m_df = pd.DataFrame()
    if "Carbs" in df.columns or "Calories" in df.columns:
        # Filter only rows that actually have meal data
        meal_mask = (
            df.get("Carbs", pd.Series(dtype=float)).notna()
            | df.get("Calories", pd.Series(dtype=float)).notna()
        )
        m_raw = df[meal_mask].copy()
        logger.debug("import_subject(%s): %d row(s) have meal data "
                     "(carbs or calories present)",
                     subject_dir.name, len(m_raw))

        if not m_raw.empty:
            def _num(col_name: str) -> pd.Series:
                return (pd.to_numeric(m_raw[col_name], errors="coerce").fillna(0.0)
                        if col_name in m_raw.columns else 0.0)

            meal_type = m_raw.get("Meal Type", "Meal").astype(str).str.strip()

            m_df = pd.DataFrame({
                "timestamp": m_raw["timestamp"],
                "food_name": meal_type.radd("Meal — "),
                "carbs_g": _num("Carbs"),
                "sugar_g": _num("Sugar"),
                "calories_kcal": _num("Calories"),
                "protein_g": _num("Protein"),
                "fat_g": _num("Fat"),
                "fiber_g": _num("Fiber"),
            })
            logger.info("import_subject(%s): extracted %d meal row(s)",
                        subject_dir.name, len(m_df))
    else:
        logger.warning("import_subject(%s): neither 'Carbs' nor 'Calories' "
                       "column found — no meals extracted",
                       subject_dir.name)

    # 4. Save to SQLite
    n_glucose = 0
    if not g_df.empty:
        try:
            n_glucose = db.save_glucose_readings(
                [{"timestamp": str(r.timestamp),
                  "glucose_mgdl": float(r.glucose_mgdl)}
                 for r in g_df.itertuples()]
            )
            logger.info("import_subject(%s): saved %d glucose reading(s) to DB",
                        subject_dir.name, n_glucose)
        except Exception:
            logger.exception("import_subject(%s): failed to save glucose "
                             "readings to DB", subject_dir.name)

    n_meals = 0
    if not m_df.empty:
        try:
            with sqlite3.connect(db.DB_PATH) as conn:
                m_df = m_df.assign(source="cgmacros", confidence=1.0)
                m_df.to_sql("meals", conn, if_exists="append", index=False)
                n_meals = len(m_df)
            logger.info("import_subject(%s): saved %d meal row(s) to DB",
                        subject_dir.name, n_meals)
        except Exception:
            logger.exception("import_subject(%s): failed to save meal rows "
                             "to DB", subject_dir.name)

    result = {
        "subject": subject_dir.name,
        "glucose_rows": n_glucose,
        "meal_rows": n_meals,
    }
    logger.info("import_subject(%s): done — glucose_rows=%d, meal_rows=%d",
                subject_dir.name, n_glucose, n_meals)
    return result


def import_all(root: str | Path, subjects: list[str] | None = None) -> list[dict]:
    root = Path(root)
    logger.info("import_all(): root=%s, explicit_subjects=%s",
                root, len(subjects) if subjects else None)

    db.init_db()

    chosen = subjects or discover_subjects(root)
    logger.info("import_all(): importing %d subject(s)", len(chosen))

    results = []
    for name in chosen:
        logger.debug("import_all(): importing %s", name)
        try:
            res = import_subject(root / name)
        except Exception:
            logger.exception("import_all(): unhandled error importing %s — "
                             "continuing with remaining subjects", name)
            results.append({"subject": name, "glucose_rows": 0,
                            "meal_rows": 0, "error": True})
            continue
        results.append(res)

    total_glucose = sum(r.get("glucose_rows", 0) for r in results)
    total_meals = sum(r.get("meal_rows", 0) for r in results)
    failed = sum(1 for r in results if r.get("error"))
    logger.info("import_all(): done — %d subject(s), %d glucose row(s), "
                "%d meal row(s)%s",
                len(results), total_glucose, total_meals,
                f", {failed} failed" if failed else "")
    return results