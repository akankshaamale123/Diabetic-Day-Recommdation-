# app/storage/cgmacros_loader.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import sqlite3

from . import db

def _to_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.to_datetime(s, unit="s", errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(series, errors="coerce")
    return out

def discover_subjects(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"CGMacros root not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("CGMacros-"))

def import_subject(subject_dir: Path) -> dict:
    # 1. Find the single CSV in the folder
    csv_files = list(subject_dir.glob("*.csv"))
    if not csv_files:
        print(f"    ⚠️ No CSV found in {subject_dir.name}")
        return {"subject": subject_dir.name, "glucose_rows": 0, "meal_rows": 0}
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    if "Timestamp" not in df.columns:
        print(f"    ⚠️ No 'Timestamp' column in {csv_path.name}")
        return {"subject": subject_dir.name, "glucose_rows": 0, "meal_rows": 0}

    # Convert timestamp
    df["timestamp"] = _to_datetime(df["Timestamp"])
    df = df.dropna(subset=["timestamp"])

    # 2. Extract Glucose (Look for Dexcom or Libre columns)
    gluc_col = None
    for col in ["Dexcom GL", "Libre GL", "glucose_value"]:
        if col in df.columns:
            gluc_col = col
            break

    g_df = pd.DataFrame()
    if gluc_col:
        g_df = df[["timestamp", gluc_col]].rename(columns={gluc_col: "glucose_mgdl"}).dropna()

    # 3. Extract Meals (Rows where 'Carbs' or 'Calories' is not empty)
    m_df = pd.DataFrame()
    if "Carbs" in df.columns or "Calories" in df.columns:
        # Filter only rows that actually have meal data
        meal_mask = df.get("Carbs", pd.Series(dtype=float)).notna() | df.get("Calories", pd.Series(dtype=float)).notna()
        m_raw = df[meal_mask].copy()
        
        if not m_raw.empty:
            def _num(col_name: str) -> pd.Series:
                return pd.to_numeric(m_raw[col_name], errors="coerce").fillna(0.0) if col_name in m_raw.columns else 0.0

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

    # 4. Save to SQLite
    n_glucose = 0
    if not g_df.empty:
        n_glucose = db.save_glucose_readings(
            [{"timestamp": str(r.timestamp), "glucose_mgdl": float(r.glucose_mgdl)} for r in g_df.itertuples()]
        )

    n_meals = 0
    if not m_df.empty:
        with sqlite3.connect(db.DB_PATH) as conn:
            m_df = m_df.assign(source="cgmacros", confidence=1.0)
            m_df.to_sql("meals", conn, if_exists="append", index=False)
            n_meals = len(m_df)

    return {
        "subject": subject_dir.name,
        "glucose_rows": n_glucose,
        "meal_rows": n_meals,
    }

def import_all(root: str | Path, subjects: list[str] | None = None) -> list[dict]:
    root = Path(root)
    db.init_db()
    chosen = subjects or discover_subjects(root)
    
    results = []
    for name in chosen:
        print(f"[CGMacros] importing {name} …", end=" ")
        res = import_subject(root / name)
        print(f"({res['glucose_rows']} glucose, {res['meal_rows']} meals)")
        results.append(res)
    return results