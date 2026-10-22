# app/scripts/build_response_model.py
"""
Train the meal->glucose response model from CGMacros.
Run from the `app` folder:   python scripts/build_response_model.py

Needs: pandas numpy scikit-learn joblib
Output: data/processed/response_training.csv  and  data/models/glucose_response.joblib
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from pipeline.response_model import FEATURES, OFFSETS, MODEL_PATH   # noqa: E402

ROOT = APP / "data" / "raw" / "CGMacros"
OUT_CSV = APP / "data" / "processed" / "response_training.csv"

COLS = {"Carbs": "carbs_g", "Sugar": "sugar_g", "Fiber": "fiber_g",
        "Protein": "protein_g", "Fat": "fat_g", "Calories": "calories_kcal"}


def _to_dt(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.to_datetime(s, unit="s", errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(series, errors="coerce")
    return out


def _secs(ts: pd.Series) -> np.ndarray:
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def _has_reading_near(gt: np.ndarray, t: float, tol: float = 600) -> bool:
    i = np.searchsorted(gt, t)
    near = []
    if i < len(gt): near.append(abs(gt[i] - t))
    if i > 0:       near.append(abs(gt[i - 1] - t))
    return bool(near) and min(near) <= tol


def build_training_table() -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("CGMacros-")):
        csvs = list(sub.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        df.columns = [c.strip() for c in df.columns]
        if "Timestamp" not in df.columns:
            continue
        df["timestamp"] = _to_dt(df["Timestamp"])
        df = df.dropna(subset=["timestamp"])

        gcol = next((c for c in ["Dexcom GL", "Libre GL", "glucose_value"] if c in df.columns), None)
        if gcol is None:
            continue
        g = df[["timestamp", gcol]].dropna().sort_values("timestamp")
        if len(g) < 50:
            continue
        gt, gv = _secs(g["timestamp"]), g[gcol].to_numpy(dtype=float)

        meal_mask = df.get("Carbs", pd.Series(dtype=float)).notna() | \
                    df.get("Calories", pd.Series(dtype=float)).notna()
        meals = df[meal_mask].sort_values("timestamp")
        mt = _secs(meals["timestamp"])

        for k, (_, r) in enumerate(meals.iterrows()):
            t0 = mt[k]
            if k + 1 < len(mt) and mt[k + 1] - t0 < 180 * 60:
                continue                      # next meal would contaminate the window
            if not (_has_reading_near(gt, t0) and _has_reading_near(gt, t0 + 180 * 60)):
                continue
            feats = {new: float(pd.to_numeric(r.get(old), errors="coerce") or 0.0)
                     for old, new in COLS.items()}
            if feats["carbs_g"] == 0 and feats["calories_kcal"] == 0:
                continue
            base = float(np.interp(t0, gt, gv))
            row = {"subject": sub.name, **feats}
            for o in OFFSETS:
                row[f"d{o}"] = float(np.interp(t0 + o * 60, gt, gv) - base)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"CGMacros folder not found: {ROOT}")
    table = build_training_table()
    if len(table) < 30:
        raise SystemExit(f"Only {len(table)} usable meals found — check the CSV columns.")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"Training meals: {len(table)} from {table['subject'].nunique()} subjects")

    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    X = table[FEATURES]
    Y = table[[f"d{o}" for o in OFFSETS]]
    mk = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                       random_state=0, n_jobs=-1)

    # honest check: predict subjects the model has never seen
    peak_idx = OFFSETS.index(60)
    errs = []
    for tr, te in GroupKFold(n_splits=5).split(X, Y, table["subject"]):
        m = mk().fit(X.iloc[tr], Y.iloc[tr])
        errs.append(np.abs(m.predict(X.iloc[te])[:, peak_idx] - Y.iloc[te].iloc[:, peak_idx]))
    print(f"Cross-subject MAE of the rise at 60 min: {np.concatenate(errs).mean():.1f} mg/dL "
          f"(spread of real rises: std {Y.iloc[:, peak_idx].std():.1f})")

    model = mk().fit(X, Y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()