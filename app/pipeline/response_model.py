# app/pipeline/response_model.py
"""
Meal -> glucose response.

Trained on CGMacros (scripts/build_response_model.py): for every meal it learns
how much glucose rises (relative to the level at meal time) at
0,15,30,45,60,90,120,150,180 min.

  predict_delta_curve(nutrition)  -> rise at those offsets
  what_if(nutrition)              -> less-sugar / actual / extra-sugar curves
  simulate(meals, start, end, baseline) -> estimated glucose timeline

These are ESTIMATES from a population model, not measurements.
If the trained model file is missing, a simple rule-of-thumb is used instead.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g", "calories_kcal"]
OFFSETS = [0, 15, 30, 45, 60, 90, 120, 150, 180]          # minutes after the meal
MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "glucose_response.joblib"

_model = None
_model_loaded = False


def _get_model():
    global _model, _model_loaded
    if not _model_loaded:
        _model_loaded = True
        try:
            import joblib
            if MODEL_PATH.is_file():
                _model = joblib.load(MODEL_PATH)
        except Exception as e:                      # model optional
            print(f"[response_model] could not load model: {e}")
    return _model


def using_trained_model() -> bool:
    return _get_model() is not None


def _heuristic(n: dict) -> np.ndarray:
    carbs, sugar = n.get("carbs_g") or 0, n.get("sugar_g") or 0
    fiber, protein = n.get("fiber_g") or 0, n.get("protein_g") or 0
    peak = 0.9 * carbs + 0.5 * sugar - 1.5 * fiber - 0.1 * protein
    peak = float(np.clip(peak, 0, 220))
    shape = np.array([0, 0.25, 0.7, 0.95, 1.0, 0.75, 0.4, 0.15, 0.0])
    return peak * shape


def predict_delta_curve(n: dict) -> np.ndarray:
    """Glucose rise (mg/dL) above the level at meal time, at OFFSETS."""
    model = _get_model()
    if model is None:
        return _heuristic(n)
    x = pd.DataFrame([{f: float(n.get(f) or 0.0) for f in FEATURES}])
    y = np.asarray(model.predict(x)[0], dtype=float)
    y[0] = 0.0
    return np.clip(y, -15.0, None)


def what_if(n: dict, sugar_cut: float = 0.5, sugar_add_g: float = 20.0) -> dict:
    """
    Three curves for the SAME meal:
      less  - sugar reduced by `sugar_cut` (sugar is part of carbs, so carbs drop too)
      actual- what the user ate
      extra - `sugar_add_g` more sugar (adds the same grams of carbs)
    Ordering less <= actual <= extra is enforced (more sugar never lowers the rise).
    """
    base = {k: float(n.get(k) or 0.0) for k in FEATURES}
    actual = predict_delta_curve(base)

    cut_g = base["sugar_g"] * sugar_cut
    less_n = dict(base, sugar_g=base["sugar_g"] - cut_g,
                  carbs_g=max(base["carbs_g"] - cut_g, 0.0),
                  calories_kcal=max(base["calories_kcal"] - 4 * cut_g, 0.0))
    extra_n = dict(base, sugar_g=base["sugar_g"] + sugar_add_g,
                   carbs_g=base["carbs_g"] + sugar_add_g,
                   calories_kcal=base["calories_kcal"] + 4 * sugar_add_g)

    less = np.minimum(predict_delta_curve(less_n), actual)
    extra = np.maximum(predict_delta_curve(extra_n), actual)
    return {"less": less, "actual": actual, "extra": extra,
            "sugar_cut_g": round(cut_g, 1), "sugar_add_g": sugar_add_g}


def _full_curve(delta: np.ndarray):
    """Append a return-to-baseline point at 240 min."""
    return OFFSETS + [240], list(delta) + [0.0]


def simulate(meals: list[tuple], start, end, baseline: float,
             step_min: int = 5) -> pd.DataFrame:
    """
    meals    : [(timestamp, nutrition_dict), ...]  (include meals up to 4 h before `start`)
    returns  : DataFrame(timestamp, glucose_mgdl) every `step_min` minutes
    Each meal's rise is added on top of the baseline (superposition).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    grid = pd.date_range(start, end, freq=f"{step_min}min")
    t = (grid - start).total_seconds().to_numpy() / 60.0
    y = np.full(len(grid), float(baseline))
    for ts, nut in meals:
        m0 = (pd.Timestamp(ts) - start).total_seconds() / 60.0
        xs, ys = _full_curve(predict_delta_curve(nut))
        y += np.interp(t - m0, xs, ys, left=0.0, right=0.0)
    return pd.DataFrame({"timestamp": grid, "glucose_mgdl": y})