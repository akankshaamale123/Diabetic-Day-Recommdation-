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

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = ["carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g", "calories_kcal"]
OFFSETS = [0, 15, 30, 45, 60, 90, 120, 150, 180]          # minutes after the meal
MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "glucose_response.joblib"

_model = None
_model_loaded = False


def _get_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True
    logger.info("_get_model(): loading response model from %s", MODEL_PATH)

    try:
        import joblib
    except Exception:
        logger.exception("_get_model(): failed to import joblib — falling "
                         "back to heuristic")
        return _model

    if not MODEL_PATH.is_file():
        logger.warning("_get_model(): model file not found at %s — using "
                       "heuristic", MODEL_PATH)
        return _model

    try:
        _model = joblib.load(MODEL_PATH)
        logger.info("_get_model(): model loaded successfully (type=%s)",
                    type(_model).__name__)
    except Exception:
        logger.exception("_get_model(): failed to load model from %s — "
                         "falling back to heuristic", MODEL_PATH)

    return _model


def using_trained_model() -> bool:
    loaded = _get_model() is not None
    logger.debug("using_trained_model(): %s", loaded)
    return loaded


def _heuristic(n: dict) -> np.ndarray:
    carbs = n.get("carbs_g") or 0
    sugar = n.get("sugar_g") or 0
    fiber = n.get("fiber_g") or 0
    protein = n.get("protein_g") or 0

    peak = 0.9 * carbs + 0.5 * sugar - 1.5 * fiber - 0.1 * protein
    peak = float(np.clip(peak, 0, 220))
    shape = np.array([0, 0.25, 0.7, 0.95, 1.0, 0.75, 0.4, 0.15, 0.0])
    curve = peak * shape

    logger.debug("_heuristic(): carbs=%.1f sugar=%.1f fiber=%.1f "
                 "protein=%.1f → peak=%.1f mg/dL",
                 carbs, sugar, fiber, protein, peak)
    return curve


def predict_delta_curve(n: dict) -> np.ndarray:
    """Glucose rise (mg/dL) above the level at meal time, at OFFSETS."""
    model = _get_model()

    if model is None:
        logger.debug("predict_delta_curve(): using heuristic "
                     "(no trained model); inputs=%s", n)
        return _heuristic(n)

    x = pd.DataFrame([{f: float(n.get(f) or 0.0) for f in FEATURES}])
    try:
        y = np.asarray(model.predict(x)[0], dtype=float)
    except Exception:
        logger.exception("predict_delta_curve(): model.predict() failed for "
                         "features=%s — falling back to heuristic", x.to_dict("records"))
        return _heuristic(n)

    y[0] = 0.0
    y = np.clip(y, -15.0, None)
    logger.debug("predict_delta_curve(): model peak=%.1f mg/dL at offset=%d min",
                 float(np.max(y)), OFFSETS[int(np.argmax(y))])
    return y


def what_if(n: dict, sugar_cut: float = 0.5, sugar_add_g: float = 20.0) -> dict:
    """
    Three curves for the SAME meal:
      less  - sugar reduced by `sugar_cut` (sugar is part of carbs, so carbs drop too)
      actual- what the user ate
      extra - `sugar_add_g` more sugar (adds the same grams of carbs)
    Ordering less <= actual <= extra is enforced (more sugar never lowers the rise).
    """
    logger.info("what_if(): sugar_cut=%.2f, sugar_add_g=%.1fg, "
                "meal_carbs=%.1f meal_sugar=%.1f",
                sugar_cut, sugar_add_g,
                float(n.get("carbs_g") or 0), float(n.get("sugar_g") or 0))

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

    logger.info("what_if(): peaks — actual=%.1f, less=%.1f (saved %.1f), "
                "extra=%.1f (added %.1f); cut=%.1fg sugar",
                float(actual.max()), float(less.max()),
                float(actual.max() - less.max()),
                float(extra.max()),
                float(extra.max() - actual.max()),
                cut_g)

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
    logger.info("simulate(): window %s → %s, baseline=%.1f mg/dL, "
                "meals=%d, step=%dmin",
                start, end, float(baseline), len(meals or []), step_min)

    if not meals:
        logger.warning("simulate(): no meals passed — timeline will be flat "
                       "at baseline")

    grid = pd.date_range(start, end, freq=f"{step_min}min")
    t = (grid - start).total_seconds().to_numpy() / 60.0
    y = np.full(len(grid), float(baseline))

    for idx, (ts, nut) in enumerate(meals or []):
        m0 = (pd.Timestamp(ts) - start).total_seconds() / 60.0
        try:
            xs, ys = _full_curve(predict_delta_curve(nut))
        except Exception:
            logger.exception("simulate(): failed to compute curve for meal "
                             "#%d at %s — skipping", idx, ts)
            continue
        y += np.interp(t - m0, xs, ys, left=0.0, right=0.0)
        logger.debug("simulate(): meal #%d at %s contributed peak=%.1f mg/dL",
                     idx, ts, float(np.max(ys)))

    logger.info("simulate(): produced %d point(s); glucose min=%.1f "
                "max=%.1f mean=%.1f",
                len(y), float(y.min()), float(y.max()), float(y.mean()))
    return pd.DataFrame({"timestamp": grid, "glucose_mgdl": y})