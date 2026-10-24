# app/pipeline/nutrition_calculator.py
import json
import logging
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rapidfuzz import process, fuzz

from Prompts import _NUTRITION_PROMPT
from client import generate_text

logger = logging.getLogger(__name__)


# ---------------- dataset load (import-time) ----------------
logger.info("Loading nutrition dataset: adarshzolekar/foods-nutrition-dataset")
try:
    _dataset = load_dataset("adarshzolekar/foods-nutrition-dataset")
    _df = _dataset["train"].to_pandas()
    _df["Food Items"] = _df["Food Items"].astype(str).str.strip()
    _food_names = _df["Food Items"].tolist()
    logger.info("Nutrition dataset loaded: %d row(s), %d unique food name(s)",
                len(_df), len(set(_food_names)))
except Exception:
    logger.exception("Failed to load nutrition dataset — the app cannot "
                     "resolve foods without it")
    raise


NUTRIENTS = {
    "Energy kcal": "calories_kcal",
    "Carbs": "carbs_g",
    "Protein(g)": "protein_g",
    "Fat(g)": "fat_g",
    "Freesugar(g)": "sugar_g",
    "Fibre(g)": "fiber_g",
    "Cholestrol(mg)": "cholesterol_mg",
    "Calcium(mg)": "calcium_mg",
}


# ---------------- LLM cache ----------------
_CACHE_PATH = Path(__file__).parent / "llm_nutrition_cache.json"


def _load_cache() -> dict:
    if not _CACHE_PATH.is_file():
        logger.info("_load_cache(): no cache file at %s — starting empty",
                    _CACHE_PATH)
        return {}
    try:
        cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        logger.info("_load_cache(): loaded %d cached LLM nutrition entr(ies) "
                    "from %s", len(cache), _CACHE_PATH)
        return cache
    except json.JSONDecodeError:
        logger.exception("_load_cache(): cache file %s is corrupt — "
                         "starting empty", _CACHE_PATH)
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("_save_cache(): wrote %d entr(ies) to %s",
                     len(cache), _CACHE_PATH)
    except Exception:
        logger.exception("_save_cache(): failed to write cache to %s",
                         _CACHE_PATH)


_LLM_CACHE = _load_cache()


def find_food(name, threshold=85):
    match = process.extractOne(name, _food_names, scorer=fuzz.WRatio)
    if match is None:
        logger.debug("find_food(%r): no fuzzy match returned", name)
        return None
    matched_name, score = match[0], match[1]
    if score < threshold:
        logger.debug("find_food(%r): best match %r scored %.0f < threshold %d "
                     "— treated as not found", name, matched_name, score, threshold)
        return None
    row = _df[_df["Food Items"] == matched_name].iloc[0]
    logger.debug("find_food(%r): matched %r (score=%.0f)", name, matched_name, score)
    return matched_name, row, score


def _strip_and_parse(text: str) -> dict | None:
    """Reuse the same robust parsing you already have."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    for idx, candidate in enumerate(
            (text, text[text.find("{"): text.rfind("}") + 1])):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            logger.debug("_strip_and_parse(): success on attempt %d", idx + 1)
            return parsed
        except json.JSONDecodeError as e:
            logger.debug("_strip_and_parse(): attempt %d failed (%s)", idx + 1, e)
            continue
    logger.debug("_strip_and_parse(): could not parse model output "
                 "(len=%d, preview=%r)", len(text), text[:200])
    return None


def estimate_nutrition_from_llm(food_name: str) -> dict | None:
    """
    Ask the LLM for per-100 g nutrient values.
    Returns dict with per-100 g values + confidence, or None on failure.
    Cached on disk so the same food is only queried once.
    """
    key = food_name.strip().lower()
    if key in _LLM_CACHE:
        logger.debug("estimate_nutrition_from_llm(%r): cache hit", food_name)
        return _LLM_CACHE[key]

    logger.info("estimate_nutrition_from_llm(%r): cache miss — calling LLM",
                food_name)
    prompt = _NUTRITION_PROMPT.format(food=food_name)
    try:
        raw = generate_text(prompt, stream=False)
    except Exception:
        logger.exception("estimate_nutrition_from_llm(%r): generate_text() "
                         "failed", food_name)
        return None

    if not raw:
        logger.warning("estimate_nutrition_from_llm(%r): empty LLM response",
                       food_name)
        return None

    parsed = _strip_and_parse(raw)
    if not parsed or not parsed.get("per_100g"):
        logger.warning("estimate_nutrition_from_llm(%r): unparseable or "
                       "missing per_100g in LLM response", food_name)
        return None

    # validate keys
    expected = set(NUTRIENTS.values())
    per_100g = {k: parsed["per_100g"].get(k) for k in expected}
    if all(v is None for v in per_100g.values()):
        logger.warning("estimate_nutrition_from_llm(%r): LLM returned no "
                       "usable nutrient values (keys=%s)",
                       food_name, list(parsed["per_100g"].keys()))
        return None

    result = {
        "per_100g": per_100g,
        "confidence": float(parsed.get("confidence", 0.0)),
        "notes": parsed.get("notes", ""),
    }
    _LLM_CACHE[key] = result
    _save_cache(_LLM_CACHE)
    logger.info("estimate_nutrition_from_llm(%r): estimated (confidence=%.2f), "
                "cache now has %d entr(ies)",
                food_name, result["confidence"], len(_LLM_CACHE))
    return result


def calculate_nutrition(items, basis_grams=100, use_llm_fallback=True):
    logger.info("calculate_nutrition(): %d item(s), basis=%dg, "
                "llm_fallback=%s", len(items or []), basis_grams,
                use_llm_fallback)

    results = []
    totals = {v: 0.0 for v in NUTRIENTS.values()}
    unmatched = []
    llm_estimated = []

    for item in items or []:
        name = item["name"]
        grams = float(item["grams"])

        if grams <= 0:
            logger.error("calculate_nutrition(): invalid portion for %r "
                         "(grams=%s)", name, grams)
            raise ValueError(f"Invalid portion: {name}")

        match = find_food(name)
        factor = grams / basis_grams

        # ---------- Case 1: DB match ----------
        if match is not None:
            matched_name, row, score = match
            nutrition = {}
            for column, output in NUTRIENTS.items():
                value = pd.to_numeric(row[column], errors="coerce")
                if pd.isna(value):
                    nutrition[output] = None
                else:
                    amount = float(value) * factor
                    nutrition[output] = round(amount, 2)
                    totals[output] += amount

            results.append({
                "requested_food": name,
                "matched_food": matched_name,
                "portion_grams": grams,
                "match_score": score,
                "source": "database",
                "confidence": round(score / 100, 2),
                "nutrition": nutrition,
            })
            logger.debug("calculate_nutrition(): %r → DB match %r "
                         "(score=%.0f, %.0fg)", name, matched_name, score, grams)
            continue

        # ---------- Case 2: LLM fallback ----------
        if use_llm_fallback:
            estimate = estimate_nutrition_from_llm(name)
            if estimate:
                nutrition = {}
                for output, value in estimate["per_100g"].items():
                    if value is None:
                        nutrition[output] = None
                    else:
                        amount = float(value) * factor
                        nutrition[output] = round(amount, 2)
                        totals[output] += amount

                results.append({
                    "requested_food": name,
                    "matched_food": name,
                    "portion_grams": grams,
                    "match_score": None,
                    "source": "llm_estimate",
                    "confidence": estimate["confidence"],
                    "notes": estimate["notes"],
                    "nutrition": nutrition,
                })
                llm_estimated.append(name)
                logger.debug("calculate_nutrition(): %r → LLM estimate "
                             "(confidence=%.2f, %.0fg)",
                             name, estimate["confidence"], grams)
                continue

        # ---------- Case 3: nothing worked ----------
        unmatched.append(name)
        logger.warning("calculate_nutrition(): %r could not be resolved "
                       "(no DB match, LLM fallback=%s)", name, use_llm_fallback)

    summary = {
        "items": results,
        "total_nutrition": {k: round(v, 2) for k, v in totals.items()},
        "unmatched_foods": unmatched,
        "llm_estimated_items": llm_estimated,
        "has_unverified_values": bool(llm_estimated),
    }
    logger.info(
        "calculate_nutrition(): done — db=%d, llm=%d, unmatched=%d, "
        "kcal=%.0f, carbs=%.0fg, sugar=%.0fg",
        len(results) - len(llm_estimated),
        len(llm_estimated),
        len(unmatched),
        totals.get("calories_kcal", 0.0),
        totals.get("carbs_g", 0.0),
        totals.get("sugar_g", 0.0),
    )
    return summary