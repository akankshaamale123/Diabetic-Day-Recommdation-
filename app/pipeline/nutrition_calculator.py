# app/pipeline/nutrition_calculator.py
from datasets import load_dataset
from pathlib import Path
from rapidfuzz import process, fuzz
from Prompts import _NUTRITION_PROMPT
from client import generate_text
import pandas as pd
import json, re


_dataset = load_dataset("adarshzolekar/foods-nutrition-dataset")
_df = _dataset["train"].to_pandas()
_df["Food Items"] = _df["Food Items"].astype(str).str.strip()
_food_names = _df["Food Items"].tolist()

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
    if _CACHE_PATH.is_file():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                           encoding="utf-8")

_LLM_CACHE = _load_cache()


def find_food(name, threshold=85):
    match = process.extractOne(name, _food_names, scorer=fuzz.WRatio)
    if match is None or match[1] < threshold:
        return None
    matched_name = match[0]
    row = _df[_df["Food Items"] == matched_name].iloc[0]
    return matched_name, row, match[1]

def _strip_and_parse(text: str) -> dict | None:
    """Reuse the same robust parsing you already have."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

def estimate_nutrition_from_llm(food_name: str) -> dict | None:
    """
    Ask the LLM for per-100 g nutrient values.
    Returns dict with per-100 g values + confidence, or None on failure.
    Cached on disk so the same food is only queried once.
    """
    key = food_name.strip().lower()
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    prompt = _NUTRITION_PROMPT.format(food=food_name)
    try:
        raw = generate_text(prompt, stream=False)
        if not raw:
            return None
        parsed = _strip_and_parse(raw)
    except Exception as e:
        print(f"[LLM fallback] failed for '{food_name}': {e}")
        return None

    if not parsed or not parsed.get("per_100g"):
        return None

    # validate keys
    expected = set(NUTRIENTS.values())
    per_100g = {k: parsed["per_100g"].get(k) for k in expected}
    if all(v is None for v in per_100g.values()):
        return None

    result = {
        "per_100g": per_100g,
        "confidence": float(parsed.get("confidence", 0.0)),
        "notes": parsed.get("notes", ""),
    }
    _LLM_CACHE[key] = result
    _save_cache(_LLM_CACHE)
    return result


def calculate_nutrition(items, basis_grams=100, use_llm_fallback=True):
    results = []
    totals = {v: 0.0 for v in NUTRIENTS.values()}
    unmatched = []
    llm_estimated = []

    for item in items:
        name = item["name"]
        grams = float(item["grams"])

        if grams <= 0:
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
                continue

        # ---------- Case 3: nothing worked ----------
        unmatched.append(name)

    return {
        "items": results,
        "total_nutrition": {k: round(v, 2) for k, v in totals.items()},
        "unmatched_foods": unmatched,
        "llm_estimated_items": llm_estimated,   # <-- surface this to the UI
        "has_unverified_values": bool(llm_estimated),
    }