# app/pipeline/meal_advisor.py
"""
Breakfast check + lunch suggestion for a diabetic user.

Flow:
  photo -> analyze_meal() -> total_nutrition
        -> assess_meal()          : good / okay / not_suitable + reasons
        -> suggest_lunch()        : carb budget + foods from the nutrition DB

IMPORTANT: DEFAULT_TARGETS are generic placeholders. Replace them with the
user's clinician-approved targets (see `targets=` argument).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# per-meal limits (placeholders — set from clinician-approved targets)
DEFAULT_TARGETS = {
    "breakfast": {"carbs_g": 45, "sugar_g": 10, "fiber_min_g": 5,
                  "protein_min_g": 12, "calories_max": 450},
    "lunch":     {"carbs_g": 60, "sugar_g": 12, "fiber_min_g": 7,
                  "protein_min_g": 20, "calories_max": 600},
}

TIPS = [
    "Fill half the plate with non-starchy vegetables or salad.",
    "Pair every carb with protein (dal, paneer, egg, curd, chicken, fish).",
    "Prefer whole grains (jowar, bajra, ragi, brown rice) over refined ones.",
    "A 10-15 minute walk after eating helps blunt the glucose rise.",
]


# ----------------------------------------------------------------
# 1. Is this meal OK for this user?
# ----------------------------------------------------------------
# def assess_meal(totals: dict, meal_type: str = "breakfast",
#                 targets: dict | None = None,
#                 detected_names: list[str] | None = None,
#                 risky_foods: list[str] | None = None) -> dict:
#     """ ... (original implementation, kept for reference) ... """

def assess_meal(totals, meal_type="breakfast", targets=None,
                detected_names=None, risky_foods=None):
    logger.info("assess_meal(): meal_type=%s, risky_foods=%d, "
                "detected_names=%d",
                meal_type,
                len(risky_foods or []),
                len(detected_names or []))
    logger.debug("assess_meal(): totals=%s", totals)

    try:
        from . import rules
        rs = rules.evaluate(totals, meal_type, risky_foods, detected_names)
        v, label = rules.verdict(rs)
    except Exception:
        logger.exception("assess_meal(): rules evaluation failed "
                         "(meal_type=%s)", meal_type)
        raise

    n_rules = len(rs or [])
    logger.info("assess_meal(): verdict=%s (%s) with %d rule(s)",
                v, label, n_rules)
    if n_rules:
        logger.debug("assess_meal(): fired rules=%s",
                     [getattr(r, "title", type(r).__name__) for r in rs])

    return {
        "verdict": v, "label": label, "rules": [r.__dict__ for r in rs],
        "meal_type": meal_type,
        **{k: round(float(totals.get(k) or 0), 1)
           for k in ("carbs_g", "sugar_g", "fiber_g", "protein_g",
                     "calories_kcal")},
    }


# ----------------------------------------------------------------
# 2. What to eat at lunch
# ----------------------------------------------------------------
def suggest_lunch(assessment: dict, targets: dict | None = None,
                  risky_foods: list[str] | None = None, n: int = 6) -> dict:
    logger.info("suggest_lunch(): verdict=%s, n=%d, risky_foods=%d",
                assessment.get("verdict"), n, len(risky_foods or []))

    # imported here so this module loads without the HF dataset
    from pipeline.nutrition_calculator import _df

    tg = targets or DEFAULT_TARGETS
    over = max(0.0, assessment["carbs_g"] - tg["breakfast"]["carbs_g"])
    lunch_carbs = max(20.0, tg["lunch"]["carbs_g"] - over)   # cut lunch if breakfast was heavy
    logger.debug("suggest_lunch(): breakfast_carbs=%.1f over=%.1f "
                 "lunch_budget=%.1f", assessment["carbs_g"], over, lunch_carbs)

    d = _df.copy()
    logger.debug("suggest_lunch(): nutrition DB rows (raw)=%d", len(d))

    for c in ["Carbs", "Fibre(g)", "Protein(g)", "Freesugar(g)", "Energy kcal"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Carbs", "Fibre(g)", "Protein(g)", "Freesugar(g)"])
    d = d.drop_duplicates(subset="Food Items")
    logger.debug("suggest_lunch(): rows after numeric coercion + dropna + "
                 "dedupe=%d", len(d))

    d = d[(d["Freesugar(g)"] <= 5) & ((d["Fibre(g)"] >= 2) | (d["Protein(g)"] >= 8))]
    logger.debug("suggest_lunch(): rows after low-sugar/high-fibre-or-protein "
                 "filter=%d", len(d))

    for r in risky_foods or []:
        before = len(d)
        d = d[~d["Food Items"].str.lower().str.contains(r.lower(), regex=False)]
        removed = before - len(d)
        if removed:
            logger.debug("suggest_lunch(): removed %d row(s) matching risky "
                         "food %r", removed, r)

    if d.empty:
        logger.warning("suggest_lunch(): no candidate foods left after "
                       "filtering (risky_foods=%d) — returning empty plan",
                       len(risky_foods or []))
        return {
            "intro": "No suitable lunch candidates found for your profile.",
            "lunch_carb_budget_g": round(lunch_carbs, 0),
            "suggestions": [],
            "tips": TIPS,
        }

    d["score"] = (d["Protein(g)"] * 1.0 + d["Fibre(g)"] * 2.0
                  - d["Freesugar(g)"] * 2.0 - d["Carbs"] * 0.3)
    top = d.sort_values("score", ascending=False).head(n)
    logger.debug("suggest_lunch(): selected top %d candidate(s)", len(top))

    picks = []
    per_item_carbs = lunch_carbs / 2          # leave room for a second item
    for _, r in top.iterrows():
        carbs100 = max(float(r["Carbs"]), 0.1)
        grams = min(200.0, per_item_carbs / carbs100 * 100)
        grams = max(50.0, round(grams / 10) * 10)
        f = grams / 100
        picks.append({
            "food": r["Food Items"], "portion_g": grams,
            "carbs_g": round(float(r["Carbs"]) * f, 1),
            "protein_g": round(float(r["Protein(g)"]) * f, 1),
            "fiber_g": round(float(r["Fibre(g)"]) * f, 1),
        })

    if assessment["verdict"] == "good":
        intro = "Breakfast was fine. Keep lunch balanced:"
    else:
        intro = ("Breakfast was heavy on carbs/sugar, so keep lunch lighter "
                 f"(about {lunch_carbs:.0f} g carbs):")

    logger.info("suggest_lunch(): returning %d suggestion(s); budget=%.0f g carbs",
                len(picks), lunch_carbs)
    return {"intro": intro, "lunch_carb_budget_g": round(lunch_carbs, 0),
            "suggestions": picks, "tips": TIPS}


# ----------------------------------------------------------------
# 3. One call used by the app
# ----------------------------------------------------------------
def breakfast_advice(analysis: dict, risky_foods: list[str] | None = None,
                     targets: dict | None = None) -> dict:
    """`analysis` = the dict returned by analyze_meal(image_path)."""
    logger.info("breakfast_advice(): risky_foods=%d", len(risky_foods or []))

    vision = analysis.get("vision", {}) or {}
    nutrition = analysis.get("nutrition", {}) or {}
    names = [f.get("normalized_food_name") or f.get("food_name")
             for f in vision.get("food_items", [])]
    names = [n for n in names if n]
    logger.debug("breakfast_advice(): detected foods=%s", names)

    assessment = assess_meal(nutrition.get("total_nutrition", {}), "breakfast",
                             targets, names, risky_foods)
    lunch = suggest_lunch(assessment, targets, risky_foods)

    logger.info("breakfast_advice(): done — breakfast verdict=%s, "
                "%d lunch suggestion(s)",
                assessment.get("verdict"), len(lunch.get("suggestions", [])))
    return {"assessment": assessment, "lunch": lunch}


def breakfast_check(image_path: str, login_date=None, end=None,
                    targets: dict | None = None) -> dict:
    """Photo in -> verdict + lunch plan out (uses user's glucose history)."""
    from pipeline import analyze_meal
    from pipeline.glucose_analysis import summarize_patterns
    from pipeline.glucose_charts import default_login_date

    logger.info("breakfast_check(): image=%s", image_path)

    try:
        login_date = login_date if login_date is not None else default_login_date()
        end = end if end is not None else pd.Timestamp.now()
        logger.debug("breakfast_check(): window %s → %s",
                     pd.Timestamp(login_date).date(), pd.Timestamp(end).date())

        risky = summarize_patterns(login_date, end).get("high_response_foods", [])
        logger.info("breakfast_check(): %d risky food(s) from history: %s",
                    len(risky), risky)

        result = analyze_meal(image_path)
    except Exception:
        logger.exception("breakfast_check(): failed while loading history or "
                         "analyzing meal (image=%s)", image_path)
        raise

    try:
        merged = {**result, **breakfast_advice(result, risky, targets)}
    except Exception:
        logger.exception("breakfast_check(): breakfast_advice() failed for "
                         "image=%s", image_path)
        raise

    verdict = (merged.get("assessment") or {}).get("verdict")
    logger.info("breakfast_check(): done — verdict=%s", verdict)
    return merged


def to_markdown(advice: dict) -> str:
    logger.debug("to_markdown(): building markdown for advice")
    a, l = advice["assessment"], advice["lunch"]
    icon = {"good": "✅", "okay": "🟡", "not_suitable": "🔴"}[a["verdict"]]
    md = f"### {icon} Breakfast: {a['label']}\n"
    md += (f"Carbs {a['carbs_g']} g · Sugar {a['sugar_g']} g · Fibre {a['fiber_g']} g · "
           f"Protein {a['protein_g']} g · {a['calories_kcal']} kcal\n\n")
    for rule in advice["assessment"].get("rules", []):
        md += f"- **{rule['title']}** — {rule['detail']}\n"
        for ev in rule.get("evidence", []):
            md += f"  - _Evidence: {ev}_\n"
    md += f"\n### 🍛 For lunch\n{l['intro']}\n\n"
    md += "| Food | Portion | Carbs | Protein | Fibre |\n|---|---|---|---|---|\n"
    for s in l["suggestions"]:
        md += (f"| {s['food']} | {s['portion_g']:.0f} g | {s['carbs_g']} g | "
               f"{s['protein_g']} g | {s['fiber_g']} g |\n")
    md += "\n" + "\n".join(f"- {t}" for t in l["tips"])
    md += "\n\n*General guidance only — follow your clinician's targets.*"
    return md