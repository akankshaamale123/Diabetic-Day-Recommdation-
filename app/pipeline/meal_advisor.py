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

import pandas as pd

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
def assess_meal(totals: dict, meal_type: str = "breakfast",
                targets: dict | None = None,
                detected_names: list[str] | None = None,
                risky_foods: list[str] | None = None) -> dict:
    """
    totals         : nutrition['total_nutrition'] from calculate_nutrition()
    risky_foods    : foods that spiked THIS user's glucose before
                     (summarize_patterns(...)['high_response_foods'])
    """
    t = (targets or DEFAULT_TARGETS)[meal_type]
    carbs = totals.get("carbs_g", 0) or 0
    sugar = totals.get("sugar_g", 0) or 0
    fiber = totals.get("fiber_g", 0) or 0
    protein = totals.get("protein_g", 0) or 0
    kcal = totals.get("calories_kcal", 0) or 0

    issues, points = [], 0
    if carbs > t["carbs_g"]:
        heavy = carbs > 1.3 * t["carbs_g"]
        points += 2 if heavy else 1
        issues.append(f"Carbs {carbs:.0f} g is above the {t['carbs_g']} g limit for {meal_type}.")
    if sugar > t["sugar_g"]:
        points += 1
        issues.append(f"Sugar {sugar:.0f} g is above {t['sugar_g']} g.")
    if fiber < t["fiber_min_g"]:
        points += 1
        issues.append(f"Fibre {fiber:.0f} g is low (aim for {t['fiber_min_g']} g+).")
    if protein < t["protein_min_g"]:
        points += 1
        issues.append(f"Protein {protein:.0f} g is low (aim for {t['protein_min_g']} g+).")
    if kcal > t["calories_max"]:
        issues.append(f"Calories {kcal:.0f} kcal are above {t['calories_max']} kcal.")

    # personal history: did these foods spike this user before?
    personal = []
    for name in detected_names or []:
        for risky in risky_foods or []:
            if risky.lower() in name.lower() or name.lower() in risky.lower():
                personal.append(name)
    if personal:
        points += 2
        issues.append("Your own glucose history shows spikes after: "
                      + ", ".join(sorted(set(personal))) + ".")

    verdict = "good" if points == 0 else "okay" if points <= 2 else "not_suitable"
    label = {"good": "Good choice for you",
             "okay": "Acceptable, but could be better",
             "not_suitable": "Not ideal for diabetes"}[verdict]
    return {"verdict": verdict, "label": label, "score_points": points,
            "issues": issues, "meal_type": meal_type,
            "carbs_g": round(carbs, 1), "sugar_g": round(sugar, 1),
            "fiber_g": round(fiber, 1), "protein_g": round(protein, 1),
            "calories_kcal": round(kcal, 1)}


# ----------------------------------------------------------------
# 2. What to eat at lunch
# ----------------------------------------------------------------
def suggest_lunch(assessment: dict, targets: dict | None = None,
                  risky_foods: list[str] | None = None, n: int = 6) -> dict:
    # imported here so this module loads without the HF dataset
    from pipeline.nutrition_calculator import _df

    tg = targets or DEFAULT_TARGETS
    over = max(0.0, assessment["carbs_g"] - tg["breakfast"]["carbs_g"])
    lunch_carbs = max(20.0, tg["lunch"]["carbs_g"] - over)   # cut lunch if breakfast was heavy

    d = _df.copy()
    for c in ["Carbs", "Fibre(g)", "Protein(g)", "Freesugar(g)", "Energy kcal"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Carbs", "Fibre(g)", "Protein(g)", "Freesugar(g)"])
    d = d.drop_duplicates(subset="Food Items")

    d = d[(d["Freesugar(g)"] <= 5) & ((d["Fibre(g)"] >= 2) | (d["Protein(g)"] >= 8))]
    for r in risky_foods or []:
        d = d[~d["Food Items"].str.lower().str.contains(r.lower(), regex=False)]

    d["score"] = (d["Protein(g)"] * 1.0 + d["Fibre(g)"] * 2.0
                  - d["Freesugar(g)"] * 2.0 - d["Carbs"] * 0.3)
    top = d.sort_values("score", ascending=False).head(n)

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
    return {"intro": intro, "lunch_carb_budget_g": round(lunch_carbs, 0),
            "suggestions": picks, "tips": TIPS}


# ----------------------------------------------------------------
# 3. One call used by the app
# ----------------------------------------------------------------
def breakfast_advice(analysis: dict, risky_foods: list[str] | None = None,
                     targets: dict | None = None) -> dict:
    """`analysis` = the dict returned by analyze_meal(image_path)."""
    vision = analysis.get("vision", {})
    nutrition = analysis.get("nutrition", {})
    names = [f.get("normalized_food_name") or f.get("food_name")
             for f in vision.get("food_items", [])]
    assessment = assess_meal(nutrition.get("total_nutrition", {}), "breakfast",
                             targets, [n for n in names if n], risky_foods)
    lunch = suggest_lunch(assessment, targets, risky_foods)
    return {"assessment": assessment, "lunch": lunch}


def breakfast_check(image_path: str, login_date=None, end=None,
                    targets: dict | None = None) -> dict:
    """Photo in -> verdict + lunch plan out (uses user's glucose history)."""
    from pipeline import analyze_meal
    from pipeline.glucose_analysis import summarize_patterns
    from pipeline.glucose_charts import default_login_date

    login_date = login_date if login_date is not None else default_login_date()
    end = end if end is not None else pd.Timestamp.now()
    risky = summarize_patterns(login_date, end).get("high_response_foods", [])
    result = analyze_meal(image_path)
    return {**result, **breakfast_advice(result, risky, targets)}


def to_markdown(advice: dict) -> str:
    a, l = advice["assessment"], advice["lunch"]
    icon = {"good": "✅", "okay": "🟡", "not_suitable": "🔴"}[a["verdict"]]
    md = f"### {icon} Breakfast: {a['label']}\n"
    md += (f"Carbs {a['carbs_g']} g · Sugar {a['sugar_g']} g · Fibre {a['fiber_g']} g · "
           f"Protein {a['protein_g']} g · {a['calories_kcal']} kcal\n\n")
    for i in a["issues"]:
        md += f"- {i}\n"
    md += f"\n### 🍛 For lunch\n{l['intro']}\n\n"
    md += "| Food | Portion | Carbs | Protein | Fibre |\n|---|---|---|---|---|\n"
    for s in l["suggestions"]:
        md += (f"| {s['food']} | {s['portion_g']:.0f} g | {s['carbs_g']} g | "
               f"{s['protein_g']} g | {s['fiber_g']} g |\n")
    md += "\n" + "\n".join(f"- {t}" for t in l["tips"])
    md += "\n\n*General guidance only — follow your clinician's targets.*"
    return md