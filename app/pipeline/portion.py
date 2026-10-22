# app/pipeline/portions.py
"""
Turn the vision result into an EDITABLE (food, grams) list, and back.

The vision model sometimes returns estimated_quantity_grams = null. Instead of
dropping the food (which made nutrition empty), we pre-fill a default portion,
mark it "assumed", and let the user correct it before nutrition is calculated.
"""
from __future__ import annotations

import math

DEFAULT_GRAMS = 200.0          # typical single-dish serving when the model gives no number


def vision_to_rows(vision: dict) -> list[list]:
    """-> [[food, grams, note], ...] for the editable table."""
    rows = []
    for f in vision.get("food_items", []):
        name = f.get("normalized_food_name") or f.get("food_name")
        if not name:
            continue
        try:
            grams = float(f.get("estimated_quantity_grams"))
        except (TypeError, ValueError):
            grams = None
        if grams is None or math.isnan(grams) or grams <= 0:
            rows.append([name, DEFAULT_GRAMS, "assumed portion — please check"])
        else:
            conf = (f.get("portion_confidence") or 0) * 100
            rows.append([name, round(grams), f"AI estimate ({conf:.0f}% sure) — please check"])
    return rows


def rows_to_items(table) -> list[dict]:
    """Editable table (DataFrame or list of rows) -> [{'name','grams'}] for calculate_nutrition."""
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    items = []
    for r in rows:
        if not r or len(r) < 2:
            continue
        name = str(r[0]).strip() if r[0] is not None else ""
        try:
            grams = float(r[1])
        except (TypeError, ValueError):
            continue
        if name and name.lower() != "nan" and grams > 0 and not math.isnan(grams):
            items.append({"name": name, "grams": grams})
    return items