# app/pipeline/portions.py
"""
Turn the vision result into an EDITABLE (food, grams) list, and back.

The vision model sometimes returns estimated_quantity_grams = null. Instead of
dropping the food (which made nutrition empty), we pre-fill a default portion,
mark it "assumed", and let the user correct it before nutrition is calculated.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

DEFAULT_GRAMS = 200.0          # typical single-dish serving when the model gives no number


def vision_to_rows(vision: dict) -> list[list]:
    """-> [[food, grams, note], ...] for the editable table."""
    food_items = (vision or {}).get("food_items", []) or []
    logger.info("vision_to_rows(): %d food item(s) in vision result",
                len(food_items))

    rows = []
    skipped_no_name = 0
    assumed = 0
    estimated = 0

    for idx, f in enumerate(food_items):
        name = f.get("normalized_food_name") or f.get("food_name")
        if not name:
            skipped_no_name += 1
            logger.debug("vision_to_rows(): skipping item #%d — no name "
                         "(keys=%s)", idx, list(f.keys()))
            continue

        try:
            grams = float(f.get("estimated_quantity_grams"))
        except (TypeError, ValueError):
            grams = None

        if grams is None or math.isnan(grams) or grams <= 0:
            rows.append([name, DEFAULT_GRAMS, "assumed portion — please check"])
            assumed += 1
            logger.debug("vision_to_rows(): %r had no usable grams (%r) — "
                         "using DEFAULT_GRAMS=%.0f",
                         name, f.get("estimated_quantity_grams"), DEFAULT_GRAMS)
        else:
            conf = (f.get("portion_confidence") or 0) * 100
            rows.append([name, round(grams), f"AI estimate ({conf:.0f}% sure) — please check"])
            estimated += 1
            logger.debug("vision_to_rows(): %r → %.0fg (confidence=%.0f%%)",
                         name, grams, conf)

    logger.info("vision_to_rows(): produced %d row(s) — %d from AI estimate, "
                "%d assumed, %d skipped (no name)",
                len(rows), estimated, assumed, skipped_no_name)
    if skipped_no_name:
        logger.warning("vision_to_rows(): %d item(s) skipped because they had "
                       "no food name", skipped_no_name)
    return rows


def rows_to_items(table) -> list[dict]:
    """Editable table (DataFrame or list of rows) -> [{'name','grams'}] for calculate_nutrition."""
    try:
        rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    except Exception:
        logger.exception("rows_to_items(): could not read table (type=%s)",
                         type(table).__name__)
        return []

    logger.debug("rows_to_items(): received %d raw row(s)", len(rows))

    items = []
    skipped = 0
    for idx, r in enumerate(rows):
        if not r or len(r) < 2:
            skipped += 1
            logger.debug("rows_to_items(): skipping row #%d — too few cells "
                         "(%r)", idx, r)
            continue

        name = str(r[0]).strip() if r[0] is not None else ""
        try:
            grams = float(r[1])
        except (TypeError, ValueError):
            skipped += 1
            logger.debug("rows_to_items(): skipping row #%d — grams not "
                         "numeric (name=%r, grams=%r)", idx, name, r[1])
            continue

        if not name or name.lower() == "nan":
            skipped += 1
            logger.debug("rows_to_items(): skipping row #%d — empty food name "
                         "(name=%r)", idx, r[0])
            continue
        if grams <= 0 or math.isnan(grams):
            skipped += 1
            logger.debug("rows_to_items(): skipping row #%d — invalid grams "
                         "(name=%r, grams=%r)", idx, name, grams)
            continue

        items.append({"name": name, "grams": grams})

    logger.info("rows_to_items(): kept %d/%d valid item(s) (%d skipped)",
                len(items), len(rows), skipped)
    if items:
        logger.debug("rows_to_items(): items=%s", items)
    else:
        logger.warning("rows_to_items(): no valid items — caller should "
                       "prompt the user to fix the food table")
    return items