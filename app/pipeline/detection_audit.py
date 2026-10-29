# pipeline/detection_audit.py
"""Quality check + optional 'what is missing?' second pass for food detection."""
from __future__ import annotations

import logging
from Prompts import _AUDIT_PROMPT

from pipeline.image_inference import detect_foods

logger = logging.getLogger(__name__)

# tunables
LOW_ITEM_CONF   = 0.85     # identification_confidence below this = suspicious
LOW_PORTION     = 0.70     # portion_confidence below this = suspicious
MIN_EXPECTED    = 3        # fewer items than this for a meal = suspicious
UNCERTAIN_WORDS = (
    "partial", "occluded", "hidden", "unclear", "cannot see",
    "not visible", "overlap", "covered", "obscured",
)


def assess_detection_quality(vision: dict) -> dict:
    """
    Cheap, no-LLM check of a detection result.

    Returns
    -------
    {
      "ok": bool,
      "reasons": [str],
      "suggestion": str,
      "should_ask_another_photo": bool,
      "item_count": int,
    }
    """
    items        = vision.get("food_items", []) or []
    unidentified = vision.get("unidentified_items", []) or []
    reasons: list[str] = []

    # 1) too few items for a meal
    if len(items) < MIN_EXPECTED:
        reasons.append(f"Only {len(items)} food item(s) detected — likely incomplete.")

    # 2) low per-item identification confidence
    low_conf = [it for it in items
                if float(it.get("identification_confidence") or 1.0) < LOW_ITEM_CONF]
    if low_conf:
        names = ", ".join(it.get("food_name", "?") for it in low_conf[:3])
        reasons.append(f"Low identification confidence on: {names}")

    # 3) model couldn't identify some regions
    if unidentified:
        reasons.append(f"{len(unidentified)} region(s) could not be identified.")

    # 4) model itself asked for confirmation
    if vision.get("requires_user_confirmation"):
        reasons.append("The model asked for user confirmation.")

    # 5) overall text hints at occlusion / partial view
    ou = (vision.get("overall_uncertainty") or "").lower()
    if any(w in ou for w in UNCERTAIN_WORDS):
        reasons.append("The model indicated some items are partly hidden or unclear.")

    # 6) many portion estimates are shaky
    shaky = [it for it in items
             if float(it.get("portion_confidence") or 1.0) < LOW_PORTION]
    if len(shaky) >= max(2, len(items) // 3):
        reasons.append(f"Portion estimate uncertain on {len(shaky)} item(s).")

    ok = len(reasons) == 0
    suggestion = (
        "Detection looks complete."
        if ok else
        "For a more accurate result, please upload another photo of the same plate "
        "from a different angle — for example, straight from the top, or from a 45° angle."
    )

    logger.info("assess_detection_quality(): items=%d ok=%s reasons=%d",
                len(items), ok, len(reasons))
    return {
        "ok": ok,
        "reasons": reasons,
        "suggestion": suggestion,
        "should_ask_another_photo": not ok,
        "item_count": len(items),
    }


# ------------------------------------------------------------------ optional LLM audit


def audit_missing_items(image_path: str, vision: dict) -> list[dict]:
    """
    Ask the LLM once more to spot anything the first pass missed.
    Returns the list of *new* items only (possibly empty).
    """
    found = [it.get("food_name", "?") for it in vision.get("food_items", [])]
    if not found:
        return []

    prompt = _AUDIT_PROMPT.format(found=", ".join(found))
    try:
        extra = detect_foods(image_path, user_prompt=prompt)
    except Exception:
        logger.exception("audit_missing_items(): second pass failed — ignoring")
        return []

    new_items = extra.get("food_items", []) or []
    logger.info("audit_missing_items(): second pass returned %d candidate(s)",
                len(new_items))
    return new_items