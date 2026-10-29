# app/pipeline/__init__.py
import logging

from .image_inference import detect_foods
from .nutrition_calculator import calculate_nutrition

__all__ = ["detect_foods", "calculate_nutrition", "analyze_meal"]

logger = logging.getLogger(__name__)


def _to_nutrition_items(vision_result: dict) -> list:
    """
    Convert the VL JSON into the [{'name':..., 'grams':...}] format
    that calculate_nutrition() expects.
    """
    items = []
    food_items = vision_result.get("food_items", []) or []
    logger.debug("_to_nutrition_items(): received %d vision food item(s)",
                 len(food_items))

    for idx, food in enumerate(food_items):
        name = (food.get("normalized_food_name")
                or food.get("food_name"))
        grams = food.get("estimated_quantity_grams")

        if not name or grams is None or grams <= 0:
            # skip items we can't estimate
            logger.debug(
                "_to_nutrition_items(): skipping item #%d (name=%r, grams=%r)",
                idx, name, grams,
            )
            continue

        items.append({"name": name, "grams": float(grams)})

    logger.debug("_to_nutrition_items(): kept %d/%d item(s) -> %s",
                 len(items), len(food_items), items)
    return items


def analyze_meal(image_path: str, user_prompt: str = None) -> dict:
    """
    End-to-end: image -> detected foods -> nutrition totals.
    Returns a single dict ready to show to the user.
    """
    prompt = user_prompt or "Identify all visible food items."
    logger.info("analyze_meal(): start image=%s prompt=%r", image_path, prompt)

    try:
        vision = detect_foods(image_path, prompt)
    except Exception:
        logger.exception("analyze_meal(): detect_foods() failed for image=%s",
                         image_path)
        raise

    detected = vision.get("food_items", []) if isinstance(vision, dict) else []
    logger.info("analyze_meal(): detection returned %d food item(s)",
                len(detected))

    nutrition_input = _to_nutrition_items(vision)
    if nutrition_input:
        logger.debug("analyze_meal(): nutrition input = %s", nutrition_input)
        try:
            nutrition = calculate_nutrition(nutrition_input)
        except Exception:
            logger.exception(
                "analyze_meal(): calculate_nutrition() failed for items=%s",
                nutrition_input,
            )
            raise
    else:
        logger.warning(
            "analyze_meal(): no usable food items after conversion; "
            "returning empty nutrition for image=%s",
            image_path,
        )
        nutrition = {
            "items": [], "total_nutrition": {}, "unmatched_foods": []
        }

    totals = nutrition.get("total_nutrition", {}) if isinstance(nutrition, dict) else {}
    logger.info(
        "analyze_meal(): done items=%d kcal=%.0f carbs=%.0fg sugar=%.0fg "
        "unmatched=%d",
        len(nutrition_input),
        totals.get("calories_kcal", 0) or 0,
        totals.get("carbs_g", 0) or 0,
        totals.get("sugar_g", 0) or 0,
        len(nutrition.get("unmatched_foods", []) or []) if isinstance(nutrition, dict) else 0,
    )
    if isinstance(nutrition, dict) and nutrition.get("llm_estimated_items"):
        logger.info("analyze_meal(): LLM-estimated items = %s",
                    nutrition["llm_estimated_items"])
    if isinstance(nutrition, dict) and nutrition.get("unmatched_foods"):
        logger.warning("analyze_meal(): unmatched foods = %s",
                       nutrition["unmatched_foods"])

    return {
        "vision": vision,
        "nutrition": nutrition,
    }