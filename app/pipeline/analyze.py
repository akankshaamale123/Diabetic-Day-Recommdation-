# app/pipeline/__init__.py
from .image_inference import detect_foods
from .nutrition_calculator import calculate_nutrition

__all__ = ["detect_foods", "calculate_nutrition", "analyze_meal"]


def _to_nutrition_items(vision_result: dict) -> list:
    """
    Convert the VL JSON into the [{'name':..., 'grams':...}] format
    that calculate_nutrition() expects.
    """
    items = []
    for food in vision_result.get("food_items", []):
        name = (food.get("normalized_food_name")
                or food.get("food_name"))
        grams = food.get("estimated_quantity_grams")

        if not name or grams is None or grams <= 0:
            # skip items we can't estimate
            continue

        items.append({"name": name, "grams": float(grams)})

    return items


def analyze_meal(image_path: str, user_prompt: str = None) -> dict:
    """
    End-to-end: image -> detected foods -> nutrition totals.
    Returns a single dict ready to show to the user.
    """
    prompt = user_prompt or "Identify all visible food items."

    vision = detect_foods(image_path, prompt)

    nutrition_input = _to_nutrition_items(vision)
    print("++++++++++++++++++++++++++++++++++++",nutrition_input)
    nutrition = calculate_nutrition(nutrition_input) if nutrition_input else {
        "items": [], "total_nutrition": {}, "unmatched_foods": []
    }
    print("==================================================",nutrition)

    return {
        "vision": vision,
        "nutrition": nutrition,
    }