import json
import logging
import re

from client import generate_image

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    stripped = text.strip()
    had_fence = stripped.startswith("```") or stripped.endswith("```")
    result = re.sub(r"^```(?:json)?\s*", "", stripped)
    result = re.sub(r"\s*```$", "", result)
    if had_fence:
        logger.debug("_strip_code_fences(): removed code fence(s) "
                     "(before=%d chars, after=%d chars)",
                     len(stripped), len(result.strip()))
    return result.strip()


def _extract_json(text: str) -> dict:
    """Try hard to parse a JSON object out of the model output."""
    cleaned = _strip_code_fences(text)
    logger.debug("_extract_json(): attempting direct JSON parse (%d chars)",
                 len(cleaned))

    try:
        data = json.loads(cleaned)
        logger.debug("_extract_json(): direct parse succeeded")
        return data
    except json.JSONDecodeError as e:
        logger.debug("_extract_json(): direct parse failed (%s); "
                     "falling back to brace extraction", e)

    # Fallback: grab the first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        logger.debug("_extract_json(): brace span found [%d:%d], retrying",
                     start, end + 1)
        try:
            data = json.loads(cleaned[start:end + 1])
            logger.debug("_extract_json(): brace-extracted parse succeeded")
            return data
        except json.JSONDecodeError as e:
            logger.debug("_extract_json(): brace-extracted parse failed: %s", e)

    logger.error("_extract_json(): could not parse JSON from model output "
                 "(len=%d, preview=%r)",
                 len(text), text[:500])
    raise ValueError(f"Model did not return valid JSON:\n{text[:500]}")


def detect_foods(image_path: str,
                 user_prompt: str = "Identify all visible food items.") -> dict:
    """
    Runs Qwen-VL on the image and returns the parsed JSON dict
    with the schema defined in Qwen_VL_prompts.
    """
    logger.info("detect_foods(): image=%s prompt=%r", image_path, user_prompt)

    user_prompt= """Look at the image very carefully.
    List every separate food item on the plate.
    Do not group foods together.
    Include small items, side items, and partially hidden items.
    For each item give:
    - food name
    - estimated grams
    - where it is on the plate (top-left, center, bottom-right, etc.)
    After listing, check again: did you miss any small or partly hidden item?"""
    
    try:
        logger.info(f"IMG PATH = {image_path} | {user_prompt}")
        raw = generate_image(image_path, prompt=user_prompt, stream=False) 
        logger.info(f"[RAW] {raw}")
    except Exception:
        logger.exception("detect_foods(): generate_image() failed for %s",
                         image_path)
        raise

    if not raw:
        logger.error("detect_foods(): empty response from VL model "
                     "(image=%s)", image_path)
        raise RuntimeError("Empty response from VL model.")

    logger.info("detect_foods(): received raw response (%d chars)", len(raw))
    logger.debug("detect_foods(): raw response preview=%r", raw[:500])

    try:
        parsed = _extract_json(raw)
    except ValueError:
        logger.exception("detect_foods(): JSON extraction failed for image=%s",
                         image_path)
        raise

    n_items = len(parsed.get("food_items", []) or []) if isinstance(parsed, dict) else 0
    logger.info("detect_foods(): parsed OK, %d food item(s) detected", n_items)
    return parsed