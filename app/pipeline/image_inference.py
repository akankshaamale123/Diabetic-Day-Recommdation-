import json
import re
from client import generate_image



def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Try hard to parse a JSON object out of the model output."""
    cleaned = _strip_code_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Model did not return valid JSON:\n{text[:500]}")


def detect_foods(image_path: str,
                 user_prompt: str = "Identify all visible food items.") -> dict:
    """
    Runs Qwen-VL on the image and returns the parsed JSON dict
    with the schema defined in Qwen_VL_prompts.
    """
    raw = generate_image(image_path, prompt=user_prompt, stream=False)

    if not raw:
        raise RuntimeError("Empty response from VL model.")
    print(raw)
    return _extract_json(raw)