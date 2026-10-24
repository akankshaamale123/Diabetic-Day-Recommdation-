# app/pipeline/vision_pipeline.py
"""
Vision upgrades on top of image_inference.detect_foods:

  check_quality(path)              -> {"ok", "issues", "warnings"}
  detect_multi(paths)              -> merged {"food_items": [...], ...}
  ocr_packaged(path)               -> {"nutrition_facts": {...}, "product": str}
  ocr_menu(path)                   -> {"dishes": [{"name", "price"}], ...}

Aggregation rules:
  - same food name (fuzzy) across images -> ONE item, grams summed
  - per-item `identification_confidence` = max across images
  - per-item `portion_confidence` = min across images (conservative)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from pipeline.image_inference import detect_foods, _extract_json
from client import generate_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- quality
def _laplacian_variance(img: Image.Image) -> float:
    """Higher = sharper. <50 is usually blurry."""
    gray = img.convert("L")
    lap = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(lap).var[0])


def check_quality(path: str) -> dict:
    try:
        img = Image.open(path)
    except Exception as e:
        logger.exception("check_quality(): cannot open image %s", path)
        return {"ok": False, "issues": [f"cannot open image: {e}"], "warnings": []}

    w, h = img.size
    issues, warnings = [], []

    if w < 600 or h < 600:
        issues.append(f"image is small ({w}×{h}) — use ≥ 800px")

    sharpness = _laplacian_variance(img)
    if sharpness < 50:
        issues.append("image looks blurry")

    mean = float(ImageStat.Stat(img.convert("L")).mean[0])
    if mean < 40:
        issues.append("image is very dark")
    if mean > 230:
        warnings.append("image is over-exposed")

    ok = not issues
    logger.info(
        "check_quality(%s): ok=%s size=%dx%d sharpness=%.0f mean_brightness=%.0f "
        "issues=%s warnings=%s",
        Path(path).name, ok, w, h, sharpness, mean, issues, warnings,
    )
    return {"ok": ok, "issues": issues,
            "warnings": warnings, "size": [w, h]}


# ---------------------------------------------------------------- multi-image
def _dedupe(foods: list[dict], name_threshold: float = 0.85) -> list[dict]:
    """Merge same food name across images (fuzzy)."""
    from rapidfuzz import fuzz

    logger.info("_dedupe(): %d input item(s), threshold=%.2f",
                len(foods), name_threshold)

    merged: list[dict] = []
    skipped_no_name = 0
    merge_events = 0

    for f in foods:
        name = (f.get("normalized_food_name") or f.get("food_name") or "").strip()
        if not name:
            skipped_no_name += 1
            logger.debug("_dedupe(): skipping item with no name (keys=%s)",
                         list(f.keys()))
            continue

        g = float(f.get("estimated_quantity_grams") or 0)
        hit = next((m for m in merged
                    if fuzz.WRatio(name, m["normalized_food_name"]) / 100
                       >= name_threshold), None)

        if hit:
            prev_grams = float(hit.get("estimated_quantity_grams") or 0)
            hit["estimated_quantity_grams"] = prev_grams + g
            hit["identification_confidence"] = max(
                hit.get("identification_confidence") or 0,
                f.get("identification_confidence") or 0)
            hit["portion_confidence"] = min(
                hit.get("portion_confidence") or 1,
                f.get("portion_confidence") or 1)
            hit["_from_images"] = hit.get("_from_images", 1) + 1
            merge_events += 1
            logger.debug(
                "_dedupe(): merged %r into %r — grams %.1f + %.1f = %.1f, "
                "now from %d image(s)",
                name, hit["normalized_food_name"], prev_grams, g,
                hit["estimated_quantity_grams"], hit["_from_images"],
            )
        else:
            merged.append({**f, "normalized_food_name": name,
                           "estimated_quantity_grams": g, "_from_images": 1})

    logger.info("_dedupe(): %d unique food(s) after merging — %d merge(s), "
                "%d skipped (no name)",
                len(merged), merge_events, skipped_no_name)
    if skipped_no_name:
        logger.warning("_dedupe(): %d item(s) skipped because they had no "
                       "food name", skipped_no_name)
    return merged


def detect_multi(paths: list[str]) -> dict:
    logger.info("detect_multi(): processing %d image(s)", len(paths or []))

    items: list[dict] = []
    warnings: list[str] = []
    skipped_quality = 0
    failed_inference = 0

    for idx, p in enumerate(paths or []):
        logger.debug("detect_multi(): image #%d %s", idx, Path(p).name)
        q = check_quality(p)
        if not q["ok"]:
            warnings.append(f"{Path(p).name}: " + ", ".join(q["issues"]))
            skipped_quality += 1
            logger.warning("detect_multi(): skipping %s due to quality: %s",
                           Path(p).name, q["issues"])
            continue

        try:
            detected = detect_foods(p).get("food_items", [])
            items += detected
            logger.debug("detect_multi(): %s → %d food item(s) detected",
                         Path(p).name, len(detected))
        except Exception as e:
            warnings.append(f"{Path(p).name}: {e}")
            failed_inference += 1
            logger.exception("detect_multi(): detection failed for %s",
                             Path(p).name)

    deduped = _dedupe(items)
    logger.info(
        "detect_multi(): done — %d food(s) after dedupe; "
        "%d image(s) skipped for quality, %d failed inference",
        len(deduped), skipped_quality, failed_inference,
    )

    return {
        "food_items": deduped,
        "requires_user_confirmation": True,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- OCR modes
_MENU_PROMPT = """Extract every dish from this menu photo. Return JSON:
{"dishes": [{"name": "...", "price": "...", "notes": "vegetarian? spicy?"}],
 "restaurant": "", "confidence": 0.0}
Return JSON only."""

_PACKAGED_PROMPT = """This is a packaged food. Read the nutrition panel.
Return JSON:
{"product": "", "serving_size_g": 0,
 "per_serving": {"calories_kcal":0,"carbs_g":0,"sugar_g":0,
                 "protein_g":0,"fat_g":0,"fiber_g":0,"sodium_mg":0},
 "confidence": 0.0}
Return JSON only."""


def _ocr(path: str, prompt: str) -> dict:
    logger.info("_ocr(): %s", Path(path).name)

    q = check_quality(path)
    if not q["ok"]:
        logger.warning("_ocr(): rejecting %s — quality issues: %s",
                       Path(path).name, q["issues"])
        return {"error": "; ".join(q["issues"])}

    try:
        raw = generate_image(path, prompt=prompt, stream=False) or ""
        logger.debug("_ocr(): %s — raw response %d chars",
                     Path(path).name, len(raw))
    except Exception as e:
        logger.exception("_ocr(): generate_image() failed for %s",
                         Path(path).name)
        return {"error": str(e)}

    try:
        parsed = _extract_json(raw)
        logger.info("_ocr(): %s — parsed OK (top-level keys=%s)",
                    Path(path).name, list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
        return parsed
    except Exception as e:
        logger.exception("_ocr(): JSON extraction failed for %s "
                         "(raw preview=%r)", Path(path).name, raw[:200])
        return {"error": str(e)}


def ocr_menu(path: str) -> dict:
    logger.info("ocr_menu(): %s", Path(path).name)
    return _ocr(path, _MENU_PROMPT)


def ocr_packaged(path: str) -> dict:
    logger.info("ocr_packaged(): %s", Path(path).name)
    return _ocr(path, _PACKAGED_PROMPT)