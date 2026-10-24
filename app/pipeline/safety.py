# app/pipeline/safety.py
"""
Medical trust layer.

Single source of truth for:
  - the disclaimer shown with EVERY advice / chart
  - refusal patterns (medication dosing, diagnosis, emergencies)
  - evidence strings that the rule engine cites
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "⚠️ **This is general nutrition guidance based on published guidelines "
    "(ADA 2024, ICMR 2020) and a population-level glucose model — "
    "it is not a diagnosis and not a substitute for your clinician. "
    "Do not change insulin or medication based on this app.**"
)

EVIDENCE = {
    "ada_carbs_45_60": (
        "ADA Standards of Care 2024, Nutrition Therapy: 45–60 g carbs "
        "per meal for most adults with diabetes (individualised)."),
    "ada_sugar_added": (
        "ADA 2024: limit added sugars to <10% of daily calories; "
        "prefer low-glycaemic carbohydrate sources."),
    "icmr_fiber": (
        "ICMR Dietary Guidelines 2020: 25–30 g dietary fibre per day "
        "(~7–10 g per main meal helps blunt glucose rise)."),
    "protein_floor": (
        "1.0–1.2 g/kg/day protein for adults with type 2 diabetes "
        "(ADA 2024) — ~12–20 g per main meal."),
    "post_meal_walk": (
        "Reynolds et al. 2016 (Diabetologia): 10–15 min post-meal walking "
        "reduces postprandial glucose AUC by up to 30%."),
    "glycaemic_index": (
        "ICMR 2020 & ADA 2024: prefer low-GI (<55) whole grains "
        "(jowar, bajra, ragi, brown rice) over refined grains."),
    "personal_history": (
        "Model-estimated response from your own logged meals "
        "(RF trained on CGMacros). Population-level estimate."),
}

# Hard refusal — never overridable by the LLM
_REFUSAL_PATTERNS = [
    ("insulin dose", "How much insulin"),
    ("medication change", "Stopping or changing medication"),
    ("diagnosis", "Diagnosing a condition"),
    ("emergency", "Chest pain, severe hypoglycaemia, or DKA symptoms"),
]

REFUSAL_MESSAGE = (
    "I can't help with medication doses, diagnoses, or emergencies. "
    "**If you feel very low (shaking, sweating, confusion) eat 15 g fast sugar "
    "and call your doctor. If you have chest pain or trouble breathing, call emergency services.** "
    "For anything else, please ask your clinician."
)

# Flat trigger list — every entry is a substring match against the lowercased message.
_TRIGGERS = (
    "insulin dose", "how much insulin", "units of insulin",
    "stop my medication", "change my dose",
    "am i diabetic", "do i have diabetes",
    "chest pain", "can't breathe", "passed out",
)


def is_unsafe(message: str) -> bool:
    """
    Return True if the message asks for medication dosing, diagnosis, or
    describes an emergency. This must never be bypassable by the LLM — the
    caller checks this BEFORE building a prompt.
    """
    m = (message or "").lower()

    if not m:
        # Empty message is not unsafe; caller handles empty separately.
        logger.debug("is_unsafe(): empty message → False")
        return False

    for trigger in _TRIGGERS:
        if trigger in m:
            # Log the *trigger string* (safe — from our controlled list),
            # not the raw user message (may contain PHI).
            logger.warning(
                "is_unsafe(): message flagged unsafe — matched trigger %r "
                "(message_len=%d)", trigger, len(message),
            )
            return True

    # Not unsafe — log only at DEBUG so the per-turn chat flow isn't noisy.
    logger.debug("is_unsafe(): message passed safety check (message_len=%d)",
                 len(message))
    return False


def with_disclaimer(text: str) -> str:
    appended = f"{text.rstrip()}\n\n---\n{DISCLAIMER}"
    logger.debug("with_disclaimer(): appended disclaimer "
                 "(input_len=%d, output_len=%d)", len(text), len(appended))
    return appended


# Log once at import time so the safety layer's readiness is visible in the
# startup log even if nothing in this module is called later.
logger.info("Safety layer loaded: %d evidence key(s), %d refusal trigger(s)",
            len(EVIDENCE), len(_TRIGGERS))