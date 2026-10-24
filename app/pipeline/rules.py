# app/pipeline/rules.py
"""
Rule engine for meal assessment. Every issue carries evidence
so the UI can explain *why* a verdict was reached.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from .safety import EVIDENCE

logger = logging.getLogger(__name__)

Severity = Literal["info", "warn", "danger"]


@dataclass
class RuleResult:
    id: str
    severity: Severity
    title: str
    detail: str
    evidence: list[str] = field(default_factory=list)


# per-meal clinician-adjustable limits
LIMITS = {
    "breakfast": {"carbs_g": 45, "sugar_g": 10, "fiber_g": 5,
                  "protein_g": 12, "kcal": 450},
    "lunch":     {"carbs_g": 60, "sugar_g": 12, "fiber_g": 7,
                  "protein_g": 20, "kcal": 600},
    "dinner":    {"carbs_g": 60, "sugar_g": 12, "fiber_g": 7,
                  "protein_g": 20, "kcal": 600},
    "snack":     {"carbs_g": 20, "sugar_g": 5,  "fiber_g": 2,
                  "protein_g": 5,  "kcal": 200},
}

# 2-point thresholds -> verdict buckets
_WEIGHTS = {"info": 0, "warn": 1, "danger": 2}


def evaluate(totals: dict, meal_type: str,
             risky_foods: list[str] | None = None,
             detected_names: list[str] | None = None) -> list[RuleResult]:
    # -- entry: log inputs (key numbers + personalization counts) --
    logger.info(
        "evaluate(): meal_type=%s, risky_foods=%d, detected_names=%d",
        meal_type, len(risky_foods or []), len(detected_names or []),
    )

    lim = LIMITS.get(meal_type)
    if lim is None:
        logger.warning("evaluate(): unknown meal_type=%r — falling back to "
                       "'lunch' limits", meal_type)
        lim = LIMITS["lunch"]

    r: list[RuleResult] = []

    carbs   = float(totals.get("carbs_g")   or 0)
    sugar   = float(totals.get("sugar_g")   or 0)
    fiber   = float(totals.get("fiber_g")   or 0)
    protein = float(totals.get("protein_g") or 0)
    kcal    = float(totals.get("calories_kcal") or 0)

    logger.debug(
        "evaluate(): inputs — carbs=%.1f/%.0f sugar=%.1f/%.0f "
        "fiber=%.1f/%.0f protein=%.1f/%.0f kcal=%.0f/%.0f",
        carbs, lim["carbs_g"], sugar, lim["sugar_g"],
        fiber, lim["fiber_g"], protein, lim["protein_g"],
        kcal, lim["kcal"],
    )

    if carbs > lim["carbs_g"]:
        over = carbs / lim["carbs_g"]
        severity = "danger" if over > 1.3 else "warn"
        logger.info("evaluate(): carbs_over fired (carbs=%.0fg, limit=%d, "
                    "ratio=%.2f → severity=%s)",
                    carbs, lim["carbs_g"], over, severity)
        r.append(RuleResult(
            id="carbs_over",
            severity=severity,
            title=f"Carbs {carbs:.0f} g exceeds {lim['carbs_g']} g",
            detail=(f"Your {meal_type} had {carbs:.0f} g carbs. "
                    f"The guideline limit for {meal_type} is {lim['carbs_g']} g."),
            evidence=[EVIDENCE["ada_carbs_45_60"]]))

    if sugar > lim["sugar_g"]:
        logger.info("evaluate(): sugar_over fired (sugar=%.0fg, limit=%d)",
                    sugar, lim["sugar_g"])
        r.append(RuleResult(
            id="sugar_over", severity="warn",
            title=f"Sugar {sugar:.0f} g exceeds {lim['sugar_g']} g",
            detail="Added/free sugar pushes glucose up quickly.",
            evidence=[EVIDENCE["ada_sugar_added"]]))

    if fiber < lim["fiber_g"]:
        logger.info("evaluate(): fiber_low fired (fiber=%.0fg, min=%d)",
                    fiber, lim["fiber_g"])
        r.append(RuleResult(
            id="fiber_low", severity="warn",
            title=f"Fibre {fiber:.0f} g < {lim['fiber_g']} g",
            detail="Fibre slows glucose absorption — your meal is low in it.",
            evidence=[EVIDENCE["icmr_fiber"]]))

    if protein < lim["protein_g"]:
        logger.info("evaluate(): protein_low fired (protein=%.0fg, min=%d)",
                    protein, lim["protein_g"])
        r.append(RuleResult(
            id="protein_low", severity="warn",
            title=f"Protein {protein:.0f} g < {lim['protein_g']} g",
            detail="Pairing carbs with protein blunts the glucose spike.",
            evidence=[EVIDENCE["protein_floor"]]))

    if kcal > lim["kcal"]:
        logger.info("evaluate(): kcal_over fired (kcal=%.0f, limit=%d)",
                    kcal, lim["kcal"])
        r.append(RuleResult(
            id="kcal_over", severity="info",
            title=f"{kcal:.0f} kcal above {lim['kcal']} kcal",
            detail="High-calorie meals contribute to weight gain over time.",
            evidence=[]))

    # personal history
    hits = []
    for name in detected_names or []:
        for risky in risky_foods or []:
            if risky.lower() in name.lower() or name.lower() in risky.lower():
                hits.append(name)
                break
    if hits:
        unique_hits = sorted(set(hits))
        logger.info("evaluate(): personal_risk fired — detected foods matched "
                    "user's risky history: %s", unique_hits)
        r.append(RuleResult(
            id="personal_risk", severity="danger",
            title="History of spikes with: " + ", ".join(unique_hits),
            detail="Your own logged meals spiked glucose after these foods before.",
            evidence=[EVIDENCE["personal_history"]]))
    elif risky_foods:
        logger.debug("evaluate(): personal_risk not triggered — detected "
                     "foods %s had no overlap with risky_foods %s",
                     detected_names or [], risky_foods)

    # always give a positive when nothing fired
    if not r:
        logger.info("evaluate(): no rules fired → 'all_good' positive result")
        r.append(RuleResult(
            id="all_good", severity="info",
            title="Balanced meal",
            detail="Carbs, sugar, fibre and protein are all within range.",
            evidence=[EVIDENCE["ada_carbs_45_60"]]))

    # -- exit: summarize fired rule ids + severities --
    logger.info(
        "evaluate(): fired %d rule(s) — ids=%s severities=%s",
        len(r),
        [rule.id for rule in r],
        [rule.severity for rule in r],
    )
    return r


def verdict(rules: list[RuleResult]) -> tuple[str, str]:
    """('good'|'okay'|'not_suitable', human label)"""
    # Exclude the positive-only marker from scoring
    scoring = [r for r in rules if r.id not in ("all_good",)]
    score = sum(_WEIGHTS[r.severity] for r in scoring)

    if score == 0:
        v, label = "good", "Good choice for you"
    elif score <= 2:
        v, label = "okay", "Acceptable, but could be better"
    else:
        v, label = "not_suitable", "Not ideal for diabetes"

    logger.info(
        "verdict(): %d rule(s) counted (excluding 'all_good'), "
        "weighted score=%d → verdict=%s (%s)",
        len(scoring), score, v, label,
    )
    if scoring and score >= 3:
        logger.debug("verdict(): danger rules driving 'not_suitable' = %s",
                     [r.id for r in scoring if r.severity == "danger"])
    return v, label