# app/pipeline/meal_coach.py
"""
Conversational nutrition coach with patient context.

  coach_reply(user, history, message) -> str

Stateless: rebuilds a structured prompt each turn using:
  - patient profile
  - last 7 days of meals (calories, carbs, protein, fibre roll-up)
  - personal risky foods from glucose history
  - current rules verdict for the last saved meal
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from client import generate_text
from storage import user_db
from pipeline import rules as rule_engine
from pipeline.safety import (DISCLAIMER, REFUSAL_MESSAGE, is_unsafe,
                             with_disclaimer)

logger = logging.getLogger(__name__)


_SYSTEM = """You are a diabetes nutrition coach for an app.
You help a user understand and improve their meals.

RULES YOU MUST FOLLOW:
1. Never give medication advice (insulin doses, changing pills) and never
   diagnose. If asked, say you can't help and defer to the clinician.
2. Base answers ONLY on the PATIENT CONTEXT and EVIDENCE below. Do not
   invent numbers.
3. When you suggest a meal, keep it to Indian / South-Asian foods unless
   the user asks otherwise, and give approximate grams and macros.
4. Every answer ends with the phrase: "Follow your clinician's targets."
5. Keep answers under 180 words. Use short bullets, not long paragraphs.
"""


def _recent_rollup(user_id: int, days: int = 7) -> dict:
    now = pd.Timestamp.now()
    logger.debug("_recent_rollup(): user_id=%s, days=%d, window %s → %s",
                 user_id, days,
                 (now - pd.Timedelta(days=days)).date(), now.date())

    try:
        df = user_db.load_meals(user_id, now - pd.Timedelta(days=days), now)
    except Exception:
        logger.exception("_recent_rollup(): load_meals() failed for user_id=%s",
                         user_id)
        return {"meals": 0}

    if df.empty:
        logger.info("_recent_rollup(): no meals in last %d day(s) for user_id=%s",
                    days, user_id)
        return {"meals": 0}

    rollup = {
        "meals": int(len(df)),
        "avg_carbs_g": round(float(df["carbs_g"].mean()), 1),
        "avg_protein_g": round(float(df["protein_g"].mean()), 1),
        "avg_fiber_g": round(float(df["fiber_g"].mean()), 1),
        "avg_sugar_g": round(float(df["sugar_g"].mean()), 1),
        "last_meal": {
            "when": str(df.iloc[-1]["timestamp"])[:16],
            "type": df.iloc[-1]["meal_type"],
            "carbs_g": round(float(df.iloc[-1]["carbs_g"]), 1),
            "sugar_g": round(float(df.iloc[-1]["sugar_g"]), 1),
            "fiber_g": round(float(df.iloc[-1]["fiber_g"]), 1),
            "protein_g": round(float(df.iloc[-1]["protein_g"]), 1),
        },
    }
    logger.info("_recent_rollup(): %d meal(s) for user_id=%s "
                "(avg_carbs=%.1f, avg_sugar=%.1f)",
                rollup["meals"], user_id,
                rollup["avg_carbs_g"], rollup["avg_sugar_g"])
    return rollup


def _risky_foods(user_id: int) -> list[str]:
    try:
        from pipeline.glucose_analysis import summarize_patterns
        from pipeline.user_charts import journey_start
        u = {"id": user_id, "created_at": "1970-01-01"}  # placeholder
        start = pd.Timestamp.now() - pd.Timedelta(days=30)
        s = summarize_patterns(start, pd.Timestamp.now())
        risky = s.get("high_response_foods", [])
        logger.debug("_risky_foods(): user_id=%s → %d risky food(s): %s",
                     user_id, len(risky), risky)
        return risky
    except Exception:
        logger.exception("_risky_foods(): failed to compute risky foods for "
                         "user_id=%s — returning empty list", user_id)
        return []


def _build_context(user: dict) -> dict:
    logger.debug("_build_context(): building context for user_id=%s",
                 user.get("id"))
    ctx = {
        "diabetes_type": user.get("diabetes_type", "type2"),
        "baseline_mgdl": float(user.get("baseline_mgdl", 130)),
        "last_7_days": _recent_rollup(user["id"]),
        "risky_foods": _risky_foods(user["id"]),
        "rules_limits": rule_engine.LIMITS,
    }
    logger.debug("_build_context(): context built — meals=%s, risky_foods=%d",
                 ctx["last_7_days"].get("meals"),
                 len(ctx["risky_foods"]))
    return ctx


def _content_to_text(content) -> str:
    """Gradio 'messages' content can be a str OR a list of blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "image":
                    parts.append("[image attached]")
                elif block.get("type") == "file":
                    parts.append("[file attached]")
        return " ".join(parts).strip()
    return str(content or "").strip()


def _history_to_text(history: list[dict], max_turns: int = 6) -> str:
    turns = history[-max_turns:] if history else []
    out = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Coach"
        out.append(f"{role}: {_content_to_text(t.get('content'))}")
    logger.debug("_history_to_text(): kept %d/%d turn(s) for prompt",
                 len(turns), len(history or []))
    return "\n".join(out)


def coach_reply(user: dict, history: list[dict], message: str,
                extra_context: dict | None = None) -> str:
    """extra_context: dict injected into the prompt — e.g. the meal the
    user just analyzed in the Upload tab. Pass None for plain chat."""
    logger.info("coach_reply(): user_id=%s, message_len=%d, history_len=%d, "
                "has_extra_context=%s",
                user.get("id"), len(message or ""), len(history or []),
                bool(extra_context))

    # 1. Hard refusal first — never let the LLM near this
    if is_unsafe(message):
        logger.warning("coach_reply(): unsafe message detected — returning "
                       "refusal for user_id=%s", user.get("id"))
        return with_disclaimer(REFUSAL_MESSAGE)

    # 2. Build prompt (patient context + optional in-progress meal)
    try:
        ctx = _build_context(user)
    except Exception:
        logger.exception("coach_reply(): _build_context() failed for "
                         "user_id=%s — falling back to minimal context",
                         user.get("id"))
        ctx = {"diabetes_type": user.get("diabetes_type", "type2")}

    if extra_context:
        ctx["in_progress_meal"] = extra_context
        logger.debug("coach_reply(): injected in_progress_meal "
                     "(source=%s)", (extra_context or {}).get("source"))

    prompt = (
        f"{_SYSTEM}\n\n"
        f"PATIENT CONTEXT (JSON):\n{json.dumps(ctx, indent=2, default=str)}\n\n"
        f"CONVERSATION SO FAR:\n{_history_to_text(history)}\n\n"
        f"USER: {message}\n\n"
        f"Respond as the Coach. End with 'Follow your clinician's targets.'"
    )
    logger.debug("coach_reply(): prompt built (%d chars)", len(prompt))

    # 3. Call LLM
    try:
        reply = generate_text(prompt, stream=False).strip()
        logger.info("coach_reply(): LLM replied (%d chars)", len(reply))
    except Exception as e:
        logger.exception("coach_reply(): generate_text() failed for "
                         "user_id=%s — using safe fallback", user.get("id"))
        reply = (f"Sorry — the coach is unavailable right now ({e}). "
                 "Here's a safe default: keep carbs per meal at 45–60 g, "
                 "pair every carb with protein, and walk 10–15 minutes after eating.")

    # 4. Guardrails
    if "clinician" not in reply.lower():
        logger.debug("coach_reply(): appending clinician reminder to reply")
        reply += "\n\nFollow your clinician's targets."
    return with_disclaimer(reply) 