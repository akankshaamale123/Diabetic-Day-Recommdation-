# app/app.py — Diabetic Diet Assistant (upload meal -> advice -> graphs)
# The user is fixed (see current_user.py); login is handled by another team.
from __future__ import annotations
import logging
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go


# from current_user import get_current_user
from pipeline.image_inference import detect_foods
from pipeline import analyze_meal
from pipeline.meal_advisor import assess_meal, breakfast_advice, to_markdown
from pipeline.nutrition_calculator import calculate_nutrition
from pipeline.portion import vision_to_rows, rows_to_items
from pipeline.user_charts import (
    journey_days, journey_start, model_note,
    plot_today, plot_whatif, whatif_text,
    plot_journey, plot_today_vs_past, compare_table, plot_nutrition_radar , plot_user_scatter  
)
from pipeline.safety import DISCLAIMER, with_disclaimer
from storage import user_db
from pipeline.vision_pipeline import detect_multi, ocr_menu, ocr_packaged
from pipeline.meal_coach import coach_reply


# ------------------------------------------------------------ logging
logger = logging.getLogger(__name__)


MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]
LABELS = {
    "calories_kcal": "Calories (kcal)", "carbs_g": "Carbs (g)",
    "protein_g": "Protein (g)", "fat_g": "Fat (g)", "sugar_g": "Sugar (g)",
    "fiber_g": "Fiber (g)", "cholesterol_mg": "Cholesterol (mg)",
    "calcium_mg": "Calcium (mg)",
}


DEMO_USER = {
    "username": "demo_user",
    "diabetes_type": "type2",
    "baseline_mgdl": 140.0,                 # typical fasting glucose
    "journey_start": "2026-10-15T08:00:00", # "login date": day 1 of the journey
}


def get_current_user() -> dict:
    user_db.init_user_tables()
    user = user_db.get_or_create_user(
        DEMO_USER["username"],
        DEMO_USER["diabetes_type"],
        DEMO_USER["baseline_mgdl"],
        created_at=DEMO_USER["journey_start"],
    )
    logger.info(
        "Current user resolved: id=%s username=%s diabetes_type=%s",
        user.get("id"), user.get("username"), user.get("diabetes_type"),
    )
    return user


CURRENT_USER = get_current_user()

def get_all_users():
    user_db.init_user_tables()
    with user_db._session() as c:
        rows = c.execute("SELECT id, username FROM users").fetchall()
    if not rows:
        return [f"{CURRENT_USER['id']}: {CURRENT_USER['username']}"]
    return [f"{r[0]}: {r[1]}" for r in rows]

def load_user_from_choice(choice: str):
    user_id = int(choice.split(":")[0])
    with user_db._session() as c:
        row = c.execute("SELECT id, username, diabetes_type, baseline_mgdl, created_at FROM users WHERE id=?", (user_id,)).fetchone()
    return {
        "id": row[0], "username": row[1], "diabetes_type": row[2],
        "baseline_mgdl": row[3], "journey_start": row[4]
    }


def _paths(files):
    """Normalise whatever Gradio gives us -> list[str]."""
    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [str(files)]
    out = []
    for f in files:
        p = getattr(f, "path", None) or getattr(f, "name", None) or f
        if p:
            out.append(str(p))
    return out


def detect(file_list, mode):
    paths = _paths(file_list)
    logger.info("detect() called: mode=%s, files=%d", mode, len(paths))
    if not paths:
        logger.warning("detect() aborted: no images uploaded")
        raise gr.Error("Upload at least one image.")

    # helper: hide the extra-photo UI when we don't need it
    hide = gr.update(visible=False)

    if mode == "Restaurant Food":
        out = ocr_menu(paths[0])
        if out.get("error"):
            logger.error("Menu OCR failed: %s", out["error"])
            raise gr.Error(out["error"])
        rows = [[d.get("name", ""), 1, d.get("notes", "")]
                for d in out.get("dishes", [])]
        logger.info("Menu parsed: %d dishes", len(rows))
        return (rows,
                f"### 📋 Menu parsed (confidence {out.get('confidence', 0):.0%})",
                None, hide, hide)

    if mode == "Packaged food":
        out = ocr_packaged(paths[0])
        if out.get("error"):
            logger.error("Packaged OCR failed: %s", out["error"])
            raise gr.Error(out["error"])
        n = out.get("per_serving", {})
        rows = [[out.get("product", "product"),
                 out.get("serving_size_g", 100),
                 f"from label, confidence {out.get('confidence', 0):.0%}"]]
        logger.info("Packaged food parsed: product=%s", out.get("product"))
        return (rows,
                f"### 📦 Label parsed\nper serving: {n.get('carbs_g', '?')} g carbs, "
                f"{n.get('sugar_g', '?')} g sugar",
                None, hide, hide)

    # ---- meal mode ----
    out = detect_multi(paths)
    rows = vision_to_rows(out)
    quality = out.get("quality", {}) or {}

    if not rows:
        logger.warning("detect(): no food detected in %d image(s)", len(paths))
        # show the extra-photo UI: a second angle may rescue detection
        return (
            [["", 100, "AI could not detect food. Please type manually."]],
            "### 🍽️ No food detected automatically.\n"
            "Try uploading another photo from a different angle, "
            "or click **+** on the table to add foods manually.",
            quality,
            gr.update(visible=True),
            gr.update(visible=True),
        )

    warn = "\n".join(f"- {w}" for w in out.get("warnings", []))

    # build the quality warning block if detection looks weak
    need_more = bool(quality.get("should_ask_another_photo"))
    quality_md = ""
    if need_more:
        reasons = "\n".join(f"- {r}" for r in quality.get("reasons", []))
        quality_md = (
            "\n\n---\n\n"
            "⚠️ **I might have missed some items on this plate:**\n"
            f"{reasons}\n\n"
            f"**{quality.get('suggestion', '')}**"
        )

    logger.info("Detection produced %d food row(s); ask_more=%s",
                len(rows), need_more)

    return (
        rows,
        "### 🍽️ Review foods\n"
        + (f"**Warnings:**\n{warn}\n\n" if warn else "")
        + "Edit the name or grams, then press **Calculate nutrition**."
        + quality_md,
        quality,
        gr.update(visible=need_more),
        gr.update(visible=need_more),
    )

def detect_with_extra(orig_files, extra_files, mode):
    """Merge first-pass photos + extra-angle photos and re-run detection."""
    orig  = _paths(orig_files)
    extra = _paths(extra_files)
    logger.info("detect_with_extra(): %d original + %d extra",
                len(orig), len(extra))
    if not extra:
        raise gr.Error("Upload at least one extra photo first.")
    return detect(orig + extra, mode)

# ------------------------------------------------------------ Step 2: nutrition + advice
def calculate(table, meal_type):
    logger.info("calculate() called: meal_type=%s", meal_type)
    items = rows_to_items(table)
    if not items:
        logger.warning("calculate() aborted: no valid items in food table")
        raise gr.Error("Add at least one food with grams greater than 0.")

    nutrition = calculate_nutrition(items) 
    logger.info(f"NUTRITION {nutrition}") 
    totals = nutrition.get("total_nutrition", {})
    if not any(totals.values()):
        unmatched = nutrition.get("unmatched_foods", [])
        logger.warning("calculate() aborted: nutrition totals empty; unmatched=%s",
                       unmatched)
        raise gr.Error("Nutrition could not be calculated for these foods. "
                       f"Not found: {', '.join(unmatched) or 'unknown'}")

    logger.info(
        "Nutrition calculated: items=%d, kcal=%.0f, carbs=%.0fg, sugar=%.0fg",
        len(items),
        totals.get("calories_kcal", 0) or 0,
        totals.get("carbs_g", 0) or 0,
        totals.get("sugar_g", 0) or 0,
    )
    if nutrition.get("llm_estimated_items"):
        logger.info("LLM-estimated items: %s", nutrition["llm_estimated_items"])
    if nutrition.get("unmatched_foods"):
        logger.warning("Unmatched foods: %s", nutrition["unmatched_foods"])

    md = "### 📊 Total Nutrition\n| Nutrient | Amount |\n|---|---|\n"
    md += "".join(f"| {LABELS.get(k, k)} | {v} |\n" for k, v in totals.items())
    if nutrition.get("llm_estimated_items"):
        md += f"\n⚠️ AI-estimated values for: {', '.join(nutrition['llm_estimated_items'])}\n"
    if nutrition.get("unmatched_foods"):
        md += f"\n❌ Not found in database: {', '.join(nutrition['unmatched_foods'])}\n"

    md += "\n---\n"
    if meal_type == "breakfast":
        fake_vision = {"food_items": [{"food_name": i["name"]} for i in items]}
        md += to_markdown(breakfast_advice({"vision": fake_vision, "nutrition": nutrition}))
    else:
        a = assess_meal(totals, "lunch")
        logger.debug("Meal verdict: %s (%d rule(s))", a["label"], len(a.get("rules", [])))
        md += f"### Verdict: {a['label']}\n"
        for rule in a.get("rules", []):
            md += f"- **{rule['title']}** — {rule['detail']}\n"

    # be defensive: not every verdict carries an "issues" list
        if isinstance(a, dict):
            label  = a.get("label") or a.get("verdict") or "Unknown"
            issues = a.get("issues") or a.get("warnings") or a.get("flags") or []
        else:
            label  = getattr(a, "label", None) or getattr(a, "verdict", None) or "Unknown"
            issues = (getattr(a, "issues", None)
                      or getattr(a, "warnings", None)
                      or getattr(a, "flags", None) or [])
        issues = list(issues or [])

        logger.debug("Meal verdict: %s (%d issue(s))", label, len(issues))
        md += f"### Verdict: {label}\n"
        if issues:
            md += "".join(f"- {i}\n" for i in issues)
        else:
            md += "_No issues detected for this meal._\n"

    md += "\n\nPress **Save to my diary** to add it to your graphs."

    return with_disclaimer(md), {"totals": totals, "foods": items}, gr.update(interactive=True)


# ------------------------------------------------------------ Step 3: save -> graphs
def save_meal(pending, meal_type, when, user):
    logger.info("save_meal() called: meal_type=%s, when=%r", meal_type, when)

    # Defensive: this branch should be unreachable because the button is
    # disabled, but the user can still trigger it via keyboard/tab order.
    if not pending or not pending.get("totals"):
        logger.warning("save_meal() blocked: no pending analysis "
                       "(calculate was not run, or table was edited)")
        return (
            "⚠️ **Please press “2. Calculate nutrition” before saving.**",
            gr.update(), gr.update(),      # leave the previous plots untouched
            None,                          # pending is still None
            gr.update(interactive=False),  # keep the button disabled
        )

    try:
        ts = pd.Timestamp(when) if str(when).strip() else pd.Timestamp.now()
    except Exception:
        logger.exception("save_meal(): invalid timestamp %r", when)
        raise gr.Error("Meal time must look like 2026-10-21 08:30 "
                       "(or leave empty for now).")

    try:
        user_db.add_meal(user["id"], ts, meal_type, pending["totals"], pending["foods"])
        logger.info("Meal saved: user_id=%s, type=%s, ts=%s, foods=%d",
                    user["id"], meal_type, ts, len(pending.get("foods", [])))
    except Exception:
        logger.exception("save_meal(): failed to persist meal for user_id=%s", user["id"])
        raise gr.Error("Could not save the meal. Please try again.")

    day = ts.strftime("%Y-%m-%d")
    note = (f"✅ Saved {meal_type} at {ts:%Y-%m-%d %H:%M}.  \n"
            f"{whatif_text(user, day)}  \n*{model_note()}*")

    # pending=None so a second click can't double-save; disable the button too
    return (with_disclaimer(note), plot_today(user, day), plot_today_vs_past(user, day),
            None, gr.update(interactive=False))


# ------------------------------------------------------------ Graphs tab
def show_charts(day, user):
    logger.info("show_charts() called: day=%s, user=%s", day, user.get("username"))
    try:
        from pipeline.glucose_charts import _empty
        from pipeline.user_charts import journey_start

        # 1. NEW: Use the user-specific scatter plot
        scatter_plot = plot_user_scatter(user, day) 

        # 2. Radar charts
        radar_today = plot_nutrition_radar(user, day=day, is_weekly=False)
        radar_week = plot_nutrition_radar(user, day=day, is_weekly=True)

        return (
            plot_today(user, day), 
            plot_whatif(user, day),
            plot_today_vs_past(user, day), 
            compare_table(user, day),
            plot_journey(user), 
            whatif_text(user, day) + f"  \n*{model_note()}*",
            scatter_plot,
            radar_today,   
            radar_week     
        )
        
    except Exception as e:
        logger.exception("show_charts() failed")
        from pipeline.glucose_charts import _empty
        empty = _empty("Error", str(e))
        return (empty, empty, empty, empty, empty, f"Error: {str(e)}", empty, empty, empty)
    
def chat_fn(message, history, user, pending, food_table):
    if not (message or "").strip():
        logger.debug("chat_fn(): empty message, ignoring")
        return history or [], ""

    logger.info("chat_fn(): user=%s, message_len=%d, history_len=%d",
                user.get("username"), len(message), len(history or []))

    history = list(history or [])
    history.append({"role": "user", "content": message})

    extra = _context_from(pending, food_table)
    try:
        reply = coach_reply(user, history[:-1], message, extra_context=extra)
    except Exception:
        logger.exception("chat_fn(): coach_reply failed for user=%s",
                         user.get("username"))
        reply = "Sorry, I couldn't process that just now. Please try again."

    history.append({"role": "assistant", "content": reply})
    return history, ""


def _context_from(pending, food_table):
    """
    Best-effort meal context. Priority:
      1. pending['totals']  -> full analyzed nutrition
      2. food_table rows    -> detected foods (grams only, no macros)
      3. None               -> no meal yet
    """
    if pending and pending.get("totals"):
        logger.debug("_context_from(): using analyzed meal (foods=%d)",
                     len(pending.get("foods", [])))
        return {
            "source": "analyzed_meal",
            "totals": pending["totals"],
            "foods": pending.get("foods", []),
        }

    try:
        rows = food_table.values.tolist() if hasattr(food_table, "values") else (food_table or [])
    except Exception:
        logger.exception("_context_from(): could not read food_table")
        rows = []

    foods = []
    for r in rows:
        if not r or len(r) < 2:
            continue
        name = str(r[0]).strip() if r[0] is not None else ""
        try:
            grams = float(r[1])
        except (TypeError, ValueError):
            continue
        if name and name.lower() != "nan" and grams > 0:
            foods.append({"name": name, "grams": grams})

    if not foods:
        logger.debug("_context_from(): no meal context available")
        return None
    logger.debug("_context_from(): using detected (not analyzed) meal, foods=%d",
                 len(foods))
    return {
        "source": "detected_not_analyzed",
        "foods": foods,
        "note": ("Foods from the uploaded photo. Nutrition totals not yet "
                 "calculated — estimate macros from these foods if the user asks."),
    }


def ask_about_last_meal(user, pending, food_table):
    logger.info("ask_about_last_meal() called for user=%s", user.get("username"))
    ctx = _context_from(pending, food_table)
    if ctx is None:
        logger.info("ask_about_last_meal(): no meal context, prompting user")
        return ([{"role": "assistant",
                  "content": ("I don't see a meal yet. Upload a photo and press "
                              "**1. Detect foods** — then I can review it even "
                              "before you calculate nutrition.")}], "")

    if ctx["source"] == "analyzed_meal":
        t = ctx["totals"]
        q = (f"Please review the meal I just analyzed. Totals: "
             f"{t.get('carbs_g', 0):.0f} g carbs, {t.get('sugar_g', 0):.0f} g sugar, "
             f"{t.get('fiber_g', 0):.0f} g fibre, {t.get('protein_g', 0):.0f} g protein, "
             f"{t.get('calories_kcal', 0):.0f} kcal. "
             "Why might this be risky for me, and what should I change next time?")
    else:
        food_str = ", ".join(f"{f['name']} ({f['grams']:.0f} g)"
                             for f in ctx["foods"])
        q = (f"I just uploaded a meal photo. The foods detected are: {food_str}. "
             "Is this a good meal for my diabetes? Why or why not? "
             "What should I add, remove, or swap to make it better?")

    try:
        reply = coach_reply(user, [], q, extra_context=ctx)
    except Exception:
        logger.exception("ask_about_last_meal(): coach_reply failed")
        reply = "Sorry, I couldn't analyze that just now. Please try again."
    return ([{"role": "user", "content": q},
             {"role": "assistant", "content": reply}], "")


# ------------------------------------------------------------ UI
with gr.Blocks(title="Diabetic Diet Assistant") as demo:
    gr.Markdown("# 🥗 Diabetic Diet Assistant")
    # gr.Markdown(f"User: **{CURRENT_USER['username']}** · journey started "
    #             f"**{journey_start(CURRENT_USER).date()}**")

    with gr.Row():
        user_choices = get_all_users()
        user_dropdown = gr.Dropdown(
            choices=user_choices, 
            value=user_choices[0], 
            label="👤 Select User (Testing)"
        )
        user_info_md = gr.Markdown()
    
    user_state = gr.State(CURRENT_USER)
    pending_state = gr.State(None)

    # Update user state when dropdown changes
    def on_user_change(choice):
        new_user = load_user_from_choice(choice)
        info = f"User: **{new_user['username']}** · journey started **{journey_start(new_user).date()}**"
        # Clear pending state when switching users
        return new_user, None, info

    user_dropdown.change(
        on_user_change,
        inputs=[user_dropdown],
        outputs=[user_state, pending_state, user_info_md]
    )
    
    # Trigger the change once to set the initial text
    demo.load(lambda: on_user_change(user_choices[0]), outputs=[user_state, pending_state, user_info_md])
    user_state = gr.State(CURRENT_USER)
    pending_state = gr.State(None)

    with gr.Tab("1 · Upload meal"):
        with gr.Row():
            with gr.Column():
                meal_type = gr.Dropdown(MEAL_TYPES, value="breakfast", label="Meal")
                scan_mode = gr.Radio(
                    ["Meal (single/multi)", "Restaurant Food", "Packaged food"],
                    value="Meal (single/multi)", label="Scan mode")
                files = gr.File(label="Image(s) - Upload multiple angles if needed",
                                file_count="multiple", file_types=["image"])
                detect_btn = gr.Button("1. Detect foods", variant="primary")
            with gr.Column():
                detect_md = gr.Markdown()
                food_table = gr.Dataframe(
                    headers=["Food", "Grams", "Note"],
                    datatype=["str", "number", "str"],
                    col_count=(3, "fixed"), row_count=(1, "dynamic"),
                    interactive=True, label="Foods and portions (editable)")

                # ── NEW: extra-angle upload flow (hidden until detection looks weak) ──
                quality_state = gr.State(None)
                extra_files   = gr.File(
                    label="📸 Additional photo(s) — try a different angle",
                    file_count="multiple", file_types=["image"], visible=False)
                redetect_btn  = gr.Button(
                    "🔄 Re-detect with these extra photos",
                    variant="secondary", visible=False)

                calc_btn = gr.Button("2. Calculate nutrition", variant="primary")
        advice_md = gr.Markdown()
        with gr.Row():
            when = gr.Textbox(label="Meal time (optional, e.g. 2026-10-21 08:30)")
            save_btn = gr.Button("3. Save to my diary", interactive=False)   # <-- start disabled
        save_msg = gr.Markdown()
        p_today_up = gr.Plot(label="Today — estimated glucose")
        p_cmp_up = gr.Plot(label="Today vs your past days")

    with gr.Tab("2 · My graphs"):
        days = journey_days(CURRENT_USER)
        day_dd = gr.Dropdown(choices=days, value=days[-1], label="Day")
        show_btn = gr.Button("Show graphs", variant="primary")
        whatif_md = gr.Markdown()
        p_day = gr.Plot(label="Selected day")
        p_whatif = gr.Plot(label="What if you ate less / more sugar?")
        p_cmp = gr.Plot(label="Selected day vs your past days")
        stats = gr.Dataframe(label="Numbers")
        p_journey = gr.Plot(label="My journey since login")
        p_scatter = gr.Plot(label="Glucose vs Carbs Scatter Plot")
        
        with gr.Row():
            p_radar_today = gr.Plot(label="Selected Day Nutrition Profile")
            p_radar_week = gr.Plot(label="Weekly Nutrition Profile")

        detect_btn.click(
            detect,
            [files, scan_mode],
            [food_table, detect_md, quality_state, extra_files, redetect_btn],
        )

        redetect_btn.click(
            detect_with_extra,
            [files, extra_files, scan_mode],
            [food_table, detect_md, quality_state, extra_files, redetect_btn],
        )

    with gr.Tab("3 · AI Coach"):
        gr.Markdown(
            "### 🤖 Ask your nutrition coach\n"
            "I can see your **last 7 days of meals**, your **personal risky foods**, "
            "and the **meal you just analyzed** in the Upload tab. "
            "Ask me why a meal is risky, what to eat next, or for a swap."
        )
        chatbot = gr.Chatbot(height=520,
                             label="Conversation", show_label=True)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="e.g. Why was my breakfast risky? "
                            "Suggest a low-carb vegetarian dinner.",
                scale=5, show_label=False, container=False)
            send_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            ask_meal_btn = gr.Button("Ask about the meal I just analyzed",
                                     size="sm")
            clear_btn = gr.Button("Clear chat", size="sm")

    # calculate now also flips save_btn on
    calc_btn.click(calculate, [food_table, meal_type],
                   [advice_md, pending_state, save_btn])

    # save now also flips save_btn back off
    save_btn.click(save_meal,
                   [pending_state, meal_type, when, user_state],
                   [save_msg, p_today_up, p_cmp_up, pending_state, save_btn])

    # if the user edits the food table after calculating, force a recalculation
    food_table.change(
        lambda: (None, gr.update(interactive=False)),
        None, [pending_state, save_btn],
    )

    show_btn.click(show_charts, [day_dd, user_state],
                   [p_day, p_whatif, p_cmp, stats, p_journey, whatif_md, p_scatter, p_radar_today, p_radar_week])

    send_btn.click(chat_fn,
                   [msg, chatbot, user_state, pending_state, food_table],
                   [chatbot, msg])
    msg.submit(chat_fn,
               [msg, chatbot, user_state, pending_state, food_table],
               [chatbot, msg])
    ask_meal_btn.click(ask_about_last_meal,
                       [user_state, pending_state, food_table],
                       [chatbot, msg])
    clear_btn.click(lambda: [], None, chatbot)

    # log when the demo actually starts
    demo.load(lambda: logger.info("Gradio UI loaded for user=%s",
                                  CURRENT_USER["username"]))


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
#     )
#     logger.info("Starting Diabetic Diet Assistant on 127.0.0.1:2000")
#     demo.launch(server_name="127.0.0.1", server_port=2000)