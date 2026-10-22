# app/app.py — Diabetic Diet Assistant (upload meal -> advice -> graphs)
# The user is fixed (see current_user.py); login is handled by another team.
from __future__ import annotations
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
    plot_journey, plot_today_vs_past, compare_table,
)
from storage import user_db



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
    return user_db.get_or_create_user(
        DEMO_USER["username"],
        DEMO_USER["diabetes_type"],
        DEMO_USER["baseline_mgdl"],
        created_at=DEMO_USER["journey_start"],
    )


CURRENT_USER = get_current_user()


def detect(image_path):
    if image_path is None:
        raise gr.Error("Upload a meal photo first.")
    try:
        vision = detect_foods(image_path, "Identify all visible food items.")
    except Exception as e:
        raise gr.Error(f"Could not analyze the photo: {e}")
 
    rows = vision_to_rows(vision)
    if not rows:
        raise gr.Error("No food was detected. Try a clearer photo, or type the foods in the table.")
    note = ("### 🍽️ Check the foods and portions\n"
            "Edit the **name** or **grams** if they look wrong (you can add rows), "
            "then press **Calculate nutrition**.")
    return rows, note
 
 
# ------------------------------------------------------------ Step 2: nutrition + advice
def calculate(table, meal_type):
    items = rows_to_items(table)
    if not items:
        raise gr.Error("Add at least one food with grams greater than 0.")
 
    nutrition = calculate_nutrition(items)
    totals = nutrition.get("total_nutrition", {})
    if not any(totals.values()):
        raise gr.Error("Nutrition could not be calculated for these foods. "
                       f"Not found: {', '.join(nutrition.get('unmatched_foods', [])) or 'unknown'}")
 
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
        a = assess_meal(totals, "lunch")                    # lunch limits for lunch/dinner/snack
        md += f"### Verdict: {a['label']}\n" + "".join(f"- {i}\n" for i in a["issues"])
    md += "\n\nPress **Save to my diary** to add it to your graphs."
    return md, {"totals": totals, "foods": items}
 
 
# ------------------------------------------------------------ Step 3: save -> graphs
def save_meal(pending, meal_type, when, user):
    if not pending:
        raise gr.Error("Press 'Calculate nutrition' first, then save.")
    try:
        ts = pd.Timestamp(when) if str(when).strip() else pd.Timestamp.now()
    except Exception:
        raise gr.Error("Meal time must look like 2026-10-21 08:30 (or leave empty for now).")
    user_db.add_meal(user["id"], ts, meal_type, pending["totals"], pending["foods"])
    day = ts.strftime("%Y-%m-%d")
    note = (f"✅ Saved {meal_type} at {ts:%Y-%m-%d %H:%M}.  \n"
            f"{whatif_text(user, day)}  \n*{model_note()}*")
    # pending=None so pressing Save twice cannot store the meal twice
    return note, plot_today(user, day), plot_today_vs_past(user, day), None
 
 
# ------------------------------------------------------------ Graphs tab
def show_charts(day, user):
    try:
        # Import required modules for scatter plot
        from pipeline.glucose_charts import plot_scatter
        from pipeline.user_charts import journey_start
        from pipeline.glucose_charts import default_login_date

        # Get default login date for scatter plot
        login_date = journey_start(user)
        end = pd.Timestamp.now()

        # Create scatter plot for the selected day
        scatter_plot = plot_scatter(day, login_date, end)

        return (plot_today(user, day), plot_whatif(user, day),
                plot_today_vs_past(user, day), compare_table(user, day),
                plot_journey(user), whatif_text(user, day) + f"  \n*{model_note()}*", scatter_plot)
    except Exception as e:
        print(f"Error generating charts: {e}")
        # Return empty charts as fallback
        from pipeline.glucose_charts import _empty
        empty_chart = _empty("Error", f"Chart generation failed: {str(e)}")
        return (plot_today(user, day), plot_whatif(user, day),
                plot_today_vs_past(user, day), compare_table(user, day),
                plot_journey(user), whatif_text(user, day) + f"  \n*{model_note()}*", empty_chart)
 
 
# ------------------------------------------------------------ UI
with gr.Blocks(title="Diabetic Diet Assistant") as demo:
    gr.Markdown("# 🥗 Diabetic Diet Assistant")
    gr.Markdown(f"User: **{CURRENT_USER['username']}** · journey started "
                f"**{journey_start(CURRENT_USER).date()}**")
    user_state = gr.State(CURRENT_USER)
    pending_state = gr.State(None)
 
    with gr.Tab("1 · Upload meal"):
        with gr.Row():
            with gr.Column():
                meal_type = gr.Dropdown(MEAL_TYPES, value="breakfast", label="Meal")
                image = gr.Image(type="filepath", label="Meal photo")
                detect_btn = gr.Button("1. Detect foods", variant="primary")
            with gr.Column():
                detect_md = gr.Markdown()
                food_table = gr.Dataframe(
                    headers=["Food", "Grams", "Note"],
                    datatype=["str", "number", "str"],
                    col_count=(3, "fixed"), row_count=(1, "dynamic"),
                    interactive=True, label="Foods and portions (editable)")
                calc_btn = gr.Button("2. Calculate nutrition", variant="primary")
        advice_md = gr.Markdown()
        with gr.Row():
            when = gr.Textbox(label="Meal time (optional, e.g. 2026-10-21 08:30)")
            save_btn = gr.Button("3. Save to my diary")
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
 
    detect_btn.click(detect, [image], [food_table, detect_md])
    calc_btn.click(calculate, [food_table, meal_type], [advice_md, pending_state])
    save_btn.click(save_meal, [pending_state, meal_type, when, user_state],
                   [save_msg, p_today_up, p_cmp_up, pending_state])
    show_btn.click(show_charts, [day_dd, user_state],
                   [p_day, p_whatif, p_cmp, stats, p_journey, whatif_md, p_scatter])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=2000)