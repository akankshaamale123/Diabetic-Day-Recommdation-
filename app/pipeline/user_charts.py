# app/pipeline/user_charts.py
"""
Charts for a logged-in user. ONE chart per figure.

All glucose lines are ESTIMATES built from the meals the user uploaded
(response model trained on CGMacros). If the user also logged real readings
(user_db.add_glucose), they are drawn on the day chart as a second line.

  plot_today(user, day)          estimated glucose for that day + meal markers
  plot_whatif(user, day, meal_id) same meal: less sugar vs ate vs extra sugar
  plot_journey(user, end)        every day since login, one line
  plot_today_vs_past(user, day)  that day vs the user's usual (past) day
  compare_table(user, day)       numbers behind the comparison
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from storage import user_db
from pipeline.response_model import (OFFSETS, simulate, what_if,
                                     using_trained_model)

logger = logging.getLogger(__name__)

LOW, HIGH = 70.0, 180.0
LEAD = pd.Timedelta(hours=4)          # earlier meals still affect the start of a day


# ---------------------------------------------------------------- helpers
def _nut(row) -> dict:
    return {k: float(row[k] or 0.0) for k in user_db.NUTRIENT_COLS}


def _meal_tuples(meals: pd.DataFrame) -> list:
    return [(r["timestamp"], _nut(r)) for _, r in meals.iterrows()]


def journey_start(user: dict) -> pd.Timestamp:
    """Login date, or the first meal day if the user backfilled older meals."""
    login = user_db.login_date(user)
    meals = user_db.load_meals(user["id"])
    start = login
    if not meals.empty:
        earliest_meal_day = meals["timestamp"].min().normalize()
        if earliest_meal_day < login:
            start = earliest_meal_day
            logger.info("journey_start(): backfilled meals moved journey start "
                        "from %s → %s (user_id=%s)",
                        login.date(), start.date(), user.get("id"))
        else:
            logger.debug("journey_start(): start=%s (login date, user_id=%s)",
                         start.date(), user.get("id"))
    else:
        logger.debug("journey_start(): no meals yet — start=login date %s "
                     "(user_id=%s)", start.date(), user.get("id"))
    return start


def journey_days(user: dict) -> list[str]:
    start = journey_start(user)
    days = pd.date_range(start, pd.Timestamp.now().normalize(), freq="D")
    logger.info("journey_days(): %d day(s) from %s to today (user_id=%s)",
                len(days), start.date(), user.get("id"))
    return [d.strftime("%Y-%m-%d") for d in days]


def _estimate(user: dict, start, end, step_min: int = 5):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    logger.debug("_estimate(): user_id=%s window %s → %s (step=%dmin)",
                 user.get("id"), start, end, step_min)

    meals = user_db.load_meals(user["id"], start - LEAD, end)

    if not meals.empty:
        meals = meals.copy()
        meals["timestamp"] = pd.to_datetime(meals["timestamp"], errors="coerce")
        meals = meals.dropna(subset=["timestamp"]).sort_values("timestamp")
        logger.debug("_estimate(): %d meal(s) in window (incl. %s lead-in)",
                     len(meals), LEAD)
    else:
        # ensure the column exists with the right dtype even when empty
        meals = pd.DataFrame(columns=user_db.NUTRIENT_COLS + ["id", "timestamp", "meal_type"])
        meals["timestamp"] = pd.to_datetime(meals["timestamp"])
        logger.debug("_estimate(): no meals in window — simulating flat baseline")

    sim = simulate(_meal_tuples(meals), start, end, user["baseline_mgdl"], step_min)
    sim = sim.copy()
    sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
    logger.debug("_estimate(): simulation produced %d point(s)", len(sim))
    return sim, meals


def _day_range(day):
    d0 = pd.Timestamp(day).normalize()
    end = d0 + pd.Timedelta(days=1)
    now = pd.Timestamp.now().floor("5min")
    truncated = False
    if d0 <= now < end:               # today: don't draw the future
        end = max(now, d0 + pd.Timedelta(minutes=5))
        truncated = True
    if truncated:
        logger.debug("_day_range(): day=%s is today — truncating end to %s",
                     d0.date(), end)
    return d0, end


def _style(fig, title, x_title="Time", y_title="Estimated glucose (mg/dL)", height=430):
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      template="plotly_white", height=height,
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _band(fig):
    fig.add_hrect(y0=LOW, y1=HIGH, fillcolor="rgba(0,180,0,0.08)", line_width=0)
    fig.add_hline(y=HIGH, line_dash="dash", line_color="red", line_width=1)


def _empty(title, msg):
    logger.debug("_empty(): placeholder figure title=%r msg=%r", title, msg)
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, template="plotly_white", height=300)
    return fig


def model_note() -> str:
    trained = using_trained_model()
    logger.debug("model_note(): trained_model=%s", trained)
    return ("Estimates from a model trained on CGMacros data."
            if trained else
            "Estimates from a simple rule of thumb (trained model not built yet).")


# ---------------------------------------------------------------- charts
def plot_today(user: dict, day) -> go.Figure:
    d0, end = _day_range(day)
    logger.info("plot_today(): user_id=%s day=%s", user.get("id"), d0.date())
    title = f"Estimated glucose — {d0.date()}"

    sim, meals = _estimate(user, d0, end)
    day_meals = meals[meals["timestamp"] >= d0]
    measured = user_db.load_glucose(user["id"], d0, end)
    logger.debug("plot_today(): day_meals=%d measured_readings=%d",
                 len(day_meals), len(measured))

    if day_meals.empty and measured.empty:
        logger.warning("plot_today(): no meals and no measured readings for "
                       "%s (user_id=%s) — returning placeholder",
                       d0.date(), user.get("id"))
        return _empty(title, "No meals uploaded for this day yet")

    fig = go.Figure(go.Scatter(
        x=sim["timestamp"], y=sim["glucose_mgdl"], mode="lines",
        name="Estimated (from your meals)", line=dict(color="#1f77b4", width=2.5),
        hovertemplate="%{x|%H:%M}<br>%{y:.0f} mg/dL<extra></extra>"))
    _band(fig)

    if not day_meals.empty:
        at = pd.merge_asof(day_meals.sort_values("timestamp"),
                           sim.rename(columns={"glucose_mgdl": "g"}),
                           on="timestamp", direction="nearest")
        fig.add_trace(go.Scatter(
            x=at["timestamp"], y=at["g"], mode="markers", name="Meal",
            marker=dict(color="orange", size=12, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=[f"{r.meal_type}: {r.carbs_g:.0f} g carbs, {r.sugar_g:.0f} g sugar"
                  for r in at.itertuples()],
            hovertemplate="%{x|%H:%M}<br>%{text}<extra></extra>"))
        logger.debug("plot_today(): added %d meal marker(s)", len(at))

    if not measured.empty:
        fig.add_trace(go.Scatter(
            x=measured["timestamp"], y=measured["glucose_mgdl"], mode="lines+markers",
            name="Measured", line=dict(color="black", width=1.5)))
        logger.debug("plot_today(): added measured-readings trace (%d points)",
                     len(measured))

    return _style(fig, title)


def _pick_meal(user: dict, day, meal_id=None):
    d0 = pd.Timestamp(day).normalize()
    meals = user_db.load_meals(user["id"], d0, d0 + pd.Timedelta(days=1))
    if meals.empty:
        logger.debug("_pick_meal(): no meals on %s (user_id=%s)",
                     d0.date(), user.get("id"))
        return None

    if meal_id is not None:
        if (meals["id"] == meal_id).any():
            logger.debug("_pick_meal(): explicit meal_id=%s selected on %s",
                         meal_id, d0.date())
            return meals[meals["id"] == meal_id].iloc[0]
        logger.warning("_pick_meal(): requested meal_id=%s not found on %s "
                       "(user_id=%s) — falling back to latest meal",
                       meal_id, d0.date(), user.get("id"))

    latest = meals.iloc[-1]
    logger.debug("_pick_meal(): fallback to latest meal on %s — id=%s, "
                 "type=%s", d0.date(), latest.get("id"), latest.get("meal_type"))
    return latest


def plot_whatif(user: dict, day, meal_id=None) -> go.Figure:
    logger.info("plot_whatif(): user_id=%s day=%s meal_id=%s",
                user.get("id"), pd.Timestamp(day).date(), meal_id)
    meal = _pick_meal(user, day, meal_id)
    title = f"What if? — {pd.Timestamp(day).date()}"
    if meal is None:
        logger.warning("plot_whatif(): no meal available on %s — placeholder",
                       pd.Timestamp(day).date())
        return _empty(title, "No meal on this day yet")

    nut = _nut(meal)
    w = what_if(nut)
    base = float(user["baseline_mgdl"])
    xs = np.arange(0, 241, 5)

    def curve(delta):
        return base + np.interp(xs, OFFSETS + [240], list(delta) + [0.0])

    peak = lambda d: float(np.max(d))
    logger.info("plot_whatif(): peaks — actual=+%.0f less=+%.0f extra=+%.0f "
                "(sugar=%.0fg, cut=%.0fg, add=%.0fg)",
                peak(w["actual"]), peak(w["less"]), peak(w["extra"]),
                nut["sugar_g"], w["sugar_cut_g"], w["sugar_add_g"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=curve(w["less"]), mode="lines",
        name=f"Less sugar (−{w['sugar_cut_g']:.0f} g): peak +{peak(w['less']):.0f}",
        line=dict(color="#2ca02c", width=2.5)))
    fig.add_trace(go.Scatter(
        x=xs, y=curve(w["actual"]), mode="lines",
        name=f"What you ate ({nut['sugar_g']:.0f} g sugar): peak +{peak(w['actual']):.0f}",
        line=dict(color="#1f77b4", width=3.5)))
    fig.add_trace(go.Scatter(
        x=xs, y=curve(w["extra"]), mode="lines",
        name=f"Extra sugar (+{w['sugar_add_g']:.0f} g): peak +{peak(w['extra']):.0f}",
        line=dict(color="#d62728", width=2.5)))
    _band(fig)
    fig = _style(fig, f"What if? — {meal['meal_type']} on {pd.Timestamp(meal['timestamp']).date()}",
                 x_title="Minutes after the meal")
    fig.update_xaxes(dtick=30)
    return fig


def whatif_text(user: dict, day, meal_id=None) -> str:
    meal = _pick_meal(user, day, meal_id)
    if meal is None:
        logger.debug("whatif_text(): no meal on %s — returning empty string",
                     pd.Timestamp(day).date())
        return ""
    w = what_if(_nut(meal))
    p = lambda k: float(np.max(w[k]))
    logger.debug("whatif_text(): actual=+%.0f less=+%.0f extra=+%.0f",
                 p("actual"), p("less"), p("extra"))
    return (f"**{meal['meal_type'].title()}** — estimated rise in glucose: "
            f"**+{p('actual'):.0f} mg/dL** as eaten, "
            f"**+{p('less'):.0f}** with {w['sugar_cut_g']:.0f} g less sugar, "
            f"**+{p('extra'):.0f}** with {w['sugar_add_g']:.0f} g extra sugar.")


def plot_journey(user: dict, end=None) -> go.Figure:
    start = journey_start(user)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().floor("15min")
    logger.info("plot_journey(): user_id=%s window %s → %s",
                user.get("id"), start.date(), end.date())

    title = f"My journey — {start.date()} → {end.date()}"
    sim, meals = _estimate(user, start, end, step_min=15)
    if meals.empty:
        logger.warning("plot_journey(): no meals in window %s → %s — "
                       "returning placeholder", start.date(), end.date())
        return _empty(title, "Upload your first meal to start your journey")

    fig = go.Figure(go.Scatter(
        x=sim["timestamp"], y=sim["glucose_mgdl"], mode="lines",
        name="Estimated glucose", line=dict(color="#1f77b4", width=1.6),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<extra></extra>"))
    _band(fig)
    logger.debug("plot_journey(): %d meal(s), %d sim points", len(meals), len(sim))
    return _style(fig, title)


def _past_days_profile(user: dict, day):
    """Hourly-ish profile of the user's past days that had meals."""
    d0 = pd.Timestamp(day).normalize()
    start = journey_start(user)

    if d0 <= start:
        logger.debug("_past_days_profile(): day=%s ≤ journey_start=%s — "
                     "no past window (user_id=%s)",
                     d0.date(), start.date(), user.get("id"))
        return None

    sim, meals = _estimate(user, start, d0, step_min=5)
    if meals.empty or sim.empty:                     # <-- the missing guard
        logger.debug("_past_days_profile(): no meals/sim in [%s, %s) — "
                     "returning None (meals=%d, sim=%d)",
                     start.date(), d0.date(), len(meals), len(sim))
        return None

    sim = sim[sim["timestamp"] < d0]
    if sim.empty:
        logger.debug("_past_days_profile(): sim has no points before %s — "
                     "returning None", d0.date())
        return None

    meal_days = set(meals["timestamp"].dt.normalize())
    sim = sim[sim["timestamp"].dt.normalize().isin(meal_days)]
    if sim.empty:
        logger.debug("_past_days_profile(): sim days didn't intersect meal "
                     "days — returning None (meal_days=%d)",
                     len(meal_days))
        return None

    logger.debug("_past_days_profile(): returning %d past-sim points across "
                 "%d meal day(s)", len(sim), len(meal_days))
    return sim


def _bins(sim: pd.DataFrame) -> pd.Series:
    t = sim["timestamp"]
    hrs = ((t.dt.hour * 60 + t.dt.minute) // 30) * 0.5
    binned = sim.groupby(hrs)["glucose_mgdl"].mean()
    logger.debug("_bins(): %d half-hour bin(s) from %d row(s)",
                 len(binned), len(sim))
    return binned


def plot_today_vs_past(user: dict, day) -> go.Figure:
    d0, end = _day_range(day)
    logger.info("plot_today_vs_past(): user_id=%s day=%s",
                user.get("id"), d0.date())
    title = f"{d0.date()} vs your usual day"

    past = _past_days_profile(user, d0)
    today, _ = _estimate(user, d0, end)
    if past is None:
        logger.warning("plot_today_vs_past(): no usable past-day profile for "
                       "%s (user_id=%s) — placeholder",
                       d0.date(), user.get("id"))
        return _empty(title, "Need at least one earlier day with meals to compare")

    p, t = _bins(past), _bins(today)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p.values, mode="lines",
                             name="Your past days (average)",
                             line=dict(color="grey", width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(x=t.index, y=t.values, mode="lines",
                             name=str(d0.date()), line=dict(color="#1f77b4", width=3)))
    _band(fig)
    fig = _style(fig, title, x_title="Hour of day")
    fig.update_xaxes(dtick=2, range=[0, 24])
    return fig


def compare_table(user: dict, day) -> pd.DataFrame:
    d0, end = _day_range(day)
    logger.info("compare_table(): user_id=%s day=%s", user.get("id"), d0.date())

    def stats(sim: pd.DataFrame) -> dict:
        v = sim["glucose_mgdl"]
        return {"peak_mgdl": round(float(v.max()), 0),
                "mean_mgdl": round(float(v.mean()), 0),
                "time_above_180_%": round(float((v > HIGH).mean() * 100), 1)}

    today, _ = _estimate(user, d0, end)
    rows = {str(d0.date()): stats(today)}
    logger.debug("compare_table(): today stats = %s", rows[str(d0.date())])

    past = _past_days_profile(user, d0)
    if past is not None:
        per_day = [stats(g) for _, g in past.groupby(past["timestamp"].dt.normalize())]
        rows["Your past days (average)"] = {
            k: round(float(np.mean([s[k] for s in per_day])), 1) for k in per_day[0]}
        logger.debug("compare_table(): averaged %d past day(s)", len(per_day))
    else:
        logger.info("compare_table(): no past-day profile available — table "
                    "will only show today's row")
    return pd.DataFrame(rows).T.reset_index(names="period")



# Add to the bottom of pipeline/user_charts.py

def plot_nutrition_radar(user: dict, day: str = None, is_weekly: bool = False) -> go.Figure:
    """
    Plots a Radar Chart showing all 8 nutrients for a single day or the whole week.
    Normalizes values to % of Daily Target so all units (kcal, g, mg) can share one graph.
    """
    if is_weekly:
        start = pd.Timestamp.now().normalize() - pd.Timedelta(days=6)
        title = "Weekly Nutrition Profile (% of Daily Target)"
        multiplier = 7
    else:
        d0 = pd.Timestamp(day).normalize() if day else pd.Timestamp.now().normalize()
        start = d0
        title = f"Nutrition Profile — {start.date()} (% of Daily Target)"
        multiplier = 1

    end = start + pd.Timedelta(days=1 if not is_weekly else 7) - pd.Timedelta(seconds=1)

    try:
        meals = user_db.load_meals(user["id"], start, end)
        if meals.empty:
            return _empty(title, "No meals logged for this period")

        # 1. Define targets
        targets = {
            "calories_kcal": (1800 * multiplier, "Calories"),
            "carbs_g": (150 * multiplier, "Carbs"),
            "protein_g": (60 * multiplier, "Protein"),
            "fat_g": (50 * multiplier, "Fat"),
            "sugar_g": (25 * multiplier, "Sugar"),
            "fiber_g": (30 * multiplier, "Fiber"),
            "cholesterol_mg": (200 * multiplier, "Cholesterol"),
            "calcium_mg": (1000 * multiplier, "Calcium"),
        }

        # 2. Only sum columns that actually exist in the database
        available_cols = [c for c in targets.keys() if c in meals.columns]
        totals = meals[available_cols].sum()

        categories = []
        percentages = []

        for key, (target, label) in targets.items():
            val = totals.get(key, 0) # Safely gets 0 if column is missing
            pct = (val / target) * 100
            
            # Only plot categories that have data or exist
            if key in available_cols:
                categories.append(label)
                percentages.append(round(pct, 1))

        # Close the radar loop
        categories.append(categories[0])
        percentages.append(percentages[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=percentages, theta=categories, fill='toself',
            name='Your Intake', line_color='#1f77b4',
            hovertemplate="%{theta}<br>%{r:.1f}% of target<extra></extra>"
        ))
        
        # Add 100% reference
        fig.add_trace(go.Scatterpolar(
            r=[100]*len(categories), theta=categories, mode='lines',
            name='100% Daily Target', line=dict(color='rgba(255, 0, 0, 0.4)', width=2, dash='dash')
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(percentages) + 20, 120)], ticksuffix="%")),
            showlegend=True, title=title, template="plotly_white", height=500,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        return fig
    except Exception as e:
        logger.exception("plot_nutrition_radar() failed")
        return _empty(title, f"Error: {str(e)}")



def plot_user_scatter(user: dict, day: str) -> go.Figure:
    """Scatter plot of glucose vs carbs using the user's actual data."""
    d0 = pd.Timestamp(day).normalize()
    end = d0 + pd.Timedelta(days=1)
    title = f"Glucose vs Carbs — {d0.date()}"

    try:
        meals = user_db.load_meals(user["id"], d0, end)
        glucose = user_db.load_glucose(user["id"], d0, end)

        if meals.empty or glucose.empty:
            return _empty(title, "Not enough data for scatter plot")

        # Merge meals and glucose to find pairs
        merged = pd.merge_asof(
            meals.sort_values("timestamp"),
            glucose.sort_values("timestamp"),
            on="timestamp", direction="backward", tolerance=pd.Timedelta(minutes=120)
        ).dropna(subset=["glucose_mgdl", "carbs_g"])

        if merged.empty:
            return _empty(title, "No meals matched with glucose readings")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged['carbs_g'], y=merged['glucose_mgdl'],
            mode='markers',
            marker=dict(color='blue', size=10, opacity=0.7),
            text=[f"{r.meal_type}" for r in merged.itertuples()],
            hovertemplate="<b>%{text}</b><br>Carbs: %{x:.0f} g<br>Glucose: %{y:.0f} mg/dL<extra></extra>"
        ))

        fig.update_layout(
            title=title, xaxis_title="Carbs (g)", yaxis_title="Glucose (mg/dL)",
            template="plotly_white", height=400
        )
        return fig
    except Exception as e:
        logger.exception("plot_user_scatter() failed")
        return _empty(title, f"Error: {str(e)}")
    
    