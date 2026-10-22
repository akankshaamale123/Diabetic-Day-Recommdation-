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

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from storage import user_db
from pipeline.response_model import (OFFSETS, simulate, what_if,
                                     using_trained_model)

LOW, HIGH = 70.0, 180.0
LEAD = pd.Timedelta(hours=4)          # earlier meals still affect the start of a day


# ---------------------------------------------------------------- helpers
def _nut(row) -> dict:
    return {k: float(row[k] or 0.0) for k in user_db.NUTRIENT_COLS}


def _meal_tuples(meals: pd.DataFrame) -> list:
    return [(r["timestamp"], _nut(r)) for _, r in meals.iterrows()]


def journey_start(user: dict) -> pd.Timestamp:
    """Login date, or the first meal day if the user backfilled older meals."""
    start = user_db.login_date(user)
    meals = user_db.load_meals(user["id"])
    if not meals.empty:
        start = min(start, meals["timestamp"].min().normalize())
    return start


def journey_days(user: dict) -> list[str]:
    days = pd.date_range(journey_start(user), pd.Timestamp.now().normalize(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]


def _estimate(user: dict, start, end, step_min: int = 5):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    meals = user_db.load_meals(user["id"], start - LEAD, end)
    sim = simulate(_meal_tuples(meals), start, end, user["baseline_mgdl"], step_min)
    return sim, meals


def _day_range(day):
    d0 = pd.Timestamp(day).normalize()
    end = d0 + pd.Timedelta(days=1)
    now = pd.Timestamp.now().floor("5min")
    if d0 <= now < end:               # today: don't draw the future
        end = max(now, d0 + pd.Timedelta(minutes=5))
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
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, template="plotly_white", height=300)
    return fig


def model_note() -> str:
    return ("Estimates from a model trained on CGMacros data."
            if using_trained_model() else
            "Estimates from a simple rule of thumb (trained model not built yet).")


# ---------------------------------------------------------------- charts
def plot_today(user: dict, day) -> go.Figure:
    d0, end = _day_range(day)
    title = f"Estimated glucose — {d0.date()}"
    sim, meals = _estimate(user, d0, end)
    day_meals = meals[meals["timestamp"] >= d0]
    if day_meals.empty and user_db.load_glucose(user["id"], d0, end).empty:
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

    measured = user_db.load_glucose(user["id"], d0, end)
    if not measured.empty:
        fig.add_trace(go.Scatter(
            x=measured["timestamp"], y=measured["glucose_mgdl"], mode="lines+markers",
            name="Measured", line=dict(color="black", width=1.5)))
    return _style(fig, title)


def _pick_meal(user: dict, day, meal_id=None):
    d0 = pd.Timestamp(day).normalize()
    meals = user_db.load_meals(user["id"], d0, d0 + pd.Timedelta(days=1))
    if meals.empty:
        return None
    if meal_id is not None and (meals["id"] == meal_id).any():
        return meals[meals["id"] == meal_id].iloc[0]
    return meals.iloc[-1]                              # latest meal of the day


def plot_whatif(user: dict, day, meal_id=None) -> go.Figure:
    meal = _pick_meal(user, day, meal_id)
    title = f"What if? — {pd.Timestamp(day).date()}"
    if meal is None:
        return _empty(title, "No meal on this day yet")

    nut = _nut(meal)
    w = what_if(nut)
    base = float(user["baseline_mgdl"])
    xs = np.arange(0, 241, 5)

    def curve(delta):
        return base + np.interp(xs, OFFSETS + [240], list(delta) + [0.0])

    peak = lambda d: float(np.max(d))
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
        return ""
    w = what_if(_nut(meal))
    p = lambda k: float(np.max(w[k]))
    return (f"**{meal['meal_type'].title()}** — estimated rise in glucose: "
            f"**+{p('actual'):.0f} mg/dL** as eaten, "
            f"**+{p('less'):.0f}** with {w['sugar_cut_g']:.0f} g less sugar, "
            f"**+{p('extra'):.0f}** with {w['sugar_add_g']:.0f} g extra sugar.")


def plot_journey(user: dict, end=None) -> go.Figure:
    start = journey_start(user)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().floor("15min")
    title = f"My journey — {start.date()} → {end.date()}"
    sim, meals = _estimate(user, start, end, step_min=15)
    if meals.empty:
        return _empty(title, "Upload your first meal to start your journey")
    fig = go.Figure(go.Scatter(
        x=sim["timestamp"], y=sim["glucose_mgdl"], mode="lines",
        name="Estimated glucose", line=dict(color="#1f77b4", width=1.6),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<extra></extra>"))
    _band(fig)
    return _style(fig, title)


def _past_days_profile(user: dict, day):
    """Hourly-ish profile of the user's past days that had meals."""
    d0 = pd.Timestamp(day).normalize()
    start = journey_start(user)
    if d0 <= start:
        return None
    sim, meals = _estimate(user, start, d0, step_min=5)
    sim = sim[sim["timestamp"] < d0]
    meal_days = set(meals["timestamp"].dt.normalize())
    sim = sim[sim["timestamp"].dt.normalize().isin(meal_days)]
    return None if sim.empty else sim


def _bins(sim: pd.DataFrame) -> pd.Series:
    t = sim["timestamp"]
    hrs = ((t.dt.hour * 60 + t.dt.minute) // 30) * 0.5
    return sim.groupby(hrs)["glucose_mgdl"].mean()


def plot_today_vs_past(user: dict, day) -> go.Figure:
    d0, end = _day_range(day)
    title = f"{d0.date()} vs your usual day"
    past = _past_days_profile(user, d0)
    today, _ = _estimate(user, d0, end)
    if past is None:
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

    def stats(sim: pd.DataFrame) -> dict:
        v = sim["glucose_mgdl"]
        return {"peak_mgdl": round(float(v.max()), 0),
                "mean_mgdl": round(float(v.mean()), 0),
                "time_above_180_%": round(float((v > HIGH).mean() * 100), 1)}

    today, _ = _estimate(user, d0, end)
    rows = {str(d0.date()): stats(today)}
    past = _past_days_profile(user, d0)
    if past is not None:
        per_day = [stats(g) for _, g in past.groupby(past["timestamp"].dt.normalize())]
        rows["Your past days (average)"] = {
            k: round(float(np.mean([s[k] for s in per_day])), 1) for k in per_day[0]}
    return pd.DataFrame(rows).T.reset_index(names="period")