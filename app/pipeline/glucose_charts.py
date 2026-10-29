# app/pipeline/glucose_charts.py
"""
Simple glucose charts — ONE chart per figure.

  plot_day(day)                     -> glucose line for one day (+ meal markers)
  plot_day_carbs(day)               -> carbs per meal for that day (separate figure)
  plot_overall(login_date, end)     -> glucose line from the login date until now
  plot_daily_average(login_date, end)-> one dot per day (daily mean glucose)
  plot_compare(day, login_date, end)-> that day vs the user's usual day (by hour)
  plot_scatter(day, login_date, end)-> scatter plot of glucose vs carbs
  compare_stats(day, login_date, end)-> table: this day vs overall
  save_all(...)                     -> writes every chart as its own HTML file
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from storage import db

logger = logging.getLogger(__name__)

LOW = 70.0
HIGH = 180.0


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------
def default_login_date():
    """First day that has data = the day the user started (login date)."""
    days = db.available_days()
    if not days:
        logger.error("default_login_date(): no data in the database yet")
        raise ValueError("No data in the database yet.")
    login = pd.Timestamp(days[0]).normalize()
    logger.debug("default_login_date(): resolved to %s (from %d available day(s))",
                 login.date(), len(days))
    return login


def _day_bounds(day):
    day = pd.Timestamp(day).normalize()
    return day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _style(fig: go.Figure, title: str, y_title: str = "mg/dL",
           x_title: str = "Time", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        template="plotly_white", height=height, showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def _target_band(fig: go.Figure) -> None:
    fig.add_hrect(y0=LOW, y1=HIGH, fillcolor="rgba(0,180,0,0.08)", line_width=0)
    fig.add_hline(y=HIGH, line_dash="dash", line_color="red", line_width=1)


def _empty(title: str, msg: str) -> go.Figure:
    logger.debug("_empty(): placeholder figure title=%r msg=%r", title, msg)
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, template="plotly_white", height=300)
    return fig


# ----------------------------------------------------------------
# 1. One day
# ----------------------------------------------------------------
def plot_day(day) -> go.Figure:
    start, end = _day_bounds(day)
    logger.info("plot_day(): day=%s", start.date())

    g = db.load_glucose(start, end)
    m = db.load_meals(start, end)
    logger.debug("plot_day(): glucose_rows=%d meal_rows=%d", len(g), len(m))

    title = f"Glucose — {start.date()}"
    if g.empty:
        logger.warning("plot_day(): no glucose readings for %s", start.date())
        return _empty(title, "No glucose readings for this day")

    fig = go.Figure(go.Scatter(
        x=g["timestamp"], y=g["glucose_mgdl"], mode="lines",
        name="Glucose", line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%H:%M}<br>%{y:.0f} mg/dL<extra></extra>",
    ))
    _target_band(fig)

    # meal markers: a dot on the glucose line at each meal time
    if not m.empty:
        g_sorted = g.sort_values("timestamp")
        y_at_meal = pd.merge_asof(
            m.sort_values("timestamp")[["timestamp", "food_name"]],
            g_sorted[["timestamp", "glucose_mgdl"]],
            on="timestamp", direction="nearest",
        )
        fig.add_trace(go.Scatter(
            x=y_at_meal["timestamp"], y=y_at_meal["glucose_mgdl"],
            mode="markers", name="Meal",
            marker=dict(color="orange", size=11, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=y_at_meal["food_name"],
            hovertemplate="%{x|%H:%M}<br>%{text}<extra></extra>",
        ))
        logger.debug("plot_day(): added %d meal marker(s) to %s",
                     len(y_at_meal), start.date())
    else:
        logger.debug("plot_day(): no meals to annotate for %s", start.date())

    return _style(fig, title)


def plot_day_carbs(day) -> go.Figure:
    start, end = _day_bounds(day)
    logger.info("plot_day_carbs(): day=%s", start.date())

    m = db.load_meals(start, end)
    logger.debug("plot_day_carbs(): meal_rows=%d", len(m))

    title = f"Carbs per meal — {start.date()}"
    if m.empty:
        logger.warning("plot_day_carbs(): no meals logged for %s", start.date())
        return _empty(title, "No meals logged for this day")

    fig = go.Figure(go.Bar(
        x=m["timestamp"], y=m["carbs_g"], marker_color="orange",
        text=m["food_name"],
        hovertemplate="%{x|%H:%M}<br>%{text}<br>%{y:.0f} g<extra></extra>",
    ))
    fig = _style(fig, title, y_title="Carbs (g)")
    fig.update_layout(showlegend=False)
    return fig


# ----------------------------------------------------------------
# 2. Overall — from login date to now
# ----------------------------------------------------------------
def plot_overall(login_date=None, end=None) -> go.Figure:
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    logger.info("plot_overall(): window %s → %s",
                login_date.date(), end.date())

    title = f"Overall glucose — {login_date.date()} → {end.date()}"
    g = db.load_glucose(login_date, end)
    logger.debug("plot_overall(): glucose_rows=%d", len(g))

    if g.empty:
        logger.warning("plot_overall(): no glucose readings in window")
        return _empty(title, "No glucose readings in this period")

    fig = go.Figure(go.Scatter(
        x=g["timestamp"], y=g["glucose_mgdl"], mode="lines", name="Glucose",
        line=dict(color="#1f77b4", width=1.4),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<extra></extra>",
    ))
    _target_band(fig)
    return _style(fig, title)


def plot_daily_average(login_date=None, end=None) -> go.Figure:
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    logger.info("plot_daily_average(): window %s → %s",
                login_date.date(), end.date())

    title = f"Daily average glucose — {login_date.date()} → {end.date()}"
    g = db.load_glucose(login_date, end)
    logger.debug("plot_daily_average(): glucose_rows=%d", len(g))

    if g.empty:
        logger.warning("plot_daily_average(): no glucose readings in window")
        return _empty(title, "No glucose readings in this period")

    daily = (g.assign(day=g["timestamp"].dt.normalize())
              .groupby("day", as_index=False)["glucose_mgdl"].mean())
    logger.debug("plot_daily_average(): %d day(s) after aggregation", len(daily))

    fig = go.Figure(go.Scatter(
        x=daily["day"], y=daily["glucose_mgdl"], mode="lines+markers",
        name="Daily mean", line=dict(color="#ff7f0e", width=2),
        hovertemplate="%{x|%d %b}<br>%{y:.0f} mg/dL<extra></extra>",
    ))
    _target_band(fig)
    return _style(fig, title, x_title="Day")


# ----------------------------------------------------------------
# 3. Compare: this day vs the user's usual day
# ----------------------------------------------------------------
def _hourly_profile(g: pd.DataFrame) -> pd.Series:
    t = g["timestamp"]
    hour = t.dt.hour + t.dt.minute / 60.0
    return g.groupby(hour.round(0))["glucose_mgdl"].mean()


def plot_compare(day, login_date=None, end=None) -> go.Figure:
    """Two lines only: the chosen day, and the average of all OTHER days."""
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    d0, d1 = _day_bounds(day)
    logger.info("plot_compare(): day=%s vs baseline window %s → %s",
                d0.date(), login_date.date(), end.date())

    title = f"{d0.date()} vs your usual day (since {login_date.date()})"

    g_day = db.load_glucose(d0, d1)
    g_all = db.load_glucose(login_date, end)
    g_other = g_all[(g_all["timestamp"] < d0) | (g_all["timestamp"] > d1)]
    logger.debug("plot_compare(): day_rows=%d other_rows=%d", len(g_day), len(g_other))

    if g_day.empty or g_other.empty:
        logger.warning("plot_compare(): not enough data (day_empty=%s, "
                       "other_empty=%s)", g_day.empty, g_other.empty)
        return _empty(title, "Not enough data to compare")

    day_prof = _hourly_profile(g_day)
    usual_prof = _hourly_profile(g_other)
    logger.debug("plot_compare(): day_profile_hours=%d usual_profile_hours=%d",
                 len(day_prof), len(usual_prof))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=usual_prof.index, y=usual_prof.values, mode="lines",
        name="Usual day (average)", line=dict(color="grey", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=day_prof.index, y=day_prof.values, mode="lines+markers",
        name=str(d0.date()), line=dict(color="#1f77b4", width=2.5),
    ))
    _target_band(fig)
    fig = _style(fig, title, x_title="Hour of day")
    fig.update_xaxes(dtick=2, range=[0, 24])
    return fig


# ----------------------------------------------------------------
# 4. Scatter plot: glucose vs carbs
# ----------------------------------------------------------------
def plot_scatter(day, login_date=None, end=None) -> go.Figure:
    """Scatter plot of glucose vs carbs consumed on the same day."""
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    d0, d1 = _day_bounds(day)
    logger.info("plot_scatter(): day=%s", d0.date())

    title = f"Glucose vs Carbs — {d0.date()}"

    try:
        g = db.load_glucose(d0, d1)
        m = db.load_meals(d0, d1)
        logger.debug("plot_scatter(): glucose_rows=%d meal_rows=%d", len(g), len(m))

        # Filter to only meals that have glucose readings
        if g.empty or m.empty:
            logger.warning("plot_scatter(): insufficient data "
                           "(glucose_empty=%s, meal_empty=%s)",
                           g.empty, m.empty)
            return _empty(title, "Not enough data for scatter plot")

        # Only include meals that have carb data and glucose readings
        meal_glucose = pd.merge_asof(
    m.sort_values("timestamp"),
    g.sort_values("timestamp"),
    on="timestamp",
    direction="backward",
    tolerance=pd.Timedelta(minutes=120),
)

        # Filter to only valid entries
        meal_glucose = meal_glucose.dropna(subset=['glucose_mgdl', 'carbs_g'])
        logger.debug("plot_scatter(): %d valid meal-glucose pair(s) after "
                     "merge + NaN drop", len(meal_glucose))

        if meal_glucose.empty:
            logger.warning("plot_scatter(): no valid meal-glucose pairs for %s "
                           "(tolerance=30 min)", d0.date())
            return _empty(title, "Not enough data for scatter plot")

        # Create scatter plot with minimal implementation
        fig = go.Figure()

        # Add scatter points - very simple implementation
        fig.add_trace(go.Scatter(
            x=meal_glucose['carbs_g'],
            y=meal_glucose['glucose_mgdl'],
            mode='markers',
            marker=dict(
                color='blue',
                size=8,
                opacity=0.7
            ),
            name="Glucose vs Carbs"
        ))

        # Simple styling
        fig.update_layout(
            title=title,
            xaxis_title="Carbs (g)",
            yaxis_title="Glucose (mg/dL)",
            template="plotly_white",
            height=400,
            showlegend=True
        )

        logger.info("plot_scatter(): built figure with %d point(s) for %s",
                    len(meal_glucose), d0.date())
        return fig
    except Exception as e:
        logger.exception("plot_scatter(): failed for day=%s", d0.date())
        # Return an error plot
        return _empty(title, f"Error generating scatter plot: {str(e)}")


def compare_stats(day, login_date=None, end=None) -> pd.DataFrame:
    """Numbers behind the comparison: this day vs everything since login."""
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    d0, d1 = _day_bounds(day)
    logger.info("compare_stats(): day=%s vs window %s → %s",
                d0.date(), login_date.date(), end.date())

    def stats(g: pd.DataFrame) -> dict:
        if g.empty:
            logger.debug("compare_stats(): empty frame → {} stats")
            return {}
        v = g["glucose_mgdl"]
        return {
            "readings": int(len(v)),
            "mean_mgdl": round(float(v.mean()), 1),
            "min_mgdl": round(float(v.min()), 1),
            "max_mgdl": round(float(v.max()), 1),
            "time_in_range_%": round(float(((v >= LOW) & (v <= HIGH)).mean() * 100), 1),
            "time_above_%": round(float((v > HIGH).mean() * 100), 1),
            "time_below_%": round(float((v < LOW).mean() * 100), 1),
        }

    day_stats = stats(db.load_glucose(d0, d1))
    overall_stats = stats(db.load_glucose(login_date, end))
    logger.info("compare_stats(): day_readings=%s overall_readings=%s",
                day_stats.get("readings"), overall_stats.get("readings"))

    rows = {
        str(d0.date()): day_stats,
        f"Overall since {login_date.date()}": overall_stats,
    }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------
# Save: every chart in its own file
# ----------------------------------------------------------------
def save_all(day, login_date=None, end=None, out_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(day).normalize()
    logger.info("save_all(): day=%s out_dir=%s", day.date(), out)

    # Build figures individually so one failure doesn't sink the whole batch
    specs = [
        (f"day_{day.date()}",                lambda: plot_day(day)),
        (f"day_carbs_{day.date()}",          lambda: plot_day_carbs(day)),
        ("overall",                          lambda: plot_overall(login_date, end)),
        ("daily_average",                    lambda: plot_daily_average(login_date, end)),
        (f"compare_{day.date()}",            lambda: plot_compare(day, login_date, end)),
        (f"scatter_{day.date()}",            lambda: plot_scatter(day, login_date, end)),
    ]

    paths = {}
    for name, build in specs:
        try:
            fig = build()
            p = out / f"{name}.html"
            fig.write_html(p)
            paths[name] = p
            logger.debug("save_all(): wrote %s", p)
        except Exception:
            logger.exception("save_all(): failed to build/write chart %r", name)

    logger.info("save_all(): wrote %d/%d chart(s) to %s",
                len(paths), len(specs), out)
    return paths