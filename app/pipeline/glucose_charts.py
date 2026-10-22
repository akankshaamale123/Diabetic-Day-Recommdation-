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

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from storage import db

LOW = 70.0
HIGH = 180.0


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------
def default_login_date():
    """First day that has data = the day the user started (login date)."""
    days = db.available_days()
    if not days:
        raise ValueError("No data in the database yet.")
    return pd.Timestamp(days[0]).normalize()


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
    g = db.load_glucose(start, end)
    m = db.load_meals(start, end)
    title = f"Glucose — {start.date()}"
    if g.empty:
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
    return _style(fig, title)


def plot_day_carbs(day) -> go.Figure:
    start, end = _day_bounds(day)
    m = db.load_meals(start, end)
    title = f"Carbs per meal — {start.date()}"
    if m.empty:
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
    title = f"Overall glucose — {login_date.date()} → {end.date()}"
    g = db.load_glucose(login_date, end)
    if g.empty:
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
    title = f"Daily average glucose — {login_date.date()} → {end.date()}"
    g = db.load_glucose(login_date, end)
    if g.empty:
        return _empty(title, "No glucose readings in this period")
    daily = (g.assign(day=g["timestamp"].dt.normalize())
              .groupby("day", as_index=False)["glucose_mgdl"].mean())
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
    title = f"{d0.date()} vs your usual day (since {login_date.date()})"

    g_day = db.load_glucose(d0, d1)
    g_all = db.load_glucose(login_date, end)
    g_other = g_all[(g_all["timestamp"] < d0) | (g_all["timestamp"] > d1)]
    if g_day.empty or g_other.empty:
        return _empty(title, "Not enough data to compare")

    day_prof = _hourly_profile(g_day)
    usual_prof = _hourly_profile(g_other)

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
    title = f"Glucose vs Carbs — {d0.date()}"

    try:
        g = db.load_glucose(d0, d1)
        m = db.load_meals(d0, d1)

        # Filter to only meals that have glucose readings
        if g.empty or m.empty:
            return _empty(title, "Not enough data for scatter plot")

        # Only include meals that have carb data and glucose readings
        meal_glucose = pd.merge_asof(
            m.sort_values("timestamp"),
            g.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=30)
        )

        # Filter to only valid entries
        meal_glucose = meal_glucose.dropna(subset=['glucose_mgdl', 'carbs_g'])

        if meal_glucose.empty:
            return _empty(title, "Not enough data for scatter plot")

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points with colors based on glucose levels
        colors = []
        for _, row in meal_glucose.iterrows():
            glucose = row['glucose_mgdl']
            if glucose < LOW:
                colors.append('blue')
            elif glucose > HIGH:
                colors.append('red')
            else:
                colors.append('green')

        # Create the scatter plot with proper formatting
        scatter_trace = go.Scatter(
            x=meal_glucose['carbs_g'],
            y=meal_glucose['glucose_mgdl'],
            mode='markers',
            marker=dict(
                color=colors,
                size=10,
                opacity=0.7
            ),
            text=[f"Meal: {row['food_name']}<br>Carbs: {row['carbs_g']:.0f}g<br>Glucose: {row['glucose_mgdl']:.0f}mg/dL"
                  for _, row in meal_glucose.iterrows()],
            hovertemplate="%{text}<extra></extra>",
            name="Glucose vs Carbs"
        )

        fig.add_trace(scatter_trace)

        # Add trend line with error handling
        try:
            if len(meal_glucose) >= 2:
                # Fit a simple linear regression
                z = np.polyfit(meal_glucose['carbs_g'], meal_glucose['glucose_mgdl'], 1)
                p = np.poly1d(z)
                trend_x = meal_glucose['carbs_g']
                trend_y = p(trend_x)

                fig.add_trace(go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode='lines',
                    name='Trend line',
                    line=dict(color='black', width=2, dash='dot'),
                    showlegend=True
                ))
        except Exception as e:
            # If trend line calculation fails, continue without it
            pass

        # Add target bands
        _target_band(fig)

        # Style the chart
        fig = _style(fig, title, x_title="Carbs (g)", y_title="Glucose (mg/dL)")

        return fig
    except Exception as e:
        # Handle any other exceptions
        print(f"Error in plot_scatter: {e}")
        return _empty(title, f"Error generating scatter plot: {str(e)}")


def compare_stats(day, login_date=None, end=None) -> pd.DataFrame:
    """Numbers behind the comparison: this day vs everything since login."""
    login_date = pd.Timestamp(login_date) if login_date is not None else default_login_date()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    d0, d1 = _day_bounds(day)

    def stats(g: pd.DataFrame) -> dict:
        if g.empty:
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

    rows = {
        str(d0.date()): stats(db.load_glucose(d0, d1)),
        f"Overall since {login_date.date()}": stats(db.load_glucose(login_date, end)),
    }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------
# Save: every chart in its own file
# ----------------------------------------------------------------
def save_all(day, login_date=None, end=None, out_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(day).normalize()

    figs = {
        f"day_{day.date()}": plot_day(day),
        f"day_carbs_{day.date()}": plot_day_carbs(day),
        "overall": plot_overall(login_date, end),
        "daily_average": plot_daily_average(login_date, end),
        f"compare_{day.date()}": plot_compare(day, login_date, end),
        f"scatter_{day.date()}": plot_scatter(day, login_date, end),
    }
    paths = {}
    for name, fig in figs.items():
        p = out / f"{name}.html"
        fig.write_html(p)
        paths[name] = p
    return paths