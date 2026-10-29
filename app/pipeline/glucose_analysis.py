# app/pipeline/glucose_analysis.py
"""
AI 4 — Glucose pattern analysis.

Reads meals + CGM readings from the SQLite store (`storage.db`).
No sample data, no hardcoded dates — every function takes a real
time window drawn from the user's history.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from storage import db

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Load + align
# ----------------------------------------------------------------
def load_window(start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load glucose + meals for a real past window (both may be empty)."""
    logger.debug("load_window(): start=%s end=%s", start, end)
    g = db.load_glucose(start, end)
    m = db.load_meals(start, end)
    logger.info("load_window(): glucose_rows=%d meal_rows=%d for %s → %s",
                len(g), len(m),
                pd.Timestamp(start).date(), pd.Timestamp(end).date())
    return g, m


# ----------------------------------------------------------------
# Single‑day chart (glucose line + meal markers + carbs per meal)
# ----------------------------------------------------------------
def plot_glucose_day(
    day,
    spike_threshold: float = 180.0,
    low_threshold: float = 70.0,
) -> go.Figure:
    """
    Chart for ONE past day pulled from the DB.
    Top subplot : glucose line (mg/dL) with meal vertical markers
    Bottom      : carbs per meal (bar) — gives nutrition context
    """
    day = pd.Timestamp(day).normalize()
    day_end = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    logger.info("plot_glucose_day(): day=%s spike=%s low=%s",
                day.date(), spike_threshold, low_threshold)

    g = db.load_glucose(day, day_end)
    m = db.load_meals(day, day_end)
    logger.debug("plot_glucose_day(): glucose_rows=%d meal_rows=%d",
                 len(g), len(m))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.06,
        subplot_titles=("Glucose", "Carbs per meal (g)"),
    )

    # --- glucose line ---
    if not g.empty:
        fig.add_trace(
            go.Scatter(
                x=g["timestamp"], y=g["glucose_mgdl"],
                mode="lines+markers", name="Glucose",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
                hovertemplate="%{x|%H:%M}<br>%{y:.0f} mg/dL<extra></extra>",
            ),
            row=1, col=1,
        )
    else:
        logger.warning("plot_glucose_day(): no glucose readings for %s", day.date())
        fig.add_annotation(
            text="No glucose readings for this day",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False, row=1, col=1,
        )

    # target band (visual only — not data)
    fig.add_hrect(
        y0=low_threshold, y1=spike_threshold,
        fillcolor="rgba(0,180,0,0.08)", line_width=0,
        row=1, col=1,
    )
    fig.add_hline(y=spike_threshold, line_dash="dash",
                  line_color="red", row=1, col=1)

    # --- meal markers on top ---
    for _, meal in m.iterrows():
        fig.add_vline(x=meal["timestamp"], line_dash="dot",
                      line_color="orange", line_width=1.2,
                      row=1, col=1)

    # --- carbs per meal on bottom ---
    if not m.empty:
        agg = (
            m.assign(minute=m["timestamp"].dt.floor("15min"))
             .groupby(["minute", "food_name"], as_index=False)["carbs_g"]
             .sum()
        )
        fig.add_trace(
            go.Bar(
                x=agg["minute"], y=agg["carbs_g"],
                name="Carbs",
                marker_color="orange",
                text=agg["food_name"],
                hovertemplate="%{x|%H:%M}<br>%{text}<br>%{y:.0f} g carbs<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="mg/dL", row=1, col=1)
    fig.update_yaxes(title_text="g", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_layout(
        title=f"Glucose + Meals — {day.date()}",
        template="plotly_white",
        height=520,
        hovermode="x unified",
        showlegend=False,
    )
    logger.debug("plot_glucose_day(): figure built for %s", day.date())
    return fig


# ----------------------------------------------------------------
# Multi‑day time series — "each day their glucose graph"
# ----------------------------------------------------------------
def plot_glucose_history(start, end,
                         spike_threshold: float = 180.0) -> go.Figure:
    """
    Continuous glucose line across the full past window, with
    day separators and a daily mean overlay.
    """
    logger.info("plot_glucose_history(): window %s → %s (spike=%s)",
                pd.Timestamp(start).date(), pd.Timestamp(end).date(),
                spike_threshold)
    g, m = load_window(start, end)

    fig = go.Figure()

    if not g.empty:
        fig.add_trace(
            go.Scatter(
                x=g["timestamp"], y=g["glucose_mgdl"],
                mode="lines", name="Glucose",
                line=dict(color="#1f77b4", width=1.6),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f}<extra></extra>",
            )
        )

        # daily mean overlay
        daily = (
            g.assign(day=g["timestamp"].dt.normalize())
             .groupby("day", as_index=False)["glucose_mgdl"].mean()
        )
        fig.add_trace(
            go.Scatter(
                x=daily["day"], y=daily["glucose_mgdl"],
                mode="lines+markers", name="Daily mean",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
            )
        )

        # meal markers
        for _, meal in m.iterrows():
            fig.add_vline(x=meal["timestamp"],
                          line_dash="dot", line_color="grey",
                          line_width=0.7)
        logger.debug("plot_glucose_history(): %d glucose points, "
                     "%d daily means, %d meal markers",
                     len(g), len(daily), len(m))
    else:
        logger.warning("plot_glucose_history(): empty glucose data for window "
                       "%s → %s",
                       pd.Timestamp(start).date(), pd.Timestamp(end).date())

    fig.add_hline(y=spike_threshold, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"Glucose history — {pd.Timestamp(start).date()} → {pd.Timestamp(end).date()}",
        xaxis_title="Time", yaxis_title="mg/dL",
        template="plotly_white", height=460,
    )
    return fig


# ----------------------------------------------------------------
# Post‑meal response metric per meal (2 h window)
# ----------------------------------------------------------------
def _meal_metrics(g: pd.DataFrame, meal: pd.Series,
                  window_minutes: int = 120,
                  spike_threshold: float = 180.0) -> dict | None:
    t0 = meal["timestamp"]
    t1 = t0 + pd.Timedelta(minutes=window_minutes)
    w = g[(g["timestamp"] >= t0) & (g["timestamp"] <= t1)]
    if len(w) < 3:
        logger.debug("_meal_metrics(): insufficient readings (%d) for meal "
                     "%s at %s — skipped",
                     len(w), meal.get("food_name"), t0)
        return None

    baseline = float(w.iloc[0]["glucose_mgdl"])
    peak_idx = w["glucose_mgdl"].idxmax()
    peak = float(w.loc[peak_idx, "glucose_mgdl"])
    peak_ts = w.loc[peak_idx, "timestamp"]
    delta = peak - baseline
    time_to_peak = (peak_ts - t0).total_seconds() / 60.0

    y = np.clip(w["glucose_mgdl"].values - baseline, 0, None)
    t = (w["timestamp"] - t0).dt.total_seconds().values / 60.0
    iauc = float(np.trapezoid(y, t)) if len(y) > 1 else 0.0

    result = {
        "timestamp": meal["timestamp"],
        "food_name": meal["food_name"],
        "carbs_g": meal.get("carbs_g"),
        "sugar_g": meal.get("sugar_g"),
        "baseline_mgdl": round(baseline, 1),
        "peak_mgdl": round(peak, 1),
        "delta_mgdl": round(delta, 1),
        "time_to_peak_min": round(time_to_peak, 1),
        "iauc_above_baseline": round(iauc, 1),
        "spike_occurred": peak >= spike_threshold,
    }
    logger.debug(
        "_meal_metrics(): %s @ %s → base=%.1f peak=%.1f Δ=%.1f spike=%s",
        result["food_name"], t0, baseline, peak, delta,
        result["spike_occurred"],
    )
    return result


def analyze_meal_responses(start, end,
                           window_minutes: int = 120,
                           spike_threshold: float = 180.0) -> pd.DataFrame:
    """Real historical meal‑by‑meal glucose responses."""
    logger.info("analyze_meal_responses(): window %s → %s (window=%dmin, spike=%s)",
                pd.Timestamp(start).date(), pd.Timestamp(end).date(),
                window_minutes, spike_threshold)
    g, m = load_window(start, end)
    if g.empty or m.empty:
        logger.warning("analyze_meal_responses(): empty input "
                       "(glucose=%d, meals=%d) — returning empty frame",
                       len(g), len(m))
        return pd.DataFrame()

    rows = []
    skipped = 0
    for _, meal in m.iterrows():
        r = _meal_metrics(g, meal, window_minutes, spike_threshold)
        if r is not None:
            rows.append(r)
        else:
            skipped += 1

    df = pd.DataFrame(rows)
    logger.info("analyze_meal_responses(): %d meal(s) analysed, %d skipped",
                len(df), skipped)
    return df


# ----------------------------------------------------------------
# Pattern summary from real history
# ----------------------------------------------------------------
def summarize_patterns(start, end,
                       high_delta: float = 50.0,
                       spike_threshold: float = 180.0) -> dict:
    """Which foods repeatedly pushed glucose up in this window."""
    logger.info("summarize_patterns(): window %s → %s (high_delta=%s)",
                pd.Timestamp(start).date(), pd.Timestamp(end).date(),
                high_delta)
    resp = analyze_meal_responses(start, end,
                                  spike_threshold=spike_threshold)
    if resp.empty:
        logger.warning("summarize_patterns(): no meal-response data in window")
        return {"message": "No meal‑response data in this window."}

    per_food = (
        resp.groupby("food_name")
            .agg(
                times_eaten=("food_name", "size"),
                avg_delta=("delta_mgdl", "mean"),
                max_delta=("delta_mgdl", "max"),
                avg_peak=("peak_mgdl", "mean"),
                spikes=("spike_occurred", "sum"),
                avg_carbs=("carbs_g", "mean"),
            )
            .reset_index()
            .sort_values("avg_delta", ascending=False)
    )
    flagged = per_food[per_food["avg_delta"] >= high_delta]

    logger.info(
        "summarize_patterns(): meals=%d, flagged_foods=%d (%s)",
        len(resp), len(flagged), flagged["food_name"].tolist(),
    )

    return {
        "window": f"{pd.Timestamp(start).date()} → {pd.Timestamp(end).date()}",
        "meals_analysed": int(len(resp)),
        "high_response_foods": flagged["food_name"].tolist(),
        "per_food": per_food.round(1).to_dict(orient="records"),
        "overall_avg_delta": round(float(resp["delta_mgdl"].mean()), 1),
        "spike_rate_pct": round(float(resp["spike_occurred"].mean() * 100), 1),
    }


# ----------------------------------------------------------------
# Daily nutrition rollup (nutrition per day from real meals)
# ----------------------------------------------------------------
def daily_nutrition(start, end) -> pd.DataFrame:
    logger.info("daily_nutrition(): window %s → %s",
                pd.Timestamp(start).date(), pd.Timestamp(end).date())
    _, m = load_window(start, end)
    if m.empty:
        logger.warning("daily_nutrition(): no meals in window — empty frame")
        return pd.DataFrame()
    m = m.assign(day=m["timestamp"].dt.normalize())
    df = (
        m.groupby("day", as_index=False)
         .agg(calories_kcal=("calories_kcal", "sum"),
              carbs_g=("carbs_g", "sum"),
              sugar_g=("sugar_g", "sum"),
              protein_g=("protein_g", "sum"),
              fat_g=("fat_g", "sum"),
              fiber_g=("fiber_g", "sum"))
         .round(1)
    )
    logger.info("daily_nutrition(): %d day(s) rolled up", len(df))
    return df


# ----------------------------------------------------------------
# Save charts to disk (no display dependency)
# ----------------------------------------------------------------
def save_day_charts(start, end, out_dir: str | Path = "reports") -> list[Path]:
    """
    For every day in the window, write one HTML file
    reports/glucose_YYYY-MM-DD.html.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("save_day_charts(): window %s → %s, out_dir=%s",
                pd.Timestamp(start).date(), pd.Timestamp(end).date(), out)

    days = pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize(), freq="D")
    logger.debug("save_day_charts(): %d day(s) to render", len(days))

    written = []
    for d in days:
        try:
            fig = plot_glucose_day(d)
            path = out / f"glucose_{d.date()}.html"
            fig.write_html(path)
            written.append(path)
            logger.debug("save_day_charts(): wrote %s", path)
        except Exception:
            logger.exception("save_day_charts(): failed to render/write chart "
                             "for day=%s", d.date())

    logger.info("save_day_charts(): wrote %d/%d chart(s) to %s",
                len(written), len(days), out)
    return written