"""LangGraph tools for the coffee ops copilot.

Four tools wrapping DemandForecaster and RevenueAnalyzer:
- forecast_demand: volume forecast + stocking recommendations (daily or window scope)
- get_sales_summary: historical sales aggregations
- get_revenue_insights: daypart performance and product mix analysis
- get_model_insights: explain how the forecasting model works and its accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from scipy.stats import poisson

from models.forecaster import (
    BUSINESS_HOUR_BUCKETS,
    DemandForecaster,
    allocate_daily_minimum_levels,
    assign_demand_tiers,
    bucket_label,
    overlapping_buckets,
)
from models.analyzer import RevenueAnalyzer
from db import get_overrides_for_context

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

forecaster = DemandForecaster(DATA_DIR)
_analyzer = RevenueAnalyzer(forecaster.raw_data)

SAFETY_LEVELS: dict[str, float] = {
    "conservative": 0.95,
    "normal": 0.90,
    "lean": 0.75,
}

GROUP_BY_COLUMNS: dict[str, str] = {
    "product": "coffee_name",
    "daypart": "hour_bucket",
    "day_of_week": "weekday",
    "machine": "machine_id",
}

WINDOW_MODE_CTA = (
    "For a full-day stocking plan (06:00–21:00), ask to plan the whole day "
    "or say e.g. “Stock Machine 1 for tomorrow.”"
)

# Must match the `hour_window` string in daily blocks so DB overrides line up.
DAY_PLAN_HOUR_WINDOW = "full day (06:00–21:00)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_summary_lines(
    peak_map: dict[str, str],
    top_products: list[str],
    limit: int = 4,
) -> list[str]:
    lines: list[str] = []
    for p in top_products[:limit]:
        hb = peak_map.get(p)
        if hb:
            lines.append(f"{p}: busiest in {bucket_label(hb)}")
    return lines


def _machine_block_day(
    mid: str,
    target_date: str,
    percentile: float,
) -> dict[str, Any]:
    dt = pd.Timestamp(target_date)
    buckets = BUSINESS_HOUR_BUCKETS
    volume = forecaster.predict_volume(target_date, buckets)

    total_mean = sum(volume[mid][b]["mean"] for b in buckets)
    total_upper = int(poisson.ppf(percentile, total_mean)) if total_mean > 0 else 0

    props = forecaster.global_mix_proportions_machine_1()
    products = forecaster.machine_1_product_names()
    if not products:
        return {
            "date": target_date,
            "day_of_week": dt.day_name(),
            "plan_scope": "day",
            "framing": "stocking_recommendations",
            "hour_window": DAY_PLAN_HOUR_WINDOW,
            "buckets_used": buckets,
            "total_volume_mean": round(total_mean, 2),
            "total_volume_upper": total_upper,
            "safety_percentile": percentile,
            "capacity_note": None,
            "recommendations": [],
            "recommendations_by_tier": {"high": [], "moderate": [], "keep_stocked": []},
            "peak_demand_hints": [],
            "mix_fallback": mid == "machine_2",
            "past_adjustments": get_overrides_for_context(mid, DAY_PLAN_HOUR_WINDOW),
        }

    expected: dict[str, float] = {}
    for p in products:
        prop = props.get(p, 0.0)
        expected[p] = round(total_upper * prop, 3)

    levels, cap_note = allocate_daily_minimum_levels(
        products,
        expected,
        total_upper,
        min_floor=0,
    )
    tiers_map = assign_demand_tiers(products, expected)
    peak_bucket = forecaster.peak_hour_bucket_per_product()

    recs: list[dict[str, Any]] = []
    by_tier: dict[str, list[dict[str, Any]]] = {
        "high": [],
        "moderate": [],
        "keep_stocked": [],
    }
    for p in sorted(products, key=lambda x: (-expected.get(x, 0), x)):
        tier = tiers_map.get(p, "keep_stocked")
        row = {
            "product": p,
            "recommended_stock": int(levels.get(p, 0)),
            "expected_demand": float(expected.get(p, 0.0)),
            "proportion": round(props.get(p, 0.0), 4),
            "demand_tier": tier,
            "confidence": DemandForecaster.confidence_label(mid, len(buckets), plan_scope="day"),
        }
        recs.append(row)
        by_tier[tier].append(row)

    top_by_exp = sorted(products, key=lambda x: -expected.get(x, 0))[:5]
    peak_hints = _peak_summary_lines(peak_bucket, top_by_exp)

    return {
        "date": target_date,
        "day_of_week": dt.day_name(),
        "plan_scope": "day",
        "framing": "stocking_recommendations",
        "hour_window": DAY_PLAN_HOUR_WINDOW,
        "buckets_used": buckets,
        "total_volume_mean": round(total_mean, 2),
        "total_volume_upper": total_upper,
        "safety_percentile": percentile,
        "capacity_note": cap_note,
        "recommendations": recs,
        "recommendations_by_tier": by_tier,
        "peak_demand_hints": peak_hints,
        "mix_fallback": mid == "machine_2",
        "past_adjustments": get_overrides_for_context(mid, DAY_PLAN_HOUR_WINDOW),
    }


def _machine_block_window(
    mid: str,
    target_date: str,
    start_hour: int,
    end_hour: int,
    percentile: float,
) -> dict[str, Any]:
    dt = pd.Timestamp(target_date)
    buckets = overlapping_buckets(start_hour, end_hour)
    volume = forecaster.predict_volume(target_date, buckets)

    total_mean = sum(volume[mid][b]["mean"] for b in buckets)
    total_upper = int(poisson.ppf(percentile, total_mean)) if total_mean > 0 else 0

    mix_result = forecaster.get_merged_mix(buckets, mid)
    props = mix_result["proportions"]
    is_mix_fallback = mix_result["fallback"]
    if not props:
        props = forecaster.global_mix_proportions_machine_1()
        is_mix_fallback = True
    products = forecaster.machine_1_product_names()
    hour_window = f"{start_hour:02d}:00-{end_hour:02d}:00"

    if not products:
        return {
            "date": target_date,
            "day_of_week": dt.day_name(),
            "plan_scope": "window",
            "framing": "stocking_recommendations",
            "hour_window": hour_window,
            "buckets_used": buckets,
            "total_volume_mean": round(total_mean, 2),
            "total_volume_upper": total_upper,
            "safety_percentile": percentile,
            "capacity_note": None,
            "recommendations": [],
            "recommendations_by_tier": {"high": [], "moderate": [], "keep_stocked": []},
            "peak_demand_hints": [],
            "window_mode_cta": WINDOW_MODE_CTA,
            "mix_fallback": is_mix_fallback,
            "past_adjustments": get_overrides_for_context(mid, hour_window),
        }

    expected: dict[str, float] = {}
    for p in products:
        prop = props.get(p, 0.0)
        expected[p] = round(total_upper * prop, 3)

    levels, cap_note = allocate_daily_minimum_levels(
        products,
        expected,
        total_upper,
        min_floor=0,
    )
    tiers_map = assign_demand_tiers(products, expected)
    peak_bucket = forecaster.peak_hour_bucket_per_product()

    recs: list[dict[str, Any]] = []
    by_tier: dict[str, list[dict[str, Any]]] = {
        "high": [],
        "moderate": [],
        "keep_stocked": [],
    }
    for p in sorted(products, key=lambda x: (-expected.get(x, 0), x)):
        tier = tiers_map.get(p, "keep_stocked")
        row = {
            "product": p,
            "recommended_stock": int(levels.get(p, 0)),
            "expected_demand": float(expected.get(p, 0.0)),
            "proportion": round(props.get(p, 0.0), 4),
            "demand_tier": tier,
            "confidence": DemandForecaster.confidence_label(mid, len(buckets), plan_scope="window"),
        }
        recs.append(row)
        by_tier[tier].append(row)

    top_by_exp = sorted(products, key=lambda x: -expected.get(x, 0))[:5]
    peak_hints = _peak_summary_lines(peak_bucket, top_by_exp)

    return {
        "date": target_date,
        "day_of_week": dt.day_name(),
        "plan_scope": "window",
        "framing": "stocking_recommendations",
        "hour_window": hour_window,
        "buckets_used": buckets,
        "total_volume_mean": round(total_mean, 2),
        "total_volume_upper": total_upper,
        "safety_percentile": percentile,
        "capacity_note": cap_note,
        "recommendations": recs,
        "recommendations_by_tier": by_tier,
        "peak_demand_hints": peak_hints,
        "window_mode_cta": WINDOW_MODE_CTA,
        "mix_fallback": is_mix_fallback,
        "past_adjustments": get_overrides_for_context(mid, hour_window),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def forecast_demand(
    target_date: str,
    plan_scope: str = "window",
    start_hour: int = 6,
    end_hour: int = 21,
    safety_level: str = "normal",
    machine_id: str | None = None,
) -> dict[str, Any]:
    """Forecast demand and stocking recommendations for a future day or time window.

    Use plan_scope \"day\" for full-day stocking recommendations (default for broad
    questions like \"plan tomorrow\" or \"stock Machine 1\"). Use \"window\" when the
    user gives a specific time range (e.g. 7--10am). Always returns both machines.

    Args:
        target_date: ISO date string (e.g. \"2024-10-07\")
        plan_scope: \"day\" = full business day (06--21) stocking recommendations; \
\"window\" = time-range recommendations for [start_hour, end_hour).
        start_hour: Window start (only used when plan_scope is \"window\")
        end_hour: Window end (only used when plan_scope is \"window\")
        safety_level: \"conservative\", \"normal\", or \"lean\"
        machine_id: \"machine_1\" or \"machine_2\" to focus the UI on one machine, \
or omit/null for both. Data is always returned for both machines.
    """
    percentile = SAFETY_LEVELS.get(safety_level, 0.90)
    scope = (plan_scope or "window").strip().lower()
    if scope not in ("day", "window"):
        scope = "window"

    machines = ["machine_1", "machine_2"]
    result: dict[str, Any] = {}
    for mid in machines:
        if scope == "day":
            result[mid] = _machine_block_day(mid, target_date, percentile)
        else:
            result[mid] = _machine_block_window(
                mid, target_date, start_hour, end_hour, percentile,
            )

    if machine_id and machine_id in result:
        result["focus_machines"] = [machine_id]
    else:
        result["focus_machines"] = list(k for k in result if k in machines)

    return result


@tool
def get_sales_summary(
    start_date: str,
    end_date: str,
    machine_id: str | None = None,
    group_by: str = "product",
) -> dict[str, Any]:
    """Summarize historical sales data for a date range.

    Use this for questions about past performance: best sellers,
    busiest times, revenue totals, or machine comparisons.

    Args:
        start_date: Start of range (ISO date, e.g. "2024-06-01")
        end_date: End of range (ISO date, e.g. "2024-06-30")
        machine_id: "machine_1", "machine_2", or omit for both
        group_by: "product", "daypart", "day_of_week", or "machine"
    """
    df = forecaster.raw_data.copy()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    df = df.loc[(df["sale_date"] >= start) & (df["sale_date"] <= end)]

    if machine_id:
        df = df.loc[df["machine_id"] == machine_id]

    if df.empty:
        return {
            "summary": [],
            "total_transactions": 0,
            "total_revenue": 0.0,
            "date_range": f"{start_date} to {end_date}",
            "grouped_by": group_by,
            "insight": "No data for the selected filters.",
        }

    group_col = GROUP_BY_COLUMNS.get(group_by, "coffee_name")

    grouped = (
        df.groupby(group_col, as_index=False)
        .agg(transactions=("money", "size"), revenue=("money", "sum"))
    )
    grouped["revenue"] = grouped["revenue"].round(2)
    grouped = grouped.sort_values("revenue", ascending=False)

    return {
        "summary": grouped.to_dict(orient="records"),
        "total_transactions": int(df.shape[0]),
        "total_revenue": round(float(df["money"].sum()), 2),
        "date_range": f"{start_date} to {end_date}",
        "grouped_by": group_by,
    }


@tool
def get_revenue_insights(
    focus: str = "daypart",
    date_range_start: str | None = None,
    date_range_end: str | None = None,
) -> dict[str, Any]:
    """Analyze revenue patterns and product mix for business insights.

    Use this for questions about revenue gaps between time periods,
    top/bottom sellers by daypart, or opportunities to improve sales.

    Args:
        focus: "daypart" for revenue by time-of-day, or "product_mix" for sellers by daypart
        date_range_start: Optional start date (ISO format)
        date_range_end: Optional end date (ISO format)
    """
    date_range = None
    if date_range_start and date_range_end:
        date_range = (date_range_start, date_range_end)

    if focus == "product_mix":
        return _analyzer.product_mix_insights(date_range=date_range)
    return _analyzer.daypart_performance(date_range=date_range)


@tool
def get_model_insights() -> dict[str, Any]:
    """Explain how the forecasting model works and how accurate it is.

    Returns the demand trend, busiest time windows, machine comparison,
    and model accuracy metrics (MAE, prediction-interval coverage, test
    set size). Use when the manager asks "why", "how accurate", or
    "how does the model work".
    """
    coefficients = forecaster.get_coefficients_summary()
    metrics = forecaster.eval_metrics
    return {
        "coefficients": coefficients,
        "eval_metrics": metrics,
    }


ALL_TOOLS = [forecast_demand, get_sales_summary, get_revenue_insights, get_model_insights]
