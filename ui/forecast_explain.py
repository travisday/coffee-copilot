"""Ops-friendly explanations for forecast tool output and forecast_data normalization."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import streamlit as st

from models.forecaster import bucket_label

_LOG = logging.getLogger("coffee_copilot.ui.forecast")


def _canonical_machine_key(key: object) -> str | None:
    if key is None:
        return None
    s = str(key).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("machine_1", "machine1", "m1"):
        return "machine_1"
    if s in ("machine_2", "machine2", "m2"):
        return "machine_2"
    return None


def normalize_forecast_data(data: dict | None) -> dict | None:
    """Map variant keys (e.g. 'Machine 1', 'm1') to machine_1 / machine_2.

    Unknown keys whose values look like per-machine forecast blocks are assigned
    to empty canonical slots in encounter order.
    """
    if not data or not isinstance(data, dict):
        return None

    out: dict[str, dict[str, Any]] = {}
    unknown_blocks: list[dict[str, Any]] = []

    for k, v in data.items():
        if k == "focus_machines":
            out["focus_machines"] = v
            continue
        if not isinstance(v, dict):
            continue
        canon = _canonical_machine_key(k)
        if canon:
            out[canon] = v
        elif _looks_like_machine_forecast_block(v):
            unknown_blocks.append(v)
        else:
            out[str(k)] = v

    for slot in ("machine_1", "machine_2"):
        if slot not in out and unknown_blocks:
            out[slot] = unknown_blocks.pop(0)

    if not out:
        return data
    _LOG.debug(
        "normalize_forecast_data: keys_in=%s keys_out=%s",
        list(data.keys()),
        list(out.keys()),
    )
    return out


def _looks_like_machine_forecast_block(v: dict) -> bool:
    return "recommendations" in v or "total_volume_mean" in v


def iter_forecast_machines(forecast_data: dict):
    """Stable machine_1 then machine_2, then any other keys (legacy messages)."""
    order = ("machine_1", "machine_2")
    seen: set[str] = set()
    for mid in order:
        if mid in forecast_data and isinstance(forecast_data[mid], dict):
            seen.add(mid)
            yield mid, forecast_data[mid]
    for mid, data in forecast_data.items():
        if mid not in seen and isinstance(data, dict):
            yield mid, data


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.0f}%"


def _safety_sentence(percentile: float, plan_scope: str | None) -> str:
    p = int(round(100 * float(percentile)))
    scope = (plan_scope or "window").lower()
    if scope == "day":
        return (
            f"Stocking targets use a **{p}th percentile** cap on **full-day** "
            "expected total drinks, split proportionally across products by "
            "historical sales mix."
        )
    return (
        f"Stocking targets for this window use a **{p}th percentile** cap on expected "
        "drinks in that period, split proportionally across products by time-specific "
        "sales mix (higher = more buffer). Change the safety level in the sidebar."
    )


def _render_forecast_demand_explained(tc: dict, forecast_data: dict | None) -> None:
    inputs = tc.get("inputs") or {}
    out = tc.get("output")
    fd = normalize_forecast_data(out if isinstance(out, dict) else forecast_data)

    st.markdown("**What we used**")
    target = inputs.get("target_date", "—")
    sh, eh = inputs.get("start_hour"), inputs.get("end_hour")
    plan_scope = (inputs.get("plan_scope") or "window").strip().lower()
    if fd:
        d_any = next(iter_forecast_machines(fd), None)
        if d_any:
            _, block0 = d_any
            plan_scope = (block0.get("plan_scope") or plan_scope or "window").strip().lower()

    safety = inputs.get("safety_level", "normal")
    if plan_scope == "day":
        st.markdown(
            f"- **Date:** {target}  \n"
            f"- **Scope:** Full business day (06:00–21:00), all 3-hour buckets summed  \n"
            f"- **Safety level:** {safety}  \n"
            f"- **Stocking target:** total forecasted demand at the chosen safety level, "
            f"split proportionally across products"
        )
    else:
        win = (
            f"{int(sh):02d}:00–{int(eh):02d}:00"
            if sh is not None and eh is not None
            else "—"
        )
        st.markdown(
            f"- **Date:** {target}  \n"
            f"- **Time window:** {win}  \n"
            f"- **Safety level:** {safety}"
        )

    if not fd:
        st.caption("Forecast numbers were not attached; see technical details if enabled.")
        return

    st.markdown("**How this maps to the data**")
    first = next(iter_forecast_machines(fd), None)
    if first:
        _, d0 = first
        buckets = d0.get("buckets_used") or []
        if buckets:
            bl = ", ".join(bucket_label(b) for b in buckets)
            if plan_scope == "day":
                st.markdown(
                    f"Volume is summed across **all business 3-hour buckets** for the day: {bl}."
                )
            else:
                st.markdown(
                    f"Forecasts use **3-hour buckets** overlapping your window: {bl}."
                )

    st.markdown("**Volume and uncertainty (per machine)**")
    safety_pct = 0.9
    for mid, d in iter_forecast_machines(fd):
        label = mid.replace("_", " ").title()
        mean = d.get("total_volume_mean", "—")
        upper = d.get("total_volume_upper", "—")
        safety_pct = float(d.get("safety_percentile", safety_pct))
        if mid == "machine_2":
            unc = (
                "**Uncertainty:** Product mix follows Machine 1 patterns (limited "
                "Machine 2 data). Treat quantities as a starting point."
            )
        else:
            unc = (
                "**Uncertainty:** The model does not treat weekdays differently yet; "
                "numbers reflect time-of-day, machine, and long-run growth."
            )

        st.markdown(f"**{label}**")
        st.markdown(
            f"- **Model prediction (average rate):** {mean} drinks  \n"
            f"- **Planning upper bound ({_pct(safety_pct)} percentile):** {upper} drinks  \n"
            f"- {unc}"
        )
        if d.get("mix_fallback"):
            st.caption(
                "Product mix for this machine uses Machine 1’s historical mix — "
                "you know local preferences better."
            )
        if plan_scope == "day" and d.get("total_volume_upper") is not None:
            st.caption(
                f"Total stocking target: **{d['total_volume_upper']}** units "
                f"(forecasted demand at {d.get('safety_percentile', 0.9):.0%} safety level)."
            )
        cn = d.get("capacity_note")
        if cn:
            st.warning(str(cn))

    st.markdown(
        "The model predicts an average *rate*, not a discrete count. "
        "We convert that rate into an integer planning bound using the Poisson "
        "distribution at your chosen safety level, then split that total across "
        "products by their historical sales mix."
    )

    st.markdown(_safety_sentence(safety_pct, plan_scope))

    st.markdown(
        "**Stocking recommendations (upper bound × mix, proportional split)**"
        if plan_scope == "day"
        else "**Stocking recommendations for this window (upper bound × time-specific mix)**"
    )
    rows = []
    for mid, d in iter_forecast_machines(fd):
        mlabel = mid.replace("_", " ").title()
        for r in d.get("recommendations") or []:
            prop = r.get("proportion")
            pct_s = _pct(prop) if prop is not None else "—"
            row = {
                "Machine": mlabel,
                "Product": r.get("product", "—"),
                "Rec. stock": r.get("recommended_stock", "—"),
                "Expected (fractional)": r.get("expected_demand", "—"),
                "Share of mix": pct_s,
                "Note": r.get("confidence", ""),
            }
            tier = r.get("demand_tier")
            if tier:
                row["Tier"] = tier
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No per-product lines in this run (very low volume).")


def _render_sales_summary_explained(tc: dict) -> None:
    out = tc.get("output")
    if not isinstance(out, dict):
        return
    st.markdown("**Sales summary**")
    st.caption(
        f"{out.get('total_transactions', 0)} transactions · "
        f"${out.get('total_revenue', 0):,.2f} revenue · "
        f"{out.get('date_range', '')}"
    )
    insight = out.get("insight")
    if insight:
        st.info(insight)
    summary = out.get("summary") or []
    if isinstance(summary, list) and summary:
        df = pd.DataFrame(summary[:12])
        st.dataframe(df, hide_index=True, width="stretch")


def _render_revenue_insights_explained(tc: dict) -> None:
    out = tc.get("output")
    if not isinstance(out, dict):
        return
    st.markdown("**Revenue insight**")
    ins = out.get("insight")
    if ins:
        st.markdown(str(ins))
    if "by_machine" in out:
        for mid, rows in (out.get("by_machine") or {}).items():
            st.caption(mid.replace("_", " ").title())
            if isinstance(rows, list) and rows:
                st.dataframe(pd.DataFrame(rows[:8]), hide_index=True, width="stretch")
    if "by_daypart" in out:
        bd = out.get("by_daypart") or {}
        if isinstance(bd, dict) and bd:
            for hb, block in list(bd.items())[:4]:
                st.caption(f"Daypart **{hb}**")
                if not isinstance(block, dict):
                    continue
                top3 = block.get("top_3") or []
                if top3:
                    st.dataframe(
                        pd.DataFrame(top3),
                        hide_index=True,
                        width="stretch",
                    )


def _render_model_insights_explained(tc: dict) -> None:
    out = tc.get("output")
    if not isinstance(out, dict):
        return

    coefficients = out.get("coefficients") or []
    metrics = out.get("eval_metrics") or {}

    if coefficients:
        st.markdown("**What the model learned**")
        for c in coefficients:
            interp = c.get("interpretation", "")
            if interp:
                st.markdown(f"- {interp}")

    if metrics:
        st.markdown("**Model accuracy**")
        mae = metrics.get("mae")
        coverage = metrics.get("interval_coverage_90")
        n_test = metrics.get("n_test")
        parts: list[str] = []
        if mae is not None:
            parts.append(f"**MAE:** {mae:.2f} drinks per 3-hour window")
        if coverage is not None:
            parts.append(
                f"**90% prediction-interval coverage:** {coverage:.0%} of "
                "actuals fall inside the interval"
            )
        if n_test is not None:
            parts.append(f"**Test set size:** {n_test} observations")
        for p in parts:
            st.markdown(f"- {p}")


def _show_debug_json(tool_calls: list[dict]) -> None:
    env = os.environ.get("COFFEE_COPILOT_DEBUG", "").lower() in ("1", "true", "yes")
    if not env and not st.session_state.get("copilot_debug_tools"):
        return
    with st.expander("Technical details (debug)"):
        st.json(tool_calls)


def render_why_this_recommendation(
    tool_calls: list[dict] | None,
    forecast_data: dict | None,
) -> None:
    """Transparency expander: plain language + tables; optional raw JSON."""
    if not tool_calls:
        return

    with st.expander("Why this recommendation"):
        st.caption(
            "How we estimated volume, uncertainty, and stocking recommendations per "
            "product — without spreadsheet jargon."
        )
        for i, tc in enumerate(tool_calls):
            if i:
                st.divider()
            name = tc.get("name") or ""
            if name == "forecast_demand":
                _render_forecast_demand_explained(tc, forecast_data)
            elif name == "get_sales_summary":
                _render_sales_summary_explained(tc)
            elif name == "get_revenue_insights":
                _render_revenue_insights_explained(tc)
            elif name == "get_model_insights":
                _render_model_insights_explained(tc)
            else:
                st.markdown(f"**Tool:** `{name}`")
                if tc.get("inputs"):
                    st.caption("Inputs (summary)")
                    st.json(tc.get("inputs"))
                if tc.get("output") is not None:
                    st.caption("Output (summary)")
                    st.json(tc.get("output"))

        _show_debug_json(tool_calls)
