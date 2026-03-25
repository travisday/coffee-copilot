"""Override sliders and confirmed-plan rendering for forecast messages."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from db import get_overrides_for_message, save_override
from ui.forecast_explain import iter_forecast_machines, normalize_forecast_data

_LOG = logging.getLogger("coffee_copilot.ui.overrides")


# ── Helpers ──────────────────────────────────────────────────────────


def _plan_final_session_key(message_id: str, machine_id: str) -> str:
    return f"_plan_final_{message_id}_{machine_id}"


def _clear_forecast_acceptance(message_id: str) -> None:
    for mid in ("machine_1", "machine_2"):
        st.session_state.pop(f"_accepted_{message_id}_{mid}", None)
        st.session_state.pop(f"_plan_final_{message_id}_{mid}", None)


def _model_and_final_maps(
    message_id: str,
    machine_id: str,
    recs: list,
) -> tuple[dict[str, int], dict[str, int]]:
    """Model baseline from forecast recs; final merges session snapshot or DB overrides."""
    model_map = {r["product"]: int(r["recommended_stock"]) for r in recs}
    snap = st.session_state.get(_plan_final_session_key(message_id, machine_id))
    if isinstance(snap, dict) and snap:
        final_map = {}
        for r in recs:
            p = r["product"]
            final_map[p] = int(snap[p]) if p in snap else model_map[p]
        return model_map, final_map

    final_map = dict(model_map)
    for row in get_overrides_for_message(message_id):
        if row.get("machine_id") != machine_id:
            continue
        p = row.get("product")
        if p in final_map:
            final_map[p] = int(row["adjusted_rec"])
    return model_map, final_map


def _render_confirmed_plan_summary(
    recs: list,
    model_by_product: dict[str, int],
    final_by_product: dict[str, int],
) -> None:
    out = []
    for r in recs:
        product = r["product"]
        m = model_by_product[product]
        f = final_by_product.get(product, m)
        delta = f - m
        out.append({
            "Product": product,
            "Model suggestion": m,
            "Your plan": f,
            "Change": f"{delta:+d}" if delta else "—",
        })
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")


# ── Dialog ───────────────────────────────────────────────────────────


@st.dialog("Why are you adjusting?")
def override_reason_dialog() -> None:
    ctx = st.session_state["_override_ctx"]
    machine_label = ctx["machine_id"].replace("_", " ").title()
    changes = {p: (o, a) for p, (o, a) in ctx["slider_values"].items() if o != a}
    for product, (original, adjusted) in changes.items():
        st.markdown(f"**{product}:** {original} → {adjusted}")
    reason = st.text_input(
        "Add a note (optional — helps future recommendations)",
        placeholder="e.g. event nearby, machine was down yesterday…",
    )
    c1, c2 = st.columns(2)
    if c1.button("Confirm", type="primary", width="stretch"):
        for product, (original, adjusted) in changes.items():
            save_override(
                message_id=ctx["message_id"],
                machine_id=ctx["machine_id"],
                date=ctx["date"],
                hour_window=ctx["hour_window"],
                product=product,
                original_rec=original,
                adjusted_rec=adjusted,
                reason=reason or None,
            )
        st.session_state[ctx["accepted_key"]] = True
        st.session_state[
            _plan_final_session_key(ctx["message_id"], ctx["machine_id"])
        ] = {p: a for p, (o, a) in ctx["slider_values"].items()}
        del st.session_state["_override_ctx"]
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        del st.session_state["_override_ctx"]
        st.rerun()


# ── Main render function ─────────────────────────────────────────────


def render_overrides(forecast_data: dict, message_id: str) -> None:
    fd = normalize_forecast_data(forecast_data) or forecast_data
    if not isinstance(fd, dict) or not fd:
        return

    m1 = isinstance(fd.get("machine_1"), dict)
    m2 = isinstance(fd.get("machine_2"), dict)
    if not m1 or not m2:
        _LOG.warning(
            "forecast_data missing machine block(s): machine_1=%s machine_2=%s keys=%s",
            m1,
            m2,
            list(fd.keys()),
        )
        st.warning(
            "Expected data for both machines was incomplete. "
            "If sliders are missing for one machine, try **Clear** in the header and ask again."
        )

    rows = list(iter_forecast_machines(fd))
    if not rows:
        return

    focus = fd.get("focus_machines")
    if isinstance(focus, list) and focus:
        rows = [(mid, data) for mid, data in rows if mid in focus]

    def _accepted(mid: str) -> bool:
        return bool(st.session_state.get(f"_accepted_{message_id}_{mid}"))

    def _resolved(mid: str, data: dict) -> bool:
        if _accepted(mid):
            return True
        if not data.get("recommendations"):
            return True
        return False

    def _has_pending_confirmation() -> bool:
        return any(
            bool(data.get("recommendations")) and not _accepted(mid)
            for mid, data in rows
        )

    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            "### Your plan — adjust and confirm"
            if _has_pending_confirmation()
            else "### Confirmed plan"
        )
    with h2:
        if st.button(
            "Reset",
            key=f"reset_acc_{message_id}",
            help="Clear confirmed state for both machines on this answer.",
        ):
            _clear_forecast_acceptance(message_id)
            st.rerun()

    if all(_resolved(mid, d) for mid, d in rows):
        st.success("Stocking plans confirmed for all machines.")

    for machine_id, data in rows:
        recs = data.get("recommendations", [])
        machine_label = machine_id.replace("_", " ").title()
        window = data.get("hour_window", "")
        day = data.get("day_of_week", "")

        if _accepted(machine_id):
            if recs:
                model_map, final_map = _model_and_final_maps(
                    message_id, machine_id, recs,
                )
                with st.container(border=True):
                    st.markdown(f"**{machine_label}** · {day} · **{window}**")
                    st.success(f"**{machine_label}** — plan confirmed.")
                    st.caption(
                        "Model suggestion vs. what you locked in for this run."
                    )
                    _render_confirmed_plan_summary(recs, model_map, final_map)
                    if st.button(
                        "Change plan",
                        key=f"chg_{message_id}_{machine_id}",
                        help="Re-open sliders for this machine only.",
                    ):
                        st.session_state.pop(
                            f"_accepted_{message_id}_{machine_id}", None,
                        )
                        st.session_state.pop(
                            _plan_final_session_key(message_id, machine_id),
                            None,
                        )
                        st.rerun()
            else:
                with st.container(border=True):
                    st.success(f"**{machine_label}** — plan confirmed.")
            continue

        if not recs:
            with st.container(border=True):
                st.info(
                    f"**{machine_label}** — "
                    "no line items to adjust (expected volume is very low for this window)."
                )
            continue

        past_adj = data.get("past_adjustments") or []
        latest_adj: dict[str, int] = {}
        for adj in past_adj:
            p = adj.get("product") or adj.get("name", "")
            if p and p not in latest_adj:
                latest_adj[p] = int(adj["adjusted_rec"])

        with st.container(border=True):
            st.markdown(f"**{machine_label}** · {day} · **{window}**")
            by_tier = data.get("recommendations_by_tier")
            if isinstance(by_tier, dict) and by_tier:
                for label, key in (
                    ("Check first (high demand)", "high"),
                    ("Moderate demand", "moderate"),
                    ("Keep stocked", "keep_stocked"),
                ):
                    tier_recs = by_tier.get(key) or []
                    if not tier_recs:
                        continue
                    names = ", ".join(
                        f"**{r['product']}**: {int(r['recommended_stock'])}"
                        for r in tier_recs
                    )
                    st.markdown(f"*{label}:* {names}")

            accepted_key = f"_accepted_{message_id}_{machine_id}"
            slider_values: dict[str, tuple[int, int]] = {}
            ncols = min(4, max(1, len(recs)))
            for chunk_start in range(0, len(recs), ncols):
                chunk = recs[chunk_start : chunk_start + ncols]
                cols = st.columns(len(chunk))
                for ci, rec in enumerate(chunk):
                    product = rec["product"]
                    model_rec = int(rec["recommended_stock"])
                    default = latest_adj.get(product, model_rec)
                    cap = max(model_rec * 3, default * 3, 5)
                    help_text = f"Suggested stock: {model_rec}."
                    if product in latest_adj:
                        help_text += f" You previously adjusted to {latest_adj[product]}."
                    val = cols[ci].slider(
                        product,
                        min_value=0,
                        max_value=cap,
                        value=default,
                        step=1,
                        key=f"sl_{message_id}_{machine_id}_{product}_{chunk_start + ci}",
                        help=help_text,
                    )
                    slider_values[product] = (model_rec, val)

            has_changes = any(o != a for o, a in slider_values.values())

            if has_changes:
                if st.button(
                    "Confirm with changes",
                    key=f"accept_{message_id}_{machine_id}",
                    type="primary",
                ):
                    st.session_state["_override_ctx"] = {
                        "message_id": message_id,
                        "machine_id": machine_id,
                        "date": data.get("date"),
                        "hour_window": data.get("hour_window"),
                        "slider_values": dict(slider_values),
                        "accepted_key": accepted_key,
                    }
                    st.rerun()
            else:
                if st.button(
                    "Confirm stocking plan",
                    key=f"accept_{message_id}_{machine_id}",
                    type="primary",
                ):
                    st.session_state[accepted_key] = True
                    st.session_state[_plan_final_session_key(message_id, machine_id)] = {
                        r["product"]: int(r["recommended_stock"]) for r in recs
                    }
                    st.rerun()
