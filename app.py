"""Streamlit frontend for Coffee Ops Copilot."""

from __future__ import annotations

import ast
import json
import logging
import uuid
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_agent
from agent.tools import _forecaster
from ui.forecast_explain import (
    iter_forecast_machines,
    normalize_forecast_data,
    render_why_this_recommendation,
)
from db import (
    clear_conversation,
    get_conversation_history,
    get_feedback_comments,
    get_feedback_stats,
    get_overrides_for_message,
    init_db,
    save_feedback,
    save_message,
    save_override,
)

st.set_page_config(page_title="Coffee Ops Copilot", page_icon="☕", layout="wide")
init_db()

_LOG = logging.getLogger("coffee_copilot.ui")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ── Cached resources ────────────────────────────────────────────────

@st.cache_resource
def get_agent():
    return build_agent()


@st.cache_data
def get_quick_stats() -> dict:
    df = _forecaster.raw_data
    return {
        "total_transactions": len(df),
        "total_revenue": round(float(df["money"].sum()), 2),
        "top_product": df["coffee_name"].value_counts().idxmax(),
        "date_range_start": str(df["sale_date"].min()),
        "date_range_end": str(df["sale_date"].max()),
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _safe_json_dumps(obj) -> str:
    def _default(o):
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=_default)


def _parse_tool_content(raw_content) -> object:
    """Best-effort parse of a ToolMessage's content into a Python object."""
    if isinstance(raw_content, dict):
        return raw_content
    if isinstance(raw_content, list):
        # LangChain may wrap JSON in content blocks (list of str or dict).
        text_parts: list[str] = []
        for block in raw_content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))
                elif "content" in block:
                    text_parts.append(str(block["content"]))
        if text_parts:
            merged = "".join(text_parts)
            try:
                return json.loads(merged)
            except (json.JSONDecodeError, TypeError):
                try:
                    return ast.literal_eval(merged)
                except Exception:
                    return merged
        return raw_content
    if not isinstance(raw_content, str):
        return str(raw_content)
    try:
        return json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(raw_content)
    except Exception:
        return raw_content


def _is_forecast_payload(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return any(
        isinstance(v, dict) and "recommendations" in v
        for v in obj.values()
    )


def _forecast_from_tool_calls(tool_calls: list[dict]) -> dict | None:
    """Recover structured forecast when ToolMessage id matching failed."""
    for tc in tool_calls:
        if tc.get("name") != "forecast_demand":
            continue
        out = tc.get("output")
        if _is_forecast_payload(out):
            return out
    return None


def parse_agent_response(
    response_messages: list, input_count: int
) -> tuple[str, list[dict], dict | None]:
    """Extract final text, tool-call details, and forecast data."""
    new_msgs = response_messages[input_count:]

    tool_calls: list[dict] = []
    forecast_data: dict | None = None
    final_content = ""
    tool_messages_seen = 0

    for msg in new_msgs:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc["name"],
                        "inputs": tc["args"],
                        "id": tc.get("id"),
                    })
            if msg.content:
                final_content = msg.content

        elif isinstance(msg, ToolMessage) or hasattr(msg, "tool_call_id"):
            parsed = _parse_tool_content(msg.content)
            tool_messages_seen += 1
            tid = getattr(msg, "tool_call_id", None)

            matched = False
            for tc in tool_calls:
                if tc.get("id") == tid:
                    tc["output"] = parsed
                    matched = True
                    break
            if not matched and tid is None and tool_calls:
                # Some providers omit tool_call_id; align by order within this turn.
                idx = tool_messages_seen - 1
                if 0 <= idx < len(tool_calls) and "output" not in tool_calls[idx]:
                    tool_calls[idx]["output"] = parsed
                    matched = True

            if _is_forecast_payload(parsed):
                forecast_data = parsed

    if forecast_data is None:
        forecast_data = _forecast_from_tool_calls(tool_calls)

    if forecast_data is not None:
        forecast_data = normalize_forecast_data(forecast_data) or forecast_data

    if any(tc.get("name") == "forecast_demand" for tc in tool_calls):
        if forecast_data is None:
            _LOG.warning(
                "forecast_demand ran but structured forecast_data was not recovered "
                "(tool_messages=%s, tool_calls=%s)",
                tool_messages_seen,
                [tc.get("name") for tc in tool_calls],
            )
            for tc in tool_calls:
                if tc.get("name") == "forecast_demand":
                    _LOG.warning(
                        "forecast_demand output type=%s preview=%s",
                        type(tc.get("output")).__name__,
                        repr(tc.get("output"))[:500],
                    )
        else:
            n_main = sum(
                len(v.get("recommendations") or [])
                for v in forecast_data.values()
                if isinstance(v, dict)
            )
            m1_ok = isinstance(forecast_data.get("machine_1"), dict)
            m2_ok = isinstance(forecast_data.get("machine_2"), dict)
            _LOG.info(
                "forecast_data: keys=%s machine_1=%s machine_2=%s line_items=%d",
                list(forecast_data.keys()),
                m1_ok,
                m2_ok,
                n_main,
            )

    return final_content, tool_calls, forecast_data


def _build_lc_messages(
    messages: list[dict], safety_level: str, default_machine: str
) -> list:
    """Convert session-state messages to LangChain message objects."""
    lc: list = []
    for m in messages:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc.append(AIMessage(content=m["content"]))

    if lc and isinstance(lc[-1], HumanMessage):
        hints: list[str] = []
        if safety_level != "normal":
            hints.append(f"safety preference: {safety_level}")
        if default_machine != "Both":
            machine_id = default_machine.lower().replace(" ", "_")
            hints.append(f"default machine: {machine_id}")
        if hints:
            note = " [User settings — " + ", ".join(hints) + "]"
            lc[-1] = HumanMessage(content=lc[-1].content + note)

        fb_comments = get_feedback_comments(limit=5)
        if fb_comments:
            quoted = ", ".join(f'"{c["comment"]}"' for c in fb_comments)
            lc[-1] = HumanMessage(
                content=lc[-1].content + f" [Past user feedback: {quoted}]"
            )

        today = date.today()
        tomorrow = today + timedelta(days=1)
        lc[-1] = HumanMessage(
            content=lc[-1].content
            + (
                f" [Calendar — today is {today.isoformat()}; "
                f"tomorrow is {tomorrow.isoformat()}. "
                "Use forecast_demand target_date as the ISO date of the day being "
                f"planned (e.g. tomorrow = {tomorrow.isoformat()}).]"
            )
        )

    return lc


# ── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    stats = get_quick_stats()

    st.markdown("### Dataset Overview")
    st.markdown(
        f"**Transactions:** {stats['total_transactions']:,}  \n"
        f"**Revenue:** ${stats['total_revenue']:,.2f}  \n"
        f"**Top product:** {stats['top_product']}  \n"
        f"**Period:** {stats['date_range_start']} to {stats['date_range_end']}"
    )

    fb_stats = get_feedback_stats()
    total_rated = fb_stats["thumbs_up"] + fb_stats["thumbs_down"]
    if total_rated > 0:
        st.divider()
        st.markdown("### Copilot Quality")
        st.markdown(
            f"**{fb_stats['thumbs_up']} / {total_rated}** responses rated helpful"
        )

    st.divider()

    st.caption(
        f"Planning calendar: today **{date.today().isoformat()}** "
        f"(forecasts use this for dates & growth trend)"
    )

    st.markdown("### Settings")
    safety_level = st.selectbox(
        "Safety level",
        ["conservative", "normal", "lean"],
        index=1,
        help="How much buffer to add. Conservative = stock more (95th percentile), Lean = stock less (75th percentile).",
    )
    default_machine = st.selectbox(
        "Default machine",
        ["Both", "Machine 1", "Machine 2"],
        index=0,
        help="Which machine to focus on by default. You can always override this in your question.",
    )
    st.checkbox(
        "Show technical tool details (debug)",
        value=False,
        key="copilot_debug_tools",
        help="Shows raw tool JSON under “Why this recommendation”. "
        "Or set COFFEE_COPILOT_DEBUG=1 in the environment.",
    )


# ── Session state ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    history = get_conversation_history()
    st.session_state.messages = []
    for m in history:
        raw_fd = (m["metadata"] or {}).get("forecast_data")
        st.session_state.messages.append({
            "role": m["role"],
            "content": m["content"],
            "message_id": (m["metadata"] or {}).get("message_id", str(uuid.uuid4())),
            "tool_calls": (m["metadata"] or {}).get("tool_calls"),
            "forecast_data": normalize_forecast_data(raw_fd) if raw_fd else None,
        })


# ── Title bar ───────────────────────────────────────────────────────

title_col, btn_col = st.columns([8, 1])
title_col.title("☕ Coffee Ops Copilot")
if btn_col.button("Clear", type="secondary"):
    clear_conversation()
    st.session_state.messages = []
    st.rerun()


# ── Suggested queries ───────────────────────────────────────────────

SUGGESTIONS = [
    "Plan tomorrow morning for both machines",
    "What are the busiest times of day?",
    "Where are we leaving money?",
    "Compare the two machines",
]

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SUGGESTIONS))
    for i, text in enumerate(SUGGESTIONS):
        if cols[i].button(text, key=f"suggest_{i}"):
            st.session_state["_pending_query"] = text
            st.rerun()


# ── Dialogs ──────────────────────────────────────────────────────────


def _plan_final_session_key(message_id: str, machine_id: str) -> str:
    return f"_plan_final_{message_id}_{machine_id}"


@st.dialog("Why are you adjusting?")
def _override_reason_dialog() -> None:
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


@st.dialog("What could be better?")
def _feedback_comment_dialog() -> None:
    ctx = st.session_state["_feedback_ctx"]
    st.markdown("Your feedback helps improve future recommendations.")
    comment = st.text_input(
        "What would you change?",
        placeholder="e.g. forecasts feel too low, too much math detail…",
    )
    c1, c2 = st.columns(2)
    if c1.button("Submit", type="primary", width="stretch"):
        save_feedback(ctx["message_id"], -1, comment=comment or None)
        st.session_state[f"_fb_{ctx['message_id']}"] = -1
        del st.session_state["_feedback_ctx"]
        st.rerun()
    if c2.button("Skip", width="stretch"):
        save_feedback(ctx["message_id"], -1)
        st.session_state[f"_fb_{ctx['message_id']}"] = -1
        del st.session_state["_feedback_ctx"]
        st.rerun()


# ── Render helpers (charts, overrides, feedback) ────────────────────

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
            "Δ": f"{delta:+d}" if delta else "—",
        })
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")


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
                    ("Keep stocked (at least 1)", "keep_stocked"),
                ):
                    tier_recs = by_tier.get(key) or []
                    if not tier_recs:
                        continue
                    names = ", ".join(
                        f"**{r['product']}**: ≥{int(r['recommended_stock'])}"
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
                    help_text = f"Minimum level suggestion: {model_rec}."
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


def render_feedback(message_id: str) -> None:
    fb_key = f"_fb_{message_id}"

    if st.session_state.get(fb_key):
        rating = st.session_state[fb_key]
        st.caption(
            "Thanks for the feedback!" if rating == 1
            else "Noted — we’ll use this to improve phrasing and recommendations."
        )
        return

    st.caption("**Was this helpful?** Your ratings help tune future answers.")
    c1, c2, _ = st.columns([1, 1, 10])
    if c1.button("👍", key=f"up_{message_id}", help="This answer was useful — helps the system learn your preferences"):
        save_feedback(message_id, 1)
        st.session_state[fb_key] = 1
        st.rerun()
    if c2.button("👎", key=f"down_{message_id}", help="This answer missed the mark — tell us what to improve"):
        st.session_state["_feedback_ctx"] = {"message_id": message_id}
        st.rerun()


def render_assistant_extras(msg: dict) -> None:
    """Order: answer text is rendered by the caller first, then this adds
    transparency (why), overrides (sliders), and feedback."""
    tool_calls = msg.get("tool_calls")
    fd = msg.get("forecast_data")
    if fd is not None:
        fd = normalize_forecast_data(fd) or fd

    if tool_calls:
        render_why_this_recommendation(tool_calls, fd if isinstance(fd, dict) else None)

    if fd:
        render_overrides(fd, msg["message_id"])

    render_feedback(msg["message_id"])


# ── Open pending dialogs ─────────────────────────────────────────────

if "_override_ctx" in st.session_state:
    _override_reason_dialog()

if "_feedback_ctx" in st.session_state:
    _feedback_comment_dialog()


# ── Render existing messages ────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg)


# ── Chat input ──────────────────────────────────────────────────────

pending = st.session_state.pop("_pending_query", None)
user_input = st.chat_input("Ask me anything… (e.g. 'Plan tomorrow 7-10am')")
query = pending or user_input

if query:
    user_id = str(uuid.uuid4())
    user_msg = {"role": "user", "content": query, "message_id": user_id}
    st.session_state.messages.append(user_msg)
    save_message("user", query, {"message_id": user_id})

    with st.chat_message("user"):
        st.markdown(query)

    lc_messages = _build_lc_messages(
        st.session_state.messages, safety_level, default_machine,
    )

    agent = get_agent()
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = agent.invoke({"messages": lc_messages})
                content, tool_calls, forecast_data = parse_agent_response(
                    response["messages"], len(lc_messages),
                )
            except Exception as exc:
                content = f"Sorry, something went wrong: {exc}"
                tool_calls = []
                forecast_data = None

        msg_id = str(uuid.uuid4())
        assistant_msg: dict = {
            "role": "assistant",
            "content": content,
            "message_id": msg_id,
            "tool_calls": tool_calls,
            "forecast_data": forecast_data,
        }
        st.markdown(content)
        render_assistant_extras(assistant_msg)

    st.session_state.messages.append(assistant_msg)
    save_message(
        "assistant",
        content,
        json.loads(_safe_json_dumps({
            "message_id": msg_id,
            "tool_calls": tool_calls,
            "forecast_data": forecast_data,
        })),
    )
