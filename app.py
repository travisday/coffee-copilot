"""Streamlit frontend for Coffee Ops Copilot."""

from __future__ import annotations

import ast
import json
import logging
import uuid
from datetime import date, timedelta

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_agent
from agent.tools import forecaster
from ui.feedback import feedback_comment_dialog, render_feedback
from ui.forecast_explain import (
    normalize_forecast_data,
    render_why_this_recommendation,
)
from ui.overrides import override_reason_dialog, render_overrides
from db import (
    clear_conversation,
    get_conversation_history,
    get_feedback_comments,
    get_feedback_stats,
    get_recent_overrides,
    init_db,
    save_message,
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
    df = forecaster.raw_data
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
        user_text = lc[-1].content
        sections: list[str] = []

        # -- settings
        settings_parts: list[str] = [f"safety: {safety_level}"]
        if default_machine != "Both":
            machine_id = default_machine.lower().replace(" ", "_")
            settings_parts.append(f"machine: {machine_id}")
        else:
            settings_parts.append("machine: both")
        sections.append("# SETTINGS\n" + ", ".join(settings_parts))

        # -- previous feedback
        fb_comments = get_feedback_comments(limit=5)
        if fb_comments:
            quoted = ", ".join(f'"{c["comment"]}"' for c in fb_comments)
            sections.append("# PREVIOUS FEEDBACK\n" + quoted)

        # -- past stocking overrides
        recent_overrides = get_recent_overrides(limit=20)
        if recent_overrides:
            lines: list[str] = []
            for ov in recent_overrides:
                machine = (ov["machine_id"] or "").replace("_", " ").title()
                delta = ov["adjusted_rec"] - ov["original_rec"]
                direction = f"+{delta}" if delta > 0 else str(delta)
                entry = f"- {machine} {ov['product']}: {ov['original_rec']}→{ov['adjusted_rec']} ({direction})"
                if ov.get("reason"):
                    entry += f" reason: {ov['reason']}"
                lines.append(entry)
            sections.append("# PAST STOCKING OVERRIDES\n" + "\n".join(lines))

        # -- calendar
        today = date.today()
        tomorrow = today + timedelta(days=1)
        sections.append(
            "# CALENDAR\n"
            f"Today: {today.isoformat()}, Tomorrow: {tomorrow.isoformat()}\n"
            "Use forecast_demand target_date as ISO date for the day being planned."
        )

        injected = (
            "\n---\n"
            "# USER MESSAGE\n"
            f"{user_text}\n\n"
            + "\n\n".join(sections)
            + "\n---"
        )
        lc[-1] = HumanMessage(content=injected)

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
    ("📊", "Plan all of tomorrow"),
    ("⏰", "What are the busiest hours?"),
    ("💰", "Where are we leaving money?"),
    ("🔍", "How accurate is the model?"),
]

st.markdown("**Quick questions:**")
cols = st.columns(len(SUGGESTIONS))
for i, (icon, text) in enumerate(SUGGESTIONS):
    if cols[i].button(f"{icon}  {text}", key=f"suggest_{i}", use_container_width=True):
        st.session_state["_pending_query"] = text
        st.rerun()


# ── Render helper (thin orchestrator) ────────────────────────────────


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
    override_reason_dialog()

if "_feedback_ctx" in st.session_state:
    feedback_comment_dialog()


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
