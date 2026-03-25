"""Feedback buttons and comment dialog for assistant messages."""

from __future__ import annotations

import streamlit as st

from db import save_feedback


# ── Dialog ───────────────────────────────────────────────────────────


@st.dialog("What could be better?")
def feedback_comment_dialog() -> None:
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


# ── Main render function ─────────────────────────────────────────────


def render_feedback(message_id: str) -> None:
    fb_key = f"_fb_{message_id}"

    if st.session_state.get(fb_key):
        rating = st.session_state[fb_key]
        st.caption(
            "Thanks for the feedback!" if rating == 1
            else "Noted — we'll use this to improve phrasing and recommendations."
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
