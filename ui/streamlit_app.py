"""Streamlit chat UI for the sorakAi gateway.

Run with::

    pip install -r requirements-ui.txt
    streamlit run ui/streamlit_app.py

The app is a thin wrapper around the gateway's ``/v1/query`` and
``/v1/agent`` endpoints; it never imports anything from
``sorakai.chains.*`` directly so the UI can be deployed independently of
the services it talks to.

Pure request/response helpers live in :mod:`ui.client`. This module is
the actual Streamlit entrypoint, so importing Streamlit at module scope is
expected and keeps project-wide imports top-level.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import httpx
import streamlit as st

from ui.client import (
    DEFAULT_SESSION_ID,
    ChatTarget,
    ask_gateway,
)

DEFAULT_GATEWAY_URL = os.environ.get("UI_GATEWAY_URL", "http://127.0.0.1:8000")


def main() -> None:  # pragma: no cover - exercised manually via `streamlit run`
    """Paint the chat UI."""
    st.set_page_config(page_title="sorakAi chat", page_icon=":speech_balloon:")
    st.title("sorakAi chat")

    with st.sidebar:
        st.markdown("### Gateway")
        gateway_url = st.text_input("URL", value=DEFAULT_GATEWAY_URL)
        api_key = st.text_input(
            "API key (Bearer)",
            type="password",
            help="Optional - leave blank when the gateway has no GATEWAY_API_KEY set.",
        )
        target = st.radio(
            "Mode",
            options=("chain", "agent"),
            index=0,
            help="`chain` is the fast LCEL path; `agent` runs the LangGraph route/grade/critique loop.",
        )
        session_id = st.text_input("Session ID", value=DEFAULT_SESSION_ID)
        max_steps = None
        if target == "agent":
            max_steps = st.slider("Max agent steps", min_value=1, max_value=10, value=4)

    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            meta = entry.get("meta") or {}
            if meta:
                st.caption(_format_meta(meta))

    if prompt := st.chat_input("Ask about the knowledge base..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_thinking..._")
            try:
                answer = ask_gateway(
                    gateway_url=gateway_url,
                    target=cast(ChatTarget, target),
                    question=prompt,
                    session_id=session_id,
                    api_key=api_key,
                    max_steps=max_steps,
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text or str(exc)
                placeholder.error(f"Gateway {exc.response.status_code}: {detail}")
                st.session_state.history.append({"role": "assistant", "content": f"_error: {detail}_"})
            except httpx.RequestError as exc:
                placeholder.error(f"Gateway unreachable: {exc}")
                st.session_state.history.append({"role": "assistant", "content": f"_unreachable: {exc}_"})
            else:
                placeholder.markdown(answer.answer or "_(empty response)_")
                meta = {
                    "sources_used": answer.sources_used,
                    "route": answer.route,
                    "steps_used": answer.steps_used,
                    "trace": " -> ".join(answer.trace) if answer.trace else None,
                }
                cleaned = {k: v for k, v in meta.items() if v not in (None, "", 0)}
                if cleaned:
                    st.caption(_format_meta(cleaned))
                st.session_state.history.append({"role": "assistant", "content": answer.answer, "meta": cleaned})


def _format_meta(meta: Mapping[str, Any]) -> str:
    """Render an answer's meta footer as a single inline string."""
    return " | ".join(f"{k}: {v}" for k, v in meta.items())


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "DEFAULT_GATEWAY_URL",
    "main",
]
