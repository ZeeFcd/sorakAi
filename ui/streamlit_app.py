"""Streamlit chat UI for the sorakAi gateway (Wave 10).

Run with::

    pip install -r requirements-ui.txt
    streamlit run ui/streamlit_app.py

The app is a thin wrapper around the gateway's ``/v1/query`` and
``/v1/agent`` endpoints; it never imports anything from
``sorakai.chains.*`` directly so the UI can be deployed independently of
the services it talks to.

To keep the module unit-testable without a Streamlit runtime, the pure
helpers (request building, response parsing, header builders) live at
module scope and are exercised by ``tests/test_ui_client.py``. The
Streamlit-specific painting logic lives in :func:`main` (which the
package's ``__init__`` deliberately does NOT import).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

DEFAULT_GATEWAY_URL = os.environ.get("UI_GATEWAY_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_SESSION_ID = "ui-default"
"""Stable session id so refreshing the page keeps multi-turn history."""

ChatTarget = Literal["chain", "agent"]


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    """Flat shape both the chain and agent paths reduce to."""

    answer: str
    sources_used: int
    session_id: str | None = None
    route: str | None = None
    steps_used: int | None = None
    trace: tuple[str, ...] = ()


def build_query_payload(question: str, session_id: str | None) -> dict[str, Any]:
    """Build the JSON body for ``POST /v1/query``.

    Empty / whitespace ``session_id`` is dropped so the gateway treats
    the call as stateless and skips Redis writes.
    """
    payload: dict[str, Any] = {"question": question}
    sid = (session_id or "").strip()
    if sid:
        payload["session_id"] = sid
        payload["use_chat_history"] = True
    return payload


def build_agent_payload(
    question: str,
    session_id: str | None,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /v1/agent``.

    ``max_steps`` is left unset when ``None`` so the RAG service falls
    back to its server-side default (``settings.agent_max_steps``).
    """
    payload = build_query_payload(question, session_id)
    if max_steps is not None:
        payload["max_steps"] = max_steps
    return payload


def build_headers(api_key: str | None) -> dict[str, str]:
    """Bearer header builder.

    Returns an empty dict when ``api_key`` is unset so the call works
    against a gateway that left ``GATEWAY_API_KEY`` unconfigured (the
    default local dev posture).
    """
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def parse_chain_response(payload: Mapping[str, Any]) -> ChatAnswer:
    """Coerce a ``/v1/query`` JSON body into the :class:`ChatAnswer` view."""
    return ChatAnswer(
        answer=str(payload.get("answer") or ""),
        sources_used=int(payload.get("sources_used") or 0),
        session_id=_optional_str(payload.get("session_id")),
    )


def parse_agent_response(payload: Mapping[str, Any]) -> ChatAnswer:
    """Coerce a ``/v1/agent`` JSON body into the :class:`ChatAnswer` view."""
    raw_steps = payload.get("steps_used")
    return ChatAnswer(
        answer=str(payload.get("answer") or ""),
        sources_used=int(payload.get("sources_used") or 0),
        session_id=_optional_str(payload.get("session_id")),
        route=_optional_str(payload.get("route")),
        steps_used=int(raw_steps) if raw_steps is not None else None,
        trace=tuple(str(x) for x in (payload.get("trace") or [])),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def ask_gateway(
    *,
    gateway_url: str,
    target: ChatTarget,
    question: str,
    session_id: str | None,
    api_key: str | None,
    max_steps: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> ChatAnswer:
    """Send one question to the gateway and return the parsed answer.

    ``client`` is injected by the tests; production callers leave it
    ``None`` and we open a per-call ``httpx.Client`` (good enough for
    the latency profile of one user typing into Streamlit). The
    gateway's HTTP error payloads are surfaced verbatim so the user can
    see ``401 Unauthorized`` etc. inline.
    """
    base = gateway_url.rstrip("/")
    if target == "chain":
        path = "/v1/query"
        body = build_query_payload(question, session_id)
    else:
        path = "/v1/agent"
        body = build_agent_payload(question, session_id, max_steps=max_steps)
    headers = build_headers(api_key)

    if client is None:
        with httpx.Client(timeout=timeout_seconds) as owned:
            response = owned.post(base + path, json=body, headers=headers)
    else:
        response = client.post(base + path, json=body, headers=headers)
    response.raise_for_status()
    payload = cast(Mapping[str, Any], response.json())
    if target == "chain":
        return parse_chain_response(payload)
    return parse_agent_response(payload)


# ---------------------------------------------------------------------------
# Streamlit runtime entry point (only invoked by ``streamlit run``)
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - exercised manually via `streamlit run`
    """Paint the chat UI. Imports ``streamlit`` lazily to keep the
    module importable in CI (where the dep is intentionally absent)."""
    import streamlit as st  # noqa: PLC0415  - lazy import keeps the test
    # suite from requiring the streamlit extra to merely import this
    # module's helpers.

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
    "DEFAULT_SESSION_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "ChatAnswer",
    "ChatTarget",
    "ask_gateway",
    "build_agent_payload",
    "build_headers",
    "build_query_payload",
    "main",
    "parse_agent_response",
    "parse_chain_response",
]
