"""Pure gateway client helpers for the Streamlit UI.

This module intentionally has no Streamlit import. It is safe to import in
unit tests and in environments that only installed the runtime service
dependencies. The actual Streamlit app lives in :mod:`ui.streamlit_app`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

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
    """Build the JSON body for ``POST /v1/query``."""
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
    """Build the JSON body for ``POST /v1/agent``."""
    payload = build_query_payload(question, session_id)
    if max_steps is not None:
        payload["max_steps"] = max_steps
    return payload


def build_headers(api_key: str | None) -> dict[str, str]:
    """Return the optional bearer-auth headers for gateway requests."""
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
    """Send one question to the gateway and return the parsed answer."""
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


__all__ = [
    "DEFAULT_SESSION_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "ChatAnswer",
    "ChatTarget",
    "ask_gateway",
    "build_agent_payload",
    "build_headers",
    "build_query_payload",
    "parse_agent_response",
    "parse_chain_response",
]
