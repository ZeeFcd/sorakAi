"""Wave 10 Streamlit UI helper tests.

The Streamlit runtime is intentionally NOT a dev/test dependency (it's
under the optional ``requirements-ui.txt`` extra). The pure helpers in
:mod:`ui.streamlit_app` are import-safe without it, and this file pins
their behaviour so the gateway contract stays in sync with the UI.
"""

from __future__ import annotations

import httpx
import pytest

from ui.streamlit_app import (
    ChatAnswer,
    ask_gateway,
    build_agent_payload,
    build_headers,
    build_query_payload,
    parse_agent_response,
    parse_chain_response,
)

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def test_build_query_payload_includes_session_when_set() -> None:
    body = build_query_payload("hi", "sess-1")
    assert body == {"question": "hi", "session_id": "sess-1", "use_chat_history": True}


def test_build_query_payload_drops_blank_session() -> None:
    body = build_query_payload("hi", "   ")
    assert body == {"question": "hi"}


def test_build_agent_payload_inherits_query_shape_plus_max_steps() -> None:
    body = build_agent_payload("hi", "sess-1", max_steps=7)
    assert body == {
        "question": "hi",
        "session_id": "sess-1",
        "use_chat_history": True,
        "max_steps": 7,
    }


def test_build_agent_payload_skips_max_steps_when_none() -> None:
    body = build_agent_payload("hi", None)
    assert "max_steps" not in body


def test_build_headers_empty_when_no_key() -> None:
    assert build_headers(None) == {}
    assert build_headers("") == {}
    assert build_headers("   ") == {}


def test_build_headers_sets_bearer_when_key_present() -> None:
    assert build_headers("hunter2") == {"Authorization": "Bearer hunter2"}


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def test_parse_chain_response_extracts_core_fields() -> None:
    parsed = parse_chain_response({"answer": "42", "sources_used": 3, "session_id": "s-1", "context_preview": "..."})
    assert parsed.answer == "42"
    assert parsed.sources_used == 3
    assert parsed.session_id == "s-1"
    assert parsed.route is None


def test_parse_agent_response_extracts_route_and_trace() -> None:
    parsed = parse_agent_response(
        {
            "answer": "Mars has Phobos.",
            "sources_used": 2,
            "session_id": None,
            "route": "kb",
            "steps_used": 1,
            "trace": ["route", "retrieve", "grade", "generate", "critique"],
        }
    )
    assert parsed.answer.startswith("Mars")
    assert parsed.route == "kb"
    assert parsed.steps_used == 1
    assert parsed.trace == ("route", "retrieve", "grade", "generate", "critique")
    assert parsed.session_id is None


def test_chat_answer_handles_missing_optional_fields() -> None:
    parsed = parse_chain_response({})
    assert parsed == ChatAnswer(answer="", sources_used=0)


# ---------------------------------------------------------------------------
# ask_gateway end-to-end
# ---------------------------------------------------------------------------


def _stub_client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_ask_gateway_posts_to_chain_path_and_returns_parsed_answer() -> None:
    seen_url: list[str] = []
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_url.append(str(request.url))
        seen_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={"answer": "ok", "sources_used": 1, "session_id": "s"},
        )

    with _stub_client(handler) as client:
        answer = ask_gateway(
            gateway_url="http://gw.test",
            target="chain",
            question="ping",
            session_id="s",
            api_key="hunter2",
            client=client,
        )

    assert answer.answer == "ok"
    assert seen_url == ["http://gw.test/v1/query"]
    assert seen_headers[0]["authorization"] == "Bearer hunter2"


def test_ask_gateway_routes_agent_target_to_agent_endpoint() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "answer": "via-agent",
                "sources_used": 0,
                "route": "chitchat",
                "steps_used": 0,
                "trace": [],
            },
        )

    with _stub_client(handler) as client:
        answer = ask_gateway(
            gateway_url="http://gw.test/",
            target="agent",
            question="hi",
            session_id=None,
            api_key=None,
            max_steps=2,
            client=client,
        )

    assert seen_urls == ["http://gw.test/v1/agent"]
    assert answer.route == "chitchat"


def test_ask_gateway_raises_on_4xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Missing bearer credentials"})

    with _stub_client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        ask_gateway(
            gateway_url="http://gw.test",
            target="chain",
            question="?",
            session_id=None,
            api_key=None,
            client=client,
        )
