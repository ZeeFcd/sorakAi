"""End-to-end tests for ``POST /v1/agent`` + ``/stream`` variants."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sorakai.chains.agent_graph import build_agent_graph, build_tool_registry
from sorakai.rag.app import create_app


def _scripted_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _replace_app_graph(app: Any, llm: FakeListChatModel) -> None:
    """Swap the lifespan-built agent graph for one wired to a scripted LLM.

    The lifespan also builds the real Ollama-backed graph; for tests we
    want full control over LLM responses, so we rebuild the graph using
    the same KB the seeder put in place.
    """
    from sorakai.common.config import get_settings

    settings = get_settings()
    vector_store = app.state.vector_store
    chat_store = app.state.chat_store
    registry = build_tool_registry(settings, vector_store)
    graph, _ = build_agent_graph(settings, vector_store, chat_store, llm=llm, registry=registry)
    app.state.agent_graph = graph
    app.state.agent_tools = registry


def test_agent_endpoint_happy_path(seed_kb: Any) -> None:
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["Pyramids are in Egypt"])
        _replace_app_graph(app, _scripted_llm("kb", "good", "Pyramids are in Egypt.", "ok"))
        r = client.post("/v1/agent", json={"question": "where are the pyramids"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"].startswith("Pyramids")
        assert body["route"] == "kb"
        assert body["trace"] == ["route", "retrieve", "grade", "generate", "critique"]
        assert body["sources_used"] >= 1
        assert any(tc["name"] == "kb_search" for tc in body["tool_calls"])


def test_agent_endpoint_chitchat_works_with_empty_kb() -> None:
    """Agent doesn't 404 on empty KB - it routes to chitchat and answers."""
    app = create_app()
    with TestClient(app) as client:
        _replace_app_graph(app, _scripted_llm("chitchat", "hello there"))
        r = client.post("/v1/agent", json={"question": "hi", "session_id": None})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["route"] == "chitchat"
        assert body["answer"] == "hello there"
        assert body["sources_used"] == 0


def test_agent_endpoint_max_steps_override(seed_kb: Any) -> None:
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["some context"])
        _replace_app_graph(app, _scripted_llm("kb", "good", "answer", "ok"))
        r = client.post(
            "/v1/agent",
            json={"question": "ping", "max_steps": 2, "use_chat_history": False},
        )
        assert r.status_code == 200
        body = r.json()
        # Successful path uses one ``retrieve`` step.
        assert body["steps_used"] >= 1


def test_agent_endpoint_rejects_bad_session_id(seed_kb: Any) -> None:
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["x"])
        r = client.post("/v1/agent", json={"question": "x", "session_id": "invalid/session"})
        assert r.status_code == 400


def test_agent_endpoint_dim_guard_returns_409(seed_kb: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the stored meta disagrees with the live provider/model, we return
    409 without spending any LLM tokens."""
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["x"])
        # Override stored meta's provider so the pre-flight trips.
        from sorakai.common.kb_meta import KBMeta

        async def _bad_read() -> KBMeta:
            return KBMeta(provider="some-other-provider", model="some-other-model", dim=256)

        monkeypatch.setattr(app.state.kb_meta, "read", _bad_read)
        r = client.post("/v1/agent", json={"question": "x"})
        assert r.status_code == 409
        body = r.json()
        assert body["detail"]["error"] == "embedding_metadata_mismatch"


def test_agent_stream_emits_sse_frames(seed_kb: Any) -> None:
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["Pyramids are in Egypt"])
        _replace_app_graph(app, _scripted_llm("kb", "good", "Pyramids are in Egypt.", "ok"))
        with client.stream("POST", "/v1/agent/stream", json={"question": "pyramids"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())

        # First frame is keep-alive comment, then one frame per node, then a done event.
        events = [chunk for chunk in body.split("\n\n") if chunk.strip()]
        assert events[0].startswith(":")  # keep-alive comment
        data_lines = [line for line in body.splitlines() if line.startswith("data:")]
        assert data_lines  # at least one data frame
        assert any('"node"' in line for line in data_lines)
        assert "event: done" in body


def test_query_stream_emits_done_event(seed_kb: Any) -> None:
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["The fox jumps over the lazy dog"])
        with client.stream("POST", "/v1/query/stream", json={"question": "foxes"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        assert "event: done" in body


def test_query_stream_empty_kb_returns_404() -> None:
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/v1/query/stream", json={"question": "x"})
        assert r.status_code == 404


def test_agent_endpoint_response_schema_round_trips(seed_kb: Any) -> None:
    """The AgentResponse pydantic schema must serialise cleanly - the
    handler builds tool_calls from the live ToolCall dataclass."""
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["context one", "context two"])
        _replace_app_graph(app, _scripted_llm("kb", "good", "answer", "ok"))
        r = client.post("/v1/agent", json={"question": "anything"})
        assert r.status_code == 200
        # Round-trip through JSON to assert no non-serialisable values.
        body = json.loads(r.text)
        assert set(body) >= {"answer", "sources_used", "session_id", "route", "steps_used", "trace", "tool_calls"}
        for tc in body["tool_calls"]:
            assert {"name", "input", "output_summary", "duration_ms"} <= set(tc)
