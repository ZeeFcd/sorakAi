"""Wave 7 SSE framing tests."""

from __future__ import annotations

import json

from sorakai.common.sse import (
    SSE_KEEPALIVE,
    format_agent_event,
    format_chain_event,
    sse_event,
)


def test_sse_event_single_line() -> None:
    out = sse_event({"x": 1})
    assert out == 'data: {"x": 1}\n\n'


def test_sse_event_named_event_and_id() -> None:
    out = sse_event({"x": 1}, event="custom", event_id="42")
    assert out.startswith("event: custom\nid: 42\n")
    assert out.endswith("\n\n")


def test_sse_event_multiline_body_splits_into_data_lines() -> None:
    out = sse_event("line-1\nline-2")
    lines = out.splitlines()
    assert "data: line-1" in lines
    assert "data: line-2" in lines


def test_sse_event_empty_body() -> None:
    out = sse_event("")
    assert "data: " in out


def test_sse_keepalive_constant_is_a_valid_comment_frame() -> None:
    assert SSE_KEEPALIVE.startswith(":")
    assert SSE_KEEPALIVE.endswith("\n\n")


def test_format_chain_event_token_passthrough() -> None:
    class _Chunk:
        content = "hello"

    out = format_chain_event({"event": "on_chat_model_stream", "data": {"chunk": _Chunk()}})
    assert out == {"type": "token", "text": "hello"}


def test_format_chain_event_drops_empty_token() -> None:
    class _Chunk:
        content = ""

    out = format_chain_event({"event": "on_chat_model_stream", "data": {"chunk": _Chunk()}})
    assert out is None


def test_format_chain_event_retrieval_count() -> None:
    out = format_chain_event({"event": "on_retriever_end", "data": {"output": [object(), object(), object()]}})
    assert out == {"type": "retrieval", "sources_used": 3}


def test_format_chain_event_unknown_returns_none() -> None:
    assert format_chain_event({"event": "on_unknown", "data": {}}) is None


def test_format_agent_event_node_only() -> None:
    out = format_agent_event({"route": {}})
    assert out == {"type": "node", "node": "route"}


def test_format_agent_event_with_answer_and_grade() -> None:
    out = format_agent_event({"generate": {"answer": "the answer", "last_grade": "good"}})
    assert out is not None
    assert out["answer"] == "the answer"
    assert out["grade"] == "good"


def test_format_agent_event_with_tool_calls() -> None:
    class _Call:
        name = "kb_search"

    out = format_agent_event({"retrieve": {"tool_calls": [_Call(), _Call()]}})
    assert out is not None
    assert out["tool_calls"] == ["kb_search", "kb_search"]


def test_format_agent_event_rejects_multi_node_payload() -> None:
    # ``astream`` always yields one-node dicts; defensive check.
    assert format_agent_event({"a": {}, "b": {}}) is None
    assert format_agent_event({}) is None


def test_sse_event_roundtrip_through_json() -> None:
    """A consumer should be able to parse ``data:`` lines back into JSON."""
    frame = sse_event({"k": 1, "v": [1, 2, 3]})
    body = "".join(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))
    parsed = json.loads(body)
    assert parsed == {"k": 1, "v": [1, 2, 3]}
