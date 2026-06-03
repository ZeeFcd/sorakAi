"""Server-Sent Events framing used by the Wave 7 streaming endpoints.

We expose two helpers:

- :func:`sse_event` - serialise one ``{event, data, id?}`` payload into a
  valid SSE frame. Multi-line payloads are split into multiple ``data:``
  lines per spec.
- :func:`format_chain_event` / :func:`format_agent_event` - turn the raw
  events emitted by ``chain.astream_events`` / ``graph.astream`` into the
  small, stable JSON we expose to clients (so we don't lock the API to the
  LangChain internal event shape).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

SSE_KEEPALIVE = ": keep-alive\n\n"


def sse_event(data: Mapping[str, Any] | str, *, event: str | None = None, event_id: str | None = None) -> str:
    """Encode one SSE frame.

    ``data`` is JSON-encoded when given a mapping, sent as-is when given
    a string. Newlines inside the body are preserved using multi-``data:``
    lines as required by the spec.
    """
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    if event_id:
        lines.append(f"id: {event_id}")
    for line in body.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def format_chain_event(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Project a LangChain ``astream_events`` event onto our public shape.

    Returns ``None`` for events we deliberately don't surface (e.g.
    ``on_chain_start`` for internal Runnables). Emitting a small, stable
    schema means client code doesn't break when LangChain bumps its
    internal event names.
    """
    name = str(event.get("event", ""))
    data = event.get("data", {}) or {}
    if name == "on_chat_model_stream":
        chunk = data.get("chunk")
        text = getattr(chunk, "content", "") if chunk is not None else ""
        if not text:
            return None
        return {"type": "token", "text": str(text)}
    if name == "on_retriever_end":
        docs = data.get("output") or data.get("documents") or []
        return {"type": "retrieval", "sources_used": len(docs)}
    if name == "on_chain_end" and event.get("name") in {"_execute", "RunnableSequence"}:
        output = data.get("output")
        if isinstance(output, dict) and "answer" in output:
            return {
                "type": "final",
                "answer": str(output.get("answer", "")),
                "sources_used": int(output.get("sources_used", 0) or 0),
            }
    return None


def format_agent_event(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Project a LangGraph ``astream`` event onto our public shape.

    ``astream`` yields ``{node_name: state_delta}`` per node completion.
    We forward node visits + tool calls + the final answer; deltas without
    user-visible content are dropped.
    """
    if not event:
        return None
    if len(event) != 1:
        return None
    node, delta = next(iter(event.items()))
    if not isinstance(delta, dict):
        return None
    payload: dict[str, Any] = {"type": "node", "node": str(node)}
    if delta.get("tool_calls"):
        payload["tool_calls"] = [getattr(tc, "name", str(tc)) for tc in delta["tool_calls"]]
    if delta.get("answer"):
        payload["answer"] = str(delta["answer"])
    if "last_grade" in delta:
        payload["grade"] = str(delta["last_grade"])
    if "last_critique" in delta:
        payload["critique"] = str(delta["last_critique"])
    return payload


__all__ = ["SSE_KEEPALIVE", "format_agent_event", "format_chain_event", "sse_event"]
