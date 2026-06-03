"""Wave 7 agent tool tests.

The tools are deliberately tiny + side-effect-free at import time so
testing them needs no fixtures beyond a seeded in-memory KB - the same
pattern the chain + retriever tests use.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sorakai.chains.retriever import VectorStoreRetriever
from sorakai.chains.tools import (
    CalcTool,
    KBSearchTool,
    ToolError,
    ToolRegistry,
    WebSearchTool,
    run_tool,
    safe_calc,
)
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.infra.embeddings.char import CharPseudoEmbeddings
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore

# ---------- safe_calc ----------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1+2", 3.0),
        ("2*3 + 4", 10.0),
        ("(10 - 2) / 4", 2.0),
        ("3 % 2", 1.0),
        ("2 ** 8", 256.0),
        ("-5 + 3", -2.0),
        ("7 // 2", 3.0),
        ("1.5 + 2.5", 4.0),
    ],
)
def test_safe_calc_evaluates_arithmetic(expr: str, expected: float) -> None:
    assert math.isclose(safe_calc(expr), expected)


@pytest.mark.parametrize(
    "bad_expr",
    [
        "__import__('os').system('ls')",
        "1 if True else 2",
        "[1, 2, 3]",
        "True",
        "'a' + 'b'",
        "len([])",
        "x + 1",
    ],
)
def test_safe_calc_rejects_non_arithmetic(bad_expr: str) -> None:
    with pytest.raises(ToolError):
        safe_calc(bad_expr)


def test_safe_calc_blank_input() -> None:
    with pytest.raises(ToolError):
        safe_calc("   ")


def test_safe_calc_invalid_syntax() -> None:
    with pytest.raises(ToolError):
        safe_calc("1 +")


def test_safe_calc_exponent_cap_blocks_cpu_bomb() -> None:
    with pytest.raises(ToolError, match="exponent"):
        safe_calc("2 ** 1000000")


# ---------- CalcTool ---------------------------------------------------------


@pytest.mark.asyncio
async def test_calc_tool_wraps_safe_calc() -> None:
    out = await CalcTool().ainvoke(expr="3 * 14")
    assert out == 42.0


@pytest.mark.asyncio
async def test_calc_tool_raises_on_bad_input() -> None:
    with pytest.raises(ToolError):
        await CalcTool().ainvoke(expr="open('etc/passwd')")


# ---------- WebSearchTool ----------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_disabled_returns_empty() -> None:
    out = await WebSearchTool(enabled=False).ainvoke(query="latest news")
    assert out == []


@pytest.mark.asyncio
async def test_web_search_enabled_without_provider_raises() -> None:
    """Misconfigured enabled-but-no-provider must surface, not silently no-op."""
    with pytest.raises(ToolError, match="no provider"):
        await WebSearchTool(enabled=True).ainvoke(query="x")


@pytest.mark.asyncio
async def test_web_search_blank_query_rejected() -> None:
    with pytest.raises(ToolError):
        await WebSearchTool(enabled=False).ainvoke(query="   ")


# ---------- KBSearchTool -----------------------------------------------------


async def _make_kb(chunks: list[tuple[str, str]]) -> KnowledgeStoreVectorStore:
    store = InMemoryKnowledgeStore()
    embedder = CharPseudoEmbeddings()
    for doc_id, text in chunks:
        vec = (await embedder.aembed_documents([text]))[0]
        await store.append_document(doc_id, f"{doc_id}.txt", [text], [np.asarray(vec, dtype=float)])
    return KnowledgeStoreVectorStore(store)


@pytest.mark.asyncio
async def test_kb_search_returns_documents() -> None:
    vstore = await _make_kb(
        [
            ("d1", "Pyramids of Giza are in Egypt"),
            ("d2", "Eiffel Tower is in Paris"),
        ]
    )
    retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharPseudoEmbeddings(), k=2)
    docs = await KBSearchTool(retriever=retriever).ainvoke(query="pyramids", k=1)
    assert len(docs) == 1
    assert "Pyramids" in docs[0].page_content or "Eiffel" in docs[0].page_content


@pytest.mark.asyncio
async def test_kb_search_blank_query_rejected() -> None:
    vstore = await _make_kb([("d1", "anything")])
    retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharPseudoEmbeddings(), k=1)
    with pytest.raises(ToolError):
        await KBSearchTool(retriever=retriever).ainvoke(query="")


@pytest.mark.asyncio
async def test_kb_search_restores_retriever_k_after_override() -> None:
    """Overriding ``k`` for one call must not leak into subsequent calls -
    the retriever is shared with the chain and other agent runs."""
    vstore = await _make_kb([("d1", "x"), ("d2", "y")])
    retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharPseudoEmbeddings(), k=5)
    tool = KBSearchTool(retriever=retriever)
    await tool.ainvoke(query="anything", k=1)
    assert retriever.k == 5


# ---------- ToolRegistry / run_tool ------------------------------------------


@pytest.mark.asyncio
async def test_registry_get_and_call() -> None:
    registry = ToolRegistry()
    registry.register(CalcTool())
    call = await run_tool(registry, "calc", expr="2+2")
    assert call.name == "calc"
    assert call.output == 4.0
    assert call.error is None
    assert call.duration_ms >= 0.0
    assert call.input == {"expr": "2+2"}


@pytest.mark.asyncio
async def test_registry_run_captures_tool_error() -> None:
    """Tool errors land in ``ToolCall.error`` instead of propagating; the
    agent graph relies on this so a bad tool call only kills the current
    step, not the whole graph."""
    registry = ToolRegistry()
    registry.register(CalcTool())
    call = await run_tool(registry, "calc", expr="bogus +")
    assert call.output is None
    assert call.error is not None
    assert "calc" in call.error


@pytest.mark.asyncio
async def test_registry_get_unknown_raises_tool_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="unknown tool"):
        registry.get("missing")


def test_registry_register_rejects_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(CalcTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CalcTool())


def test_registry_register_requires_name_and_ainvoke() -> None:
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        registry.register(object())
