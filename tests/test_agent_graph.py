"""Wave 7 LangGraph agent tests.

Drives the graph end-to-end with :class:`FakeListChatModel` so we can
script the exact sequence of route/grade/critique decisions, and asserts
on the resulting node trace + tool call ledger.
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sorakai.chains.agent_graph import ainvoke_agent, build_agent_graph, build_tool_registry
from sorakai.common.chat_history import InMemoryChatHistoryStore
from sorakai.common.config import get_settings
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.infra.embeddings.char import CharPseudoEmbeddings
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore


async def _seed_store(chunks: list[tuple[str, str]]) -> KnowledgeStoreVectorStore:
    store = InMemoryKnowledgeStore()
    embedder = CharPseudoEmbeddings()
    for doc_id, text in chunks:
        vec = (await embedder.aembed_documents([text]))[0]
        await store.append_document(doc_id, f"{doc_id}.txt", [text], [np.asarray(vec, dtype=float)])
    return KnowledgeStoreVectorStore(store)


# Each scripted response below corresponds to one LLM ``ainvoke`` from a node.
# The order is: route -> (retrieve [no llm]) -> grade -> [rewrite -> retrieve -> grade]?
# -> generate -> critique -> (loop?) -> END.


@pytest.mark.asyncio
async def test_happy_kb_path_route_retrieve_grade_generate_critique() -> None:
    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids of Giza are in Egypt"), ("d2", "Eiffel Tower is in Paris")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "kb",  # route
            "good",  # grade
            "Pyramids are in Egypt.",  # generate
            "ok",  # critique
        ]
    )
    graph, registry = build_agent_graph(settings, vstore, chat, llm=llm)

    result = await ainvoke_agent(graph, question="where are the pyramids", session_id=None, max_steps=4)

    assert result["trace"] == ["route", "retrieve", "grade", "generate", "critique"]
    assert result["route"] == "kb"
    assert result["answer"].startswith("Pyramids")
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "kb_search"
    assert "kb_search" in registry.names()


@pytest.mark.asyncio
async def test_chitchat_route_skips_retrieval() -> None:
    settings = get_settings()
    vstore = await _seed_store([("d1", "ignored context")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "chitchat",  # route
            "Hi there!",  # generate
            # critique is short-circuited for chitchat (no LLM call)
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)

    result = await ainvoke_agent(graph, question="hi", session_id=None, max_steps=4)

    assert result["trace"] == ["route", "generate", "critique"]
    assert result["answer"] == "Hi there!"
    assert result["tool_calls"] == []
    assert result["route"] == "chitchat"


@pytest.mark.asyncio
async def test_weak_grade_triggers_rewrite_then_retrieve_again() -> None:
    """The classic weak->rewrite->retrieve loop. After one retry the grader
    accepts the new context and we head to generate + critique."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids are in Egypt"), ("d2", "Skis are for snow")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "kb",  # route
            "weak",  # grade (round 1) -> rewrite
            "Egypt pyramids history",  # rewrite output (used as new query)
            "good",  # grade (round 2) -> generate
            "Pyramids are in Egypt.",  # generate
            "ok",  # critique
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)

    result = await ainvoke_agent(graph, question="pyramids?", session_id=None, max_steps=4)

    assert result["trace"] == [
        "route",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "critique",
    ]
    assert len([tc for tc in result["tool_calls"] if tc.name == "kb_search"]) == 2


@pytest.mark.asyncio
async def test_critique_retry_loops_back_to_rewrite() -> None:
    """The first generated answer is graded ``retry`` so we rewrite +
    re-retrieve + re-generate before the second critique accepts."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids are in Egypt")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "kb",  # route
            "good",  # grade round 1
            "off-topic stuff",  # generate round 1
            "retry",  # critique round 1 -> rewrite
            "better query",  # rewrite
            "good",  # grade round 2
            "Pyramids are in Egypt.",  # generate round 2
            "ok",  # critique round 2
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)
    result = await ainvoke_agent(graph, question="pyramids?", session_id=None, max_steps=6)

    assert result["trace"][0] == "route"
    assert result["trace"][-1] == "critique"
    assert result["answer"].startswith("Pyramids")


@pytest.mark.asyncio
async def test_max_steps_short_circuits_grade_loop() -> None:
    """If the grader keeps returning ``weak`` we still produce something
    once we burn through the step budget - the graph never hangs."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "irrelevant context")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "kb",  # route
            "weak",  # grade 1
            "rewrite1",  # rewrite 1
            "weak",  # grade 2
            "rewrite2",  # rewrite 2
            # After 2 retrieves we're at step=2; with max_steps=2 the grade
            # branch flips to ``good`` regardless of LLM output.
            "Best-effort answer.",  # generate
            "ok",  # critique
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)
    result = await ainvoke_agent(graph, question="anything", session_id=None, max_steps=2)
    assert result["answer"]
    assert result["step"] >= 2


@pytest.mark.asyncio
async def test_invalid_llm_label_falls_back_to_default() -> None:
    """A classifier that returns junk must not stall the graph - the
    fallback route is ``kb`` so we still answer."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "anything")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "I AM A POEM",  # route -> defaults to kb
            "good",  # grade
            "Some answer.",  # generate
            "ok",  # critique
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)
    result = await ainvoke_agent(graph, question="x", session_id=None, max_steps=3)
    assert result["route"] == "kb"
    assert result["answer"] == "Some answer."


@pytest.mark.asyncio
async def test_session_history_persists_after_agent_run() -> None:
    settings = get_settings()
    vstore = await _seed_store([("d1", "x")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(
        responses=[
            "chitchat",  # route (skip retrieval, no critique LLM call)
            "Echo.",  # generate
        ]
    )
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)
    await ainvoke_agent(graph, question="hello", session_id="user-7", max_steps=3)

    msgs = await chat.get_messages("user-7")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "Echo."


@pytest.mark.asyncio
async def test_build_tool_registry_wires_all_three_tools() -> None:
    settings = get_settings()
    vstore = await _seed_store([("d1", "x")])
    registry = build_tool_registry(settings, vstore)
    assert sorted(registry.names()) == ["calc", "kb_search", "web_search"]
