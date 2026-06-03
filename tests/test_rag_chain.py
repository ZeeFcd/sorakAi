"""Tests for the Wave 6 LCEL ``rag_chain``.

The chain is built with a :class:`FakeListChatModel` so we never depend on
Ollama (or any cloud) and so we can assert the exact message sequence the
prompt produces - the same Wave 1 contract :file:`test_llm_shim.py` used to
guard, just one layer up.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sorakai.chains.prompts import RAG_SYSTEM_PROMPT
from sorakai.chains.rag_chain import (
    CONTEXT_SEPARATOR,
    ainvoke_rag,
    build_rag_chain,
    format_docs,
    render_context_preview,
)
from sorakai.common.chat_history import InMemoryChatHistoryStore
from sorakai.common.config import get_settings
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.infra.embeddings.char import CharPseudoEmbeddings as CharEmbeddingsAdapter
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore


class _RecordingFakeChatModel(FakeListChatModel):
    # Intentionally a class attribute so test assertions can read the last
    # captured message list from the type, not the instance (the chain owns
    # the instance and we never have a reference back from the test).
    last_messages: ClassVar[list[BaseMessage]] = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        type(self).last_messages = list(messages)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _acall(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        type(self).last_messages = list(messages)
        return await super()._acall(messages, stop=stop, run_manager=run_manager, **kwargs)


async def _seed_store(texts: list[tuple[str, str]]) -> KnowledgeStoreVectorStore:
    store = InMemoryKnowledgeStore()
    embedder = CharEmbeddingsAdapter()
    for doc_id, text in texts:
        vec = (await embedder.aembed_documents([text]))[0]
        await store.append_document(doc_id, f"{doc_id}.txt", [text], [np.asarray(vec, dtype=float)])
    return KnowledgeStoreVectorStore(store)


# ---------- format_docs --------------------------------------------------------


def test_format_docs_joins_with_separator() -> None:
    from langchain_core.documents import Document

    out = format_docs([Document(page_content="A"), Document(page_content="B")])
    assert out == f"A{CONTEXT_SEPARATOR}B"


def test_format_docs_dedupes_and_drops_empty() -> None:
    from langchain_core.documents import Document

    docs = [
        Document(page_content="A"),
        Document(page_content="  "),
        Document(page_content="A"),
        Document(page_content="B"),
    ]
    out = format_docs(docs)
    assert out == f"A{CONTEXT_SEPARATOR}B"


def test_render_context_preview_truncates() -> None:
    assert render_context_preview("x" * 10, max_chars=20) == "x" * 10
    long = "y" * 500
    out = render_context_preview(long, max_chars=400)
    assert out.endswith("…")
    assert len(out) == 401


# ---------- chain end-to-end ---------------------------------------------------


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingFakeChatModel]:
    """Pin the LLM factory to a single-response recorder for assertions."""
    _RecordingFakeChatModel.last_messages = []
    instance = _RecordingFakeChatModel(responses=["chain-answer"])
    monkeypatch.setattr("sorakai.chains.rag_chain.get_chat_model", lambda _s: instance)
    return _RecordingFakeChatModel


@pytest.mark.asyncio
async def test_chain_returns_dict_with_answer_context_sources(
    fake_chat_model: type[_RecordingFakeChatModel],
) -> None:
    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids are in Egypt"), ("d2", "Skis are used in winter sports")])
    chat = InMemoryChatHistoryStore()
    chain, _ = await build_rag_chain(settings, vstore, chat)

    out = await ainvoke_rag(chain, question="where are the pyramids", session_id=None)
    assert out["answer"] == "chain-answer"
    assert "Pyramids" in out["context"] or "Skis" in out["context"]
    assert out["sources_used"] >= 1
    # System prompt + user turn (no history, stateless session).
    msgs = fake_chat_model.last_messages
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == RAG_SYSTEM_PROMPT
    assert isinstance(msgs[-1], HumanMessage)
    assert "where are the pyramids" in str(msgs[-1].content)
    # The user template shape is preserved (so the LLM sees a stable
    # contract regardless of which retriever produced the context).
    user_content = str(msgs[-1].content)
    assert user_content.startswith("Knowledge base context:\n")
    assert "\n\nQuestion:\n" in user_content


@pytest.mark.asyncio
async def test_chain_writes_session_history(
    fake_chat_model: type[_RecordingFakeChatModel],
) -> None:
    del fake_chat_model
    settings = get_settings()
    vstore = await _seed_store([("d1", "any context")])
    chat = InMemoryChatHistoryStore()
    chain, _ = await build_rag_chain(settings, vstore, chat)

    await ainvoke_rag(chain, question="first", session_id="user-A")
    msgs = await chat.get_messages("user-A")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "first"
    assert msgs[1]["content"] == "chain-answer"


@pytest.mark.asyncio
async def test_chain_replays_history_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _RecordingFakeChatModel(responses=["second-answer"])
    _RecordingFakeChatModel.last_messages = []
    monkeypatch.setattr("sorakai.chains.rag_chain.get_chat_model", lambda _s: instance)

    settings = get_settings()
    vstore = await _seed_store([("d1", "any context")])
    chat = InMemoryChatHistoryStore()
    await chat.append_pair("user-B", "earlier-q", "earlier-a")

    chain, _ = await build_rag_chain(settings, vstore, chat)
    await ainvoke_rag(chain, question="follow-up", session_id="user-B")

    msgs = _RecordingFakeChatModel.last_messages
    roles = [type(m).__name__ for m in msgs]
    assert roles[0] == "SystemMessage"
    # The MessagesPlaceholder injects the prior turns between system + user.
    assert "HumanMessage" in roles
    assert "AIMessage" in roles
    # The new user turn is always last.
    assert isinstance(msgs[-1], HumanMessage)
    assert "follow-up" in str(msgs[-1].content)


@pytest.mark.asyncio
async def test_chain_sync_invoke_raises(fake_chat_model: type[_RecordingFakeChatModel]) -> None:
    del fake_chat_model
    settings = get_settings()
    vstore = await _seed_store([("d1", "x")])
    chain, _ = await build_rag_chain(settings, vstore, InMemoryChatHistoryStore())
    with pytest.raises(NotImplementedError):
        chain.invoke({"question": "x"})


@pytest.mark.asyncio
async def test_chain_stateless_when_no_session_id(
    fake_chat_model: type[_RecordingFakeChatModel],
) -> None:
    del fake_chat_model
    settings = get_settings()
    vstore = await _seed_store([("d1", "the context")])
    chat = InMemoryChatHistoryStore()
    chain, _ = await build_rag_chain(settings, vstore, chat)

    await ainvoke_rag(chain, question="hi", session_id=None)
    # The stateless session id we use internally must not leak as a session.
    assert await chat.get_messages("user-X") == []
    # But the internal session does exist (so RunnableWithMessageHistory's
    # output_messages_key persistence has somewhere to land). We don't
    # promise a specific name; only that it isn't an empty string.
    sessions = chat._sessions
    assert any(sid for sid in sessions if sid)


@pytest.mark.asyncio
async def test_chain_with_hybrid_retriever_enabled(
    monkeypatch: pytest.MonkeyPatch,
    fake_chat_model: type[_RecordingFakeChatModel],
) -> None:
    del fake_chat_model
    monkeypatch.setenv("HYBRID_RETRIEVER_ENABLED", "true")
    get_settings.cache_clear()
    settings = get_settings()
    vstore = await _seed_store(
        [
            ("d1", "Polar bears live in the Arctic"),
            ("d2", "Penguins live in Antarctica"),
            ("d3", "Pyramids are in Egypt"),
        ]
    )
    chain, retriever = await build_rag_chain(settings, vstore, InMemoryChatHistoryStore())

    from sorakai.chains.retriever import HybridRetriever

    assert isinstance(retriever, HybridRetriever)
    out = await ainvoke_rag(chain, question="polar bears", session_id=None)
    assert out["sources_used"] >= 1
    assert retriever.initialized is True


def test_chain_persistence_survives_event_loop_isolation() -> None:
    """End-to-end smoke test driven from a sync test (mirrors handler invocation)."""

    async def run() -> dict[str, object]:
        instance = _RecordingFakeChatModel(responses=["one"])
        # We can't monkeypatch from sync; emulate via direct chain construction.
        from langchain_core.runnables.history import RunnableWithMessageHistory

        from sorakai.chains.rag_chain import _build_inner_runnable, _make_history_factory, _SessionAwareChain
        from sorakai.chains.retriever import VectorStoreRetriever

        vstore = await _seed_store([("d1", "blue whales")])
        chat = InMemoryChatHistoryStore()
        retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=1)
        inner = _build_inner_runnable(instance, retriever)
        with_history = RunnableWithMessageHistory(
            inner,
            _make_history_factory(chat),
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="answer",
        )
        chain = _SessionAwareChain(with_history)
        return await chain.ainvoke({"question": "whales", "session_id": "smoke"})

    out = asyncio.run(run())
    assert out["answer"] == "one"
    assert out["sources_used"] >= 1
