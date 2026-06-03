"""LCEL ``rag_chain`` factory (Wave 6).

The chain **never** imports a concrete provider - it always asks the
factories. Swapping LLM or embeddings providers is one env var:

    LLM_PROVIDER=stub EMBEDDING_PROVIDER=char         # tests
    LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama     # local prod

Wire flow:

    {"question", "session_id"}
        │
        ▼  (RunnableWithMessageHistory injects "history" from chat_store)
    inner: async (question, history) ->
                {"answer": str, "context": str, "sources_used": int}
        │
        ▼  RunnableWithMessageHistory persists the user question + AI answer
    {"answer": str, "context": str, "sources_used": int}

The dict-shaped output is what lets the handler still produce the legacy
``context_preview`` / ``sources_used`` response fields - we don't ask the
LLM to summarise its own input back to us.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from sorakai.chains.history import SorakaiChatMessageHistory
from sorakai.chains.prompts import build_rag_prompt
from sorakai.chains.retriever import (
    HybridRetriever,
    NoopReranker,
    Reranker,
    VectorStoreRetriever,
)
from sorakai.common.chat_history import (
    InMemoryChatHistoryStore,
    RedisChatHistoryStore,
)
from sorakai.common.config import Settings
from sorakai.core.logging import get_logger
from sorakai.infra.embeddings import get_embeddings
from sorakai.infra.llm import get_chat_model
from sorakai.infra.vector_store import VectorStore

logger = get_logger(__name__)


CONTEXT_SEPARATOR = "\n\n---\n\n"
"""How retrieved chunks are joined into the single ``{context}`` slot.

The same separator the Wave 0..5 handler used so existing prompts /
fine-tunes that rely on the marker keep working unchanged."""


def format_docs(docs: list[Any]) -> str:
    """Merge retrieved chunks into the single ``context`` string.

    Empty / whitespace-only chunks are dropped; duplicates by
    ``page_content`` are deduplicated (preserving the first occurrence) so
    the LLM doesn't see the same paragraph twice when BM25 and the vector
    retriever both promote it.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for d in docs:
        text = str(getattr(d, "page_content", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return CONTEXT_SEPARATOR.join(parts)


HistoryFactory = Callable[[str], SorakaiChatMessageHistory]


def _make_history_factory(
    chat_store: RedisChatHistoryStore | InMemoryChatHistoryStore,
) -> HistoryFactory:
    def _factory(session_id: str) -> SorakaiChatMessageHistory:
        return SorakaiChatMessageHistory(chat_store, session_id)

    return _factory


async def _build_retriever(
    settings: Settings,
    vector_store: VectorStore,
    *,
    reranker: Reranker | None,
) -> VectorStoreRetriever | HybridRetriever:
    embeddings = get_embeddings(settings)
    vector_retriever = VectorStoreRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        k=settings.rag_top_k,
    )
    if not settings.hybrid_retriever_enabled:
        return vector_retriever
    # BM25 corpus is snapshotted lazily on first query so seed-then-ask
    # patterns (tests and post-startup ingestions) work without extra wiring.
    return HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store=vector_store,
        bm25_weight=settings.hybrid_bm25_weight,
        vector_weight=settings.hybrid_vector_weight,
        top_k=settings.rag_top_k,
        rerank_top_n=settings.rerank_top_n,
        reranker=reranker if settings.reranker_enabled else None,
    )


def _build_inner_runnable(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever | HybridRetriever,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """The retriever -> prompt -> LLM -> dict step the history wrapper drives."""
    prompt = build_rag_prompt()
    parser = StrOutputParser()

    async def _execute(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", ""))
        history: Sequence[BaseMessage] = inputs.get("history", []) or []
        docs = await retriever.ainvoke(question)
        context = format_docs(docs)
        prompt_value = prompt.invoke(
            {
                "question": question,
                "context": context,
                "history": list(history),
            }
        )
        response = await llm.ainvoke(prompt_value.to_messages())
        return {
            "answer": parser.invoke(response),
            "context": context,
            "sources_used": len(docs),
        }

    return RunnableLambda(_execute)


async def build_rag_chain(
    settings: Settings,
    vector_store: VectorStore,
    chat_store: RedisChatHistoryStore | InMemoryChatHistoryStore,
    *,
    reranker: Reranker | None = None,
) -> tuple[Runnable[dict[str, Any], dict[str, Any]], VectorStoreRetriever | HybridRetriever]:
    """Return ``(chain, retriever)``.

    The chain expects ``{"question": str, "session_id": str | None}`` and
    emits ``{"answer": str, "context": str, "sources_used": int}``. Returning
    the retriever too lets the handler reuse it for things like
    ``/v1/explain`` (Wave 10) without rebuilding the BM25 snapshot.
    """
    llm = get_chat_model(settings)
    retriever = await _build_retriever(
        settings,
        vector_store,
        reranker=reranker or NoopReranker(),
    )
    inner = _build_inner_runnable(llm, retriever)

    with_history = RunnableWithMessageHistory(
        inner,
        _make_history_factory(chat_store),
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )

    chain: Runnable[dict[str, Any], dict[str, Any]] = _SessionAwareChain(with_history)
    logger.info(
        "RAG chain built: llm=%s embeddings=%s vector_store=%s hybrid=%s rerank=%s top_k=%s",
        settings.llm_provider,
        settings.embedding_provider,
        settings.vector_store,
        settings.hybrid_retriever_enabled,
        settings.reranker_enabled,
        settings.rag_top_k,
    )
    return chain, retriever


class _SessionAwareChain(Runnable[dict[str, Any], dict[str, Any]]):
    """Thin facade so handlers can call ``chain.ainvoke(payload)`` without
    juggling the ``RunnableWithMessageHistory`` ``configurable`` block.

    ``payload`` may carry ``session_id``; if it does, the value is moved
    into the runnable config (where ``RunnableWithMessageHistory`` expects
    it) and the inner ``ainvoke`` is dispatched. If absent, a synthetic
    stateless session id is used so the chain still runs (and the in-memory
    write is discarded by callers that don't read the same session back).
    """

    def __init__(self, inner: RunnableWithMessageHistory) -> None:
        self._inner = inner

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = dict(input)
        session_id = str(payload.pop("session_id", "") or "__stateless__")
        merged_config = cast("RunnableConfig", {**(config or {})})
        configurable = dict(merged_config.get("configurable") or {})
        configurable.setdefault("session_id", session_id)
        merged_config["configurable"] = configurable
        result = await self._inner.ainvoke(payload, config=merged_config, **kwargs)
        return cast("dict[str, Any]", result)

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del input, config, kwargs
        raise NotImplementedError("RAG chain is async-only; use ``await chain.ainvoke(payload)``.")


def render_context_preview(context: str, *, max_chars: int = 400) -> str:
    """Replicate the legacy ``/v1/query`` ``context_preview`` shape."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "…"


async def ainvoke_rag(
    chain: Runnable[dict[str, Any], dict[str, Any]],
    *,
    question: str,
    session_id: str | None,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Public helper so handlers don't have to know the payload shape.

    ``callbacks`` is forwarded into the runnable's ``RunnableConfig`` so
    Wave 8's :class:`~sorakai.common.mlflow_callback.MlflowChainCallback`
    (or any other :class:`langchain_core.callbacks.BaseCallbackHandler`)
    can observe LLM + retrieval + tool calls happening inside the chain.
    """
    payload: dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    config: RunnableConfig | None = cast(RunnableConfig, {"callbacks": callbacks}) if callbacks else None
    result: dict[str, Any] = await chain.ainvoke(payload, config=config)
    return result


__all__ = [
    "CONTEXT_SEPARATOR",
    "HistoryFactory",
    "ainvoke_rag",
    "build_rag_chain",
    "format_docs",
    "render_context_preview",
]
