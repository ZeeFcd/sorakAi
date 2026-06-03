from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from sorakai import __version__
from sorakai.chains.agent_graph import ainvoke_agent, build_agent_graph
from sorakai.chains.rag_chain import (
    ainvoke_rag,
    build_rag_chain,
    render_context_preview,
)
from sorakai.chains.tools import ToolCall
from sorakai.common.chat_history import (
    InMemoryChatHistoryStore,
    RedisChatHistoryStore,
    create_chat_store,
    validate_session_id,
)
from sorakai.common.config import get_settings
from sorakai.common.kb_meta import (
    KBMeta,
    KBMetaStore,
    RedisKBMetaStore,
    create_kb_meta_store,
)
from sorakai.common.logging_utils import get_logger
from sorakai.common.middleware import install_common_middleware
from sorakai.common.mlflow_callback import MlflowChainCallback
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import (
    AgentRequest,
    AgentResponse,
    AgentToolCallEntry,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from sorakai.common.sse import (
    SSE_KEEPALIVE,
    format_agent_event,
    format_chain_event,
    sse_event,
)
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store
from sorakai.common.telemetry import (
    configure_tracing,
    instrument_fastapi,
    instrument_httpx,
    span,
)
from sorakai.core.errors import DimensionMismatchError
from sorakai.core.logging import configure_logging
from sorakai.infra.vector_store import VectorStore, get_vector_store
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore
from sorakai.infra.vector_store.qdrant import QdrantVectorStore

logger = get_logger("sorakai.rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, log_format=settings.log_format)
    configure_tracing("sorakai-rag", settings, version=__version__)
    instrument_fastapi(app)
    instrument_httpx()

    store = create_store(settings.redis_url)
    chat_concrete = create_chat_store(
        settings.redis_url,
        ttl_seconds=settings.chat_history_ttl_seconds,
        max_messages=settings.chat_history_max_messages,
    )
    # ``create_chat_store`` returns the abstract base for backwards
    # compatibility, but Wave 6's chain needs to know which concrete
    # backend it's talking to so it can use the right async write path.
    if not isinstance(chat_concrete, RedisChatHistoryStore | InMemoryChatHistoryStore):
        raise TypeError(f"unsupported chat history backend: {type(chat_concrete).__name__}")
    chat: RedisChatHistoryStore | InMemoryChatHistoryStore = chat_concrete
    kb_meta = create_kb_meta_store(settings.redis_url)

    # Wave 6: the chain is what the handler actually calls. The VectorStore
    # is picked via the Wave 5 factory; we reuse the KnowledgeStore the
    # handler is going to keep using for dim-guard reads so memory/redis
    # backends stay consistent. Qdrant gets its own client (independent of
    # the KnowledgeStore) - that's intentional, Wave 7 wires the same chain
    # against the same Qdrant.
    if settings.vector_store in ("memory", "redis"):
        vector_store: VectorStore = KnowledgeStoreVectorStore(store)
    else:
        vector_store = get_vector_store(settings)

    chain, retriever = await build_rag_chain(settings, vector_store, chat)
    agent_graph, agent_tools = build_agent_graph(settings, vector_store, chat)

    app.state.store = store
    app.state.chat_store = chat
    app.state.kb_meta = kb_meta
    app.state.vector_store = vector_store
    app.state.rag_chain = chain
    app.state.retriever = retriever
    app.state.agent_graph = agent_graph
    app.state.agent_tools = agent_tools
    logger.info(
        "RAG service started (redis=%s, llm_provider=%s, embedding_provider=%s, vector_store=%s)",
        bool(settings.redis_url),
        settings.llm_provider,
        settings.embedding_provider,
        settings.vector_store,
    )

    try:
        yield
    finally:
        if isinstance(store, RedisKnowledgeStore):
            await store.aclose()
        if isinstance(chat, RedisChatHistoryStore):
            await chat.aclose()
        if isinstance(kb_meta, RedisKBMetaStore):
            await kb_meta.aclose()
        if isinstance(vector_store, QdrantVectorStore):
            await vector_store.aclose()
        logger.info("RAG service shutdown")


def _mlflow_callbacks(
    settings: Any,
    *,
    run_name: str,
    static_params: dict[str, Any],
) -> list[Any] | None:
    """Build the callback list passed into chain/agent ``RunnableConfig``.

    Returns ``None`` (not ``[]``) when MLflow tracking is disabled so the
    chain skips the ``callbacks`` config branch entirely - no overhead in
    test runs or in deployments without an MLflow server.
    """
    if not settings.mlflow_callback_enabled or not settings.mlflow_tracking_uri:
        return None
    return [
        MlflowChainCallback(
            experiment_name="sorakai-rag",
            run_name=run_name,
            tracking_uri=settings.mlflow_tracking_uri,
            static_params=static_params,
        )
    ]


def _raise_dim_mismatch(exc: DimensionMismatchError) -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "error": "embedding_metadata_mismatch",
            "message": str(exc),
            "expected": {
                "provider": exc.expected_provider,
                "model": exc.expected_model,
                "dim": exc.expected_dim,
            },
            "actual": {
                "provider": exc.actual_provider,
                "model": exc.actual_model,
                "dim": exc.actual_dim,
            },
            "hint": (
                "Re-ingest the corpus with the current provider/model, or POST to /v1/documents with replace_kb=true."
            ),
        },
    ) from exc


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi RAG",
        description="Retrieval and LLM answer generation",
        version=__version__,
        lifespan=lifespan,
    )

    install_common_middleware(app, get_settings(), service="rag")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(service="rag")

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    async def ready(request: Request) -> ReadinessResponse:
        store: KnowledgeStore = request.app.state.store
        ok_store = await store.ping()
        if not ok_store:
            return ReadinessResponse(ready=False, service="rag", detail="store_unreachable")
        return ReadinessResponse(ready=True, service="rag")

    @app.post("/v1/query", response_model=QueryResponse, tags=["rag"])
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        store: KnowledgeStore = request.app.state.store
        kb_meta: KBMetaStore = request.app.state.kb_meta
        chain = request.app.state.rag_chain
        settings = get_settings()

        try:
            sid = validate_session_id(body.session_id) if body.use_chat_history else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        # Wave 6 dim guard: compare stored meta against the live settings'
        # (provider, model). We don't need to run the embedding here just to
        # learn the dim - within a given (provider, model) it's fixed - so the
        # cheap pre-flight catches model swaps before we kick off the chain.
        stored_meta = await kb_meta.read()
        if stored_meta is not None:
            candidate = KBMeta(
                provider=settings.embedding_provider,
                model=settings.ollama_embedding_model,
                dim=stored_meta.dim,  # placeholder; chain will surface real dim mismatches
            )
            if (stored_meta.provider, stored_meta.model) != (candidate.provider, candidate.model):
                _raise_dim_mismatch(
                    DimensionMismatchError(
                        expected_provider=stored_meta.provider,
                        expected_model=stored_meta.model,
                        expected_dim=stored_meta.dim,
                        actual_provider=candidate.provider,
                        actual_model=candidate.model,
                        actual_dim=candidate.dim,
                    )
                )

        # Cheap empty-KB check so we keep the legacy 404 contract.
        summaries = await store.list_documents()
        if not summaries:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents in knowledge base")

        callbacks = _mlflow_callbacks(
            settings,
            run_name="query",
            static_params={"service": "rag", "top_k": body.top_k, "session": bool(sid)},
        )

        with span("rag.query", session=bool(sid), top_k=body.top_k):
            try:
                result: dict[str, Any] = await ainvoke_rag(
                    chain,
                    question=body.question,
                    session_id=sid,
                    callbacks=callbacks,
                )
            except ValueError as e:
                # The vector store / retrieval layer raises ValueError when dims
                # diverge mid-flight (e.g. KB was rewritten between the meta read
                # and the actual search). Translate to the same 409 the dim guard
                # uses so clients have a single error code to handle.
                if "dim" in str(e).lower():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={"error": "embedding_dim_mismatch", "message": str(e)},
                    ) from e
                raise

        context = str(result.get("context", ""))
        sources_used = int(result.get("sources_used", 0))
        if not context:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not retrieve context")

        answer = str(result.get("answer", ""))

        return QueryResponse(
            answer=answer,
            context_preview=render_context_preview(context),
            sources_used=sources_used,
            session_id=sid,
        )

    @app.post("/v1/query/stream", tags=["rag"])
    async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        """SSE streaming variant of ``/v1/query``.

        Emits ``token`` events as the LLM produces output, plus a final
        ``done`` event with ``{answer, sources_used}``. Errors are sent as
        an ``error`` event so clients can render them inline instead of
        losing the connection mid-stream.
        """
        chain = request.app.state.rag_chain
        store: KnowledgeStore = request.app.state.store
        try:
            sid = validate_session_id(body.session_id) if body.use_chat_history else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        if not await store.list_documents():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents in knowledge base")

        payload: dict[str, Any] = {"question": body.question}
        if sid:
            payload["session_id"] = sid

        async def _gen() -> AsyncIterator[bytes]:
            try:
                async for ev in chain.astream_events(payload, version="v2"):
                    projected = format_chain_event(ev)
                    if projected is not None:
                        yield sse_event(projected).encode("utf-8")
                yield sse_event({"type": "done"}, event="done").encode("utf-8")
            except Exception as exc:
                logger.exception("chain stream failed: %s", exc)
                yield sse_event({"type": "error", "message": str(exc)}, event="error").encode("utf-8")

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.post("/v1/agent", response_model=AgentResponse, tags=["agent"])
    async def agent(body: AgentRequest, request: Request) -> AgentResponse:
        """Run the Wave 7 LangGraph agent."""
        graph = request.app.state.agent_graph
        store: KnowledgeStore = request.app.state.store
        kb_meta: KBMetaStore = request.app.state.kb_meta
        cfg = get_settings()
        try:
            sid = validate_session_id(body.session_id) if body.use_chat_history else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        # Same cheap pre-flight as /v1/query so dim mismatches surface
        # before we spend any LLM tokens on routing.
        stored_meta = await kb_meta.read()
        if stored_meta is not None and (stored_meta.provider, stored_meta.model) != (
            cfg.embedding_provider,
            cfg.ollama_embedding_model,
        ):
            _raise_dim_mismatch(
                DimensionMismatchError(
                    expected_provider=stored_meta.provider,
                    expected_model=stored_meta.model,
                    expected_dim=stored_meta.dim,
                    actual_provider=cfg.embedding_provider,
                    actual_model=cfg.ollama_embedding_model,
                    actual_dim=stored_meta.dim,
                )
            )

        # An empty KB doesn't 404 here: the agent can still chitchat (the
        # route node may classify as such) - keeping the endpoint useful for
        # smalltalk and tool-only flows like ``calc``.
        await store.list_documents()

        max_steps = int(body.max_steps or cfg.agent_max_steps)
        callbacks = _mlflow_callbacks(
            cfg,
            run_name="agent",
            static_params={"service": "rag-agent", "max_steps": max_steps, "session": bool(sid)},
        )
        with span("agent.run", session=bool(sid), max_steps=max_steps):
            try:
                result = await ainvoke_agent(
                    graph,
                    question=body.question,
                    session_id=sid,
                    max_steps=max_steps,
                    callbacks=callbacks,
                )
            except ValueError as e:
                if "dim" in str(e).lower():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={"error": "embedding_dim_mismatch", "message": str(e)},
                    ) from e
                raise

        tool_calls: list[ToolCall] = list(result.get("tool_calls", []) or [])
        tool_call_entries = [_summarise_tool_call(tc) for tc in tool_calls]

        return AgentResponse(
            answer=str(result.get("answer", "")),
            sources_used=len(result.get("docs", []) or []),
            session_id=sid,
            route=str(result.get("route", "kb")),
            steps_used=int(result.get("step", 0) or 0),
            trace=list(result.get("trace", []) or []),
            tool_calls=tool_call_entries,
        )

    @app.post("/v1/agent/stream", tags=["agent"])
    async def agent_stream(body: AgentRequest, request: Request) -> StreamingResponse:
        graph = request.app.state.agent_graph
        cfg = get_settings()
        try:
            sid = validate_session_id(body.session_id) if body.use_chat_history else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        max_steps = int(body.max_steps or cfg.agent_max_steps)
        initial: dict[str, Any] = {
            "question": body.question,
            "query": body.question,
            "session_id": sid,
            "history": [],
            "docs": [],
            "context": "",
            "answer": "",
            "route": "kb",
            "step": 0,
            "max_steps": max_steps,
            "trace": [],
            "tool_calls": [],
        }

        async def _gen() -> AsyncIterator[bytes]:
            yield SSE_KEEPALIVE.encode("utf-8")
            try:
                async for chunk in graph.astream(initial):
                    projected = format_agent_event(chunk)
                    if projected is not None:
                        yield sse_event(projected).encode("utf-8")
                yield sse_event({"type": "done"}, event="done").encode("utf-8")
            except Exception as exc:
                logger.exception("agent stream failed: %s", exc)
                yield sse_event({"type": "error", "message": str(exc)}, event="error").encode("utf-8")

        return StreamingResponse(_gen(), media_type="text/event-stream")

    register_bundled_openapi_routes(app, "rag")
    return app


def _summarise_tool_call(call: ToolCall) -> AgentToolCallEntry:
    """Reduce a :class:`ToolCall` to a transport-safe schema row.

    - ``kb_search`` outputs are lists of Documents; we summarise as
      ``"<n> docs"`` (the actual context is already in the answer).
    - Anything else gets stringified through ``str(...)`` truncated to a
      sensible length to keep the response body small.
    """
    output = call.output
    if isinstance(output, list):
        summary = f"{len(output)} item(s)"
    elif output is None:
        summary = ""
    else:
        rendered = str(output)
        summary = rendered if len(rendered) <= 400 else rendered[:400] + "…"
    return AgentToolCallEntry(
        name=call.name,
        input=dict(call.input),
        output_summary=summary,
        duration_ms=float(call.duration_ms),
        error=call.error,
    )


app = create_app()
