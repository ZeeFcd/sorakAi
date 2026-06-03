from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sorakai import __version__
from sorakai.chains.rag_chain import (
    ainvoke_rag,
    build_rag_chain,
    render_context_preview,
)
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
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.mlflow_tracking import log_params_metrics, mlflow_run
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import HealthResponse, QueryRequest, QueryResponse, ReadinessResponse
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store
from sorakai.core.errors import DimensionMismatchError
from sorakai.core.logging import configure_logging
from sorakai.infra.vector_store import VectorStore, get_vector_store
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore
from sorakai.infra.vector_store.qdrant import QdrantVectorStore

logger = get_logger("sorakai.rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

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

    app.state.store = store
    app.state.chat_store = chat
    app.state.kb_meta = kb_meta
    app.state.vector_store = vector_store
    app.state.rag_chain = chain
    app.state.retriever = retriever
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


def _install_cors(app: FastAPI) -> None:
    """Install CORS. ``*`` and credentials are never combined (browsers reject it)."""
    origins = get_settings().cors_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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

    _install_cors(app)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exc(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

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

        try:
            result: dict[str, Any] = await ainvoke_rag(chain, question=body.question, session_id=sid)
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

        with mlflow_run("sorakai-rag", run_name="query"):
            log_params_metrics(
                {"service": "rag", "top_k": float(body.top_k), "session": float(bool(sid))},
                {
                    "context_len": float(len(context)),
                    "answer_len": float(len(answer)),
                    "sources_used": float(sources_used),
                },
            )

        return QueryResponse(
            answer=answer,
            context_preview=render_context_preview(context),
            sources_used=sources_used,
            session_id=sid,
        )

    register_bundled_openapi_routes(app, "rag")
    return app


app = create_app()
