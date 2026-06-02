from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sorakai import __version__
from sorakai.common.chat_history import RedisChatHistoryStore, create_chat_store, validate_session_id
from sorakai.common.config import get_settings
from sorakai.common.embedding import embed_chunks
from sorakai.common.kb_meta import (
    KBMeta,
    KBMetaStore,
    RedisKBMetaStore,
    create_kb_meta_store,
)
from sorakai.common.llm import ask_llm
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.mlflow_tracking import log_params_metrics, mlflow_run
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.retrieval import retrieve_top_k_context
from sorakai.common.schemas import HealthResponse, QueryRequest, QueryResponse, ReadinessResponse
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store
from sorakai.core.errors import DimensionMismatchError
from sorakai.core.logging import configure_logging

logger = get_logger("sorakai.rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = create_store(settings.redis_url)
    chat = create_chat_store(
        settings.redis_url,
        ttl_seconds=settings.chat_history_ttl_seconds,
        max_messages=settings.chat_history_max_messages,
    )
    kb_meta = create_kb_meta_store(settings.redis_url)
    app.state.store = store
    app.state.chat_store = chat
    app.state.kb_meta = kb_meta
    logger.info(
        "RAG service started (redis=%s, llm_provider=%s, embedding_provider=%s)",
        bool(settings.redis_url),
        settings.llm_provider,
        settings.embedding_provider,
    )
    yield
    if isinstance(store, RedisKnowledgeStore):
        await store.aclose()
    if isinstance(chat, RedisChatHistoryStore):
        await chat.aclose()
    if isinstance(kb_meta, RedisKBMetaStore):
        await kb_meta.aclose()
    logger.info("RAG service shutdown")


def _install_cors(app: FastAPI) -> None:
    """Install the CORS middleware.

    Browsers reject ``Access-Control-Allow-Origin: *`` together with
    ``Access-Control-Allow-Credentials: true``, so we never combine them.
    """
    origins = get_settings().cors_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi RAG",
        description="Retrieval and LLM answer generation",
        version=__version__,
        lifespan=lifespan,
    )

    _install_cors(app)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
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
        chat_store = request.app.state.chat_store
        kb_meta: KBMetaStore = request.app.state.kb_meta
        settings = get_settings()

        try:
            sid = validate_session_id(body.session_id) if body.use_chat_history else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        loaded = await store.load_flat()
        if not loaded:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents in knowledge base")
        chunks, embeddings = loaded
        q_emb = (await embed_chunks([body.question]))[0]

        # Dim guard: query must use the same provider/model/dim as the stored KB.
        stored_meta = await kb_meta.read()
        if stored_meta is not None:
            candidate = KBMeta(
                provider=settings.embedding_provider,
                model=settings.ollama_embedding_model,
                dim=q_emb.size,
            )
            if not stored_meta.matches(candidate):
                exc = DimensionMismatchError(
                    expected_provider=stored_meta.provider,
                    expected_model=stored_meta.model,
                    expected_dim=stored_meta.dim,
                    actual_provider=candidate.provider,
                    actual_model=candidate.model,
                    actual_dim=candidate.dim,
                )
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
                            "Re-ingest the corpus with the current provider/model, "
                            "or POST to /v1/documents with replace_kb=true."
                        ),
                    },
                ) from exc

        context, n_sources = retrieve_top_k_context(q_emb, embeddings, chunks, top_k=body.top_k)
        if not context:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not retrieve context")

        prior: list[dict[str, str]] = []
        if sid:
            prior = await chat_store.get_messages(sid)

        answer = await ask_llm(body.question, context, conversation=prior or None)

        if sid:
            await chat_store.append_pair(sid, body.question, answer)

        preview = context[:400] + ("…" if len(context) > 400 else "")

        with mlflow_run("sorakai-rag", run_name="query"):
            log_params_metrics(
                {"service": "rag", "top_k": float(body.top_k), "session": float(bool(sid))},
                {
                    "context_len": float(len(context)),
                    "answer_len": float(len(answer)),
                    "sources_used": float(n_sources),
                },
            )

        return QueryResponse(
            answer=answer,
            context_preview=preview,
            sources_used=n_sources,
            session_id=sid,
        )

    register_bundled_openapi_routes(app, "rag")
    return app


app = create_app()
