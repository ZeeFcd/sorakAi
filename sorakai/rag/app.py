from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sorakai import __version__
from sorakai.common.config import get_settings
from sorakai.common.embedding import embed_chunks
from sorakai.common.llm import ask_llm
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.mlflow_tracking import log_params_metrics, mlflow_run
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.retrieval import retrieve_best_chunk
from sorakai.common.schemas import HealthResponse, QueryRequest, QueryResponse, ReadinessResponse
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store

logger = get_logger("sorakai.rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = create_store(settings.redis_url)
    app.state.store = store
    logger.info("RAG service started (redis=%s)", bool(settings.redis_url))
    yield
    if isinstance(store, RedisKnowledgeStore):
        await store.aclose()
    logger.info("RAG service shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi RAG",
        description="Retrieval and LLM answer generation",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        # Only verify store reachability so the pod can become Ready before first ingest.
        ok_store = await store.ping()
        if not ok_store:
            return ReadinessResponse(ready=False, service="rag", detail="store_unreachable")
        return ReadinessResponse(ready=True, service="rag")

    @app.post("/v1/query", response_model=QueryResponse, tags=["rag"])
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        store: KnowledgeStore = request.app.state.store
        loaded = await store.load()
        if not loaded:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents in knowledge base")
        chunks, embeddings = loaded
        q_emb = embed_chunks([body.question])[0]
        context = retrieve_best_chunk(q_emb, embeddings, chunks)
        if not context:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not retrieve context")

        answer = ask_llm(body.question, context)
        preview = context[:280] + ("…" if len(context) > 280 else "")

        with mlflow_run("sorakai-rag", run_name="query"):
            log_params_metrics({"service": "rag"}, {"context_len": float(len(context)), "answer_len": float(len(answer))})

        return QueryResponse(answer=answer, context_preview=preview)

    register_bundled_openapi_routes(app, "rag")
    return app


app = create_app()
