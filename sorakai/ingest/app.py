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
from sorakai.common.ingest import process_file
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.mlflow_tracking import log_params_metrics, mlflow_run
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import DocumentIngestRequest, DocumentIngestResponse, HealthResponse, ReadinessResponse
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store

logger = get_logger("sorakai.ingest")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = create_store(settings.redis_url)
    app.state.store = store
    logger.info("Ingest service started (redis=%s)", bool(settings.redis_url))
    yield
    if isinstance(store, RedisKnowledgeStore):
        await store.aclose()
    logger.info("Ingest service shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi Ingest",
        description="Document chunking, embedding, and KB persistence",
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
        return HealthResponse(service="ingest")

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    async def ready(request: Request) -> ReadinessResponse:
        store: KnowledgeStore = request.app.state.store
        ok = await store.ping()
        if not ok:
            return ReadinessResponse(ready=False, service="ingest", detail="store_unreachable")
        return ReadinessResponse(ready=True, service="ingest")

    @app.post(
        "/v1/documents",
        response_model=DocumentIngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def ingest_document(body: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
        store: KnowledgeStore = request.app.state.store
        chunks = process_file(body.content, body.chunk_size)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced from content")
        vectors = embed_chunks(chunks)
        await store.save(chunks, vectors)

        with mlflow_run("sorakai-ingest", run_name=f"ingest-{body.filename}"):
            log_params_metrics(
                {"filename": body.filename, "chunk_size": body.chunk_size, "service": "ingest"},
                {"num_chunks": float(len(chunks))},
            )

        return DocumentIngestResponse(
            message=f"Stored {len(chunks)} chunks for '{body.filename}'",
            num_chunks=len(chunks),
            filename=body.filename,
        )

    register_bundled_openapi_routes(app, "ingest")
    return app


app = create_app()
