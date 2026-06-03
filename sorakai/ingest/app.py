from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sorakai import __version__
from sorakai.common.config import get_settings
from sorakai.common.embedding import embed_chunks
from sorakai.common.ingest import chunk_document
from sorakai.common.kb_meta import (
    KBMeta,
    KBMetaStore,
    RedisKBMetaStore,
    create_kb_meta_store,
)
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.mlflow_tracking import log_params_metrics, mlflow_run
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import (
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    HealthResponse,
    ReadinessResponse,
    new_document_id,
)
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore, create_store
from sorakai.core.errors import DimensionMismatchError
from sorakai.core.logging import configure_logging

logger = get_logger("sorakai.ingest")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = create_store(settings.redis_url)
    kb_meta = create_kb_meta_store(settings.redis_url)
    app.state.store = store
    app.state.kb_meta = kb_meta
    logger.info(
        "Ingest service started (redis=%s, embedding_provider=%s)",
        bool(settings.redis_url),
        settings.embedding_provider,
    )
    yield
    if isinstance(store, RedisKnowledgeStore):
        await store.aclose()
    if isinstance(kb_meta, RedisKBMetaStore):
        await kb_meta.aclose()
    logger.info("Ingest service shutdown")


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
        title="sorakAi Ingest",
        description="Document chunking, embedding, and KB persistence",
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
        kb_meta: KBMetaStore = request.app.state.kb_meta
        settings = get_settings()

        chunks = chunk_document(
            body.content,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
            filename=body.filename,
            mime_type=body.mime_type,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced from content")
        vectors = await embed_chunks(chunks)

        # Dim guard: every chunk-vector must share a dim, and the KB must
        # have been built (or be being built) with the same provider/model/dim.
        dims = {v.size for v in vectors}
        if len(dims) != 1:
            raise HTTPException(status_code=500, detail=f"Embedding provider returned mixed dims: {sorted(dims)}")
        candidate = KBMeta(
            provider=settings.embedding_provider,
            model=settings.ollama_embedding_model,
            dim=next(iter(dims)),
        )
        # Append path: pre-flight the meta check BEFORE touching the store so
        # mismatched ingests never write chunks they'd just have to roll back.
        if not body.replace_kb:
            try:
                await kb_meta.ensure_compatible(candidate)
            except DimensionMismatchError as exc:
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
                        "hint": "Re-ingest the corpus with the current provider/model, or pass replace_kb=true.",
                    },
                ) from exc

        doc_id = body.document_id or new_document_id()
        if body.replace_kb:
            # Wave 4: store write happens BEFORE the meta swap so a meta-write
            # failure leaves a self-consistent (chunks + stale-meta) state that
            # the dim guard surfaces as 409 on the next request - never the
            # silent corruption the meta-first ordering used to produce.
            await store.replace_kb_with_document(
                doc_id,
                body.filename,
                chunks,
                vectors,
                mime_type=body.mime_type,
            )
            await kb_meta.reset_to(candidate)
        else:
            await store.append_document(
                doc_id,
                body.filename,
                chunks,
                vectors,
                mime_type=body.mime_type,
            )

        with mlflow_run("sorakai-ingest", run_name=f"ingest-{body.filename}"):
            log_params_metrics(
                {
                    "filename": body.filename,
                    "chunk_size": body.chunk_size,
                    "chunk_overlap": body.chunk_overlap,
                    "mime_type": body.mime_type or "auto",
                    "service": "ingest",
                    "replace_kb": body.replace_kb,
                },
                {"num_chunks": float(len(chunks))},
            )

        return DocumentIngestResponse(
            message=f"Stored {len(chunks)} chunks for '{body.filename}' (append)"
            if not body.replace_kb
            else f"Replaced KB with {len(chunks)} chunks from '{body.filename}'",
            num_chunks=len(chunks),
            filename=body.filename,
            document_id=doc_id,
        )

    @app.get(
        "/v1/documents",
        response_model=DocumentListResponse,
        tags=["documents"],
    )
    async def list_documents(request: Request) -> DocumentListResponse:
        store: KnowledgeStore = request.app.state.store
        summaries = await store.list_documents()
        return DocumentListResponse(
            documents=[
                DocumentSummaryResponse(
                    doc_id=s.doc_id,
                    filename=s.filename,
                    chunk_count=s.chunk_count,
                    mime=s.mime,
                )
                for s in summaries
            ],
            total=len(summaries),
        )

    @app.delete(
        "/v1/documents/{doc_id}",
        response_model=DocumentDeleteResponse,
        tags=["documents"],
    )
    async def delete_document(doc_id: str, request: Request) -> DocumentDeleteResponse:
        if not doc_id.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="doc_id must not be empty")
        store: KnowledgeStore = request.app.state.store
        removed = await store.delete_document(doc_id)
        if removed == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document with doc_id={doc_id!r}",
            )
        return DocumentDeleteResponse(
            doc_id=doc_id,
            removed_chunks=removed,
            message=f"Removed {removed} chunks for document {doc_id!r}",
        )

    register_bundled_openapi_routes(app, "ingest")
    return app


app = create_app()
