from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException, Request, status

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
from sorakai.common.logging_utils import get_logger
from sorakai.common.middleware import install_common_middleware
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
from sorakai.common.telemetry import (
    configure_tracing,
    instrument_fastapi,
    instrument_httpx,
    span,
)
from sorakai.core.errors import DimensionMismatchError
from sorakai.core.logging import configure_logging
from sorakai.infra.vector_store.base import VectorDoc, VectorStore
from sorakai.infra.vector_store.factory import get_vector_store

logger = get_logger("sorakai.ingest")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, log_format=settings.log_format)
    configure_tracing("sorakai-ingest", settings, version=__version__)
    instrument_fastapi(app)
    instrument_httpx()
    store = create_store(settings.redis_url)
    vector_store = get_vector_store(settings)
    kb_meta = create_kb_meta_store(settings.redis_url)
    app.state.store = store
    app.state.vector_store = vector_store
    app.state.kb_meta = kb_meta
    logger.info(
        "Ingest service started (redis=%s, embedding_provider=%s)",
        bool(settings.redis_url),
        settings.embedding_provider,
    )
    yield
    if isinstance(store, RedisKnowledgeStore):
        await store.aclose()
    await vector_store.aclose()
    if isinstance(kb_meta, RedisKBMetaStore):
        await kb_meta.aclose()
    logger.info("Ingest service shutdown")


def _record_ingest_metrics(
    settings: Any,
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log one MLflow run with the ingest job's params + metrics.

    Silently no-ops when ``mlflow_tracking_uri`` is unset or when MLflow
    raises (network blip, missing experiment, etc.). Centralised here so
    Wave 8's structlog migration removed the last call site of the
    legacy ``mlflow_tracking.mlflow_run`` context manager.
    """
    if not settings.mlflow_tracking_uri:
        return
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("sorakai-ingest")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
    except Exception as exc:
        logger.warning("MLflow ingest run skipped: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi Ingest",
        description="Document chunking, embedding, and KB persistence",
        version=__version__,
        lifespan=lifespan,
    )

    install_common_middleware(app, get_settings(), service="ingest")

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

        with span(
            "ingest.chunk",
            filename=body.filename,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        ):
            chunks = chunk_document(
                body.content,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
                filename=body.filename,
                mime_type=body.mime_type,
            )
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced from content")
        with span("ingest.embed", num_chunks=len(chunks)):
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
        vector_store: VectorStore = request.app.state.vector_store
        vector_docs = _to_vector_docs(
            doc_id=doc_id,
            filename=body.filename,
            chunks=chunks,
            vectors=vectors,
            mime_type=body.mime_type,
        )
        if body.replace_kb:
            # Wave 4: store write happens BEFORE the meta swap so a meta-write
            # failure leaves a self-consistent (chunks + stale-meta) state that
            # the dim guard surfaces as 409 on the next request - never the
            # silent corruption the meta-first ordering used to produce.
            await _replace_vector_store_with_document(vector_store, vector_docs)
            await store.replace_kb_with_document(
                doc_id,
                body.filename,
                chunks,
                vectors,
                mime_type=body.mime_type,
            )
            await kb_meta.reset_to(candidate)
        else:
            await vector_store.upsert(vector_docs)
            await store.append_document(
                doc_id,
                body.filename,
                chunks,
                vectors,
                mime_type=body.mime_type,
            )

        # MLflow logging for the ingest pipeline stays at module scope (no
        # LangChain callback to plug in) but we route the writes through
        # the same helper so a missing tracking URI silently no-ops just
        # like in the RAG handler.
        _record_ingest_metrics(
            settings,
            run_name=f"ingest-{body.filename}",
            params={
                "filename": body.filename,
                "chunk_size": body.chunk_size,
                "chunk_overlap": body.chunk_overlap,
                "mime_type": body.mime_type or "auto",
                "service": "ingest",
                "replace_kb": body.replace_kb,
            },
            metrics={"num_chunks": float(len(chunks))},
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
        vector_store: VectorStore = request.app.state.vector_store
        removed = await store.delete_document(doc_id)
        if removed == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document with doc_id={doc_id!r}",
            )
        await vector_store.delete_doc(doc_id)
        return DocumentDeleteResponse(
            doc_id=doc_id,
            removed_chunks=removed,
            message=f"Removed {removed} chunks for document {doc_id!r}",
        )

    register_bundled_openapi_routes(app, "ingest")
    return app


def _to_vector_docs(
    *,
    doc_id: str,
    filename: str,
    chunks: list[str],
    vectors: list[Any],
    mime_type: str | None,
) -> list[VectorDoc]:
    total = len(chunks)
    return [
        VectorDoc(
            page_content=chunk,
            embedding=vector,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx,
                "chunk_total": total,
                "mime": mime_type,
            },
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]


async def _replace_vector_store_with_document(
    vector_store: VectorStore,
    docs: list[VectorDoc],
) -> None:
    for summary in await vector_store.list_docs():
        await vector_store.delete_doc(summary.doc_id)
    await vector_store.upsert(docs)


app = create_app()
