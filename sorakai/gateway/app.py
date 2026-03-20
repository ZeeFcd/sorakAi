from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sorakai import __version__
from sorakai.common.config import get_settings
from sorakai.common.logging_utils import get_logger, new_request_id, request_id_ctx
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)

logger = get_logger("sorakai.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    app.state.http = httpx.AsyncClient(timeout=timeout)
    logger.info("Gateway started")
    yield
    await app.state.http.aclose()
    logger.info("Gateway shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="sorakAi Gateway",
        description="BFF orchestrating ingest and RAG microservices",
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "documents", "description": "Ingest pipeline"},
            {"name": "rag", "description": "Question answering"},
            {"name": "ops", "description": "Health and readiness"},
        ],
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

    def client(request: Request) -> httpx.AsyncClient:
        return request.app.state.http

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(service="gateway")

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    async def ready(request: Request) -> ReadinessResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        errors: list[str] = []
        for name, url in (("ingest", cfg.ingest_service_url), ("rag", cfg.rag_service_url)):
            try:
                r = await http.get(f"{url.rstrip('/')}/health")
                if r.status_code != 200:
                    errors.append(f"{name}:{r.status_code}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}:{e!s}")
        if errors:
            return ReadinessResponse(ready=False, service="gateway", detail=";".join(errors))
        return ReadinessResponse(ready=True, service="gateway")

    @app.get("/", tags=["ops"])
    async def root() -> dict[str, str]:
        cfg = get_settings()
        return {
            "service": "sorakAi-gateway",
            "docs": "/docs",
            "ingest_upstream": cfg.ingest_service_url,
            "rag_upstream": cfg.rag_service_url,
        }

    @app.post(
        "/api/v1/documents",
        response_model=DocumentIngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def proxy_ingest(body: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.ingest_service_url.rstrip('/')}/v1/documents"
        try:
            r = await http.post(url, json=body.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"ingest_unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return DocumentIngestResponse.model_validate(r.json())

    @app.post("/api/v1/query", response_model=QueryResponse, tags=["rag"])
    async def proxy_query(body: QueryRequest, request: Request) -> QueryResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.rag_service_url.rstrip('/')}/v1/query"
        try:
            r = await http.post(url, json=body.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"rag_unreachable: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:  # noqa: BLE001
                detail = r.text
            raise HTTPException(status_code=r.status_code, detail=detail)
        return QueryResponse.model_validate(r.json())

    register_bundled_openapi_routes(app, "gateway")
    return app


app = create_app()
