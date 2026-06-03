from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse

from sorakai import __version__
from sorakai.common.config import get_settings
from sorakai.common.logging_utils import get_logger
from sorakai.common.middleware import install_common_middleware
from sorakai.common.openapi_bundle import register_bundled_openapi_routes
from sorakai.common.schemas import (
    AgentRequest,
    AgentResponse,
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from sorakai.common.security import install_gateway_security
from sorakai.common.telemetry import (
    configure_tracing,
    instrument_fastapi,
    instrument_httpx,
)
from sorakai.core.logging import configure_logging

logger = get_logger("sorakai.gateway")

API_V1_LEGACY_PREFIX = "/api/v1"
V1_PREFIX = "/v1"
"""Wave 10: ``/v1/*`` is the new canonical surface; ``/api/v1/*`` stays
as a 308 redirect for one release to give clients a deprecation window.
``308 Permanent Redirect`` preserves both the request method and the
body, so POSTs forwarded to ``/api/v1/query`` keep working."""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, log_format=settings.log_format)
    configure_tracing("sorakai-gateway", settings, version=__version__)
    instrument_fastapi(app)
    instrument_httpx()
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    app.state.http = httpx.AsyncClient(timeout=timeout)
    logger.info("Gateway started")
    yield
    await app.state.http.aclose()
    logger.info("Gateway shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="sorakAi Gateway",
        description="BFF orchestrating ingest and RAG microservices",
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "documents", "description": "Ingest pipeline"},
            {"name": "rag", "description": "Question answering"},
            {"name": "ops", "description": "Health and readiness"},
            {"name": "compat", "description": "Legacy /api/v1 -> /v1 redirects"},
        ],
    )

    install_common_middleware(app, settings, service="gateway")
    auth_dep, rate_limit_dep = install_gateway_security(app, settings)
    # ``Depends(...)`` wrappers so we can reuse the exact instance across
    # every guarded route while still letting tests swap the underlying
    # dependency via ``app.dependency_overrides``.
    auth = Depends(auth_dep)
    rate_limit = Depends(rate_limit_dep)
    secured = [auth, rate_limit]

    def client(request: Request) -> httpx.AsyncClient:
        http: httpx.AsyncClient = request.app.state.http
        return http

    # ------------------------------------------------------------------
    # Ops (unauthenticated)
    # ------------------------------------------------------------------

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
            except Exception as e:
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
            "api_base": V1_PREFIX,
        }

    # ------------------------------------------------------------------
    # /v1/* canonical proxies (guarded by bearer auth + rate limit)
    # ------------------------------------------------------------------

    @app.post(
        f"{V1_PREFIX}/documents",
        response_model=DocumentIngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
        dependencies=secured,
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

    @app.get(
        f"{V1_PREFIX}/documents",
        response_model=DocumentListResponse,
        tags=["documents"],
        dependencies=secured,
    )
    async def proxy_list_documents(request: Request) -> DocumentListResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.ingest_service_url.rstrip('/')}/v1/documents"
        try:
            r = await http.get(url)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"ingest_unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return DocumentListResponse.model_validate(r.json())

    @app.delete(
        f"{V1_PREFIX}/documents/{{doc_id}}",
        response_model=DocumentDeleteResponse,
        tags=["documents"],
        dependencies=secured,
    )
    async def proxy_delete_document(doc_id: str, request: Request) -> DocumentDeleteResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.ingest_service_url.rstrip('/')}/v1/documents/{doc_id}"
        try:
            r = await http.delete(url)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"ingest_unreachable: {e}") from e
        if r.status_code >= 400:
            detail = _safe_detail(r)
            raise HTTPException(status_code=r.status_code, detail=detail)
        return DocumentDeleteResponse.model_validate(r.json())

    @app.post(f"{V1_PREFIX}/query", response_model=QueryResponse, tags=["rag"], dependencies=secured)
    async def proxy_query(body: QueryRequest, request: Request) -> QueryResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.rag_service_url.rstrip('/')}/v1/query"
        try:
            r = await http.post(url, json=body.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"rag_unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=_safe_detail(r))
        return QueryResponse.model_validate(r.json())

    @app.post(f"{V1_PREFIX}/agent", response_model=AgentResponse, tags=["agent"], dependencies=secured)
    async def proxy_agent(body: AgentRequest, request: Request) -> AgentResponse:
        cfg = get_settings()
        http: httpx.AsyncClient = client(request)
        url = f"{cfg.rag_service_url.rstrip('/')}/v1/agent"
        try:
            r = await http.post(url, json=body.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"rag_unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=_safe_detail(r))
        return AgentResponse.model_validate(r.json())

    @app.post(f"{V1_PREFIX}/query/stream", tags=["rag"], dependencies=secured)
    async def proxy_query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        return await _proxy_stream(request, "/v1/query/stream", body.model_dump())

    @app.post(f"{V1_PREFIX}/agent/stream", tags=["agent"], dependencies=secured)
    async def proxy_agent_stream(body: AgentRequest, request: Request) -> StreamingResponse:
        return await _proxy_stream(request, "/v1/agent/stream", body.model_dump())

    # ------------------------------------------------------------------
    # /api/v1/* compatibility redirects (one-release deprecation window)
    # ------------------------------------------------------------------

    @app.api_route(
        f"{API_V1_LEGACY_PREFIX}/{{rest_of_path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        tags=["compat"],
        include_in_schema=False,
    )
    async def legacy_api_v1_redirect(rest_of_path: str, request: Request) -> RedirectResponse:
        query = request.url.query
        target = f"{V1_PREFIX}/{rest_of_path}" + (f"?{query}" if query else "")
        # 308 preserves method + body across the redirect (per RFC 7538),
        # which the legacy curl one-liners around the team rely on.
        return RedirectResponse(
            url=target,
            status_code=status.HTTP_308_PERMANENT_REDIRECT,
        )

    register_bundled_openapi_routes(app, "gateway")
    return app


def _safe_detail(response: httpx.Response) -> Any:
    """Pull ``detail`` out of an upstream JSON error, falling back to raw text.

    Centralised so every proxy renders upstream failures identically and
    we don't leak HTML 500 pages into the gateway's JSON response shape.
    """
    try:
        payload: dict[str, Any] = cast(dict[str, Any], response.json())
        return payload.get("detail", response.text)
    except ValueError:
        return response.text


async def _proxy_stream(
    request: Request,
    path: str,
    payload: dict[str, object],
) -> StreamingResponse:
    """Forward an SSE request to the RAG service, streaming bytes through.

    The upstream client is opened per-request so the response lifetime is
    bounded by the iterator; ``app.state.http`` is reserved for the
    short-lived JSON proxies above.
    """
    cfg = get_settings()
    url = f"{cfg.rag_service_url.rstrip('/')}{path}"

    async def _gen() -> AsyncIterator[bytes]:
        timeout = httpx.Timeout(cfg.request_timeout_seconds, read=None)
        async with httpx.AsyncClient(timeout=timeout) as upstream:
            try:
                async with upstream.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        yield (
                            f'event: error\ndata: {{"status":{resp.status_code},"detail":{body.decode("utf-8", "replace")!r}}}\n\n'.encode()
                        )
                        return
                    async for chunk in resp.aiter_raw():
                        yield chunk
            except httpx.RequestError as e:
                yield (f'event: error\ndata: {{"error":"rag_unreachable","message":{str(e)!r}}}\n\n').encode()

    return StreamingResponse(_gen(), media_type="text/event-stream")


app = create_app()
