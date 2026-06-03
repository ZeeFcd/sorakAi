"""Shared HTTP middleware for the three FastAPI services (Wave 10).

Before Wave 10 each app re-implemented CORS, the request-id middleware,
and the catch-all 500 handler. Three copies drifted (subtle differences
in header casing, the time-format, and the exception logger). This
module is the single source of truth.

Public surface:

- :func:`install_cors` — read CORS origins from settings and wire
  the FastAPI ``CORSMiddleware`` with the credential-safe defaults
  (browsers reject ``*`` together with ``allow_credentials=True``).
- :func:`install_request_id` — accept or mint an ``X-Request-ID``,
  bind it into structlog's context, time the request, and reflect both
  on the response.
- :func:`install_exception_handler` — log unhandled errors with the
  caller's logger and surface a generic 500 JSON payload so we never
  leak stack traces.
- :func:`install_request_size_limit` — reject requests whose
  ``Content-Length`` exceeds ``settings.request_max_bytes`` with a
  413 before the body is buffered into memory.
- :func:`install_common_middleware` — convenience helper that wires
  all of the above in the canonical order, used by every app's
  ``create_app``.

Wave 10 also introduced :mod:`sorakai.common.security` for the gateway-only
bearer auth + rate-limit hooks; those are deliberately kept out of this
module because ingest/RAG are internal-only and never authenticated.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sorakai.common.config import Settings
from sorakai.common.logging_utils import (
    bind_request_id,
    clear_request_context,
    new_request_id,
)
from sorakai.core.logging import get_logger

_logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time"
"""HTTP header names exposed by the request-id middleware.

Constants so clients (and the Streamlit UI under ``ui/``) don't have to
hard-code the strings. The headers are intentionally stable across
services so an operator can correlate ``X-Request-ID`` end-to-end
without service-specific knowledge."""


def install_cors(app: FastAPI, settings: Settings) -> None:
    """Add the standard CORS middleware to ``app``.

    Origins come from :attr:`Settings.cors_origins`; when ``*`` is
    present we suppress credentials because browsers refuse the
    combination (treating it as a CORS error and dropping the
    response). Callers that need cookie-based auth must pin explicit
    origins.
    """
    origins = settings.cors_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def install_request_id(app: FastAPI) -> None:
    """Bind structlog request-id context for every HTTP request.

    Accepts an inbound ``X-Request-ID`` (so a gateway-level id flows
    through to the upstream services) or mints a fresh one. The id is
    reflected on the response and the elapsed wall-clock time is
    surfaced in ``X-Process-Time`` so curl users can see latency
    without enabling tracing.
    """

    @app.middleware("http")
    async def _request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        bind_request_id(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers[REQUEST_ID_HEADER] = rid
        response.headers[PROCESS_TIME_HEADER] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
        return response


def install_exception_handler(app: FastAPI, *, service: str) -> None:
    """Translate unhandled exceptions into a stable 500 JSON shape.

    ``service`` only flavours the log message (the response stays the
    same across services so clients can build one error renderer).
    Errors are logged via the per-service logger named
    ``sorakai.<service>`` so the rendered structlog line carries the
    right logger context.
    """
    service_logger = get_logger(f"sorakai.{service}")

    @app.exception_handler(Exception)
    async def _unhandled_exc(_: Request, exc: Exception) -> JSONResponse:
        service_logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


class _RequestSizeLimitMiddleware:
    """Tiny ASGI middleware that rejects oversized requests.

    Lives here (rather than in :mod:`security`) because it is a
    universally-safe HTTP hygiene rule rather than an authn/authz
    concern. Strategy:

    - For methods that carry a body (POST/PUT/PATCH) we look at
      ``Content-Length`` and refuse 413 if it exceeds the configured
      cap. We do **not** buffer the body to measure it; clients that
      misreport ``Content-Length`` either crash uvicorn's HTTP parser
      (and never reach a handler) or get a 400 later, which is the
      right outcome for a misbehaving client.
    - ``GET``/``DELETE``/``HEAD``/``OPTIONS`` skip the check.
    - ``max_bytes <= 0`` disables the middleware entirely.
    """

    _CHECKED_METHODS = frozenset({"POST", "PUT", "PATCH"})

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._max_bytes <= 0:
            await self._app(scope, receive, send)
            return
        method = str(scope.get("method", "")).upper()
        if method not in self._CHECKED_METHODS:
            await self._app(scope, receive, send)
            return
        content_length = self._extract_content_length(scope)
        if content_length is None or content_length <= self._max_bytes:
            await self._app(scope, receive, send)
            return
        await self._send_413(send, observed=content_length)

    @staticmethod
    def _extract_content_length(scope: Scope) -> int | None:
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"content-length":
                try:
                    return int(header_value)
                except ValueError:
                    return None
        return None

    async def _send_413(self, send: Send, *, observed: int) -> None:
        payload = (f'{{"detail":"Request body too large: {observed} > {self._max_bytes} bytes"}}').encode()
        await send(_start_message(413, len(payload)))
        await send({"type": "http.response.body", "body": payload, "more_body": False})


def _start_message(status_code: int, body_length: int) -> Message:
    return {
        "type": "http.response.start",
        "status": status_code,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(body_length).encode("ascii")),
        ],
    }


def install_request_size_limit(app: FastAPI, settings: Settings) -> None:
    """Wire the request-size limit middleware from settings."""
    app.add_middleware(_RequestSizeLimitMiddleware, max_bytes=settings.request_max_bytes)


def install_common_middleware(
    app: FastAPI,
    settings: Settings,
    *,
    service: str,
) -> None:
    """Wire CORS + request-size limit + request-id + 500 handler.

    Order matters in Starlette:

    1. ``install_cors`` first so CORS headers wrap every other layer
       (and pre-flight ``OPTIONS`` short-circuits before our checks).
    2. ``install_request_size_limit`` next so the cap is enforced
       *before* FastAPI starts buffering the request body.
    3. ``install_request_id`` so context is bound for the size-limit's
       few synthetic responses too.
    4. ``install_exception_handler`` last so it catches everything that
       slips through.
    """
    install_cors(app, settings)
    install_request_size_limit(app, settings)
    install_request_id(app)
    install_exception_handler(app, service=service)


__all__ = [
    "PROCESS_TIME_HEADER",
    "REQUEST_ID_HEADER",
    "install_common_middleware",
    "install_cors",
    "install_exception_handler",
    "install_request_id",
    "install_request_size_limit",
]
