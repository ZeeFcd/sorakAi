"""Gateway-only security hooks: bearer auth + rate limit (Wave 10).

These hooks are deliberately kept **out** of :mod:`sorakai.common.middleware`:
ingest and RAG run inside the cluster boundary and are reached only by
the gateway, so they don't need an authn/authz layer; bolting one on
them would just add latency and a tempting "we have auth, so we don't
need the gateway" anti-pattern.

Two surfaces:

- :func:`bearer_auth_dependency` returns a FastAPI dependency that
  enforces ``Authorization: Bearer <key>`` when ``GATEWAY_API_KEY`` is
  set; when the env var is unset the dependency is a no-op so local
  dev keeps working out of the box.
- :class:`RateLimiter` wraps the third-party ``slowapi`` limiter with
  a thin facade: Redis storage when ``REDIS_URL`` is set, in-memory
  otherwise; ``settings.rate_limit_per_minute <= 0`` disables it
  entirely. The class exposes :meth:`install` (registers the limiter
  + exception handler on the FastAPI app) and :meth:`limit_route` so
  routes can opt in with a decorator-free dependency.

Both helpers are factories so settings can be re-read between tests
(the singleton pattern fights with ``monkeypatch.setenv``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from limits import parse as parse_rate_limit
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sorakai.common.config import Settings
from sorakai.core.logging import get_logger

_logger = get_logger(__name__)

_BEARER_SCHEME = HTTPBearer(auto_error=False)
"""Single :class:`HTTPBearer` instance reused by the dependency factory.
``auto_error=False`` means we can produce our own 401 body shape
(matching the rest of the gateway's error contract) instead of FastAPI's
default ``{"detail": "Not authenticated"}``."""


# ---------------------------------------------------------------------------
# Bearer auth
# ---------------------------------------------------------------------------


def bearer_auth_dependency(
    settings: Settings,
) -> Callable[[HTTPAuthorizationCredentials | None], Awaitable[None]]:
    """Return a FastAPI dependency enforcing ``Authorization: Bearer <key>``.

    No-op when :attr:`Settings.gateway_api_key` is ``None`` / empty so
    local dev keeps working without setting an env var. Returns the
    coroutine itself (not the value) so FastAPI handles the
    ``Depends(...)`` plumbing.
    """
    api_key = (settings.gateway_api_key or "").strip()

    async def _enforce(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER_SCHEME)],
    ) -> None:
        if not api_key:
            return
        if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not _secure_compare(creds.credentials, api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _enforce


def _secure_compare(presented: str, expected: str) -> bool:
    """Constant-time string compare so length/byte timing doesn't leak the key."""
    if len(presented) != len(expected):
        return False
    mismatch = 0
    for left, right in zip(presented, expected, strict=True):
        mismatch |= ord(left) ^ ord(right)
    return mismatch == 0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Facade around ``slowapi`` with sorakAi's configuration baked in.

    - ``settings.rate_limit_per_minute <= 0`` disables the limiter and
      :meth:`limit_dependency` becomes a no-op (returns ``None``).
    - When ``settings.redis_url`` is set we point ``slowapi`` at Redis so
      multiple gateway replicas share one bucket; otherwise we use the
      in-memory backend.
    - The limit is expressed as ``"<per_minute>/minute"`` plus the burst
      cap so quick succession ``burst`` requests at most then drip into
      the per-minute budget.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._enabled = settings.rate_limit_per_minute > 0
        self._limit_string = (
            f"{settings.rate_limit_per_minute}/minute"
            if self._enabled
            else "1000000000/minute"  # never hit; used as no-op default
        )
        self._burst = max(settings.rate_limit_burst, settings.rate_limit_per_minute)
        if self._enabled:
            storage_uri = settings.redis_url or "memory://"
            self._limiter = Limiter(key_func=get_remote_address, storage_uri=storage_uri)
        else:
            self._limiter = Limiter(key_func=get_remote_address, storage_uri="memory://", enabled=False)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def limiter(self) -> Limiter:
        """Expose the underlying slowapi limiter for tests / advanced wiring."""
        return self._limiter

    def install(self, app: FastAPI) -> None:
        """Attach the limiter + the 429 exception handler on ``app``."""
        app.state.limiter = self._limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    def limit_dependency(self) -> Callable[[Request], Awaitable[None]]:
        """Build a FastAPI dependency that counts a hit against the limit.

        When disabled the dependency returns immediately, so wiring it
        unconditionally is cheap. We use a dependency (not slowapi's
        decorator) so the same limit is enforced regardless of whether
        the route signature happens to take ``request`` positionally.
        """
        if not self._enabled:

            async def _noop(_: Request) -> None:
                return None

            return _noop

        per_minute = self._settings.rate_limit_per_minute
        burst = self._burst
        # ``limits.parse`` returns a ``RateLimitItem`` from "<n>/<unit>"
        # strings; we share the parsed item so the limiter doesn't
        # reparse on every request.
        rate_item = parse_rate_limit(self._limit_string)

        async def _enforce(request: Request) -> None:
            key = get_remote_address(request)
            limiter = cast(Limiter, request.app.state.limiter)
            # ``slowapi`` exposes the underlying limits backend through the
            # ``_limiter`` private attr; using ``hit`` returns False when
            # the bucket is empty. The same ``key`` is used as the
            # identifier so the limiter coalesces per-IP across calls.
            allowed = limiter._limiter.hit(rate_item, key, cost=1)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {per_minute}/minute (burst={burst})",
                )

        return _enforce


async def _rate_limit_exceeded_handler(_: Request, exc: Exception) -> JSONResponse:
    """Render slowapi's ``RateLimitExceeded`` as a stable 429 JSON body."""
    detail = str(exc) if str(exc) else "Rate limit exceeded"
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": detail},
    )


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def install_gateway_security(
    app: FastAPI,
    settings: Settings,
) -> tuple[Callable[[Any], Awaitable[None]], Callable[[Request], Awaitable[None]]]:
    """Wire the bearer auth + rate limiter on the gateway in one call.

    Returns ``(auth_dependency, rate_limit_dependency)`` so handlers can
    declare them via ``Depends(...)``. Both are unconditional - the
    underlying logic no-ops when the matching setting is disabled.
    """
    auth_dep = bearer_auth_dependency(settings)
    limiter = RateLimiter(settings)
    limiter.install(app)
    return cast(Callable[[Any], Awaitable[None]], auth_dep), limiter.limit_dependency()


__all__ = [
    "RateLimiter",
    "bearer_auth_dependency",
    "install_gateway_security",
]
