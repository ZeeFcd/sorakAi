"""Wave 10 shared HTTP middleware tests.

The unit under test is :mod:`sorakai.common.middleware`. Every helper is
verified through a tiny FastAPI app + TestClient so we exercise the
actual ASGI stack rather than mocking Starlette internals.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from sorakai.common.config import Settings, get_settings
from sorakai.common.middleware import (
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    install_common_middleware,
    install_cors,
    install_exception_handler,
    install_request_id,
    install_request_size_limit,
)


def _settings(**overrides: object) -> Settings:
    """Build a fresh Settings reading nothing from disk."""
    defaults: dict[str, object] = {
        "cors_origins": ["*"],
        "request_max_bytes": 1024,
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def _ping_app(settings: Settings | None = None, *, service: str = "ingest") -> FastAPI:
    """Minimal app with one POST + one GET for middleware exercises."""
    app = FastAPI()
    install_common_middleware(app, settings or _settings(), service=service)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"pong": "1"}

    @app.post("/echo")
    def echo(body: dict[str, str]) -> dict[str, str]:
        return body

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("synthetic failure")

    return app


def test_install_cors_emits_allow_origin_for_wildcard() -> None:
    app = FastAPI()
    install_cors(app, _settings(cors_origins=["*"]))

    @app.get("/x")
    def x() -> dict[str, str]:
        return {"x": "1"}

    with TestClient(app) as client:
        r = client.options(
            "/x",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_install_cors_locks_to_explicit_origin_with_credentials() -> None:
    app = FastAPI()
    install_cors(app, _settings(cors_origins=["https://only.me"]))

    @app.get("/x")
    def x() -> dict[str, str]:
        return {"x": "1"}

    with TestClient(app) as client:
        r = client.options(
            "/x",
            headers={
                "Origin": "https://only.me",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://only.me"
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_install_request_id_returns_minted_id_on_response() -> None:
    app = FastAPI()
    install_request_id(app)

    @app.get("/x")
    def x() -> dict[str, str]:
        return {"x": "1"}

    with TestClient(app) as client:
        r = client.get("/x")
    assert r.status_code == 200
    minted = r.headers.get(REQUEST_ID_HEADER)
    assert minted and len(minted) >= 8
    assert r.headers.get(PROCESS_TIME_HEADER, "").endswith("ms")


def test_install_request_id_preserves_inbound_header() -> None:
    app = FastAPI()
    install_request_id(app)

    @app.get("/x")
    def x() -> dict[str, str]:
        return {"x": "1"}

    with TestClient(app) as client:
        r = client.get("/x", headers={REQUEST_ID_HEADER: "rid-from-edge"})
    assert r.headers[REQUEST_ID_HEADER] == "rid-from-edge"


def test_install_exception_handler_returns_generic_500() -> None:
    app = _ping_app(service="rag")
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}


def test_install_exception_handler_does_not_swallow_http_exception() -> None:
    """HTTP errors raised by routes should still flow through unchanged."""
    app = FastAPI()
    install_exception_handler(app, service="rag")

    @app.get("/forbidden")
    def forbidden() -> None:
        raise HTTPException(status_code=403, detail="nope")

    with TestClient(app) as client:
        r = client.get("/forbidden")
    assert r.status_code == 403
    assert r.json() == {"detail": "nope"}


def test_request_size_limit_blocks_oversized_body() -> None:
    app = _ping_app(_settings(request_max_bytes=32))
    big_body = "x" * 200
    with TestClient(app) as client:
        r = client.post(
            "/echo",
            content=f'{{"k": "{big_body}"}}',
            headers={"content-type": "application/json"},
        )
    assert r.status_code == 413
    assert "Request body too large" in r.json()["detail"]


def test_request_size_limit_allows_within_budget() -> None:
    app = _ping_app(_settings(request_max_bytes=1024))
    with TestClient(app) as client:
        r = client.post("/echo", json={"k": "tiny"})
    assert r.status_code == 200
    assert r.json() == {"k": "tiny"}


def test_request_size_limit_skips_get_methods() -> None:
    app = _ping_app(_settings(request_max_bytes=8))
    with TestClient(app) as client:
        r = client.get("/ping", headers={"content-length": "1024"})
    assert r.status_code == 200


def test_request_size_limit_disabled_when_max_bytes_zero() -> None:
    app = _ping_app(_settings(request_max_bytes=0))
    with TestClient(app) as client:
        r = client.post("/echo", json={"k": "x" * 4096})
    assert r.status_code == 200


def test_install_request_size_limit_uses_settings_value() -> None:
    """Sanity-check the helper threads settings.request_max_bytes through."""
    app = FastAPI()
    install_request_size_limit(app, _settings(request_max_bytes=16))

    @app.post("/echo")
    def echo(body: dict[str, str]) -> dict[str, str]:
        return body

    with TestClient(app) as client:
        r = client.post("/echo", json={"k": "x" * 200})
    assert r.status_code == 413


def test_install_common_middleware_wires_all_layers(monkeypatch: object) -> None:
    """A single helper call wires CORS + size limit + request-id + 500 handler."""
    get_settings.cache_clear()
    app = _ping_app(_settings(cors_origins=["*"], request_max_bytes=64))

    with TestClient(app, raise_server_exceptions=False) as client:
        r_ok = client.get("/ping")
        r_oversize = client.post("/echo", json={"k": "x" * 200})
        r_boom = client.get("/boom")

    assert r_ok.status_code == 200
    assert REQUEST_ID_HEADER in r_ok.headers
    assert r_oversize.status_code == 413
    assert r_boom.status_code == 500
