"""Wave 10 gateway security tests.

Covers the bearer-auth dependency, the constant-time compare, and the
slowapi-backed rate limiter via a tiny FastAPI app + TestClient.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from sorakai.common.config import Settings
from sorakai.common.security import (
    RateLimiter,
    _secure_compare,
    bearer_auth_dependency,
    install_gateway_security,
)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "cors_origins": ["*"],
        "rate_limit_per_minute": 0,
        "rate_limit_burst": 20,
        "gateway_api_key": None,
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def _auth_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    enforce = bearer_auth_dependency(settings)

    @app.get("/protected", dependencies=[Depends(enforce)])
    def protected() -> dict[str, str]:
        return {"ok": "1"}

    return app


# ---------------------------------------------------------------------------
# _secure_compare
# ---------------------------------------------------------------------------


def test_secure_compare_matches_equal_strings() -> None:
    assert _secure_compare("hunter2", "hunter2") is True


def test_secure_compare_rejects_length_mismatch() -> None:
    assert _secure_compare("hunter2", "hunter22") is False


def test_secure_compare_rejects_value_mismatch() -> None:
    assert _secure_compare("hunter2", "hunter1") is False


# ---------------------------------------------------------------------------
# bearer_auth_dependency
# ---------------------------------------------------------------------------


def test_bearer_auth_disabled_when_key_unset_allows_anonymous_calls() -> None:
    app = _auth_app(_settings(gateway_api_key=None))
    with TestClient(app) as client:
        r = client.get("/protected")
    assert r.status_code == 200
    assert r.json() == {"ok": "1"}


def test_bearer_auth_disabled_when_key_blank() -> None:
    app = _auth_app(_settings(gateway_api_key="   "))
    with TestClient(app) as client:
        r = client.get("/protected")
    assert r.status_code == 200


def test_bearer_auth_rejects_missing_header() -> None:
    app = _auth_app(_settings(gateway_api_key="hunter2"))
    with TestClient(app) as client:
        r = client.get("/protected")
    assert r.status_code == 401
    assert r.headers.get("www-authenticate", "").lower().startswith("bearer")
    assert "Missing" in r.json()["detail"]


def test_bearer_auth_rejects_wrong_key() -> None:
    app = _auth_app(_settings(gateway_api_key="hunter2"))
    with TestClient(app) as client:
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert "Invalid" in r.json()["detail"]


def test_bearer_auth_rejects_non_bearer_scheme() -> None:
    app = _auth_app(_settings(gateway_api_key="hunter2"))
    with TestClient(app) as client:
        r = client.get("/protected", headers={"Authorization": "Basic Zm9vOmJhcg=="})
    assert r.status_code == 401


def test_bearer_auth_accepts_correct_key() -> None:
    app = _auth_app(_settings(gateway_api_key="hunter2"))
    with TestClient(app) as client:
        r = client.get("/protected", headers={"Authorization": "Bearer hunter2"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


@pytest.fixture
def limiter_app() -> Iterator[tuple[FastAPI, RateLimiter]]:
    limiter = RateLimiter(_settings(rate_limit_per_minute=2))
    app = FastAPI()
    limiter.install(app)
    dep = limiter.limit_dependency()

    @app.get("/limited", dependencies=[Depends(dep)])
    def limited() -> dict[str, str]:
        return {"ok": "1"}

    yield app, limiter


def test_rate_limiter_disabled_when_per_minute_zero() -> None:
    limiter = RateLimiter(_settings(rate_limit_per_minute=0))
    assert limiter.enabled is False


def test_rate_limiter_dependency_is_noop_when_disabled() -> None:
    limiter = RateLimiter(_settings(rate_limit_per_minute=0))
    app = FastAPI()
    limiter.install(app)
    dep = limiter.limit_dependency()

    @app.get("/open", dependencies=[Depends(dep)])
    def open_route() -> dict[str, str]:
        return {"ok": "1"}

    with TestClient(app) as client:
        for _ in range(10):
            r = client.get("/open")
            assert r.status_code == 200


def test_rate_limiter_allows_up_to_budget(limiter_app: tuple[FastAPI, RateLimiter]) -> None:
    app, _ = limiter_app
    with TestClient(app) as client:
        first = client.get("/limited")
        second = client.get("/limited")
    assert first.status_code == 200
    assert second.status_code == 200


def test_rate_limiter_rejects_burst_over_budget(
    limiter_app: tuple[FastAPI, RateLimiter],
) -> None:
    app, _ = limiter_app
    with TestClient(app) as client:
        client.get("/limited")
        client.get("/limited")
        blocked = client.get("/limited")
    assert blocked.status_code == 429
    assert "Rate limit exceeded" in blocked.json()["detail"]


def test_install_gateway_security_returns_callable_dependencies() -> None:
    app = FastAPI()
    auth_dep, rl_dep = install_gateway_security(app, _settings(gateway_api_key=None, rate_limit_per_minute=0))
    assert callable(auth_dep)
    assert callable(rl_dep)
    assert hasattr(app.state, "limiter")
