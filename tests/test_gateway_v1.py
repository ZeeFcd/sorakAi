"""Wave 10 gateway tests: canonical /v1 surface + legacy /api/v1 redirects.

The pre-existing ``tests/test_gateway.py`` covers the JSON proxy
contracts via ``/api/v1/*``; this file pins:

- the new canonical paths under ``/v1/*`` are wired and proxy correctly,
- ``/api/v1/*`` still resolves (via the 308 redirect),
- bearer auth gates the canonical paths when ``GATEWAY_API_KEY`` is set,
- rate-limit responses surface as 429 (when enabled),
- the request-size limit middleware engages on oversized bodies.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sorakai.common.config import get_settings
from sorakai.gateway.app import create_app


@pytest.fixture
def gateway_app(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    get_settings.cache_clear()
    return create_app()


# ---------------------------------------------------------------------------
# /v1 canonical paths
# ---------------------------------------------------------------------------


@respx.mock
def test_v1_query_proxies_to_rag(gateway_app):
    respx.post("http://rag.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={"answer": "42", "context_preview": "...", "sources_used": 1, "session_id": None},
        )
    )
    with TestClient(gateway_app) as client:
        r = client.post("/v1/query", json={"question": "what?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "42"


@respx.mock
def test_v1_documents_list_proxies_to_ingest(gateway_app):
    respx.get("http://ingest.test/v1/documents").mock(
        return_value=httpx.Response(200, json={"documents": [], "total": 0})
    )
    with TestClient(gateway_app) as client:
        r = client.get("/v1/documents")
    assert r.status_code == 200
    assert r.json()["total"] == 0


# ---------------------------------------------------------------------------
# /api/v1 -> /v1 redirects
# ---------------------------------------------------------------------------


@respx.mock
def test_legacy_api_v1_get_redirects_with_308(gateway_app):
    respx.get("http://ingest.test/v1/documents").mock(
        return_value=httpx.Response(200, json={"documents": [], "total": 0})
    )
    with TestClient(gateway_app) as client:
        # follow_redirects=False so we can inspect the 308 status.
        r = client.get("/api/v1/documents", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/v1/documents")


@respx.mock
def test_legacy_api_v1_post_redirects_and_clients_follow(gateway_app):
    respx.post("http://rag.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={"answer": "42", "context_preview": "...", "sources_used": 1, "session_id": None},
        )
    )
    with TestClient(gateway_app) as client:
        # Default follow_redirects=True; 308 preserves the POST body.
        r = client.post("/api/v1/query", json={"question": "?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "42"


@respx.mock
def test_legacy_api_v1_preserves_query_string(gateway_app):
    with TestClient(gateway_app) as client:
        r = client.get("/api/v1/anything?foo=bar&baz=qux", follow_redirects=False)
    assert r.status_code == 308
    assert "foo=bar" in r.headers["location"]
    assert "baz=qux" in r.headers["location"]


# ---------------------------------------------------------------------------
# Bearer auth
# ---------------------------------------------------------------------------


@respx.mock
def test_v1_query_rejects_when_key_required_and_missing(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    monkeypatch.setenv("GATEWAY_API_KEY", "hunter2")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/v1/query", json={"question": "?"})
    assert r.status_code == 401


@respx.mock
def test_v1_query_accepts_when_bearer_matches(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    monkeypatch.setenv("GATEWAY_API_KEY", "hunter2")
    get_settings.cache_clear()
    respx.post("http://rag.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={"answer": "42", "context_preview": "...", "sources_used": 1, "session_id": None},
        )
    )
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/v1/query",
            json={"question": "?"},
            headers={"Authorization": "Bearer hunter2"},
        )
    assert r.status_code == 200


@respx.mock
def test_health_and_ready_remain_unauthenticated(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    monkeypatch.setenv("GATEWAY_API_KEY", "hunter2")
    get_settings.cache_clear()
    respx.get("http://ingest.test/health").mock(return_value=httpx.Response(200, json={"service": "ingest"}))
    respx.get("http://rag.test/health").mock(return_value=httpx.Response(200, json={"service": "rag"}))
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200


# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------


def test_oversized_v1_query_returns_413(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    monkeypatch.setenv("REQUEST_MAX_BYTES", "32")
    get_settings.cache_clear()
    app = create_app()
    big = "x" * 1024
    with TestClient(app) as client:
        r = client.post(
            "/v1/query",
            content=f'{{"question": "{big}"}}',
            headers={"content-type": "application/json"},
        )
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


@respx.mock
def test_rate_limit_engages_after_budget(monkeypatch):
    monkeypatch.setenv("INGEST_SERVICE_URL", "http://ingest.test")
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("RATE_LIMIT_BURST", "2")
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()
    respx.post("http://rag.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={"answer": "ok", "context_preview": "", "sources_used": 0, "session_id": None},
        )
    )
    app = create_app()
    with TestClient(app) as client:
        first = client.post("/v1/query", json={"question": "?"})
        second = client.post("/v1/query", json={"question": "?"})
        third = client.post("/v1/query", json={"question": "?"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
