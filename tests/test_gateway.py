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


@respx.mock
def test_gateway_root_and_health(gateway_app):
    with TestClient(gateway_app) as client:
        assert client.get("/health").json()["service"] == "gateway"
        root = client.get("/").json()
        assert root["service"] == "sorakAi-gateway"


@respx.mock
def test_gateway_proxy_ingest(gateway_app):
    respx.post("http://ingest.test/v1/documents").mock(
        return_value=httpx.Response(
            201,
            json={
                "message": "ok",
                "num_chunks": 2,
                "filename": "a.py",
            },
        )
    )
    with TestClient(gateway_app) as client:
        r = client.post("/api/v1/documents", json={"filename": "a.py", "content": "hello world " * 20})
        assert r.status_code == 201
        assert r.json()["num_chunks"] == 2


@respx.mock
def test_gateway_proxy_query(gateway_app):
    respx.post("http://rag.test/v1/query").mock(
        return_value=httpx.Response(
            200,
            json={"answer": "42", "context_preview": "def foo"},
        )
    )
    with TestClient(gateway_app) as client:
        r = client.post("/api/v1/query", json={"question": "?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "42"


@respx.mock
def test_gateway_ready_upstream_unhealthy(gateway_app):
    respx.get("http://ingest.test/health").mock(return_value=httpx.Response(503))
    respx.get("http://rag.test/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "service": "rag"})
    )
    with TestClient(gateway_app) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is False


@respx.mock
def test_gateway_ready_all_ok(gateway_app):
    respx.get("http://ingest.test/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "service": "ingest"})
    )
    respx.get("http://rag.test/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "service": "rag"})
    )
    with TestClient(gateway_app) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True
