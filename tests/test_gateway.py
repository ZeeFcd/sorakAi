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
                "document_id": "00000000-0000-0000-0000-000000000001",
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
            json={
                "answer": "42",
                "context_preview": "def foo",
                "sources_used": 1,
                "session_id": None,
            },
        )
    )
    with TestClient(gateway_app) as client:
        r = client.post("/api/v1/query", json={"question": "?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "42"


@respx.mock
def test_gateway_proxy_agent(gateway_app):
    respx.post("http://rag.test/v1/agent").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Pyramids are in Egypt.",
                "sources_used": 2,
                "session_id": "user-7",
                "route": "kb",
                "steps_used": 1,
                "trace": ["route", "retrieve", "grade", "generate", "critique"],
                "tool_calls": [
                    {
                        "name": "kb_search",
                        "input": {"query": "pyramids", "k": 5},
                        "output_summary": "2 item(s)",
                        "duration_ms": 12.3,
                        "error": None,
                    }
                ],
            },
        )
    )
    with TestClient(gateway_app) as client:
        r = client.post("/api/v1/agent", json={"question": "pyramids", "session_id": "user-7"})
        assert r.status_code == 200
        body = r.json()
        assert body["route"] == "kb"
        assert body["tool_calls"][0]["name"] == "kb_search"


@respx.mock
def test_gateway_proxy_agent_stream_forwards_bytes(gateway_app):
    """Streaming proxy must pipe upstream SSE frames through untouched."""
    sse_body = b'event: open\ndata: {"type":"node","node":"route"}\n\nevent: done\ndata: {"type":"done"}\n\n'
    respx.post("http://rag.test/v1/agent/stream").mock(
        return_value=httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )
    )
    with (
        TestClient(gateway_app) as client,
        client.stream("POST", "/api/v1/agent/stream", json={"question": "x"}) as r,
    ):
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes())
    assert b'data: {"type":"node","node":"route"}' in body
    assert b"event: done" in body


@respx.mock
def test_gateway_proxy_query_stream_forwards_bytes(gateway_app):
    sse_body = b'data: {"type":"token","text":"hi"}\n\nevent: done\ndata: {"type":"done"}\n\n'
    respx.post("http://rag.test/v1/query/stream").mock(
        return_value=httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )
    )
    with (
        TestClient(gateway_app) as client,
        client.stream("POST", "/api/v1/query/stream", json={"question": "?"}) as r,
    ):
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert b"event: done" in body


@respx.mock
def test_gateway_proxy_list_documents(gateway_app):
    respx.get("http://ingest.test/v1/documents").mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [
                    {"doc_id": "doc-1", "filename": "a.txt", "chunk_count": 3, "mime": None},
                    {"doc_id": "doc-2", "filename": "b.md", "chunk_count": 1, "mime": "text/markdown"},
                ],
                "total": 2,
            },
        )
    )
    with TestClient(gateway_app) as client:
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert {d["doc_id"] for d in body["documents"]} == {"doc-1", "doc-2"}


@respx.mock
def test_gateway_proxy_delete_document(gateway_app):
    respx.delete("http://ingest.test/v1/documents/doc-1").mock(
        return_value=httpx.Response(
            200,
            json={"doc_id": "doc-1", "removed_chunks": 3, "message": "Removed 3 chunks for document 'doc-1'"},
        )
    )
    with TestClient(gateway_app) as client:
        r = client.delete("/api/v1/documents/doc-1")
        assert r.status_code == 200
        assert r.json()["removed_chunks"] == 3


@respx.mock
def test_gateway_proxy_delete_propagates_404(gateway_app):
    respx.delete("http://ingest.test/v1/documents/missing").mock(
        return_value=httpx.Response(404, json={"detail": "No document with doc_id='missing'"})
    )
    with TestClient(gateway_app) as client:
        r = client.delete("/api/v1/documents/missing")
        assert r.status_code == 404
        assert "missing" in r.json()["detail"]


@respx.mock
def test_gateway_ready_upstream_unhealthy(gateway_app):
    respx.get("http://ingest.test/health").mock(return_value=httpx.Response(503))
    respx.get("http://rag.test/health").mock(return_value=httpx.Response(200, json={"status": "ok", "service": "rag"}))
    with TestClient(gateway_app) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is False


@respx.mock
def test_gateway_ready_all_ok(gateway_app):
    respx.get("http://ingest.test/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "service": "ingest"})
    )
    respx.get("http://rag.test/health").mock(return_value=httpx.Response(200, json={"status": "ok", "service": "rag"}))
    with TestClient(gateway_app) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True
