from fastapi.testclient import TestClient

from sorakai.rag.app import create_app


def test_rag_health_ready():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["service"] == "rag"
        assert client.get("/ready").json()["ready"] is True


def test_rag_query_after_seed(seed_kb):
    app = create_app()
    with TestClient(app) as client:
        seed_kb(app, ["def foo():\n    return 42\n"])
        r = client.post("/v1/query", json={"question": "what is foo"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "answer" in body
        assert "context_preview" in body


def test_rag_query_empty_kb():
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/v1/query", json={"question": "anything"})
        assert r.status_code == 404
