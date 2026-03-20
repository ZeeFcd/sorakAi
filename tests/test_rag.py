import numpy as np
from fastapi.testclient import TestClient

from sorakai.rag.app import create_app
from tests.conftest import run_async


def test_rag_health_ready():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["service"] == "rag"
        assert client.get("/ready").json()["ready"] is True


def test_rag_query_after_seed():
    app = create_app()
    with TestClient(app) as client:
        run_async(
            client.app.state.store.save(
                ["def foo():\n    return 42\n"],
                [np.array([1.0, 2.0, 3.0], dtype=float)],
            )
        )
        r = client.post("/v1/query", json={"question": "what is foo"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "context_preview" in body


def test_rag_query_empty_kb():
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/v1/query", json={"question": "anything"})
        assert r.status_code == 404
