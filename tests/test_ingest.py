from fastapi.testclient import TestClient

from sorakai.ingest.app import create_app


def test_ingest_health_and_ready():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["service"] == "ingest"
        assert client.get("/ready").json()["ready"] is True


def test_ingest_document():
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents",
            json={"filename": "x.py", "content": "def foo():\n    return 42\n" * 5, "chunk_size": 50},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["num_chunks"] >= 1
        assert "x.py" in data["message"]
