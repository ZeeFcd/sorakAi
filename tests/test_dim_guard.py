"""End-to-end Wave 2 dim-guard behaviour through the ingest + RAG handlers.

These tests run with the autouse ``EMBEDDING_PROVIDER=char`` from
``conftest`` so we don't need Ollama running; switching providers is
simulated by flipping ``OLLAMA_EMBEDDING_MODEL`` (which is what the guard
metadata records together with ``provider`` and ``dim``).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from sorakai.common.config import get_settings
from sorakai.common.kb_meta import KBMeta
from sorakai.ingest.app import create_app as create_ingest
from sorakai.rag.app import create_app as create_rag


def test_first_ingest_writes_kb_meta(run_async, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-A")
    app = create_ingest()
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents",
            json={"filename": "x.txt", "content": "hello world " * 5, "chunk_size": 50},
        )
        assert r.status_code == 201, r.text
        meta = run_async(app.state.kb_meta.read())
        assert meta is not None
        assert meta.model == "model-A"
        assert meta.dim > 0


def test_second_ingest_with_different_model_returns_409(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-A")
    app = create_ingest()
    with TestClient(app) as client:
        r1 = client.post(
            "/v1/documents",
            json={"filename": "a.txt", "content": "lorem ipsum " * 5, "chunk_size": 50},
        )
        assert r1.status_code == 201, r1.text

        # Force a "new provider/model" without restarting the app.
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-B")
        get_settings.cache_clear()

        r2 = client.post(
            "/v1/documents",
            json={"filename": "b.txt", "content": "dolor sit amet " * 5, "chunk_size": 50},
        )
        assert r2.status_code == 409, r2.text
        body = r2.json()["detail"]
        assert body["error"] == "embedding_metadata_mismatch"
        assert body["expected"]["model"] == "model-A"
        assert body["actual"]["model"] == "model-B"


def test_replace_kb_true_resets_meta_to_new_model(run_async, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-A")
    app = create_ingest()
    with TestClient(app) as client:
        r1 = client.post(
            "/v1/documents",
            json={"filename": "a.txt", "content": "first " * 10, "chunk_size": 50},
        )
        assert r1.status_code == 201, r1.text

        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-B")
        get_settings.cache_clear()

        r2 = client.post(
            "/v1/documents",
            json={
                "filename": "b.txt",
                "content": "second " * 10,
                "chunk_size": 50,
                "replace_kb": True,
            },
        )
        assert r2.status_code == 201, r2.text
        meta = run_async(app.state.kb_meta.read())
        assert meta is not None
        assert meta.model == "model-B"


def test_rag_query_with_mismatched_meta_returns_409(run_async, monkeypatch, seed_kb) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-A")
    app = create_rag()
    with TestClient(app) as client:
        seed_kb(app, ["Paris is the capital of France."])

        # Manually overwrite stored meta to simulate the KB having been built
        # by a different model than the live query path will use.
        run_async(app.state.kb_meta.write(KBMeta(provider="char", model="other-model", dim=256)))

        r = client.post("/v1/query", json={"question": "What is the capital?"})
        assert r.status_code == 409, r.text
        body = r.json()["detail"]
        assert body["error"] == "embedding_metadata_mismatch"
        assert body["expected"]["model"] == "other-model"
