"""Wave 4 ingest endpoints: ``GET /v1/documents`` + ``DELETE /v1/documents/{id}``.

Includes the re-ingest invariant (POSTing the same ``document_id`` twice yields
the second version, never both). The dim-guard interaction with ``replace_kb``
gets its own dedicated test below to lock the post-Wave-4 ordering in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from sorakai.infra.vector_store.base import VectorStore
from sorakai.ingest.app import create_app


def _post(client: TestClient, **body: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunk_size": 100,
        "chunk_overlap": 0,
    }
    payload.update(body)
    r = client.post("/v1/documents", json=payload)
    assert r.status_code == 201, r.text
    return dict(r.json())


def test_list_documents_empty() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert body == {"documents": [], "total": 0}


def test_list_documents_after_ingest() -> None:
    app = create_app()
    with TestClient(app) as c:
        _post(c, filename="a.txt", content="hello " * 20, document_id="doc-a")
        _post(
            c, filename="b.md", content="# title\n\n" + ("para " * 20), document_id="doc-b", mime_type="text/markdown"
        )
        body = c.get("/v1/documents").json()
        assert body["total"] == 2
        by_id = {d["doc_id"]: d for d in body["documents"]}
        assert set(by_id) == {"doc-a", "doc-b"}
        assert by_id["doc-a"]["filename"] == "a.txt"
        assert by_id["doc-b"]["filename"] == "b.md"
        assert by_id["doc-b"]["mime"] == "text/markdown"
        assert by_id["doc-a"]["chunk_count"] >= 1


def test_ingest_writes_configured_vector_store(run_async) -> None:
    app = create_app()
    with TestClient(app) as c:
        _post(c, filename="vector.txt", content="vector searchable content " * 20, document_id="doc-vector")

        async def _assert_vector_store_written() -> None:
            vector_store: VectorStore = app.state.vector_store
            summaries = await vector_store.list_docs()
            assert [s.doc_id for s in summaries] == ["doc-vector"]
            hits = await vector_store.search(np.ones(256, dtype=np.float32), k=1)
            assert hits
            assert hits[0].metadata["doc_id"] == "doc-vector"

        run_async(_assert_vector_store_written())


def test_reingest_same_doc_id_overwrites() -> None:
    app = create_app()
    with TestClient(app) as c:
        r1 = _post(c, filename="x.txt", content="version one " * 30, document_id="doc-x")
        _post(c, filename="x.txt", content="version two ", document_id="doc-x")
        body = c.get("/v1/documents").json()
        assert body["total"] == 1
        only = body["documents"][0]
        assert only["doc_id"] == "doc-x"
        assert only["chunk_count"] < int(r1["num_chunks"])


def test_delete_document_endpoint(run_async) -> None:
    app = create_app()
    with TestClient(app) as c:
        _post(c, filename="a.txt", content="aaa " * 30, document_id="doc-a")
        _post(c, filename="b.txt", content="bbb " * 30, document_id="doc-b")
        r = c.delete("/v1/documents/doc-a")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["doc_id"] == "doc-a"
        assert body["removed_chunks"] >= 1
        remaining = c.get("/v1/documents").json()
        assert remaining["total"] == 1
        assert remaining["documents"][0]["doc_id"] == "doc-b"

        async def _assert_vector_store_deleted() -> None:
            vector_store: VectorStore = app.state.vector_store
            summaries = await vector_store.list_docs()
            assert {s.doc_id for s in summaries} == {"doc-b"}

        run_async(_assert_vector_store_deleted())


def test_delete_unknown_doc_id_returns_404() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.delete("/v1/documents/never-existed")
        assert r.status_code == 404
        assert "never-existed" in r.json()["detail"]


def test_replace_kb_true_drops_prior_documents() -> None:
    app = create_app()
    with TestClient(app) as c:
        _post(c, filename="a.txt", content="aaa " * 20, document_id="doc-a")
        _post(c, filename="b.txt", content="bbb " * 20, document_id="doc-b")
        _post(c, filename="c.txt", content="ccc " * 20, document_id="doc-c", replace_kb=True)
        body = c.get("/v1/documents").json()
        assert body["total"] == 1
        assert body["documents"][0]["doc_id"] == "doc-c"
