"""Wave 3 metadata propagation: chunk_total / mime travel from request to store."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sorakai.ingest.app import create_app as create_ingest


def test_chunk_total_and_mime_in_stored_entries(run_async) -> None:
    app = create_ingest()
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents",
            json={
                "filename": "post.md",
                "content": (
                    "# Title\n\n" + ("Lots of intro content here. " * 10) + "\n\n"
                    "## Section A\n\n" + ("Body A content goes here. " * 10) + "\n\n"
                    "## Section B\n\n" + ("Body B content goes here. " * 10) + "\n"
                ),
                "chunk_size": 80,
                "chunk_overlap": 0,
                "mime_type": "text/markdown",
            },
        )
        assert r.status_code == 201, r.text
        total = r.json()["num_chunks"]
        assert total >= 1

        entries = run_async(app.state.store._read_entries())
        assert len(entries) == total
        for e in entries:
            assert e["filename"] == "post.md"
            assert e["chunk_total"] == total
            assert e["mime"] == "text/markdown"
            # Indices form a contiguous 0..N-1 sequence.
        assert sorted(e["chunk_index"] for e in entries) == list(range(total))


def test_mime_defaults_to_none_when_not_provided(run_async) -> None:
    app = create_ingest()
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents",
            json={
                "filename": "raw.txt",
                "content": "hello " * 50,
                "chunk_size": 100,
                "chunk_overlap": 0,
            },
        )
        assert r.status_code == 201, r.text
        entries = run_async(app.state.store._read_entries())
        assert entries
        assert all(e["mime"] is None for e in entries)


def test_chunker_is_language_aware_python_one_chunk(run_async) -> None:
    """Short Python file fits in one chunk and the doc-level metadata is consistent."""
    app = create_ingest()
    src = "def add(a, b):\n    '''Return a + b.'''\n    return a + b\n"
    with TestClient(app) as client:
        r = client.post(
            "/v1/documents",
            json={
                "filename": "mod.py",
                "content": src,
                "chunk_size": 500,
                "chunk_overlap": 0,
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["num_chunks"] == 1
        entries = run_async(app.state.store._read_entries())
        assert len(entries) == 1
        assert entries[0]["chunk_total"] == 1
        assert entries[0]["chunk_index"] == 0
