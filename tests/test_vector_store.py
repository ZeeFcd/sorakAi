"""Wave 5 :class:`VectorStore` Protocol: behaviour parity across backends.

The matrix below runs every behavioural test against the three shipped
adapters:

- ``KnowledgeStoreVectorStore`` wrapping an ``InMemoryKnowledgeStore``
  (``VECTOR_STORE=memory``)
- ``KnowledgeStoreVectorStore`` wrapping a ``RedisKnowledgeStore`` backed by
  ``fakeredis`` (``VECTOR_STORE=redis``)
- ``QdrantVectorStore`` backed by ``AsyncQdrantClient(":memory:")``
  (``VECTOR_STORE=qdrant``)

Anything backend-specific (Qdrant collection lazy-create, error-on-filter
for the simple adapter, ...) lives in its own ``test_<backend>_*`` test
below the matrix.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import fakeredis.aioredis
import numpy as np
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from sorakai.common.store import InMemoryKnowledgeStore, RedisKnowledgeStore
from sorakai.core.errors import RetrievalError, StoreError
from sorakai.infra.vector_store.base import DocSummary, Hit, VectorDoc, VectorStore
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore
from sorakai.infra.vector_store.qdrant import (
    QdrantVectorStore,
    _point_id,
    build_qdrant_vector_store_from_url,
)


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _doc(doc_id: str, index: int, total: int, *, text: str, seed: int, filename: str = "f.txt") -> VectorDoc:
    return VectorDoc(
        page_content=text,
        embedding=_vec(seed),
        metadata={
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": index,
            "chunk_total": total,
            "mime": "text/plain",
        },
    )


@pytest_asyncio.fixture(params=["memory", "redis", "qdrant"])
async def vstore(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    if request.param == "memory":
        store: VectorStore = KnowledgeStoreVectorStore(InMemoryKnowledgeStore())
        yield store
        await store.aclose()
        return
    if request.param == "redis":
        fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
        with patch("redis.asyncio.from_url", return_value=fake):
            redis_store = RedisKnowledgeStore("redis://localhost:6379/0")
        store = KnowledgeStoreVectorStore(redis_store)
        yield store
        await store.aclose()
        return
    if request.param == "qdrant":
        # Each test gets its own client + collection to avoid cross-test leakage.
        client = AsyncQdrantClient(":memory:")
        qstore = QdrantVectorStore(client=client, collection=f"sorakai_test_{id(request)}")
        yield qstore
        await qstore.aclose()
        return
    raise AssertionError(f"unknown backend: {request.param!r}")


# ---- Behavioural matrix -----------------------------------------------------


async def test_empty_store_returns_no_docs(vstore: VectorStore) -> None:
    assert await vstore.list_docs() == []


async def test_empty_search_returns_no_hits(vstore: VectorStore) -> None:
    assert await vstore.search(_vec(0), k=3) == []


async def test_upsert_then_list(vstore: VectorStore) -> None:
    await vstore.upsert(
        [
            _doc("d1", 0, 2, text="alpha", seed=1, filename="a.txt"),
            _doc("d1", 1, 2, text="bravo", seed=2, filename="a.txt"),
            _doc("d2", 0, 1, text="charlie", seed=3, filename="b.md"),
        ]
    )
    summaries = {s.doc_id: s for s in await vstore.list_docs()}
    assert set(summaries) == {"d1", "d2"}
    assert summaries["d1"].filename == "a.txt"
    assert summaries["d1"].chunk_count == 2
    assert summaries["d2"].chunk_count == 1


async def test_reingest_same_doc_id_overwrites(vstore: VectorStore) -> None:
    await vstore.upsert(
        [
            _doc("d", 0, 2, text="v1.a", seed=1),
            _doc("d", 1, 2, text="v1.b", seed=2),
        ]
    )
    await vstore.upsert([_doc("d", 0, 1, text="v2.a", seed=3)])
    summaries = await vstore.list_docs()
    assert len(summaries) == 1
    assert summaries[0].chunk_count == 1


async def test_delete_doc_returns_chunk_count(vstore: VectorStore) -> None:
    await vstore.upsert(
        [
            _doc("d1", 0, 2, text="x", seed=1),
            _doc("d1", 1, 2, text="y", seed=2),
            _doc("d2", 0, 1, text="z", seed=3),
        ]
    )
    removed = await vstore.delete_doc("d1")
    assert removed == 2
    remaining = await vstore.list_docs()
    assert [s.doc_id for s in remaining] == ["d2"]


async def test_delete_unknown_doc_returns_zero(vstore: VectorStore) -> None:
    await vstore.upsert([_doc("d1", 0, 1, text="x", seed=1)])
    assert await vstore.delete_doc("never-existed") == 0


async def test_search_ranks_by_cosine(vstore: VectorStore) -> None:
    """The chunk whose embedding equals the query vector must be top-1."""
    query = _vec(42)
    docs = [
        VectorDoc(
            page_content=f"chunk-{i}",
            embedding=_vec(i),
            metadata={
                "doc_id": "d",
                "filename": "f.txt",
                "chunk_index": i,
                "chunk_total": 6,
                "mime": None,
            },
        )
        for i in range(5)
    ]
    docs.append(
        VectorDoc(
            page_content="exact-match",
            embedding=query,
            metadata={
                "doc_id": "d",
                "filename": "f.txt",
                "chunk_index": 99,
                "chunk_total": 6,
                "mime": None,
            },
        )
    )
    await vstore.upsert(docs)
    hits = await vstore.search(query, k=3)
    assert hits, "search must return at least one hit"
    assert hits[0].page_content == "exact-match"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    # Hits must be sorted by descending score.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_search_returns_metadata_round_trip(vstore: VectorStore) -> None:
    await vstore.upsert([_doc("d", 0, 1, text="hello", seed=7, filename="hello.txt")])
    hits = await vstore.search(_vec(7), k=1)
    assert len(hits) == 1
    h = hits[0]
    assert isinstance(h, Hit)
    assert h.page_content == "hello"
    assert h.metadata["doc_id"] == "d"
    assert h.metadata["filename"] == "hello.txt"
    assert h.metadata["chunk_index"] == 0
    assert h.metadata["chunk_total"] == 1


async def test_ping(vstore: VectorStore) -> None:
    assert await vstore.ping() is True


# ---- KnowledgeStoreVectorStore-specific -------------------------------------


async def test_knowledge_store_adapter_refuses_filters() -> None:
    """Filters are server-side only; the simple adapter must fail loud."""
    store = KnowledgeStoreVectorStore(InMemoryKnowledgeStore())
    await store.upsert([_doc("d", 0, 1, text="x", seed=1)])
    with pytest.raises(RetrievalError, match="qdrant"):
        await store.search(_vec(1), k=1, filters={"doc_id": "d"})


async def test_knowledge_store_adapter_rejects_invalid_metadata() -> None:
    store = KnowledgeStoreVectorStore(InMemoryKnowledgeStore())
    bad = VectorDoc(page_content="x", embedding=_vec(1), metadata={})
    with pytest.raises(StoreError, match="doc_id"):
        await store.upsert([bad])


# ---- QdrantVectorStore-specific ---------------------------------------------


async def test_qdrant_creates_collection_on_first_upsert() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantVectorStore(client=client, collection="late_create_kb")
    assert await client.collection_exists("late_create_kb") is False
    await store.upsert([_doc("d", 0, 1, text="x", seed=1)])
    assert await client.collection_exists("late_create_kb") is True
    await store.aclose()


async def test_qdrant_supports_doc_id_filter() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantVectorStore(client=client, collection="filter_kb")
    await store.upsert(
        [
            _doc("d1", 0, 1, text="alpha", seed=1, filename="a.txt"),
            _doc("d2", 0, 1, text="bravo", seed=2, filename="b.txt"),
        ]
    )
    hits = await store.search(_vec(2), k=2, filters={"doc_id": "d2"})
    assert hits, "filtered search must still return hits"
    assert all(h.metadata["doc_id"] == "d2" for h in hits)
    await store.aclose()


async def test_qdrant_search_on_missing_collection_returns_empty() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantVectorStore(client=client, collection="never_created_kb")
    assert await store.search(_vec(1), k=5) == []
    assert await store.list_docs() == []
    assert await store.delete_doc("d") == 0
    await store.aclose()


def test_qdrant_point_ids_are_deterministic() -> None:
    assert _point_id("doc-a", 0) == _point_id("doc-a", 0)
    assert _point_id("doc-a", 0) != _point_id("doc-a", 1)
    assert _point_id("doc-a", 0) != _point_id("doc-b", 0)


def test_qdrant_factory_from_memory_url() -> None:
    store = build_qdrant_vector_store_from_url(":memory:", "kb")
    assert isinstance(store, QdrantVectorStore)


# ---- Factory ----------------------------------------------------------------


def test_factory_picks_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorakai.common.config import get_settings
    from sorakai.infra.vector_store import get_vector_store

    monkeypatch.setenv("VECTOR_STORE", "memory")
    get_settings.cache_clear()
    store = get_vector_store(get_settings())
    assert isinstance(store, KnowledgeStoreVectorStore)
    assert isinstance(store.knowledge_store, InMemoryKnowledgeStore)


def test_factory_picks_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorakai.common.config import get_settings
    from sorakai.infra.vector_store import get_vector_store

    monkeypatch.setenv("VECTOR_STORE", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    get_settings.cache_clear()
    with patch("redis.asyncio.from_url", return_value=fakeredis.aioredis.FakeRedis(decode_responses=True)):
        store = get_vector_store(get_settings())
    assert isinstance(store, KnowledgeStoreVectorStore)
    assert isinstance(store.knowledge_store, RedisKnowledgeStore)


def test_factory_redis_without_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorakai.common.config import get_settings
    from sorakai.core.errors import ConfigError
    from sorakai.infra.vector_store import get_vector_store

    monkeypatch.setenv("VECTOR_STORE", "redis")
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="REDIS_URL"):
        get_vector_store(get_settings())


def test_factory_qdrant_with_memory_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorakai.common.config import get_settings
    from sorakai.infra.vector_store import get_vector_store

    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION", "factory_test_kb")
    get_settings.cache_clear()
    store = get_vector_store(get_settings())
    assert isinstance(store, QdrantVectorStore)


def test_factory_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorakai.common.config import Settings
    from sorakai.core.errors import ConfigError
    from sorakai.infra.vector_store import get_vector_store

    # Bypass Pydantic literal validation by constructing Settings then
    # poking the field directly - simulates a future backend name that
    # made it past validation but lacks a builder.
    settings = Settings()
    object.__setattr__(settings, "vector_store", "milvus")
    with pytest.raises(ConfigError, match="milvus"):
        get_vector_store(settings)


def test_register_vector_store_extends_registry() -> None:
    from sorakai.infra.vector_store.factory import (
        VECTOR_STORE_REGISTRY,
        register_vector_store,
    )

    class _StubStore:  # minimal Protocol-compatible stub
        async def upsert(self, docs: list[VectorDoc]) -> None:
            return None

        async def delete_doc(self, doc_id: str) -> int:
            return 0

        async def list_docs(self) -> list[DocSummary]:
            return []

        async def search(self, query_vec: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[Hit]:
            return []

        async def ping(self) -> bool:
            return True

        async def aclose(self) -> None:
            return None

    register_vector_store("stub_test_only", lambda _s: _StubStore())
    try:
        assert "stub_test_only" in VECTOR_STORE_REGISTRY
    finally:
        VECTOR_STORE_REGISTRY.pop("stub_test_only", None)


# ---- Smoke: VectorStore Protocol runtime check ------------------------------


def test_all_adapters_are_runtime_checkable_against_protocol() -> None:
    """``isinstance`` against the Protocol must pass for every shipped adapter."""
    mem_store = KnowledgeStoreVectorStore(InMemoryKnowledgeStore())
    client = AsyncQdrantClient(":memory:")
    qstore = QdrantVectorStore(client=client, collection="proto_check")
    try:
        assert isinstance(mem_store, VectorStore)
        assert isinstance(qstore, VectorStore)
    finally:
        # Don't leak the in-memory qdrant client between modules.
        sys.modules.pop("qdrant_client.local.qdrant_local", None)
