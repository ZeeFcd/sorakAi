"""Wave 4 store behaviour: composite keys, re-ingest, list, delete.

The matrix below runs every behavioural test against both backends:

- :class:`~sorakai.common.store.InMemoryKnowledgeStore` (the default in CI)
- :class:`~sorakai.common.store.RedisKnowledgeStore` backed by ``fakeredis``
  so we get the real pipeline / HSCAN code path without a Redis container.

Anything that's specific to the Redis pipeline (atomic ``replace_kb``,
``ck:<doc_id>:<chunk_index>`` field layout) lives in its own ``test_redis_*``
test below the matrix.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import fakeredis.aioredis
import numpy as np
import pytest

from sorakai.common.store import (
    CHUNK_FIELD_PREFIX,
    KB_CHUNKS_HASH_KEY,
    InMemoryKnowledgeStore,
    KnowledgeStore,
    RedisKnowledgeStore,
    create_store,
)


def _vec(seed: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


@pytest.fixture(params=["memory", "redis"], ids=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[KnowledgeStore]:
    if request.param == "memory":
        yield InMemoryKnowledgeStore()
        return
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.from_url", return_value=fake):
        yield RedisKnowledgeStore("redis://localhost:6379/0")


# ---- list_documents ----------------------------------------------------------


async def test_list_documents_empty(store: KnowledgeStore) -> None:
    assert await store.list_documents() == []


async def test_list_documents_groups_by_doc_id(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["t1", "t2"], [_vec(1), _vec(2)], mime_type="text/plain")
    await store.append_document("d2", "b.md", ["m1"], [_vec(3)], mime_type="text/markdown")
    summaries = await store.list_documents()
    by_id = {s.doc_id: s for s in summaries}
    assert set(by_id) == {"d1", "d2"}
    assert by_id["d1"].filename == "a.txt"
    assert by_id["d1"].chunk_count == 2
    assert by_id["d1"].mime == "text/plain"
    assert by_id["d2"].chunk_count == 1
    assert by_id["d2"].mime == "text/markdown"


# ---- re-ingest is idempotent ------------------------------------------------


async def test_reingest_same_doc_id_does_not_duplicate(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["v1.a", "v1.b"], [_vec(1), _vec(2)])
    await store.append_document("d1", "a.txt", ["v2.a"], [_vec(3)])
    flat = await store.load_flat()
    assert flat is not None
    chunks, _ = flat
    assert chunks == ["v2.a"]
    summaries = await store.list_documents()
    assert len(summaries) == 1
    assert summaries[0].chunk_count == 1


async def test_reingest_other_docs_untouched(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["a1"], [_vec(1)])
    await store.append_document("d2", "b.txt", ["b1", "b2"], [_vec(2), _vec(3)])
    await store.append_document("d1", "a.txt", ["a1-v2", "a2-v2"], [_vec(4), _vec(5)])
    summaries = {s.doc_id: s for s in await store.list_documents()}
    assert summaries["d1"].chunk_count == 2
    assert summaries["d2"].chunk_count == 2


# ---- delete_document ---------------------------------------------------------


async def test_delete_document_returns_chunk_count(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["x", "y", "z"], [_vec(i) for i in range(3)])
    removed = await store.delete_document("d1")
    assert removed == 3
    assert await store.list_documents() == []


async def test_delete_document_unknown_returns_zero(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["x"], [_vec(1)])
    removed = await store.delete_document("does-not-exist")
    assert removed == 0
    summaries = await store.list_documents()
    assert len(summaries) == 1


async def test_delete_document_leaves_siblings_alone(store: KnowledgeStore) -> None:
    await store.append_document("d1", "a.txt", ["a1"], [_vec(1)])
    await store.append_document("d2", "b.txt", ["b1", "b2"], [_vec(2), _vec(3)])
    removed = await store.delete_document("d1")
    assert removed == 1
    summaries = {s.doc_id: s for s in await store.list_documents()}
    assert set(summaries) == {"d2"}
    assert summaries["d2"].chunk_count == 2


# ---- replace_kb_with_document -----------------------------------------------


async def test_replace_kb_with_document_drops_everything_else(store: KnowledgeStore) -> None:
    await store.append_document("old1", "old1.txt", ["o1"], [_vec(1)])
    await store.append_document("old2", "old2.txt", ["o2"], [_vec(2)])
    await store.replace_kb_with_document("new", "new.txt", ["n1", "n2"], [_vec(3), _vec(4)], mime_type="text/plain")
    summaries = await store.list_documents()
    assert len(summaries) == 1
    assert summaries[0].doc_id == "new"
    assert summaries[0].chunk_count == 2
    assert summaries[0].mime == "text/plain"


# ---- Redis-specific guarantees ----------------------------------------------


async def test_redis_uses_composite_field_keys() -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.from_url", return_value=fake):
        store = RedisKnowledgeStore("redis://localhost:6379/0")
    await store.append_document("doc-42", "f.txt", ["x", "y"], [_vec(1), _vec(2)])
    fields = sorted(cast("list[str]", await fake.hkeys(KB_CHUNKS_HASH_KEY)))
    assert fields == [f"{CHUNK_FIELD_PREFIX}:doc-42:0", f"{CHUNK_FIELD_PREFIX}:doc-42:1"]


async def test_redis_replace_kb_is_one_write_visible_atomically() -> None:
    """``replace_kb_with_document`` must atomically swap chunks via MULTI/EXEC.

    We don't simulate a partial failure inside ``fakeredis`` (it would need a
    custom pipeline mock); instead we assert the post-condition that matters
    end-to-end: the hash contains *only* the new doc's fields and nothing
    from the previous doc. The pipeline is the only way to keep that
    invariant under concurrent readers.
    """
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.from_url", return_value=fake):
        store = RedisKnowledgeStore("redis://localhost:6379/0")
    await store.append_document("old", "old.txt", ["o1", "o2", "o3"], [_vec(i) for i in range(3)])
    await store.replace_kb_with_document("new", "new.txt", ["n1"], [_vec(99)])
    fields = sorted(cast("list[str]", await fake.hkeys(KB_CHUNKS_HASH_KEY)))
    assert fields == [f"{CHUNK_FIELD_PREFIX}:new:0"]


async def test_redis_legacy_uuid_entries_still_readable() -> None:
    """Pre-Wave-4 entries used ``ck:<uuid>`` keys without ``chunk_total``."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    import json as _json
    import uuid as _uuid

    payload_a = _json.dumps(
        {
            "text": "legacy-a",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "doc_id": "legacy",
            "filename": "old.txt",
            "chunk_index": 0,
        }
    )
    payload_b = _json.dumps(
        {
            "text": "legacy-b",
            "embedding": [0.5, 0.6, 0.7, 0.8],
            "doc_id": "legacy",
            "filename": "old.txt",
            "chunk_index": 1,
        }
    )
    await fake.hset(
        KB_CHUNKS_HASH_KEY,
        mapping={
            f"ck:{_uuid.uuid4()}-a": payload_a,
            f"ck:{_uuid.uuid4()}-b": payload_b,
        },
    )
    with patch("redis.asyncio.from_url", return_value=fake):
        store = RedisKnowledgeStore("redis://localhost:6379/0")
    summaries = await store.list_documents()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.doc_id == "legacy"
    assert s.chunk_count == 2  # inferred from sibling grouping


async def test_redis_delete_falls_back_for_legacy_uuid_keys() -> None:
    """Legacy ``ck:<uuid>`` keys can't be matched by prefix, so deletion
    must transparently rewrite the hash via the abstract base implementation."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    import json as _json

    await fake.hset(
        KB_CHUNKS_HASH_KEY,
        mapping={
            "ck:legacy-uuid-1": _json.dumps(
                {
                    "text": "x",
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "doc_id": "legacy",
                    "filename": "old.txt",
                    "chunk_index": 0,
                }
            ),
            "ck:legacy-uuid-2": _json.dumps(
                {
                    "text": "y",
                    "embedding": [0.5, 0.6, 0.7, 0.8],
                    "doc_id": "keep",
                    "filename": "keep.txt",
                    "chunk_index": 0,
                }
            ),
        },
    )
    with patch("redis.asyncio.from_url", return_value=fake):
        store = RedisKnowledgeStore("redis://localhost:6379/0")
    removed = await store.delete_document("legacy")
    assert removed == 1
    summaries = {s.doc_id for s in await store.list_documents()}
    assert summaries == {"keep"}


# ---- create_store factory ---------------------------------------------------


def test_create_store_picks_redis_when_url_present() -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.from_url", return_value=fake):
        s = create_store("redis://localhost:6379/0")
    assert isinstance(s, RedisKnowledgeStore)


def test_create_store_picks_memory_when_url_missing() -> None:
    assert isinstance(create_store(None), InMemoryKnowledgeStore)
