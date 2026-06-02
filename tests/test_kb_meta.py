"""Tests for :mod:`sorakai.common.kb_meta`."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from sorakai.common.kb_meta import (
    InMemoryKBMetaStore,
    KBMeta,
    RedisKBMetaStore,
    create_kb_meta_store,
)
from sorakai.core.errors import DimensionMismatchError, StoreError


def test_create_kb_meta_store_picks_in_memory_when_no_redis() -> None:
    assert isinstance(create_kb_meta_store(None), InMemoryKBMetaStore)


def test_kb_meta_matches_compares_all_fields() -> None:
    a = KBMeta(provider="ollama", model="nomic-embed-text", dim=768)
    assert a.matches(KBMeta(provider="ollama", model="nomic-embed-text", dim=768))
    assert not a.matches(KBMeta(provider="ollama", model="nomic-embed-text", dim=512))
    assert not a.matches(KBMeta(provider="ollama", model="other-model", dim=768))
    assert not a.matches(KBMeta(provider="char", model="nomic-embed-text", dim=768))


def test_in_memory_round_trip_and_clear(run_async) -> None:
    store = InMemoryKBMetaStore()
    assert run_async(store.read()) is None
    meta = KBMeta(provider="char", model="test", dim=256)
    run_async(store.write(meta))
    assert run_async(store.read()) == meta
    run_async(store.clear())
    assert run_async(store.read()) is None


def test_ensure_compatible_writes_when_empty(run_async) -> None:
    store = InMemoryKBMetaStore()
    meta = KBMeta(provider="ollama", model="m", dim=384)
    result = run_async(store.ensure_compatible(meta))
    assert result == meta
    assert run_async(store.read()) == meta


def test_ensure_compatible_returns_existing_on_match(run_async) -> None:
    store = InMemoryKBMetaStore()
    meta = KBMeta(provider="ollama", model="m", dim=384)
    run_async(store.write(meta))
    result = run_async(store.ensure_compatible(meta))
    assert result == meta


def test_ensure_compatible_raises_on_mismatch(run_async) -> None:
    store = InMemoryKBMetaStore()
    run_async(store.write(KBMeta(provider="ollama", model="m1", dim=384)))
    with pytest.raises(DimensionMismatchError) as exc_info:
        run_async(store.ensure_compatible(KBMeta(provider="ollama", model="m2", dim=384)))
    exc = exc_info.value
    assert exc.expected_model == "m1"
    assert exc.actual_model == "m2"


def test_reset_to_overwrites(run_async) -> None:
    store = InMemoryKBMetaStore()
    old = KBMeta(provider="ollama", model="old", dim=384)
    new = KBMeta(provider="ollama", model="new", dim=768)
    run_async(store.write(old))
    run_async(store.reset_to(new))
    assert run_async(store.read()) == new


def test_redis_round_trip_with_fakeredis(monkeypatch, run_async) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "sorakai.common.kb_meta.redis.from_url",
        lambda *_a, **_k: fake,
    )
    store = RedisKBMetaStore("redis://fake")
    assert run_async(store.read()) is None

    meta = KBMeta(provider="ollama", model="nomic-embed-text", dim=768)
    run_async(store.write(meta))
    assert run_async(store.read()) == meta

    run_async(store.clear())
    assert run_async(store.read()) is None


def test_redis_corrupt_meta_raises_store_error(monkeypatch, run_async) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "sorakai.common.kb_meta.redis.from_url",
        lambda *_a, **_k: fake,
    )
    store = RedisKBMetaStore("redis://fake")

    async def _setup_and_read() -> None:
        await fake.hset(
            "sorakai:kb:meta",
            mapping={"provider": "ollama", "model": "m", "dim": "not-an-int"},
        )
        await store.read()

    with pytest.raises(StoreError, match="Corrupt KB meta"):
        run_async(_setup_and_read())
