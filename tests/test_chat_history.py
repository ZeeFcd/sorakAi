"""Behavioural tests for the chat-history backends.

The Redis case is exercised via ``fakeredis.aioredis``: a wire-compatible
in-process Redis that supports pipelines + ``LTRIM`` + ``EXPIRE`` and lets us
test the atomic write path without spinning up a real server.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from sorakai.common.chat_history import (
    ChatHistoryStore,
    InMemoryChatHistoryStore,
    RedisChatHistoryStore,
    create_chat_store,
    validate_session_id,
)


def test_validate_session_id_rejects_bad_input() -> None:
    assert validate_session_id(None) is None
    assert validate_session_id("u_1.alpha-2") == "u_1.alpha-2"
    with pytest.raises(ValueError):
        validate_session_id("with spaces")
    with pytest.raises(ValueError):
        validate_session_id("x" * 200)


def test_in_memory_store_caps_at_max_messages(run_async) -> None:
    store: ChatHistoryStore = InMemoryChatHistoryStore(max_messages=4)
    for i in range(5):
        run_async(store.append_pair("sess", f"q{i}", f"a{i}"))
    msgs = run_async(store.get_messages("sess"))
    assert len(msgs) == 4
    assert msgs[0] == {"role": "user", "content": "q3"}
    assert msgs[-1] == {"role": "assistant", "content": "a4"}


def test_create_chat_store_picks_in_memory_when_no_redis() -> None:
    assert isinstance(create_chat_store(None), InMemoryChatHistoryStore)


def test_redis_store_atomic_under_concurrent_writers(monkeypatch, run_async) -> None:
    """Five concurrent ``append_pair`` calls must produce exactly 10 messages.

    The legacy GET-modify-SET implementation lost writes under this load.
    """

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "sorakai.common.chat_history.redis.from_url",
        lambda *_a, **_k: fake,
    )
    store = RedisChatHistoryStore("redis://fake", ttl_seconds=60, max_messages=100)

    async def _run() -> list[dict[str, str]]:
        await asyncio.gather(*(store.append_pair("sess", f"q{i}", f"a{i}") for i in range(5)))
        return await store.get_messages("sess")

    msgs = run_async(_run())
    assert len(msgs) == 10, msgs
    assert sum(1 for m in msgs if m["role"] == "user") == 5
    assert sum(1 for m in msgs if m["role"] == "assistant") == 5


def test_redis_store_trims_to_max_messages(monkeypatch, run_async) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "sorakai.common.chat_history.redis.from_url",
        lambda *_a, **_k: fake,
    )
    store = RedisChatHistoryStore("redis://fake", ttl_seconds=60, max_messages=4)

    async def _run() -> list[dict[str, str]]:
        for i in range(5):
            await store.append_pair("sess", f"q{i}", f"a{i}")
        return await store.get_messages("sess")

    msgs = run_async(_run())
    assert len(msgs) == 4
    assert msgs[0]["content"] == "q3"


def test_redis_store_skips_corrupt_entries(monkeypatch, run_async) -> None:
    """Bad JSON or wrong shape must be silently dropped, not crash the reader."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "sorakai.common.chat_history.redis.from_url",
        lambda *_a, **_k: fake,
    )
    store = RedisChatHistoryStore("redis://fake", ttl_seconds=60, max_messages=10)

    async def _run() -> list[dict[str, str]]:
        await store.append_pair("sess", "hi", "hello")
        await fake.rpush("sorakai:chat:sess", "{not-json")
        await fake.rpush("sorakai:chat:sess", '{"role":"narrator","content":"x"}')
        return await store.get_messages("sess")

    msgs = run_async(_run())
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
