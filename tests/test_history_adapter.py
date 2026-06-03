"""Tests for :class:`sorakai.chains.history.SorakaiChatMessageHistory`."""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sorakai.chains.history import SorakaiChatMessageHistory
from sorakai.common.chat_history import (
    CHAT_KEY_PREFIX,
    InMemoryChatHistoryStore,
    RedisChatHistoryStore,
)


@pytest.mark.asyncio
async def test_aget_messages_empty() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-1")
    assert await history.aget_messages() == []


@pytest.mark.asyncio
async def test_aadd_messages_in_memory_preserves_order() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-1")
    await history.aadd_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    await history.aadd_messages([HumanMessage(content="more"), AIMessage(content="ok")])

    msgs = await history.aget_messages()
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert [str(m.content) for m in msgs] == ["hi", "hello", "more", "ok"]


@pytest.mark.asyncio
async def test_aadd_messages_drops_system_messages() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-1")
    await history.aadd_messages([SystemMessage(content="ignore me"), HumanMessage(content="kept")])
    msgs = await history.aget_messages()
    assert [type(m).__name__ for m in msgs] == ["HumanMessage"]


@pytest.mark.asyncio
async def test_aclear_in_memory() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-1")
    await history.aadd_messages([HumanMessage(content="x")])
    assert await history.aget_messages()

    await history.aclear()
    assert await history.aget_messages() == []


@pytest.mark.asyncio
async def test_aadd_messages_redis_pipeline_atomic() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = RedisChatHistoryStore.__new__(RedisChatHistoryStore)
    store._redis = fake_redis  # type: ignore[attr-defined]
    store._ttl = 60  # type: ignore[attr-defined]
    store._max = 40  # type: ignore[attr-defined]

    history = SorakaiChatMessageHistory(store, "sess-redis")
    await history.aadd_messages([HumanMessage(content="q1"), AIMessage(content="a1")])

    msgs = await history.aget_messages()
    assert [str(m.content) for m in msgs] == ["q1", "a1"]

    raw = await fake_redis.lrange(f"{CHAT_KEY_PREFIX}sess-redis", 0, -1)
    assert len(raw) == 2


@pytest.mark.asyncio
async def test_max_messages_window_trims_on_overflow() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = RedisChatHistoryStore.__new__(RedisChatHistoryStore)
    store._redis = fake_redis  # type: ignore[attr-defined]
    store._ttl = 60  # type: ignore[attr-defined]
    store._max = 4  # type: ignore[attr-defined]

    history = SorakaiChatMessageHistory(store, "sess-window")
    for i in range(5):
        await history.aadd_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])

    msgs = await history.aget_messages()
    assert len(msgs) == 4
    assert [str(m.content) for m in msgs] == ["q3", "a3", "q4", "a4"]


def test_sync_property_outside_loop_runs_to_completion() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-sync")
    history.add_messages([HumanMessage(content="hi")])
    msgs = history.messages
    assert [str(m.content) for m in msgs] == ["hi"]


@pytest.mark.asyncio
async def test_sync_property_inside_loop_raises() -> None:
    store = InMemoryChatHistoryStore()
    history = SorakaiChatMessageHistory(store, "sess-loop")
    # Inside a running loop the sync surface must refuse rather than block.
    import warnings

    with warnings.catch_warnings(), pytest.raises(RuntimeError):
        warnings.simplefilter("ignore", RuntimeWarning)
        _ = history.messages


@pytest.mark.asyncio
async def test_concurrent_appends_dont_lose_messages() -> None:
    """Regression: the Wave 1 atomic pipeline survives parallel writers; the
    adapter must preserve that property when several chains share a store."""
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = RedisChatHistoryStore.__new__(RedisChatHistoryStore)
    store._redis = fake_redis  # type: ignore[attr-defined]
    store._ttl = 60  # type: ignore[attr-defined]
    store._max = 100  # type: ignore[attr-defined]

    history = SorakaiChatMessageHistory(store, "sess-concurrent")

    async def add_pair(i: int) -> None:
        await history.aadd_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])

    await asyncio.gather(*(add_pair(i) for i in range(10)))
    msgs = await history.aget_messages()
    # 10 pairs => 20 messages.
    assert len(msgs) == 20
    # Roles alternate H/A pairwise (each pair is atomic).
    role_pairs = [(type(msgs[i]).__name__, type(msgs[i + 1]).__name__) for i in range(0, 20, 2)]
    assert all(p == ("HumanMessage", "AIMessage") for p in role_pairs)
