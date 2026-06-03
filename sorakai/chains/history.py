"""Adapter: expose :class:`sorakai.common.chat_history.ChatHistoryStore` as a LangChain :class:`BaseChatMessageHistory`.

``RunnableWithMessageHistory`` wants a per-session history object that
implements both sync (``messages``, ``add_messages``) and async
(``aget_messages``, ``aadd_messages``) surfaces. Our Wave 1 Redis store is
async-first, so:

- The async methods are the canonical path - they hit the Redis pipeline
  directly and stay multi-replica safe.
- The sync methods are best-effort: ``add_messages`` and ``messages`` use
  ``asyncio.run`` when called outside a running loop (e.g. unit tests), and
  raise loudly if invoked from inside one (``RuntimeError``). The RAG chain
  is only ever ``ainvoke``'d from the FastAPI handler so the async path is
  what actually runs in production.

Each LangChain ``HumanMessage`` / ``AIMessage`` is persisted as a single
``role + content`` row in the underlying store. We deliberately store user
and assistant turns one at a time so a chain that emits the human input
first and the AI reply second still produces a chronologically-correct
transcript - the Wave 1 ``append_pair`` is reserved for the legacy
non-chain code path that knows it always has both halves.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Coroutine, Sequence
from typing import Any, TypeVar

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from sorakai.common.chat_history import (
    CHAT_KEY_PREFIX,
    InMemoryChatHistoryStore,
    RedisChatHistoryStore,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _role_for(message: BaseMessage) -> str | None:
    """Map a LangChain message to our role string. Unknown roles are dropped."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        # System prompts are baked into the chain prompt, not persisted.
        return None
    return None


def _message_for(role: str, content: str) -> BaseMessage | None:
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return None


class SorakaiChatMessageHistory(BaseChatMessageHistory):
    """LangChain history shim over our async :class:`ChatHistoryStore`."""

    def __init__(
        self,
        store: RedisChatHistoryStore | InMemoryChatHistoryStore,
        session_id: str,
    ) -> None:
        self._store = store
        self._session_id = session_id

    # ---- async path (canonical, used by RunnableWithMessageHistory.ainvoke) ----

    async def aget_messages(self) -> list[BaseMessage]:
        raw = await self._store.get_messages(self._session_id)
        out: list[BaseMessage] = []
        for row in raw:
            msg = _message_for(row.get("role", ""), row.get("content", ""))
            if msg is not None:
                out.append(msg)
        return out

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if isinstance(self._store, RedisChatHistoryStore):
            await self._append_redis(messages)
            return
        # In-memory backend: store one message at a time so emit order is
        # preserved even when only a single AI message is added.
        for m in messages:
            role = _role_for(m)
            if role is None:
                continue
            mem = self._store
            hist = mem._sessions.setdefault(self._session_id, [])
            hist.append({"role": role, "content": str(m.content)})
            if len(hist) > mem._max:
                mem._sessions[self._session_id] = hist[-mem._max :]

    async def aclear(self) -> None:
        if isinstance(self._store, RedisChatHistoryStore):
            await self._store._redis.delete(f"{CHAT_KEY_PREFIX}{self._session_id}")
            return
        self._store._sessions.pop(self._session_id, None)

    # ---- sync path (best-effort; raises inside a running loop) -----------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        # BaseChatMessageHistory declares ``messages`` as a writeable attribute;
        # using a property is the canonical override pattern across the
        # LangChain ecosystem (FileChatMessageHistory, RedisChatMessageHistory,
        # etc.) so we silence the override check here.
        return self._run_sync(self.aget_messages())

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._run_sync(self.aadd_messages(messages))

    def clear(self) -> None:
        self._run_sync(self.aclear())

    # ---- internals -------------------------------------------------------

    async def _append_redis(self, messages: Sequence[BaseMessage]) -> None:
        store = self._store
        assert isinstance(store, RedisChatHistoryStore)
        payloads: list[str] = []
        for m in messages:
            role = _role_for(m)
            if role is None:
                continue
            payloads.append(json.dumps({"role": role, "content": str(m.content)}, ensure_ascii=False))
        if not payloads:
            return
        key = f"{CHAT_KEY_PREFIX}{self._session_id}"
        async with store._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *payloads)
            pipe.ltrim(key, -store._max, -1)
            if store._ttl > 0:
                pipe.expire(key, store._ttl)
            await pipe.execute()

    @staticmethod
    def _run_sync(coro: Coroutine[Any, Any, _T] | Awaitable[_T]) -> _T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not isinstance(coro, Coroutine):
                raise TypeError("expected coroutine for sync execution") from None
            result: _T = asyncio.run(coro)
            return result
        raise RuntimeError(
            "SorakaiChatMessageHistory sync methods cannot run inside an event loop; "
            "use the async (aget_messages / aadd_messages / aclear) path via RunnableWithMessageHistory.ainvoke."
        )
