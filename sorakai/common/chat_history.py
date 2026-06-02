"""Session-scoped chat history for multi-turn RAG.

The Redis backend uses a list per session with **atomic** ``RPUSH`` +
``LTRIM`` + ``EXPIRE`` inside a single pipeline (formerly read-modify-write
``GET``/``SET``, which lost messages under parallel turns). The in-memory
backend keeps the same Protocol so tests don't need Redis.

Storage layout
--------------

- ``sorakai:chat:<session_id>`` -> Redis ``LIST`` of JSON-encoded
  ``{"role": "user"|"assistant", "content": str}`` messages.
- TTL is refreshed on every append; ``chat_history_max_messages`` caps the
  list length via ``LTRIM``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import redis.asyncio as redis

CHAT_KEY_PREFIX = "sorakai:chat:"
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
DEFAULT_MAX_MESSAGES = 40


def validate_session_id(session_id: str | None) -> str | None:
    """Return a sanitised ``session_id`` or raise ``ValueError`` on bad input."""
    if not session_id:
        return None
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("session_id must be 1-128 chars: letters, digits, ._-")
    return session_id


class ChatHistoryStore(ABC):
    """Abstract backend interface (ISP: only what call sites need)."""

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Return prior messages in OpenAI-style format (role + content)."""

    @abstractmethod
    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Append one user + one assistant turn, atomically with respect to other writers."""


class InMemoryChatHistoryStore(ChatHistoryStore):
    """Single-process backend. Not safe across replicas - use Redis in prod."""

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._max = max_messages

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        hist = self._sessions.setdefault(session_id, [])
        hist.append({"role": "user", "content": user_text})
        hist.append({"role": "assistant", "content": assistant_text})
        if len(hist) > self._max:
            self._sessions[session_id] = hist[-self._max :]


class RedisChatHistoryStore(ChatHistoryStore):
    """Multi-replica safe Redis backend (atomic ``RPUSH`` + ``LTRIM`` + ``EXPIRE``)."""

    def __init__(
        self,
        url: str,
        ttl_seconds: int = 604_800,
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ) -> None:
        self._redis: redis.Redis[str] = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._max = max_messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{CHAT_KEY_PREFIX}{session_id}"

    @staticmethod
    def _decode(raw: str) -> dict[str, str] | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        role = data.get("role")
        content = data.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        return {"role": role, "content": content}

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        raw_list: list[str] = await self._redis.lrange(self._key(session_id), 0, -1)
        out: list[dict[str, str]] = []
        for raw in raw_list:
            decoded = self._decode(raw)
            if decoded is not None:
                out.append(decoded)
        return out

    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        key = self._key(session_id)
        user_msg = json.dumps({"role": "user", "content": user_text}, ensure_ascii=False)
        assistant_msg = json.dumps({"role": "assistant", "content": assistant_text}, ensure_ascii=False)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, user_msg, assistant_msg)
            pipe.ltrim(key, -self._max, -1)
            if self._ttl > 0:
                pipe.expire(key, self._ttl)
            await pipe.execute()

    async def aclose(self) -> None:
        # ``aclose`` exists at runtime in redis-py >= 5.0 (preferred over deprecated ``close``);
        # the bundled ``types-redis`` 4.6 stubs predate that rename, so the attr check needs an ignore.
        await self._redis.aclose()  # type: ignore[attr-defined]


def create_chat_store(
    redis_url: str | None,
    ttl_seconds: int = 604_800,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> ChatHistoryStore:
    """Pick the right backend based on whether Redis is configured."""
    if redis_url:
        return RedisChatHistoryStore(redis_url, ttl_seconds=ttl_seconds, max_messages=max_messages)
    return InMemoryChatHistoryStore(max_messages=max_messages)
