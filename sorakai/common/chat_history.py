"""Session-scoped chat history for multi-turn RAG (Redis or in-process)."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

MAX_MESSAGES = 40  # cap stored turns (~20 user/assistant pairs)
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
CHAT_KEY_PREFIX = "sorakai:chat:"


def validate_session_id(session_id: str | None) -> str | None:
    if session_id is None or session_id == "":
        return None
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("session_id must be 1–128 chars: letters, digits, ._-")
    return session_id


class ChatHistoryStore(ABC):
    @abstractmethod
    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """OpenAI-style messages: role in user|assistant, content str."""

    @abstractmethod
    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        pass


class InMemoryChatHistoryStore(ChatHistoryStore):
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        hist = self._sessions.setdefault(session_id, [])
        hist.append({"role": "user", "content": user_text})
        hist.append({"role": "assistant", "content": assistant_text})
        if len(hist) > MAX_MESSAGES:
            self._sessions[session_id] = hist[-MAX_MESSAGES:]


class RedisChatHistoryStore(ChatHistoryStore):
    def __init__(self, url: str, ttl_seconds: int = 604_800) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{CHAT_KEY_PREFIX}{session_id}"

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return [m for m in data if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m]

    async def append_pair(self, session_id: str, user_text: str, assistant_text: str) -> None:
        key = self._key(session_id)
        hist = await self.get_messages(session_id)
        hist.append({"role": "user", "content": user_text})
        hist.append({"role": "assistant", "content": assistant_text})
        if len(hist) > MAX_MESSAGES:
            hist = hist[-MAX_MESSAGES:]
        payload = json.dumps(hist, ensure_ascii=False)
        if self._ttl > 0:
            await self._redis.setex(key, self._ttl, payload)
        else:
            await self._redis.set(key, payload)

    async def aclose(self) -> None:
        await self._redis.aclose()


def create_chat_store(redis_url: str | None, ttl_seconds: int = 604_800) -> ChatHistoryStore:
    if redis_url:
        return RedisChatHistoryStore(redis_url, ttl_seconds=ttl_seconds)
    return InMemoryChatHistoryStore()
