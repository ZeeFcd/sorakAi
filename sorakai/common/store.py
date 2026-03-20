"""Shared knowledge-base storage: Redis (multi-replica) or in-process (single instance)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

KB_KEY = "sorakai:kb"


class KnowledgeStore(ABC):
    @abstractmethod
    async def save(self, chunks: list[str], embeddings: list[np.ndarray]) -> None:
        pass

    @abstractmethod
    async def load(self) -> tuple[list[str], list[np.ndarray]] | None:
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if backend is reachable (always True for in-memory)."""


class InMemoryKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._embeddings: list[np.ndarray] = []

    async def save(self, chunks: list[str], embeddings: list[np.ndarray]) -> None:
        self._chunks = list(chunks)
        self._embeddings = [np.array(e, dtype=float) for e in embeddings]

    async def load(self) -> tuple[list[str], list[np.ndarray]] | None:
        if not self._chunks:
            return None
        return list(self._chunks), [np.array(e, dtype=float) for e in self._embeddings]

    async def ping(self) -> bool:
        return True


class RedisKnowledgeStore(KnowledgeStore):
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url, decode_responses=True)

    async def save(self, chunks: list[str], embeddings: list[np.ndarray]) -> None:
        payload: dict[str, Any] = {
            "chunks": chunks,
            "embeddings": [e.astype(float).tolist() for e in embeddings],
        }
        await self._redis.set(KB_KEY, json.dumps(payload))

    async def load(self) -> tuple[list[str], list[np.ndarray]] | None:
        raw = await self._redis.get(KB_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        chunks = data["chunks"]
        embeddings = [np.array(row, dtype=float) for row in data["embeddings"]]
        return chunks, embeddings

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()


def create_store(redis_url: str | None) -> KnowledgeStore:
    if redis_url:
        return RedisKnowledgeStore(redis_url)
    return InMemoryKnowledgeStore()
