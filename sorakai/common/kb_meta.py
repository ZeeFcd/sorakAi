"""Knowledge-base metadata: what provider/model/dim built the current KB.

Wave 2 introduces this guard so the silent zero-padding that lived in
``sorakai.common.retrieval._pad_to_same_length`` cannot mask a query being
embedded by a different model (and therefore a different vector space) than
the stored chunks. The metadata is written on first ingest and verified on
every ingest + query; a mismatch raises :class:`~sorakai.core.errors.DimensionMismatchError`,
which the HTTP handlers translate to a ``409 Conflict``.

Storage layout (Redis):

- ``sorakai:kb:meta`` -> Redis ``HASH`` with fields ``provider``, ``model``, ``dim``.

In-memory backend mirrors the same Protocol for tests and single-process dev.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import redis.asyncio as redis

from sorakai.core.errors import DimensionMismatchError, StoreError

KB_META_KEY = "sorakai:kb:meta"


def _to_str(x: str | bytes) -> str:
    """Normalise a redis value that may come back as bytes (some clients) or str.

    Avoids relying on the connection's ``decode_responses=True`` setting being
    honoured for every command - notably ``HGETALL`` returns bytes in some
    fakeredis versions even when the flag is set.
    """
    return x.decode("utf-8") if isinstance(x, bytes) else x


@dataclass(frozen=True, slots=True)
class KBMeta:
    """Identity of the embedding model that built the current KB."""

    provider: str
    model: str
    dim: int

    def matches(self, other: KBMeta) -> bool:
        return (self.provider, self.model, self.dim) == (other.provider, other.model, other.dim)


class KBMetaStore(ABC):
    """Read/write/clear interface for the KB-identity record (ISP-narrow)."""

    @abstractmethod
    async def read(self) -> KBMeta | None: ...

    @abstractmethod
    async def write(self, meta: KBMeta) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    async def ensure_compatible(self, candidate: KBMeta) -> KBMeta:
        """Write ``candidate`` if no meta yet; else raise on mismatch, return on match.

        Returns the (possibly newly-written) meta that the KB is committed to.
        Convenience helper used from both ingest and RAG handlers.
        """
        existing = await self.read()
        if existing is None:
            await self.write(candidate)
            return candidate
        if not existing.matches(candidate):
            raise DimensionMismatchError(
                expected_provider=existing.provider,
                expected_model=existing.model,
                expected_dim=existing.dim,
                actual_provider=candidate.provider,
                actual_model=candidate.model,
                actual_dim=candidate.dim,
            )
        return existing

    async def reset_to(self, meta: KBMeta) -> None:
        """``replace_kb=True`` path: drop old identity and stamp the new one."""
        await self.clear()
        await self.write(meta)


class InMemoryKBMetaStore(KBMetaStore):
    """Single-process backend. Use in tests and when Redis isn't configured."""

    def __init__(self) -> None:
        self._meta: KBMeta | None = None

    async def read(self) -> KBMeta | None:
        return self._meta

    async def write(self, meta: KBMeta) -> None:
        self._meta = meta

    async def clear(self) -> None:
        self._meta = None


class RedisKBMetaStore(KBMetaStore):
    """Redis-backed identity store. Uses a HASH so future fields can land additively."""

    def __init__(self, url: str) -> None:
        self._redis: redis.Redis[str] = redis.from_url(url, decode_responses=True)

    async def read(self) -> KBMeta | None:
        raw = await self._redis.hgetall(KB_META_KEY)
        if not raw:
            return None
        # Some redis / fakeredis versions ignore ``decode_responses=True`` for
        # ``HGETALL`` in async mode and return ``bytes`` keys + values. Normalise
        # here so the rest of the code can index with str literals.
        data = {_to_str(k): _to_str(v) for k, v in raw.items()}
        try:
            return KBMeta(
                provider=data["provider"],
                model=data["model"],
                dim=int(data["dim"]),
            )
        except (KeyError, ValueError) as exc:
            raise StoreError(f"Corrupt KB meta at {KB_META_KEY!r}: {data!r}") from exc

    async def write(self, meta: KBMeta) -> None:
        await self._redis.hset(
            KB_META_KEY,
            mapping={"provider": meta.provider, "model": meta.model, "dim": str(meta.dim)},
        )

    async def clear(self) -> None:
        await self._redis.delete(KB_META_KEY)

    async def aclose(self) -> None:
        # See note in chat_history.py: types-redis 4.6 stubs predate the ``aclose`` rename.
        await self._redis.aclose()  # type: ignore[attr-defined]


def create_kb_meta_store(redis_url: str | None) -> KBMetaStore:
    """Pick the right backend based on whether Redis is configured."""
    if redis_url:
        return RedisKBMetaStore(redis_url)
    return InMemoryKBMetaStore()
