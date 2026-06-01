"""Shared knowledge-base: in-memory list or Redis HASH (one field per chunk).

**Redis**

- **KB chunks**: ``sorakai:kb:chunks`` — Redis *HASH*: field ``ck:<uuid>`` → JSON
  ``{text, embedding, doc_id, filename, chunk_index}``. Appends use **HSET** only.

**Chat history** (RAG) uses separate keys — see ``chat_history.py``.

For large corpora or ANN search, consider a dedicated vector DB (Qdrant, pgvector, …).
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np

KB_CHUNKS_HASH_KEY = "sorakai:kb:chunks"


class ChunkEntry(TypedDict):
    text: str
    embedding: list[float]
    doc_id: str
    filename: str
    chunk_index: int


def _entries_to_flat(entries: list[ChunkEntry]) -> tuple[list[str], list[np.ndarray]]:
    chunks = [e["text"] for e in entries]
    embeddings = [np.array(e["embedding"], dtype=float) for e in entries]
    return chunks, embeddings


def _entry_to_json(entry: ChunkEntry) -> str:
    return json.dumps(dict(entry), ensure_ascii=False)


class KnowledgeStore(ABC):
    async def clear_all(self) -> None:
        await self._write_entries([])

    async def append_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        entries = await self._read_entries()
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            entries.append(
                {
                    "text": text,
                    "embedding": emb.astype(float).tolist(),
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                }
            )
        await self._write_entries(entries)

    async def replace_entire_kb(
        self, chunks: list[str], embeddings: list[np.ndarray], filename: str = "inline"
    ) -> None:
        """Replace the whole KB with a single synthetic document."""
        doc_id = str(uuid.uuid4())
        new_entries: list[ChunkEntry] = []
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            new_entries.append(
                {
                    "text": text,
                    "embedding": emb.astype(float).tolist(),
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                }
            )
        await self._write_entries(new_entries)

    async def save(self, chunks: list[str], embeddings: list[np.ndarray]) -> None:
        """Replace entire KB with one synthetic document (tests / simple wipe)."""
        await self.replace_entire_kb(chunks, embeddings, filename="inline")

    async def load(self) -> tuple[list[str], list[np.ndarray]] | None:
        flat = await self.load_flat()
        if not flat:
            return None
        chunks, embeddings = flat
        return chunks, embeddings

    async def load_flat(self) -> tuple[list[str], list[np.ndarray]] | None:
        entries = await self._read_entries()
        if not entries:
            return None
        return _entries_to_flat(entries)

    @abstractmethod
    async def _read_entries(self) -> list[ChunkEntry]:
        pass

    @abstractmethod
    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        pass

    @abstractmethod
    async def ping(self) -> bool:
        pass


class InMemoryKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._entries: list[ChunkEntry] = []

    async def _read_entries(self) -> list[ChunkEntry]:
        return list(self._entries)

    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        self._entries = list(entries)

    async def ping(self) -> bool:
        return True


class RedisKnowledgeStore(KnowledgeStore):
    """KB as HASH (one field per chunk)."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url, decode_responses=True)

    async def _read_entries(self) -> list[ChunkEntry]:
        raw_map = await self._redis.hgetall(KB_CHUNKS_HASH_KEY)
        entries: list[ChunkEntry] = []
        for v in raw_map.values():
            d = json.loads(v)
            entries.append(
                {
                    "text": d["text"],
                    "embedding": list(d["embedding"]),
                    "doc_id": d["doc_id"],
                    "filename": d["filename"],
                    "chunk_index": int(d["chunk_index"]),
                }
            )
        entries.sort(key=lambda e: (e["doc_id"], e["chunk_index"]))
        return entries

    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        await self._redis.delete(KB_CHUNKS_HASH_KEY)
        if not entries:
            return
        mapping = {f"ck:{uuid.uuid4()}": _entry_to_json(e) for e in entries}
        await self._redis.hset(KB_CHUNKS_HASH_KEY, mapping=mapping)

    async def append_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        if not chunks:
            return
        mapping: dict[str, str] = {}
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            entry: ChunkEntry = {
                "text": text,
                "embedding": emb.astype(float).tolist(),
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
            }
            mapping[f"ck:{uuid.uuid4()}"] = _entry_to_json(entry)
        await self._redis.hset(KB_CHUNKS_HASH_KEY, mapping=mapping)

    async def clear_all(self) -> None:
        await self._redis.delete(KB_CHUNKS_HASH_KEY)

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
