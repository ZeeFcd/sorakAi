"""Knowledge base storage: in-memory and Redis HASH backends.

**Layout**

- Each chunk is keyed by ``ck:<doc_id>:<chunk_index>`` (Wave 4). Re-ingesting
  the same ``doc_id`` overwrites cleanly instead of leaking duplicates the way
  the Wave 0..3 ``ck:<uuid>`` keying did.
- The Redis backend keeps everything under a single hash
  (``sorakai:kb:chunks``), so we get atomic multi-field writes through a
  pipeline without touching multiple top-level keys.
- All write paths run inside a ``MULTI``/``EXEC`` Redis pipeline, including the
  ``replace_kb`` flow that pairs the chunk wipe with the new chunk batch.

**Reading legacy entries**

Pre-Wave-3 entries lack ``chunk_total`` + ``mime``; pre-Wave-4 entries used
``ck:<uuid>`` keys and we infer ``chunk_total`` from siblings sharing the same
``doc_id``. Both are tolerated by the read path so live KBs survive an upgrade.

For large corpora or ANN search, swap the backend out via the Wave 5
``VectorStore`` Protocol.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KB_CHUNKS_HASH_KEY = "sorakai:kb:chunks"
CHUNK_FIELD_PREFIX = "ck"


class ChunkEntry(TypedDict):
    text: str
    embedding: list[float]
    doc_id: str
    filename: str
    chunk_index: int
    chunk_total: int
    mime: str | None


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """One row in :meth:`KnowledgeStore.list_documents`."""

    doc_id: str
    filename: str
    chunk_count: int
    mime: str | None


def _chunk_field(doc_id: str, chunk_index: int) -> str:
    """Composite Redis hash field key for one chunk."""
    return f"{CHUNK_FIELD_PREFIX}:{doc_id}:{chunk_index}"


def _entry_to_json(entry: ChunkEntry) -> str:
    return json.dumps(dict(entry), ensure_ascii=False)


def _decode(value: str | bytes) -> str:
    """Some ``fakeredis`` versions ignore ``decode_responses=True`` for HGETALL."""
    return value.decode("utf-8") if isinstance(value, bytes | bytearray) else value


def _entries_to_flat(entries: list[ChunkEntry]) -> tuple[list[str], list[np.ndarray]]:
    chunks = [e["text"] for e in entries]
    embeddings = [np.array(e["embedding"], dtype=float) for e in entries]
    return chunks, embeddings


def _summaries_from_entries(entries: list[ChunkEntry]) -> list[DocumentSummary]:
    """Group entries by ``doc_id`` -> :class:`DocumentSummary`.

    Order is stable: documents are returned in the order their first chunk
    was first seen in ``entries``. This keeps API responses deterministic
    for clients that paginate by truncation.
    """
    by_doc: dict[str, list[ChunkEntry]] = {}
    for e in entries:
        by_doc.setdefault(e["doc_id"], []).append(e)
    return [
        DocumentSummary(
            doc_id=doc_id,
            filename=group[0]["filename"],
            chunk_count=len(group),
            mime=group[0]["mime"],
        )
        for doc_id, group in by_doc.items()
    ]


class KnowledgeStore(ABC):
    """Abstract KB. Concrete backends override the I/O primitives."""

    # ---- read path -------------------------------------------------------

    @abstractmethod
    async def _read_entries(self) -> list[ChunkEntry]:
        """Return every chunk, sorted by ``(doc_id, chunk_index)``."""

    @abstractmethod
    async def ping(self) -> bool:
        """Cheap liveness probe."""

    async def load_flat(self) -> tuple[list[str], list[np.ndarray]] | None:
        entries = await self._read_entries()
        if not entries:
            return None
        return _entries_to_flat(entries)

    async def load(self) -> tuple[list[str], list[np.ndarray]] | None:
        return await self.load_flat()

    async def list_documents(self) -> list[DocumentSummary]:
        return _summaries_from_entries(await self._read_entries())

    # ---- write path (default impl rewrites the whole entry list) ---------

    @abstractmethod
    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        """Replace the entire stored entry list with ``entries``."""

    async def clear_all(self) -> None:
        await self._write_entries([])

    async def append_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
        *,
        mime_type: str | None = None,
    ) -> None:
        """Atomically replace any prior chunks for ``doc_id`` with the new batch.

        Re-ingesting the same ``doc_id`` is idempotent: old chunks are dropped
        in the same write that adds the new ones, so partial-failure halfway
        through cannot leave duplicates behind. Empty ``chunks`` deletes
        anything stored under that ``doc_id``.
        """
        kept = [e for e in await self._read_entries() if e["doc_id"] != doc_id]
        total = len(chunks)
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            kept.append(
                {
                    "text": text,
                    "embedding": emb.astype(float).tolist(),
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_total": total,
                    "mime": mime_type,
                }
            )
        await self._write_entries(kept)

    async def delete_document(self, doc_id: str) -> int:
        """Remove every chunk for ``doc_id``. Returns the count removed."""
        entries = await self._read_entries()
        kept = [e for e in entries if e["doc_id"] != doc_id]
        removed = len(entries) - len(kept)
        if removed == 0:
            return 0
        await self._write_entries(kept)
        return removed

    async def replace_kb_with_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
        *,
        mime_type: str | None = None,
    ) -> None:
        """Wipe everything and store a single document, in one write."""
        new_entries: list[ChunkEntry] = []
        total = len(chunks)
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            new_entries.append(
                {
                    "text": text,
                    "embedding": emb.astype(float).tolist(),
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_total": total,
                    "mime": mime_type,
                }
            )
        await self._write_entries(new_entries)


class InMemoryKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._entries: list[ChunkEntry] = []

    async def _read_entries(self) -> list[ChunkEntry]:
        out = list(self._entries)
        out.sort(key=lambda e: (e["doc_id"], e["chunk_index"]))
        return out

    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        self._entries = list(entries)

    async def ping(self) -> bool:
        return True


class RedisKnowledgeStore(KnowledgeStore):
    """KB as a single Redis HASH, one field per chunk.

    Every public write goes through a pipeline so callers see the change
    atomically: ``replace_kb_with_document`` pairs the ``DEL`` of the whole
    hash with the ``HSET`` of the new chunks, ``append_document`` pairs the
    ``HDEL`` of the old chunks for that ``doc_id`` with the ``HSET`` of the
    new ones, and ``delete_document`` is a single multi-field ``HDEL``.
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url, decode_responses=True)

    async def _hgetall(self) -> dict[str, str]:
        raw = await self._redis.hgetall(KB_CHUNKS_HASH_KEY)
        return {_decode(k): _decode(v) for k, v in raw.items()}

    async def _read_entries(self) -> list[ChunkEntry]:
        raw_map = await self._hgetall()
        # Group pre-Wave-4 entries (``ck:<uuid>``) so we can infer
        # ``chunk_total`` from siblings sharing the same doc_id.
        legacy_groups: dict[str, int] = {}
        parsed: list[dict[str, Any]] = []
        for raw in raw_map.values():
            d = cast("dict[str, Any]", json.loads(raw))
            parsed.append(d)
            doc_id = str(d.get("doc_id", ""))
            if "chunk_total" not in d and doc_id:
                legacy_groups[doc_id] = legacy_groups.get(doc_id, 0) + 1

        entries: list[ChunkEntry] = []
        for d in parsed:
            doc_id = str(d["doc_id"])
            raw_total = d.get("chunk_total")
            chunk_total = raw_total if isinstance(raw_total, int) and raw_total >= 0 else legacy_groups.get(doc_id, -1)
            mime_value = d.get("mime")
            mime: str | None = mime_value if isinstance(mime_value, str) else None
            entries.append(
                {
                    "text": str(d["text"]),
                    "embedding": [float(x) for x in d["embedding"]],
                    "doc_id": doc_id,
                    "filename": str(d["filename"]),
                    "chunk_index": int(d["chunk_index"]),
                    "chunk_total": int(chunk_total),
                    "mime": mime,
                }
            )
        entries.sort(key=lambda e: (e["doc_id"], e["chunk_index"]))
        return entries

    async def _write_entries(self, entries: list[ChunkEntry]) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(KB_CHUNKS_HASH_KEY)
            if entries:
                mapping = {_chunk_field(e["doc_id"], e["chunk_index"]): _entry_to_json(e) for e in entries}
                pipe.hset(KB_CHUNKS_HASH_KEY, mapping=mapping)
            await pipe.execute()

    async def _scan_fields_for_doc(self, doc_id: str) -> AsyncIterator[str]:
        match = f"{CHUNK_FIELD_PREFIX}:{doc_id}:*"
        cursor = 0
        while True:
            cursor, batch = await self._redis.hscan(KB_CHUNKS_HASH_KEY, cursor=cursor, match=match)
            for field in batch:
                yield _decode(field)
            if cursor == 0:
                return

    async def append_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
        *,
        mime_type: str | None = None,
    ) -> None:
        existing = [field async for field in self._scan_fields_for_doc(doc_id)]
        total = len(chunks)
        mapping: dict[str, str] = {}
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            entry: ChunkEntry = {
                "text": text,
                "embedding": emb.astype(float).tolist(),
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "chunk_total": total,
                "mime": mime_type,
            }
            mapping[_chunk_field(doc_id, i)] = _entry_to_json(entry)
        async with self._redis.pipeline(transaction=True) as pipe:
            stale = [f for f in existing if f not in mapping]
            if stale:
                pipe.hdel(KB_CHUNKS_HASH_KEY, *stale)
            if mapping:
                pipe.hset(KB_CHUNKS_HASH_KEY, mapping=mapping)
            await pipe.execute()

    async def delete_document(self, doc_id: str) -> int:
        fields = [field async for field in self._scan_fields_for_doc(doc_id)]
        if not fields:
            # Legacy ``ck:<uuid>`` entries can't be found by prefix; fall back
            # to the abstract delete which rewrites the entire hash.
            return await super().delete_document(doc_id)
        removed = await self._redis.hdel(KB_CHUNKS_HASH_KEY, *fields)
        return int(removed)

    async def replace_kb_with_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
        *,
        mime_type: str | None = None,
    ) -> None:
        total = len(chunks)
        mapping: dict[str, str] = {}
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            entry: ChunkEntry = {
                "text": text,
                "embedding": emb.astype(float).tolist(),
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "chunk_total": total,
                "mime": mime_type,
            }
            mapping[_chunk_field(doc_id, i)] = _entry_to_json(entry)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(KB_CHUNKS_HASH_KEY)
            if mapping:
                pipe.hset(KB_CHUNKS_HASH_KEY, mapping=mapping)
            await pipe.execute()

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
