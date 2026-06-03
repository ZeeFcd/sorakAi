"""Adapter: expose a :class:`KnowledgeStore` (Wave 4) as a :class:`VectorStore` (Wave 5).

Used for ``VECTOR_STORE=redis`` and ``VECTOR_STORE=memory``. The whole point
is that this wave introduces the Protocol *without* changing on-disk layout
or the Wave 4 atomicity contracts - the adapter just translates between the
two surfaces.

The adapter also owns ``search``: it loads every chunk (the Wave 4
``KnowledgeStore`` is dense-in-memory-on-read), stacks the embeddings into
a matrix, and runs the same vectorised cosine top-k that powered Wave 2.
That's fine for small/medium KBs - true ANN ranking is the
:class:`~sorakai.infra.vector_store.qdrant.QdrantVectorStore`'s job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sorakai.common.retrieval import cosine_top_k, stack_embeddings
from sorakai.common.store import KnowledgeStore, RedisKnowledgeStore
from sorakai.core.errors import RetrievalError, StoreError
from sorakai.infra.vector_store.base import DocSummary, Hit, VectorDoc


class KnowledgeStoreVectorStore:
    """Bridge an existing :class:`KnowledgeStore` into the :class:`VectorStore` Protocol."""

    def __init__(self, knowledge_store: KnowledgeStore) -> None:
        self._store = knowledge_store

    @property
    def knowledge_store(self) -> KnowledgeStore:
        return self._store

    async def upsert(self, docs: list[VectorDoc]) -> None:
        if not docs:
            return
        # Group by doc_id so each underlying ``append_document`` call gets a
        # complete batch and benefits from the Wave 4 atomic HDEL+HSET swap.
        grouped: dict[str, list[VectorDoc]] = {}
        order: list[str] = []
        for d in docs:
            doc_id = self._required_str(d.metadata, "doc_id")
            if doc_id not in grouped:
                order.append(doc_id)
            grouped.setdefault(doc_id, []).append(d)

        for doc_id in order:
            batch = sorted(
                grouped[doc_id],
                key=lambda v: self._required_int(v.metadata, "chunk_index"),
            )
            filename = self._required_str(batch[0].metadata, "filename")
            mime = batch[0].metadata.get("mime")
            mime_type = mime if isinstance(mime, str) or mime is None else None
            chunks = [d.page_content for d in batch]
            embeddings = [np.asarray(d.embedding, dtype=np.float32) for d in batch]
            await self._store.append_document(
                doc_id,
                filename,
                chunks,
                embeddings,
                mime_type=mime_type,
            )

    async def delete_doc(self, doc_id: str) -> int:
        return await self._store.delete_document(doc_id)

    async def list_docs(self) -> list[DocSummary]:
        summaries = await self._store.list_documents()
        return [
            DocSummary(
                doc_id=s.doc_id,
                filename=s.filename,
                chunk_count=s.chunk_count,
                mime=s.mime,
            )
            for s in summaries
        ]

    async def search(
        self,
        query_vec: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        if filters:
            raise RetrievalError(
                "KnowledgeStoreVectorStore does not support payload filters - "
                "switch to VECTOR_STORE=qdrant for server-side filtering."
            )
        entries = await self._read_entries()
        if not entries:
            return []
        embeddings = [np.asarray(e["embedding"], dtype=np.float32) for e in entries]
        matrix = stack_embeddings(embeddings)
        indices, scores = cosine_top_k(query_vec, matrix, k=k)
        hits: list[Hit] = []
        for idx, score in zip(indices.tolist(), scores.tolist(), strict=True):
            entry = entries[int(idx)]
            hits.append(
                Hit(
                    page_content=entry["text"],
                    metadata={
                        "doc_id": entry["doc_id"],
                        "filename": entry["filename"],
                        "chunk_index": entry["chunk_index"],
                        "chunk_total": entry["chunk_total"],
                        "mime": entry["mime"],
                    },
                    score=float(score),
                )
            )
        return hits

    async def ping(self) -> bool:
        return await self._store.ping()

    async def aclose(self) -> None:
        if isinstance(self._store, RedisKnowledgeStore):
            await self._store.aclose()

    async def _read_entries(self) -> list[dict[str, Any]]:
        # ``_read_entries`` is the documented extension point on
        # :class:`KnowledgeStore` (used by both backends + the Wave 3 tests).
        entries = await self._store._read_entries()
        return [dict(e) for e in entries]

    @staticmethod
    def _required_str(metadata: dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise StoreError(f"VectorDoc.metadata[{key!r}] must be a non-empty string, got {value!r}")
        return value

    @staticmethod
    def _required_int(metadata: dict[str, Any], key: str) -> int:
        value = metadata.get(key)
        if not isinstance(value, int):
            raise StoreError(f"VectorDoc.metadata[{key!r}] must be an int, got {value!r}")
        return value
