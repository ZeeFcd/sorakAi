"""Qdrant adapter for :class:`~sorakai.infra.vector_store.base.VectorStore`.

Uses :class:`qdrant_client.AsyncQdrantClient` so every operation is native
async. The collection is created lazily on the first :meth:`upsert` (we only
know the vector dim once the first batch shows up - this matches the Wave 2
dim-guard philosophy: don't pre-commit a vector space before you've seen one).

Point ids are derived deterministically from ``(doc_id, chunk_index)`` so
re-ingest of the same chunk overwrites the previous version - same contract
as Wave 4 introduced for the Redis backend.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from sorakai.core.errors import StoreError
from sorakai.core.logging import get_logger
from sorakai.infra.vector_store.base import DocSummary, Hit, VectorDoc

logger = get_logger(__name__)

# Stable namespace for deterministic UUID5 point ids. Bumping this is a
# destructive migration - existing points will be orphaned.
_POINT_NAMESPACE = uuid.UUID("6f8c6f4e-2d3a-4f0d-9b1a-7c9f5d9d5d5e")


class QdrantVectorStore:
    """Qdrant-backed :class:`VectorStore`.

    Parameters
    ----------
    client:
        An :class:`AsyncQdrantClient` (real HTTP/gRPC or ``:memory:`` for tests).
        The adapter does not own the client's lifecycle by default; pass
        ``owns_client=True`` to have :meth:`aclose` close it.
    collection:
        Collection name. Created lazily on first upsert.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        *,
        owns_client: bool = True,
    ) -> None:
        self._client = client
        self._collection = collection
        self._owns_client = owns_client
        self._collection_ready = False

    # ---- public API ------------------------------------------------------

    async def upsert(self, docs: list[VectorDoc]) -> None:
        if not docs:
            return
        await self._ensure_collection(int(docs[0].embedding.size))

        # Group by doc_id so we get the Wave-4 "re-ingest overwrites" contract:
        # any stale chunks for a doc_id we're about to write are dropped first.
        # We can't rely on UUID5 point-id collisions alone because a re-ingest
        # may produce fewer chunks than the previous run (leaving orphans).
        affected_doc_ids = {self._required_str(d.metadata, "doc_id") for d in docs}
        for doc_id in affected_doc_ids:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=_doc_filter(doc_id)),
                wait=True,
            )

        points: list[PointStruct] = []
        for d in docs:
            doc_id = self._required_str(d.metadata, "doc_id")
            chunk_index = self._required_int(d.metadata, "chunk_index")
            vec = np.asarray(d.embedding, dtype=np.float32)
            payload: dict[str, Any] = {
                "doc_id": doc_id,
                "filename": self._required_str(d.metadata, "filename"),
                "chunk_index": chunk_index,
                "chunk_total": int(d.metadata.get("chunk_total", -1)),
                "mime": d.metadata.get("mime"),
                "text": d.page_content,
            }
            points.append(
                PointStruct(
                    id=_point_id(doc_id, chunk_index),
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def delete_doc(self, doc_id: str) -> int:
        if not await self._collection_exists():
            return 0
        # Server-side count BEFORE delete so the return value matches the
        # number of chunks actually removed.
        count = await self._client.count(
            collection_name=self._collection,
            count_filter=_doc_filter(doc_id),
            exact=True,
        )
        removed = int(count.count)
        if removed == 0:
            return 0
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=_doc_filter(doc_id)),
            wait=True,
        )
        return removed

    async def list_docs(self) -> list[DocSummary]:
        if not await self._collection_exists():
            return []
        # Scroll the whole collection projecting only the metadata we need.
        # Fine for small/medium KBs; large multi-million-chunk corpora should
        # query a sidecar metadata collection instead - left for later waves.
        per_doc: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        offset: Any = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=512,
                offset=offset,
                with_payload=["doc_id", "filename", "mime"],
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                doc_id = payload.get("doc_id")
                if not isinstance(doc_id, str):
                    continue
                entry = per_doc.get(doc_id)
                if entry is None:
                    per_doc[doc_id] = {
                        "filename": str(payload.get("filename", "")),
                        "mime": payload.get("mime"),
                        "count": 1,
                    }
                    order.append(doc_id)
                else:
                    entry["count"] += 1
            if offset is None:
                break
        return [
            DocSummary(
                doc_id=doc_id,
                filename=str(per_doc[doc_id]["filename"]),
                chunk_count=int(per_doc[doc_id]["count"]),
                mime=per_doc[doc_id]["mime"] if isinstance(per_doc[doc_id]["mime"], str) else None,
            )
            for doc_id in order
        ]

    async def search(
        self,
        query_vec: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        if not await self._collection_exists():
            return []
        if query_vec.size == 0:
            return []
        result = await self._client.query_points(
            collection_name=self._collection,
            query=np.asarray(query_vec, dtype=np.float32).tolist(),
            limit=max(1, k),
            query_filter=_payload_filter(filters),
            with_payload=True,
            with_vectors=False,
        )
        hits: list[Hit] = []
        for sp in result.points:
            payload = sp.payload or {}
            text = payload.get("text", "")
            hits.append(
                Hit(
                    page_content=str(text),
                    metadata={
                        "doc_id": payload.get("doc_id"),
                        "filename": payload.get("filename"),
                        "chunk_index": payload.get("chunk_index"),
                        "chunk_total": payload.get("chunk_total"),
                        "mime": payload.get("mime"),
                    },
                    score=float(sp.score),
                )
            )
        return hits

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception:
            # Any failure (network, bad URL, server down) means "not reachable".
            return False
        return True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    # ---- internals -------------------------------------------------------

    async def _collection_exists(self) -> bool:
        if self._collection_ready:
            return True
        try:
            exists = await self._client.collection_exists(self._collection)
        except UnexpectedResponse:
            exists = False
        if exists:
            self._collection_ready = True
        return exists

    async def _ensure_collection(self, dim: int) -> None:
        if dim <= 0:
            raise StoreError(f"Refusing to create Qdrant collection with non-positive dim {dim}")
        if await self._collection_exists():
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._collection_ready = True
        logger.info("Created Qdrant collection %s (dim=%d, distance=cosine)", self._collection, dim)

    # ---- shared validation helpers --------------------------------------

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


def _point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 so re-ingest overwrites instead of duplicating."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


def _doc_filter(doc_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def _payload_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Map a flat ``{field: value}`` dict to a Qdrant must-match filter.

    Only equality matches are supported in Wave 5 - that's enough for the
    Wave 6 LCEL chain to filter by ``doc_id`` (e.g. "answer using only this
    document"). Range / full-text / nested filters can land in a later wave
    by translating the same dict into richer ``FieldCondition``s.
    """
    if not filters:
        return None
    must = [FieldCondition(key=str(k), match=MatchValue(value=v)) for k, v in filters.items()]
    return Filter(must=must)


def build_qdrant_vector_store_from_url(url: str, collection: str) -> QdrantVectorStore:
    """Factory helper: create an :class:`AsyncQdrantClient` from a URL.

    Supports HTTP, gRPC, and the ``":memory:"`` test transport. The returned
    store owns the client and will close it from :meth:`aclose`.
    """
    client = AsyncQdrantClient(url=url) if url != ":memory:" else AsyncQdrantClient(":memory:")
    return QdrantVectorStore(client=client, collection=collection, owns_client=True)
