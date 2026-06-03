"""Provider-agnostic vector-store interface.

The Wave 1 / Wave 2 work made *which LLM and which embedder run* a
configuration choice. Wave 5 does the same for *where chunks + vectors
live*: ingest and (later) RAG depend on the :class:`VectorStore` Protocol
defined here instead of importing a concrete backend.

Today the registry ships:

- ``KnowledgeStoreVectorStore`` (``VECTOR_STORE=redis`` and ``=memory``) -
  a thin adapter over the existing Wave 4 :mod:`sorakai.common.store`. Keeps
  the entire test suite green and lets us swap backends without touching
  the ingest / RAG handlers in this wave.
- :class:`~sorakai.infra.vector_store.qdrant.QdrantVectorStore` -
  Qdrant via the official async client; collection per env, cosine
  distance, payload carries the Wave 3 chunk metadata.

Future backends (pgvector, Milvus, Weaviate, Redis Search-VECTOR, ...) are
one adapter file + one ``register_vector_store(...)`` call away.

Design notes
------------

- Embeddings are **precomputed by the caller**. The dim-guard
  (:mod:`sorakai.common.kb_meta`) needs to validate a vector before any
  side effect lands; pushing the embed call inside the vector store would
  make that check awkward and force every backend to learn the embedder
  factory. ``VectorDoc`` therefore bundles ``page_content + embedding +
  metadata`` and ``search`` takes a precomputed ``query_vec``.
- ``upsert`` is **idempotent per ``doc_id``**: re-uploading a doc replaces
  its existing chunks atomically (the contract Wave 4 introduced for the
  Redis backend). Qdrant gets the same semantics via point-id derivation
  from ``(doc_id, chunk_index)``.
- ``search`` returns :class:`Hit` rows already sorted by descending score
  (cosine similarity, higher = better) so callers don't have to know the
  backend's native ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorDoc:
    """One chunk to upsert: text + precomputed embedding + Wave 3 metadata.

    ``metadata`` is required to carry at minimum ``doc_id``, ``filename``,
    ``chunk_index``, ``chunk_total`` and ``mime`` so adapters can derive
    deterministic point ids, support ``delete_doc(doc_id)``, and return
    rich :class:`Hit` rows from :meth:`VectorStore.search`.
    """

    page_content: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocSummary:
    """Per-document roll-up returned by :meth:`VectorStore.list_docs`."""

    doc_id: str
    filename: str
    chunk_count: int
    mime: str | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieval result, sorted by descending ``score`` in the response list."""

    page_content: str
    metadata: dict[str, Any]
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """The minimal surface every KB backend must implement."""

    async def upsert(self, docs: list[VectorDoc]) -> None:
        """Add or replace chunks. Re-using a ``doc_id`` overwrites cleanly."""

    async def delete_doc(self, doc_id: str) -> int:
        """Remove every chunk for ``doc_id``. Returns the count removed."""

    async def list_docs(self) -> list[DocSummary]:
        """Roll-up by ``doc_id``: filename, chunk_count, mime."""

    async def search(
        self,
        query_vec: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Top-``k`` cosine matches, sorted by descending score.

        ``filters`` is forwarded as-is to backends that support payload
        filtering (Qdrant); adapters that don't yet implement filtering
        raise :class:`~sorakai.core.errors.RetrievalError` when given a
        non-empty filter so callers never silently get unfiltered results.
        """

    async def ping(self) -> bool:
        """Cheap liveness probe (used by ``/ready``)."""

    async def aclose(self) -> None:
        """Release any underlying connections/clients."""
