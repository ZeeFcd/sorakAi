"""LangChain retrievers wired on top of the Wave 5 :class:`VectorStore`.

Two adapters:

- :class:`VectorStoreRetriever` - the bare minimum: embed the query through
  the embeddings adapter, ask the vector store for ``top_k``, return
  LangChain :class:`~langchain_core.documents.Document` rows. Used directly
  when ``HYBRID_RETRIEVER_ENABLED=false``.
- :class:`HybridRetriever` - lexical (BM25) + semantic (vector) fused via
  Reciprocal Rank Fusion. BM25 is rebuilt lazily from the corpus snapshot
  visible at the start of the chain build; we don't try to keep it live with
  ingest mutations in Wave 6 (a sidecar invalidation hook is a Wave 8
  observability concern). For small/medium KBs this is fine; large KBs
  should turn the hybrid off (and rely on a real ANN reranker upstream of
  Qdrant) until that hook lands.

The optional ``rerank`` hook is a Protocol so Wave 7+ can drop in a real
``bge-reranker-base`` (or Cohere / Voyage) implementation without touching
the chain. The default no-op reranker keeps Wave 6 local-only fast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from sorakai.core.logging import get_logger
from sorakai.infra.embeddings import Embeddings
from sorakai.infra.vector_store import Hit, VectorStore
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore
from sorakai.infra.vector_store.qdrant import QdrantVectorStore

logger = get_logger(__name__)

_RRF_K = 60
"""Reciprocal Rank Fusion constant (Cormack et al., 2009). 60 is the
canonical value; smaller K sharpens the head, larger K flattens fusion."""


def _hit_to_document(hit: Hit) -> Document:
    metadata = dict(hit.metadata)
    metadata["score"] = float(hit.score)
    return Document(page_content=hit.page_content, metadata=metadata)


def _document_key(doc: Document) -> tuple[str, int]:
    """Stable identity for fusion: ``(doc_id, chunk_index)``.

    Falls back to the raw page content when metadata is incomplete (e.g.
    docs produced by something other than our adapters).
    """
    meta = doc.metadata or {}
    doc_id = str(meta.get("doc_id") or "")
    chunk_index = int(meta.get("chunk_index", -1))
    if doc_id:
        return (doc_id, chunk_index)
    # ``hash(str)`` is stable per-process, sufficient for in-request fusion.
    return ("__no_doc_id__", hash(doc.page_content))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]],
    *,
    weights: Sequence[float] | None = None,
    k: int = _RRF_K,
    top_n: int | None = None,
) -> list[Document]:
    """Fuse multiple ranked lists into one via weighted Reciprocal Rank Fusion.

    ``rankings[i]`` is the ranked list emitted by retriever ``i``; ranks
    inside each list are 1-based positions (0 = best). ``weights`` defaults
    to equal weighting. Documents are deduplicated by
    :func:`_document_key` and ordered by descending fused score; the winning
    document instance is whichever ranking placed it highest (so the
    metadata reflects the strongest evidence).
    """
    if not rankings:
        return []
    n = len(rankings)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != rankings length {n}")

    fused_scores: dict[tuple[str, int], float] = {}
    best_doc: dict[tuple[str, int], tuple[float, Document]] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank_idx, doc in enumerate(ranking):
            key = _document_key(doc)
            contribution = weight / (k + rank_idx + 1)
            fused_scores[key] = fused_scores.get(key, 0.0) + contribution
            # Keep the highest-ranking instance of each doc (lowest rank_idx).
            existing = best_doc.get(key)
            if existing is None or rank_idx < existing[0]:
                best_doc[key] = (rank_idx, doc)

    ordered_keys = sorted(fused_scores, key=lambda k_: -fused_scores[k_])
    if top_n is not None:
        ordered_keys = ordered_keys[:top_n]
    out: list[Document] = []
    for key in ordered_keys:
        _, doc = best_doc[key]
        # Annotate the fused score so downstream prompts/loggers can see it.
        new_meta = dict(doc.metadata)
        new_meta["rrf_score"] = float(fused_scores[key])
        out.append(Document(page_content=doc.page_content, metadata=new_meta))
    return out


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + lowercase tokenizer for BM25."""
    return [tok for tok in text.lower().split() if tok]


@runtime_checkable
class Reranker(Protocol):
    """Pluggable cross-encoder reranker (Wave 7+ ships a real implementation)."""

    async def arerank(self, query: str, docs: list[Document], *, top_n: int) -> list[Document]: ...


class NoopReranker:
    """Default reranker: returns the top ``top_n`` of the fused list unchanged."""

    async def arerank(self, query: str, docs: list[Document], *, top_n: int) -> list[Document]:
        del query
        return docs[:top_n]


class VectorStoreRetriever(BaseRetriever):
    """Bridge a Wave 5 :class:`VectorStore` to LangChain's :class:`BaseRetriever`."""

    vector_store: VectorStore
    embeddings: Embeddings
    k: int = 5
    filters: dict[str, Any] | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        vec = await self.embeddings.aembed_query(query)
        hits = await self.vector_store.search(np.asarray(vec, dtype=np.float32), self.k, self.filters)
        return [_hit_to_document(h) for h in hits]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # The chain is always ``ainvoke``-ed; the sync path is provided so
        # ``LangSmith`` callbacks that probe both surfaces don't blow up.
        del query, run_manager
        raise NotImplementedError("VectorStoreRetriever is async-only; use ``ainvoke`` / ``aget_relevant_documents``.")


class HybridRetriever(BaseRetriever):
    """BM25 + vector retriever fused via Reciprocal Rank Fusion.

    The BM25 index is built **lazily on first query** (and cached) rather
    than at construction time. That keeps the lifespan startup cheap and
    makes "seed corpus, then ask" test patterns work without any rebuild
    plumbing. Wave 8 wires a ``KB changed`` invalidation hook so live
    ingestions in long-running services pick up changes mid-flight; until
    then call :meth:`arebuild` from the ingest path if you need it.
    """

    vector_retriever: VectorStoreRetriever
    vector_store: VectorStore
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    top_k: int = 5
    rerank_top_n: int = 5
    reranker: Reranker | None = None
    corpus_docs: list[Document] = Field(default_factory=list)
    bm25: BM25Okapi | None = None
    initialized: bool = False

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    async def arebuild(self) -> None:
        """Snapshot the corpus and rebuild BM25.

        Idempotent. Safe to call from a write path after large ingestions
        or whenever you want fresh lexical ranking. Reads no LLM state.
        """
        docs = await snapshot_corpus(self.vector_store)
        self.corpus_docs = docs
        if docs:
            self.bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs])
        else:
            self.bm25 = None
        self.initialized = True

    async def _ensure_built(self) -> None:
        if not self.initialized:
            await self.arebuild()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        await self._ensure_built()
        vector_docs = await self.vector_retriever._aget_relevant_documents(
            query,
            run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
        )
        if self.bm25 is None or not self.corpus_docs:
            return vector_docs[: self.top_k]

        scores = self.bm25.get_scores(_tokenize(query))
        # rank-bm25 returns float scores; sort descending and take the same
        # top-N as the vector path so RRF gets a fair fight.
        top_n = max(self.top_k * 2, len(vector_docs))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
        bm25_docs: list[Document] = []
        for i in order:
            if not math.isfinite(scores[i]) or scores[i] <= 0:
                continue
            base = self.corpus_docs[i]
            meta = dict(base.metadata)
            meta["bm25_score"] = float(scores[i])
            bm25_docs.append(Document(page_content=base.page_content, metadata=meta))

        fused = reciprocal_rank_fusion(
            [bm25_docs, vector_docs],
            weights=[self.bm25_weight, self.vector_weight],
            top_n=self.rerank_top_n,
        )
        if self.reranker is None:
            return fused[: self.top_k]
        reranked = await self.reranker.arerank(query, fused, top_n=self.top_k)
        return reranked[: self.top_k]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del query, run_manager
        raise NotImplementedError("HybridRetriever is async-only; use ``ainvoke``.")


async def snapshot_corpus(vector_store: VectorStore) -> list[Document]:
    """Pull every chunk out of a vector store, as :class:`Document` rows.

    Used by :class:`HybridRetriever` to seed BM25. Backends dispatch:

    - :class:`KnowledgeStoreVectorStore` reuses its in-memory mirror.
    - :class:`QdrantVectorStore` scrolls the underlying collection
      (payload-only - no vectors transferred).

    Returns ``[]`` for any other implementation; the hybrid retriever then
    transparently degrades to vector-only retrieval, logging a one-line
    warning so misconfigurations surface in observability.
    """
    if isinstance(vector_store, KnowledgeStoreVectorStore):
        entries = await vector_store._read_entries()
        return [
            Document(
                page_content=str(e["text"]),
                metadata={
                    "doc_id": e["doc_id"],
                    "filename": e["filename"],
                    "chunk_index": e["chunk_index"],
                    "chunk_total": e["chunk_total"],
                    "mime": e["mime"],
                },
            )
            for e in entries
        ]
    if isinstance(vector_store, QdrantVectorStore):
        return await _snapshot_qdrant_corpus(vector_store)
    logger.warning(
        "snapshot_corpus: %s does not implement a corpus scroll; hybrid retriever falls back to vector-only",
        type(vector_store).__name__,
    )
    return []


async def _snapshot_qdrant_corpus(vector_store: Any) -> list[Document]:
    """Scroll a Qdrant collection in pages, return :class:`Document` rows.

    Pulls payload only (``with_vectors=False``) so the BM25 build stays
    cheap even for collections with multi-thousand-dim vectors.
    """
    if not await vector_store._collection_exists():
        return []
    out: list[Document] = []
    offset: Any = None
    while True:
        records, offset = await vector_store._client.scroll(
            collection_name=vector_store._collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            payload = rec.payload or {}
            out.append(
                Document(
                    page_content=str(payload.get("text", "")),
                    metadata={
                        "doc_id": payload.get("doc_id"),
                        "filename": payload.get("filename"),
                        "chunk_index": payload.get("chunk_index"),
                        "chunk_total": payload.get("chunk_total"),
                        "mime": payload.get("mime"),
                    },
                )
            )
        if offset is None:
            break
    return out
