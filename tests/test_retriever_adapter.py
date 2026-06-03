"""Tests for :class:`VectorStoreRetriever`, :class:`HybridRetriever`, and RRF.

Backed by an in-memory :class:`KnowledgeStoreVectorStore` so the tests
exercise the same adapter the Wave 6 RAG chain uses in production for the
``VECTOR_STORE=memory`` configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

from sorakai.chains.retriever import (
    HybridRetriever,
    NoopReranker,
    VectorStoreRetriever,
    reciprocal_rank_fusion,
)
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.infra.embeddings.char import CharPseudoEmbeddings as CharEmbeddingsAdapter
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore


async def _make_store(chunks: list[tuple[str, str]]) -> KnowledgeStoreVectorStore:
    """``chunks`` = ``[(doc_id, text), ...]``."""
    store = InMemoryKnowledgeStore()
    embedder = CharEmbeddingsAdapter()
    for doc_id, text in chunks:
        vec = (await embedder.aembed_documents([text]))[0]
        await store.append_document(doc_id, f"{doc_id}.txt", [text], [np.asarray(vec, dtype=float)])
    return KnowledgeStoreVectorStore(store)


# ---------- VectorStoreRetriever ------------------------------------------------


@pytest.mark.asyncio
async def test_vector_store_retriever_returns_documents_with_scores() -> None:
    vstore = await _make_store(
        [
            ("d1", "Apples and oranges grow on trees"),
            ("d2", "Cars drive on roads"),
            ("d3", "Bananas are yellow fruits"),
        ]
    )
    retriever = VectorStoreRetriever(
        vector_store=vstore,
        embeddings=CharEmbeddingsAdapter(),
        k=2,
    )
    docs = await retriever.ainvoke("fruit")
    assert len(docs) == 2
    for d in docs:
        assert isinstance(d, Document)
        assert "score" in d.metadata
        assert "doc_id" in d.metadata


@pytest.mark.asyncio
async def test_vector_store_retriever_sync_path_raises() -> None:
    vstore = await _make_store([("d1", "anything")])
    retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=1)
    with pytest.raises(NotImplementedError):
        retriever.invoke("x")


# ---------- Reciprocal Rank Fusion ---------------------------------------------


def _doc(doc_id: str, chunk_index: int, content: str = "") -> Document:
    return Document(page_content=content or doc_id, metadata={"doc_id": doc_id, "chunk_index": chunk_index})


def test_rrf_promotes_documents_ranked_high_by_both_lists() -> None:
    a = [_doc("d1", 0), _doc("d2", 0), _doc("d3", 0)]
    b = [_doc("d2", 0), _doc("d1", 0), _doc("d3", 0)]
    fused = reciprocal_rank_fusion([a, b])
    # d1 and d2 each appear top-2 in one list and top-1 in the other; d3 is
    # consistently last in both. Either d1 or d2 wins depending on rounding,
    # but d3 must lose.
    assert [d.metadata["doc_id"] for d in fused[:2]] == sorted(["d1", "d2"])
    assert fused[2].metadata["doc_id"] == "d3"


def test_rrf_respects_weights() -> None:
    a = [_doc("d1", 0), _doc("d2", 0)]  # heavy weight: d1 wins
    b = [_doc("d2", 0), _doc("d1", 0)]  # heavy weight: d2 wins
    heavy_a = reciprocal_rank_fusion([a, b], weights=[10.0, 1.0])
    heavy_b = reciprocal_rank_fusion([a, b], weights=[1.0, 10.0])
    assert heavy_a[0].metadata["doc_id"] == "d1"
    assert heavy_b[0].metadata["doc_id"] == "d2"


def test_rrf_dedupes_and_annotates_score() -> None:
    a = [_doc("d1", 0), _doc("d1", 0)]
    b = [_doc("d1", 0)]
    fused = reciprocal_rank_fusion([a, b])
    assert len(fused) == 1
    assert "rrf_score" in fused[0].metadata


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_weights_length_must_match_rankings() -> None:
    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion([[_doc("d", 0)]], weights=[1.0, 2.0])


def test_rrf_top_n_truncates() -> None:
    docs = [_doc(f"d{i}", 0) for i in range(10)]
    fused = reciprocal_rank_fusion([docs], top_n=3)
    assert len(fused) == 3


# ---------- HybridRetriever ----------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_retriever_lazy_builds_on_first_query() -> None:
    vstore = await _make_store([("d1", "Quantum computing uses qubits")])
    vector_retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=1)
    hybrid = HybridRetriever(vector_retriever=vector_retriever, vector_store=vstore, top_k=1)

    assert hybrid.initialized is False
    assert hybrid.bm25 is None

    docs = await hybrid.ainvoke("qubits")
    assert hybrid.initialized is True
    assert hybrid.bm25 is not None
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_hybrid_retriever_arebuild_picks_up_new_chunks() -> None:
    vstore = await _make_store([("d1", "Original text here")])
    vector_retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=2)
    hybrid = HybridRetriever(vector_retriever=vector_retriever, vector_store=vstore, top_k=2)
    await hybrid._ensure_built()
    assert hybrid.bm25 is not None
    initial_corpus_len = len(hybrid.corpus_docs)

    embedder = CharEmbeddingsAdapter()
    new_vec = (await embedder.aembed_documents(["Fresh new chunk added"]))[0]
    await vstore._store.append_document(
        "d2",
        "d2.txt",
        ["Fresh new chunk added"],
        [np.asarray(new_vec, dtype=float)],
    )

    await hybrid.arebuild()
    assert len(hybrid.corpus_docs) > initial_corpus_len


@pytest.mark.asyncio
async def test_hybrid_retriever_falls_back_to_vector_when_corpus_empty() -> None:
    store = InMemoryKnowledgeStore()
    vstore = KnowledgeStoreVectorStore(store)
    vector_retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=2)
    hybrid = HybridRetriever(vector_retriever=vector_retriever, vector_store=vstore, top_k=2)
    docs = await hybrid.ainvoke("anything")
    assert docs == []
    assert hybrid.bm25 is None


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_top_k() -> None:
    vstore = await _make_store([(f"d{i}", f"Document about topic {i} with shared keyword sample") for i in range(8)])
    vector_retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=5)
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store=vstore,
        top_k=3,
        rerank_top_n=8,
    )
    docs = await hybrid.ainvoke("sample")
    assert len(docs) == 3


@pytest.mark.asyncio
async def test_snapshot_corpus_dispatches_to_qdrant_scroll() -> None:
    """``snapshot_corpus`` must work over Qdrant too so the hybrid retriever
    isn't silently degraded when :envvar:`VECTOR_STORE=qdrant`."""
    qdrant_client = pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient

    from sorakai.chains.retriever import snapshot_corpus
    from sorakai.infra.vector_store.base import VectorDoc
    from sorakai.infra.vector_store.qdrant import QdrantVectorStore

    del qdrant_client
    qvs = QdrantVectorStore(
        client=AsyncQdrantClient(":memory:"),
        collection="hybrid_snapshot",
        owns_client=True,
    )
    try:
        embedder = CharEmbeddingsAdapter()
        texts = ["bears live in forests", "fish swim in rivers", "birds fly through air"]
        vecs = await embedder.aembed_documents(texts)
        docs = [
            VectorDoc(
                page_content=t,
                embedding=np.asarray(v, dtype=float),
                metadata={
                    "doc_id": f"qd-{i}",
                    "filename": f"qd-{i}.txt",
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "mime": None,
                },
            )
            for i, (t, v) in enumerate(zip(texts, vecs, strict=True))
        ]
        await qvs.upsert(docs)

        snapshot = await snapshot_corpus(qvs)
        assert {d.page_content for d in snapshot} == set(texts)
        assert all(d.metadata.get("doc_id", "").startswith("qd-") for d in snapshot)
    finally:
        await qvs.aclose()


@pytest.mark.asyncio
async def test_snapshot_corpus_warns_on_unknown_backend(caplog: pytest.LogCaptureFixture) -> None:
    """Non-recognised vector stores must degrade with a logged warning, not crash."""
    from sorakai.chains.retriever import snapshot_corpus

    class _MysteryStore:
        async def list_docs(self):  # type: ignore[no-untyped-def]
            return []

    with caplog.at_level("WARNING"):
        out = await snapshot_corpus(_MysteryStore())  # type: ignore[arg-type]
    assert out == []
    assert any("does not implement a corpus scroll" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_retriever_with_noop_reranker_truncates_to_top_k() -> None:
    vstore = await _make_store([(f"d{i}", f"text {i} with sample keyword") for i in range(5)])
    vector_retriever = VectorStoreRetriever(vector_store=vstore, embeddings=CharEmbeddingsAdapter(), k=5)
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store=vstore,
        top_k=2,
        rerank_top_n=5,
        reranker=NoopReranker(),
    )
    docs = await hybrid.ainvoke("sample")
    assert len(docs) == 2
