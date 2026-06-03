"""LCEL pipelines (Wave 6).

The contents of this package are intentionally never imported by the
infrastructure layer (``sorakai.infra.*``) - chains compose factories, not
the other way around. Adding a new chain or wiring a different prompt is
done here without touching adapters or services.
"""

from __future__ import annotations

from sorakai.chains.rag_chain import build_rag_chain
from sorakai.chains.retriever import HybridRetriever, VectorStoreRetriever, reciprocal_rank_fusion

__all__ = [
    "HybridRetriever",
    "VectorStoreRetriever",
    "build_rag_chain",
    "reciprocal_rank_fusion",
]
