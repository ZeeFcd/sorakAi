"""Vector-store adapters (Wave 5).

Public surface mirrors the LLM / Embeddings factories: callers depend on the
Protocol + factory, not on concrete backend classes.
"""

from __future__ import annotations

from sorakai.infra.vector_store.base import DocSummary, Hit, VectorDoc, VectorStore
from sorakai.infra.vector_store.factory import (
    VECTOR_STORE_REGISTRY,
    get_vector_store,
    register_vector_store,
)

__all__ = [
    "VECTOR_STORE_REGISTRY",
    "DocSummary",
    "Hit",
    "VectorDoc",
    "VectorStore",
    "get_vector_store",
    "register_vector_store",
]
