"""Embeddings adapters and factory.

Same shape as :mod:`sorakai.infra.llm`: ``base`` re-exports
``langchain_core``'s ``Embeddings`` Protocol; ``ollama`` and ``char`` provide
the two shipped adapters; ``factory`` exposes the env-driven registry.
"""

from sorakai.infra.embeddings.base import Embeddings
from sorakai.infra.embeddings.factory import (
    EMBEDDINGS_REGISTRY,
    EmbeddingsBuilder,
    get_embeddings,
    register_embeddings,
)

__all__ = [
    "EMBEDDINGS_REGISTRY",
    "Embeddings",
    "EmbeddingsBuilder",
    "get_embeddings",
    "register_embeddings",
]
