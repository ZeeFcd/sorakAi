"""Vector-store factory: env-driven lookup over a top-level registry.

Same OCP / DIP shape as :mod:`sorakai.infra.llm.factory` and
:mod:`sorakai.infra.embeddings.factory`. Adding a new backend is one file
under ``sorakai/infra/vector_store/`` plus one entry in
:data:`VECTOR_STORE_REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Callable

from sorakai.common.config import Settings
from sorakai.common.store import InMemoryKnowledgeStore, RedisKnowledgeStore
from sorakai.core.errors import ConfigError
from sorakai.infra.vector_store.base import VectorStore
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore
from sorakai.infra.vector_store.qdrant import build_qdrant_vector_store_from_url

VectorStoreBuilder = Callable[[Settings], VectorStore]


def _build_memory(_: Settings) -> VectorStore:
    return KnowledgeStoreVectorStore(InMemoryKnowledgeStore())


def _build_redis(settings: Settings) -> VectorStore:
    if not settings.redis_url:
        raise ConfigError("VECTOR_STORE=redis requires REDIS_URL to be set.")
    return KnowledgeStoreVectorStore(RedisKnowledgeStore(settings.redis_url))


def _build_qdrant(settings: Settings) -> VectorStore:
    return build_qdrant_vector_store_from_url(settings.qdrant_url, settings.qdrant_collection)


VECTOR_STORE_REGISTRY: dict[str, VectorStoreBuilder] = {
    "memory": _build_memory,
    "redis": _build_redis,
    "qdrant": _build_qdrant,
}


def register_vector_store(name: str, builder: VectorStoreBuilder) -> None:
    """Register or replace a vector-store builder under ``name``."""
    VECTOR_STORE_REGISTRY[name] = builder


def get_vector_store(settings: Settings) -> VectorStore:
    """Return the configured :class:`VectorStore` instance."""
    try:
        builder = VECTOR_STORE_REGISTRY[settings.vector_store]
    except KeyError as exc:
        raise ConfigError(
            f"Unknown VECTOR_STORE={settings.vector_store!r}; registered: {sorted(VECTOR_STORE_REGISTRY)}"
        ) from exc
    return builder(settings)
