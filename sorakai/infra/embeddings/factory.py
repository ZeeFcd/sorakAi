"""Embeddings factory: env-driven lookup over a top-level registry.

Same OCP / DIP shape as :mod:`sorakai.infra.llm.factory`.
"""

from __future__ import annotations

from collections.abc import Callable

from sorakai.common.config import Settings
from sorakai.core.errors import ConfigError
from sorakai.infra.embeddings.base import Embeddings
from sorakai.infra.embeddings.char import build_char_embeddings
from sorakai.infra.embeddings.ollama import build_ollama_embeddings

EmbeddingsBuilder = Callable[[Settings], Embeddings]

EMBEDDINGS_REGISTRY: dict[str, EmbeddingsBuilder] = {
    "ollama": build_ollama_embeddings,
    "char": build_char_embeddings,
}


def register_embeddings(name: str, builder: EmbeddingsBuilder) -> None:
    """Register or replace an embeddings builder under ``name``."""
    EMBEDDINGS_REGISTRY[name] = builder


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the configured :class:`Embeddings` instance."""
    try:
        builder = EMBEDDINGS_REGISTRY[settings.embedding_provider]
    except KeyError as exc:
        raise ConfigError(
            f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}; registered: {sorted(EMBEDDINGS_REGISTRY)}"
        ) from exc
    return builder(settings)
