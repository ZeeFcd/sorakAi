"""Typed exception hierarchy for sorakAi.

Every failure raised by sorakAi code should be a subclass of
:class:`SorakaiError`. This lets handlers catch a single, well-known type
instead of relying on ``except Exception`` (which the engineering rules
ban via ``BLE001``). New subclasses get added next to the layer that owns
them - for example, a future ``AgentError`` would live in
``sorakai.chains.errors`` and inherit from :class:`SorakaiError`.
"""

from __future__ import annotations


class SorakaiError(Exception):
    """Base class for all sorakAi errors."""


class ConfigError(SorakaiError):
    """Invalid or missing configuration (env vars, settings, etc.)."""


class StoreError(SorakaiError):
    """Knowledge-base / vector-store / chat-history backend failure."""


class LLMError(SorakaiError):
    """LLM provider call failed (network, model, or response shape)."""


class EmbeddingError(SorakaiError):
    """Embeddings provider call failed."""


class DimensionMismatchError(EmbeddingError):
    """A query / chunk vector's dimension or model differs from what the KB was built with.

    Carries the conflicting metadata so handlers can render a 409 explaining
    whether the caller needs to re-ingest or pass ``replace_kb=true``.
    """

    def __init__(
        self,
        *,
        expected_provider: str,
        expected_model: str,
        expected_dim: int,
        actual_provider: str,
        actual_model: str,
        actual_dim: int,
    ) -> None:
        self.expected_provider = expected_provider
        self.expected_model = expected_model
        self.expected_dim = expected_dim
        self.actual_provider = actual_provider
        self.actual_model = actual_model
        self.actual_dim = actual_dim
        super().__init__(
            f"Embedding metadata mismatch: KB was built with "
            f"provider={expected_provider!r} model={expected_model!r} dim={expected_dim}, "
            f"but caller is using provider={actual_provider!r} model={actual_model!r} dim={actual_dim}."
        )


class RetrievalError(SorakaiError):
    """Retrieval pipeline produced no usable context."""
