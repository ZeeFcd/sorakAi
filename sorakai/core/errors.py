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


class RetrievalError(SorakaiError):
    """Retrieval pipeline produced no usable context."""
