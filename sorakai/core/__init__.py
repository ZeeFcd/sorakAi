"""Core building blocks shared across services.

This package is the new home for cross-cutting concerns (config, errors,
logging, schemas, etc.) as the overhaul plan migrates code out of
``sorakai.common``. New code should import from here; legacy modules in
``sorakai.common`` remain as compatibility re-exports until each wave
retires them.
"""

from sorakai.core.errors import (
    ConfigError,
    EmbeddingError,
    LLMError,
    RetrievalError,
    SorakaiError,
    StoreError,
)
from sorakai.core.logging import configure_logging, get_logger

__all__ = [
    "ConfigError",
    "EmbeddingError",
    "LLMError",
    "RetrievalError",
    "SorakaiError",
    "StoreError",
    "configure_logging",
    "get_logger",
]
