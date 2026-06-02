"""Ollama embeddings adapter.

Wave 1 ships the minimal wrapper around ``OllamaEmbeddings`` from
``langchain-ollama``. Wave 2 enhances this module with explicit batching,
bounded concurrency, and a switch between the new ``/api/embed`` endpoint and
the legacy ``/api/embeddings`` one.
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from sorakai.common.config import Settings
from sorakai.infra.embeddings.base import Embeddings


def build_ollama_embeddings(settings: Settings) -> Embeddings:
    """Construct ``OllamaEmbeddings`` from project settings."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
