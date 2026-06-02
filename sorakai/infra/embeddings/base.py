"""Canonical embeddings abstract type for sorakAi.

We reuse ``langchain_core.embeddings.Embeddings`` so any host that already
has a LangChain adapter (Ollama, ``sentence-transformers``, OpenAI, ...) can
be slotted in by writing a tiny factory and registering it.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

__all__ = ["Embeddings"]
