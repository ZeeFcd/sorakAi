"""Thin embeddings shim used by ingest and RAG.

Delegates to :func:`sorakai.infra.embeddings.get_embeddings`; the actual
provider-specific code lives in ``sorakai/infra/embeddings/<provider>.py``.
The shim exists for two reasons: (1) it keeps the legacy import path
``from sorakai.common.embedding import embed_chunks`` working while we move
code into ``sorakai.infra``, and (2) it converts to ``numpy`` arrays for the
current cosine-similarity retrieval code (Wave 2 stacks these into a matrix).
"""

from __future__ import annotations

import numpy as np

from sorakai.common.config import get_settings
from sorakai.core.logging import get_logger
from sorakai.infra.embeddings import get_embeddings

logger = get_logger(__name__)


async def embed_chunks(chunks: list[str]) -> list[np.ndarray]:
    """Embed ``chunks`` with the currently-configured provider."""
    if not chunks:
        return []
    settings = get_settings()
    embeddings = get_embeddings(settings)
    logger.info("Embedding %d chunks via provider=%s", len(chunks), settings.embedding_provider)
    vectors = await embeddings.aembed_documents(chunks)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


async def embed_query(text: str) -> np.ndarray:
    """Embed a single query string with the currently-configured provider."""
    settings = get_settings()
    embeddings = get_embeddings(settings)
    vector = await embeddings.aembed_query(text)
    return np.asarray(vector, dtype=np.float32)
