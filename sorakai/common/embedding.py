from typing import List

import numpy as np

from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.embedding")


def embed_chunks(chunks: List[str]) -> list[np.ndarray]:
    """Deterministic pseudo-embeddings for MVP (replace with sentence-transformers / API in production)."""
    logger.info("Embedding %s chunks", len(chunks))
    return [np.array([float(ord(c) % 128) for c in chunk[:512]], dtype=float) for chunk in chunks]
