import numpy as np

from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.retrieval")


def _pad_to_same_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = max(a.size, b.size)
    aa = np.zeros(m, dtype=float)
    bb = np.zeros(m, dtype=float)
    aa[: a.size] = a.astype(float).ravel()
    bb[: b.size] = b.astype(float).ravel()
    return aa, bb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _pad_to_same_length(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve_best_chunk(
    query_embedding: np.ndarray,
    stored_embeddings: list[np.ndarray],
    chunks: list[str],
) -> str:
    if not stored_embeddings or not chunks:
        logger.warning("Empty store")
        return ""
    sims = [cosine_similarity(query_embedding, emb) for emb in stored_embeddings]
    best_idx = int(np.argmax(sims))
    logger.info("Best similarity: %.4f", sims[best_idx])
    return chunks[best_idx]
