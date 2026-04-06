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
    ctx, _n = retrieve_top_k_context(query_embedding, stored_embeddings, chunks, top_k=1)
    return ctx


def retrieve_top_k_context(
    query_embedding: np.ndarray,
    stored_embeddings: list[np.ndarray],
    chunks: list[str],
    top_k: int = 5,
) -> tuple[str, int]:
    """
    Return merged context from the top-k most similar chunks (by cosine similarity).
    Deduplicates identical chunk text. Second return value is number of chunks merged.
    """
    if not stored_embeddings or not chunks or len(stored_embeddings) != len(chunks):
        logger.warning("Empty or mismatched store")
        return "", 0

    k = max(1, min(top_k, len(chunks)))
    sims = [cosine_similarity(query_embedding, emb) for emb in stored_embeddings]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

    seen_text: set[str] = set()
    selected: list[str] = []
    for idx in ranked:
        if len(selected) >= k:
            break
        text = chunks[idx]
        if text in seen_text:
            continue
        seen_text.add(text)
        selected.append(text)

    if not selected:
        return "", 0

    logger.info("Top similarity: %.4f (using %s chunks)", sims[ranked[0]], len(selected))
    merged = "\n\n---\n\n".join(selected)
    return merged, len(selected)
