"""Vectorised cosine-similarity retrieval (Wave 2).

The Wave 0/1 implementation called :func:`cosine_similarity` once per stored
chunk and silently zero-padded mismatched dimensions via
``_pad_to_same_length`` - that mask hid the bug where a query embedded by
provider/model A was being scored against chunks embedded by B.

This module now:

- Stacks all stored vectors into a single ``(N, D)`` matrix and computes
  ``query @ matrix.T`` in one BLAS call (massively faster on large KBs and
  shifts overall query latency back onto the LLM where it belongs).
- Raises :class:`~sorakai.core.errors.DimensionMismatchError`-style
  ``ValueError`` when stored vectors disagree on dim, instead of padding.

The dimension *provider/model* guard lives one layer up in
:mod:`sorakai.common.kb_meta` so the handlers can respond with the right
HTTP status; this module only guarantees mathematical correctness.
"""

from __future__ import annotations

import numpy as np

from sorakai.core.logging import get_logger

logger = get_logger("sorakai.retrieval")

_EPS = 1e-12


def stack_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Stack a list of 1-D vectors into a ``(N, D)`` matrix.

    Raises ``ValueError`` if any pair of vectors has a different length;
    this guards against the silent zero-padding bug fixed in Wave 2.
    """
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    first_dim = embeddings[0].size
    for i, emb in enumerate(embeddings):
        if emb.ndim != 1:
            raise ValueError(f"Embedding at index {i} is not 1-D (got shape {emb.shape!r})")
        if emb.size != first_dim:
            raise ValueError(
                f"Embedding at index {i} has dim {emb.size}, expected {first_dim}; "
                "this usually means the KB contains chunks from a different embedding model."
            )
    return np.stack([e.astype(np.float32, copy=False) for e in embeddings])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors of the same dim."""
    if a.size != b.size:
        raise ValueError(f"Cosine similarity requires equal dims (got {a.size} vs {b.size})")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_top_k(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(top_k_indices, top_k_scores)`` via a single matmul.

    Both arrays are sorted by descending score. ``matrix`` is treated as
    ``(N, D)``; ``query`` as ``(D,)``. Empty inputs return empty arrays.
    """
    if matrix.size == 0 or query.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if query.size != matrix.shape[1]:
        raise ValueError(
            f"Query dim {query.size} does not match stored dim {matrix.shape[1]}; "
            "re-ingest the KB with the current embedding model or pass replace_kb=true."
        )

    q = query.astype(np.float32, copy=False)
    q_norm = float(np.linalg.norm(q))
    row_norms = np.linalg.norm(matrix, axis=1)
    if q_norm < _EPS:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    scores = (matrix @ q) / (row_norms * q_norm + _EPS)
    k = max(1, min(k, scores.size))
    # ``argpartition`` is O(N) for the top-k, then we sort just those k entries.
    top_unsorted = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[top_unsorted])
    top_indices = top_unsorted[order]
    return top_indices.astype(np.int64), scores[top_indices].astype(np.float32)


def retrieve_best_chunk(
    query_embedding: np.ndarray,
    stored_embeddings: list[np.ndarray],
    chunks: list[str],
) -> str:
    """Back-compat helper: return the best chunk (or '' if KB empty)."""
    ctx, _n = retrieve_top_k_context(query_embedding, stored_embeddings, chunks, top_k=1)
    return ctx


def retrieve_top_k_context(
    query_embedding: np.ndarray,
    stored_embeddings: list[np.ndarray],
    chunks: list[str],
    top_k: int = 5,
) -> tuple[str, int]:
    """Return merged context from the top-k most similar chunks.

    Deduplicates identical chunk text. Second return value is number of
    chunks merged into the context string.

    Raises ``ValueError`` (via :func:`stack_embeddings` / :func:`cosine_top_k`)
    when the KB or query has inconsistent dims - that's the dim-guard hook the
    handlers translate to a ``409``.
    """
    if not stored_embeddings or not chunks or len(stored_embeddings) != len(chunks):
        logger.warning("Empty or mismatched store")
        return "", 0

    matrix = stack_embeddings(stored_embeddings)
    top_indices, top_scores = cosine_top_k(query_embedding, matrix, k=top_k)
    if top_indices.size == 0:
        return "", 0

    seen_text: set[str] = set()
    selected: list[str] = []
    for idx in top_indices:
        text = chunks[int(idx)]
        if text in seen_text:
            continue
        seen_text.add(text)
        selected.append(text)

    if not selected:
        return "", 0

    logger.info("Top similarity: %.4f (using %d chunks)", float(top_scores[0]), len(selected))
    return "\n\n---\n\n".join(selected), len(selected)
