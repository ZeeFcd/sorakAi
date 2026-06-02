"""Tests for the vectorised retrieval module (Wave 2)."""

from __future__ import annotations

import numpy as np
import pytest

from sorakai.common.retrieval import (
    cosine_similarity,
    cosine_top_k,
    retrieve_best_chunk,
    retrieve_top_k_context,
    stack_embeddings,
)


def test_stack_embeddings_builds_matrix() -> None:
    matrix = stack_embeddings([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32


def test_stack_embeddings_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="different embedding model"):
        stack_embeddings([np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0])])


def test_stack_embeddings_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="not 1-D"):
        stack_embeddings([np.array([[1.0, 0.0]])])


def test_stack_embeddings_empty_returns_empty_matrix() -> None:
    matrix = stack_embeddings([])
    assert matrix.shape == (0, 0)


def test_cosine_similarity_basic() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    c = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, c) == pytest.approx(0.0)

    d = np.array([-1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, d) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_yields_zero() -> None:
    z = np.zeros(3, dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(z, v) == 0.0
    assert cosine_similarity(v, z) == 0.0


def test_cosine_similarity_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal dims"):
        cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_cosine_top_k_orders_by_descending_score() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx, scores = cosine_top_k(query, matrix, k=3)
    assert idx.tolist() == [0, 2, 1]
    assert scores[0] > scores[1] > scores[2]


def test_cosine_top_k_query_dim_mismatch_raises() -> None:
    matrix = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="re-ingest"):
        cosine_top_k(np.array([1.0, 0.0]), matrix, k=1)


def test_cosine_top_k_handles_zero_query() -> None:
    matrix = np.eye(3, dtype=np.float32)
    idx, scores = cosine_top_k(np.zeros(3, dtype=np.float32), matrix, k=1)
    assert idx.size == 0
    assert scores.size == 0


def test_cosine_top_k_empty_matrix() -> None:
    idx, scores = cosine_top_k(np.array([1.0, 0.0]), np.empty((0, 0)), k=1)
    assert idx.size == 0
    assert scores.size == 0


def test_retrieve_top_k_context_deduplicates() -> None:
    chunks = ["a", "a", "b"]
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    query = np.array([1.0, 0.0], dtype=np.float32)
    merged, n = retrieve_top_k_context(query, embeddings, chunks, top_k=3)
    assert n == 2
    assert merged.startswith("a")
    assert "b" in merged


def test_retrieve_top_k_context_empty_store() -> None:
    merged, n = retrieve_top_k_context(np.array([1.0]), [], [], top_k=1)
    assert merged == ""
    assert n == 0


def test_retrieve_top_k_context_mismatched_inputs_returns_empty() -> None:
    merged, n = retrieve_top_k_context(
        np.array([1.0]),
        [np.array([1.0]), np.array([0.0])],
        ["only one chunk"],
        top_k=1,
    )
    assert merged == ""
    assert n == 0


def test_retrieve_best_chunk_returns_top_one() -> None:
    chunks = ["far", "close"]
    embeddings = [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert retrieve_best_chunk(query, embeddings, chunks) == "close"


def test_retrieve_top_k_propagates_dim_guard() -> None:
    chunks = ["a"]
    embeddings = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    with pytest.raises(ValueError, match="re-ingest"):
        retrieve_top_k_context(np.array([1.0]), embeddings, chunks, top_k=1)
