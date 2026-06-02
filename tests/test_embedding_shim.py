"""Tests for the ``sorakai.common.embedding`` shim."""

from __future__ import annotations

import numpy as np

from sorakai.common.embedding import embed_chunks, embed_query


def test_embed_chunks_returns_float32_ndarrays(run_async) -> None:
    vectors = run_async(embed_chunks(["hello", "world"]))
    assert len(vectors) == 2
    for v in vectors:
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float32


def test_embed_chunks_handles_empty_input(run_async) -> None:
    assert run_async(embed_chunks([])) == []


def test_embed_query_returns_float32_ndarray(run_async) -> None:
    v = run_async(embed_query("hello"))
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    assert v.size > 0


def test_char_embeddings_are_deterministic(run_async) -> None:
    a = run_async(embed_chunks(["abc"]))[0]
    b = run_async(embed_chunks(["abc"]))[0]
    np.testing.assert_array_equal(a, b)
