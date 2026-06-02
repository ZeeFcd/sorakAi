"""Test fixtures shared across the suite.

The previous version exposed ``run_async`` as a module-level helper that other
test files imported with ``from tests.conftest import run_async``. That is a
test-on-test cross-import; Wave 0 of the overhaul plan replaces it with a
proper pytest fixture so test files have a single dependency surface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import numpy as np
import pytest
from fastapi import FastAPI

from sorakai.common.config import get_settings
from sorakai.common.embedding import embed_chunks
from sorakai.common.kb_meta import KBMeta, KBMetaStore
from sorakai.common.store import KnowledgeStore

T = TypeVar("T")


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force ``get_settings()`` to re-read env between tests and pin offline providers.

    - ``Settings`` is ``@lru_cache``d, so without clearing the cache
      ``monkeypatch.setenv`` would silently no-op after the first call.
    - We pin ``LLM_PROVIDER=stub`` and ``EMBEDDING_PROVIDER=char`` so the suite
      runs fully offline regardless of whether Ollama happens to be reachable.
      Tests that exercise a different provider opt in by re-setting these
      vars *before* depending on a settings/store/chat fixture.
    """
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "char")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def run_async() -> Callable[[Awaitable[T]], T]:
    """Run an awaitable to completion from a synchronous test."""

    def _runner(coro: Awaitable[T]) -> T:
        return asyncio.run(coro)

    return _runner


@pytest.fixture
def seed_kb(run_async: Callable[[Awaitable[None]], None]) -> Callable[..., list[np.ndarray]]:
    """Seed a FastAPI app's KB consistently using the configured embedding provider.

    This is the Wave 2 replacement for tests that used to hand-craft
    ``np.array([1.0, 2.0, 3.0])`` and stuff it into the store directly: those
    vectors mismatched the live query embeddings and only worked because of
    the silent ``_pad_to_same_length`` bug we just removed. The fixture
    embeds the chunks with the same provider the query will use, then
    stamps the matching :class:`KBMeta` so the dim guard is happy.

    Returns the list of stored vectors so tests can assert on them.
    """

    def _seeder(
        app: FastAPI,
        chunks: list[str],
        *,
        doc_id: str = "doc-test",
        filename: str = "seed.txt",
    ) -> list[np.ndarray]:
        async def _run() -> list[np.ndarray]:
            settings = get_settings()
            vectors = await embed_chunks(chunks)
            store: KnowledgeStore = app.state.store
            kb_meta: KBMetaStore = app.state.kb_meta
            await store.append_document(doc_id, filename, chunks, vectors)
            await kb_meta.write(
                KBMeta(
                    provider=settings.embedding_provider,
                    model=settings.ollama_embedding_model,
                    dim=vectors[0].size,
                )
            )
            return vectors

        return run_async(_run())

    return _seeder
