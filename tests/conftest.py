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

import pytest

from sorakai.common.config import get_settings

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
