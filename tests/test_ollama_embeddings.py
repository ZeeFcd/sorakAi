"""Tests for :mod:`sorakai.infra.embeddings.ollama` (Wave 2).

Uses ``respx`` to intercept the real ``httpx.AsyncClient`` traffic so we can
assert on requests/responses without spinning up an Ollama server.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from sorakai.core.errors import EmbeddingError
from sorakai.infra.embeddings.ollama import (
    BATCH_PATH,
    LEGACY_PATH,
    OllamaEmbeddingsAdapter,
)

BASE = "http://ollama.test"


def _adapter(
    *,
    batch: int = 64,
    concurrency: int = 4,
    use_batch_endpoint: bool = True,
) -> OllamaEmbeddingsAdapter:
    return OllamaEmbeddingsAdapter(
        base_url=BASE,
        model="nomic-embed-text",
        batch_size=batch,
        concurrency=concurrency,
        timeout_seconds=5.0,
        use_batch_endpoint=use_batch_endpoint,
    )


@respx.mock
def test_batched_endpoint_returns_vectors(run_async) -> None:
    route = respx.post(f"{BASE}{BATCH_PATH}").mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
        )
    )
    adapter = _adapter()
    vectors = run_async(adapter.aembed_documents(["hello", "world"]))
    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert route.called
    sent = route.calls[0].request.read()
    assert b'"input":["hello","world"]' in sent
    assert b'"model":"nomic-embed-text"' in sent


@respx.mock
def test_multi_batch_splits_requests(run_async) -> None:
    # batch=2 forces 3 inputs to be split into two requests of size 2 + 1.
    def _responder(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        n = len(payload["input"])
        return httpx.Response(200, json={"embeddings": [[float(i)] for i in range(n)]})

    route = respx.post(f"{BASE}{BATCH_PATH}").mock(side_effect=_responder)
    adapter = _adapter(batch=2)
    vectors = run_async(adapter.aembed_documents(["a", "b", "c"]))
    assert len(vectors) == 3
    assert route.call_count == 2


@respx.mock
def test_batched_endpoint_404_falls_back_to_legacy(run_async) -> None:
    respx.post(f"{BASE}{BATCH_PATH}").mock(return_value=httpx.Response(404, text="not found"))
    legacy = respx.post(f"{BASE}{LEGACY_PATH}").mock(
        side_effect=[
            httpx.Response(200, json={"embedding": [1.0, 0.0]}),
            httpx.Response(200, json={"embedding": [0.0, 1.0]}),
        ]
    )
    adapter = _adapter()
    vectors = run_async(adapter.aembed_documents(["a", "b"]))
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert legacy.call_count == 2


@respx.mock
def test_force_legacy_via_settings(run_async) -> None:
    batch_route = respx.post(f"{BASE}{BATCH_PATH}")
    legacy = respx.post(f"{BASE}{LEGACY_PATH}").mock(return_value=httpx.Response(200, json={"embedding": [0.5, 0.5]}))
    adapter = _adapter(use_batch_endpoint=False)
    vectors = run_async(adapter.aembed_documents(["x"]))
    assert vectors == [[0.5, 0.5]]
    assert legacy.called
    assert not batch_route.called


@respx.mock
def test_empty_inputs_skipped_and_zero_padded(run_async) -> None:
    route = respx.post(f"{BASE}{BATCH_PATH}").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0, 2.0, 3.0]]})
    )
    adapter = _adapter()
    vectors = run_async(adapter.aembed_documents(["", "hello", "   "]))
    assert vectors[0] == [0.0, 0.0, 0.0]
    assert vectors[1] == [1.0, 2.0, 3.0]
    assert vectors[2] == [0.0, 0.0, 0.0]
    # Only the non-blank input should have been sent.
    sent = route.calls[0].request.read()
    assert b'"input":["hello"]' in sent


def test_all_empty_inputs_raises(run_async) -> None:
    adapter = _adapter()
    with pytest.raises(EmbeddingError, match="All inputs were empty"):
        run_async(adapter.aembed_documents(["", "  ", "\n"]))


def test_query_empty_raises(run_async) -> None:
    adapter = _adapter()
    with pytest.raises(EmbeddingError, match="empty query"):
        run_async(adapter.aembed_query("   "))


@respx.mock
def test_concurrency_bounded_by_semaphore(run_async) -> None:
    """With concurrency=2 and batch=1, three inputs should peak at 2 in-flight."""
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _responder(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    respx.post(f"{BASE}{BATCH_PATH}").mock(side_effect=_responder)
    adapter = _adapter(batch=1, concurrency=2)
    run_async(adapter.aembed_documents(["a", "b", "c", "d"]))
    assert max_in_flight <= 2


@respx.mock
def test_batched_endpoint_error_status_raises(run_async) -> None:
    respx.post(f"{BASE}{BATCH_PATH}").mock(return_value=httpx.Response(500, text="boom"))
    adapter = _adapter()
    with pytest.raises(EmbeddingError, match="500"):
        run_async(adapter.aembed_documents(["a"]))


@respx.mock
def test_batched_endpoint_bad_shape_raises(run_async) -> None:
    respx.post(f"{BASE}{BATCH_PATH}").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})  # 1 vec for 2 inputs
    )
    adapter = _adapter()
    with pytest.raises(EmbeddingError, match="Unexpected"):
        run_async(adapter.aembed_documents(["a", "b"]))


@respx.mock
def test_aembed_query_uses_batched_endpoint(run_async) -> None:
    route = respx.post(f"{BASE}{BATCH_PATH}").mock(return_value=httpx.Response(200, json={"embeddings": [[7.0]]}))
    adapter = _adapter()
    vec = run_async(adapter.aembed_query("hello"))
    assert vec == [7.0]
    assert route.called
