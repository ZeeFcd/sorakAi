"""Ollama embeddings adapter (Wave 2).

Implements the full :class:`~langchain_core.embeddings.Embeddings` protocol
on top of ``httpx.AsyncClient`` so we can:

- Batch chunks into ``OLLAMA_EMBED_BATCH``-sized lists and POST them in one
  call to ``/api/embed`` (Ollama's modern batched endpoint).
- Fall back automatically to the legacy per-input ``/api/embeddings`` endpoint
  when ``/api/embed`` returns 404 (older Ollama servers) or when
  ``OLLAMA_EMBED_USE_BATCH_ENDPOINT=false`` forces the legacy path.
- Bound concurrency with ``asyncio.Semaphore(OLLAMA_EMBED_CONCURRENCY)`` so
  we don't open hundreds of sockets against a tiny local server.
- Filter empty / whitespace-only inputs *before* the network call (Ollama
  errors on empty prompts) and reinsert zero-vectors at the original indices
  so the returned list aligns 1:1 with the input list (callers expect that).
- Use a dedicated ``OLLAMA_EMBED_TIMEOUT_SECONDS`` (separate from the gateway
  proxy timeout) since embeddings runs are often much slower than chat.

The shape of this file is the canonical template for future provider
adapters: keep all provider-specific knobs in ``Settings`` and behind one
factory function (``build_ollama_embeddings``) so the rest of the code only
sees :class:`Embeddings`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from sorakai.common.config import Settings
from sorakai.core.errors import EmbeddingError
from sorakai.core.logging import get_logger
from sorakai.infra.embeddings.base import Embeddings

logger = get_logger(__name__)

BATCH_PATH = "/api/embed"
LEGACY_PATH = "/api/embeddings"


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


class OllamaEmbeddingsAdapter(Embeddings):
    """Batched, concurrent, dimension-stable Ollama embeddings client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        batch_size: int,
        concurrency: int,
        timeout_seconds: float,
        use_batch_endpoint: bool,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = httpx.Timeout(timeout_seconds)
        # ``_use_batch_endpoint`` starts as the configured preference and may
        # be flipped to False at runtime the first time ``/api/embed`` 404s
        # (older Ollama). Subsequent calls in this process skip the failed
        # endpoint entirely.
        self._use_batch_endpoint = use_batch_endpoint

    # ---- Embeddings protocol ------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # LangChain's sync API is rarely used in our async stack, but the
        # Embeddings protocol requires it. Delegate to the async path.
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        if _is_blank(text):
            raise EmbeddingError("Cannot embed an empty query string.")
        vectors = await self._embed([text])
        return vectors[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Track which input indices are blank so we can reinsert zero-vectors
        # at the end and keep the output list aligned with ``texts``.
        blank_indices = {i for i, t in enumerate(texts) if _is_blank(t)}
        live_indices = [i for i in range(len(texts)) if i not in blank_indices]
        live_texts = [texts[i] for i in live_indices]

        if not live_texts:
            # No real content; we still need a sensible dim. Round-trip a
            # single space-padded probe so callers downstream can reason
            # about dim without us inventing a constant.
            raise EmbeddingError("All inputs were empty or whitespace; refusing to embed.")

        live_vectors = await self._embed(live_texts)
        if len(live_vectors) != len(live_texts):
            raise EmbeddingError(f"Ollama returned {len(live_vectors)} vectors for {len(live_texts)} inputs.")

        dim = len(live_vectors[0])
        zero = [0.0] * dim
        out: list[list[float]] = []
        live_iter = iter(live_vectors)
        for i in range(len(texts)):
            out.append(zero if i in blank_indices else next(live_iter))
        return out

    # ---- internals ----------------------------------------------------

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Split ``texts`` into batches and dispatch them concurrently."""
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            tasks = [self._embed_batch(client, batch) for batch in batches]
            results = await asyncio.gather(*tasks)
        # Flatten while preserving batch order.
        flat: list[list[float]] = []
        for r in results:
            flat.extend(r)
        return flat

    async def _embed_batch(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        async with self._semaphore:
            if self._use_batch_endpoint:
                vectors = await self._try_batch_endpoint(client, batch)
                if vectors is not None:
                    return vectors
                # Sticky downgrade: stop hitting /api/embed for the rest of
                # this adapter's lifetime once we've seen it 404.
                logger.info("Ollama /api/embed unavailable; falling back to legacy /api/embeddings")
                self._use_batch_endpoint = False
            return await self._call_legacy(client, batch)

    async def _try_batch_endpoint(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]] | None:
        """Return vectors on success, ``None`` on a 404 (signal to fall back)."""
        try:
            response = await client.post(
                BATCH_PATH,
                json={"model": self._model, "input": batch},
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama batched embed request failed: {exc}") from exc

        if response.status_code == 404:
            return None
        if response.is_error:
            raise EmbeddingError(f"Ollama /api/embed returned {response.status_code}: {response.text[:300]}")

        payload: dict[str, Any] = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise EmbeddingError(f"Unexpected /api/embed response shape: {payload!r}")
        return [[float(x) for x in vec] for vec in embeddings]

    async def _call_legacy(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        async def _one(text: str) -> list[float]:
            try:
                response = await client.post(
                    LEGACY_PATH,
                    json={"model": self._model, "prompt": text},
                )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Ollama legacy embed request failed: {exc}") from exc
            if response.is_error:
                raise EmbeddingError(f"Ollama /api/embeddings returned {response.status_code}: {response.text[:300]}")
            payload = response.json()
            vec = payload.get("embedding")
            if not isinstance(vec, list):
                raise EmbeddingError(f"Unexpected /api/embeddings response shape: {payload!r}")
            return [float(x) for x in vec]

        # The semaphore is already held by the caller; legacy fans out
        # sequentially within a batch to respect the same concurrency bound.
        return [await _one(t) for t in batch]


def build_ollama_embeddings(settings: Settings) -> Embeddings:
    """Construct the batched / concurrent Ollama embeddings adapter from settings."""
    return OllamaEmbeddingsAdapter(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        batch_size=settings.ollama_embed_batch,
        concurrency=settings.ollama_embed_concurrency,
        timeout_seconds=settings.ollama_embed_timeout_seconds,
        use_batch_endpoint=settings.ollama_embed_use_batch_endpoint,
    )
