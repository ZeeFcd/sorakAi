"""Chunk embeddings: semantic (OpenAI / Ollama) or legacy char fallback."""

from __future__ import annotations

from typing import List

import httpx
import numpy as np

from sorakai.common.config import get_settings
from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.embedding")


def _embed_char(chunks: List[str]) -> list[np.ndarray]:
    """Deterministic pseudo-embeddings (no semantics) for tests / offline dev."""
    return [np.array([float(ord(c) % 128) for c in chunk[:512]], dtype=float) for chunk in chunks]


async def _embed_openai(chunks: List[str], model: str, api_key: str, base_url: str | None) -> list[np.ndarray]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
    # Single batch when possible (OpenAI supports multiple inputs per request)
    resp = await client.embeddings.create(model=model, input=list(chunks))
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [np.array(item.embedding, dtype=np.float32) for item in ordered]


async def _embed_ollama(
    chunks: List[str],
    base_url: str,
    model: str,
    timeout: float,
) -> list[np.ndarray]:
    base = base_url.rstrip("/")
    out: list[np.ndarray] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, text in enumerate(chunks):
            r = await client.post(
                f"{base}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not emb:
                raise RuntimeError(f"Ollama embeddings: missing 'embedding' in response for chunk {i}")
            out.append(np.array(emb, dtype=np.float32))
    return out


async def embed_chunks(chunks: List[str]) -> list[np.ndarray]:
    """
    Embed text chunks for retrieval.

    Provider from ``EMBEDDING_PROVIDER``:

    - ``char`` — fast local pseudo-vectors (default; good for CI / no network).
    - ``openai`` — ``OPENAI_API_KEY`` + ``OPENAI_EMBEDDING_MODEL`` (default ``text-embedding-3-small``).
      Optional ``OPENAI_EMBEDDINGS_BASE_URL`` for Azure / proxies (not the Ollama chat URL).
    - ``ollama`` — ``OLLAMA_EMBED_BASE_URL`` (e.g. ``http://ollama:11434``) + ``OLLAMA_EMBEDDING_MODEL``
      (e.g. ``nomic-embed-text``); calls ``POST /api/embeddings`` per chunk.
    """
    if not chunks:
        return []

    settings = get_settings()
    provider = (settings.embedding_provider or "char").strip().lower()
    logger.info("Embedding %s chunks via provider=%s", len(chunks), provider)

    if provider == "char":
        return _embed_char(chunks)

    if provider == "openai":
        key = settings.openai_api_key
        if not key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        base = settings.openai_embeddings_base_url
        return await _embed_openai(chunks, settings.openai_embedding_model, key, base)

    if provider == "ollama":
        base = settings.ollama_embed_base_url
        if not base:
            raise RuntimeError("EMBEDDING_PROVIDER=ollama requires OLLAMA_EMBED_BASE_URL (e.g. http://ollama:11434)")
        return await _embed_ollama(
            chunks,
            base,
            settings.ollama_embedding_model,
            settings.request_timeout_seconds * 4,
        )

    logger.warning("Unknown EMBEDDING_PROVIDER=%r; using char fallback", provider)
    return _embed_char(chunks)
