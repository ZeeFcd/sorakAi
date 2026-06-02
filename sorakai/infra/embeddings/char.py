"""Deterministic pseudo-embeddings adapter (tests / offline dev).

The encoding is ``ord(c) % 128`` per character, truncated to 512 characters.
Vectors are intentionally **variable-length** - matching the legacy MVP
behaviour - which makes them unsuitable for ``Embeddings`` clients that
assume a fixed dim. Production code paths will hit the dimension guard
introduced in Wave 2 and raise instead of silently padding.
"""

from __future__ import annotations

from sorakai.common.config import Settings
from sorakai.infra.embeddings.base import Embeddings


class CharPseudoEmbeddings(Embeddings):
    """Variable-length pseudo-vectors that need no network and no model weights."""

    _MAX_CHARS = 512

    def _encode(self, text: str) -> list[float]:
        return [float(ord(c) % 128) for c in text[: self._MAX_CHARS]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def build_char_embeddings(_settings: Settings) -> Embeddings:
    """Build the pseudo-embeddings adapter. Settings-independent on purpose."""
    return CharPseudoEmbeddings()
