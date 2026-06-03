"""Deterministic pseudo-embeddings adapter (tests / offline dev).

Encodes each character as ``ord(c) % 128`` and produces a **fixed-dimension**
vector by zero-padding (or truncating) to ``CHAR_EMBED_DIM`` floats. The
fixed-dim property is what makes this adapter usable with the Wave 2
dimension guard - real embedding models always emit a constant ``dim`` per
model, and the test stub must do the same to be a faithful stand-in.

These vectors carry no semantic meaning: they are intended only for tests
and offline dev where the code path matters but the retrieval quality does
not.
"""

from __future__ import annotations

from sorakai.common.config import Settings
from sorakai.infra.embeddings.base import Embeddings

CHAR_EMBED_DIM = 256
"""Fixed output dimension. Picked to comfortably fit short inputs while still
being cheap to allocate per call. Bumping this only invalidates KBs that were
ingested with the old value - the Wave 2 dim guard catches that and tells
the operator to re-ingest."""


class CharPseudoEmbeddings(Embeddings):
    """Fixed-dim ``CHAR_EMBED_DIM`` pseudo-vectors with no network / no weights."""

    def _encode(self, text: str) -> list[float]:
        # Pre-allocate so short inputs are zero-padded and long inputs are
        # truncated to the same dim; this is the invariant the dim guard relies on.
        out = [0.0] * CHAR_EMBED_DIM
        for i, c in enumerate(text[:CHAR_EMBED_DIM]):
            out[i] = float(ord(c) % 128)
        return out

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
