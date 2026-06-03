"""Document chunking - language-aware via ``langchain-text-splitters`` (Wave 3).

The legacy implementation was a 1-line ``[content[i:i+n] for i in ...]``
char-window splitter that happily cut function bodies, list literals, and
markdown lists in half. Wave 3 replaces it with LangChain's
:class:`RecursiveCharacterTextSplitter`, which prefers semantically-meaningful
boundaries (newlines, then sentences, then words, then characters) and accepts
language-specific separator lists for code and prose.

Public API
----------

- :func:`chunk_document` - the single entry point used by the ingest handler.
- :func:`select_splitter` - factory that picks a splitter from
  ``filename`` / ``mime_type`` hints; exposed so tests can pin behaviour.
- :data:`SplitterStrategy` - the small enum returned by
  :func:`detect_strategy` for diagnostics + logging.

Adding a new language is one entry in :data:`_FILENAME_LANGUAGE_MAP` plus one
in :data:`_MIME_LANGUAGE_MAP`. Splitter choice is intentionally **not**
provider-pluggable via env (different inputs want different splitters; that
choice belongs to the caller, not a global setting).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from sorakai.core.logging import get_logger

logger = get_logger(__name__)


class SplitterStrategy(StrEnum):
    """Tag describing which splitter :func:`select_splitter` picked."""

    PYTHON = "python"
    MARKDOWN = "markdown"
    RECURSIVE = "recursive"


# Map filename suffix -> Language. Lowercased; suffix includes the dot.
_FILENAME_LANGUAGE_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
}

# Map MIME type -> Language. Lowercased; trailing parameters (e.g. charset)
# are stripped before lookup.
_MIME_LANGUAGE_MAP: dict[str, Language] = {
    "text/x-python": Language.PYTHON,
    "application/x-python": Language.PYTHON,
    "text/markdown": Language.MARKDOWN,
    "text/x-markdown": Language.MARKDOWN,
}

_LANGUAGE_STRATEGY: dict[Language, SplitterStrategy] = {
    Language.PYTHON: SplitterStrategy.PYTHON,
    Language.MARKDOWN: SplitterStrategy.MARKDOWN,
}


@dataclass(frozen=True, slots=True)
class _SplitterChoice:
    strategy: SplitterStrategy
    splitter: TextSplitter


def _normalise_mime(mime_type: str | None) -> str | None:
    if mime_type is None:
        return None
    return mime_type.split(";", 1)[0].strip().lower() or None


def detect_strategy(filename: str | None, mime_type: str | None) -> SplitterStrategy:
    """Pick a splitter strategy from the available hints.

    ``mime_type`` wins over ``filename`` so explicit caller intent beats
    guessing. Both default to :attr:`SplitterStrategy.RECURSIVE`.
    """
    mime = _normalise_mime(mime_type)
    if mime is not None:
        lang = _MIME_LANGUAGE_MAP.get(mime)
        if lang is not None:
            return _LANGUAGE_STRATEGY[lang]
    if filename:
        suffix = PurePosixPath(filename).suffix.lower()
        lang = _FILENAME_LANGUAGE_MAP.get(suffix)
        if lang is not None:
            return _LANGUAGE_STRATEGY[lang]
    return SplitterStrategy.RECURSIVE


def select_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    filename: str | None = None,
    mime_type: str | None = None,
) -> _SplitterChoice:
    """Build the right splitter for ``filename`` / ``mime_type``.

    Always returns a configured :class:`TextSplitter` plus the strategy tag
    so callers can log / surface which splitter ran.
    """
    strategy = detect_strategy(filename, mime_type)
    if strategy is SplitterStrategy.PYTHON:
        splitter: TextSplitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy is SplitterStrategy.MARKDOWN:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return _SplitterChoice(strategy=strategy, splitter=splitter)


def chunk_document(
    content: str,
    *,
    chunk_size: int,
    chunk_overlap: int = 50,
    filename: str | None = None,
    mime_type: str | None = None,
) -> list[str]:
    """Split ``content`` into chunks using a language-aware splitter.

    Empty / whitespace-only ``content`` yields an empty list. The splitter
    drops any zero-length pieces LangChain may produce at boundary edges.
    """
    if not content or not content.strip():
        return []
    choice = select_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        filename=filename,
        mime_type=mime_type,
    )
    chunks = [c for c in choice.splitter.split_text(content) if c]
    logger.info(
        "Chunked document filename=%s mime=%s strategy=%s chunk_size=%d overlap=%d -> %d chunks",
        filename,
        mime_type,
        choice.strategy.value,
        chunk_size,
        chunk_overlap,
        len(chunks),
    )
    return chunks


def process_file(file_content: str, chunk_size: int = 500) -> list[str]:
    """Back-compat shim around :func:`chunk_document`.

    Kept so anything that still imports ``process_file`` (e.g. notebooks or
    out-of-tree consumers) keeps working without code changes. New callers
    should use :func:`chunk_document` directly so they can opt into overlap
    and language-aware splitting.
    """
    return chunk_document(file_content, chunk_size=chunk_size, chunk_overlap=0)
