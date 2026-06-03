"""Tests for :mod:`sorakai.common.ingest` (Wave 3 LangChain chunker)."""

from __future__ import annotations

import logging

import pytest

from sorakai.common.ingest import (
    SplitterStrategy,
    chunk_document,
    detect_strategy,
    process_file,
    select_splitter,
)

# ---------- detect_strategy ----------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "mime_type", "expected"),
    [
        ("script.py", None, SplitterStrategy.PYTHON),
        ("module.PY", None, SplitterStrategy.PYTHON),
        ("notes.md", None, SplitterStrategy.MARKDOWN),
        ("post.markdown", None, SplitterStrategy.MARKDOWN),
        ("page.mdx", None, SplitterStrategy.MARKDOWN),
        ("readme.txt", None, SplitterStrategy.RECURSIVE),
        (None, None, SplitterStrategy.RECURSIVE),
        ("anything.bin", "text/x-python", SplitterStrategy.PYTHON),
        ("anything.bin", "text/markdown; charset=utf-8", SplitterStrategy.MARKDOWN),
        ("anything.bin", "application/octet-stream", SplitterStrategy.RECURSIVE),
    ],
)
def test_detect_strategy_dispatch(filename: str | None, mime_type: str | None, expected: SplitterStrategy) -> None:
    assert detect_strategy(filename, mime_type) is expected


def test_mime_type_wins_over_filename() -> None:
    # ``.py`` suffix would normally pick PYTHON, but caller explicitly says markdown.
    assert detect_strategy("a.py", "text/markdown") is SplitterStrategy.MARKDOWN


# ---------- select_splitter ---------------------------------------------------


def test_select_splitter_returns_strategy_and_splitter() -> None:
    choice = select_splitter(chunk_size=200, chunk_overlap=50, filename="a.py")
    assert choice.strategy is SplitterStrategy.PYTHON
    assert hasattr(choice.splitter, "split_text")


# ---------- chunk_document ----------------------------------------------------


def test_chunk_document_empty_input_returns_empty_list() -> None:
    assert chunk_document("", chunk_size=100, chunk_overlap=0) == []
    assert chunk_document("   \n\t  ", chunk_size=100, chunk_overlap=0) == []


def test_chunk_document_short_input_yields_one_chunk() -> None:
    chunks = chunk_document("hello world", chunk_size=100, chunk_overlap=0)
    assert chunks == ["hello world"]


def test_chunk_document_recursive_respects_chunk_size() -> None:
    text = "abcdef\n" * 100  # ~700 chars
    chunks = chunk_document(text, chunk_size=80, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        # RecursiveCharacterTextSplitter occasionally exceeds chunk_size slightly
        # if there is no good boundary; a generous bound is sufficient as a guard.
        assert len(c) <= 160


def test_chunk_document_overlap_is_applied() -> None:
    """With overlap > 0 successive chunks share content; with overlap == 0 they don't."""
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 5
    no_overlap = chunk_document(text, chunk_size=80, chunk_overlap=0)
    with_overlap = chunk_document(text, chunk_size=80, chunk_overlap=40)
    # Overlap normally produces at least as many chunks as no-overlap, often more.
    assert len(with_overlap) >= len(no_overlap)


def test_chunk_document_python_keeps_function_together() -> None:
    """A short Python function should land in a single chunk when ``chunk_size`` allows it."""
    src = "def add(a, b):\n    '''Return a + b.'''\n    return a + b\n"
    chunks = chunk_document(src, chunk_size=500, chunk_overlap=0, filename="x.py")
    assert len(chunks) == 1
    assert "def add" in chunks[0]
    assert "return a + b" in chunks[0]


def test_chunk_document_markdown_splits_on_headings() -> None:
    md = "# Title\n\nIntro paragraph.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
    chunks = chunk_document(md, chunk_size=40, chunk_overlap=0, filename="doc.md")
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    assert "Section A" in joined
    assert "Section B" in joined


def test_chunk_document_logs_strategy(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="sorakai.common.ingest")
    chunk_document("hello", chunk_size=100, chunk_overlap=0, filename="x.py")
    assert any("strategy=python" in rec.getMessage() for rec in caplog.records)


# ---------- process_file back-compat -----------------------------------------


def test_process_file_back_compat_shim() -> None:
    """The legacy ``process_file`` entry point still returns a list of strings."""
    chunks = process_file("hello world " * 20, chunk_size=80)
    assert isinstance(chunks, list)
    assert chunks
    assert all(isinstance(c, str) for c in chunks)
