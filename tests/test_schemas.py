"""Tests for :mod:`sorakai.common.schemas` validators (Wave 3 additions)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sorakai.common.schemas import DocumentIngestRequest


def test_document_ingest_request_defaults() -> None:
    req = DocumentIngestRequest(filename="x.txt", content="hello world")
    assert req.chunk_size == 500
    assert req.chunk_overlap == 50
    assert req.mime_type is None
    assert req.replace_kb is False


def test_chunk_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        DocumentIngestRequest(
            filename="x.txt",
            content="hello world",
            chunk_size=100,
            chunk_overlap=100,
        )
    with pytest.raises(ValidationError, match="chunk_overlap"):
        DocumentIngestRequest(
            filename="x.txt",
            content="hello world",
            chunk_size=100,
            chunk_overlap=200,
        )


def test_chunk_overlap_zero_is_allowed() -> None:
    req = DocumentIngestRequest(
        filename="x.txt",
        content="hello world",
        chunk_size=50,
        chunk_overlap=0,
    )
    assert req.chunk_overlap == 0


def test_mime_type_echoed_through() -> None:
    req = DocumentIngestRequest(
        filename="a.md",
        content="# title",
        mime_type="text/markdown",
    )
    assert req.mime_type == "text/markdown"


def test_blank_document_id_becomes_none() -> None:
    req = DocumentIngestRequest(filename="x.txt", content="c", document_id="   ")
    assert req.document_id is None
