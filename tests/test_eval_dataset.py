"""Wave 9 dataset loader tests.

Covers: required fields, mode validation, comment / empty-line handling,
duplicate-id rejection, corpus filtering by extension, and the
``load_default_corpus`` smoke that the in-tree golden + corpus parse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sorakai.core.errors import SorakaiError
from sorakai.eval.dataset import (
    EvalCase,
    EvalDatasetError,
    load_dataset,
    load_default_corpus,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_load_dataset_parses_jsonl_with_comments_and_blanks(tmp_path: Path) -> None:
    body = """
# a comment line
{"id": "a", "question": "Q a", "expected_substrings": ["foo"], "expected_doc_ids": ["d1"]}

{"id": "b", "question": "Q b", "expected_substrings_mode": "all", "expected_substrings": ["x", "y"]}
"""
    path = _write(tmp_path, "g.jsonl", body)
    ds = load_dataset(path)
    assert len(ds) == 2
    assert ds.cases[0].id == "a"
    assert ds.cases[0].expected_substrings == ("foo",)
    assert ds.cases[0].expected_doc_ids == ("d1",)
    assert ds.cases[1].expected_substrings_mode == "all"


def test_load_dataset_missing_question_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.jsonl", '{"id": "a"}\n')
    with pytest.raises(EvalDatasetError, match="missing required keys"):
        load_dataset(path)


def test_load_dataset_unknown_mode_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.jsonl", '{"id": "a", "question": "Q", "expected_substrings_mode": "majority"}\n')
    with pytest.raises(EvalDatasetError, match="unknown mode"):
        load_dataset(path)


def test_load_dataset_duplicate_ids_raises(tmp_path: Path) -> None:
    body = '{"id": "a", "question": "Q1"}\n{"id": "a", "question": "Q2"}\n'
    path = _write(tmp_path, "g.jsonl", body)
    with pytest.raises(EvalDatasetError, match="duplicate"):
        load_dataset(path)


def test_load_dataset_bad_json_reports_line_number(tmp_path: Path) -> None:
    body = '{"id": "a", "question": "Q"}\n{ not json\n'
    path = _write(tmp_path, "g.jsonl", body)
    with pytest.raises(EvalDatasetError, match=":2 invalid JSON"):
        load_dataset(path)


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalDatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_eval_dataset_error_is_sorakai_error() -> None:
    assert issubclass(EvalDatasetError, SorakaiError)


def test_load_dataset_with_corpus_reads_md_files_only(tmp_path: Path) -> None:
    golden = _write(tmp_path, "g.jsonl", '{"id": "a", "question": "Q"}\n')
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("# Alpha", encoding="utf-8")
    (corpus / "beta.txt").write_text("Beta body", encoding="utf-8")
    (corpus / "ignored.png").write_text("nope", encoding="utf-8")
    (corpus / ".hidden").write_text("nope", encoding="utf-8")

    ds = load_dataset(golden, corpus)
    doc_ids = [doc_id for doc_id, _ in ds.corpus]
    assert doc_ids == ["alpha", "beta"]


def test_load_dataset_empty_corpus_dir_raises(tmp_path: Path) -> None:
    golden = _write(tmp_path, "g.jsonl", '{"id": "a", "question": "Q"}\n')
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(EvalDatasetError, match=r"no \.md/\.txt"):
        load_dataset(golden, corpus)


def test_load_dataset_missing_corpus_dir_raises(tmp_path: Path) -> None:
    golden = _write(tmp_path, "g.jsonl", '{"id": "a", "question": "Q"}\n')
    with pytest.raises(EvalDatasetError, match="corpus directory not found"):
        load_dataset(golden, tmp_path / "absent")


def test_load_default_corpus_parses_in_tree_golden() -> None:
    ds = load_default_corpus()
    assert len(ds) >= 12
    assert all(isinstance(c, EvalCase) for c in ds.cases)
    expected_doc_ids = {"pyramids", "eiffel", "mars", "python", "git", "sorakai"}
    actual = {doc_id for doc_id, _ in ds.corpus}
    assert expected_doc_ids.issubset(actual)
