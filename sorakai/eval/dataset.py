"""Loaders for the Wave 9 golden Q/A set + the on-disk corpus.

The dataset shape is JSONL so it stays diff-friendly and easy to grow
case-by-case. Each line decodes to one :class:`EvalCase`; unknown JSON
keys are dropped (forward compatibility - a Wave 10 case may carry
extra fields the runner ignores).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sorakai.core.errors import SorakaiError


class EvalDatasetError(SorakaiError):
    """Raised on a malformed golden file or missing corpus document."""


SubstringMode = Literal["any", "all"]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question / expected-answer row from the golden JSONL.

    Attributes:
        id: Stable identifier; used as the MLflow run name suffix and as
            the dict key in the per-case score map. Must be unique.
        question: The user-facing prompt sent to the chain or agent.
        expected_substrings: Strings the answer must contain. The mode
            controls whether ``any`` of them suffices (default) or
            ``all`` must appear.
        expected_substrings_mode: ``"any"`` (default) or ``"all"``.
        expected_doc_ids: Document IDs (the file stem without the
            ``.md`` suffix) that should appear in the retrieved context;
            used by :func:`context_precision_at_k`.
        tags: Free-form labels for filtering / slicing the result set.
    """

    id: str
    question: str
    expected_substrings: tuple[str, ...] = ()
    expected_substrings_mode: SubstringMode = "any"
    expected_doc_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A bundle of cases plus the corpus the chain/agent should retrieve from."""

    cases: tuple[EvalCase, ...]
    corpus: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Tuple of ``(doc_id, text)`` pairs ingested into the eval KB."""

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)


def load_dataset(
    golden_path: Path,
    corpus_dir: Path | None = None,
) -> EvalDataset:
    """Read ``golden.jsonl`` + (optionally) the matching corpus directory.

    The golden file is parsed line-by-line so a single malformed row only
    fails that row (with the offending line number in the error message),
    keeping the rest of the dataset usable.
    """
    if not golden_path.exists():
        raise EvalDatasetError(f"golden dataset not found: {golden_path}")

    cases: list[EvalCase] = []
    for lineno, raw in enumerate(golden_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"{golden_path}:{lineno} invalid JSON: {exc}") from exc
        cases.append(_case_from_row(row, lineno=lineno, source=golden_path))

    _check_unique_ids(cases, golden_path)

    corpus: tuple[tuple[str, str], ...] = ()
    if corpus_dir is not None:
        corpus = _read_corpus(corpus_dir)

    return EvalDataset(cases=tuple(cases), corpus=corpus)


def load_default_corpus() -> EvalDataset:
    """Load the dataset checked in under ``tests/eval/`` next to the repo root."""
    root = _repo_root()
    return load_dataset(
        golden_path=root / "tests" / "eval" / "golden.jsonl",
        corpus_dir=root / "tests" / "eval" / "corpus",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _case_from_row(row: dict[str, object], *, lineno: int, source: Path) -> EvalCase:
    if not isinstance(row, dict):
        raise EvalDatasetError(f"{source}:{lineno} expected JSON object, got {type(row).__name__}")
    required = ("id", "question")
    missing = [k for k in required if k not in row]
    if missing:
        raise EvalDatasetError(f"{source}:{lineno} missing required keys: {missing}")
    mode_raw = str(row.get("expected_substrings_mode") or "any").lower()
    if mode_raw not in ("any", "all"):
        raise EvalDatasetError(f"{source}:{lineno} unknown mode: {mode_raw!r}")
    return EvalCase(
        id=str(row["id"]),
        question=str(row["question"]),
        expected_substrings=tuple(_string_list(row.get("expected_substrings") or [])),
        expected_substrings_mode=mode_raw,  # type: ignore[arg-type]
        expected_doc_ids=tuple(_string_list(row.get("expected_doc_ids") or [])),
        tags=tuple(_string_list(row.get("tags") or [])),
    )


def _string_list(value: object) -> Sequence[str]:
    if not isinstance(value, list):
        raise EvalDatasetError(f"expected list of strings, got {type(value).__name__}")
    return [str(v) for v in value]


def _check_unique_ids(cases: Sequence[EvalCase], source: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)
    if duplicates:
        raise EvalDatasetError(f"{source} has duplicate case ids: {sorted(set(duplicates))}")


def _read_corpus(corpus_dir: Path) -> tuple[tuple[str, str], ...]:
    if not corpus_dir.is_dir():
        raise EvalDatasetError(f"corpus directory not found: {corpus_dir}")
    docs: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix not in (".md", ".txt"):
            continue
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    if not docs:
        raise EvalDatasetError(f"corpus directory {corpus_dir} contains no .md/.txt files")
    return tuple(docs)


def _repo_root() -> Path:
    """Return the repo root (``tests/eval/...`` lives under it).

    Walks up from this file until we find a directory containing
    ``pyproject.toml`` so the harness works when sorakai is installed
    from a source checkout (the typical eval flow).
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise EvalDatasetError("could not locate repo root (no pyproject.toml above sorakai.eval)")
