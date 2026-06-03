"""In-tree scoring metrics for the Wave 9 eval harness.

These are deliberately lightweight (no LLM-as-judge, no ragas required)
so the harness runs in a few seconds on the stub LLM and in CI:

- :func:`answer_contains_expected` — 1.0 if the answer covers the
  expected substrings (per the case's ``any`` / ``all`` mode), else 0.0.
- :func:`context_precision_at_k` — fraction of retrieved documents
  that belong to the expected set (precision@k flavour, where ``k`` is
  whatever the retriever actually returned).

If ``ragas`` is installed (optional extra) we surface its metric names
in :func:`maybe_score_with_ragas`; otherwise that helper returns an
empty dict so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sorakai.eval.dataset import EvalCase

try:
    import ragas as _ragas
except ImportError:  # pragma: no cover - exercised on hosts without the extra
    _ragas = None

PASS_RATE_KEY = "pass_rate"
"""Aggregate key that flips to 1.0 when ``answer_contains_expected`` is 1.0
for every case in the dataset; the eval CLI uses this as the regression
gate by default."""


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Per-case score bundle. ``extras`` carries metric-specific data
    (e.g. matched substrings) for human-readable reports."""

    answer_contains_expected: float
    context_precision_at_k: float
    extras: Mapping[str, Any]


def answer_contains_expected(case: EvalCase, answer: str) -> tuple[float, Mapping[str, Any]]:
    """Return ``(score, extras)`` for the substring-containment metric.

    Empty ``expected_substrings`` is treated as a free pass (1.0) so a
    case author can opt out of this metric by leaving the list empty.
    """
    needles = case.expected_substrings
    if not needles:
        return 1.0, {"matched": [], "missing": [], "needles": []}
    answer_lc = answer.lower()
    matched = [n for n in needles if n.lower() in answer_lc]
    missing = [n for n in needles if n not in matched]
    score = (1.0 if not missing else 0.0) if case.expected_substrings_mode == "all" else (1.0 if matched else 0.0)
    return score, {"matched": matched, "missing": missing, "needles": list(needles)}


def context_precision_at_k(case: EvalCase, retrieved_doc_ids: Sequence[str]) -> tuple[float, Mapping[str, Any]]:
    """Precision-at-k flavour: ``|retrieved ∩ expected| / |retrieved|``.

    Returns 0.0 when nothing was retrieved AND we expected something
    (so the metric punishes silent retrieval failures); 1.0 when the
    case did not declare any expected docs (opt-out, same semantics as
    :func:`answer_contains_expected`).
    """
    expected = set(case.expected_doc_ids)
    if not expected:
        return 1.0, {"retrieved": list(retrieved_doc_ids), "expected": []}
    if not retrieved_doc_ids:
        return 0.0, {"retrieved": [], "expected": sorted(expected)}
    hits = sum(1 for d in retrieved_doc_ids if d in expected)
    score = hits / float(len(retrieved_doc_ids))
    return score, {
        "retrieved": list(retrieved_doc_ids),
        "expected": sorted(expected),
        "hits": hits,
    }


def score_case(
    case: EvalCase,
    *,
    answer: str,
    retrieved_doc_ids: Sequence[str],
) -> CaseScore:
    """Score one case against the chain/agent output."""
    a, a_extra = answer_contains_expected(case, answer)
    p, p_extra = context_precision_at_k(case, retrieved_doc_ids)
    return CaseScore(
        answer_contains_expected=a,
        context_precision_at_k=p,
        extras={"answer": a_extra, "context": p_extra},
    )


def aggregate_scores(scores: Sequence[CaseScore]) -> dict[str, float]:
    """Reduce per-case scores into a flat metric dict.

    Three aggregates are exposed:

    - ``mean_answer_contains_expected`` — average of the substring metric.
    - ``mean_context_precision_at_k`` — average of the retrieval metric.
    - ``pass_rate`` — fraction of cases that scored 1.0 on the substring
      metric (the canonical CI regression gate).
    """
    if not scores:
        return {
            "cases": 0.0,
            "mean_answer_contains_expected": 0.0,
            "mean_context_precision_at_k": 0.0,
            PASS_RATE_KEY: 0.0,
        }
    answer_vals = [s.answer_contains_expected for s in scores]
    context_vals = [s.context_precision_at_k for s in scores]
    return {
        "cases": float(len(scores)),
        "mean_answer_contains_expected": statistics.mean(answer_vals),
        "mean_context_precision_at_k": statistics.mean(context_vals),
        PASS_RATE_KEY: float(sum(1 for v in answer_vals if v >= 1.0)) / float(len(scores)),
    }


def maybe_score_with_ragas(
    case: EvalCase,
    *,
    answer: str,
    retrieved_contexts: Sequence[str],
) -> Mapping[str, float]:
    """Hook for the optional ``ragas`` extra (Wave 9 stays no-op unless
    the user installs it). When present, we return ``ragas`` metric
    scores; when absent, an empty dict, so the runner can merge the
    result unconditionally.

    The Wave 9 implementation intentionally stops at "is the dependency
    available?" - integrating ragas's evaluator with a local LLM is
    larger than this wave needs. Keeping the public signature stable
    means Wave 10+ can fill the body without breaking callers.
    """
    del case, answer, retrieved_contexts
    if _ragas is None:
        return {}
    return {}


__all__ = [
    "PASS_RATE_KEY",
    "CaseScore",
    "aggregate_scores",
    "answer_contains_expected",
    "context_precision_at_k",
    "maybe_score_with_ragas",
    "score_case",
]
