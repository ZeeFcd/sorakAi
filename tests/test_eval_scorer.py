"""Wave 9 scorer unit tests.

The scorer is pure-Python and deterministic, so the test set covers
every branch (any/all mode, empty inputs, retrieval opt-out) instead of
relying on golden snapshots.
"""

from __future__ import annotations

from sorakai.eval.dataset import EvalCase
from sorakai.eval.scorer import (
    PASS_RATE_KEY,
    aggregate_scores,
    answer_contains_expected,
    context_precision_at_k,
    maybe_score_with_ragas,
    score_case,
)


def _case(
    *,
    cid: str = "c1",
    needles: tuple[str, ...] = (),
    mode: str = "any",
    docs: tuple[str, ...] = (),
) -> EvalCase:
    return EvalCase(
        id=cid,
        question=f"q-{cid}",
        expected_substrings=needles,
        expected_substrings_mode=mode,
        expected_doc_ids=docs,
    )


def test_answer_contains_expected_any_passes_on_first_hit() -> None:
    case = _case(needles=("Phobos", "Deimos"))
    score, extras = answer_contains_expected(case, "Mars has Phobos.")
    assert score == 1.0
    assert extras["matched"] == ["Phobos"]
    assert extras["missing"] == ["Deimos"]


def test_answer_contains_expected_any_misses_all_returns_zero() -> None:
    case = _case(needles=("Khufu",))
    score, extras = answer_contains_expected(case, "no match here")
    assert score == 0.0
    assert extras["matched"] == []
    assert extras["missing"] == ["Khufu"]


def test_answer_contains_expected_all_requires_every_substring() -> None:
    case = _case(needles=("Phobos", "Deimos"), mode="all")
    assert answer_contains_expected(case, "Mars has Phobos.")[0] == 0.0
    assert answer_contains_expected(case, "Mars has Phobos and Deimos.")[0] == 1.0


def test_answer_contains_expected_is_case_insensitive() -> None:
    case = _case(needles=("PHOBOS",))
    score, extras = answer_contains_expected(case, "phobos is a moon.")
    assert score == 1.0
    assert extras["matched"] == ["PHOBOS"]


def test_answer_contains_expected_empty_needles_is_a_free_pass() -> None:
    case = _case(needles=())
    score, extras = answer_contains_expected(case, "")
    assert score == 1.0
    assert extras["matched"] == []


def test_context_precision_at_k_hits_over_total() -> None:
    case = _case(docs=("mars",))
    score, extras = context_precision_at_k(case, ["mars", "git", "eiffel", "python"])
    assert score == 0.25
    assert extras["hits"] == 1
    assert extras["expected"] == ["mars"]


def test_context_precision_at_k_all_match_returns_one() -> None:
    case = _case(docs=("mars", "python"))
    score, _ = context_precision_at_k(case, ["mars", "python"])
    assert score == 1.0


def test_context_precision_at_k_no_retrieved_returns_zero_when_expected() -> None:
    case = _case(docs=("mars",))
    score, extras = context_precision_at_k(case, [])
    assert score == 0.0
    assert extras["expected"] == ["mars"]


def test_context_precision_at_k_no_expected_is_a_free_pass() -> None:
    case = _case(docs=())
    score, _ = context_precision_at_k(case, [])
    assert score == 1.0


def test_score_case_combines_both_metrics() -> None:
    case = _case(needles=("Phobos",), docs=("mars",))
    score = score_case(case, answer="Phobos", retrieved_doc_ids=["mars", "eiffel"])
    assert score.answer_contains_expected == 1.0
    assert score.context_precision_at_k == 0.5
    assert "answer" in score.extras
    assert "context" in score.extras


def test_aggregate_scores_empty_returns_zeros() -> None:
    metrics = aggregate_scores([])
    assert metrics["cases"] == 0.0
    assert metrics[PASS_RATE_KEY] == 0.0


def test_aggregate_scores_pass_rate_counts_only_perfect_substring_scores() -> None:
    case = _case(needles=("Phobos",), docs=("mars",))
    scores = [
        score_case(case, answer="Phobos", retrieved_doc_ids=["mars"]),
        score_case(case, answer="no hit", retrieved_doc_ids=["git"]),
        score_case(case, answer="Phobos and Deimos", retrieved_doc_ids=["mars", "git"]),
    ]
    metrics = aggregate_scores(scores)
    assert metrics["cases"] == 3.0
    assert metrics[PASS_RATE_KEY] == 2.0 / 3.0
    assert metrics["mean_answer_contains_expected"] == 2.0 / 3.0
    assert metrics["mean_context_precision_at_k"] == (1.0 + 0.0 + 0.5) / 3.0


def test_maybe_score_with_ragas_returns_empty_when_extra_missing() -> None:
    case = _case()
    assert maybe_score_with_ragas(case, answer="x", retrieved_contexts=[]) == {}
