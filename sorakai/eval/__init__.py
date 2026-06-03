"""Wave 9 evaluation harness for the LCEL chain and the LangGraph agent.

The harness is intentionally tiny:

- :mod:`sorakai.eval.dataset` loads the golden JSONL Q/A set + the on-disk
  corpus checked into ``tests/eval/corpus/`` and seeds an in-memory
  knowledge store from it.
- :mod:`sorakai.eval.scorer` implements two in-tree metrics
  (``answer_contains_expected`` and ``context_precision_at_k``) plus the
  aggregation helper :func:`aggregate_scores`.
- :mod:`sorakai.eval.runner` wires the seed store + the chain / agent and
  iterates the dataset, returning an :class:`EvalResult` the CLI (or a
  test) can assert against.

Cloud-only providers stay disabled by default; the harness picks providers
from settings exactly the same way the production handlers do, so the
exact same chain is being evaluated.
"""

from sorakai.eval.dataset import EvalCase, EvalDataset, load_dataset, load_default_corpus
from sorakai.eval.runner import EvalCaseResult, EvalResult, EvalTarget, run_eval
from sorakai.eval.scorer import (
    PASS_RATE_KEY,
    aggregate_scores,
    answer_contains_expected,
    context_precision_at_k,
    score_case,
)

__all__ = [
    "PASS_RATE_KEY",
    "EvalCase",
    "EvalCaseResult",
    "EvalDataset",
    "EvalResult",
    "EvalTarget",
    "aggregate_scores",
    "answer_contains_expected",
    "context_precision_at_k",
    "load_dataset",
    "load_default_corpus",
    "run_eval",
    "score_case",
]
