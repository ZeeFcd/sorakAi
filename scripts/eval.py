#!/usr/bin/env python3
"""Wave 9 eval harness CLI.

Examples
--------

    # Run the LCEL chain end-to-end with the configured provider
    # (defaults to Ollama from .env). Exits non-zero if the pass rate
    # drops below ``--min-pass-rate`` (the regression gate CI uses).
    python scripts/eval.py --target chain

    # Run the agent, log a single MLflow run per case + an aggregate
    # row, and dump the per-case results as JSON for archival.
    python scripts/eval.py --target agent --mlflow --json out/eval.json

    # Smoke run a subset of the golden set (handy while editing prompts).
    python scripts/eval.py --target chain --limit 3 --min-pass-rate 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from sorakai.common.config import get_settings  # noqa: E402
from sorakai.common.mlflow_callback import MlflowChainCallback  # noqa: E402
from sorakai.core.logging import configure_logging, get_logger  # noqa: E402
from sorakai.eval.dataset import EvalDataset, load_default_corpus  # noqa: E402
from sorakai.eval.runner import EvalResult, EvalTarget, run_eval  # noqa: E402
from sorakai.eval.scorer import PASS_RATE_KEY  # noqa: E402

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2

DEFAULT_EXPERIMENT = "sorakai-eval"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/eval.py",
        description="Run the sorakAi golden Q/A set through the chain or agent.",
    )
    parser.add_argument(
        "--target",
        choices=("chain", "agent"),
        default="chain",
        help="Which entry point to evaluate (default: chain).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases (handy for fast iteration on prompts).",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.7,
        help=(
            "Regression gate: exit non-zero when the pass_rate metric drops below "
            "this value. Set to 0.0 to disable the gate."
        ),
    )
    parser.add_argument(
        "--min-context-precision",
        type=float,
        default=0.0,
        help="Optional gate on mean_context_precision_at_k.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Wire the MlflowChainCallback so each case is logged as an MLflow run.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name when --mlflow is set (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to dump the full result (cases + metrics) as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-case console table; only print the aggregate summary.",
    )
    return parser


def _slice(dataset: EvalDataset, limit: int | None) -> EvalDataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return EvalDataset(cases=dataset.cases[:limit], corpus=dataset.corpus)


def _print_case_table(result: EvalResult) -> None:
    width_id = max((len(r.case_id) for r in result.cases), default=4)
    header = f"{'case_id':<{width_id}} {'ans':>4} {'ctx':>5} {'lat_ms':>8}  question"
    print(header)
    print("-" * len(header))
    for r in result.cases:
        ctx_display = f"{r.score.context_precision_at_k:.2f}"
        print(
            f"{r.case_id:<{width_id}} "
            f"{int(r.score.answer_contains_expected):>4} "
            f"{ctx_display:>5} "
            f"{r.latency_ms:>8.1f}  "
            f"{r.question}"
        )


def _print_summary(result: EvalResult) -> None:
    metrics = result.metrics
    print()
    print(f"target              : {result.target}")
    print(f"cases               : {int(metrics.get('cases', 0))}")
    print(f"pass_rate           : {metrics.get(PASS_RATE_KEY, 0.0):.3f}")
    print(f"answer_contains_avg : {metrics.get('mean_answer_contains_expected', 0.0):.3f}")
    print(f"context_precision   : {metrics.get('mean_context_precision_at_k', 0.0):.3f}")
    print(f"mean_latency_ms     : {metrics.get('mean_latency_ms', 0.0):.1f}")


def _result_to_json(result: EvalResult) -> dict[str, Any]:
    return {
        "target": result.target,
        "metrics": dict(result.metrics),
        "cases": [_case_to_json(c) for c in result.cases],
    }


def _case_to_json(case: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": case.case_id,
        "question": case.question,
        "answer": case.answer,
        "retrieved_doc_ids": list(case.retrieved_doc_ids),
        "latency_ms": case.latency_ms,
        "ragas": dict(case.ragas),
    }
    score = case.score
    payload["score"] = {
        "answer_contains_expected": score.answer_contains_expected,
        "context_precision_at_k": score.context_precision_at_k,
        "extras": _to_jsonable(score.extras),
    }
    return payload


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _build_mlflow_callback(args: argparse.Namespace) -> MlflowChainCallback | None:
    if not args.mlflow:
        return None
    settings = get_settings()
    return MlflowChainCallback(
        experiment_name=args.experiment,
        run_name=f"eval-{args.target}",
        tracking_uri=settings.mlflow_tracking_uri,
        static_params={"eval_target": args.target},
    )


async def _amain(args: argparse.Namespace) -> int:
    configure_logging()
    logger = get_logger("sorakai.eval.cli")

    settings = get_settings()
    dataset = _slice(load_default_corpus(), args.limit)
    if not dataset.cases:
        logger.error("dataset has zero cases after --limit slicing")
        return EXIT_USAGE

    callback = _build_mlflow_callback(args)
    result = await run_eval(
        target=cast(EvalTarget, args.target),
        settings=settings,
        dataset=dataset,
        mlflow_callback=callback,
    )

    if not args.quiet:
        _print_case_table(result)
    _print_summary(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_result_to_json(result), indent=2) + "\n")
        logger.info("wrote eval results to %s", args.json)

    pass_rate = result.metrics.get(PASS_RATE_KEY, 0.0)
    context_precision = result.metrics.get("mean_context_precision_at_k", 0.0)
    failures: list[str] = []
    if pass_rate < args.min_pass_rate:
        failures.append(f"pass_rate {pass_rate:.3f} < min {args.min_pass_rate:.3f}")
    if context_precision < args.min_context_precision:
        failures.append(f"mean_context_precision_at_k {context_precision:.3f} < min {args.min_context_precision:.3f}")
    if failures:
        for msg in failures:
            print(f"REGRESSION: {msg}", file=sys.stderr)
        return EXIT_REGRESSION
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
