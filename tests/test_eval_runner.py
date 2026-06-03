"""Wave 9 eval runner tests.

We script the LLM with :class:`FakeListChatModel` so the harness is
deterministic and the assertions cover both metrics (substring and
context precision) end-to-end.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sorakai.common.config import get_settings
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.eval.dataset import EvalCase, EvalDataset
from sorakai.eval.runner import (
    EvalCaseResult,
    EvalResult,
    _log_eval_to_mlflow,
    _RetrievalCapture,
    run_eval,
    seed_eval_store,
)
from sorakai.eval.scorer import PASS_RATE_KEY
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore

CORPUS = (
    (
        "mars",
        "Mars has two natural moons, Phobos and Deimos. A day on Mars is about 24 hours and 37 minutes.",
    ),
    (
        "python",
        "Python was created by Guido van Rossum and first released in 1991. CPython is the reference implementation.",
    ),
    (
        "eiffel",
        "The Eiffel Tower in Paris stands 330 metres tall and was built between 1887 and 1889.",
    ),
)


def _dataset() -> EvalDataset:
    return EvalDataset(
        cases=(
            EvalCase(
                id="mars-moons",
                question="What are the moons of Mars?",
                expected_substrings=("Phobos", "Deimos"),
                expected_substrings_mode="all",
                expected_doc_ids=("mars",),
            ),
            EvalCase(
                id="py-creator",
                question="Who created Python?",
                expected_substrings=("Guido",),
                expected_doc_ids=("python",),
            ),
        ),
        corpus=CORPUS,
    )


def _fresh_store() -> KnowledgeStoreVectorStore:
    return KnowledgeStoreVectorStore(InMemoryKnowledgeStore())


@pytest.mark.asyncio
async def test_seed_eval_store_writes_chunks_for_every_doc() -> None:
    store = _fresh_store()
    total = await seed_eval_store(store, _dataset())
    assert total >= len(CORPUS)
    summaries = await store.list_docs()
    seen = {s.doc_id for s in summaries}
    assert seen == {"mars", "python", "eiffel"}


@pytest.mark.asyncio
async def test_seed_eval_store_no_corpus_returns_zero() -> None:
    store = _fresh_store()
    empty_ds = EvalDataset(cases=(), corpus=())
    assert await seed_eval_store(store, empty_ds) == 0


@pytest.mark.asyncio
async def test_run_eval_chain_passes_when_llm_answers_correctly() -> None:
    settings = get_settings()
    llm = FakeListChatModel(
        responses=[
            "Mars has Phobos and Deimos as moons.",
            "Python was created by Guido van Rossum.",
        ]
    )
    result = await run_eval(
        target="chain",
        settings=settings,
        dataset=_dataset(),
        llm=llm,
        vector_store=_fresh_store(),
    )

    assert isinstance(result, EvalResult)
    assert result.target == "chain"
    assert len(result.cases) == 2
    assert all(isinstance(c, EvalCaseResult) for c in result.cases)
    assert result.metrics[PASS_RATE_KEY] == 1.0
    assert result.metrics["mean_answer_contains_expected"] == 1.0


@pytest.mark.asyncio
async def test_run_eval_chain_captures_retrieved_doc_ids() -> None:
    settings = get_settings()
    llm = FakeListChatModel(
        responses=[
            "Mars has Phobos and Deimos.",
            "Python by Guido.",
        ]
    )
    result = await run_eval(
        target="chain",
        settings=settings,
        dataset=_dataset(),
        llm=llm,
        vector_store=_fresh_store(),
    )

    for case in result.cases:
        assert case.retrieved_doc_ids, f"no doc ids captured for {case.id}"
        assert all(d in {"mars", "python", "eiffel"} for d in case.retrieved_doc_ids)


@pytest.mark.asyncio
async def test_run_eval_chain_fails_when_answer_misses_substrings() -> None:
    settings = get_settings()
    llm = FakeListChatModel(responses=["irrelevant", "irrelevant"])
    result = await run_eval(
        target="chain",
        settings=settings,
        dataset=_dataset(),
        llm=llm,
        vector_store=_fresh_store(),
    )
    assert result.metrics[PASS_RATE_KEY] == 0.0


@pytest.mark.asyncio
async def test_run_eval_agent_executes_full_graph() -> None:
    settings = get_settings()
    # 4 responses per case: route, grade, generate, critique.
    llm = FakeListChatModel(
        responses=[
            "kb",
            "good",
            "Mars has Phobos and Deimos.",
            "ok",
            "kb",
            "good",
            "Python was created by Guido van Rossum.",
            "ok",
        ]
    )
    result = await run_eval(
        target="agent",
        settings=settings,
        dataset=_dataset(),
        llm=llm,
        vector_store=_fresh_store(),
    )
    assert result.target == "agent"
    assert result.metrics[PASS_RATE_KEY] == 1.0
    assert all(c.retrieved_doc_ids for c in result.cases)


@pytest.mark.asyncio
async def test_run_eval_records_per_case_latency() -> None:
    settings = get_settings()
    llm = FakeListChatModel(responses=["x", "y"])
    result = await run_eval(
        target="chain",
        settings=settings,
        dataset=_dataset(),
        llm=llm,
        vector_store=_fresh_store(),
    )
    assert all(c.latency_ms >= 0.0 for c in result.cases)
    assert result.metrics["mean_latency_ms"] >= 0.0


def test_retrieval_capture_deduplicates_and_skips_unknown_ids() -> None:
    capture = _RetrievalCapture()
    capture.on_retriever_end(
        [
            Document(page_content="x", metadata={"doc_id": "mars"}),
            Document(page_content="y", metadata={"doc_id": "mars"}),
            Document(page_content="z", metadata={"doc_id": "python"}),
            Document(page_content="w", metadata={}),  # no doc_id -> skipped
        ],
        run_id=uuid4(),
    )
    assert capture.doc_ids == ("mars", "python")
    capture.reset()
    assert len(capture.doc_ids) == 0


def test_log_eval_to_mlflow_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval should never break because the MLflow server is flaky."""

    class _BoomCallback:
        experiment_name = "x"
        run_name = "y"
        is_active = False

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tracking server down")

    monkeypatch.setattr("sorakai.eval.runner.mlflow.set_experiment", _boom)

    _log_eval_to_mlflow(_BoomCallback(), target="chain", metrics={"pass_rate": 1.0}, n_cases=1)


def test_log_eval_to_mlflow_records_metrics_when_callback_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {"metrics": {}, "params": {}, "experiment": None, "run_name": None}

    class _Cb:
        experiment_name = "exp-x"
        run_name = "run-y"
        is_active = False

    def _set_experiment(name: str) -> None:
        recorded["experiment"] = name

    def _start_run(*, run_name: str | None = None) -> None:
        recorded["run_name"] = run_name

    def _log_metric(key: str, value: float) -> None:
        recorded["metrics"][key] = value

    def _log_param(key: str, value: Any) -> None:
        recorded["params"][key] = value

    monkeypatch.setattr("sorakai.eval.runner.mlflow.set_experiment", _set_experiment)
    monkeypatch.setattr("sorakai.eval.runner.mlflow.start_run", _start_run)
    monkeypatch.setattr("sorakai.eval.runner.mlflow.log_metric", _log_metric)
    monkeypatch.setattr("sorakai.eval.runner.mlflow.log_param", _log_param)

    _log_eval_to_mlflow(_Cb(), target="chain", metrics={"pass_rate": 0.9}, n_cases=3)

    assert recorded["experiment"] == "exp-x"
    assert recorded["run_name"] == "run-y"
    assert recorded["metrics"] == {"eval_pass_rate": 0.9}
    assert recorded["params"] == {"eval_target": "chain", "eval_cases": 3}


@pytest.mark.asyncio
async def test_run_eval_unknown_target_rejected() -> None:
    settings = get_settings()
    with pytest.raises(ValueError, match="unknown eval target"):
        await run_eval(
            target="bogus",
            settings=settings,
            dataset=_dataset(),
            llm=FakeListChatModel(responses=[]),
            vector_store=_fresh_store(),
        )
