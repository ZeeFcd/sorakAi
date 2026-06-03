"""Tests for the Wave 8 MLflow callback.

We don't talk to a real MLflow server; the callback is exercised against
a fake `mlflow` module replaced via ``monkeypatch.setattr`` so we can
assert what would have been logged. The buffered metrics live in
``callback.summary`` so tests don't need to scrape the global state.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from sorakai.common.mlflow_callback import MAX_PER_CALL_METRICS, MlflowChainCallback


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        class _Info:
            pass

        self.info = _Info()
        self.info.run_id = run_id


class _FakeMlflow:
    """Stand-in for ``mlflow`` so the callback never touches the network.

    Recorded calls live on ``recorded_*`` attributes so the method names
    don't shadow the data buckets.
    """

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[str, float]] = []
        self.tags: dict[str, str] = {}
        self.ended = False
        self.recorded_uri: str | None = None
        self.recorded_experiment: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.recorded_uri = uri

    def set_experiment(self, name: str) -> None:
        self.recorded_experiment = name

    def start_run(self, run_name: str | None = None) -> _FakeRun:
        return _FakeRun(run_id=f"run-{run_name}")

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics.append((key, value))

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def end_run(self) -> None:
        self.ended = True


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflow:
    """Replace the ``mlflow`` module inside the callback with a fake."""
    fake = _FakeMlflow()
    monkeypatch.setattr("sorakai.common.mlflow_callback.mlflow", fake)
    return fake


def test_callback_opens_run_on_first_chain_start(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test", run_name="rag")
    cb.on_chain_start({}, {"question": "?"}, run_id=uuid4())
    assert cb.is_active
    assert fake_mlflow.recorded_experiment == "sorakai-test"


def test_callback_closes_run_on_root_chain_end_and_logs_metrics(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test", run_name="rag")
    root_id = uuid4()
    cb.on_chain_start({}, {"question": "?"}, run_id=root_id)
    cb.on_chain_end({"answer": "hello"}, run_id=root_id)

    assert fake_mlflow.ended
    # Always-emitted aggregate metrics.
    keys = [k for k, _ in fake_mlflow.metrics]
    assert "llm_calls" in keys
    assert "retrievals" in keys
    assert "answer_len" in keys
    metric_map = dict(fake_mlflow.metrics)
    assert metric_map["answer_len"] == 5.0


def test_callback_records_static_params(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback(
        "sorakai-test",
        run_name="rag",
        static_params={"service": "rag", "top_k": 5, "session": True},
    )
    cb.on_chain_start({}, {}, run_id=uuid4())
    assert fake_mlflow.params == {"service": "rag", "top_k": "5", "session": "True"}


def test_callback_aggregates_llm_calls_and_latency(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)

    for _ in range(3):
        llm_id = uuid4()
        cb.on_llm_start({}, ["prompt"], run_id=llm_id)
        time.sleep(0.001)
        cb.on_llm_end(
            LLMResult(
                generations=[[Generation(text="x")]],
                llm_output={"token_usage": {"total_tokens": 7}},
            ),
            run_id=llm_id,
        )

    cb.on_chain_end({"answer": "ok"}, run_id=root_id)

    assert cb.summary["llm_calls"] == 3
    assert cb.summary["tokens_total"] == 21
    assert cb.summary["llm_latency_ms_total"] > 0


def test_callback_per_call_metrics_capped(fake_mlflow: _FakeMlflow) -> None:
    """Per-call ``llm_call_N_latency_ms`` metrics are capped so a chatty
    agent doesn't explode the MLflow UI; aggregates remain accurate."""
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)
    for _ in range(MAX_PER_CALL_METRICS + 3):
        llm_id = uuid4()
        cb.on_llm_start({}, ["p"], run_id=llm_id)
        cb.on_llm_end(LLMResult(generations=[[Generation(text="x")]]), run_id=llm_id)
    cb.on_chain_end({"answer": ""}, run_id=root_id)

    per_call_keys = [k for k, _ in fake_mlflow.metrics if k.startswith("llm_call_")]
    assert len(per_call_keys) == MAX_PER_CALL_METRICS
    assert cb.summary["llm_calls"] == MAX_PER_CALL_METRICS + 3


def test_callback_buffers_retrievals(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)
    r1 = uuid4()
    cb.on_retriever_start({}, "query", run_id=r1)
    cb.on_retriever_end([Document(page_content="a"), Document(page_content="b")], run_id=r1)
    r2 = uuid4()
    cb.on_retriever_start({}, "query2", run_id=r2)
    cb.on_retriever_end([Document(page_content="c")], run_id=r2)
    cb.on_chain_end({"answer": ""}, run_id=root_id)

    assert cb.summary["retrievals"] == 2
    assert cb.summary["docs_retrieved"] == 3


def test_callback_records_tool_calls(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)
    t1 = uuid4()
    cb.on_tool_start({}, "kb_search", run_id=t1)
    cb.on_tool_end("ok", run_id=t1)
    cb.on_chain_end({"answer": ""}, run_id=root_id)

    assert cb.summary["tool_calls"] == 1
    assert cb.summary["tool_latency_ms_total"] >= 0.0


def test_callback_chain_error_tags_mlflow_run(fake_mlflow: _FakeMlflow) -> None:
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)
    cb.on_chain_error(RuntimeError("boom"), run_id=root_id)

    assert fake_mlflow.tags.get("chain_error") == "RuntimeError"
    assert fake_mlflow.ended


def test_callback_nested_chains_share_one_mlflow_run(fake_mlflow: _FakeMlflow) -> None:
    """Nested chains (e.g. RunnableWithMessageHistory wrapping the inner
    chain) must NOT open a second MLflow run; the buffer is per-request."""
    cb = MlflowChainCallback("sorakai-test")
    root_id = uuid4()
    child_id = uuid4()
    cb.on_chain_start({}, {}, run_id=root_id)
    cb.on_chain_start({}, {}, run_id=child_id, parent_run_id=root_id)
    assert cb.is_active
    cb.on_chain_end({"answer": ""}, run_id=child_id, parent_run_id=root_id)
    assert cb.is_active  # nested end should NOT close the run
    cb.on_chain_end({"answer": ""}, run_id=root_id)
    assert not cb.is_active
    assert fake_mlflow.ended


def test_callback_swallows_mlflow_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flaky tracking server must NEVER break a user request."""

    class _BrokenMlflow(_FakeMlflow):
        def start_run(self, run_name: str | None = None) -> _FakeRun:
            raise RuntimeError("tracking server down")

    broken = _BrokenMlflow()
    monkeypatch.setattr("sorakai.common.mlflow_callback.mlflow", broken)

    cb = MlflowChainCallback("sorakai-test")
    # Should not raise.
    cb.on_chain_start({}, {}, run_id=uuid4())
    cb.on_chain_end({"answer": "x"}, run_id=cb._root_run_id or uuid4())
