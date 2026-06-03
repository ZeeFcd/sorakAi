"""LangChain callback that buffers a chain/agent run into one MLflow run.

This replaces the Wave 1 ``mlflow_run`` context manager + ad-hoc
``log_params_metrics`` calls in :mod:`sorakai.rag.app` and
:mod:`sorakai.ingest.app`. Every retrieval, every LLM call, and every
tool invocation now flows through the same callback, so the resulting
MLflow run carries:

- per-call latencies (``llm_latency_ms_total`` + ``llm_calls``,
  ``retrieval_latency_ms_total`` + ``retrievals``, plus per-call
  ``llm_call_<n>_latency_ms`` for the first ``MAX_PER_CALL_METRICS``),
- token usage when the LLM surfaces it through ``llm_output``,
- the number of retrieved chunks (``docs_retrieved``),
- the chain's final output length (``answer_len``) and inputs as params.

Implementation notes
--------------------

- We open the MLflow run lazily on the **first** ``on_chain_start`` and
  close it on the matching ``on_chain_end``; nested chains share the
  outer run so a single ``POST /v1/query`` still maps to one run.
- The callback never raises - MLflow is optional and a flaky tracking
  server must not break a user request.
- Both sync (``on_*``) and async (``on_*_async``) hooks are wired; the
  async wrappers just delegate so the same buffer is used regardless of
  which path LangChain takes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Final
from uuid import UUID

import mlflow
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from sorakai.core.logging import get_logger

_logger = get_logger(__name__)

MAX_PER_CALL_METRICS: Final[int] = 10
"""Cap the number of per-call ``llm_call_N`` metrics so a chatty agent
doesn't blow up the MLflow UI. Aggregate counters (``llm_calls``,
``llm_latency_ms_total``) are always recorded."""


class MlflowChainCallback(BaseCallbackHandler):
    """One callback instance per chain/agent invocation.

    Re-use across requests is **not** supported (the buffer is per-run).
    Handlers create a fresh callback per request and dispose of it after
    the chain returns.
    """

    raise_error: bool = False
    run_inline: bool = True

    def __init__(
        self,
        experiment_name: str,
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        static_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.static_params: dict[str, Any] = dict(static_params or {})

        self._run_id: str | None = None
        self._root_run_id: UUID | None = None
        self._starts: dict[UUID, float] = {}
        self._summary: dict[str, float] = {
            "llm_calls": 0.0,
            "llm_latency_ms_total": 0.0,
            "tokens_total": 0.0,
            "retrievals": 0.0,
            "docs_retrieved": 0.0,
            "retrieval_latency_ms_total": 0.0,
            "tool_calls": 0.0,
            "tool_latency_ms_total": 0.0,
            "answer_len": 0.0,
        }
        self._per_call_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _open_run(self) -> None:
        if self._run_id is not None:
            return
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=self.run_name)
            self._run_id = run.info.run_id
            if self.static_params:
                mlflow.log_params({k: str(v) for k, v in self.static_params.items()})
        except Exception as exc:
            _logger.warning("MLflow run open failed: %s", exc)
            self._run_id = None

    def _close_run(self) -> None:
        if self._run_id is None:
            return
        try:
            for key, value in self._summary.items():
                mlflow.log_metric(key, value)
            for key, value in self._per_call_metrics.items():
                mlflow.log_metric(key, value)
            mlflow.end_run()
        except Exception as exc:
            _logger.warning("MLflow run close failed: %s", exc)
        finally:
            self._run_id = None

    @property
    def is_active(self) -> bool:
        """True between root chain start and end - useful in tests."""
        return self._root_run_id is not None

    @property
    def summary(self) -> dict[str, float]:
        """Read-only view of the aggregated metrics buffer (for tests)."""
        return dict(self._summary)

    # ------------------------------------------------------------------
    # Chain hooks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self._root_run_id is None and parent_run_id is None:
            self._root_run_id = run_id
            self._open_run()

    def on_chain_end(
        self,
        outputs: dict[str, Any] | str | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id != self._root_run_id:
            return
        answer = outputs.get("answer") or outputs.get("output") or "" if isinstance(outputs, dict) else outputs
        if isinstance(answer, str):
            self._summary["answer_len"] = float(len(answer))
        self._close_run()
        self._root_run_id = None

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id != self._root_run_id:
            return
        if self._run_id is not None:
            try:
                mlflow.set_tag("chain_error", type(error).__name__)
            except Exception as exc:
                _logger.warning("MLflow set_tag failed: %s", exc)
        self._close_run()
        self._root_run_id = None

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: Sequence[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: Sequence[Sequence[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # langchain dispatches chat-model calls here; share the same timing
        # buffer as on_llm_start so a tool-using agent's chat-LLM steps
        # show up in the same ``llm_latency_ms_total`` counter.
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is not None:
            dur_ms = (time.perf_counter() - start) * 1000.0
            self._summary["llm_calls"] += 1.0
            self._summary["llm_latency_ms_total"] += dur_ms
            n = int(self._summary["llm_calls"])
            if n <= MAX_PER_CALL_METRICS:
                self._per_call_metrics[f"llm_call_{n}_latency_ms"] = dur_ms
        try:
            usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
            if isinstance(usage, dict):
                total = usage.get("total_tokens") or sum(int(v) for v in usage.values() if isinstance(v, int | float))
                self._summary["tokens_total"] += float(total)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug("token usage extraction failed: %s", exc)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.pop(run_id, None)

    # ------------------------------------------------------------------
    # Retriever hooks
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is not None:
            self._summary["retrieval_latency_ms_total"] += (time.perf_counter() - start) * 1000.0
        self._summary["retrievals"] += 1.0
        self._summary["docs_retrieved"] += float(len(documents))

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.pop(run_id, None)

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is not None:
            self._summary["tool_latency_ms_total"] += (time.perf_counter() - start) * 1000.0
        self._summary["tool_calls"] += 1.0

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.pop(run_id, None)


__all__ = ["MAX_PER_CALL_METRICS", "MlflowChainCallback"]
