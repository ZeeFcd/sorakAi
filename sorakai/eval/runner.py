"""Glue between the eval dataset, the chain/agent under test, and the scorer.

The runner is intentionally provider-agnostic: it asks
:func:`sorakai.infra.vector_store.factory.get_vector_store` for an empty
backend, ingests the dataset's corpus through the same
:func:`sorakai.common.ingest.chunk_document` +
:func:`sorakai.common.embedding.embed_chunks` pipeline production uses,
then runs whichever target (``"chain"`` or ``"agent"``) the caller asked
for. With ``LLM_PROVIDER=stub`` the answer-substring metric is meaningless
(the stub doesn't answer); ``context_precision_at_k`` is still useful and
the unit tests pin the LLM via dependency injection.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

import mlflow
import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from sorakai.chains.agent_graph import ainvoke_agent, build_agent_graph
from sorakai.chains.rag_chain import ainvoke_rag, build_rag_chain
from sorakai.common.chat_history import InMemoryChatHistoryStore
from sorakai.common.config import Settings
from sorakai.common.embedding import embed_chunks
from sorakai.common.ingest import chunk_document
from sorakai.common.mlflow_callback import MlflowChainCallback
from sorakai.core.logging import get_logger
from sorakai.eval.dataset import EvalDataset
from sorakai.eval.scorer import (
    PASS_RATE_KEY,
    CaseScore,
    aggregate_scores,
    maybe_score_with_ragas,
    score_case,
)
from sorakai.infra.vector_store import VectorStore
from sorakai.infra.vector_store.base import VectorDoc
from sorakai.infra.vector_store.factory import get_vector_store

logger = get_logger(__name__)

EvalTarget = Literal["chain", "agent"]

_DEFAULT_EVAL_CHUNK_SIZE = 400
_DEFAULT_EVAL_CHUNK_OVERLAP = 50
_EVAL_SESSION_PREFIX = "eval-"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """One row of an eval run."""

    case_id: str
    question: str
    answer: str
    retrieved_doc_ids: tuple[str, ...]
    score: CaseScore
    latency_ms: float
    ragas: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregate eval bundle (CLI prints / MLflow logs this)."""

    target: EvalTarget
    cases: tuple[EvalCaseResult, ...]
    metrics: Mapping[str, float]

    @property
    def pass_rate(self) -> float:
        return float(self.metrics.get(PASS_RATE_KEY, 0.0))


# ---------------------------------------------------------------------------
# Retrieval capture callback
# ---------------------------------------------------------------------------


class _RetrievalCapture(BaseCallbackHandler):
    """Records the ``doc_id`` of every document returned by a retriever.

    LangChain dispatches ``on_retriever_end`` for any
    :class:`~langchain_core.retrievers.BaseRetriever` in the chain, so this
    handler observes both the LCEL chain's :class:`VectorStoreRetriever`
    and (when wired) the agent's KB-search tool.
    """

    raise_error: bool = False
    run_inline: bool = True

    def __init__(self) -> None:
        self._docs: list[Document] = []

    def reset(self) -> None:
        self._docs.clear()

    @property
    def doc_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for doc in self._docs:
            doc_id = str((doc.metadata or {}).get("doc_id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
        return tuple(out)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._docs.extend(documents)


# ---------------------------------------------------------------------------
# Corpus seeding
# ---------------------------------------------------------------------------


async def seed_eval_store(
    vector_store: VectorStore,
    dataset: EvalDataset,
    *,
    chunk_size: int = _DEFAULT_EVAL_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_EVAL_CHUNK_OVERLAP,
) -> int:
    """Ingest the dataset's corpus into ``vector_store``.

    Returns the total number of chunks written so callers (and tests)
    can sanity-check the seed actually populated the KB.
    """
    if not dataset.corpus:
        return 0
    total = 0
    for doc_id, content in dataset.corpus:
        chunks = chunk_document(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            filename=f"{doc_id}.md",
        )
        if not chunks:
            continue
        vectors = await embed_chunks(chunks)
        docs = [
            VectorDoc(
                page_content=chunk,
                embedding=np.asarray(vec, dtype=np.float32),
                metadata={
                    "doc_id": doc_id,
                    "filename": f"{doc_id}.md",
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "mime": "text/markdown",
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await vector_store.upsert(docs)
        total += len(docs)
    logger.info("eval corpus seeded: docs=%d chunks=%d", len(dataset.corpus), total)
    return total


# ---------------------------------------------------------------------------
# Target runners
# ---------------------------------------------------------------------------


async def _run_chain_case(
    chain: Any,
    case_id: str,
    question: str,
    *,
    callbacks: list[Any],
) -> tuple[str, float]:
    started = time.perf_counter()
    result = await ainvoke_rag(
        chain,
        question=question,
        session_id=f"{_EVAL_SESSION_PREFIX}{case_id}",
        callbacks=callbacks,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = str(result.get("answer") or "")
    return answer, latency_ms


async def _run_agent_case(
    graph: Any,
    case_id: str,
    question: str,
    *,
    settings: Settings,
    callbacks: list[Any],
) -> tuple[str, float]:
    started = time.perf_counter()
    state = await ainvoke_agent(
        graph,
        question=question,
        session_id=f"{_EVAL_SESSION_PREFIX}{case_id}",
        max_steps=settings.agent_max_steps,
        callbacks=callbacks,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = str(state.get("answer") or "")
    return answer, latency_ms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_eval(
    *,
    target: EvalTarget,
    settings: Settings,
    dataset: EvalDataset,
    llm: BaseChatModel | None = None,
    vector_store: VectorStore | None = None,
    mlflow_callback: MlflowChainCallback | None = None,
) -> EvalResult:
    """Drive ``dataset`` through the configured chain or agent.

    The function builds an isolated in-memory chat-history store so each
    eval case gets a fresh session and the regression score doesn't
    depend on prior conversation state.

    Returns an :class:`EvalResult` the CLI prints and (optionally) logs
    to MLflow.
    """
    actual_store: VectorStore = vector_store or get_vector_store(settings)
    await seed_eval_store(actual_store, dataset)

    chat_store = InMemoryChatHistoryStore(max_messages=settings.chat_history_max_messages)

    chain: Any = None
    graph: Any = None
    if target == "chain":
        chain, _ = await build_rag_chain(settings, actual_store, chat_store, llm=llm)
    elif target == "agent":
        graph, _ = build_agent_graph(settings, actual_store, chat_store, llm=llm)
    else:  # pragma: no cover - exhaustive Literal already enforces this
        raise ValueError(f"unknown eval target: {target!r}")

    results: list[EvalCaseResult] = []
    for case in dataset:
        capture = _RetrievalCapture()
        callbacks: list[Any] = [capture]
        if mlflow_callback is not None:
            callbacks.append(mlflow_callback)
        if target == "chain":
            answer, latency_ms = await _run_chain_case(chain, case.id, case.question, callbacks=callbacks)
        else:
            answer, latency_ms = await _run_agent_case(
                graph, case.id, case.question, settings=settings, callbacks=callbacks
            )
        retrieved = capture.doc_ids
        case_score = score_case(case, answer=answer, retrieved_doc_ids=retrieved)
        ragas_scores = maybe_score_with_ragas(case, answer=answer, retrieved_contexts=[])
        results.append(
            EvalCaseResult(
                case_id=case.id,
                question=case.question,
                answer=answer,
                retrieved_doc_ids=retrieved,
                score=case_score,
                latency_ms=latency_ms,
                ragas=ragas_scores,
            )
        )

    metrics = dict(aggregate_scores([r.score for r in results]))
    metrics["mean_latency_ms"] = sum(r.latency_ms for r in results) / float(len(results)) if results else 0.0
    if mlflow_callback is not None:
        _log_eval_to_mlflow(mlflow_callback, target=target, metrics=metrics, n_cases=len(results))
    return EvalResult(target=target, cases=tuple(results), metrics=metrics)


def _log_eval_to_mlflow(
    callback: MlflowChainCallback,
    *,
    target: EvalTarget,
    metrics: Mapping[str, float],
    n_cases: int,
) -> None:
    """Best-effort dump of aggregate metrics into the active MLflow run.

    The callback opens its own run on the first chain start; we piggyback
    so the per-case latencies + the eval aggregates live in the same
    MLflow row, and we never raise (eval should never break because the
    tracking server is down).
    """
    try:
        if not callback.is_active:
            mlflow.set_experiment(callback.experiment_name)
            mlflow.start_run(run_name=callback.run_name)
        for key, value in metrics.items():
            mlflow.log_metric(f"eval_{key}", float(value))
        mlflow.log_param("eval_target", target)
        mlflow.log_param("eval_cases", n_cases)
    except Exception as exc:
        logger.warning("MLflow eval logging failed: %s", exc)


__all__ = [
    "EvalCaseResult",
    "EvalResult",
    "EvalTarget",
    "run_eval",
    "seed_eval_store",
]
