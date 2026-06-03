"""End-to-end smoke tests for the Wave 8 observability wiring.

We don't go through the FastAPI lifespan here (the in-process telemetry
plumbing is exercised in ``test_telemetry.py``); instead we wire the
MlflowChainCallback into a real :func:`build_rag_chain` call and assert
the callback observed the LLM + retrieval round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from sorakai.chains.rag_chain import ainvoke_rag, build_rag_chain
from sorakai.common.chat_history import InMemoryChatHistoryStore
from sorakai.common.config import get_settings
from sorakai.common.mlflow_callback import MlflowChainCallback
from sorakai.common.store import InMemoryKnowledgeStore
from sorakai.infra.embeddings.char import CharPseudoEmbeddings
from sorakai.infra.vector_store.knowledge_store import KnowledgeStoreVectorStore


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    class _FakeRun:
        def __init__(self, run_id: str) -> None:
            class _Info:
                pass

            self.info = _Info()
            self.info.run_id = run_id

    class _FakeMlflow:
        def __init__(self) -> None:
            self.params: dict[str, str] = {}
            self.metrics: list[tuple[str, float]] = []
            self.ended = False
            self.tags: dict[str, str] = {}

        def set_tracking_uri(self, uri: str) -> None: ...
        def set_experiment(self, name: str) -> None: ...
        def start_run(self, run_name: str | None = None) -> _FakeRun:
            return _FakeRun(run_id="fake")

        def log_params(self, params: dict[str, str]) -> None:
            self.params.update(params)

        def log_metric(self, key: str, value: float) -> None:
            self.metrics.append((key, value))

        def set_tag(self, key: str, value: str) -> None:
            self.tags[key] = value

        def end_run(self) -> None:
            self.ended = True

    fake = _FakeMlflow()
    monkeypatch.setattr("sorakai.common.mlflow_callback.mlflow", fake)
    return fake


async def _seed_store(chunks: list[tuple[str, str]]) -> KnowledgeStoreVectorStore:
    store = InMemoryKnowledgeStore()
    embedder = CharPseudoEmbeddings()
    for doc_id, text in chunks:
        vec = (await embedder.aembed_documents([text]))[0]
        await store.append_document(doc_id, f"{doc_id}.txt", [text], [np.asarray(vec, dtype=float)])
    return KnowledgeStoreVectorStore(store)


@pytest.mark.asyncio
async def test_mlflow_callback_observes_chain_invocation(fake_mlflow) -> None:
    """One ``ainvoke_rag`` call must result in one MLflow run with
    at least the LLM + retrieval counters set."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids are in Egypt")])
    chat = InMemoryChatHistoryStore()
    chain, _ = await build_rag_chain(settings, vstore, chat)

    callback = MlflowChainCallback(
        "sorakai-rag",
        run_name="query",
        tracking_uri="memory",
        static_params={"service": "rag-test"},
    )
    await ainvoke_rag(chain, question="pyramids?", session_id=None, callbacks=[callback])

    assert fake_mlflow.ended, "MLflow run was not closed"
    metric_map = dict(fake_mlflow.metrics)
    assert metric_map["llm_calls"] >= 1
    assert metric_map["retrievals"] >= 1
    assert metric_map["docs_retrieved"] >= 1
    assert metric_map["answer_len"] > 0
    assert fake_mlflow.params["service"] == "rag-test"


@pytest.mark.asyncio
async def test_chain_callbacks_optional_no_op(fake_mlflow) -> None:
    """Passing ``callbacks=None`` keeps the chain quiet and doesn't open
    an MLflow run - the cheap default path for tests + dev."""
    settings = get_settings()
    vstore = await _seed_store([("d1", "x")])
    chat = InMemoryChatHistoryStore()
    chain, _ = await build_rag_chain(settings, vstore, chat)

    await ainvoke_rag(chain, question="?", session_id=None, callbacks=None)

    assert not fake_mlflow.ended
    assert not fake_mlflow.metrics


@pytest.mark.asyncio
async def test_agent_graph_propagates_callbacks(fake_mlflow) -> None:
    """``ainvoke_agent`` must forward callbacks into the compiled graph so
    the same callback observes both chain and agent runs."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from sorakai.chains.agent_graph import ainvoke_agent, build_agent_graph

    settings = get_settings()
    vstore = await _seed_store([("d1", "Pyramids are in Egypt")])
    chat = InMemoryChatHistoryStore()
    llm = FakeListChatModel(responses=["kb", "good", "Pyramids are in Egypt.", "ok"])
    graph, _ = build_agent_graph(settings, vstore, chat, llm=llm)

    callback = MlflowChainCallback("sorakai-agent", tracking_uri="memory")
    await ainvoke_agent(
        graph,
        question="pyramids?",
        session_id=None,
        max_steps=4,
        callbacks=[callback],
    )

    assert fake_mlflow.ended
    metric_map = dict(fake_mlflow.metrics)
    # Route + grade + generate + critique = 4 LLM calls.
    assert metric_map["llm_calls"] == 4
    assert metric_map["retrievals"] >= 1
