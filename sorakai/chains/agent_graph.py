"""LangGraph agent for ``POST /v1/agent`` (Wave 7).

Topology (mirrors the plan):

```
                ┌──────────┐
                │  start   │
                └────┬─────┘
                     ▼
              ┌─────────────┐    kb        ┌──────────┐
              │    route    ├─────────────▶│ retrieve │
              └────┬────────┘              └────┬─────┘
                   │chitchat                    ▼
                   ▼                       ┌─────────┐
               ┌───────────┐  weak ◀───────┤  grade  │
               │ generate  │               └────┬────┘
               └────┬──────┘                    │ good
                    ▼                           ▼
              ┌──────────┐                 ┌──────────┐
              │ critique ├──── retry ─────▶│ rewrite  │
              └────┬─────┘                 └────┬─────┘
                   │ ok                          │
                   ▼                             ▼
                  END                       (back to retrieve)
```

Why this graph (vs the LCEL chain)
-----------------------------------

The Wave 6 LCEL chain is a fast, linear ``retrieve -> prompt -> llm`` flow
- the right tool when the question always needs the KB. The agent adds:

- **Routing**: cheap chit-chat questions skip retrieval entirely.
- **Self-correction**: a weak retrieval gets rewritten and retried; an
  off-topic answer gets re-generated.
- **Auditability**: every node visit and every tool call is recorded in
  ``state['trace']`` and ``state['tool_calls']`` so the response payload
  is useful for debugging without enabling tracing.

Determinism + cost control
--------------------------

- The LLM is only consulted on classification nodes (route / grade /
  critique) with a one-word output contract; tests pin the responses via
  :class:`langchain_core.language_models.fake_chat_models.FakeListChatModel`.
- ``state['step']`` caps loop iterations at ``settings.agent_max_steps``;
  exceeded steps short-circuit to ``generate`` with the best context so
  far so the agent always produces *something*.
- The agent never imports a concrete provider - it asks
  :func:`get_chat_model` like every other Wave 1+ entry point, so swapping
  Ollama for a future provider is one env var.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import add
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sorakai.chains.history import SorakaiChatMessageHistory
from sorakai.chains.prompts import (
    build_agent_chitchat_prompt,
    build_agent_critique_prompt,
    build_agent_grade_prompt,
    build_agent_rewrite_prompt,
    build_agent_route_prompt,
    build_rag_prompt,
)
from sorakai.chains.rag_chain import format_docs
from sorakai.chains.retriever import VectorStoreRetriever
from sorakai.chains.tools import (
    CalcTool,
    KBSearchTool,
    ToolCall,
    ToolRegistry,
    WebSearchTool,
    run_tool,
)
from sorakai.common.chat_history import InMemoryChatHistoryStore, RedisChatHistoryStore
from sorakai.common.config import Settings
from sorakai.core.logging import get_logger
from sorakai.infra.embeddings import get_embeddings
from sorakai.infra.llm import get_chat_model
from sorakai.infra.vector_store import VectorStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

Route = Literal["kb", "chitchat"]
Grade = Literal["good", "weak"]
Critique = Literal["ok", "retry"]


class AgentState(TypedDict, total=False):
    """Per-invocation state carried through the graph.

    ``trace`` and ``tool_calls`` use the ``operator.add`` reducer so each
    node can return only its delta and the merge happens server-side; the
    rest of the fields are last-writer-wins scalars.
    """

    question: str
    query: str
    session_id: str | None
    history: list[BaseMessage]
    docs: list[Document]
    context: str
    answer: str
    route: Route
    last_grade: Grade
    last_critique: Critique
    step: int
    max_steps: int
    trace: Annotated[list[str], add]
    tool_calls: Annotated[list[ToolCall], add]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_label(raw: str, allowed: Sequence[str], default: str) -> str:
    """Strip / lowercase the LLM's label and fall back to ``default`` if
    the model went off-script. The agent must always make forward progress."""
    cleaned = raw.strip().lower().splitlines()[0].strip(" .,'\"") if raw.strip() else ""
    for token in allowed:
        if cleaned.startswith(token):
            return token
    logger.warning("agent classifier returned %r, defaulting to %s", raw, default)
    return default


async def _classify(llm: BaseChatModel, prompt_value: Any, *, allowed: Sequence[str], default: str) -> str:
    response = await llm.ainvoke(prompt_value.to_messages())
    parser = StrOutputParser()
    return _normalise_label(parser.invoke(response), allowed=allowed, default=default)


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_route_node(llm: BaseChatModel) -> Any:
    prompt = build_agent_route_prompt()

    async def route_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        label = await _classify(
            llm,
            prompt.invoke({"question": question}),
            allowed=("kb", "chitchat"),
            default="kb",
        )
        return {
            "route": cast(Route, label),
            "query": question,
            "step": state.get("step", 0),
            "trace": ["route"],
        }

    return route_node


def _make_retrieve_node(registry: ToolRegistry, top_k: int) -> Any:
    async def retrieve_node(state: AgentState) -> dict[str, Any]:
        call = await run_tool(registry, "kb_search", query=state["query"], k=top_k)
        docs = cast(list[Document], call.output or [])
        return {
            "docs": docs,
            "context": format_docs(docs),
            "tool_calls": [call],
            "trace": ["retrieve"],
            "step": state.get("step", 0) + 1,
        }

    return retrieve_node


def _make_grade_node(llm: BaseChatModel) -> Any:
    prompt = build_agent_grade_prompt()

    async def grade_node(state: AgentState) -> dict[str, Any]:
        context = state.get("context", "")
        if not context.strip():
            return {"last_grade": "weak", "trace": ["grade"]}
        label = await _classify(
            llm,
            prompt.invoke({"question": state["question"], "context": context}),
            allowed=("good", "weak"),
            default="good",
        )
        return {"last_grade": label, "trace": ["grade"]}

    return grade_node


def _make_rewrite_node(llm: BaseChatModel) -> Any:
    prompt = build_agent_rewrite_prompt()
    parser = StrOutputParser()

    async def rewrite_node(state: AgentState) -> dict[str, Any]:
        response = await llm.ainvoke(
            prompt.invoke({"question": state["question"], "query": state.get("query", state["question"])}).to_messages()
        )
        new_query = parser.invoke(response).strip() or state["question"]
        return {"query": new_query, "trace": ["rewrite"]}

    return rewrite_node


def _make_generate_node(llm: BaseChatModel) -> Any:
    rag_prompt = build_rag_prompt()
    chit_prompt = build_agent_chitchat_prompt()
    parser = StrOutputParser()

    async def generate_node(state: AgentState) -> dict[str, Any]:
        history: Sequence[BaseMessage] = state.get("history", []) or []
        context = state.get("context", "")
        if state.get("route") == "chitchat" or not context.strip():
            prompt_value = chit_prompt.invoke({"question": state["question"], "history": list(history)})
        else:
            prompt_value = rag_prompt.invoke(
                {"question": state["question"], "context": context, "history": list(history)}
            )
        response = await llm.ainvoke(prompt_value.to_messages())
        return {
            "answer": parser.invoke(response).strip(),
            "trace": ["generate"],
        }

    return generate_node


def _make_critique_node(llm: BaseChatModel) -> Any:
    prompt = build_agent_critique_prompt()

    async def critique_node(state: AgentState) -> dict[str, Any]:
        # Chit-chat answers don't go through critique; the grader/router is
        # the only quality gate they need.
        if state.get("route") == "chitchat":
            return {"last_critique": "ok", "trace": ["critique"]}
        label = await _classify(
            llm,
            prompt.invoke(
                {
                    "question": state["question"],
                    "context": state.get("context", ""),
                    "answer": state.get("answer", ""),
                }
            ),
            allowed=("ok", "retry"),
            default="ok",
        )
        return {"last_critique": label, "trace": ["critique"]}

    return critique_node


# ---------------------------------------------------------------------------
# Branch functions
# ---------------------------------------------------------------------------


def _route_branch(state: AgentState) -> str:
    return state.get("route", "kb")


def _grade_branch(state: AgentState) -> str:
    """Weak grade triggers a rewrite, but only while we have budget."""
    if state.get("step", 0) >= state.get("max_steps", 4):
        return "good"
    return state.get("last_grade", "good")


def _critique_branch(state: AgentState) -> str:
    if state.get("step", 0) >= state.get("max_steps", 4):
        return "ok"
    return state.get("last_critique", "ok")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_tool_registry(
    settings: Settings,
    vector_store: VectorStore,
) -> ToolRegistry:
    """Wire up the three Wave 7 tools against the live infrastructure."""
    embeddings = get_embeddings(settings)
    retriever = VectorStoreRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        k=settings.rag_top_k,
    )
    registry = ToolRegistry()
    registry.register(KBSearchTool(retriever=retriever))
    registry.register(CalcTool())
    registry.register(WebSearchTool(enabled=settings.web_search_enabled))
    return registry


def build_agent_graph(
    settings: Settings,
    vector_store: VectorStore,
    chat_store: RedisChatHistoryStore | InMemoryChatHistoryStore,
    *,
    llm: BaseChatModel | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[CompiledStateGraph[AgentState, Any, Any, Any], ToolRegistry]:
    """Compile the graph + return the tool registry the caller can audit.

    ``llm`` and ``registry`` are injection points for tests; production
    callers leave them ``None`` and the factories take over (same pattern
    as :func:`sorakai.chains.rag_chain.build_rag_chain`).
    """
    actual_llm = llm or get_chat_model(settings)
    actual_registry = registry or build_tool_registry(settings, vector_store)

    builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
    builder.add_node("route", _make_route_node(actual_llm))
    builder.add_node("retrieve", _make_retrieve_node(actual_registry, settings.rag_top_k))
    builder.add_node("grade", _make_grade_node(actual_llm))
    builder.add_node("rewrite", _make_rewrite_node(actual_llm))
    builder.add_node("generate", _make_generate_node(actual_llm))
    builder.add_node("critique", _make_critique_node(actual_llm))

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", _route_branch, {"kb": "retrieve", "chitchat": "generate"})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", _grade_branch, {"good": "generate", "weak": "rewrite"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges("critique", _critique_branch, {"ok": END, "retry": "rewrite"})

    compiled = builder.compile()
    # Lift the chat-store reference so the public ``ainvoke_agent`` wrapper
    # can hydrate / persist history without the graph having to.
    compiled._sorakai_chat_store = chat_store  # type: ignore[attr-defined]
    logger.info(
        "Agent graph built: llm=%s vector_store=%s max_steps=%s web_search=%s",
        settings.llm_provider,
        settings.vector_store,
        settings.agent_max_steps,
        settings.web_search_enabled,
    )
    return compiled, actual_registry


# ---------------------------------------------------------------------------
# Public invocation helper
# ---------------------------------------------------------------------------


async def ainvoke_agent(
    graph: CompiledStateGraph[AgentState, Any, Any, Any],
    *,
    question: str,
    session_id: str | None,
    max_steps: int,
) -> AgentState:
    """Run the agent end-to-end, hydrating + persisting chat history.

    Returns the final :class:`AgentState` (which the handler reshapes into
    the ``AgentResponse`` schema).
    """
    chat_store = cast(
        "RedisChatHistoryStore | InMemoryChatHistoryStore",
        graph._sorakai_chat_store,  # type: ignore[attr-defined]
    )
    history: list[BaseMessage] = []
    history_adapter: SorakaiChatMessageHistory | None = None
    if session_id:
        history_adapter = SorakaiChatMessageHistory(chat_store, session_id)
        history = await history_adapter.aget_messages()

    initial: AgentState = {
        "question": question,
        "query": question,
        "session_id": session_id,
        "history": history,
        "docs": [],
        "context": "",
        "answer": "",
        "route": "kb",
        "step": 0,
        "max_steps": max_steps,
        "trace": [],
        "tool_calls": [],
    }
    # LangGraph applies the reducer when we update from inside nodes, but the
    # initial value still has to be a list; an empty list is fine.

    result = cast(AgentState, await graph.ainvoke(initial))

    if history_adapter is not None and result.get("answer"):
        await history_adapter.aadd_messages([HumanMessage(content=question), AIMessage(content=result["answer"])])
    return result


__all__ = [
    "AgentState",
    "Critique",
    "Grade",
    "Route",
    "ainvoke_agent",
    "build_agent_graph",
    "build_tool_registry",
]
