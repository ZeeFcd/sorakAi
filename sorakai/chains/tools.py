"""Tools the Wave 7 agent graph can call.

Each tool is a small async callable with a strict, dataclass-typed input
and output so the graph can stash the invocation in the audit trail
(:class:`ToolCall`) without serialising arbitrary objects. The interface
is deliberately decoupled from LangChain's ``BaseTool`` - the agent graph
calls these directly so we don't pay for the structured-tool wrapper, and
so swapping a tool for a Wave 9 evaluation harness is a one-line change.

Three tools ship in Wave 7:

- :func:`kb_search` - retrieve top-``k`` chunks from the configured
  :class:`~sorakai.infra.vector_store.base.VectorStore` via the same Wave 6
  ``VectorStoreRetriever`` the chain uses (so the agent and the chain
  agree on what "knowledge base" means).
- :func:`safe_calc` - AST-walking arithmetic evaluator. No ``eval``, no
  attribute access, no calls; exponent is capped so a malicious agent
  prompt can't pin a CPU. Used by the agent for cheap "what's 3% of 487?"
  questions without round-tripping through the LLM.
- :func:`web_search` - feature-flagged off by default
  (``WEB_SEARCH_ENABLED=false``); returns ``[]`` in that mode so the graph
  can call it unconditionally without leaking outbound HTTP. A future
  wave drops in a real SearXNG / DuckDuckGo HTML adapter behind the same
  signature.
"""

from __future__ import annotations

import ast
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document

from sorakai.chains.retriever import VectorStoreRetriever
from sorakai.core.errors import SorakaiError
from sorakai.core.logging import get_logger

logger = get_logger(__name__)


class ToolError(SorakaiError):
    """Tool refused to execute (bad input, disabled, evaluator rejected expr)."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation captured for the agent's audit trail."""

    name: str
    input: dict[str, Any]
    output: Any
    duration_ms: float
    error: str | None = None


@runtime_checkable
class AsyncTool(Protocol):
    """The shape every tool must satisfy at runtime.

    Each concrete tool keeps its own strict ``ainvoke`` signature (e.g.
    :class:`KBSearchTool` accepts ``query`` + optional ``k``); the Protocol
    just promises the duck-typed surface so :class:`ToolRegistry` can
    accept heterogenous tools in one container without forcing them all
    to share the exact same kwarg shape. Static type checking on the
    individual ``ainvoke`` signatures is what catches malformed call
    sites at the call site, not at registration.
    """

    name: str

    async def ainvoke(self, /, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# kb_search
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KBSearchTool:
    """Wrap the Wave 6 retriever so the agent can call it like any other tool."""

    retriever: VectorStoreRetriever
    name: str = "kb_search"

    async def ainvoke(self, *, query: str, k: int | None = None) -> list[Document]:
        if not query.strip():
            raise ToolError("kb_search: query is required")
        # The retriever's ``k`` is fixed at construction; honour an override
        # by temporarily swapping it so we don't allocate a new retriever
        # per call.
        original = self.retriever.k
        try:
            if k is not None:
                self.retriever.k = max(1, min(int(k), 50))
            return await self.retriever.ainvoke(query)
        finally:
            self.retriever.k = original


# ---------------------------------------------------------------------------
# safe_calc
# ---------------------------------------------------------------------------


_BINOPS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.AST], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_POWER = 50
"""Largest exponent we allow ``a**b`` to use. Guards against CPU bombs like
``2**1000000000`` without imposing a useful-case-blocking small limit."""


def safe_calc(expr: str) -> float:
    """Evaluate a numeric expression. Raises :class:`ToolError` on anything
    that isn't pure arithmetic. Result is always a :class:`float`."""
    if not expr.strip():
        raise ToolError("calc: expression is required")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"calc: invalid expression - {e}") from e
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):  # bool is a subclass of int; reject explicitly
            raise ToolError("calc: booleans are not numbers")
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ToolError(f"calc: only numeric constants allowed, got {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        bin_op_type: type[ast.AST] = type(node.op)
        if bin_op_type not in _BINOPS:
            raise ToolError(f"calc: operator {bin_op_type.__name__} not allowed")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if bin_op_type is ast.Pow and (right > _MAX_POWER or right < -_MAX_POWER):
            raise ToolError(f"calc: exponent {right} exceeds limit {_MAX_POWER}")
        return _BINOPS[bin_op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        un_op_type: type[ast.AST] = type(node.op)
        if un_op_type not in _UNARY:
            raise ToolError(f"calc: unary operator {un_op_type.__name__} not allowed")
        return _UNARY[un_op_type](_eval_node(node.operand))
    raise ToolError(f"calc: AST node {type(node).__name__} not allowed")


@dataclass(slots=True)
class CalcTool:
    name: str = "calc"

    async def ainvoke(self, *, expr: str) -> float:
        return safe_calc(expr)


# ---------------------------------------------------------------------------
# web_search (stubbed off by default)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WebSearchTool:
    """Off-by-default web search.

    The ``enabled=False`` branch returns ``[]`` and logs a one-line warning
    so the agent doesn't silently rely on it; the ``enabled=True`` branch
    is a stub today and raises :class:`ToolError` so a misconfigured prod
    deployment fails loudly instead of pretending to call the internet.
    """

    enabled: bool = False
    name: str = "web_search"

    async def ainvoke(self, *, query: str) -> list[dict[str, str]]:
        if not query.strip():
            raise ToolError("web_search: query is required")
        if not self.enabled:
            logger.debug("web_search disabled - returning empty result set")
            return []
        raise ToolError(
            "web_search: enabled but no provider configured - "
            "register a backend via WebSearchTool subclass or wait for a later wave"
        )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolRegistry:
    """Look-up by name. Used by the agent graph to dispatch tool calls.

    Stored as ``dict[str, Any]`` rather than ``dict[str, AsyncTool]`` so
    each tool can keep its own strict ``ainvoke(*, ...)`` signature
    without us widening every kwarg to ``Any`` at the call site. The
    ``register`` method does a runtime structural check so configuration
    errors still surface immediately.
    """

    tools: dict[str, Any] = field(default_factory=dict)

    def register(self, tool: Any) -> None:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"tool {tool!r} must expose a non-empty .name")
        if not callable(getattr(tool, "ainvoke", None)):
            raise TypeError(f"tool {name!r} must expose an async .ainvoke")
        if name in self.tools:
            raise ValueError(f"tool {name!r} already registered")
        self.tools[name] = tool

    def get(self, name: str) -> Any:
        try:
            return self.tools[name]
        except KeyError as e:
            raise ToolError(f"unknown tool: {name}") from e

    def names(self) -> list[str]:
        return sorted(self.tools)


async def run_tool(registry: ToolRegistry, name: str, /, **kwargs: Any) -> ToolCall:
    """Dispatch a tool by name, returning a fully-populated :class:`ToolCall`."""
    tool = registry.get(name)
    start = time.perf_counter()
    error: str | None = None
    output: Any = None
    try:
        output = await tool.ainvoke(**kwargs)
    except ToolError as e:
        error = str(e)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return ToolCall(name=name, input=dict(kwargs), output=output, duration_ms=duration_ms, error=error)


__all__ = [
    "AsyncTool",
    "CalcTool",
    "KBSearchTool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "WebSearchTool",
    "run_tool",
    "safe_calc",
]
