"""Deterministic stub chat-model used by tests and offline dev.

The stub is intentionally a real :class:`BaseChatModel` subclass (not a
free-standing function) so it composes with LCEL chains, ``RunnableWithMessageHistory``,
and the LangGraph agent later waves will introduce - exactly like any other
provider would. Tests that need to inspect what the LLM saw can monkeypatch
the registry to return a recording variant; see ``tests/test_llm_shim.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sorakai.common.config import Settings
from sorakai.infra.llm.base import BaseChatModel


def _summarise_input(messages: Sequence[BaseMessage]) -> str:
    """Compact, deterministic summary of the most recent user message + history depth.

    Kept stable so tests and humans can reason about the response without
    depending on a real model. The format is documented and may be relied on
    in tests.
    """
    last_user = next(
        (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)),
        "<no user message>",
    )
    snippet = last_user if len(last_user) <= 120 else last_user[:117] + "..."
    history_pairs = sum(1 for m in messages if isinstance(m, AIMessage))
    return f"[stub] q={snippet!r} history_pairs={history_pairs}"


class StubChatModel(BaseChatModel):
    """Returns ``_summarise_input(messages)`` for every call. No network."""

    @property
    def _llm_type(self) -> str:
        return "sorakai-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = _summarise_input(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


def build_stub_chat(_settings: Settings) -> BaseChatModel:
    """Build a stub chat model. Independent of settings on purpose."""
    return StubChatModel()
