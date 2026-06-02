"""Tests for the ``sorakai.common.llm`` shim.

We monkeypatch ``sorakai.common.llm.get_chat_model`` to return a recording
chat model so the test asserts the exact :class:`BaseMessage` sequence the
handler builds (system + prior turns + final user) - independently of the
``LLMProvider`` literal and of any concrete provider's text format.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sorakai.common.llm import SYSTEM_PROMPT, ask_llm
from sorakai.infra.llm.base import BaseChatModel


class _RecorderChatModel(BaseChatModel):
    last_messages: ClassVar[list[BaseMessage]] = []
    reply: str = "ok"

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **_kwargs):
        type(self).last_messages = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _agenerate(self, messages, stop=None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **_k):
        return self._generate(messages, stop=stop)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_RecorderChatModel]:
    instance = _RecorderChatModel()
    monkeypatch.setattr("sorakai.common.llm.get_chat_model", lambda _s: instance)
    _RecorderChatModel.last_messages = []
    _RecorderChatModel.reply = "ok"
    return _RecorderChatModel


def test_ask_llm_builds_system_plus_prior_plus_user(recorder, run_async) -> None:
    prior = [
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    ]
    answer = run_async(ask_llm("new-q", "the-context", conversation=prior))
    assert answer == "ok"

    msgs = recorder.last_messages
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == SYSTEM_PROMPT
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "earlier-q"
    assert isinstance(msgs[2], AIMessage) and msgs[2].content == "earlier-a"
    assert isinstance(msgs[3], HumanMessage)
    assert "the-context" in str(msgs[3].content)
    assert "new-q" in str(msgs[3].content)


def test_ask_llm_skips_empty_or_unknown_turns(recorder, run_async) -> None:
    prior = [
        {"role": "user", "content": ""},
        {"role": "system", "content": "should-be-ignored"},
        {"role": "assistant", "content": "kept"},
    ]
    run_async(ask_llm("q", "ctx", conversation=prior))
    roles = [type(m).__name__ for m in recorder.last_messages]
    assert roles == ["SystemMessage", "AIMessage", "HumanMessage"]


def test_ask_llm_works_without_conversation(recorder, run_async) -> None:
    answer = run_async(ask_llm("q", "ctx"))
    assert answer == "ok"
    assert len(recorder.last_messages) == 2  # system + user
