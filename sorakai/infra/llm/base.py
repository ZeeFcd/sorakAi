"""Canonical chat-model abstract type for sorakAi.

We deliberately reuse ``langchain_core``'s ``BaseChatModel`` so any host that
already has a LangChain adapter (Ollama today; OpenAI, Anthropic, Bedrock,
vLLM, LM Studio, ... tomorrow) can be slotted in by writing a tiny factory
function and registering it - no chain or handler change required.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

__all__ = ["BaseChatModel"]
