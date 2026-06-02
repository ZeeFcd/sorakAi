"""Ollama chat-model adapter.

Wave 1 ships the minimal viable wrapper around ``ChatOllama`` from
``langchain-ollama``. Performance work (streaming options, keep-alive,
JSON-mode, structured outputs) lives in later waves of the overhaul plan.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from sorakai.common.config import Settings
from sorakai.infra.llm.base import BaseChatModel


def build_ollama_chat(settings: Settings) -> BaseChatModel:
    """Construct a ``ChatOllama`` from the project's settings.

    All knobs come from :class:`sorakai.common.config.Settings` so the rest of
    the code never reads env vars directly (DIP).
    """
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
