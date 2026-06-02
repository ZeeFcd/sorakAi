"""Chat-model factory: env-driven lookup over a top-level registry.

OCP: adding a new provider is one new file under ``sorakai/infra/llm/`` plus
one entry appended to :data:`CHAT_MODEL_REGISTRY` (or one call to
:func:`register_chat_model` from a separate package). The factory itself
never changes.

DIP: all call sites depend on :class:`BaseChatModel`, never on a concrete
adapter class.
"""

from __future__ import annotations

from collections.abc import Callable

from sorakai.common.config import Settings
from sorakai.core.errors import ConfigError
from sorakai.infra.llm.base import BaseChatModel
from sorakai.infra.llm.ollama import build_ollama_chat
from sorakai.infra.llm.stub import build_stub_chat

ChatModelBuilder = Callable[[Settings], BaseChatModel]

CHAT_MODEL_REGISTRY: dict[str, ChatModelBuilder] = {
    "ollama": build_ollama_chat,
    "stub": build_stub_chat,
}


def register_chat_model(name: str, builder: ChatModelBuilder) -> None:
    """Register or replace a chat-model builder under ``name``.

    Public hook so a future provider package (or a test) can self-register at
    import time without modifying this module.
    """
    CHAT_MODEL_REGISTRY[name] = builder


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return the configured :class:`BaseChatModel` instance.

    Raises :class:`sorakai.core.errors.ConfigError` when
    ``Settings.llm_provider`` is not in :data:`CHAT_MODEL_REGISTRY`.
    """
    try:
        builder = CHAT_MODEL_REGISTRY[settings.llm_provider]
    except KeyError as exc:
        raise ConfigError(
            f"Unknown LLM_PROVIDER={settings.llm_provider!r}; registered: {sorted(CHAT_MODEL_REGISTRY)}"
        ) from exc
    return builder(settings)
