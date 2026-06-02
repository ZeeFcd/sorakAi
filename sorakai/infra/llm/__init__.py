"""Chat-model adapters and factory.

Public API:

- :class:`BaseChatModel` (re-exported from ``langchain_core``) - the abstract
  type that every adapter returns.
- :func:`get_chat_model` - factory keyed by ``Settings.llm_provider``.
- :func:`register_chat_model` - public registration hook so a future provider
  package can self-register at import time.
"""

from sorakai.infra.llm.base import BaseChatModel
from sorakai.infra.llm.factory import (
    CHAT_MODEL_REGISTRY,
    ChatModelBuilder,
    get_chat_model,
    register_chat_model,
)

__all__ = [
    "CHAT_MODEL_REGISTRY",
    "BaseChatModel",
    "ChatModelBuilder",
    "get_chat_model",
    "register_chat_model",
]
