"""Thin async LLM call site used by the RAG handler.

All provider-specific code lives behind :func:`sorakai.infra.llm.factory.get_chat_model`,
so swapping or adding providers never touches this file or the handlers.
Tests can flip ``LLM_PROVIDER`` to ``stub`` (default in tests via ``conftest``)
or monkeypatch :data:`sorakai.infra.llm.factory.CHAT_MODEL_REGISTRY` to plug in
a recorder model.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from sorakai.common.config import get_settings
from sorakai.core.logging import get_logger
from sorakai.infra.llm import get_chat_model

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You answer using the knowledge base context provided in the user's message. "
    "Earlier messages in this chat are the same conversation - stay consistent with them. "
    "If the answer is not in the context, say you do not have that information."
)


def _build_messages(
    question: str,
    context: str,
    conversation: list[dict[str, str]] | None,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in conversation or ():
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    user_content = f"Knowledge base context:\n{context}\n\nQuestion:\n{question}"
    messages.append(HumanMessage(content=user_content))
    return messages


async def ask_llm(
    question: str,
    context: str,
    *,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """Ask the currently-configured chat model.

    The function is provider-agnostic: it depends on the
    :class:`~sorakai.infra.llm.base.BaseChatModel` Protocol and the factory.
    """
    settings = get_settings()
    model = get_chat_model(settings)
    messages = _build_messages(question, context, conversation)
    logger.info("LLM call: provider=%s messages=%d", settings.llm_provider, len(messages))
    response = await model.ainvoke(messages)
    return str(response.content).strip()
