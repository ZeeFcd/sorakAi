"""Prompt templates used by Wave 6 chains.

Kept in their own module so:

- Tests can import the exact :class:`SystemMessage` text without depending
  on the chain wiring (the same shape Wave 1's ``ask_llm`` tests asserted on).
- A future per-locale / per-tenant prompt override is a single registry
  swap, not a rewrite of the chain.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = (
    "You answer using the knowledge base context provided in the user's message. "
    "Earlier messages in this chat are the same conversation - stay consistent with them. "
    "If the answer is not in the context, say you do not have that information."
)

# Single source of truth for the per-turn user message. Keeps the chain and
# any debugging tools rendering the exact same string.
RAG_USER_TEMPLATE = "Knowledge base context:\n{context}\n\nQuestion:\n{question}"


def build_rag_prompt() -> ChatPromptTemplate:
    """The RAG prompt: system + injected history placeholder + user turn."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", RAG_USER_TEMPLATE),
        ]
    )
