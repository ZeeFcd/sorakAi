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


# ---------------------------------------------------------------------------
# Wave 7 agent prompts
# ---------------------------------------------------------------------------

AGENT_ROUTE_SYSTEM = (
    "Decide whether the user's question requires looking something up in the "
    "knowledge base or can be answered directly as small talk. "
    "Respond with exactly one lowercase word: 'kb' or 'chitchat'."
)

AGENT_GRADE_SYSTEM = (
    "You decide whether retrieved context is useful for answering the question. "
    "Respond with exactly one lowercase word: 'good' if the context covers the "
    "question, 'weak' if it doesn't."
)
AGENT_GRADE_USER = "Question:\n{question}\n\nRetrieved context:\n{context}"

AGENT_REWRITE_SYSTEM = (
    "Rewrite the user's question to be a better search query. Keep proper "
    "nouns. Output only the rewritten query, no quotes, no preamble."
)
AGENT_REWRITE_USER = "Original question:\n{question}\n\nPrevious query:\n{query}"

AGENT_CHITCHAT_SYSTEM = (
    "You answer the user briefly and clearly. Earlier messages are the same "
    "conversation - stay consistent. If the user asks something you genuinely "
    "don't know, say so."
)

AGENT_CRITIQUE_SYSTEM = (
    "You assess whether an answer addresses the question. Respond with exactly "
    "one lowercase word: 'ok' if the answer is on-topic and grounded in the "
    "context, 'retry' if it is off-topic, hallucinated, or empty."
)
AGENT_CRITIQUE_USER = "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"


def build_agent_route_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", AGENT_ROUTE_SYSTEM), ("human", "Question:\n{question}")])


def build_agent_grade_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", AGENT_GRADE_SYSTEM), ("human", AGENT_GRADE_USER)])


def build_agent_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", AGENT_REWRITE_SYSTEM), ("human", AGENT_REWRITE_USER)])


def build_agent_chitchat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_CHITCHAT_SYSTEM),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )


def build_agent_critique_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", AGENT_CRITIQUE_SYSTEM), ("human", AGENT_CRITIQUE_USER)])
