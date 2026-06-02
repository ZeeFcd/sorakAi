"""Provider-agnostic adapters for external systems (LLMs, embeddings, vector stores).

Subpackages here define a small ``base`` (the abstract interface, usually a
LangChain ``BaseChatModel`` / ``Embeddings`` / a local ``Protocol``), one
adapter module per supported provider, and a ``factory`` module exposing a
registry keyed by the corresponding env-driven settings field.

Adding a new provider is intentionally local: one adapter file + one entry in
the registry dict + (if needed) one literal value in
:mod:`sorakai.common.config`. No chain, agent, or HTTP handler changes.
"""
