# Provider registry (Wave 11)

sorakAi keeps every external dependency (LLMs, embeddings, vector
stores, chat history) behind a thin factory + registry pair so a new
provider is **one new file + one entry in a dict**, never a sweep
through the codebase. This document is the canonical map of what's
registered today and what an author has to implement to add the next
one.

The three pillars all follow the same shape:

| Pillar       | Protocol module                                         | Factory module                                           | Env switch                       |
| ------------ | ------------------------------------------------------- | -------------------------------------------------------- | -------------------------------- |
| Chat LLM     | [`sorakai/infra/llm/base.py`](../sorakai/infra/llm/base.py)             | [`sorakai/infra/llm/factory.py`](../sorakai/infra/llm/factory.py)             | `LLM_PROVIDER`                   |
| Embeddings   | [`sorakai/infra/embeddings/base.py`](../sorakai/infra/embeddings/base.py) | [`sorakai/infra/embeddings/factory.py`](../sorakai/infra/embeddings/factory.py) | `EMBEDDING_PROVIDER`             |
| Vector store | [`sorakai/infra/vector_store/base.py`](../sorakai/infra/vector_store/base.py) | [`sorakai/infra/vector_store/factory.py`](../sorakai/infra/vector_store/factory.py) | `VECTOR_STORE`                   |
| Chat history | [`sorakai/common/chat_history.py`](../sorakai/common/chat_history.py)       | `create_chat_store` in the same file                     | implicit (`REDIS_URL` set or not) |

Every factory exposes a `register_<thing>(name, builder)` hook so an
out-of-tree package can self-register at import time without modifying
the registry dict directly. That is the OCP boundary: the factory
**never** changes when you add a provider.

---

## Registered providers (as of Wave 11)

### Chat LLM (`LLM_PROVIDER`)

| Name       | Builder                                                     | Notes                                                                                          |
| ---------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `ollama`   | [`build_ollama_chat`](../sorakai/infra/llm/ollama.py)       | Default. Talks to a local Ollama at `OLLAMA_BASE_URL` with `OLLAMA_CHAT_MODEL`.                |
| `stub`     | [`build_stub_chat`](../sorakai/infra/llm/stub.py)           | Deterministic test double; used by the eval CI smoke job and unit tests.                       |

### Embeddings (`EMBEDDING_PROVIDER`)

| Name       | Builder                                                                | Notes                                                                                          |
| ---------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `ollama`   | [`build_ollama_embeddings`](../sorakai/infra/embeddings/ollama.py)     | Default. Uses `OLLAMA_EMBEDDING_MODEL` (e.g. `nomic-embed-text`).                              |
| `char`     | [`build_char_embeddings`](../sorakai/infra/embeddings/char.py)         | Deterministic, network-free character-level embeddings. Used in tests + the in-memory KB demos. |

### Vector stores (`VECTOR_STORE`)

| Name       | Builder                                                                 | Notes                                                                                            |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `qdrant`   | `_build_qdrant` in [factory.py](../sorakai/infra/vector_store/factory.py) | Default in compose. `QDRANT_URL` + `QDRANT_COLLECTION`. Survives restarts via a docker volume.   |
| `redis`    | `_build_redis` in [factory.py](../sorakai/infra/vector_store/factory.py)  | Wraps `RedisKnowledgeStore`. Requires `REDIS_URL`.                                               |
| `memory`   | `_build_memory` in [factory.py](../sorakai/infra/vector_store/factory.py) | In-process `InMemoryKnowledgeStore`. Handy for tests / a quick local repl, not persistent.       |

### Chat history

Selected implicitly by the presence of `REDIS_URL`:

- `RedisChatHistoryStore` when `REDIS_URL` is set.
- `InMemoryChatHistoryStore` otherwise.

The `SorakaiChatMessageHistory` adapter bridges either backend to
LangChain's `BaseChatMessageHistory` so the RAG chain doesn't care
which one is wired.

---

## Adding a new provider

The same recipe works for all three pillars. The example uses an LLM
because it has the most moving parts; embeddings and vector stores are
identical except for the protocol they implement.

### 1. Write the adapter

```python
# sorakai/infra/llm/my_provider.py
from __future__ import annotations

from sorakai.common.config import Settings
from sorakai.infra.llm.base import BaseChatModel


def build_my_provider_chat(settings: Settings) -> BaseChatModel:
    """Return a BaseChatModel that talks to MyProvider.

    Read everything you need from ``settings`` (extend ``Settings`` first if
    you need a new env var). NEVER reach into ``os.environ`` here -
    settings is the single source of truth.
    """
    return _MyProviderChat(
        base_url=settings.my_provider_base_url,
        model=settings.my_provider_model,
        temperature=settings.llm_temperature,
    )
```

Implement the protocol's small surface (`ainvoke`, `astream`, etc.) on
`_MyProviderChat`; the protocol is intentionally tiny so adapters can
delegate to whatever client library makes sense (LangChain integrations,
raw HTTP, an SDK, ...).

### 2. Register the builder

Either add one line to the in-tree registry:

```python
# sorakai/infra/llm/factory.py
from sorakai.infra.llm.my_provider import build_my_provider_chat

CHAT_MODEL_REGISTRY: dict[str, ChatModelBuilder] = {
    "ollama": build_ollama_chat,
    "stub": build_stub_chat,
    "my_provider": build_my_provider_chat,  # <-- new
}
```

…or, from an out-of-tree package, register at import time:

```python
from sorakai.infra.llm.factory import register_chat_model
from my_pkg.adapters import build_my_provider_chat

register_chat_model("my_provider", build_my_provider_chat)
```

### 3. Extend `Settings` if you need new env vars

```python
# sorakai/common/config.py
my_provider_base_url: str = Field(default="http://localhost:9999", alias="MY_PROVIDER_BASE_URL")
my_provider_model: str = Field(default="my-llm-7b", alias="MY_PROVIDER_MODEL")
```

`populate_by_name=True` is already set on the `Settings` model, so both
`my_provider_base_url=...` (in tests) and `MY_PROVIDER_BASE_URL=...`
(in real env / `.env`) work.

### 4. Add at least one test

A protocol-level test that drives the adapter through a mock transport
is enough; the rest of the stack only consumes `BaseChatModel`, so an
adapter that passes the protocol contract Just Works in the chain and
the agent.

### 5. Document it

Append a row to the appropriate table in this file. That keeps the
"what's available?" answer in one searchable place.

---

## SOLID alignment

- **OCP**: factories never change when a provider is added. The
  registry is open for extension via the `register_*` hooks.
- **DIP**: every call site (chain, agent, ingest) depends on the
  protocol modules under `sorakai/infra/*/base.py`, never on a concrete
  adapter class.
- **LSP**: protocol contracts are written so any adapter that satisfies
  the signatures is substitutable for any other; the mypy `--strict`
  gate catches drift.
- **SRP**: each adapter file owns exactly one provider; factories do
  registry lookup only.
- **ISP**: protocols are kept narrow so an adapter doesn't have to
  implement methods the chain/agent never use.
