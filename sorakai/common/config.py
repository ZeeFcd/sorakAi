from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "stub"]
"""Concrete chat-model backends registered in
``sorakai.infra.llm.factory.CHAT_MODEL_REGISTRY``. Adding a future provider
means adding a literal here plus an adapter file under ``sorakai/infra/llm/``."""

EmbeddingProvider = Literal["char", "ollama"]
"""Concrete embeddings backends registered in
``sorakai.infra.embeddings.factory.EMBEDDINGS_REGISTRY``."""

VectorStoreBackend = Literal["memory", "redis", "qdrant"]
"""Concrete vector-store backends registered in
``sorakai.infra.vector_store.factory.VECTOR_STORE_REGISTRY``.

- ``memory`` and ``redis`` wrap :mod:`sorakai.common.store` (Wave 4 layout).
- ``qdrant`` talks to a real Qdrant server (collection per env, cosine,
  payload-carries-metadata)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Project / runtime ------------------------------------------------
    project_name: str = "sorakAi"
    environment: str = Field(default="dev", description="dev|staging|prod")
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Python logging level for sorakai.* loggers (DEBUG|INFO|WARNING|ERROR).",
    )

    # --- HTTP / CORS ------------------------------------------------------
    request_timeout_seconds: float = 30.0
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        alias="CORS_ORIGINS",
        description=(
            "Allowed CORS origins. Browsers reject `*` together with credentials, so we never "
            "set allow_credentials=True when this contains `*`. Provide JSON when overriding via "
            "env, e.g. CORS_ORIGINS='[\"https://app.example.com\"]'."
        ),
    )

    # --- Storage ----------------------------------------------------------
    redis_url: str | None = Field(default=None, description="redis://host:6379/0 — if unset, in-memory store")

    # --- Vector store (Wave 5; pluggable via sorakai.infra.vector_store.factory) ---
    vector_store: VectorStoreBackend = Field(
        default="redis",
        alias="VECTOR_STORE",
        description=(
            "Selects the KB backend. 'memory'/'redis' wrap the Wave 4 KnowledgeStore; "
            "'qdrant' talks to a real Qdrant server. Tests default to 'memory' via conftest."
        ),
    )
    qdrant_url: str = Field(
        default="http://127.0.0.1:6333",
        alias="QDRANT_URL",
        description=(
            "Qdrant endpoint URL. The literal ':memory:' starts an in-process Qdrant "
            "(useful for tests and local-only smoke runs)."
        ),
    )
    qdrant_collection: str = Field(
        default="sorakai_kb",
        alias="QDRANT_COLLECTION",
        min_length=1,
        max_length=128,
        description="Qdrant collection name. Created lazily on first ingest with cosine distance.",
    )

    # --- Upstream services (gateway-only) ---------------------------------
    ingest_service_url: str = Field(default="http://127.0.0.1:8001", alias="INGEST_SERVICE_URL")
    rag_service_url: str = Field(default="http://127.0.0.1:8002", alias="RAG_SERVICE_URL")

    # --- Chat history (RAG) ----------------------------------------------
    chat_history_ttl_seconds: int = Field(default=604_800, alias="CHAT_HISTORY_TTL_SECONDS")
    chat_history_max_messages: int = Field(
        default=40,
        alias="CHAT_HISTORY_MAX_MESSAGES",
        ge=2,
        description="Cap stored turns per session (~half this many user/assistant pairs).",
    )

    # --- LLM provider (provider-agnostic via sorakai.infra.llm.factory) --
    llm_provider: LLMProvider = Field(
        default="ollama",
        alias="LLM_PROVIDER",
        description="Selects the chat-model adapter. Tests override to `stub` via conftest.",
    )

    # --- Embeddings provider --------------------------------------------
    embedding_provider: EmbeddingProvider = Field(
        default="ollama",
        alias="EMBEDDING_PROVIDER",
        description="Selects the embeddings adapter. Tests override to `char` via conftest.",
    )

    # --- Ollama (chat + embeddings share one endpoint) -------------------
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
        description="Ollama HTTP root (`/api/chat` and `/api/embed` live under here).",
    )
    ollama_chat_model: str = Field(default="llama3.2:1b", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    # --- Ollama embeddings tuning (Wave 2) -------------------------------
    ollama_embed_batch: int = Field(
        default=64,
        alias="OLLAMA_EMBED_BATCH",
        ge=1,
        le=2048,
        description="Max chunks per `/api/embed` request body before splitting into more batches.",
    )
    ollama_embed_concurrency: int = Field(
        default=4,
        alias="OLLAMA_EMBED_CONCURRENCY",
        ge=1,
        le=64,
        description="Max concurrent in-flight embed requests (bounded by an asyncio.Semaphore).",
    )
    ollama_embed_timeout_seconds: float = Field(
        default=60.0,
        alias="OLLAMA_EMBED_TIMEOUT_SECONDS",
        gt=0,
        description="Per-request HTTP timeout for embed calls (separate from `request_timeout_seconds`).",
    )
    ollama_embed_use_batch_endpoint: bool = Field(
        default=True,
        alias="OLLAMA_EMBED_USE_BATCH_ENDPOINT",
        description=(
            "When true (default), use the modern `/api/embed` endpoint that accepts a list of inputs. "
            "Set to false to force the legacy per-input `/api/embeddings` endpoint (for older Ollama). "
            "The adapter also falls back automatically on 404 from the batched endpoint."
        ),
    )

    # --- MLflow ----------------------------------------------------------
    mlflow_tracking_uri: str | None = Field(
        default=None,
        alias="MLFLOW_TRACKING_URI",
        description="When set, ingest/RAG log runs to this MLflow tracking server.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
