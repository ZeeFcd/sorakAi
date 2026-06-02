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
