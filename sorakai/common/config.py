from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "sorakAi"
    environment: str = Field(default="dev", description="dev|staging|prod")
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Python logging level for sorakai.* loggers (DEBUG|INFO|WARNING|ERROR).",
    )

    redis_url: str | None = Field(default=None, description="redis://host:6379/0 — if unset, in-memory store")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")

    ingest_service_url: str = Field(default="http://127.0.0.1:8001", alias="INGEST_SERVICE_URL")
    rag_service_url: str = Field(default="http://127.0.0.1:8002", alias="RAG_SERVICE_URL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    # Self-hosted OpenAI-compatible API (e.g. Ollama: http://ollama:11434/v1)
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")

    request_timeout_seconds: float = 30.0

    # Chat history in Redis (0 = no expiry)
    chat_history_ttl_seconds: int = Field(default=604_800, alias="CHAT_HISTORY_TTL_SECONDS")

    # Embeddings: char | openai | ollama (see sorakai/common/embedding.py)
    embedding_provider: str = Field(default="char", alias="EMBEDDING_PROVIDER")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_embeddings_base_url: str | None = Field(default=None, alias="OPENAI_EMBEDDINGS_BASE_URL")
    ollama_embed_base_url: str | None = Field(
        default=None,
        alias="OLLAMA_EMBED_BASE_URL",
        description="Ollama root URL for /api/embeddings (not .../v1)",
    )
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
