"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API and RAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AI PDF Assistant", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Comma-separated origins for local Streamlit (Phase 4) and browsers.
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        alias="CORS_ORIGINS",
    )

    # LLM provider switch (env only). Default ollama for local free generation.
    # Never falls back between providers at runtime.
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        alias="LLM_PROVIDER",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_timeout_seconds: float = Field(default=30.0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=2, alias="OPENAI_MAX_RETRIES")

    # Local Ollama (default provider). Must remain on localhost.
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="qwen3:0.6b", alias="OLLAMA_MODEL")
    ollama_timeout_seconds: float = Field(
        default=60.0,
        alias="OLLAMA_TIMEOUT_SECONDS",
    )
    ollama_max_retries: int = Field(default=2, alias="OLLAMA_MAX_RETRIES")
    ollama_num_predict: int = Field(
        default=512,
        alias="OLLAMA_NUM_PREDICT",
        description="Max tokens for Ollama generation (num_predict).",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    # Cosine similarity via FAISS IndexFlatIP on L2-normalized vectors.
    similarity_threshold: float = Field(default=0.25, alias="SIMILARITY_THRESHOLD")
    # Approximate token budget for retrieved context (~4 chars/token).
    context_token_budget: int = Field(default=2500, alias="CONTEXT_TOKEN_BUDGET")

    data_dir: str = Field(default="data", alias="DATA_DIR")
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        alias="MAX_UPLOAD_BYTES",
        description="Maximum accepted PDF upload size in bytes (default 10 MiB).",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
