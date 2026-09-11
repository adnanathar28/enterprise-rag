from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="brd-knowledge", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    database_url: str = Field(
        default="postgresql+psycopg://brd_knowledge:brd_knowledge@localhost:5433/brd_knowledge",
        alias="DATABASE_URL",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_document_extensions: set[str] = Field(
        default={".pdf"},
        alias="ALLOWED_DOCUMENT_EXTENSIONS",
    )
    max_upload_size_bytes: int = Field(default=25 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES")
    source_storage_dir: Path = Field(default=Path("data/source_files"), alias="SOURCE_STORAGE_DIR")
    embedding_model_name: str = Field(
        default="Alibaba-NLP/gte-modernbert-base",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_model_revision: str = Field(
        default="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        alias="EMBEDDING_MODEL_REVISION",
    )
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION", gt=0)
    embedding_max_sequence_length: int = Field(
        default=8192,
        alias="EMBEDDING_MAX_SEQUENCE_LENGTH",
        gt=0,
    )
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE", gt=0)
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_preprocessing_version: str = Field(
        default="1",
        alias="EMBEDDING_PREPROCESSING_VERSION",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        alias="RERANKER_MODEL_NAME",
    )
    reranker_model_revision: str = Field(
        default="233902d25c440f23af6f7d6e94d2946bac0bee0a",
        alias="RERANKER_MODEL_REVISION",
    )
    reranker_max_sequence_length: int = Field(
        default=512,
        alias="RERANKER_MAX_SEQUENCE_LENGTH",
        gt=0,
    )
    reranker_batch_size: int = Field(default=16, alias="RERANKER_BATCH_SIZE", gt=0)
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_candidate_k: int = Field(default=40, alias="RERANKER_CANDIDATE_K", gt=0)

    llm_provider: Literal["gemini", "local_qwen"] = Field(default="gemini", alias="LLM_PROVIDER")
    local_qwen_model: str = Field(default="qwen3:8b", alias="LOCAL_QWEN_MODEL")
    local_qwen_base_url: str = Field(default="http://127.0.0.1:11434", alias="LOCAL_QWEN_BASE_URL")
    local_qwen_context_tokens: int = Field(default=8192, alias="LOCAL_QWEN_CONTEXT_TOKENS")
    local_qwen_timeout_seconds: float = Field(default=180, alias="LOCAL_QWEN_TIMEOUT_SECONDS")

    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=1.0, ge=0, le=2, alias="GEMINI_TEMPERATURE")
    gemini_timeout_seconds: float = Field(default=60.0, gt=0, alias="GEMINI_TIMEOUT_SECONDS")
    gemini_safety_margin_tokens: int = Field(default=128, ge=0, alias="GEMINI_SAFETY_MARGIN_TOKENS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
