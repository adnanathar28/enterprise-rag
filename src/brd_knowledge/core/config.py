from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="brd-knowledge", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    database_url: str = Field(
        default="postgresql+psycopg://brd_knowledge:brd_knowledge@localhost:5432/brd_knowledge",
        alias="DATABASE_URL",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_document_extensions: set[str] = Field(
        default={".pdf"},
        alias="ALLOWED_DOCUMENT_EXTENSIONS",
    )
    max_upload_size_bytes: int = Field(default=25 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES")
    source_storage_dir: Path = Field(default=Path("data/source_files"), alias="SOURCE_STORAGE_DIR")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
