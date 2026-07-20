from functools import lru_cache

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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
