from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="INSURANCE_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment name.",
    )
    log_level: str = Field(default="INFO", description="Python logging level name.")
    api_host: str = Field(default="0.0.0.0", description="Bind host for the ASGI server.")
    api_port: int = Field(default=8000, description="Bind port for the ASGI server.")
    service_name: str = Field(
        default="insurance-ai-api",
        description="Logical service name for logs and tracing.",
    )
    app_version: str = Field(default="0.1.0", description="Semantic application version string.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached for stable dependency injection)."""

    return Settings()
