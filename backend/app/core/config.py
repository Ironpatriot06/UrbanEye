"""
Application configuration.

All settings are read from environment variables (or a .env file).
New settings should be added here rather than scattered across the codebase.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object.

    Pydantic-settings reads values from the environment first, then from a
    .env file in the project root.  The lru_cache wrapper (see get_settings)
    ensures we only parse the environment once.
    """

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = (
        "postgresql+psycopg2://urbaneye:urbaneye_dev@localhost:5432/urbaneye"
    )

    # ------------------------------------------------------------------
    # API / Security
    # ------------------------------------------------------------------
    SECRET_KEY: str = "change-me-in-production"

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    APP_NAME: str = "UrbanEye+"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
