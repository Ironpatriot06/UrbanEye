"""
Application configuration.

All settings are read from environment variables (or a .env file).
New settings should be added here rather than scattered across the codebase.
"""

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
    # JWT / Security
    # ------------------------------------------------------------------
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ------------------------------------------------------------------
    # Demo account seeds (read by scripts/create_demo_data.py)
    # ------------------------------------------------------------------
    DEMO_ADMIN_EMAIL: str = "admin@urbaneye.local"
    DEMO_ADMIN_PASSWORD: str = "Admin@1234"
    DEMO_ADMIN_NAME: str = "System Admin"

    DEMO_AGENT_EMAIL: str = "agent@urbaneye.local"
    DEMO_AGENT_PASSWORD: str = "Agent@1234"
    DEMO_AGENT_NAME: str = "Field Agent"

    DEMO_USER_EMAIL: str = "user@urbaneye.local"
    DEMO_USER_PASSWORD: str = "User@1234"
    DEMO_USER_NAME: str = "Demo Citizen"

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    APP_NAME: str = "UrbanEye+"
    APP_VERSION: str = "0.2.0"
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
