"""
Application configuration.

All settings are read from environment variables (or a .env file).
New settings should be added here rather than scattered across the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/ -> repository root
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent

#: .env files, resolved to absolute paths so the same configuration loads
#: whether the backend is started from the repository root or from backend/
#: (as the README instructs).  A bare ".env" would be resolved against the
#: current working directory and silently fall back to the insecure defaults
#: below.  Later files take precedence; missing files are skipped.
ENV_FILES = (_REPO_ROOT / ".env", _BACKEND_DIR / ".env")


class Settings(BaseSettings):
    """
    Central settings object.

    Pydantic-settings reads values from the environment first, then from the
    files in ENV_FILES (the repository-root .env, optionally overridden by
    backend/.env).  The lru_cache wrapper (see get_settings) ensures we only
    parse the environment once.
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
    # Google Sign-In (OAuth 2.0 / OpenID Connect)
    # ------------------------------------------------------------------
    # Obtained from the Google Cloud Console — see docs/authentication.md.
    # Both default to "" so the rest of the application starts and every other
    # feature works without them; the Google endpoints then return HTTP 503
    # instead of failing at import time.
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    # Must match a redirect URI registered on the OAuth client EXACTLY.
    # This points at the BACKEND callback, not the frontend: the code is
    # exchanged server-side so the client secret never reaches the browser.
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    # Where the backend sends the browser once Google sign-in has completed.
    # Also the origin used to validate any post-login redirect target, so a
    # crafted `next` parameter cannot bounce a fresh token off-site.
    FRONTEND_URL: str = "http://localhost:3000"

    #: Lifetime of the OAuth `state` token.  Long enough for a human to get
    #: through the Google consent screen, short enough that a leaked state is
    #: useless later.
    OAUTH_STATE_EXPIRE_SECONDS: int = 600

    #: Send the OAuth state cookie with the Secure flag.  Must be True in any
    #: deployment served over HTTPS; False locally because http://localhost
    #: would otherwise drop the cookie.
    COOKIE_SECURE: bool = False

    # ------------------------------------------------------------------
    # Demo account seeds (read by scripts/create_demo_data.py)
    # ------------------------------------------------------------------
    DEMO_ADMIN_EMAIL: str = "admin@urbaneye.local"
    DEMO_ADMIN_PASSWORD: str = "Admin@1234"
    DEMO_ADMIN_NAME: str = "System Admin"

    DEMO_AGENT_EMAIL: str = "agent@urbaneye.local"
    DEMO_AGENT_PASSWORD: str = "Agent@1234"
    DEMO_AGENT_NAME: str = "Field Agent"

    DEMO_AGENT_EMAIL_2: str = "agent2@urbaneye.local"
    DEMO_AGENT_PASSWORD_2: str = "Agent2@1234"
    DEMO_AGENT_NAME_2: str = "Field Agent 2"

    DEMO_AGENT_EMAIL_3: str = "agent3@urbaneye.local"
    DEMO_AGENT_PASSWORD_3: str = "Agent3@1234"
    DEMO_AGENT_NAME_3: str = "Field Agent 3"

    DEMO_AGENT_EMAIL_4: str = "agent4@urbaneye.local"
    DEMO_AGENT_PASSWORD_4: str = "Agent4@1234"
    DEMO_AGENT_NAME_4: str = "Field Agent 4"

    DEMO_USER_EMAIL: str = "user@urbaneye.local"
    DEMO_USER_PASSWORD: str = "User@1234"
    DEMO_USER_NAME: str = "Demo Citizen"

    # ------------------------------------------------------------------
    # ML inference (Phase 2C) — see app/ml/__init__.py
    # ------------------------------------------------------------------
    #: Master switch.  False -> every prediction is UNAVAILABLE (ML_DISABLED)
    #: and no model file is opened; incident workflows are unaffected.
    ML_ENABLED: bool = True
    #: Directory of the serving bundle exported by
    #: UrbanEye_ML_Data_Pipeline/scripts/phase2c/export_serving_bundle.py.
    ML_BUNDLE_DIR: str = str(_BACKEND_DIR / "ml_bundle" / "urbaneye-ml-2026.10.13")
    #: SHA-256 of the bundle's MANIFEST.json.  The manifest in turn lists the
    #: SHA-256 of every file, so a bundle whose files or manifest differ from
    #: the reviewed one is refused before any pickle is opened.
    ML_BUNDLE_MANIFEST_SHA256: str = (
        "b89f0e3266c9e677bffdd22134355087dfeaf8cd262cc8281bb10f3926187c7e"
    )
    #: JSON list of service areas ({"name", "lat_min", "lat_max", "lon_min",
    #: "lon_max"}).  Empty -> the built-in Bengaluru and Chennai boxes.
    ML_SERVICE_AREAS: str = ""
    #: Log predictions automatically after an incident is created or its SLA /
    #: priority changes (best effort; never blocks the write).  Predictions are
    #: also logged when viewed, and by scripts/ml_log_predictions.py.
    ML_LOG_ON_WRITE: bool = True

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    APP_NAME: str = "UrbanEye+"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
