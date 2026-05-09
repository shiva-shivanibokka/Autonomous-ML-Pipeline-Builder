"""
core.config — centralised environment variable loading and validation.

Replaces the copy-pasted `from dotenv import load_dotenv; load_dotenv()` that
appears at the top of 15+ files across this portfolio.

Usage:
    from core.config import settings
    api_key = settings.anthropic_api_key
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ── LLM provider keys ─────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # ── E2B sandbox ───────────────────────────────────────────────────────────
    e2b_api_key: str = Field(default="", alias="E2B_API_KEY")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="autonomous-ml-pipeline", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # ── Service ports ─────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    gradio_port: int = Field(default=7860, alias="GRADIO_PORT")

    # ── Pipeline defaults ─────────────────────────────────────────────────────
    default_provider: str = Field(default="anthropic", alias="DEFAULT_PROVIDER")
    default_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="DEFAULT_MODEL"
    )
    max_correction_retries: int = Field(default=3, alias="MAX_CORRECTION_RETRIES")
    sandbox_timeout_seconds: int = Field(default=120, alias="SANDBOX_TIMEOUT_SECONDS")

    # ── HF Spaces compatibility ────────────────────────────────────────────────
    # On HF Spaces, E2B is available; set to "subprocess" for local fallback
    execution_backend: str = Field(default="e2b", alias="EXECUTION_BACKEND")

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"anthropic", "openai", "groq"}
        if v not in allowed:
            raise ValueError(f"default_provider must be one of {allowed}")
        return v

    model_config = {"populate_by_name": True, "extra": "ignore"}

    def has_provider_key(self, provider: str) -> bool:
        """Check whether a valid API key is configured for the given provider."""
        mapping = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
        }
        return bool(mapping.get(provider, "").strip())

    def get_api_key(self, provider: str) -> str:
        """Return the API key for a given provider, or empty string."""
        mapping = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
        }
        return mapping.get(provider, "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — only reads env once per process."""
    return Settings()


# Module-level alias for convenience: `from core.config import settings`
settings = get_settings()
