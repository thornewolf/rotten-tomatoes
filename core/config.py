import logging
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rotten Tomatoes Predictor"
    GEMINI_API_KEY: Optional[str] = None
    KALSHI_API_KEY: Optional[str] = None
    KALSHI_PRIVATE_KEY_PATH: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("KALSHI_PRIVATE_KEY_PATH", "KALSHI_PEM_PATH"),
        description="Path to RSA private key for Kalshi API authentication"
    )
    KALSHI_KEY_ID: Optional[str] = None
    # Quickstart market data defaults to demo environment
    KALSHI_BASE_URL: str = "https://demo-api.kalshi.co/trade-api/v2"

    @field_validator("KALSHI_PRIVATE_KEY_PATH")
    @classmethod
    def validate_private_key_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate that the private key file exists if a path is provided.
        Logs a warning but doesn't fail startup if the file is missing.
        """
        if v is None:
            return v

        key_path = Path(v)
        if not key_path.exists():
            logger.warning(
                "KALSHI_PRIVATE_KEY_PATH is set to '%s' but the file does not exist. "
                "Kalshi API requests will not be signed.",
                v
            )
        elif not key_path.is_file():
            logger.warning(
                "KALSHI_PRIVATE_KEY_PATH '%s' is not a file. "
                "Kalshi API requests will not be signed.",
                v
            )
        else:
            logger.info("Found Kalshi private key at %s", key_path.resolve())

        return v

    @field_validator("KALSHI_BASE_URL")
    @classmethod
    def validate_kalshi_url(cls, v: str) -> str:
        """Ensure the Kalshi URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"KALSHI_BASE_URL must start with http:// or https://, got: {v}")
        return v.rstrip("/")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
