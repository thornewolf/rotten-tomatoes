from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rotten Tomatoes Predictor"
    GEMINI_API_KEY: str | None = None
    KALSHI_API_KEY: str | None = None
    KALSHI_PRIVATE_KEY_PATH: str | None = Field(
        default=None,
        validation_alias=AliasChoices("KALSHI_PRIVATE_KEY_PATH", "KALSHI_PEM_PATH"),
    )
    KALSHI_KEY_ID: str | None = None
    # Quickstart market data defaults to demo environment
    KALSHI_BASE_URL: str = "https://demo-api.kalshi.co/trade-api/v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
