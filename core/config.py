from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rotten Tomatoes Predictor"
    GEMINI_API_KEY: str | None = None
    KALSHI_API_KEY: str | None = None
    KALSHI_PRIVATE_KEY_PATH: str | None = None
    KALSHI_KEY_ID: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
