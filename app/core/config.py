from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = Field(default=45.0, gt=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    prompt_path: Path = PROJECT_ROOT / "prompts" / "cleaning_inspection.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
