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
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_models: str = "gemini-3.1-flash-lite"
    gemini_timeout_seconds: float = Field(default=45.0, gt=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    prompt_path: Path = PROJECT_ROOT / "prompts" / "cleaning_inspection.txt"

    @property
    def gemini_model_chain(self) -> tuple[str, ...]:
        configured_models = [self.gemini_model, *self.gemini_fallback_models.split(",")]
        return tuple(dict.fromkeys(model.strip() for model in configured_models if model.strip()))


@lru_cache
def get_settings() -> Settings:
    return Settings()
