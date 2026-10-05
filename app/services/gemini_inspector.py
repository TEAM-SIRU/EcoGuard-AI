import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.cleaning import CleaningAssessment, GeminiCleaningAssessment


logger = logging.getLogger(__name__)
GEMINI_MAX_ATTEMPTS_PER_MODEL = 2
GEMINI_RETRY_BASE_SECONDS = 1.0


class InspectionSystemError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class GeminiApiKeyMissing(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__(
            "GEMINI_API_KEY_MISSING",
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.",
            503,
        )


class GeminiTimeout(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__("GEMINI_TIMEOUT", "Gemini 요청 시간이 초과되었습니다.", 504)


class GeminiRateLimited(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__("GEMINI_RATE_LIMITED", "Gemini API 요청 한도에 도달했습니다.", 503)


class GeminiUnavailable(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__(
            "GEMINI_UNAVAILABLE",
            "Gemini 서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            503,
        )


class GeminiApiFailure(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__("GEMINI_API_ERROR", "Gemini API 호출에 실패했습니다.", 502)


class StructuredOutputInvalid(InspectionSystemError):
    def __init__(self) -> None:
        super().__init__(
            "STRUCTURED_OUTPUT_INVALID",
            "Gemini 응답이 CleaningAssessment 스키마를 통과하지 못했습니다.",
            502,
        )


class GeminiCleaningInspector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sync_client: genai.Client | None = None
        self._async_client: Any | None = None
        self._prompt: str | None = None
        self._next_model_index = 0

    def _get_prompt(self) -> str:
        if self._prompt is None:
            try:
                self._prompt = Path(self.settings.prompt_path).read_text(encoding="utf-8")
            except OSError as exc:
                raise GeminiApiFailure() from exc
        return self._prompt

    def _get_async_client(self) -> Any:
        api_key = (
            self.settings.gemini_api_key.get_secret_value().strip()
            if self.settings.gemini_api_key is not None
            else ""
        )
        if not api_key:
            raise GeminiApiKeyMissing()
        if self._async_client is None:
            timeout_ms = int(self.settings.gemini_timeout_seconds * 1000)
            self._sync_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
            self._async_client = self._sync_client.aio
        return self._async_client

    def _log_api_failure(self, exc: Exception, status_code: int | None) -> None:
        api_key = (
            self.settings.gemini_api_key.get_secret_value()
            if self.settings.gemini_api_key is not None
            else ""
        )
        error_message = str(exc)
        if api_key:
            error_message = error_message.replace(api_key, "[REDACTED]")
        logger.error(
            "Gemini API 요청 실패(type=%s, status=%s): %s",
            type(exc).__name__,
            status_code,
            error_message,
        )

    async def inspect(self, image_bytes: bytes, mime_type: str) -> CleaningAssessment:
        client = self._get_async_client()
        configured_models = self.settings.gemini_model_chain
        if not configured_models:
            raise GeminiApiFailure()
        start_index = self._next_model_index % len(configured_models)
        self._next_model_index = (start_index + 1) % len(configured_models)
        models = configured_models[start_index:] + configured_models[:start_index]
        for model_index, model in enumerate(models):
            for attempt in range(GEMINI_MAX_ATTEMPTS_PER_MODEL):
                try:
                    response = await client.models.generate_content(
                        model=model,
                        contents=[
                            types.Part.from_text(text=self._get_prompt()),
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GeminiCleaningAssessment,
                            temperature=0,
                            max_output_tokens=1024,
                        ),
                    )
                except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
                    raise GeminiTimeout() from exc
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                    try:
                        status_code = int(status_code)
                    except (TypeError, ValueError):
                        status_code = None

                    has_fallback = model_index < len(models) - 1
                    if status_code == 429 and has_fallback:
                        logger.warning(
                            "Gemini 모델 %s 요청 한도 초과(status=429). 다음 모델 %s로 전환합니다.",
                            model,
                            models[model_index + 1],
                        )
                        break
                    if status_code == 429:
                        self._log_api_failure(exc, status_code)
                        raise GeminiRateLimited() from exc

                    if status_code == 503 and attempt < GEMINI_MAX_ATTEMPTS_PER_MODEL - 1:
                        delay_seconds = GEMINI_RETRY_BASE_SECONDS * (2**attempt)
                        logger.warning(
                            "Gemini 모델 %s 일시 오류(status=503). %.1f초 후 재시도합니다.",
                            model,
                            delay_seconds,
                        )
                        await asyncio.sleep(delay_seconds)
                        continue
                    if status_code == 503 and has_fallback:
                        logger.warning(
                            "Gemini 모델 %s를 사용할 수 없습니다(status=503). 다음 모델 %s로 전환합니다.",
                            model,
                            models[model_index + 1],
                        )
                        break
                    if status_code == 503:
                        self._log_api_failure(exc, status_code)
                        raise GeminiUnavailable() from exc

                    self._log_api_failure(exc, status_code)
                    raise GeminiApiFailure() from exc
                else:
                    logger.info("Gemini 판정 완료(model=%s)", model)
                    return self.parse_response(response)

        raise GeminiApiFailure()

    @staticmethod
    def parse_response(response: Any) -> CleaningAssessment:
        try:
            response_text = getattr(response, "text", None)
            if response_text:
                return CleaningAssessment.model_validate_json(response_text)
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                if hasattr(parsed, "model_dump"):
                    parsed = parsed.model_dump()
                return CleaningAssessment.model_validate(parsed)
            raise ValueError("Gemini response contained no structured content")
        except (ValidationError, ValueError, TypeError) as exc:
            raise StructuredOutputInvalid() from exc

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
        if self._sync_client is not None:
            self._sync_client.close()
