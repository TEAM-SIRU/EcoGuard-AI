from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api.cleaning import router as cleaning_router
from app.core.config import Settings, get_settings
from app.schemas.cleaning import ErrorResponse
from app.services.gemini_inspector import GeminiCleaningInspector, InspectionSystemError
from app.services.image_validation import ImageValidationError


def create_app(
    *, settings: Settings | None = None, inspector: GeminiCleaningInspector | None = None
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        close = getattr(app.state.inspector, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(
        title="EcoGuard AI 청소 판정 API",
        description=(
            "청소구역 사진 한 장을 검사해 PASS, FAIL 또는 REVIEW를 반환합니다. "
            "사진을 명확히 판단하기 어려우면 사람의 재확인이 필요한 REVIEW로 처리합니다."
        ),
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_tags=[
            {
                "name": "청소 판정",
                "description": "청소구역 사진과 요청 메타데이터를 보내 판정을 받습니다.",
            },
            {
                "name": "서버 상태",
                "description": "서버가 요청을 받을 수 있는지 확인합니다.",
            },
        ],
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.inspector = inspector or GeminiCleaningInspector(app_settings)
    app.include_router(cleaning_router)

    @app.get("/docs", include_in_schema=False)
    @app.get("/redoc", include_in_schema=False)
    async def korean_api_docs() -> FileResponse:
        docs_file = Path(__file__).with_name("docs.html")
        return FileResponse(docs_file, media_type="text/html; charset=utf-8")

    @app.get(
        "/health",
        tags=["서버 상태"],
        summary="서버 상태 확인",
        description="Gemini API를 호출하지 않고 서버의 응답 여부를 확인합니다.",
        response_description="서버 상태",
    )
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(ImageValidationError)
    async def image_validation_error(
        request: Request, exc: ImageValidationError
    ) -> JSONResponse:
        body = ErrorResponse(error_code=exc.error_code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(InspectionSystemError)
    async def inspection_system_error(
        request: Request, exc: InspectionSystemError
    ) -> JSONResponse:
        body = ErrorResponse(error_code=exc.error_code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    return app


app = create_app()
