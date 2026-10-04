from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

    app = FastAPI(title="EcoGuard AI V1", version="1.0.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.inspector = inspector or GeminiCleaningInspector(app_settings)
    app.include_router(cleaning_router)

    @app.get("/health", tags=["health"])
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
