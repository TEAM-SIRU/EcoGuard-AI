from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.ai.dustpan_yolo import DustpanDetector
from app.ai.zone_recognition import load_encoder
from app.core.config import settings
from app.schemas.cleaning import ErrorResponse
from app.services.errors import EcoGuardError
from app.services.inference import CleaningInference
from app.services.zone_registry import ZoneRegistry
from app.api.cleaning import router as cleaning_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = settings
    registry = ZoneRegistry.load(config.zone_registry_path, config)
    detector = None
    encoder = None
    if config.dustpan_weight_path.is_file():
        try:
            detector = DustpanDetector(config.dustpan_weight_path)
        except EcoGuardError:
            detector = None
    if config.zone_encoder_weight_path.is_file():
        try:
            encoder = load_encoder(config)
        except Exception:
            encoder = None
    app.state.config = config
    app.state.registry = registry
    app.state.inference = CleaningInference(config, registry, detector, encoder)
    app.state.inference_semaphore = asyncio.Semaphore(config.max_inference_concurrency)
    yield


app = FastAPI(title="EcoGuard AI", version="0.1.0", lifespan=lifespan)
app.include_router(cleaning_router)


@app.exception_handler(EcoGuardError)
async def ecoguard_error_handler(request: Request, exc: EcoGuardError):
    body = ErrorResponse(error={"code": exc.code, "message": exc.message, "detail": exc.detail})
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    body = ErrorResponse(error={"code": "INFERENCE_ERROR", "message": "Unexpected inference error", "detail": {}})
    return JSONResponse(status_code=500, content=body.model_dump())


@app.get("/health")
def health():
    ready, missing = CleaningInference.global_readiness(settings)
    inference = getattr(app.state, "inference", None)
    runtime_loaded = inference is not None and inference.detector is not None and inference.encoder is not None
    if not runtime_loaded:
        missing.append("runtime model objects are not loaded")
    return {"status": "ok", "models_ready": ready and runtime_loaded, "missing_assets": missing}
