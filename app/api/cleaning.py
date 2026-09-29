from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.errors import EcoGuardError, input_error
from app.services.inference import CleaningInference
from app.utils.image import validate_image

router = APIRouter()


@router.post("/api/v1/cleaning/evaluate")
async def evaluate_cleaning(
    request: Request,
    image: UploadFile = File(...),
    zone_id: str = Form(...),
    checkpoint_id: str = Form(...),
    user_id: str | None = Form(default=None),
):
    config = request.app.state.config
    contents = await image.read(config.max_upload_bytes + 1)
    validated = validate_image(contents, config)
    semaphore: asyncio.Semaphore = request.app.state.inference_semaphore
    async with semaphore:
        service: CleaningInference = request.app.state.inference
        return await run_in_threadpool(service.evaluate, validated.image, zone_id, checkpoint_id, user_id)
