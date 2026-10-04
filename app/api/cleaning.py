from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import StringConstraints, ValidationError

from app.core.config import Settings
from app.schemas.cleaning import CleaningAssessment, CleaningEvaluationResponse
from app.services.gemini_inspector import (
    GeminiCleaningInspector,
    InspectionSystemError,
    StructuredOutputInvalid,
)
from app.services.image_validation import ImageValidationError, validate_image_bytes


router = APIRouter(prefix="/api/v1/cleaning", tags=["cleaning"])
MetadataId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


def get_inspector(request: Request) -> GeminiCleaningInspector:
    return request.app.state.inspector


@router.post("/evaluate", response_model=CleaningEvaluationResponse)
async def evaluate_cleaning(
    request: Request,
    image: Annotated[UploadFile, File(description="청소구역 사진 한 장")],
    zone_id: Annotated[MetadataId, Form()],
    checkpoint_id: Annotated[MetadataId, Form()],
    user_id: Annotated[str | None, Form(max_length=128)] = None,
    inspector: GeminiCleaningInspector = Depends(get_inspector),
) -> CleaningEvaluationResponse:
    settings: Settings = request.app.state.settings
    image_bytes = await image.read(settings.max_upload_bytes + 1)
    mime_type = validate_image_bytes(
        image_bytes,
        image.content_type,
        max_upload_bytes=settings.max_upload_bytes,
        max_image_pixels=settings.max_image_pixels,
    )

    try:
        assessment = await inspector.inspect(image_bytes, mime_type)
        if not isinstance(assessment, CleaningAssessment):
            assessment = CleaningAssessment.model_validate(assessment)
    except ValidationError as exc:
        raise StructuredOutputInvalid() from exc
    except InspectionSystemError:
        raise

    return CleaningEvaluationResponse(
        **assessment.model_dump(),
        zone_id=zone_id,
        checkpoint_id=checkpoint_id,
        user_id=user_id,
    )
