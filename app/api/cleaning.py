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


router = APIRouter(prefix="/api/v1/cleaning", tags=["청소 판정"])
MetadataId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


def get_inspector(request: Request) -> GeminiCleaningInspector:
    return request.app.state.inspector


@router.post(
    "/evaluate",
    response_model=CleaningEvaluationResponse,
    summary="청소 상태 판정",
    description=(
        "사진 한 장을 Gemini API로 검사합니다. 사진을 판독하기 어렵거나 필수 조건을 "
        "확실히 판단할 수 없으면 REVIEW를 반환합니다. 구역 ID는 요청 기록용 메타데이터이며 "
        "사진 판정이나 실제 장소 검증에는 사용되지 않습니다."
    ),
    response_description="청소 상태 판정 결과와 구역 ID",
)
async def evaluate_cleaning(
    request: Request,
    image: Annotated[
        UploadFile,
        File(description="판정할 청소구역 사진 한 장(JPEG, PNG, WebP 형식)"),
    ],
    zone_id: Annotated[
        MetadataId, Form(description="요청 기록용 청소구역 ID(1~128자)")
    ],
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
    )
