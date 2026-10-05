from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Path, Request, UploadFile
from pydantic import StringConstraints, ValidationError

from app.core.config import Settings
from app.schemas.cleaning import CleaningAssessment, CleaningEvaluationResponse
from app.services.gemini_inspector import (
    GeminiCleaningInspector,
    InspectionSystemError,
    StructuredOutputInvalid,
)
from app.services.image_validation import ImageValidationError, validate_image_bytes


router = APIRouter(tags=["AI 검수"])
MetadataId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


def get_inspector(request: Request) -> GeminiCleaningInspector:
    return request.app.state.inspector


async def inspect_image(
    request: Request,
    image: UploadFile,
    inspector: GeminiCleaningInspector,
) -> CleaningAssessment:
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

    return assessment


@router.post(
    "/api/v1/cleaning/evaluate",
    response_model=CleaningEvaluationResponse,
    response_model_exclude_none=True,
    include_in_schema=False,
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
    assessment = await inspect_image(request, image, inspector)
    return CleaningEvaluationResponse(
        **assessment.model_dump(),
        zone_id=zone_id,
    )


@router.post(
    "/verifications/{verificationId}/ai-review",
    response_model=CleaningEvaluationResponse,
    response_model_exclude_none=True,
    summary="청소 사진 AI 검수",
    description=(
        "CSV 명세상 시스템 전용 경로입니다. 이 FastAPI 앱은 역할 인증을 구현하지 않으므로 "
        "인증된 애플리케이션 백엔드 뒤에서 호출해야 합니다. "
        "Gemini는 청소 사진을 판정합니다. "
        "verificationId는 백엔드 검증 항목을 가리키는 경로 값이며 이 서비스는 존재 여부를 조회하지 않습니다. "
        "zone_id는 기록용 Metadata입니다. 두 값 모두 Gemini 판정 입력으로 보내지 않습니다."
    ),
    response_description="청소 판정 결과와 구역 ID",
)
async def request_verification_ai_review(
    request: Request,
    verification_id: Annotated[
        MetadataId,
        Path(alias="verificationId", description="검증 ID (1~128자)"),
    ],
    image: Annotated[
        UploadFile,
        File(description="판정할 청소구역 사진 한 장(JPEG, PNG, WebP 형식)"),
    ],
    zone_id: Annotated[
        MetadataId,
        Form(description="요청 기록용 청소구역 ID(1~128자)"),
    ],
    inspector: GeminiCleaningInspector = Depends(get_inspector),
) -> CleaningEvaluationResponse:
    assessment = await inspect_image(request, image, inspector)
    return CleaningEvaluationResponse(
        **assessment.model_dump(),
        zone_id=zone_id,
    )
