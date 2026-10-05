from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Decision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class CleaningAssessment(BaseModel):
    model_config = ConfigDict(title="청소 판정 결과", extra="forbid")

    decision: Decision = Field(
        description=(
            "최종 판정. 모든 필수 조건을 충족하면 PASS, 확인 가능한 조건 중 하나라도 "
            "충족하지 못하면 FAIL, 사진 판독이 어려우면 REVIEW입니다."
        )
    )
    image_assessable: bool = Field(
        description="사진에서 필수 청소 조건을 모두 명확하게 판단할 수 있는지 여부"
    )
    dustpan_detected: bool = Field(description="사진에서 쓰레받이가 확인되는지 여부")
    trash_detected: bool = Field(description="사진에서 쓰레기가 보이는지 여부")
    trash_inside_dustpan: bool = Field(
        description="쓰레기가 쓰레받이 안에 담겨 있는지 여부"
    )
    cleaning_area_clean: bool = Field(
        description="사진에 보이는 청소구역에 남은 쓰레기나 오염이 없는지 여부"
    )
    needs_review: bool = Field(description="사람의 재확인이 필요한지 여부")
    reasons: list[str] = Field(description="판정 근거를 설명하는 짧고 사실적인 문장 목록")

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "CleaningAssessment":
        if self.trash_inside_dustpan and not self.trash_detected:
            raise ValueError("trash_inside_dustpan cannot be true when no trash is detected")

        if not self.image_assessable:
            expected_decision = Decision.REVIEW
        else:
            required_conditions = (
                self.dustpan_detected,
                self.trash_detected,
                self.trash_inside_dustpan,
                self.cleaning_area_clean,
            )
            expected_decision = (
                Decision.PASS if all(required_conditions) else Decision.FAIL
            )

        if self.decision is not expected_decision:
            raise ValueError(
                f"decision must be {expected_decision.value} for the supplied assessment fields"
            )
        if self.needs_review != (expected_decision is Decision.REVIEW):
            raise ValueError("needs_review must be true exactly when decision is REVIEW")
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must contain at least one non-empty explanation")
        return self


class GeminiCleaningAssessment(CleaningAssessment):
    """Gemini용 스키마. 응답은 CleaningAssessment로 다시 엄격하게 검증한다."""

    model_config = ConfigDict(title="Gemini 청소 판정 응답", extra="ignore")


class CleaningEvaluationResponse(CleaningAssessment):
    zone_id: str = Field(description="요청 기록용 청소구역 ID")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(title="오류 응답")

    error_code: str = Field(description="오류 종류를 나타내는 코드")
    message: str = Field(description="오류 내용 안내")
