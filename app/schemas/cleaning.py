from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Decision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class CleaningAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision = Field(description="PASS, FAIL, or REVIEW")
    image_assessable: bool = Field(description="Whether the photo is clear enough to inspect")
    dustpan_detected: bool = Field(description="Whether a dustpan is visibly present")
    trash_detected: bool = Field(description="Whether any trash is visible in the photo")
    trash_inside_dustpan: bool = Field(
        description="True only when visible trash is clearly inside the dustpan"
    )
    cleaning_area_clean: bool = Field(
        description="Whether the visible cleaning area has no obvious remaining trash or contamination"
    )
    needs_review: bool = Field(description="Whether a person should inspect the result")
    reasons: list[str] = Field(description="Short, factual reasons for the decision")

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "CleaningAssessment":
        if self.needs_review != (self.decision is Decision.REVIEW):
            raise ValueError("needs_review must be true exactly when decision is REVIEW")
        if not self.image_assessable and self.decision is not Decision.REVIEW:
            raise ValueError("an unassessable image must be marked REVIEW")
        if self.trash_inside_dustpan and not self.trash_detected:
            raise ValueError("trash_inside_dustpan cannot be true when no trash is detected")
        if self.decision is Decision.PASS:
            if not self.dustpan_detected or not self.cleaning_area_clean:
                raise ValueError("PASS requires a visible dustpan and a clean area")
            if self.trash_detected and not self.trash_inside_dustpan:
                raise ValueError("PASS cannot include visible trash outside the dustpan")
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must contain at least one non-empty explanation")
        return self


class CleaningEvaluationResponse(CleaningAssessment):
    zone_id: str
    checkpoint_id: str
    user_id: str | None = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
