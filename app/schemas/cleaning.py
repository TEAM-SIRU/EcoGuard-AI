from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Decision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class CleaningResult(BaseModel):
    decision: Decision
    is_passed: bool
    fail_reasons: list[str] = Field(default_factory=list)
    zone_id: str
    checkpoint_id: str
    user_id: str | None = None
    analysis: dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
