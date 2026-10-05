import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app
from app.schemas.cleaning import CleaningAssessment
from app.services.gemini_inspector import (
    GeminiApiFailure,
    GeminiCleaningInspector,
    StructuredOutputInvalid,
)


def upload(valid_png: bytes) -> dict:
    return {"image": ("stairs.png", valid_png, "image/png")}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_image_is_rejected_before_gemini(client: TestClient, fake_inspector, valid_png) -> None:
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "zone-a"},
        files={"image": ("stairs.png", b"not an image", "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_IMAGE"
    assert fake_inspector.calls == 0


@pytest.mark.parametrize(
    ("assessment", "expected_decision"),
    [
        (
            {
                "decision": "PASS",
                "image_assessable": True,
                "dustpan_detected": True,
                "trash_detected": True,
                "trash_inside_dustpan": True,
                "cleaning_area_clean": True,
                "needs_review": False,
                "reasons": ["수거한 쓰레기가 쓰레받이 안에 있고 청소구역이 깨끗합니다."],
            },
            "PASS",
        ),
        (
            {
                "decision": "FAIL",
                "image_assessable": True,
                "dustpan_detected": True,
                "trash_detected": True,
                "trash_inside_dustpan": False,
                "cleaning_area_clean": False,
                "needs_review": False,
                "reasons": ["쓰레받이 밖에 쓰레기가 남아 있습니다."],
            },
            "FAIL",
        ),
        (
            {
                "decision": "FAIL",
                "image_assessable": True,
                "dustpan_detected": True,
                "trash_detected": False,
                "trash_inside_dustpan": False,
                "cleaning_area_clean": True,
                "needs_review": False,
                "reasons": ["사진에서 수거한 쓰레기가 확인되지 않습니다."],
            },
            "FAIL",
        ),
        (
            {
                "decision": "REVIEW",
                "image_assessable": False,
                "dustpan_detected": False,
                "trash_detected": False,
                "trash_inside_dustpan": False,
                "cleaning_area_clean": False,
                "needs_review": True,
                "reasons": ["사진이 어두워 장면을 판정하기 어렵습니다."],
            },
            "REVIEW",
        ),
    ],
)
def test_cleaning_decisions_return_assessment_and_zone_id(
    client: TestClient, fake_inspector, valid_png: bytes, assessment: dict, expected_decision: str
) -> None:
    fake_inspector.result = CleaningAssessment.model_validate(assessment)
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "record-only-zone"},
        files=upload(valid_png),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == expected_decision
    assert body["zone_id"] == "record-only-zone"
    assert "checkpoint_id" not in body
    assert "user_id" not in body
    assert body["reasons"]


def test_gemini_error_is_a_system_error_not_a_cleaning_fail(
    client: TestClient, fake_inspector, valid_png: bytes
) -> None:
    fake_inspector.error = GeminiApiFailure()
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "zone-a"},
        files=upload(valid_png),
    )
    assert response.status_code == 502
    assert response.json()["error_code"] == "GEMINI_API_ERROR"
    assert "decision" not in response.json()


def test_structured_output_error_returns_system_error(
    client: TestClient, fake_inspector, valid_png: bytes
) -> None:
    fake_inspector.error = StructuredOutputInvalid()
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "zone-a"},
        files=upload(valid_png),
    )
    assert response.status_code == 502
    assert response.json()["error_code"] == "STRUCTURED_OUTPUT_INVALID"
    assert "decision" not in response.json()


def test_api_key_missing_is_a_system_error(valid_png: bytes) -> None:
    settings = Settings(_env_file=None, gemini_api_key=SecretStr(""))
    app = create_app(settings=settings, inspector=GeminiCleaningInspector(settings))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cleaning/evaluate",
            data={"zone_id": "zone-a"},
            files=upload(valid_png),
        )
    assert response.status_code == 503
    assert response.json()["error_code"] == "GEMINI_API_KEY_MISSING"


def test_structured_output_validation_error_is_a_system_error() -> None:
    malformed_response = type("Response", (), {"parsed": {"decision": "UNKNOWN"}, "text": None})()
    with pytest.raises(StructuredOutputInvalid):
        GeminiCleaningInspector.parse_response(malformed_response)


def test_pydantic_consistency_validation_rejects_invalid_pass() -> None:
    with pytest.raises(ValueError):
        CleaningAssessment(
            decision="PASS",
            image_assessable=True,
            dustpan_detected=True,
            trash_detected=False,
            trash_inside_dustpan=False,
            cleaning_area_clean=True,
            needs_review=False,
            reasons=["수거한 쓰레기가 보이지 않습니다."],
        )


def test_pydantic_consistency_validation_rejects_review_when_assessable() -> None:
    with pytest.raises(ValueError):
        CleaningAssessment(
            decision="REVIEW",
            image_assessable=True,
            dustpan_detected=True,
            trash_detected=True,
            trash_inside_dustpan=True,
            cleaning_area_clean=True,
            needs_review=True,
            reasons=["모든 필수 조건을 확인했습니다."],
        )
