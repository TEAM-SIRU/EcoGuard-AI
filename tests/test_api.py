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
        data={"zone_id": "zone-a", "checkpoint_id": "stairs-1"},
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
                "trash_detected": False,
                "trash_inside_dustpan": False,
                "cleaning_area_clean": True,
                "needs_review": False,
                "reasons": ["청소구역이 깨끗하고 쓰레받이가 보입니다."],
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
def test_cleaning_decisions_return_assessment_and_metadata(
    client: TestClient, fake_inspector, valid_png: bytes, assessment: dict, expected_decision: str
) -> None:
    fake_inspector.result = CleaningAssessment.model_validate(assessment)
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "record-only-zone", "checkpoint_id": "cp-7", "user_id": "u-2"},
        files=upload(valid_png),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == expected_decision
    assert body["zone_id"] == "record-only-zone"
    assert body["checkpoint_id"] == "cp-7"
    assert body["user_id"] == "u-2"
    assert body["reasons"]


def test_gemini_error_is_a_system_error_not_a_cleaning_fail(
    client: TestClient, fake_inspector, valid_png: bytes
) -> None:
    fake_inspector.error = GeminiApiFailure()
    response = client.post(
        "/api/v1/cleaning/evaluate",
        data={"zone_id": "zone-a", "checkpoint_id": "stairs-1"},
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
        data={"zone_id": "zone-a", "checkpoint_id": "stairs-1"},
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
            data={"zone_id": "zone-a", "checkpoint_id": "stairs-1"},
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
            trash_detected=True,
            trash_inside_dustpan=False,
            cleaning_area_clean=False,
            needs_review=False,
            reasons=["쓰레기가 남아 있습니다."],
        )
