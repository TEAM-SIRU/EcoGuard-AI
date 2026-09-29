from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (12, 12), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_runs_without_model_assets():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_ready"] is False


def test_evaluation_reports_model_not_ready_without_weights():
    with TestClient(app) as client:
        response = client.post("/api/v1/cleaning/evaluate", data={"zone_id": "unknown", "checkpoint_id": "unknown"},
                               files={"image": ("test.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_image_errors_are_reported_before_model_readiness():
    with TestClient(app) as client:
        response = client.post("/api/v1/cleaning/evaluate", data={"zone_id": "x", "checkpoint_id": "y"},
                               files={"image": ("bad.png", b"bad", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_unsupported_image_format_returns_415_before_model_readiness():
    buffer = BytesIO()
    Image.new("RGB", (12, 12), "white").save(buffer, format="TIFF")
    with TestClient(app) as client:
        response = client.post("/api/v1/cleaning/evaluate", data={"zone_id": "x", "checkpoint_id": "y"},
                               files={"image": ("bad.tiff", buffer.getvalue(), "image/tiff")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_FORMAT"
