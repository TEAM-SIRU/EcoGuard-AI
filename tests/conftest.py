from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app
from app.schemas.cleaning import CleaningAssessment


def valid_pass_assessment() -> CleaningAssessment:
    return CleaningAssessment(
        decision="PASS",
        image_assessable=True,
        dustpan_detected=True,
        trash_detected=True,
        trash_inside_dustpan=True,
        cleaning_area_clean=True,
        needs_review=False,
        reasons=["쓰레받이 안에 쓰레기가 있고 청소구역이 깨끗합니다."],
    )


class FakeInspector:
    def __init__(self) -> None:
        self.result = valid_pass_assessment()
        self.error: Exception | None = None
        self.calls = 0

    async def inspect(self, image_bytes: bytes, mime_type: str) -> CleaningAssessment:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_inspector() -> FakeInspector:
    return FakeInspector()


@pytest.fixture
def app(fake_inspector: FakeInspector):
    settings = Settings(_env_file=None, gemini_api_key=SecretStr(""))
    return create_app(settings=settings, inspector=fake_inspector)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 24), color="white").save(output, format="PNG")
    return output.getvalue()
