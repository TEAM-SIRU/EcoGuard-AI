from io import BytesIO

from PIL import Image, UnidentifiedImageError


class ImageValidationError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int = 422) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def validate_image_bytes(
    data: bytes,
    content_type: str | None,
    *,
    max_upload_bytes: int,
    max_image_pixels: int,
) -> str:
    if not data:
        raise ImageValidationError("INVALID_IMAGE", "업로드한 이미지가 비어 있습니다.")
    if len(data) > max_upload_bytes:
        raise ImageValidationError(
            "IMAGE_TOO_LARGE", "이미지 크기가 업로드 한도를 초과했습니다.", status_code=413
        )

    declared_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_image_pixels:
                raise ImageValidationError(
                    "INVALID_IMAGE", "이미지 해상도가 허용된 범위를 벗어났습니다."
                )
            image.verify()
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageValidationError(
            "INVALID_IMAGE", "유효한 JPEG, PNG 또는 WebP 이미지가 아닙니다."
        ) from exc

    expected_type = SUPPORTED_FORMATS.get(image_format or "")
    if expected_type is None or declared_type != expected_type:
        raise ImageValidationError(
            "INVALID_IMAGE", "파일 내용과 Content-Type은 JPEG, PNG 또는 WebP 형식으로 일치해야 합니다."
        )
    return expected_type
