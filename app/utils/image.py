from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, settings
from app.services.errors import input_error


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    format: str
    width: int
    height: int


def validate_image(data: bytes, config: Settings = settings) -> ValidatedImage:
    if not data:
        raise input_error("INVALID_IMAGE", "Image is empty")
    if len(data) > config.max_upload_bytes:
        raise input_error("UPLOAD_TOO_LARGE", "Image exceeds the upload size limit")
    try:
        with Image.open(BytesIO(data)) as opened:
            image_format = (opened.format or "").upper()
            width, height = opened.size
            if image_format not in config.allowed_image_formats:
                raise input_error("UNSUPPORTED_IMAGE_FORMAT", "Only JPEG and PNG images are supported", {"format": image_format})
            if width <= 0 or height <= 0 or width * height > config.max_image_pixels:
                raise input_error("INVALID_IMAGE", "Image dimensions are invalid or exceed the pixel limit", {"width": width, "height": height})
            opened.verify()
        with Image.open(BytesIO(data)) as reopened:
            return ValidatedImage(reopened.convert("RGB"), image_format, width, height)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise input_error("INVALID_IMAGE", "Image cannot be decoded") from exc
