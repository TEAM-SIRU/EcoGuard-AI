from __future__ import annotations

from dataclasses import dataclass
import math

from PIL import Image, ImageDraw

from app.services.errors import system_error


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


def validate_bbox(box: BoundingBox, width: int, height: int) -> None:
    values = (box.x1, box.y1, box.x2, box.y2)
    if not all(isinstance(value, (int, float)) for value in values):
        raise system_error("INVALID_DETECTION_RESULT", "Bounding box coordinates must be numeric")
    if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
        raise system_error("INVALID_DETECTION_RESULT", "Bounding box is outside the image or has no area", {"bbox": values, "width": width, "height": height})


def _pixel_bounds(image: Image.Image, box: BoundingBox) -> tuple[int, int, int, int]:
    width, height = image.size
    validate_bbox(box, width, height)
    left, top = int(math.floor(box.x1)), int(math.floor(box.y1))
    right, bottom = int(math.ceil(box.x2)), int(math.ceil(box.y2))
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def create_stair_view(image: Image.Image, dustpan_box: BoundingBox) -> Image.Image:
    """Keep the scene and black out the detected dustpan ROI for Zone/PatchCore."""
    rgb = image.convert("RGB")
    left, top, right, bottom = _pixel_bounds(rgb, dustpan_box)
    mask = Image.new("L", rgb.size, 255)
    ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=0)
    return Image.composite(rgb, Image.new("RGB", rgb.size), mask)


def crop_dustpan(image: Image.Image, dustpan_box: BoundingBox) -> Image.Image:
    """Crop the dustpan ROI for an independent YOLO trash-detection pass."""
    rgb = image.convert("RGB")
    return rgb.crop(_pixel_bounds(rgb, dustpan_box))
