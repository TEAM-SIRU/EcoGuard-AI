from __future__ import annotations

from dataclasses import dataclass

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
    if not all(isinstance(v, (int, float)) for v in values):
        raise system_error("INVALID_DETECTION_RESULT", "Bounding box coordinates must be numeric")
    if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
        raise system_error("INVALID_DETECTION_RESULT", "Bounding box is outside the image or has no area", {"bbox": values, "width": width, "height": height})


def create_dustpan_mask(image: Image.Image, box: BoundingBox) -> Image.Image:
    """Keep the detected dustpan area and black out the rest without cropping/resizing."""
    width, height = image.size
    validate_bbox(box, width, height)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((int(box.x1), int(box.y1), int(box.x2), int(box.y2)), fill=255)
    return Image.composite(image.convert("RGB"), Image.new("RGB", (width, height)), mask)


def trash_center_inside(dustpan: BoundingBox, trash: BoundingBox) -> bool:
    _validate_rect(dustpan)
    _validate_rect(trash)
    center_x = (trash.x1 + trash.x2) / 2
    center_y = (trash.y1 + trash.y2) / 2
    return dustpan.x1 <= center_x <= dustpan.x2 and dustpan.y1 <= center_y <= dustpan.y2


def trash_overlap_ratio(dustpan: BoundingBox, trash: BoundingBox) -> float:
    """Return intersection area divided by the trash bbox area."""
    _validate_rect(dustpan)
    _validate_rect(trash)
    intersection_width = max(0.0, min(dustpan.x2, trash.x2) - max(dustpan.x1, trash.x1))
    intersection_height = max(0.0, min(dustpan.y2, trash.y2) - max(dustpan.y1, trash.y1))
    trash_area = (trash.x2 - trash.x1) * (trash.y2 - trash.y1)
    return (intersection_width * intersection_height) / trash_area


def trash_is_inside(dustpan: BoundingBox, trash: BoundingBox, *, overlap_threshold: float) -> bool:
    """Require trash center inside the dustpan and enough of its area overlapping it."""
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in [0, 1]")
    return trash_center_inside(dustpan, trash) and trash_overlap_ratio(dustpan, trash) >= overlap_threshold


def _validate_rect(box: BoundingBox) -> None:
    if not (box.x1 < box.x2 and box.y1 < box.y2):
        raise ValueError("bounding box must have positive area")
