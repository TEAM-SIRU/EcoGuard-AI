from __future__ import annotations

from PIL import Image

from app.ai.dustpan_yolo import Detection, DustpanDetector
from app.ai.region_masker import crop_dustpan, create_stair_view, validate_bbox
from app.core.config import settings


def split_image_regions(image: Image.Image, detector: DustpanDetector) -> tuple[Image.Image, Image.Image | None, Detection | None]:
    """Return stair scene, dustpan crop, and detection; stair-only build images are allowed."""
    rgb = image.convert("RGB")
    dustpans = [item for item in detector.predict(rgb) if item.label == "dustpan"]
    if not dustpans:
        return rgb.copy(), None, None
    selected = max(dustpans, key=lambda detection: detection.confidence)
    validate_bbox(selected.bbox, *rgb.size)
    return create_stair_view(rgb, selected.bbox), crop_dustpan(rgb, selected.bbox), selected


def stair_view_for_build(image: Image.Image, detector: DustpanDetector) -> Image.Image:
    """Use stair-only training images as-is; mask a detected pan to match request-time preprocessing."""
    return split_image_regions(image, detector)[0]


def imagenet_normalized_tensor(image: Image.Image, image_size: int, mean=settings.imagenet_mean, std=settings.imagenet_std):
    """Shared resize and ImageNet normalization for reference and PatchCore features."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize, to_tensor

    tensor = to_tensor(resize(image.convert("RGB"), [image_size, image_size], interpolation=InterpolationMode.BILINEAR))
    return normalize(tensor, mean=list(mean), std=list(std)).unsqueeze(0)
