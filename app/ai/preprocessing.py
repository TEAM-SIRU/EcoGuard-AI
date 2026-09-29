from __future__ import annotations

from PIL import Image

from app.ai.dustpan_yolo import Detection, DustpanDetector
from app.ai.region_masker import create_dustpan_mask
from app.core.config import settings
from app.services.errors import system_error


def mask_image_with_dustpan(image: Image.Image, detector: DustpanDetector) -> tuple[Image.Image, Detection]:
    detections = detector.predict(image)
    dustpans = [detection for detection in detections if detection.label == "dustpan"]
    if not dustpans:
        raise system_error("DATASET_NOT_READY", "No dustpan detection available for masking this training/build image")
    selected = max(dustpans, key=lambda detection: detection.confidence)
    return create_dustpan_mask(image, selected.bbox), selected


def imagenet_normalized_tensor(image: Image.Image, image_size: int, mean=settings.imagenet_mean, std=settings.imagenet_std):
    """Shared resize and ImageNet normalization for reference and PatchCore features."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize, to_tensor

    tensor = to_tensor(resize(image.convert("RGB"), [image_size, image_size], interpolation=InterpolationMode.BILINEAR))
    return normalize(tensor, mean=list(mean), std=list(std)).unsqueeze(0)
