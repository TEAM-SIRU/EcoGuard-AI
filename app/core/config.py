from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    dataset_dir: Path = ROOT_DIR / "datasets"
    model_dir: Path = ROOT_DIR / "models"
    zone_registry_path: Path = ROOT_DIR / "configs" / "zones.yaml"
    dustpan_weight_path: Path = ROOT_DIR / "models" / "global" / "dustpan" / "best.pt"
    zone_encoder_weight_path: Path = ROOT_DIR / "models" / "global" / "zone_encoder" / "resnet18-f37072fd.pth"
    allowed_image_formats: tuple[str, ...] = ("JPEG", "PNG")
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_inference_concurrency: int = 1
    image_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # These remain unset until a dataset-backed validation/build supplies them.
    trash_inside_threshold: float | None = None
    yolo_model_yaml: str = "yolov8n.yaml"

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0 or self.max_image_pixels <= 0 or self.max_inference_concurrency <= 0 or self.image_size <= 0:
            raise ValueError("upload, pixel, concurrency, and image-size limits must be positive")
        if any(not math.isfinite(value) for value in (*self.imagenet_mean, *self.imagenet_std)):
            raise ValueError("ImageNet normalization values must be finite")
        if any(value <= 0 for value in self.imagenet_std):
            raise ValueError("ImageNet standard deviations must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        root = Path(os.getenv("ECOGUARD_ROOT", str(base.root_dir))).resolve()
        return cls(
            root_dir=root,
            dataset_dir=Path(os.getenv("ECOGUARD_DATASET_DIR", root / "datasets")).resolve(),
            model_dir=Path(os.getenv("ECOGUARD_MODEL_DIR", root / "models")).resolve(),
            zone_registry_path=Path(os.getenv("ECOGUARD_ZONE_REGISTRY", root / "configs" / "zones.yaml")).resolve(),
            dustpan_weight_path=Path(os.getenv("ECOGUARD_DUSTPAN_WEIGHT", root / "models/global/dustpan/best.pt")).resolve(),
            zone_encoder_weight_path=Path(os.getenv("ECOGUARD_ZONE_ENCODER_WEIGHT", root / "models/global/zone_encoder/resnet18-f37072fd.pth")).resolve(),
            allowed_image_formats=base.allowed_image_formats,
            max_upload_bytes=int(os.getenv("ECOGUARD_MAX_UPLOAD_BYTES", base.max_upload_bytes)),
            max_image_pixels=int(os.getenv("ECOGUARD_MAX_IMAGE_PIXELS", base.max_image_pixels)),
            max_inference_concurrency=int(os.getenv("ECOGUARD_MAX_INFERENCE_CONCURRENCY", base.max_inference_concurrency)),
            image_size=int(os.getenv("ECOGUARD_IMAGE_SIZE", base.image_size)),
            imagenet_mean=base.imagenet_mean,
            imagenet_std=base.imagenet_std,
            trash_inside_threshold=(float(os.environ["ECOGUARD_TRASH_INSIDE_THRESHOLD"]) if os.getenv("ECOGUARD_TRASH_INSIDE_THRESHOLD") else None),
            yolo_model_yaml=os.getenv("ECOGUARD_YOLO_MODEL_YAML", base.yolo_model_yaml),
        )


settings = Settings.from_env()
