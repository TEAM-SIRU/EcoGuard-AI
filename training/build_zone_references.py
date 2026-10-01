from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.ai.dustpan_yolo import load_dustpan_detector
from app.ai.preprocessing import stair_view_for_build
from app.ai.zone_recognition import create_reference_bank, load_encoder, save_reference_bank, validate_reference_threshold
from app.core.config import Settings, settings
from app.services.zone_registry import CheckpointAssets
from training.dataset_validator import patchcore_images, zone_reference_images
from training.metadata import package_versions, save_checkpoint_metadata


def build_zone_reference_assets(cp: CheckpointAssets, encoder, config: Settings = settings, force: bool = False, detector=None) -> dict:
    output = cp.reference_bank / "reference_bank.npz"
    checkpoint_dir = cp.metadata_path.parent if cp.metadata_path is not None else cp.reference_bank.parent
    if not force and (output.exists() or (checkpoint_dir / "reference_embeddings/metadata.json").exists()):
        raise FileExistsError(f"existing zone assets protected; pass --force to replace: {cp.reference_bank}")
    paths = zone_reference_images(cp.zone_id, cp.checkpoint_id, "reference", config)
    detector = detector or load_dustpan_detector(config)
    masked = [_mask_image(path, detector) for path in paths]
    embeddings = create_reference_bank(encoder, masked)

    # Explicit held-out positive/negative sets are required for an accepted threshold.
    positives = zone_reference_images(cp.zone_id, cp.checkpoint_id, "validation/positive", config)
    negatives = zone_reference_images(cp.zone_id, cp.checkpoint_id, "negative", config)
    positive_embeddings = [encoder.embed(_mask_image(path, detector)) for path in positives]
    negative_embeddings = [encoder.embed(_mask_image(path, detector)) for path in negatives]
    threshold, validation = validate_reference_threshold(positive_embeddings, negative_embeddings, embeddings)
    save_reference_bank(output, embeddings)
    save_checkpoint_metadata(checkpoint_dir, component="zone", zone_id=cp.zone_id, checkpoint_id=cp.checkpoint_id,
                             values={"reference_bank": str(output), "zone_threshold": threshold, "validation": validation,
                                     "encoder": "torchvision.resnet18", "encoder_weights": "ResNet18_Weights.IMAGENET1K_V1",
                                     "encoder_weight_file": str(config.zone_encoder_weight_path),
                                     "preprocessing": {"stair_view": "dustpan_bbox_black_when_detected", "image_size": config.image_size, "mean": config.imagenet_mean, "std": config.imagenet_std},
                                     "package_versions": package_versions("torch", "torchvision"),
                                     "embedding_dim": int(embeddings.shape[1])})
    return {"component": "zone", "asset": str(output), "threshold": threshold}


def _mask_image(path: Path, detector) -> Image.Image:
    with Image.open(path) as image:
        return stair_view_for_build(image.convert("RGB"), detector)
