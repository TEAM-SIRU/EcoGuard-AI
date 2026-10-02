from __future__ import annotations

from PIL import Image

from app.ai.zone_anomaly import PATCHCORE_EXCLUSION_MARGIN_PIXELS, PATCHCORE_FEATURE_VERSION, PATCHCORE_GRID_SIZE, anomaly_score, build_memory_bank, load_patchcore_feature_extractor, save_memory_bank
from app.core.config import Settings, settings
from app.services.zone_registry import CheckpointAssets
from training.dataset_validator import patchcore_images
from training.metadata import package_versions, save_checkpoint_metadata
from training.validation import calibrate_normal_threshold, calibrate_threshold


def build_patchcore_assets(cp: CheckpointAssets, config: Settings = settings, force: bool = False, feature_extractor=None, normal_only_threshold: bool = False) -> dict:
    output = cp.patchcore / "memory_bank.npz"
    checkpoint_dir = cp.metadata_path.parent if cp.metadata_path is not None else cp.patchcore.parent
    if not force and (output.exists() or (checkpoint_dir / "patchcore/metadata.json").exists()):
        raise FileExistsError(f"existing PatchCore assets protected; pass --force to replace: {cp.patchcore}")
    extractor = feature_extractor or load_patchcore_feature_extractor(config)
    train_paths = patchcore_images(cp.zone_id, cp.checkpoint_id, "train", "normal", config)
    normal_validation = patchcore_images(cp.zone_id, cp.checkpoint_id, "val", "normal", config)
    # Stream full-resolution photos so dataset size does not determine peak image RAM.
    memory = build_memory_bank(extractor, (_load_rgb(path) for path in train_paths))
    normal_scores = [anomaly_score(extractor.extract(_load_rgb(path)), memory) for path in normal_validation]
    if normal_only_threshold:
        threshold, validation = calibrate_normal_threshold(normal_scores)
    else:
        anomaly_validation = patchcore_images(cp.zone_id, cp.checkpoint_id, "val", "anomaly", config)
        anomaly_scores = [anomaly_score(extractor.extract(_load_rgb(path)), memory) for path in anomaly_validation]
        threshold, validation = calibrate_threshold(normal_scores, anomaly_scores)
    calibration_mode = "normal_only_provisional" if normal_only_threshold else "normal_and_anomaly"
    save_memory_bank(output, memory)
    save_checkpoint_metadata(checkpoint_dir, component="patchcore", zone_id=cp.zone_id, checkpoint_id=cp.checkpoint_id,
                             values={"memory_bank": str(output), "anomaly_threshold": threshold, "threshold_calibration_mode": calibration_mode, "validation": validation,
                                     "backbone": "resnet18", "backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
                                     "backbone_weight_file": str(config.zone_encoder_weight_path),
                                     "preprocessing": {"training_input": "stair_only_image_without_dustpan_yolo", "inference_input": "black_mask_dustpan_bbox_and_ignore_feature_roi", "feature_grid_size": PATCHCORE_GRID_SIZE, "dustpan_context_exclusion_pixels": PATCHCORE_EXCLUSION_MARGIN_PIXELS, "image_size": config.image_size, "mean": config.imagenet_mean, "std": config.imagenet_std},
                                     "feature_version": PATCHCORE_FEATURE_VERSION,
                                     "package_versions": package_versions("torch", "torchvision"),
                                     "feature_dim": int(memory.shape[1]), "train_image_count": len(train_paths)})
    return {"component": "patchcore", "asset": str(output), "threshold": threshold, "threshold_calibration_mode": calibration_mode, "validation": validation}


def _load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")
