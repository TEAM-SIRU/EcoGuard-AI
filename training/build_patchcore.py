from __future__ import annotations

from PIL import Image

from app.ai.zone_anomaly import anomaly_score, build_memory_bank, load_patchcore_feature_extractor, save_memory_bank
from app.ai.dustpan_yolo import load_dustpan_detector
from app.ai.preprocessing import mask_image_with_dustpan
from app.core.config import Settings, settings
from app.services.zone_registry import CheckpointAssets
from training.dataset_validator import patchcore_images
from training.metadata import package_versions, save_checkpoint_metadata
from training.validation import calibrate_threshold


def build_patchcore_assets(cp: CheckpointAssets, config: Settings = settings, force: bool = False, feature_extractor=None, detector=None) -> dict:
    output = cp.patchcore / "memory_bank.npz"
    checkpoint_dir = cp.metadata_path.parent if cp.metadata_path is not None else cp.patchcore.parent
    if not force and (output.exists() or (checkpoint_dir / "patchcore/metadata.json").exists()):
        raise FileExistsError(f"existing PatchCore assets protected; pass --force to replace: {cp.patchcore}")
    extractor = feature_extractor or load_patchcore_feature_extractor(config)
    detector = detector or load_dustpan_detector(config)
    train_paths = patchcore_images(cp.zone_id, cp.checkpoint_id, "train", "normal", config)
    normal_validation = patchcore_images(cp.zone_id, cp.checkpoint_id, "val", "normal", config)
    anomaly_validation = patchcore_images(cp.zone_id, cp.checkpoint_id, "val", "anomaly", config)
    train_images = [_masked(path, detector) for path in train_paths]
    memory = build_memory_bank(extractor, train_images)
    normal_scores = [anomaly_score(extractor.extract(_masked(path, detector)), memory) for path in normal_validation]
    anomaly_scores = [anomaly_score(extractor.extract(_masked(path, detector)), memory) for path in anomaly_validation]
    threshold, validation = calibrate_threshold(normal_scores, anomaly_scores)
    save_memory_bank(output, memory)
    save_checkpoint_metadata(checkpoint_dir, component="patchcore", zone_id=cp.zone_id, checkpoint_id=cp.checkpoint_id,
                             values={"memory_bank": str(output), "anomaly_threshold": threshold, "validation": validation,
                                     "backbone": "resnet18", "backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
                                     "backbone_weight_file": str(config.zone_encoder_weight_path),
                                     "preprocessing": {"mask": "outside_dustpan_black", "image_size": config.image_size, "mean": config.imagenet_mean, "std": config.imagenet_std},
                                     "package_versions": package_versions("torch", "torchvision"),
                                     "feature_dim": int(memory.shape[1]), "train_image_count": len(train_paths)})
    return {"component": "patchcore", "asset": str(output), "threshold": threshold}


def _masked(path, detector):
    with Image.open(path) as image:
        return mask_image_with_dustpan(image.convert("RGB"), detector)[0]
