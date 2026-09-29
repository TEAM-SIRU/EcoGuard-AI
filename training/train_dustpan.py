from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from PIL import Image

from app.ai.dustpan_yolo import DustpanDetector
from app.ai.region_masker import trash_center_inside, trash_overlap_ratio, validate_bbox
from app.core.config import Settings, settings
from app.services.errors import system_error
from training.dataset_validator import InvalidDataset, validate_yolo_dataset
from training.metadata import load_metadata, package_versions, save_metadata
from training.trash_threshold import TrashValidationExample, calibrate_trash_inside_threshold


def train_dustpan(config: Settings = settings, *, force: bool = False) -> dict:
    output_dir = config.dustpan_weight_path.parent
    best_path = config.dustpan_weight_path
    metadata_path = output_dir / "metadata.json"
    if best_path.exists() and not force:
        try:
            metadata = load_metadata(metadata_path)
            values = metadata["values"]
            threshold = values.get("trash_inside_threshold")
            if (metadata.get("component") == "dustpan" and values.get("pretrained") is False
                    and isinstance(threshold, (int, float)) and math.isfinite(float(threshold)) and 0 <= float(threshold) <= 1
                    and best_path.stat().st_size > 0):
                return {"component": "dustpan", "status": "already_built", "weight": str(best_path), "trash_inside_threshold": float(threshold)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise FileExistsError(f"dustpan weight exists without valid build metadata; inspect it or pass --force: {best_path}")
    dataset = validate_yolo_dataset(config)
    try:
        from ultralytics import YOLO
        # A YAML architecture creates randomly initialized weights; no pretrained .pt is loaded.
        model = YOLO(config.yolo_model_yaml)
        run_name = "dustpan-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result = model.train(
            data=str(dataset.yaml_path),
            project=str(output_dir / "runs"),
            name=run_name,
            exist_ok=False,
            pretrained=False,
            save=True,
            plots=False,
        )
        run_dir = Path(result.save_dir)
        trained_best = run_dir / "weights" / "best.pt"
        if not trained_best.is_file():
            raise RuntimeError(f"YOLO did not produce a best weight at {trained_best}")
        best_model = YOLO(str(trained_best))
        validation = best_model.val(data=str(dataset.yaml_path), verbose=False)

        detector = DustpanDetector(trained_best)
        trash_threshold, trash_validation = _calibrate_trash_threshold(detector, dataset)
        metric_results = getattr(validation, "results_dict", {})
        metrics = {str(key): float(value) for key, value in metric_results.items() if isinstance(value, (int, float))}
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = best_path.with_suffix(".pt.tmp")
        shutil.copy2(trained_best, temp_path)
        temp_path.replace(best_path)
        save_metadata(metadata_path, component="dustpan", zone_id=None, checkpoint_id=None,
                      values={"weight": str(best_path), "dataset_yaml": str(dataset.yaml_path), "train_image_count": len(dataset.train_images),
                              "validation_image_count": len(dataset.val_images), "validation_metrics": metrics,
                              "trash_inside_threshold": trash_threshold, "trash_inside_validation": trash_validation,
                              "class_names": {"0": "dustpan", "1": "trash"}, "architecture": config.yolo_model_yaml,
                              "package_versions": package_versions("ultralytics", "torch", "torchvision"), "pretrained": False})
        return {"component": "dustpan", "status": "built", "weight": str(best_path), "trash_inside_threshold": trash_threshold, "validation_metrics": metrics}
    except (InvalidDataset, FileExistsError):
        raise
    except Exception as exc:
        if hasattr(exc, "code"):
            raise
        raise system_error("TRAINING_FAILED", "Dustpan YOLO training or validation failed", {"reason": str(exc)}) from exc


def _calibrate_trash_threshold(detector: DustpanDetector, dataset) -> tuple[float, dict]:
    examples = []
    for expected_inside, paths in ((True, dataset.trash_inside_images), (False, dataset.trash_outside_images)):
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            detections = detector.predict(image)
            dustpans = [item for item in detections if item.label == "dustpan"]
            if not dustpans:
                raise RuntimeError(f"dustpan detector missed a calibrated validation image: {path}")
            dustpan = max(dustpans, key=lambda item: item.confidence)
            validate_bbox(dustpan.bbox, *image.size)
            trashes = [item for item in detections if item.label == "trash"]
            all_centers_inside = bool(trashes)
            ratios = []
            for trash in trashes:
                validate_bbox(trash.bbox, *image.size)
                all_centers_inside = all_centers_inside and trash_center_inside(dustpan.bbox, trash.bbox)
                ratios.append(trash_overlap_ratio(dustpan.bbox, trash.bbox))
            examples.append(TrashValidationExample(expected_inside, all_centers_inside, min(ratios, default=0.0)))
    return calibrate_trash_inside_threshold(examples)
