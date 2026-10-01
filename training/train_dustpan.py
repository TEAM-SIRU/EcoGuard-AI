from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from app.ai.dustpan_yolo import DustpanDetector
from app.core.config import Settings, settings
from app.services.errors import system_error
from training.dataset_validator import InvalidDataset, validate_yolo_dataset
from training.metadata import load_metadata, package_versions, save_metadata


def train_dustpan(config: Settings = settings, *, force: bool = False) -> dict:
    output_dir = config.dustpan_weight_path.parent
    best_path = config.dustpan_weight_path
    metadata_path = output_dir / "metadata.json"
    if best_path.exists() and not force:
        try:
            metadata = load_metadata(metadata_path)
            values = metadata["values"]
            if (metadata.get("component") == "dustpan" and values.get("pretrained") is False
                    and best_path.stat().st_size > 0):
                return {"component": "dustpan", "status": "already_built", "weight": str(best_path)}
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
        metric_results = getattr(validation, "results_dict", {})
        metrics = {str(key): float(value) for key, value in metric_results.items() if isinstance(value, (int, float))}
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = best_path.with_suffix(".pt.tmp")
        shutil.copy2(trained_best, temp_path)
        temp_path.replace(best_path)
        # Ensure the saved global detector can be loaded before recording it as ready.
        DustpanDetector(best_path)
        save_metadata(metadata_path, component="dustpan", zone_id=None, checkpoint_id=None,
                      values={"weight": str(best_path), "dataset_yaml": str(dataset.yaml_path), "train_image_count": len(dataset.train_images),
                              "validation_image_count": len(dataset.val_images), "validation_metrics": metrics,
                              "class_names": {"0": "dustpan", "1": "trash"}, "architecture": config.yolo_model_yaml,
                              "package_versions": package_versions("ultralytics", "torch", "torchvision"), "pretrained": False})
        return {"component": "dustpan", "status": "built", "weight": str(best_path), "validation_metrics": metrics}
    except (InvalidDataset, FileExistsError):
        raise
    except Exception as exc:
        if hasattr(exc, "code"):
            raise
        raise system_error("TRAINING_FAILED", "Dustpan YOLO training or validation failed", {"reason": str(exc)}) from exc
