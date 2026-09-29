from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.ai.region_masker import BoundingBox
from app.core.config import Settings, settings
from app.services.errors import system_error


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox


class DustpanDetector:
    def __init__(self, weight_path: Path):
        if not weight_path.is_file():
            raise system_error("MODEL_NOT_READY", "Dustpan YOLO weight is missing", {"path": str(weight_path)})
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(weight_path))
        except Exception as exc:
            raise system_error("MODEL_NOT_READY", "Dustpan YOLO weight could not be loaded", {"reason": str(exc)}) from exc

    def predict(self, image: Image.Image) -> list[Detection]:
        try:
            result = self._model.predict(source=image, verbose=False)[0]
            if result.boxes is None:
                return []
            detections = []
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                coordinates = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(Detection(label, float(box.conf.item()), BoundingBox(*coordinates)))
            return detections
        except Exception as exc:
            raise system_error("INFERENCE_ERROR", "Dustpan detector inference failed", {"reason": str(exc)}) from exc


def load_dustpan_detector(config: Settings = settings) -> DustpanDetector:
    return DustpanDetector(config.dustpan_weight_path)
