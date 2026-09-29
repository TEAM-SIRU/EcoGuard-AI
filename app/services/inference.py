from __future__ import annotations

from pathlib import Path
import math

from PIL import Image

from app.ai.dustpan_yolo import Detection, DustpanDetector
from app.ai.region_masker import create_dustpan_mask, validate_bbox, trash_is_inside
from app.ai.zone_anomaly import anomaly_score, load_memory_bank
from app.ai.zone_recognition import load_reference_bank, max_reference_similarity
from app.core.config import Settings, settings
from app.schemas.cleaning import CleaningResult
from app.services.errors import system_error
from app.services.evaluator import FinalEvaluator
from app.services.zone_registry import CheckpointAssets, ZoneRegistry
from training.metadata import load_metadata


class CleaningInference:
    def __init__(self, config: Settings, registry: ZoneRegistry, detector: DustpanDetector | None, encoder):
        self.config = config
        self.registry = registry
        self.detector = detector
        self.encoder = encoder
        self.trash_inside_threshold = self._trash_threshold(config)
        self._zone_asset_cache: dict[tuple[str, str], tuple[object, object, float, float]] = {}

    @staticmethod
    def _trash_threshold(config: Settings) -> float | None:
        if config.trash_inside_threshold is not None:
            value = config.trash_inside_threshold
            if not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
                return value
            return None
        try:
            value = load_metadata(config.dustpan_weight_path.parent / "metadata.json")["values"].get("trash_inside_threshold")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 <= float(value) <= 1:
                return float(value)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            pass
        return None

    @staticmethod
    def global_readiness(config: Settings = settings) -> tuple[bool, list[str]]:
        missing = []
        if not config.dustpan_weight_path.is_file():
            missing.append(f"dustpan weight: {config.dustpan_weight_path}")
        if not config.zone_encoder_weight_path.is_file():
            missing.append(f"zone encoder weight: {config.zone_encoder_weight_path}")
        if CleaningInference._trash_threshold(config) is None:
            missing.append("TRASH_INSIDE_THRESHOLD has not been calibrated or configured")
        return not missing, missing

    @staticmethod
    def zone_readiness(cp: CheckpointAssets) -> tuple[bool, dict]:
        root_metadata_path = cp.metadata_path or (cp.reference_bank.parent / "metadata.json")
        try:
            root = load_metadata(root_metadata_path)
            values = root["values"]
            zone = values.get("zone", {})
            patch = values.get("patchcore", {})
            if not isinstance(zone, dict) or not isinstance(patch, dict):
                raise ValueError("zone/PatchCore metadata must be mappings")
            reference_path = cp.reference_bank / "reference_bank.npz"
            memory_path = cp.patchcore / "memory_bank.npz"
            if not reference_path.is_file() or not memory_path.is_file():
                raise ValueError("reference bank or PatchCore memory bank is missing")
            if zone.get("zone_threshold") is None or patch.get("anomaly_threshold") is None:
                raise ValueError("zone/anomaly threshold is missing")
            zone_threshold = float(zone["zone_threshold"])
            anomaly_threshold = float(patch["anomaly_threshold"])
            if not math.isfinite(zone_threshold) or not -1 <= zone_threshold <= 1:
                raise ValueError("zone threshold must be finite and within [-1, 1]")
            if not math.isfinite(anomaly_threshold) or anomaly_threshold < 0:
                raise ValueError("anomaly threshold must be finite and non-negative")
            return True, {"reference_path": reference_path, "memory_path": memory_path,
                          "zone_threshold": zone_threshold, "anomaly_threshold": anomaly_threshold}
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "Zone assets or validated thresholds are missing or invalid", {"zone_id": cp.zone_id, "checkpoint_id": cp.checkpoint_id, "reason": str(exc)}) from exc

    def evaluate(self, image: Image.Image, zone_id: str, checkpoint_id: str, user_id: str | None) -> CleaningResult:
        ready, missing = self.global_readiness(self.config)
        if not ready or self.detector is None or self.encoder is None:
            missing = list(missing)
            if self.detector is None:
                missing.append("Dustpan detector failed to load")
            if self.encoder is None:
                missing.append("Zone encoder failed to load")
            raise system_error("MODEL_NOT_READY", "Required global AI assets are unavailable", {"missing": missing})

        detections = self.detector.predict(image)
        dustpans = [d for d in detections if d.label == "dustpan"]
        trashes = [d for d in detections if d.label == "trash"]
        if not dustpans:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": False}, failures=["DUSTPAN_NOT_FOUND"])
        dustpan = max(dustpans, key=lambda item: item.confidence)
        validate_bbox(dustpan.bbox, *image.size)
        for trash in trashes:
            validate_bbox(trash.bbox, *image.size)
        masked = create_dustpan_mask(image, dustpan.bbox)

        cp = self.registry.get_checkpoint(zone_id, checkpoint_id)
        ready, assets = self.zone_readiness(cp)
        if not ready:
            raise system_error("ZONE_MODEL_NOT_READY", "Zone assets are not ready")

        cache_key = (str(assets["reference_path"]), str(assets["memory_path"]))
        if cache_key not in self._zone_asset_cache:
            try:
                reference_bank = load_reference_bank(assets["reference_path"])
                memory_bank = load_memory_bank(assets["memory_path"])
                self._zone_asset_cache[cache_key] = (reference_bank, memory_bank, assets["zone_threshold"], assets["anomaly_threshold"])
            except Exception as exc:
                raise system_error("ZONE_MODEL_NOT_READY", "Zone assets could not be loaded", {"reason": str(exc)}) from exc
        reference_bank, memory_bank, zone_threshold, anomaly_threshold = self._zone_asset_cache[cache_key]
        try:
            similarity = max_reference_similarity(self.encoder.embed(masked), reference_bank)
        except Exception as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "Zone reference bank could not be used", {"reason": str(exc)}) from exc
        if similarity < zone_threshold:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_similarity": similarity, "zone_recognized": False},
                                           failures=["ZONE_NOT_RECOGNIZED"])

        try:
            score = anomaly_score(_patchcore_features(masked, self.config, self.encoder), memory_bank)
        except Exception as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "PatchCore memory bank could not be used", {"reason": str(exc)}) from exc
        if score > anomaly_threshold:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                     "anomaly_score": score, "zone_anomaly": True},
                                           failures=["ZONE_ANOMALY_DETECTED"])

        if not trashes:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                     "anomaly_score": score, "zone_anomaly": False, "trash_detected": False},
                                           failures=["TRASH_NOT_FOUND_IN_DUSTPAN"])
        outside = [not trash_is_inside(dustpan.bbox, trash.bbox, overlap_threshold=self.trash_inside_threshold) for trash in trashes]
        return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                       analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                 "anomaly_score": score, "zone_anomaly": False, "trash_detected": True,
                                                 "trash_outside": any(outside), "trash_count": len(trashes)},
                                       failures=["TRASH_OUTSIDE_DUSTPAN"] if any(outside) else [])


def _patchcore_features(image: Image.Image, config: Settings, encoder):
    # Reuse the same local ResNet weights but expose its spatial layer-4 map.
    from app.ai.zone_anomaly import PatchCoreFeatures
    if not hasattr(encoder, "patchcore_features"):
        encoder.patchcore_features = PatchCoreFeatures(config.zone_encoder_weight_path, config.image_size)
    return encoder.patchcore_features.extract(image)
