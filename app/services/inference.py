from __future__ import annotations

import math

from PIL import Image

from app.ai.dustpan_yolo import DustpanDetector
from app.ai.preprocessing import split_image_regions
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
        self._zone_asset_cache: dict[tuple[str, str], tuple[object, object, float, float]] = {}

    @staticmethod
    def global_readiness(config: Settings = settings) -> tuple[bool, list[str]]:
        missing = []
        if not config.dustpan_weight_path.is_file():
            missing.append(f"dustpan weight: {config.dustpan_weight_path}")
        if not config.zone_encoder_weight_path.is_file():
            missing.append(f"zone encoder weight: {config.zone_encoder_weight_path}")
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

        stair_view, dustpan_crop, dustpan = split_image_regions(image, self.detector)
        if dustpan is None or dustpan_crop is None:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": False}, failures=["DUSTPAN_NOT_FOUND"])

        cp = self.registry.get_checkpoint(zone_id, checkpoint_id)
        _, assets = self.zone_readiness(cp)
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
            similarity = max_reference_similarity(self.encoder.embed(stair_view), reference_bank)
        except Exception as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "Zone reference bank could not be used", {"reason": str(exc)}) from exc
        if similarity < zone_threshold:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_similarity": similarity, "zone_recognized": False,
                                                     "zone_input": "stair_scene_without_dustpan"},
                                           failures=["ZONE_NOT_RECOGNIZED"])

        try:
            score = anomaly_score(_patchcore_features(stair_view, self.config, self.encoder), memory_bank)
        except Exception as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "PatchCore memory bank could not be used", {"reason": str(exc)}) from exc
        if score > anomaly_threshold:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                     "anomaly_score": score, "zone_anomaly": True,
                                                     "stair_input": "stair_scene_without_dustpan"},
                                           failures=["ZONE_ANOMALY_DETECTED"])

        try:
            trashes = [item for item in self.detector.predict(dustpan_crop) if item.label == "trash"]
        except Exception as exc:
            raise system_error("INFERENCE_ERROR", "YOLO failed to inspect the dustpan crop", {"reason": str(exc)}) from exc
        if not trashes:
            return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                           analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                     "anomaly_score": score, "zone_anomaly": False, "trash_detected": False,
                                                     "stair_input": "stair_scene_without_dustpan", "trash_input": "dustpan_crop"},
                                           failures=["TRASH_NOT_FOUND_IN_DUSTPAN"])
        return FinalEvaluator.evaluate(zone_id=zone_id, checkpoint_id=checkpoint_id, user_id=user_id,
                                       analyses={"dustpan_detected": True, "zone_recognized": True, "zone_similarity": similarity,
                                                 "anomaly_score": score, "zone_anomaly": False, "trash_detected": True,
                                                 "trash_count": len(trashes), "stair_input": "stair_scene_without_dustpan",
                                                 "trash_input": "dustpan_crop"})


def _patchcore_features(image: Image.Image, config: Settings, encoder):
    # Reuse the same local ResNet weights but expose its spatial layer-4 map.
    from app.ai.zone_anomaly import PatchCoreFeatures
    if not hasattr(encoder, "patchcore_features"):
        encoder.patchcore_features = PatchCoreFeatures(config.zone_encoder_weight_path, config.image_size)
    return encoder.patchcore_features.extract(image)
