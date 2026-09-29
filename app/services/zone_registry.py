from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import math

import yaml

from app.core.config import Settings, settings
from app.services.errors import input_error, system_error

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


@dataclass(frozen=True)
class CheckpointAssets:
    zone_id: str
    checkpoint_id: str
    reference_bank: Path
    patchcore: Path
    zone_threshold: float | None
    anomaly_threshold: float | None
    metadata: dict
    metadata_path: Path | None = None


@dataclass(frozen=True)
class Zone:
    zone_id: str
    display_name: str
    active: bool
    checkpoints: dict[str, CheckpointAssets]


class ZoneRegistry:
    def __init__(self, zones: dict[str, Zone]):
        self._zones = zones

    @classmethod
    def load(cls, path: Path, config: Settings = settings) -> "ZoneRegistry":
        if not path.is_file():
            raise system_error("ZONE_MODEL_NOT_READY", "Zone registry file is missing", {"path": str(path)})
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict) or not isinstance(raw.get("zones", {}), dict):
                raise ValueError("top-level 'zones' must be a mapping")
            zones = {}
            for zone_id, entry in raw.get("zones", {}).items():
                _validate_id(zone_id, "zone_id")
                if not isinstance(entry, dict):
                    raise ValueError(f"zone {zone_id!r} must be a mapping")
                active = entry.get("active", True)
                if not isinstance(active, bool):
                    raise ValueError(f"active for zone {zone_id!r} must be a boolean")
                checkpoints_raw = entry.get("checkpoints", {})
                if not isinstance(checkpoints_raw, dict):
                    raise ValueError(f"checkpoints for {zone_id!r} must be a mapping")
                checkpoints = {}
                for checkpoint_id, cp in checkpoints_raw.items():
                    _validate_id(checkpoint_id, "checkpoint_id")
                    if not isinstance(cp, dict):
                        raise ValueError(f"checkpoint {zone_id}/{checkpoint_id} must be a mapping")
                    metadata = cp.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValueError(f"metadata for checkpoint {zone_id}/{checkpoint_id} must be a mapping")
                    checkpoints[checkpoint_id] = CheckpointAssets(
                        zone_id=zone_id,
                        checkpoint_id=checkpoint_id,
                        reference_bank=_asset_path(config.model_dir, cp.get("reference_bank"), f"models/zones/{zone_id}/{checkpoint_id}/reference_embeddings"),
                        patchcore=_asset_path(config.model_dir, cp.get("patchcore"), f"models/zones/{zone_id}/{checkpoint_id}/patchcore"),
                        zone_threshold=_optional_threshold(cp.get("zone_threshold"), f"{zone_id}/{checkpoint_id}.zone_threshold"),
                        anomaly_threshold=_optional_threshold(cp.get("anomaly_threshold"), f"{zone_id}/{checkpoint_id}.anomaly_threshold"),
                        metadata=metadata,
                        metadata_path=_asset_path(config.model_dir, cp.get("metadata_path"), f"models/zones/{zone_id}/{checkpoint_id}/metadata.json"),
                    )
                zones[zone_id] = Zone(zone_id, str(entry.get("display_name", zone_id)), active, checkpoints)
            return cls(zones)
        except (OSError, yaml.YAMLError, ValueError, TypeError, OverflowError) as exc:
            raise system_error("ZONE_MODEL_NOT_READY", "Zone registry is invalid", {"reason": str(exc)}) from exc

    def get_checkpoint(self, zone_id: str, checkpoint_id: str) -> CheckpointAssets:
        zone = self._zones.get(zone_id)
        if zone is None or not zone.active:
            raise input_error("UNKNOWN_ZONE", "Unknown or inactive zone", {"zone_id": zone_id})
        checkpoint = zone.checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise input_error("UNKNOWN_CHECKPOINT", "Unknown checkpoint", {"zone_id": zone_id, "checkpoint_id": checkpoint_id})
        return checkpoint

    def checkpoints(self, zone_id: str | None = None) -> list[CheckpointAssets]:
        result = []
        for zone in self._zones.values():
            if zone.active and (zone_id is None or zone.zone_id == zone_id):
                result.extend(zone.checkpoints.values())
        return result


def _validate_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")


def _asset_path(model_dir: Path, value: object, default: str) -> Path:
    if value is not None and not isinstance(value, str):
        raise ValueError("asset paths must be strings")
    path = Path(value) if isinstance(value, str) else Path(default)
    return (path if path.is_absolute() else model_dir.parent / path).resolve()


def _optional_threshold(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric or null")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
