from __future__ import annotations

import argparse
import json
import math
import sys

from app.ai.dustpan_yolo import load_dustpan_detector
from app.ai.zone_recognition import load_encoder
from app.core.config import settings
from app.services.errors import EcoGuardError
from app.services.zone_registry import ZoneRegistry
from training.build_patchcore import build_patchcore_assets
from training.build_zone_references import build_zone_reference_assets
from training.dataset_validator import DatasetNotReady, InvalidDataset, validate_registry_datasets
from training.metadata import load_metadata
from training.train_dustpan import train_dustpan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EcoGuard AI assets from explicitly supplied datasets")
    parser.add_argument("--component", choices=("dustpan", "zone", "patchcore", "all"), required=True)
    parser.add_argument("--zone-id")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--force", action="store_true", help="replace generated assets")
    return parser


def run(component: str, zone_id: str | None = None, checkpoint_id: str | None = None, force: bool = False) -> dict:
    if checkpoint_id and not zone_id:
        raise ValueError("--checkpoint-id requires --zone-id")
    registry = ZoneRegistry.load(settings.zone_registry_path, settings)
    outputs = []
    if component in {"dustpan", "all"}:
        outputs.append(train_dustpan(settings, force=force))
    if component in {"zone", "patchcore", "all"}:
        checkpoints = registry.checkpoints(zone_id)
        if zone_id and not checkpoints:
            raise DatasetNotReady([f"zone is not registered or has no active checkpoints: {zone_id}"])
        if checkpoint_id:
            checkpoints = [cp for cp in checkpoints if cp.checkpoint_id == checkpoint_id]
            if not checkpoints:
                raise DatasetNotReady([f"checkpoint is not registered for zone: {zone_id}/{checkpoint_id}"])
        if not checkpoints:
            raise DatasetNotReady(["no active Zone/Checkpoint entries exist in the registry"])

        detector = None
        encoder = None
        for cp in checkpoints:
            if component in {"zone", "all"}:
                if not force and _zone_assets_ready(cp):
                    outputs.append({"component": "zone", "status": "already_built", "asset": str(cp.reference_bank / "reference_bank.npz")})
                else:
                    validate_registry_datasets(registry, "zone", cp.zone_id, cp.checkpoint_id)
                    if detector is None:
                        detector = load_dustpan_detector(settings)
                    if encoder is None:
                        encoder = load_encoder(settings)
                    outputs.append(build_zone_reference_assets(cp, encoder, settings, force=force, detector=detector))
            if component in {"patchcore", "all"}:
                if not force and _patchcore_assets_ready(cp):
                    outputs.append({"component": "patchcore", "status": "already_built", "asset": str(cp.patchcore / "memory_bank.npz")})
                else:
                    validate_registry_datasets(registry, "patchcore", cp.zone_id, cp.checkpoint_id)
                    if detector is None:
                        detector = load_dustpan_detector(settings)
                    outputs.append(build_patchcore_assets(cp, settings, force=force, detector=detector))
    return {"status": "completed", "component": component, "results": outputs}


def _component_metadata(cp, component: str) -> dict | None:
    try:
        data = load_metadata(cp.metadata_path or (cp.reference_bank.parent / "metadata.json"))
        values = data["values"].get(component)
        return values if isinstance(values, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _zone_assets_ready(cp) -> bool:
    values = _component_metadata(cp, "zone")
    threshold = values.get("zone_threshold") if values else None
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not -1 <= float(threshold) <= 1:
        return False
    try:
        from app.ai.zone_recognition import load_reference_bank
        load_reference_bank(cp.reference_bank / "reference_bank.npz")
        return True
    except (OSError, ValueError, KeyError):
        return False


def _patchcore_assets_ready(cp) -> bool:
    values = _component_metadata(cp, "patchcore")
    threshold = values.get("anomaly_threshold") if values else None
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or float(threshold) < 0:
        return False
    try:
        from app.ai.zone_anomaly import load_memory_bank
        load_memory_bank(cp.patchcore / "memory_bank.npz")
        return True
    except (OSError, ValueError, KeyError):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.component, args.zone_id, args.checkpoint_id, args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except DatasetNotReady as exc:
        print(json.dumps({"status": "failed", "error": "DATASET_NOT_READY", "missing": exc.missing}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except InvalidDataset as exc:
        print(json.dumps({"status": "failed", "error": "INVALID_DATASET", "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(json.dumps({"status": "failed", "error": "ASSET_EXISTS", "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except EcoGuardError as exc:
        print(json.dumps({"status": "failed", "error": exc.code, "message": exc.message, "detail": exc.detail}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": "INVALID_DATASET", "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
