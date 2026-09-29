from __future__ import annotations

from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_versions(*packages: str) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def save_metadata(path: Path, *, component: str, zone_id: str | None, checkpoint_id: str | None, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "component": component,
        "zone_id": zone_id,
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "values": values,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("values"), dict):
        raise ValueError(f"invalid metadata schema: {path}")
    return payload


def save_threshold(path: Path, *, name: str, value: float, validation_summary: dict) -> None:
    save_metadata(path, component="threshold", zone_id=None, checkpoint_id=None, values={
        "name": name, "value": float(value), "validation_summary": validation_summary,
    })



def save_checkpoint_metadata(checkpoint_dir: Path, *, component: str, zone_id: str, checkpoint_id: str, values: dict) -> None:
    """Keep component records and merge them into the checkpoint's single runtime metadata file."""
    component_dir = checkpoint_dir / ("reference_embeddings" if component == "zone" else "patchcore")
    save_metadata(component_dir / "metadata.json", component=component, zone_id=zone_id, checkpoint_id=checkpoint_id, values=values)
    root_path = checkpoint_dir / "metadata.json"
    merged_values = {}
    if root_path.is_file():
        try:
            merged_values = dict(load_metadata(root_path)["values"])
        except (OSError, ValueError, KeyError, TypeError):
            merged_values = {}
    merged_values[component] = values
    save_metadata(root_path, component="zone_checkpoint", zone_id=zone_id, checkpoint_id=checkpoint_id, values=merged_values)
