from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, settings
from app.services.zone_registry import ZoneRegistry


class DatasetNotReady(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("DATASET_NOT_READY: " + "; ".join(missing))


class InvalidDataset(Exception):
    pass


@dataclass(frozen=True)
class YoloDataset:
    yaml_path: Path
    train_images: tuple[Path, ...]
    val_images: tuple[Path, ...]
    trash_inside_images: tuple[Path, ...]
    trash_outside_images: tuple[Path, ...]


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_yolo_dataset(config: Settings = settings) -> YoloDataset:
    base = config.dataset_dir
    expected = {
        "train images": base / "raw/dustpan/images/train",
        "train labels": base / "raw/dustpan/labels/train",
        "validation images": base / "raw/dustpan/images/val",
        "validation labels": base / "raw/dustpan/labels/val",
        "trash inside validation images": base / "raw/dustpan/trash_inside/inside/images",
        "trash inside validation labels": base / "raw/dustpan/trash_inside/inside/labels",
        "trash outside validation images": base / "raw/dustpan/trash_inside/outside/images",
        "trash outside validation labels": base / "raw/dustpan/trash_inside/outside/labels",
    }
    missing = [f"{label}: {path}" for label, path in expected.items() if not path.is_dir()]
    train = _images(expected["train images"])
    val = _images(expected["validation images"])
    inside = _images(expected["trash inside validation images"])
    outside = _images(expected["trash outside validation images"])
    if not train:
        missing.append(f"no images found in {expected['train images']}")
    if not val:
        missing.append(f"no images found in {expected['validation images']}")
    if not inside:
        missing.append(f"no images found in {expected['trash inside validation images']}")
    if not outside:
        missing.append(f"no images found in {expected['trash outside validation images']}")
    if missing:
        raise DatasetNotReady(missing)
    labels_by_stem: dict[str, Path] = {}
    for split in ("train", "val"):
        image_key = "train" if split == "train" else "validation"
        image_dir = expected[f"{image_key} images"]
        label_dir = expected[f"{image_key} labels"]
        for image_path in _images(image_dir):
            _validate_image(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            labels_by_stem[str(image_path.resolve())] = label_path
            if not label_path.is_file():
                raise InvalidDataset(f"missing YOLO label for {image_path}: {label_path}")
            _validate_yolo_label(label_path)
        image_stems = {path.stem for path in _images(image_dir)}
        extras = sorted(path.name for path in label_dir.glob("*.txt") if path.stem not in image_stems)
        if extras:
            raise InvalidDataset(f"labels without matching image in {label_dir}: {', '.join(extras[:10])}")
    for kind, image_paths in (("inside", inside), ("outside", outside)):
        image_dir = expected[f"trash {kind} validation images"]
        label_dir = expected[f"trash {kind} validation labels"]
        for image_path in image_paths:
            _validate_image(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise InvalidDataset(f"missing YOLO label for trash-{kind} validation image {image_path}: {label_path}")
            _validate_yolo_label(label_path, required_classes={0, 1})
    yaml_path = base / "processed/dustpan_dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "path: " + _yaml_quote(base.resolve()) + "\n"
        "train: raw/dustpan/images/train\n"
        "val: raw/dustpan/images/val\n"
        "names:\n  0: dustpan\n  1: trash\n",
        encoding="utf-8",
    )
    return YoloDataset(yaml_path, tuple(train), tuple(val), tuple(inside), tuple(outside))


def zone_reference_images(zone_id: str, checkpoint_id: str, kind: str = "reference", config: Settings = settings) -> list[Path]:
    if kind == "negative":
        folder = config.dataset_dir / "raw" / "zones" / "negative" / zone_id / checkpoint_id
    else:
        folder = config.dataset_dir / "raw" / "zones" / zone_id / checkpoint_id / kind
    images = _images(folder)
    if not images:
        raise DatasetNotReady([f"zone reference images missing or empty: {folder}"])
    for path in images:
        _validate_image(path)
    return images


def patchcore_images(zone_id: str, checkpoint_id: str, split: str, kind: str, config: Settings = settings) -> list[Path]:
    folder = config.dataset_dir / "raw" / "patchcore" / zone_id / checkpoint_id / split / kind
    images = _images(folder)
    if not images:
        raise DatasetNotReady([f"PatchCore {split}/{kind} images missing or empty: {folder}"])
    for path in images:
        _validate_image(path)
    return images


def validate_registry_datasets(registry: ZoneRegistry, component: str, zone_id: str | None, checkpoint_id: str | None) -> None:
    checkpoints = registry.checkpoints(zone_id)
    if not checkpoints:
        raise DatasetNotReady(["no active Zone/Checkpoint entries exist in the registry"] )
    selected = [cp for cp in checkpoints if checkpoint_id is None or cp.checkpoint_id == checkpoint_id]
    if checkpoint_id and not selected:
        raise DatasetNotReady([f"checkpoint is not registered for zone: {zone_id}/{checkpoint_id}"])
    missing = []
    for cp in selected:
        if component in {"zone", "all"}:
            try:
                zone_reference_images(cp.zone_id, cp.checkpoint_id, "reference")
                zone_reference_images(cp.zone_id, cp.checkpoint_id, "validation/positive")
                zone_reference_images(cp.zone_id, cp.checkpoint_id, "negative")
            except DatasetNotReady as exc:
                missing.extend(exc.missing)
        if component in {"patchcore", "all"}:
            for split, kind in (("train", "normal"), ("val", "normal"), ("val", "anomaly")):
                try:
                    patchcore_images(cp.zone_id, cp.checkpoint_id, split, kind)
                except DatasetNotReady as exc:
                    missing.extend(exc.missing)
    if missing:
        raise DatasetNotReady(missing)


def _validate_yolo_label(path: Path, required_classes: set[int] | None = None) -> None:
    observed_classes = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise InvalidDataset(f"{path}:{line_number}: expected class x_center y_center width height")
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise InvalidDataset(f"{path}:{line_number}: label contains a non-numeric value") from exc
        observed_classes.add(class_id)
        if class_id not in (0, 1):
            raise InvalidDataset(f"{path}:{line_number}: class id must be 0 (dustpan) or 1 (trash)")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1 and x - width / 2 >= 0 and x + width / 2 <= 1 and y - height / 2 >= 0 and y + height / 2 <= 1):
            raise InvalidDataset(f"{path}:{line_number}: normalized box is out of bounds or has no area")
    if required_classes and not required_classes.issubset(observed_classes):
        raise InvalidDataset(f"{path}: expected labels for classes {sorted(required_classes)}")


def _images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidDataset(f"invalid image: {path}") from exc


def _yaml_quote(value: Path) -> str:
    return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'
