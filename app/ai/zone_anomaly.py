from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.zone_recognition import l2_normalize
from app.ai.region_masker import BoundingBox, validate_bbox
from app.ai.preprocessing import imagenet_normalized_tensor
from app.core.config import Settings, settings


PATCHCORE_FEATURE_VERSION = 2
PATCHCORE_GRID_SIZE = 14
PATCHCORE_EXCLUSION_MARGIN_PIXELS = 32


class PatchCoreFeatures:
    def __init__(self, weight_path: Path = settings.zone_encoder_weight_path, image_size: int = settings.image_size):
        import torch
        from torchvision.models import resnet18

        if not weight_path.is_file():
            from app.services.errors import system_error
            raise system_error("MODEL_NOT_READY", "Local ResNet18 ImageNet weights are missing", {"path": str(weight_path)})
        model = resnet18(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        model.load_state_dict(state, strict=False)
        self._torch = torch
        self._model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1).eval()
        self._functional = torch.nn.functional
        self.image_size = image_size
        self.grid_size = PATCHCORE_GRID_SIZE

    def extract(self, image: Image.Image, ignored_bbox: BoundingBox | None = None) -> np.ndarray:
        torch = self._torch
        rgb = image.convert("RGB")
        tensor = imagenet_normalized_tensor(rgb, self.image_size)
        with torch.inference_mode():
            feature_map = self._model(tensor)
            feature_map = self._functional.adaptive_avg_pool2d(feature_map, (self.grid_size, self.grid_size))
            feature_grid = feature_map.squeeze(0).permute(1, 2, 0).cpu().numpy()

        valid = np.ones((self.grid_size, self.grid_size), dtype=bool)
        if ignored_bbox is not None:
            validate_bbox(ignored_bbox, *rgb.size)
            margin = PATCHCORE_EXCLUSION_MARGIN_PIXELS
            left = ignored_bbox.x1 * self.image_size / rgb.width - margin
            right = ignored_bbox.x2 * self.image_size / rgb.width + margin
            top = ignored_bbox.y1 * self.image_size / rgb.height - margin
            bottom = ignored_bbox.y2 * self.image_size / rgb.height + margin
            centers = (np.arange(self.grid_size, dtype=np.float32) + 0.5) * self.image_size / self.grid_size
            excluded_x = (centers >= left) & (centers <= right)
            excluded_y = (centers >= top) & (centers <= bottom)
            valid[np.ix_(excluded_y, excluded_x)] = False
        selected = feature_grid[valid]
        if not len(selected):
            raise ValueError("dustpan exclusion removed every PatchCore feature")
        return np.stack([l2_normalize(row) for row in selected]).astype(np.float32)


def build_memory_bank(feature_extractor: PatchCoreFeatures, images: list[Image.Image]) -> np.ndarray:
    if not images:
        raise ValueError("normal training images are required")
    return np.concatenate([feature_extractor.extract(image) for image in images], axis=0)


def save_memory_bank(path: Path, memory_bank: np.ndarray) -> None:
    bank = np.asarray(memory_bank, dtype=np.float32)
    if bank.ndim != 2 or not len(bank):
        raise ValueError("memory bank must be a non-empty 2D matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=bank)


def load_memory_bank(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        bank = np.asarray(archive["embeddings"], dtype=np.float32)
    if bank.ndim != 2 or not len(bank) or not np.isfinite(bank).all():
        raise ValueError(f"invalid PatchCore memory bank: {path}")
    if np.any(np.linalg.norm(bank, axis=1) == 0):
        raise ValueError(f"PatchCore memory bank contains a zero feature: {path}")
    return bank


def anomaly_score(features: np.ndarray, memory_bank: np.ndarray) -> float:
    query = np.asarray(features, dtype=np.float32)
    bank = np.asarray(memory_bank, dtype=np.float32)
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1] or not len(query) or not len(bank):
        raise ValueError("invalid query features or memory bank")
    if not np.isfinite(query).all() or not np.isfinite(bank).all():
        raise ValueError("query features and memory bank must be finite")
    # Maximum nearest-neighbor distance is used as the image-level anomaly score.
    nearest = np.sqrt(np.maximum(0, 2 - 2 * (query @ bank.T)).min(axis=1))
    return float(nearest.max())


def load_patchcore_feature_extractor(config: Settings = settings) -> PatchCoreFeatures:
    return PatchCoreFeatures(config.zone_encoder_weight_path, config.image_size)
