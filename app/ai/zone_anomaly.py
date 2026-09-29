from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.zone_recognition import l2_normalize
from app.ai.preprocessing import imagenet_normalized_tensor
from app.core.config import Settings, settings


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
        self._model = torch.nn.Sequential(*list(model.children())[:-2]).eval()
        self.image_size = image_size

    def extract(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        tensor = imagenet_normalized_tensor(image, self.image_size)
        with torch.inference_mode():
            feature = self._model(tensor).squeeze(0).permute(1, 2, 0).reshape(-1, 512).cpu().numpy()
        return np.stack([l2_normalize(row) for row in feature]).astype(np.float32)


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
