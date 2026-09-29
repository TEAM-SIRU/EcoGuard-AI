from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import Settings, settings
from app.ai.preprocessing import imagenet_normalized_tensor
from app.services.errors import system_error


class ResNet18Encoder:
    """Frozen ImageNet ResNet18 feature extractor; weights must already exist locally."""

    def __init__(self, weight_path: Path, image_size: int = settings.image_size):
        import torch
        from torchvision.models import resnet18

        if not weight_path.is_file():
            raise system_error("MODEL_NOT_READY", "Local ResNet18 ImageNet weights are missing", {"path": str(weight_path)})
        model = resnet18(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        model.load_state_dict(state, strict=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self._torch = torch
        self._model = model
        self.image_size = image_size

    def embed(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        tensor = imagenet_normalized_tensor(image, self.image_size)
        with torch.inference_mode():
            feature = self._model(tensor).squeeze(0).cpu().numpy()
        return l2_normalize(feature)


def load_encoder(config: Settings = settings) -> ResNet18Encoder:
    return ResNet18Encoder(config.zone_encoder_weight_path, config.image_size)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embedding must have a finite non-zero norm")
    return array / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = l2_normalize(np.asarray(left))
    b = l2_normalize(np.asarray(right))
    if a.shape != b.shape:
        raise ValueError("embedding dimensions do not match")
    return float(np.dot(a, b))


def create_reference_bank(encoder: ResNet18Encoder, images: list[Image.Image]) -> np.ndarray:
    if not images:
        raise ValueError("at least one reference image is required")
    return np.stack([encoder.embed(image) for image in images]).astype(np.float32)


def save_reference_bank(path: Path, embeddings: np.ndarray) -> None:
    bank = np.asarray(embeddings, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError("reference bank must be a non-empty 2D matrix")
    bank = np.stack([l2_normalize(row) for row in bank])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=bank)


def load_reference_bank(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        bank = np.asarray(archive["embeddings"], dtype=np.float32)
    if bank.ndim != 2 or bank.shape[0] == 0 or not np.isfinite(bank).all():
        raise ValueError(f"invalid reference bank: {path}")
    if np.any(np.linalg.norm(bank, axis=1) == 0):
        raise ValueError(f"reference bank contains a zero embedding: {path}")
    return bank


def max_reference_similarity(embedding: np.ndarray, bank: np.ndarray) -> float:
    normalized = l2_normalize(embedding)
    reference = np.stack([l2_normalize(row) for row in np.asarray(bank)])
    if reference.ndim != 2 or reference.shape[1] != normalized.shape[0]:
        raise ValueError("reference bank dimensions do not match")
    return float(np.max(reference @ normalized))


def validate_reference_threshold(positive_embeddings: np.ndarray, negative_embeddings: np.ndarray, bank: np.ndarray) -> tuple[float, dict]:
    from training.validation import calibrate_threshold

    positive_scores = [max_reference_similarity(item, bank) for item in positive_embeddings]
    negative_scores = [max_reference_similarity(item, bank) for item in negative_embeddings]
    threshold, summary = calibrate_threshold(negative_scores, positive_scores, high_inclusive=True)
    return threshold, {"positive_count": len(positive_scores), "negative_count": len(negative_scores), **summary}
