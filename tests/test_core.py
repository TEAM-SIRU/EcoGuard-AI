from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.ai.region_masker import BoundingBox, crop_dustpan, create_stair_view
from app.ai.zone_recognition import cosine_similarity, l2_normalize
from app.core.config import settings
from app.services.errors import EcoGuardError
from app.services.evaluator import FinalEvaluator
from app.services.zone_registry import ZoneRegistry
from app.utils.image import validate_image
from training.validation import calibrate_threshold


def png_bytes(size=(20, 12)):
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_config_load_defaults():
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_image_pixels == 20_000_000
    assert settings.max_inference_concurrency == 1


def test_image_validator_accepts_jpeg_and_png_and_rejects_invalid():
    result = validate_image(png_bytes())
    assert result.format == "PNG"
    assert result.image.size == (20, 12)
    with pytest.raises(EcoGuardError, match="Image cannot be decoded"):
        validate_image(b"not an image")


def test_image_validator_enforces_upload_and_pixel_limits():
    small_limit = replace(settings, max_upload_bytes=2)
    with pytest.raises(EcoGuardError) as too_large:
        validate_image(png_bytes(), small_limit)
    assert too_large.value.code == "UPLOAD_TOO_LARGE"
    pixel_limit = replace(settings, max_image_pixels=100)
    with pytest.raises(EcoGuardError) as too_many_pixels:
        validate_image(png_bytes(), pixel_limit)
    assert too_many_pixels.value.code == "INVALID_IMAGE"


def test_region_split_masks_stairs_and_crops_dustpan():
    image = Image.new("RGB", (20, 12), "white")
    box = BoundingBox(2, 2, 10, 10)
    stairs = create_stair_view(image, box)
    pan = crop_dustpan(image, box)
    assert stairs.size == image.size
    assert stairs.getpixel((4, 4)) == (0, 0, 0)
    assert stairs.getpixel((0, 0)) == (255, 255, 255)
    assert pan.size == (8, 8)
    with pytest.raises(EcoGuardError) as invalid:
        create_stair_view(image, BoundingBox(-1, 0, 3, 3))
    assert invalid.value.code == "INVALID_DETECTION_RESULT"


def test_l2_normalize_and_cosine_similarity():
    np.testing.assert_allclose(l2_normalize(np.array([3.0, 4.0])), [0.6, 0.8])
    assert cosine_similarity(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_threshold_is_derived_from_validation_scores():
    threshold, summary = calibrate_threshold([0.1, 0.2], [0.8, 0.9])
    assert 0.2 <= threshold < 0.8
    assert summary["balanced_accuracy"] == 1.0
    with pytest.raises(ValueError):
        calibrate_threshold([], [0.7])


def test_final_evaluator_owns_pass_fail_decision():
    passed = FinalEvaluator.evaluate(zone_id="z", checkpoint_id="c", user_id=None,
                                     analyses={"dustpan_detected": True, "zone_recognized": True,
                                               "zone_anomaly": False, "trash_detected": True})
    failed = FinalEvaluator.evaluate(zone_id="z", checkpoint_id="c", user_id="u",
                                     analyses={"dustpan_detected": False}, failures=["DUSTPAN_NOT_FOUND"])
    assert passed.decision == "PASS" and passed.is_passed
    assert failed.decision == "FAIL" and failed.fail_reasons == ["DUSTPAN_NOT_FOUND"]


def test_zone_registry_load_and_unknown_ids(tmp_path):
    registry_file = tmp_path / "zones.yaml"
    registry_file.write_text("zones:\n  zone_one:\n    active: true\n    checkpoints:\n      level_a:\n        zone_threshold: null\n        anomaly_threshold: null\n", encoding="utf-8")
    config = replace(settings, model_dir=tmp_path / "models")
    registry = ZoneRegistry.load(registry_file, config)
    assert registry.get_checkpoint("zone_one", "level_a").zone_id == "zone_one"
    with pytest.raises(EcoGuardError) as unknown_zone:
        registry.get_checkpoint("missing", "level_a")
    assert unknown_zone.value.code == "UNKNOWN_ZONE"
    with pytest.raises(EcoGuardError) as unknown_checkpoint:
        registry.get_checkpoint("zone_one", "missing")
    assert unknown_checkpoint.value.code == "UNKNOWN_CHECKPOINT"


def test_module_imports_without_model_assets():
    import app.main
    import training.run_pipeline
    import training.train_dustpan
    import training.build_zone_references
    import training.build_patchcore
    assert app.main.app.title == "EcoGuard AI"


def test_zone_assets_missing_returns_zone_model_not_ready(tmp_path):
    from app.services.inference import CleaningInference
    from app.services.zone_registry import CheckpointAssets
    checkpoint = CheckpointAssets("z", "c", tmp_path / "reference_embeddings", tmp_path / "patchcore", None, None, {})
    with pytest.raises(EcoGuardError) as missing:
        CleaningInference.zone_readiness(checkpoint)
    assert missing.value.code == "ZONE_MODEL_NOT_READY"

def test_training_orchestrator_reports_missing_dataset_without_training():
    from training.dataset_validator import DatasetNotReady
    from training.run_pipeline import run
    with pytest.raises(DatasetNotReady) as missing:
        run("all")
    assert any("no images found" in item for item in missing.value.missing)


def test_reference_bank_round_trip_and_add_new_checkpoint_without_training(tmp_path):
    from app.ai.zone_recognition import create_reference_bank, load_reference_bank, save_reference_bank
    class FixtureEncoder:
        def embed(self, image):
            return np.array([float(image.getpixel((0, 0))[0] + 1), 1.0], dtype=np.float32)
    encoder = FixtureEncoder()
    bank = create_reference_bank(encoder, [Image.new("RGB", (2, 2), (1, 0, 0)), Image.new("RGB", (2, 2), (2, 0, 0))])
    destination = tmp_path / "new_zone" / "reference_bank.npz"
    save_reference_bank(destination, bank)
    loaded = load_reference_bank(destination)
    assert loaded.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(loaded, axis=1), [1.0, 1.0])

def test_patchcore_memory_bank_round_trip_and_score(tmp_path):
    from app.ai.zone_anomaly import anomaly_score, build_memory_bank, load_memory_bank, save_memory_bank
    class FixtureFeatures:
        def extract(self, image):
            value = float(image.getpixel((0, 0))[0]) / 255
            return np.array([[value, 1 - value]], dtype=np.float32)
    bank = build_memory_bank(FixtureFeatures(), [Image.new("RGB", (1, 1), (0, 0, 0)), Image.new("RGB", (1, 1), (5, 0, 0))])
    destination = tmp_path / "patchcore" / "memory_bank.npz"
    save_memory_bank(destination, bank)
    loaded = load_memory_bank(destination)
    assert loaded.shape == (2, 2)
    assert anomaly_score(np.array([[0.0, 1.0]], dtype=np.float32), loaded) < anomaly_score(np.array([[1.0, 0.0]], dtype=np.float32), loaded)

def test_threshold_metadata_round_trip(tmp_path):
    from training.metadata import load_metadata, save_threshold
    path = tmp_path / "threshold.json"
    save_threshold(path, name="zone", value=0.72, validation_summary={"balanced_accuracy": 0.9})
    metadata = load_metadata(path)
    assert metadata["values"]["value"] == pytest.approx(0.72)

def test_training_cli_accepts_component_and_scope():
    from training.run_pipeline import build_parser
    args = build_parser().parse_args(["--component", "patchcore", "--zone-id", "zone_a", "--checkpoint-id", "cp_1", "--force"])
    assert (args.component, args.zone_id, args.checkpoint_id, args.force) == ("patchcore", "zone_a", "cp_1", True)


def _prepare_yolo_dataset(root):
    for split in ("train", "val"):
        image_dir = root / "raw" / "dustpan" / "images" / split
        label_dir = root / "raw" / "dustpan" / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(image_dir / "sample.JPG", format="JPEG")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

def test_dataset_validator_builds_yaml_and_accepts_uppercase_images(tmp_path):
    from training.dataset_validator import validate_yolo_dataset
    root = tmp_path / "datasets"
    _prepare_yolo_dataset(root)
    result = validate_yolo_dataset(replace(settings, dataset_dir=root))
    assert len(result.train_images) == 1
    assert "0: dustpan" in result.yaml_path.read_text(encoding="utf-8")

def test_dataset_validator_rejects_invalid_yolo_classes(tmp_path):
    from training.dataset_validator import InvalidDataset, validate_yolo_dataset
    root = tmp_path / "datasets"
    _prepare_yolo_dataset(root)
    (root / "raw/dustpan/labels/train/sample.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(InvalidDataset, match="class id must be 0"):
        validate_yolo_dataset(replace(settings, dataset_dir=root))


def test_zone_readiness_rejects_nan_thresholds(tmp_path):
    from app.services.inference import CleaningInference
    from app.services.zone_registry import CheckpointAssets
    import json
    checkpoint_dir = tmp_path / "zone" / "checkpoint"
    reference = checkpoint_dir / "reference_embeddings"
    patchcore = checkpoint_dir / "patchcore"
    reference.mkdir(parents=True)
    patchcore.mkdir(parents=True)
    (reference / "reference_bank.npz").write_bytes(b"fixture")
    (patchcore / "memory_bank.npz").write_bytes(b"fixture")
    metadata = checkpoint_dir / "metadata.json"
    metadata.write_text(json.dumps({"schema_version": 1, "values": {"zone": {"zone_threshold": float("nan")}, "patchcore": {"anomaly_threshold": 0.2}}}), encoding="utf-8")
    cp = CheckpointAssets("z", "c", reference, patchcore, None, None, {}, metadata)
    with pytest.raises(EcoGuardError) as invalid:
        CleaningInference.zone_readiness(cp)
    assert invalid.value.code == "ZONE_MODEL_NOT_READY"

def test_patchcore_memory_loader_rejects_non_finite_values(tmp_path):
    from app.ai.zone_anomaly import load_memory_bank
    path = tmp_path / "bad.npz"
    np.savez(path, embeddings=np.array([[float("nan"), 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="invalid PatchCore memory bank"):
        load_memory_bank(path)


def test_config_rejects_zero_inference_concurrency():
    with pytest.raises(ValueError, match="must be positive"):
        replace(settings, max_inference_concurrency=0)


def test_zone_threshold_validation_matches_inference_greater_or_equal_rule():
    threshold, summary = calibrate_threshold([0.3, 0.5], [0.5, 0.9], high_inclusive=True)
    assert summary["high_class_comparison"] == ">="
    assert summary["balanced_accuracy"] == 0.75
    assert 0.3 <= threshold <= 0.5
