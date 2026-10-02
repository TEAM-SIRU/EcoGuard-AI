from __future__ import annotations

import math


def calibrate_threshold(low_class_scores: list[float], high_class_scores: list[float], *, high_inclusive: bool = False) -> tuple[float, dict]:
    """Select a data-derived threshold separating the low-score and high-score validation classes."""
    if not low_class_scores or not high_class_scores:
        raise ValueError("both normal/positive and abnormal/negative validation examples are required")
    values = [float(value) for value in low_class_scores + high_class_scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("validation scores must be finite")
    low, high = min(values), max(values)
    candidates = [low - 1e-7, high + 1e-7]
    candidates.extend((left + right) / 2 for left, right in zip(sorted(set(values)), sorted(set(values))[1:]))
    best = max(candidates, key=lambda threshold: (_balanced_accuracy(low_class_scores, high_class_scores, threshold, high_inclusive), -threshold))
    return float(best), {
        "balanced_accuracy": _balanced_accuracy(low_class_scores, high_class_scores, best, high_inclusive),
        "high_class_comparison": ">=" if high_inclusive else ">",
        "low_class_count": len(low_class_scores),
        "high_class_count": len(high_class_scores),
        "low_class_range": [min(low_class_scores), max(low_class_scores)],
        "high_class_range": [min(high_class_scores), max(high_class_scores)],
    }


def calibrate_normal_threshold(normal_scores: list[float]) -> tuple[float, dict]:
    """Set a provisional threshold at the highest held-out normal score.

    This keeps every supplied normal validation sample below the anomaly
    threshold, but says nothing about anomaly recall until real anomalies are
    evaluated.
    """
    if not normal_scores:
        raise ValueError("normal validation examples are required")
    values = [float(value) for value in normal_scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("validation scores must be finite")
    threshold = max(values)
    return threshold, {
        "calibration_mode": "normal_only_provisional",
        "threshold_rule": "maximum_held_out_normal_score",
        "balanced_accuracy": None,
        "anomaly_detection_validated": False,
        "high_class_comparison": ">",
        "low_class_count": len(values),
        "high_class_count": 0,
        "low_class_range": [min(values), max(values)],
        "high_class_range": None,
        "normal_validation_pass_rate": 1.0,
    }


def _balanced_accuracy(low_class: list[float], high_class: list[float], threshold: float, high_inclusive: bool = False) -> float:
    if high_inclusive:
        low_correct = sum(score < threshold for score in low_class) / len(low_class)
        high_correct = sum(score >= threshold for score in high_class) / len(high_class)
    else:
        low_correct = sum(score <= threshold for score in low_class) / len(low_class)
        high_correct = sum(score > threshold for score in high_class) / len(high_class)
    return (low_correct + high_correct) / 2
