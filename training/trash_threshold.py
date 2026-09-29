from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrashValidationExample:
    expected_inside: bool
    all_centers_inside: bool
    minimum_overlap_ratio: float


def calibrate_trash_inside_threshold(examples: list[TrashValidationExample]) -> tuple[float, dict]:
    inside = [item for item in examples if item.expected_inside]
    outside = [item for item in examples if not item.expected_inside]
    if not inside or not outside:
        raise ValueError("both trash-inside and trash-outside validation images are required")
    ratios = [float(item.minimum_overlap_ratio) for item in examples]
    if any(not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("trash overlap ratios must be within [0, 1]")
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(sorted(set(ratios)), sorted(set(ratios))[1:]))
    def accuracy(threshold: float) -> float:
        correct_inside = sum(item.all_centers_inside and item.minimum_overlap_ratio >= threshold for item in inside) / len(inside)
        correct_outside = sum(not (item.all_centers_inside and item.minimum_overlap_ratio >= threshold) for item in outside) / len(outside)
        return (correct_inside + correct_outside) / 2
    threshold = max(candidates, key=lambda item: (accuracy(item), item))
    return float(threshold), {
        "balanced_accuracy": accuracy(threshold),
        "inside_count": len(inside),
        "outside_count": len(outside),
        "threshold_rule": "all trash bbox centers inside and minimum trash-area overlap ratio >= threshold",
    }
