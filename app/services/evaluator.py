from __future__ import annotations

from app.schemas.cleaning import CleaningResult, Decision


class FinalEvaluator:
    """The only place that converts model analyses into PASS/FAIL."""

    @staticmethod
    def evaluate(*, zone_id: str, checkpoint_id: str, user_id: str | None, analyses: dict, failures: list[str] | None = None) -> CleaningResult:
        reasons = list(dict.fromkeys(failures or []))
        if not reasons:
            if not analyses.get("dustpan_detected", False):
                reasons.append("DUSTPAN_NOT_FOUND")
            if not analyses.get("zone_recognized", False):
                reasons.append("ZONE_NOT_RECOGNIZED")
            if analyses.get("zone_anomaly", True):
                reasons.append("ZONE_ANOMALY_DETECTED")
            if not analyses.get("trash_detected", False):
                reasons.append("TRASH_NOT_FOUND_IN_DUSTPAN")
            if analyses.get("trash_outside", False):
                reasons.append("TRASH_OUTSIDE_DUSTPAN")
        passed = not reasons
        return CleaningResult(
            decision=Decision.PASS if passed else Decision.FAIL,
            is_passed=passed,
            fail_reasons=reasons,
            zone_id=zone_id,
            checkpoint_id=checkpoint_id,
            user_id=user_id,
            analysis=analyses,
        )
