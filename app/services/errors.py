from __future__ import annotations


class EcoGuardError(Exception):
    def __init__(self, code: str, message: str, status_code: int, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}


def input_error(code: str, message: str, detail: dict | None = None) -> EcoGuardError:
    status = {"UPLOAD_TOO_LARGE": 413, "UNSUPPORTED_IMAGE_FORMAT": 415}.get(code, 400)
    return EcoGuardError(code, message, status, detail)


def system_error(code: str, message: str, detail: dict | None = None) -> EcoGuardError:
    status = 503 if code in {"MODEL_NOT_READY", "ZONE_MODEL_NOT_READY"} else 500
    return EcoGuardError(code, message, status, detail)
