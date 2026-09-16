"""Stable nullable-number wire contract, separate from evidence-level labels."""
from __future__ import annotations

import math
from numbers import Real


def confidence_fields(value):
    """Never invent a probability from a label, score, or missing observation.

    JSON null is compatible with a nullable Java Double. A backend requiring
    a non-null probability must handle unavailable estimates explicitly.
    """
    if isinstance(value, str):
        return {"confidence": None, "confidence_kind": value or "unavailable"}
    if value is None:
        return {"confidence": None, "confidence_kind": "unavailable"}
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number in [0, 1], an evidence label, or null")
    return {"confidence": float(value), "confidence_kind": "numeric_estimate_calibration_unspecified"}
