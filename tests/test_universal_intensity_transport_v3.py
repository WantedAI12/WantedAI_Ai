from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.build_universal_intensity_transport_v3 import (
    TRANSPORT_FEATURES,
    _finite_or_none,
)


def test_transport_features_separate_measurement_fallback_and_missingness():
    names = set(TRANSPORT_FEATURES)
    assert {
        "vapor_pressure_measured",
        "vapor_pressure_from_boiling",
        "vapor_pressure_missing",
        "boiling_point_missing",
        "headspace_log10_ppm_proxy",
    } <= names


def test_finite_or_none_rejects_nonfinite_without_inventing_zero():
    assert _finite_or_none(None) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert math.isclose(_finite_or_none("1.25") or 0.0, 1.25)


def test_published_transport_v3_rejection_stays_fail_closed():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "universal_intensity_transport_v3.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["retrospective_transport_gate"]["passed"] is False
    assert value["prospective_external_gate"]["passed"] is False
    assert value["runtime"]["primary_score_weight"] == 0.0
