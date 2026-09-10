from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_universal_intensity_threshold_v4 import (
    THRESHOLD_FEATURES,
    _attach_thresholds,
)


def test_threshold_features_include_activity_and_missingness():
    assert set(THRESHOLD_FEATURES) == {
        "log10_odor_threshold_ppm",
        "odor_threshold_missing",
        "odor_activity_log10_proxy",
    }


def test_attach_thresholds_keeps_missing_as_nan_with_flag():
    row = {
        "canonical_smiles": "A",
        "transport": np.asarray([0.0] * 16),
    }
    result = _attach_thresholds([row], {})[0]["threshold"]
    assert np.isnan(result[0])
    assert result[1] == 1.0
    assert np.isnan(result[2])


def test_published_threshold_v4_rejection_stays_weight_zero():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "universal_intensity_threshold_v4.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["retrospective_threshold_gate"]["passed"] is False
    assert value["prospective_external_gate"]["passed"] is False
    assert value["runtime"]["primary_score_weight"] == 0.0
