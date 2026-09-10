from __future__ import annotations

import json
from pathlib import Path

from scripts.build_bierling_intensity_crossfit_calibration import (
    CANDIDATES,
    INNER_FOLDS,
    OUTER_FOLDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_intensity_crossfit_calibration_contract_is_nested_and_portable():
    assert OUTER_FOLDS == 5
    assert INNER_FOLDS == 4
    assert len(CANDIDATES) == 20
    assert len({row["name"] for row in CANDIDATES}) == len(CANDIDATES)


def test_published_crossfit_calibration_never_enables_runtime_when_present():
    path = ROOT / "benchmarks" / "bierling_2025_intensity_crossfit_calibration_v1.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["development_timing"] == "post_intensity_pilot_outcome"
    assert (
        report["portable_diagnostic_calibrator"]["runtime_primary_score_weight"]
        == 0.0
    )
    assert report["human_olfactory_90_percent_certified"] is False
    assert report["concentration_delta_validated"] is False
    assert report["release_gate"]["passed"] is False
    assert report["nested_crossfit"]["spearman"] == 0.5360419983047057
    assert report["nested_crossfit"]["mae"] == 10.685287854006807
    assert (
        report["bootstrap"]["calibrated_minus_ravia_spearman_95_interval"][0]
        > 0.0
    )
    assert report["bootstrap"]["ravia_minus_calibrated_mae_95_interval"][0] < 0.0
