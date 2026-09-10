from __future__ import annotations

import json
from pathlib import Path

from scripts.build_bierling_intensity_calibration_v2 import (
    CANDIDATES,
    INNER_FOLDS,
    HINGE_KNOTS,
    OUTER_FOLDS,
    OUTER_REPEATS,
    TARGET_MAE,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_contract_is_repeated_nested_and_has_a_large_improvement_target():
    assert OUTER_REPEATS == 5
    assert OUTER_FOLDS == 5
    assert INNER_FOLDS == 4
    assert TARGET_MAE == 9.0
    assert HINGE_KNOTS == (20.0, 40.0, 60.0, 80.0)
    assert len(CANDIDATES) == 49
    assert len({row["name"] for row in CANDIDATES}) == len(CANDIDATES)


def test_published_v2_never_enables_runtime_when_present():
    path = ROOT / "benchmarks" / "bierling_2025_intensity_calibration_v2.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["development_timing"] == (
        "post_intensity_pilot_outcome_after_v1_review"
    )
    assert report["final_model"]["runtime_primary_score_weight"] == 0.0
    assert report["human_olfactory_90_percent_certified"] is False
    assert report["concentration_delta_validated"] is False
    assert report["large_improvement_gate"]["passed"] is True
    assert report["repeated_nested_crossfit"]["mae"] == 8.937125435542422
    assert report["repeated_nested_crossfit"]["spearman"] == 0.612721042117401
    assert report["relative_mae_reduction_vs_ravia"] > 0.30
    assert (
        report["final_model"]["parameters"][
            "portable_parity_maximum_absolute_error"
        ]
        <= 1e-10
    )
    assert report["bootstrap"]["ravia_minus_calibrated_mae_95_interval"][0] > 0
    assert (
        report["bootstrap"]["calibrated_minus_ravia_spearman_95_interval"][0]
        > 0
    )
