from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "dream_headspace_retrospective_v3.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_headspace_dream_diagnostic_is_source_bound_and_rejected() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "dream-headspace-retrospective/v3"
    assert report["implementation"]["script_sha256"] == sha256(
        ROOT / "scripts" / "benchmark_dream_headspace_v3.py"
    )
    assert report["source"]["headspace_hub_sha256"] == sha256(
        ROOT / "benchmarks" / "headspace_sensory_hub_v1.db"
    )
    assert report["source"]["concentration_calibration_sha256"] == sha256(
        ROOT / "benchmarks" / "concentration_headspace_calibration_v1.json"
    )
    assert report["physical_model"]["component_evidence"] == {
        "measured_boiling_point_trouton_fallback": 38,
        "measured_opera_vapor_pressure_median": 120,
        "missing": 46,
    }
    assert report["physical_model"]["unique_components"] == 204
    assert report["physical_model"]["resolved_components"] == 158
    assert report["physical_model"]["absolute_headspace_claimed"] is False
    assert report["status"] == "headspace_candidate_rejected_not_point_pareto"
    assert report["selection"]["point_pareto_candidates"] == 0
    assert report["gates"]["point_pareto"]["passed"] is False
    assert report["gates"]["statistical_improvement"]["passed"] is False
    assert report["gates"]["human_ceiling_90_percent"]["passed"] is False
    assert report["gates"]["production"] == {
        "passed": False,
        "runtime_primary_score_weight": 0.0,
    }


def test_nested_training_only_residual_still_fails_external_pareto_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    residual = report["training_selection"]["pair_fixed_plus_headspace_residual"]
    assert residual["baseline_prediction_is_source_group_oof"] is True
    assert residual["residual_selection_is_nested_source_group_oof"] is True
    assert residual["outer_source_is_excluded_from_inner_residual_targets"] is True
    assert residual["selected_scale"]["alpha"] == 100000.0
    assert residual["selected_scale"]["scale"] == 1.0
    candidate = next(
        row
        for row in report["candidate_search"]
        if row["name"] == "pair_fixed_plus_headspace_residual"
    )
    assert candidate["point_pareto"] is False
    assert candidate["test"]["rmse"] < report["current_pair_v2"]["test"]["rmse"]
    assert candidate["validation"]["rmse"] > report["current_pair_v2"]["validation"][
        "rmse"
    ]
    selected = report["selection"]["selected"]
    assert selected["name"] == "pair_fixed_plus_headspace_residual"
    assert selected["test"]["rmse"] < report["current_pair_v2"]["test"]["rmse"]
    assert selected["validation"]["rmse"] > report["current_pair_v2"]["validation"][
        "rmse"
    ]
    assert report["claim_boundary"]["quantitative_mixture_composition_available"] is False
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False
