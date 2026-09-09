"""An incremental audit may reuse passes, never erase failures or inputs."""

import copy
import importlib.util
from pathlib import Path

import pytest


def load_summary():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_request_refinement.py"
    spec = importlib.util.spec_from_file_location("refinement_summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reports():
    cases = [{"id": f"case-{i}", "brief": f"same request {i}", "before_score": None if i == 0 else 85. if i == 1 else 90.,
              "after_score": None if i == 0 else 85. if i == 1 else 90., "full_profile_target_met": i > 1,
              "after_formula_id": f"formula-{i}", "gain_points": None if i == 0 else 0.} for i in range(400)]
    base = {"target_unchanged": 90, "experimental_disable_safety": False, "regressions": 0, "case_count": 400,
            "catalog_scope": "conditional_registry_prototype", "full_profile_90_case_count": 398,
            "source_sha256": {"fragrance_ai/recommender/profile_match.py": "same", "fragrance_ai/recommender/dose_refinement.py": "before"},
            "cases": cases}
    old = copy.deepcopy(base)
    new = copy.deepcopy(base)
    new["cases"] = new["cases"][:12]
    new["case_count"] = 12
    new["source_sha256"]["fragrance_ai/recommender/dose_refinement.py"] = "after"
    new["cases"][1].update(after_score=95., full_profile_target_met=True, gain_points=10., after_formula_id="improved")
    return base, new, old


def test_provenance_and_denominator_are_kept():
    result = load_summary().combine(*reports())
    assert result["case_count"] == 400 and result["full_profile_90_case_count"] == 399
    assert result["unscorable_cases"] == ["case-0"]
    assert result["validation"]["current_source_actual_requests"] == 12
    assert len(result["validation"]["carried_successful_paths"]) == 388
    assert not result["validation"]["all_400_freshly_rerun_with_current_source"]


@pytest.mark.parametrize("removed", [0, 1])
def test_failed_or_undefined_cases_cannot_be_carried_forward(removed):
    base, new, old = reports()
    new["cases"].pop(removed)
    new["case_count"] -= 1
    with pytest.raises(ValueError, match="failed or unscorable"):
        load_summary().combine(base, new, old)


@pytest.mark.parametrize("change", ["case_drop", "scorer", "regression", "control", "nonfinite"])
def test_invalid_incremental_evidence_is_rejected(change):
    base, new, old = reports()
    if change == "case_drop":
        base["cases"].pop()
    elif change == "scorer":
        new["source_sha256"]["fragrance_ai/recommender/profile_match.py"] = "different score"
    elif change == "regression":
        new["cases"][1]["after_score"] = 80.
    elif change == "control":
        new["cases"][2]["after_formula_id"] = "changed control"
    else:
        new["cases"][1]["after_score"] = float("nan")
    with pytest.raises(ValueError):
        load_summary().combine(base, new, old)
