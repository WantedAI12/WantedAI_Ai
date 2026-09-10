import hashlib
import json
from pathlib import Path

import pytest
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "bushdid_accuracy_v4.json"
MODEL = ROOT / "benchmarks" / "bushdid_accuracy_v4_catboost.json"
SCRIPT = ROOT / "scripts" / "experiment_bushdid_accuracy_v4.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bushdid_accuracy_v4_artifacts_and_metrics_are_bound():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "bushdid-outcome-aware-accuracy-v4"
    assert report["status"] == "outcome_aware_retrospective_development"
    assert report["implementation"]["script_sha256"] == _sha256(SCRIPT)
    assert report["artifact"]["model_sha256"] == _sha256(MODEL)
    assert report["artifact"]["model_bytes"] == MODEL.stat().st_size
    assert report["artifact"]["format"] == "catboost_json_no_pickle"
    assert report["artifact"]["runtime_primary_score_weight"] == 0.0
    assert report["development_contract"][
        "post_selection_intervals_descriptive_only"
    ] is True
    assert "more than 100 CatBoost configurations" in report[
        "development_contract"
    ]["selection_disclosure"]

    result = report["results"]["crossfit_compact_candidate"]
    assert result["absolute_accuracy_percent"] == pytest.approx(89.57880716951861)
    assert result["spearman"] == pytest.approx(0.665136791076642)
    assert result["human_ceiling_normalized_spearman"] == pytest.approx(
        0.7775502503084268
    )
    assert result["absolute_accuracy_percent"] > report["results"][
        "crossfit_component_overlap"
    ]["absolute_accuracy_percent"]
    assert report["results"]["bootstrap"][
        "candidate_minus_overlap_accuracy_95_interval"
    ][0] > 0.0
    assert report["target_gate"]["passed"] is False
    assert report["target_gate"]["absolute_accuracy_point_at_least_90"] is False
    assert report["target_gate"]["normalized_rank_point_at_least_0_90"] is False
    assert report["results"]["bootstrap"][
        "post_selection_model_uncertainty_included"
    ] is False


def test_bushdid_accuracy_v4_model_is_loadable_json():
    model = CatBoostRegressor()
    model.load_model(MODEL, format="json")
    assert model.tree_count_ == 600
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["development_contract"]["feature_count"] == 24
    assert len(report["development_contract"]["feature_names"]) == 24
    assert len(report["development_contract"]["fold_rows"]) == 5
