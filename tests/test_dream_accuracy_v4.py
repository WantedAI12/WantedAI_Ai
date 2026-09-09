from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "dream_accuracy_search_v4.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_router_improves_all_external_point_metrics_but_not_90_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "dream-accuracy-outcome-aware-search/v4"
    assert report["implementation"]["script_sha256"] == sha256(
        ROOT / "scripts" / "experiment_dream_accuracy_v4.py"
    )
    assert report["implementation"]["candidate_count"] == 4960
    assert report["point_pareto_candidates"] == 19
    selected = report["selected_diagnostic"]
    assert selected["name"] == "extra_trees_leaf15_blend0.35_exact10_router"
    assert selected["point_pareto"] is True
    assert all(selected["point_checks"].values())
    for split in ("test", "validation"):
        candidate = selected[split]
        baseline = report["baseline"][split]
        assert candidate["pearson"] > baseline["pearson"]
        assert candidate["spearman"] > baseline["spearman"]
        assert candidate["rmse"] < baseline["rmse"]
        assert candidate["mae"] < baseline["mae"]
    assert report["gates"]["point_pareto"]["passed"] is True
    assert report["gates"]["statistical_improvement"]["passed"] is False
    assert report["gates"]["human_ceiling_90_percent"]["passed"] is False
    assert report["gates"]["production"] == {
        "passed": False,
        "runtime_primary_score_weight": 0.0,
    }
    assert report["timing"]["eligible_for_model_selection_or_promotion"] is False
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False


def test_protocol_router_source_oof_is_better_but_small_and_post_selected() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    source_oof = report["source_group_oof_diagnostic"]
    assert source_oof["selection_used_test_and_validation_outcomes"] is True
    assert source_oof["exact10_router"]["n"] == 23
    assert source_oof["exact10_router"]["pearson"] > source_oof["exact10_baseline"][
        "pearson"
    ]
    assert source_oof["exact10_router"]["rmse"] < source_oof["exact10_baseline"][
        "rmse"
    ]
    inference = report["selected_post_selection_inference"]
    assert inference["post_selection_descriptive_only"] is True
    assert inference["human_ceiling_normalized_pearson"] < 0.90
    lower, upper = inference["human_ceiling_normalized_pearson_95_interval"]
    assert lower < 0.90
    assert upper > 0.90
