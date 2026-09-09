from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "dream_set_encoder_search_v5.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attention_set_encoder_is_reproducibly_rejected_below_v4() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "dream-attention-set-outcome-aware-search/v5"
    assert report["implementation"]["script_sha256"] == sha256(
        ROOT / "scripts" / "experiment_dream_set_encoder_v5.py"
    )
    assert report["source"]["v4_report_sha256"] == sha256(
        ROOT / "benchmarks" / "dream_accuracy_search_v4.json"
    )
    assert report["status"] == "attention_set_encoder_rejected_below_v4"
    assert len(report["candidates"]) == 4
    assert report["gates"]["point_pareto_above_v4"]["passed"] is False
    assert report["gates"]["production"] == {
        "passed": False,
        "runtime_primary_score_weight": 0.0,
    }
    selected = report["selected_diagnostic"]
    assert selected["v4_point_pareto"] is False
    assert not any(selected["v4_point_checks"].values())
    assert report["timing"]["eligible_for_selection_or_promotion"] is False
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False
