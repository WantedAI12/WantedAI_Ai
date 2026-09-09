from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_published_minnesota_scoring_adjudication_preserves_values_and_weight_zero():
    path = ROOT / "benchmarks" / "minnesota_intensity_blind_scoring_adjudication_v1.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    adjudication = value["adjudication"]
    assert adjudication["prediction_values_changed"] is False
    assert adjudication["outcome_values_changed"] is False
    assert adjudication["metric_weighting_changed_to_match_seal"] is True
    assert value["runtime_primary_score_weight"] == 0.0
    assert value["human_olfactory_90_percent_certified"] is False
