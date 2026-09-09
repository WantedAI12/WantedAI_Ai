from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_published_minnesota_acquisition_adjudication_preserves_data_and_scoring():
    path = ROOT / "benchmarks" / "minnesota_intensity_blind_outcome_acquisition_v1.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    adjudication = value["acquisition_adjudication"]
    assert adjudication["repository_actual_rows"] == {
        "original": 820,
        "retest": 250,
        "new_recruits": 560,
    }
    assert adjudication["prediction_rows_changed"] is False
    assert adjudication["outcome_rows_changed"] is False
    assert adjudication["scoring_contract_changed"] is False
