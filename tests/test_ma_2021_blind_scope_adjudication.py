from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_published_ma_scope_adjudication_never_changes_metrics_or_authority():
    path = ROOT / "benchmarks" / "ma_2021_binary_mixture_blind_adjudication_v1.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    adjudication = value["adjudication"]
    assert adjudication["row_level_outcome_workbook_opened_before_seal"] is False
    assert adjudication["fully_outcome_naive_prospective_blind"] is False
    assert adjudication["metric_values_changed"] is False
    assert adjudication["gate_values_changed"] is False
    assert value["authoritative_gate_status"]["runtime_integration_authorized"] is False
