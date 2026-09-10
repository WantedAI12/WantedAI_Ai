from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.research.headspace import predict_portable_log_vapor_pressure


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "opera_vapor_pressure_imputer_v1.json"
RUNTIME = ROOT / "benchmarks" / "opera_vapor_pressure_runtime_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opera_imputer_strict_scaffold_gate_fails_closed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    assert report["schema"] == "opera-vp-imputer-retrospective/v1"
    assert report["software"]["script_sha256"] == sha256(
        ROOT / "scripts" / "train_opera_vapor_pressure_imputer_v1.py"
    )
    assert report["runtime"]["sha256"] == sha256(RUNTIME)
    assert report["source"]["hub_sha256"] == sha256(
        ROOT / "benchmarks" / "headspace_sensory_hub_v1.db"
    )
    assert report["dataset"]["raw_observations"] == 2713
    assert report["dataset"]["unique_molecules"] == 2711
    assert report["dataset"]["strict_scaffold_test_molecules"] == 93
    assert report["dataset"]["strict_scaffold_overlap"] == 0
    assert report["dataset"]["scaffold_definition"] == (
        "bemis_murcko_ring_framework_with_all_acyclic_molecules_in_one_group"
    )
    strict = report["strict_scaffold_test"]["candidate"]
    assert strict["r2"] < 0.75
    assert strict["rmse_log10_mmhg"] > 1.5
    assert strict["absolute_error_q95_log10_mmhg"] > 2.5
    assert report["gates"]["research_imputer"]["passed"] is False
    assert report["gates"]["production"] == {
        "passed": False,
        "runtime_primary_score_weight": 0.0,
    }
    assert runtime["runtime_primary_score_weight"] == 0.0
    assert runtime["human_olfactory_90_percent_certified"] is False


def test_portable_opera_runtime_executes_and_rejects_contract_changes() -> None:
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    names = runtime["descriptor_names"]
    median = np.asarray(runtime["feature_median"], dtype=float)
    prediction = predict_portable_log_vapor_pressure(runtime, median, names)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="names"):
        predict_portable_log_vapor_pressure(runtime, median, [*names[:-1], "changed"])
    changed = json.loads(json.dumps(runtime))
    changed["feature_scale"][0] = 0.0
    with pytest.raises(ValueError, match="scale"):
        predict_portable_log_vapor_pressure(changed, median, names)
