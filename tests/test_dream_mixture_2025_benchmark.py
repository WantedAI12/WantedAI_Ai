from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("rdkit")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from scripts.benchmark_dream_mixture_2025 import (
    MixtureRepresentation,
    _mixture_feature_names,
    metrics,
    pair_features,
    predict_portable_ridge,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mixture(offset: float, component_ids: set[int]) -> MixtureRepresentation:
    return MixtureRepresentation(
        component_ids=frozenset(component_ids),
        pom_mean=np.asarray([0.1, 0.4, 0.8]) + offset,
        pom_max=np.asarray([0.2, 0.5, 0.9]) + offset,
        pom_noisy_or=np.asarray([0.3, 0.6, 0.95]) + offset,
        rdkit_mean=np.asarray([1.0, 2.0, 3.0, 4.0]) + offset,
        rdkit_std=np.asarray([0.1, 0.2, 0.3, 0.4]) + offset,
        morgan_mean=np.asarray([0.0, 0.5, 1.0, 0.5]) + offset,
    )


def test_pair_features_are_exactly_symmetric_and_contract_width_matches() -> None:
    first = _mixture(0.0, {1, 2, 3})
    second = _mixture(0.05, {3, 4})
    forward = pair_features(first, second)
    reverse = pair_features(second, first)
    names = _mixture_feature_names(
        ["p0", "p1", "p2"], ["r0", "r1", "r2", "r3"]
    )
    assert np.array_equal(forward, reverse)
    assert len(forward) == len(names)
    assert np.isfinite(forward).all()


def test_metric_contract_uses_distance_error_and_rank() -> None:
    result = metrics([0.1, 0.4, 0.8, 0.9], [0.2, 0.3, 0.7, 1.0])
    assert result["n"] == 4
    assert abs(float(result["mae"]) - 0.1) < 1e-12
    assert abs(float(result["rmse"]) - 0.1) < 1e-12
    assert float(result["pearson"]) > 0.95
    assert float(result["spearman"]) == 1.0


def test_portable_ridge_executor_is_equivalent_and_fails_closed() -> None:
    names = ["first", "second"]
    runtime = {
        "schema": "dream-mixture-portable-ridge-research/v1",
        "feature_names": names,
        "feature_contract_sha256": hashlib.sha256(
            json.dumps(
                names,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "feature_mean": [1.0, 2.0],
        "feature_scale": [2.0, 4.0],
        "coefficients": [0.5, -0.25],
        "intercept": 0.4,
        "prediction_clip": [0.0, 1.0],
    }
    result = predict_portable_ridge(runtime, np.asarray([[3.0, 6.0]]), names)
    assert np.array_equal(result, np.asarray([0.65]))
    changed = dict(runtime)
    changed["feature_scale"] = [2.0, 0.0]
    with pytest.raises(ValueError, match="invalid numeric"):
        predict_portable_ridge(changed, np.asarray([[3.0, 6.0]]), names)
    with pytest.raises(ValueError, match="feature names"):
        predict_portable_ridge(runtime, np.asarray([[3.0, 6.0]]), list(reversed(names)))


def test_frozen_dream_report_preserves_improvement_and_release_boundary() -> None:
    report_path = ROOT / "benchmarks" / "dream_mixture_2025_retrospective_v1.json"
    runtime_path = ROOT / "benchmarks" / "dream_mixture_2025_research_runtime_v1.json"
    script_path = ROOT / "scripts" / "benchmark_dream_mixture_2025.py"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert report["schema"] == "dream-mixture-retrospective-v1"
    assert report["implementation"]["script_sha256"] == _sha256(script_path)
    assert report["runtime"]["sha256"] == _sha256(runtime_path)
    assert report["selection"]["selected"] == "ridge_pommix_pom_rdkit_30000"
    assert report["test"]["candidate"]["pearson"] > report["test"][
        "current_frozen_r2"
    ]["pearson"]
    assert report["validation"]["candidate"]["pearson"] > report["validation"][
        "fixed_public_top6_ensemble"
    ]["pearson"]
    assert report["validation"]["gate_passed"] is False
    assert report["test"]["gate_passed"] is False
    assert report["timing"]["development_used_test_or_validation_labels"] is True
    assert (
        report["timing"][
            "formal_candidate_ranking_used_test_or_validation_labels"
        ]
        is False
    )
    assert report["selection"]["development_outcome_aware"] is True
    assert report["release_gate"]["passed"] is False
    assert report["release_gate"]["runtime_primary_score_weight"] == 0.0
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False
    assert report["implementation"]["portable_runtime_equivalence_max_abs_error"] < 1e-12
    assert report["implementation"]["portable_runtime_rows_checked"] == 96
    assert runtime["allow_pickle"] is False
    assert runtime["runtime_primary_score_weight"] == 0.0
