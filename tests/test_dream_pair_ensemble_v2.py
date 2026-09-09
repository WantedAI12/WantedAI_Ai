from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ogb")
pytest.importorskip("pandas")
pytest.importorskip("rdkit")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from scripts.benchmark_dream_pair_ensemble_v2 import (
    PAIR_GENERATED_EMBEDDING_SHA256,
    PAIR_WEIGHTS_SHA256,
    _pair_feature_names,
    pair_embedding_features,
    predict_portable_pair_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_hash(names: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            names,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_odor_pair_features_are_finite_and_exactly_symmetric() -> None:
    first = np.linspace(-1.0, 1.0, 128)
    second = np.linspace(0.75, -0.25, 128)
    forward = pair_embedding_features(first, second)
    reverse = pair_embedding_features(second, first)
    assert np.array_equal(forward, reverse)
    assert forward.shape == (len(_pair_feature_names()),)
    assert len(forward) == 260
    assert np.isfinite(forward).all()


def test_portable_pair_ensemble_executes_and_rejects_bad_weights() -> None:
    names = ["first", "second"]
    contract = _contract_hash(names)
    runtime = {
        "schema": "dream-pair-ensemble-portable-ridge/v2",
        "feature_names": names,
        "feature_contract_sha256": contract,
        "prediction_clip": [0.0, 1.0],
        "members": [
            {
                "weight": 0.6,
                "feature_mean": [1.0, 2.0],
                "feature_scale": [2.0, 4.0],
                "coefficients": [0.5, -0.25],
                "intercept": 0.4,
            },
            {
                "weight": 0.4,
                "feature_mean": [1.0, 2.0],
                "feature_scale": [2.0, 4.0],
                "coefficients": [0.2, 0.1],
                "intercept": 0.1,
            },
        ],
    }
    prediction = predict_portable_pair_ensemble(
        runtime, np.asarray([[3.0, 6.0]]), names
    )
    assert np.allclose(prediction, [0.55], atol=1e-15)
    changed = json.loads(json.dumps(runtime))
    changed["members"][0]["weight"] = 0.599999
    with pytest.raises(ValueError, match="weights"):
        predict_portable_pair_ensemble(changed, np.asarray([[3.0, 6.0]]), names)


def test_frozen_pair_ensemble_report_preserves_claim_boundary() -> None:
    report_path = ROOT / "benchmarks" / "dream_pair_ensemble_retrospective_v2.json"
    runtime_path = (
        ROOT / "benchmarks" / "dream_pair_ensemble_research_runtime_v2.json"
    )
    script_path = ROOT / "scripts" / "benchmark_dream_pair_ensemble_v2.py"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert report["implementation"]["script_sha256"] == _sha256(script_path)
    assert report["runtime"]["sha256"] == _sha256(runtime_path)
    assert report["source"]["odor_pair"]["weights_sha256"] == PAIR_WEIGHTS_SHA256
    assert (
        report["source"]["odor_pair"]["generated_embedding_rows_sha256"]
        == PAIR_GENERATED_EMBEDDING_SHA256
    )
    for split in ("test", "validation"):
        candidate = report[split]["candidate"]
        current = report[split]["current"]
        assert candidate["pearson"] > current["pearson"]
        assert candidate["spearman"] > current["spearman"]
        assert candidate["rmse"] < current["rmse"]
        assert candidate["mae"] < current["mae"]
    assert report["timing"]["development_used_test_and_validation_outcomes"] is True
    assert report["timing"]["post_selection_intervals_descriptive_only"] is True
    assert report["timing"]["inferentially_valid_for_promotion"] is False
    assert len(report["selection"]["weight_search"]) == 21
    assert report["gates"]["point_pareto"]["passed"] is True
    assert report["gates"]["statistical_improvement"]["passed"] is False
    assert report["gates"]["human_ceiling_90_percent"]["passed"] is False
    assert report["gates"]["production"]["passed"] is False
    assert report["gates"]["production"]["runtime_primary_score_weight"] == 0.0
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False
    assert runtime["allow_pickle"] is False
    assert runtime["runtime_primary_score_weight"] == 0.0
