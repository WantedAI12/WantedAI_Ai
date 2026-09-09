from __future__ import annotations

from scripts.build_ma_2021_mixture_calibration_v2 import (
    CANDIDATES,
    CONTINUOUS_FEATURES,
    HINGE_FEATURES,
    _balanced_folds,
    _candidates,
    _portable_predict,
)


def test_ma_v2_candidate_contract_is_unique_and_portable():
    candidates = _candidates()
    names = [row["name"] for row in candidates]
    assert candidates == CANDIDATES
    assert len(names) == len(set(names))
    assert {row["algorithm"] for row in candidates} == {
        "median",
        "ridge",
        "huber",
        "quantile",
    }


def test_ma_v2_feature_contract_excludes_post_mixture_outcomes():
    names = set(CONTINUOUS_FEATURES) | set(HINGE_FEATURES)
    assert not {"IAB", "IAmix", "IBmix", "PAB", "Group", "Repeat"} & names
    assert "baseline_strongest_component" in names
    assert "r2_similarity" in names


def test_balanced_pair_folds_are_deterministic_and_complete():
    values = [f"pair-{index}" for index in range(23)]
    first = _balanced_folds(values, folds=5, salt="test")
    second = _balanced_folds(values, folds=5, salt="test")
    assert first.tolist() == second.tolist()
    assert set(first.tolist()) == {0, 1, 2, 3, 4}
    counts = [int((first == fold).sum()) for fold in range(5)]
    assert max(counts) - min(counts) <= 1


def test_portable_median_prediction_has_exact_parity():
    parameters = {
        "algorithm": "median",
        "candidate": {
            "name": "training_median_residual",
            "algorithm": "median",
            "feature_set": "none",
        },
        "intercept": -0.125,
        "feature_names": [],
    }
    rows = [{"pair_id": "a"}, {"pair_id": "b"}]
    assert _portable_predict(parameters, rows, ()).tolist() == [-0.125, -0.125]
