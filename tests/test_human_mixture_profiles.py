"""Synthetic contract tests; scientific performance comes only from real runs."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.research.perception_validation import (
    choose_ridge, cluster_interval, fit_ridge, group_folds, predict_ridge,
    paired_profile_comparison, profile_errors, summarize_profiles, wilson_lower,
)


SPEC = importlib.util.spec_from_file_location("human_profile_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_human_mixture_profiles.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_group_folds_are_balanced_stable_and_grouped():
    groups = [str(i) for i in range(20)] + ["3", "3", "4"]
    folds = group_folds(groups)
    assert len(set(folds[np.asarray(groups) == "3"])) == 1
    assert np.array_equal(folds[::-1], group_folds(groups[::-1]))
    for fold in range(5):
        assert len({group for group, value in zip(groups, folds) if value == fold}) == 4


def test_group_folds_reject_pseudoreplication():
    with pytest.raises(ValueError):
        group_folds(["same"] * 100)


def test_ridge_portable_replay_and_feature_contract():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 7))
    y = np.abs(rng.normal(size=(30, 3)))
    model = fit_ridge(x, y, 10)
    assert np.array_equal(predict_ridge(model, x), predict_ridge(json.loads(json.dumps(model)), x))
    model["scale"][0] = 0
    with pytest.raises(ValueError):
        predict_ridge(model, x)


def test_dual_ridge_matches_primal_formula():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(7, 20)), rng.normal(size=(7, 3))
    model = fit_ridge(x, y, 3)
    z = (x - x.mean(0)) / x.std(0)
    expected = np.linalg.solve(z.T @ z + 3 * np.eye(20), z.T @ (y - y.mean(0)))
    assert np.allclose(model["coefficients"], expected, atol=1e-12)


def test_cv_uses_only_supplied_training_data():
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(20, 3)), np.abs(rng.normal(size=(20, 4)))
    _, selection = choose_ridge(x, y, [str(i // 2) for i in range(20)], alphas=(1.0, 10.0))
    assert selection["group_count"] == 10
    assert selection["selection_data"] == "training_only"
    assert len(selection["candidates"]) == 2


def test_missing_predictions_remain_in_success_denominator():
    result = summarize_profiles(np.array([[1, 0], [np.nan, np.nan]]), np.array([[1, 0], [0, 1]]), ["a", "b"])
    tolerance = result["diagnostic_tolerance"]
    assert tolerance["pass_rate_percent"] == 50
    assert tolerance["all_requested_profiles"] == 2
    assert not tolerance["actual_human_accuracy_90_authorized"]


def test_zero_profiles_cannot_manufacture_perfect_accuracy():
    result = summarize_profiles(np.zeros((100, 3)), np.zeros((100, 3)), [str(i) for i in range(100)])
    assert result["diagnostic_tolerance"]["successes"] == 0
    assert result["cosine_distance"]["mean"] is None
    assert result["mae"]["mean"] == 0


def test_duplicate_profiles_do_not_inflate_binomial_confidence():
    result = summarize_profiles(np.tile([1, 0], (100, 1)), np.tile([1, 0], (100, 1)), ["one-mixture"] * 100)
    assert result["diagnostic_tolerance"]["independent_composition_groups"] == 1
    assert result["diagnostic_tolerance"]["composition_wilson_one_sided_95_lower"] < 0.3
    assert result["mae"]["composition_bootstrap_95"] is None


def test_ninety_of_hundred_is_not_a_ninety_percent_lower_bound():
    assert 0.83 < wilson_lower(90, 100) < 0.85
    assert wilson_lower(100, 100) > 0.97


@pytest.mark.parametrize("successes,total", [(2, 1), (-1, 10), (0, 0)])
def test_invalid_binomial_counts_fail(successes, total):
    with pytest.raises(ValueError):
        wilson_lower(successes, total)


def test_cluster_interval_is_finite_and_not_fake_zero_with_one_group():
    assert cluster_interval(np.array([1.0, 3.0]), ["a", "a"]) is None
    interval = cluster_interval(np.array([1.0, 3.0]), ["a", "b"])
    assert interval == [1.0, 3.0]


def test_cosine_and_absolute_error_are_not_conflated():
    error = profile_errors(np.array([[2, 0, 1]]), np.array([[4, 0, 2]]))
    assert error["cosine_distance"][0] < 1e-12
    assert error["mae"][0] == 1
    assert "accuracy" not in error


def test_paired_comparison_uses_identical_rows_and_handles_undefined_cosine():
    baseline = np.array([[1.0, 0], [np.nan, np.nan], [0, 0]])
    candidate = np.array([[1.0, 0], [100, 100], [0, 0]])
    target = np.array([[1.0, 0], [1, 0], [0, 1]])
    result = paired_profile_comparison(baseline, candidate, target, ["a", "b", "c"])
    assert result["paired_profiles"] == 2
    assert result["paired_cosine_defined_profiles"] == 1
    assert result["baseline_mae_minus_candidate_mae"] == 0
    undefined = paired_profile_comparison(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), ["a", "b"])
    assert undefined["baseline_cosine_distance_minus_candidate"] is None
    json.dumps(undefined, allow_nan=False)


def test_stock_identity_includes_solvent_and_decimal_dilution():
    a = benchmark.condition("01", "1e-2", "PG")
    assert a == benchmark.condition("1", "0.010", "pg")
    assert a != benchmark.condition("1", "0.01", "DEP")
    assert a != benchmark.condition("1", "0.1", "pg")
    assert a == benchmark.condition("1", ".01", "1,2-propanediol")


def test_native_carrier_features_preserve_oils_and_ethanol_strengths():
    native = {"1": {"profile": [1.0] + [0.0] * 18, "odor_impact": 1.0}}
    features = [benchmark.feature(benchmark.condition("1", ".01", carrier), native)
                for carrier in ("paraffin oil", "mineral oil", "90% ethanol", "99% ethanol", "PG")]
    assert len({tuple(value[-len(benchmark.CARRIERS):]) for value in features}) == 5
    assert np.array_equal(features[-1], benchmark.feature(benchmark.condition("1", ".01", "1,2-propanediol"), native))


@pytest.mark.parametrize("value", ["0", "NaN", "Infinity", "-1", "1.1"])
def test_invalid_stock_dilution_fails(value):
    with pytest.raises(ValueError):
        benchmark.condition("1", value, "pg")


def test_formula_key_preserves_ratios_but_collapses_equivalent_copies():
    a, b = ("1", "0.1", "pg"), ("2", "0.2", "dep")
    assert benchmark.formula_key([a, b]) == benchmark.formula_key([b, a, a, b])
    assert benchmark.formula_key([a, a, b]) != benchmark.formula_key([a, b])


def test_measured_stock_mean_does_not_double_apply_dilution():
    a, b = ("1", "0.001", "pg"), ("2", "0.5", "dep")
    prediction, substitutions = benchmark.measured_mean([a, b], {a: np.array([2, 0]), b: np.array([0, 4])})
    assert np.array_equal(prediction, [1, 2])
    assert substitutions == 0


def test_nearest_stock_never_crosses_solvent_or_one_decade():
    key = ("1", "0.1", "pg")
    measured = {key: np.array([1.0, 0.0])}
    assert benchmark.measured_mean([("1", "0.1", "dep")], measured, nearest=True)[0] is None
    assert benchmark.measured_mean([("1", "0.001", "pg")], measured, nearest=True)[0] is None
    assert benchmark.measured_mean([("1", "0.01", "pg")], measured)[0] is None
    assert benchmark.measured_mean([("1", "0.01", "pg")], measured, nearest=True)[1] == 1


def test_known_missing_release_axes_are_not_filled_with_zeros():
    row = {name: "1" for name in benchmark.ENDPOINTS}
    row["Ozone"] = row["Metallic"] = "NA"
    with pytest.raises(ValueError):
        benchmark.vector(row)
    result = benchmark.vector(row, optional_release_columns=True)
    assert np.isnan(result[-2:]).all()
    assert len(benchmark.EVALUATED_ENDPOINTS) == 49
    row["Woody"] = "NA"
    with pytest.raises(ValueError):
        benchmark.vector(row, optional_release_columns=True)


def _csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def test_single_join_uses_stimulus_not_renumbered_component_id(tmp_path):
    b = benchmark
    _csv(tmp_path / b.SOURCE_FILES["components"][0], ["id", "CID", "dilution", "solvent"], [
        {"id": "1", "CID": "50", "dilution": ".1", "solvent": "dep"},
        {"id": "2", "CID": "50", "dilution": ".1", "solvent": "pg"}])
    _csv(tmp_path / b.SOURCE_FILES["stimuli"][0], ["id", "components"], [{"id": "X", "components": "2"}])
    _csv(tmp_path / b.SOURCE_FILES["task1_stimuli"][0], ["stimulus", "molecule", "dilution", "solvent"], [])
    row = {"stimulus": "X", "components": "1", "molecule": "50", "dilution": ".1", **{name: "1" for name in b.ENDPOINTS}}
    _csv(tmp_path / b.SOURCE_FILES["single"][0], list(row), [row])
    _, _, measured, audit = b.load_design(tmp_path)
    assert ("50", "0.1", "pg") in measured
    assert ("50", "0.1", "dep") not in measured
    assert audit["ignored_renumbered_single_component_ids"] == ["X"]
    row["molecule"] = "51"
    _csv(tmp_path / b.SOURCE_FILES["single"][0], list(row), [row])
    with pytest.raises(ValueError, match="CID/dilution conflict"):
        b.load_design(tmp_path)


def test_duplicate_outcomes_and_overwriting_evidence_fail(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.unique_rows([{"stimulus": "A"}, {"stimulus": "A"}], "stimulus")
    path = tmp_path / "evidence.json"
    benchmark.write_new(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        benchmark.write_new(path, {"status": "replacement"})


def test_empty_seal_cannot_skip_code_and_prediction_bindings(tmp_path):
    benchmark.write_new(tmp_path / "seal.json", {"schema": benchmark.SCHEMA, "files": {}, "code": {}})
    with pytest.raises(ValueError, match="seal schema"):
        benchmark.verify_seal(tmp_path)


def test_prepare_source_explicitly_excludes_target_outcome_parsing():
    import inspect
    source = inspect.getsource(benchmark.prepare)
    assert 'SOURCE_FILES["target_outcomes"]' not in source
    assert '"certification_90_allowed": False' in source
    assert '"runtime_promotion_allowed": False' in source
