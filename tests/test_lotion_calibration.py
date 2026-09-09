from copy import deepcopy

import pytest

from fragrance_ai.research.lotion_calibration import LotionCalibrationRequest, fit_lotion_transport
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_lotion import data, run


def calibration_data():
    simulation = data()
    simulation.update(transport_mode="bidirectional_air", times_minutes=[0., 5., 10., 20., 40., 80., 120.])
    truth = deepcopy(simulation)
    truth["materials"][0]["gas_transfer_cm_min"] = .025
    observed = run(truth)
    rows = []
    for split in ("calibration", "validation"):
        for point in observed["temporal_profile"][1:]:
            value = point["materials"][0]["air_concentration_mg_m3"]
            rows.append({"experiment_id": "synthetic-" + split, "split": split, "minutes": point["minutes"],
                "observable": "air_concentration_mg_m3", "value": value, "standard_error": max(value*.01, 1e-9),
                "source_kind": "simulated", "source_reference": "synthetic recovery test, not a physical experiment",
                "source_date": "2026-09-01"})
    return {"simulation": simulation, "observation_context_id": simulation["parameter_context_id"],
        "ingredient_id": simulation["materials"][0]["ingredient_id"], "parameter": "gas_transfer_cm_min",
        "lower_bound": .001, "upper_bound": .1, "observations": rows}


def fit(value):
    return fit_lotion_transport(LotionCalibrationRequest.model_validate(value), IngredientCatalog.load_builtin())


def test_single_coefficient_recovers_and_does_not_mutate_input_or_claim_measurements():
    value = calibration_data()
    original = deepcopy(value)
    result = fit(value)
    assert result["fitted_value"] == pytest.approx(.025, rel=1e-5)
    assert result["validation_after"]["nrmse_by_observed_rms"] < 1e-6
    assert result["numerical_validation_passed"]
    assert not result["all_observations_caller_declared_measured"]
    assert not result["automatic_promotion_allowed"] and not result["source_verified"]
    assert result["calibrated_simulation_candidate"]["materials"][0]["source_kind"] == "estimated"
    assert value == original


def test_validation_values_cannot_change_fitted_coefficient():
    value = calibration_data()
    first = fit(value)
    for row in value["observations"]:
        if row["split"] == "validation": row["value"] *= 2
    changed = fit(value)
    assert changed["fitted_value"] == first["fitted_value"]
    assert not changed["numerical_validation_passed"]
    assert changed["validation_after"]["nrmse_by_observed_rms"] > .4


@pytest.mark.parametrize("case", ["same_experiment", "duplicate", "context", "times", "bounds", "outside",
    "unknown", "no_valid", "mixed_units", "zero", "future", "boolean", "loose_limit"])
def test_invalid_calibration_data_rejected(case):
    value = calibration_data()
    if case == "same_experiment":
        for row in value["observations"]: row["experiment_id"] = "same"
    if case == "duplicate": value["observations"].append(deepcopy(value["observations"][0]))
    if case == "context": value["observation_context_id"] = "0"*64
    if case == "times": value["observations"][0]["minutes"] = 3.
    if case == "bounds": value["lower_bound"] = value["upper_bound"]
    if case == "outside": value["upper_bound"] = 1e10
    if case == "unknown": value["ingredient_id"] = "unknown"
    if case == "no_valid":
        for row in value["observations"]: row["split"] = "calibration"
    if case == "mixed_units": value["observations"][0]["observable"] = "remaining_mg_cm2"
    if case == "zero":
        for row in value["observations"]: row["value"] = 0
    if case == "future": value["observations"][0]["source_date"] = "2099-01-01"
    if case == "boolean": value["observations"][0]["value"] = True
    if case == "loose_limit": value["validation_nrmse_limit"] = .1
    with pytest.raises(ValueError):
        LotionCalibrationRequest.model_validate(value)


def test_boundary_solution_is_not_promoted_as_success():
    value = calibration_data()
    value["upper_bound"] = .015
    result = fit(value)
    assert result["at_parameter_bound"]
    assert not result["numerical_validation_passed"]


def test_fitted_candidate_connects_to_existing_lotion_optimizer():
    from tests.test_lotion_v21 import fixture, optimize
    optimization, catalog = fixture()
    value = calibration_data()
    value["simulation"]["materials"] = deepcopy(optimization["simulation"]["materials"])
    value["simulation"]["coefficient_scope"] = "dilute_fixed_base"
    value["simulation"]["coefficient_scope_reference"] = "synthetic test fixture, not calibrated lotion evidence"
    value["ingredient_id"] = catalog.ingredients[0].ingredient_id
    truth = deepcopy(value["simulation"])
    truth["materials"][0]["gas_transfer_cm_min"] = .025
    results = run(truth, catalog)
    by_time = {p["minutes"]: p["materials"][0]["air_concentration_mg_m3"] for p in results["temporal_profile"]}
    for observation in value["observations"]:
        observation["value"] = by_time[observation["minutes"]]
        observation["standard_error"] = max(observation["value"]*.01, 1e-9)
    fitted = fit_lotion_transport(LotionCalibrationRequest.model_validate(value), catalog)
    assert fitted["numerical_validation_passed"]
    optimization["simulation"] = fitted["calibrated_simulation_candidate"]
    result = optimize(optimization, catalog)
    assert result["simulation"]["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
    sources = result["simulation"]["caller_declared_parameter_sources"]
    assert any(row["reference"] == "offline-lotion-calibration:" + fitted["calibration_id"] for row in sources)
    assert not result["all_user_requirements_verified"]
