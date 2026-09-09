from copy import deepcopy
from dataclasses import replace
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fragrance_ai.platform.application_context import ApplicationContext, assess_application_context
from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion import simulate_lotion
from tests.test_application_context import context_data
from tests.test_ai_extensions import Formula


def data():
    catalog = IngredientCatalog.load_builtin()
    context = context_data()
    context.pop("release_series")
    return {"application_context": context,
        "parameter_context_id": assess_application_context(ApplicationContext.model_validate(context))["context_id"],
        "phase_components": [{"name": "water", "phase": "aqueous", "density_g_ml": 1.},
                             {"name": "test oil phase", "phase": "lipid", "density_g_ml": 1.}],
        "materials": [{"ingredient_id": catalog.ingredients[0].ingredient_id, "concentrate_percent": 100.,
            "lipid_water_partition": 10., "air_water_partition": .001, "gas_transfer_cm_min": .01,
            "skin_permeability_cm_min": .00001, "odor_threshold_mg_m3": .1,
            "source_reference": "synthetic test coefficients, not measured", "source_date": "2026-09-01", "source_kind": "simulated"}],
        "water_loss_per_min": 0., "retained_water_fraction": .1, "water_loss_source_reference": "synthetic fixture",
        "headspace_height_cm": 1., "air_exchange_per_min": 1000., "times_minutes": [0., 15., 60., 120.]}


def run(value=None, catalog=None):
    return simulate_lotion(LotionSimulationRequest.model_validate(value or data()), catalog or IngredientCatalog.load_builtin())


def rebind(value):
    value["parameter_context_id"] = assess_application_context(ApplicationContext.model_validate(value["application_context"]))["context_id"]


def test_analytic_solution_mass_and_air_convolution():
    value = data()
    result = run(value)
    capacity = .00198 * (.8 + 10 * .2)
    loss = .00002 / capacity
    time = 120.
    expected_mass = .02 * math.exp(-loss * time)
    expected_air = .00001 / capacity * .02 * (math.exp(-loss * time) - math.exp(-1000 * time)) / (1000 - loss)
    last = result["temporal_profile"][-1]["materials"][0]
    assert last["remaining_mg_cm2"] == pytest.approx(expected_mass, rel=1e-12)
    assert last["headspace_mg_cm2"] == pytest.approx(expected_air, rel=1e-12)
    assert result["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-14
    assert result["status"] == "research_simulation"
    assert result["temporal_profile"][0]["scent_profile"] is None
    assert result["human_similarity_percent"] is None
    assert result["manufacturing_approved"] is False


def test_equal_evaporation_and_ventilation_rates_are_finite():
    value = data()
    value["air_exchange_per_min"] = .00002 / (.00198 * 2.8)
    result = run(value)
    row = result["temporal_profile"][-1]["materials"][0]
    loss = value["air_exchange_per_min"]
    assert row["headspace_mg_cm2"] == pytest.approx(.5 * loss * .02 * 120 * math.exp(-loss * 120), rel=1e-12)
    assert result["status"] == "outside_open_sink_assumption"


def test_water_loss_matches_independent_ode_and_refinement_converges():
    value = data()
    value["water_loss_per_min"] = .03
    def rhs(t, y):
        capacity = .00198 * (.8 * (.1 + .9 * math.exp(-.03*t)) + 10 * .2)
        return [-.00002 / capacity * y[0]]
    reference = solve_ivp(rhs, (0., 120.), [.02], rtol=1e-11, atol=1e-13).y[0, -1]
    value["integration_step_minutes"] = 1.
    coarse = run(value)["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"]
    value["integration_step_minutes"] = .25
    fine = run(value)["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"]
    assert abs(fine-reference) < abs(coarse-reference) / 10
    assert fine == pytest.approx(reference, rel=1e-6)


def test_drying_and_lipid_retention_change_real_mass_curve():
    baseline = run()["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"]
    value = data()
    value["water_loss_per_min"] = .1
    assert run(value)["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"] < baseline
    value = data()
    value["materials"][0]["lipid_water_partition"] = 100.
    assert run(value)["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"] > baseline


def test_dose_changes_depletion_rate_not_just_amplitude():
    a = run()["temporal_profile"][-1]["materials"][0]
    value = data()
    value["application_context"]["application_mass_mg_cm2"] *= 2
    rebind(value)
    b = run(value)["temporal_profile"][-1]["materials"][0]
    assert b["remaining_mg_cm2"] / b["initial_mg_cm2"] > a["remaining_mg_cm2"] / a["initial_mg_cm2"]


def test_missing_threshold_does_not_invent_human_intensity():
    value = data()
    value["materials"][0].pop("odor_threshold_mg_m3")
    result = run(value)
    assert result["diagnostics"]["odor_threshold_coverage_percent"] == 0
    assert result["temporal_profile"][-1]["profile_basis"] == "air_mass_weighted_catalog_proxy"
    assert result["temporal_profile"][-1]["total_odor_activity_proxy"] is None


def test_mixture_profile_evolves_without_inventing_catalog_profiles():
    value = data()
    catalog = IngredientCatalog.load_builtin()
    first = replace(catalog.ingredients[0], profile={"citrus": 1.})
    second = replace(catalog.ingredients[1], profile={"woody": 1.})
    catalog = IngredientCatalog([first, second])
    row = deepcopy(value["materials"][0])
    row.update(ingredient_id=second.ingredient_id, concentrate_percent=50., lipid_water_partition=100.)
    value["materials"][0]["concentrate_percent"] = 50.
    value["materials"].append(row)
    result = run(value, catalog)
    early, late = result["temporal_profile"][1], result["temporal_profile"][-1]
    assert late["scent_profile"]["woody"] > early["scent_profile"]["woody"]
    for point in result["temporal_profile"]:
        for material in point["materials"]:
            assert sum(material[key] for key in ("remaining_mg_cm2", "headspace_mg_cm2", "skin_sink_mg_cm2", "ventilated_mg_cm2")) == pytest.approx(material["initial_mg_cm2"], abs=1e-14)


@pytest.mark.parametrize("case", ["temperature", "missing", "wo", "sum", "duplicate", "phase", "water", "time", "boolean", "nan", "work", "source", "inert"])
def test_invalid_or_mismatched_inputs_are_rejected(case):
    value = data()
    if case == "temperature": value["application_context"]["temperature_c"] = 30
    if case == "missing": value["application_context"].pop("application_mass_mg_cm2")
    if case == "wo":
        value["application_context"]["emulsion_type"] = "water_in_oil"
        rebind(value)
    if case == "sum": value["materials"][0]["concentrate_percent"] = 90.
    if case == "duplicate": value["materials"] *= 2
    if case == "phase": value["phase_components"].pop()
    if case == "water": value["phase_components"][0]["phase"] = "lipid"
    if case == "time": value["times_minutes"] = [0, 1, 1]
    if case == "boolean": value["times_minutes"] = [False, 1]
    if case == "nan": value["materials"][0]["gas_transfer_cm_min"] = float("nan")
    if case == "work":
        value["materials"] = [dict(value["materials"][0], ingredient_id=str(i), concentrate_percent=1.) for i in range(100)]
        value["integration_step_minutes"] = .01
        value["times_minutes"] = [0., 1440.]
    if case == "source": value["materials"][0]["source_reference"] = " "
    if case == "inert":
        value["application_context"]["substrate"] = "inert_surface"
        rebind(value)
    with pytest.raises(ValueError): LotionSimulationRequest.model_validate(value)


@pytest.mark.parametrize("kind", ["unknown", "blocked", "diluted"])
def test_ineligible_ingredients_do_not_get_silent_defaults(kind):
    value = data()
    catalog = IngredientCatalog.load_builtin()
    if kind == "unknown": value["materials"][0]["ingredient_id"] = "not-a-material"
    if kind == "blocked": catalog.ingredients[0] = replace(catalog.ingredients[0], blocked=True)
    if kind == "diluted": catalog.ingredients[0] = replace(catalog.ingredients[0], active_strength_percent=10.)
    with pytest.raises(ValueError): run(value, catalog)


def test_api_runs_actual_transport_without_calling_perfume_generator():
    app, rates = FastAPI(), []
    def unexpected(*args, **kwargs):
        pytest.fail("lotion simulation must not substitute the perfume generator")
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), unexpected, lambda: rates.append(1))
    with TestClient(app) as client:
        result = client.post("/v1/applications/body-lotion/simulate", json=data())
        assert result.status_code == 200
        assert result.json()["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-14
        assert len(rates) == 1
        features = client.get("/v1/ai/capabilities").json()["features"]
        assert features["body_lotion_research_transport_simulation"]
        assert features["body_lotion_matrix_model"] is False
        invalid = data()
        invalid["materials"][0]["ingredient_id"] = "unknown"
        assert client.post("/v1/applications/body-lotion/simulate", json=invalid).status_code == 422


def test_vanishing_dry_film_capacity_is_rejected_without_nan_output():
    value = data()
    value["application_context"]["base_components"][0]["mass_percent"] = 100.
    value["application_context"]["base_components"][1]["mass_percent"] = 1e-20
    value["water_loss_per_min"] = 10.
    value["retained_water_fraction"] = 0.
    rebind(value)
    with pytest.raises(ValueError, match="capacity"):
        run(value)


def test_actual_modal_app_exposes_lotion_simulation_and_contract():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    with TestClient(create_web_app(str(REGISTRY))) as client:
        result = client.post("/v1/applications/body-lotion/simulate", json=data())
        assert result.status_code == 200
        assert result.json()["status"] == "research_simulation"
        assert result.json()["calibrated_lotion_generator_enabled"] is False
        assert "/v1/applications/body-lotion/simulate" in client.get("/openapi.json").json()["paths"]


def test_response_memory_budget_is_separate_from_integration_work():
    value = data()
    value["times_minutes"] = [i * .01 for i in range(101)]
    value["materials"] = [dict(value["materials"][0], ingredient_id=str(i), concentrate_percent=.1) for i in range(1000)]
    with pytest.raises(ValueError, match="response budget"):
        LotionSimulationRequest.model_validate(value)
