from datetime import date
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.formulation_inputs import HydrolysisParameters, StorageConditions
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest, LotionSimulationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.formulation_science import permeability_estimates, storage_retention
from fragrance_ai.recommender.lotion import simulate_lotion
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs, estimate_lotion_recipe
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion, prepare_lotion_optimization
from tests.test_lotion_v21 import fixture
from tests.test_ai_extensions import Formula


def kinetic(identifier="test", **kwargs):
    return HydrolysisParameters(ingredient_id=identifier, neutral_per_min=.001,
        acid_m_inv_min=0, base_m_inv_min=0, minimum_ph=4, maximum_ph=9,
        source_reference="synthetic unit test only", source_kind="simulated", source_date=date.today(), **kwargs)


def test_first_order_storage_matches_analytic_half_life():
    conditions = StorageConditions(ph=5, days=math.log(2)/.001/1440)
    assert storage_retention(kinetic(), conditions, 1)["parent_fraction"] == pytest.approx(.5)
    assert storage_retention(kinetic(), conditions, .5)["parent_fraction"] == pytest.approx(math.sqrt(.5))


def test_ph_changes_hydrolysis_rate_and_preserves_provenance():
    params = kinetic().model_copy(update={"neutral_per_min": 0, "base_m_inv_min": 1e6})
    low = storage_retention(params, StorageConditions(ph=4, days=1), 1)
    high = storage_retention(params, StorageConditions(ph=9, days=1), 1)
    assert high["aqueous_rate_per_min"] == pytest.approx(low["aqueous_rate_per_min"] * 1e5)
    assert high["parent_fraction"] < low["parent_fraction"]
    assert not high["source_verified"]


@pytest.mark.parametrize("values", [{"ph": 15, "days": 1}, {"ph": 5, "days": -1},
    {"ph": True, "days": 1}, {"ph": 5, "days": 1, "temperature_c": 40}])
def test_invalid_storage_is_rejected(values):
    with pytest.raises(ValueError):
        StorageConditions.model_validate(values)


def test_kinetic_domain_is_not_extrapolated():
    with pytest.raises(ValueError):
        StorageConditions(ph=3, days=1, kinetics=[kinetic()])


def test_reaction_and_storage_mass_balance_and_analytic_limit():
    value, catalog = fixture()
    s = value["simulation"]
    s.update(times_minutes=[0, 10], water_loss_per_min=0)
    for row in s["materials"]:
        row.update(initial_parent_fraction=.8, aqueous_hydrolysis_per_min=.01,
                   reaction_source_reference="synthetic analytic mass balance", gas_transfer_cm_min=1e-12,
                   skin_permeability_cm_min=0)
    parsed = LotionSimulationRequest.model_validate(s)
    result = simulate_lotion(parsed, catalog)
    first, last = result["temporal_profile"]
    for i, m in enumerate(last["materials"]):
        aqueous = first["aqueous_volume_ml_cm2"]
        lipid = first["lipid_volume_ml_cm2"]
        fraction = aqueous / (aqueous + parsed.materials[i].lipid_water_partition*lipid)
        expected = m["initial_mg_cm2"] * .8 * math.exp(-.01*fraction*10)
        assert m["remaining_mg_cm2"] == pytest.approx(expected, abs=1e-10)
        assert m["skin_sink_mg_cm2"] == 0
        assert m["degraded_parent_equivalent_mg_cm2"] > m["initial_mg_cm2"]*.2
    assert result["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12


def test_robust_optimizer_uses_each_scenarios_own_threshold():
    value, catalog = fixture()
    request = LotionOptimizationRequest.model_validate(value)
    s = request.simulation.model_dump(mode="json")
    s["materials"][1]["odor_threshold_mg_m3"] *= 9
    result = optimize_lotion(request, catalog, transport_scenarios=[LotionSimulationRequest.model_validate(s)])
    assert result["score"] == pytest.approx(75, abs=.1)
    lines = result["closest_candidate"]
    assert lines[1]["concentrate_percent"] == pytest.approx(75, abs=.2)
    assert not result["profile_target_met"]


def test_uptake_constraint_changes_recipe_and_is_checked_in_final_simulation():
    value, catalog = fixture()
    value["simulation"]["materials"][0]["skin_permeability_cm_min"] = 0
    request = LotionOptimizationRequest.model_validate(value)
    base = optimize_lotion(request, catalog)
    uptake = sum(m["skin_sink_mg_cm2"] for m in base["simulation"]["temporal_profile"][-1]["materials"])
    constrained = optimize_lotion(request.model_copy(update={"maximum_modeled_uptake_mg_cm2": uptake/2}), catalog)
    actual = sum(m["skin_sink_mg_cm2"] for m in constrained["simulation"]["temporal_profile"][-1]["materials"])
    assert actual <= uptake/2 * (1+1e-8)
    assert constrained["score"] < base["score"]


def test_transition_schedule_changes_objective_not_coefficients():
    value, catalog = fixture()
    value["brief"] = "첫향은 시트러스, 잔향은 우디"
    value["simulation"]["times_minutes"] = [0,1,15,60,120]
    value["transition_schedule"] = {"opening_until_minutes": 5, "heart_until_minutes": 40}
    request = LotionOptimizationRequest.model_validate(value)
    fast = prepare_lotion_optimization(request, catalog)[0]
    value["transition_schedule"]["opening_until_minutes"] = 30
    slow = prepare_lotion_optimization(LotionOptimizationRequest.model_validate(value), catalog)[0]
    assert fast["evaluation_targets"][1]["phase"] == "heart"
    assert slow["evaluation_targets"][1]["phase"] == "opening"
    assert fast["evaluation_targets"][1]["target_profile"] != slow["evaluation_targets"][1]["target_profile"]


def test_qspr_is_numeric_but_not_skin_compatibility():
    value = permeability_estimates(154.25, 2.7)
    assert value["cm_min"] == pytest.approx(10**(-2.72+.71*2.7-.0061*154.25)/60)
    assert value["cm_min"] > 0 and not value["skin_compatibility_verified"]
    assert permeability_estimates(1500, 2)["cm_min"] is None


def test_fast_search_stops_only_after_real_profile_target_passes():
    value, catalog = fixture()
    value["search_goal"] = "reach_target"
    result = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result["score"] >= 95 and result["profile_target_met"]
    assert result["solver_calls"] == 1
    value["brief"] = "rose scent"
    failed = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert not failed["profile_target_met"] and failed["recipe"] == []


def test_brief_only_design_reaches_real_solver_without_external_inference():
    catalog = IngredientCatalog.load_builtin()
    app = FastAPI()
    calls = []
    register_ai_extensions(app, Formula, catalog, lambda *a, **kw: calls.append(True), lambda: None)
    with TestClient(app) as client:
        response = client.post("/v1/applications/body-lotion/design", json={"brief": "citrus woody scent"})
        assert response.status_code == 200
        result = response.json()
        assert result["candidate_recipe"]
        assert result["transport_simulation_calls"] + result["reused_basis_simulations"] == 6
        assert result["transport_simulation_calls"] in (3,6)
        assert result["estimation"]["external_api_calls"] == 0
        assert sum(m["concentrate_percent"] for m in result["candidate_recipe"]) == pytest.approx(100)
        assert result["skin_exposure"]["modeled_daily_sink_dose_mg_kg"] >= 0
        assert not result["skin_exposure"]["skin_compatibility_verified"]
        assert not calls
        assert client.post("/v1/applications/body-lotion/design", json={"brief": "citrus woody scent"}).status_code == 200


def test_storage_requirement_does_not_assume_missing_rates_are_zero():
    with pytest.raises(ValueError, match="no candidates"):
        build_estimated_lotion_inputs(LotionEstimateRequest(brief="woody scent", storage={"ph": 5, "days": 30,
            "minimum_parent_retention_percent": 95}), IngredientCatalog.load_builtin())


def test_storage_kinetics_feed_actual_transport_inputs():
    catalog = IngredientCatalog.load_builtin()
    request = LotionEstimateRequest(brief="woody scent")
    inputs, _, _ = build_estimated_lotion_inputs(request, catalog)
    identifier = inputs.simulation.materials[0].ingredient_id
    request = request.model_copy(update={"storage": StorageConditions(ph=5, days=30, kinetics=[kinetic(identifier)])})
    altered, scenarios, evidence = build_estimated_lotion_inputs(request, catalog)
    m = next(m for m in altered.simulation.materials if m.ingredient_id == identifier)
    assert m.initial_parent_fraction < 1 and m.aqueous_hydrolysis_per_min == .001
    assert evidence["coverage"]["missing_hydrolysis_kinetics"] > 0
    assert next(m for m in scenarios[0].materials if m.ingredient_id == identifier).initial_parent_fraction < m.initial_parent_fraction


def test_negative_only_opening_keeps_positive_global_target_and_avoidance():
    value, catalog = fixture()
    value["brief"] = "opening no sweetness, drydown woody musk"
    value["simulation"]["times_minutes"] = [0,15,60,480]
    request = LotionOptimizationRequest.model_validate(value)
    prepared, _, _ = prepare_lotion_optimization(request, catalog)
    first = prepared["evaluation_targets"][0]
    assert sum(first["target_profile"].values()) > 0
    assert "gourmand" in first["avoided"]
    result = optimize_lotion(request, catalog)
    assert result["score"] is not None
    assert all(row["score"] is not None for row in result["timepoint_assessments"])
