from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest, LotionSimulationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion import simulate_lotion
from fragrance_ai.recommender.lotion_transport import bidirectional_step
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion, prepare_lotion_optimization
from tests.test_lotion import data, run
from tests.test_ai_extensions import Formula


@pytest.mark.parametrize("rates", [(1., 2., 3., 4.), (.01, .0, .01, .01), (1e-8, .0, 1e-8, 1e-12),
    (1e-12, 1., 1e-12, 1.), (10000., .0001, 1., .001), (.002, .002, .01, .0001)])
def test_bidirectional_exact_step_matches_independent_matrix_exponential(rates):
    e, s, b, v = rates
    matrix = np.array([[-e-s,b,0,0], [e,-b-v,0,0], [s,0,0,0], [0,v,0,0]])
    expected = expm(matrix * .25) @ np.array([.02,.003,0.,0.])
    actual = np.array([x[0] for x in bidirectional_step(np.array([.02]), np.array([.003]),
        np.array([e]), np.array([s]), np.array([b]), v, .25)])
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-8)
    assert np.all(actual >= 0)
    assert actual.sum() == pytest.approx(.023, abs=1e-12)


def test_finite_air_back_transfer_replaces_invalid_open_sink_prediction():
    value = data()
    value["air_exchange_per_min"] = .001
    open_result = run(value)
    value["transport_mode"] = "bidirectional_air"
    result = run(value)
    assert open_result["status"] == "outside_open_sink_assumption"
    assert result["status"] == "research_simulation"
    assert not result["diagnostics"]["open_sink_assumption_required"]
    assert result["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"] > open_result["temporal_profile"][-1]["materials"][0]["remaining_mg_cm2"]
    assert result["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12


def test_bidirectional_water_loss_matches_independent_four_state_ode():
    value = data()
    value.update(transport_mode="bidirectional_air", water_loss_per_min=.03, air_exchange_per_min=.02)
    def rhs(t, y):
        capacity = .00198 * (.8*(.1+.9*np.exp(-.03*t)) + 2.)
        e = s = .00001 / capacity
        transfer = e*y[0] - .01*y[1]
        return [-transfer-s*y[0], transfer-.02*y[1], s*y[0], .02*y[1]]
    expected = solve_ivp(rhs, (0.,120.), [.02,0.,0.,0.], rtol=1e-11, atol=1e-13).y[:,-1]
    result = run(value)["temporal_profile"][-1]["materials"][0]
    actual = [result[key] for key in ("remaining_mg_cm2", "headspace_mg_cm2", "skin_sink_mg_cm2", "ventilated_mg_cm2")]
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=2e-6)


def fixture():
    builtin = IngredientCatalog.load_builtin()
    items = [replace(builtin.ingredients[0], ingredient_id="fixture-citrus", name="Test citrus material", aliases=(),
                profile={"citrus": 1.}, pyramid="top", max_concentrate_percent=100., price_per_kg=50., availability=1.,
                rarity="common", risk_tier=1, formulation_ready=True, blocked=False, active_strength_percent=100.),
             replace(builtin.ingredients[1], ingredient_id="fixture-wood", name="Test wood material", aliases=(),
                profile={"woody": 1.}, pyramid="base", max_concentrate_percent=100., price_per_kg=50., availability=1.,
                rarity="common", risk_tier=1, formulation_ready=True, blocked=False, active_strength_percent=100.)]
    catalog = IngredientCatalog(items)
    simulation = data()
    simulation.update(transport_mode="bidirectional_air", coefficient_scope="dilute_fixed_base",
                      coefficient_scope_reference="synthetic independent dilute coefficients for test base", times_minutes=[0.,15.,60.])
    row = simulation["materials"][0]
    row.update(ingredient_id=items[0].ingredient_id, concentrate_percent=90.)
    second = dict(row, ingredient_id=items[1].ingredient_id, concentrate_percent=10.)
    simulation["materials"].append(second)
    return {"simulation": simulation, "brief": "citrus woody scent", "minimum_air_concentration_mg_m3": 1e-8}, catalog


def optimize(value=None, catalog=None):
    default, default_catalog = fixture()
    return optimize_lotion(LotionOptimizationRequest.model_validate(value or default), catalog or default_catalog)


def test_natural_language_lotion_inverse_design_actually_changes_weights():
    result = optimize()
    assert result["profile_target_met"] and result["score"] >= 99.9
    assert result["score"] > result["baseline_score"] + 35
    assert sum(row["concentrate_percent"] for row in result["recipe"]) == pytest.approx(100.)
    assert result["recipe"][0]["concentrate_percent"] == pytest.approx(50., abs=.05)
    assert result["transport_simulation_calls"] == 2
    assert not result["all_user_requirements_verified"] and not result["manufacturing_approved"]
    assert all(row["score"] >= 99.9 for row in result["timepoint_assessments"])


def test_transport_difference_changes_optimized_formula_not_just_report():
    value, catalog = fixture()
    reference = optimize(value, catalog)
    value["simulation"]["materials"][1]["lipid_water_partition"] = 100.
    altered = optimize(value, catalog)
    lines = altered["recipe"] or altered["closest_candidate"]
    assert lines[1]["concentrate_percent"] > reference["recipe"][1]["concentrate_percent"] + 20
    assert altered["score"] >= 95.


def test_low_scores_remain_diagnostics_without_lowering_target():
    value, catalog = fixture()
    value["brief"] = "rose scent"
    result = optimize(value, catalog)
    assert not result["profile_target_met"] and result["recipe"] == []
    assert result["closest_candidate"] and result["score"] == 0
    assert result["preparation"]["effective_target"] == 95


def test_floor_and_caps_are_real_feasibility_constraints():
    value, catalog = fixture()
    value["minimum_air_concentration_mg_m3"] = 1e6
    assert optimize(value, catalog)["status"] == "no_feasible_research_formula"
    value, catalog = fixture()
    catalog.ingredients = [replace(item, max_concentrate_percent=20.) for item in catalog.ingredients]
    assert optimize(value, catalog)["status"] == "no_feasible_research_formula"
    value, catalog = fixture()
    value["max_formula_cost_per_kg"] = 1.
    assert optimize(value, catalog)["status"] == "no_feasible_research_formula"


def test_explicit_exclusions_and_notes_reach_actual_composition():
    value, catalog = fixture()
    value["brief"] = "woody scent"
    value["excluded_ingredient_ids"] = ["fixture-citrus"]
    result = optimize(value, catalog)
    assert result["recipe"][0]["ingredient_id"] == "fixture-wood" and len(result["recipe"]) == 1
    value, catalog = fixture()
    value["brief"] = "citrus woody scent, top 30%, base 70%"
    result = optimize(value, catalog)
    lines = result["recipe"] or result["closest_candidate"]
    assert next(row for row in lines if row["pyramid"] == "top")["concentrate_percent"] == pytest.approx(30.)


def test_missing_threshold_does_not_switch_weighting_after_material_is_removed():
    value, catalog = fixture()
    value["simulation"]["materials"][0]["odor_threshold_mg_m3"] = None
    value["brief"] = "woody scent"
    result = optimize(value, catalog)
    assert result["simulation"]["temporal_profile"][-1]["profile_basis"] == "air_mass_weighted_catalog_proxy"


@pytest.mark.parametrize("kind", ["scope", "scope_reference", "mode", "low_target", "missing_material", "changed_concentration", "missing_phase", "count", "unknown", "requested_missing"])
def test_unsupported_or_incomplete_requirements_are_not_silently_accepted(kind):
    value, catalog = fixture()
    if kind == "scope": value["simulation"]["coefficient_scope"] = "fixed_composition"
    if kind == "scope_reference": value["simulation"]["coefficient_scope_reference"] = None
    if kind == "mode": value["simulation"]["transport_mode"] = "open_sink"
    if kind == "low_target": value["target_similarity"] = 90
    if kind == "missing_material": value["simulation"]["materials"][0].pop("lipid_water_partition")
    if kind == "changed_concentration": value["brief"] = "woody scent 농도 3%"
    if kind == "missing_phase": value["brief"] = "opening citrus, drydown woody"
    if kind == "count": value["brief"] = "woody scent 최대 1개 원료"
    if kind == "unknown": value["simulation"]["materials"][0]["ingredient_id"] = "missing"
    if kind == "requested_missing":
        value["brief"] = "Test wood material woody scent"
        value["excluded_ingredient_ids"] = ["fixture-wood"]
    with pytest.raises(ValueError): optimize(value, catalog)


def test_routes_reuse_parser_and_catalog_without_perfume_inference():
    value, catalog = fixture()
    app, rate_calls = FastAPI(), []
    def unexpected(*args, **kwargs): pytest.fail("must not substitute perfume inference")
    register_ai_extensions(app, Formula, catalog, unexpected, lambda: rate_calls.append(1))
    with TestClient(app) as client:
        prepared = client.post("/v1/applications/body-lotion/prepare", json=value)
        assert prepared.status_code == 200
        assert prepared.json()["transport_covered_candidate_count"] == 2
        result = client.post("/v1/applications/body-lotion/optimize", json=value)
        assert result.status_code == 200 and result.json()["profile_target_met"]
        assert len(rate_calls) == 2
        repeated = client.post("/v1/applications/body-lotion/optimize", json=value)
        assert repeated.headers["X-Perfumery-Lotion-Cache"] == "hit"
        assert repeated.json() == result.json()
        assert client.get("/v1/ai/capabilities").json()["features"]["body_lotion_fixed_base_inverse_design"]
        value["target_similarity"] = 90
        assert client.post("/v1/applications/body-lotion/optimize", json=value).status_code == 422


def test_phase_minimum_is_not_replaced_by_an_optimistic_average():
    value, catalog = fixture()
    value["brief"] = "opening citrus, drydown woody"
    value["simulation"]["times_minutes"] = [0., 15., 480.]
    result = optimize(value, catalog)
    assert result["score"] == min(row["score"] for row in result["timepoint_assessments"])
    assert result["score"] < 51 and not result["profile_target_met"]
    assert result["recipe"] == [] and result["closest_candidate"]


def test_actual_modal_app_optimizes_real_catalog_with_explicit_synthetic_transport():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    builtin = IngredientCatalog.load_builtin()
    items = [item for item in builtin.ingredients if item.formulation_ready and not item.blocked and item.risk_tier == 1]
    value, _ = fixture()
    value["brief"] = "clean fresh citrus woody scent"
    value["simulation"]["materials"] = [dict(value["simulation"]["materials"][0], ingredient_id=item.ingredient_id,
        concentrate_percent=100./len(items)) for item in items]
    with TestClient(create_web_app(str(REGISTRY))) as client:
        response = client.post("/v1/applications/body-lotion/optimize", json=value)
        assert response.status_code == 200
        result = response.json()
        lines = result["recipe"] or result["closest_candidate"]
        assert lines and sum(row["concentrate_percent"] for row in lines) == pytest.approx(100.)
        by_id = {item.ingredient_id: item for item in items}
        assert all(row["concentrate_percent"] <= by_id[row["ingredient_id"]].max_concentrate_percent + 1e-8 for row in lines)
        assert result["simulation"]["model"] == "finite_dose_rapid_partition_bidirectional_v2"
        assert not result["simulation"]["parameter_source_verified"]
        assert result["score"] == min(row["score"] for row in result["timepoint_assessments"])
        # This fixture is an API/solver integration check, never lotion accuracy evidence.
        print({"fixture": "real_catalog_synthetic_transport", "score": result["score"],
            "status": result["status"], "material_count": len(lines), "solver_calls": result["solver_calls"]})


def test_solver_timeout_is_not_reported_as_infeasibility(monkeypatch):
    from types import SimpleNamespace
    import fragrance_ai.recommender.lotion_optimizer as module
    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: SimpleNamespace(status=1, success=False))
    # This test exercises the all-recovery-unavailable path. Separate tests
    # cover a successful block/conic recovery after the LP times out.
    monkeypatch.setattr('fragrance_ai.recommender.accord_trials.accord_trials', lambda *a, **k: iter(()))
    monkeypatch.setattr('fragrance_ai.recommender.lotion_conic.solve_full_profile',
        lambda **k: (None, {'solver_calls': 0, 'status': 'unavailable', 'solver_incomplete': True}))
    result = optimize()
    assert result["search_incomplete"] and result["closest_candidate"]
    assert result["score"] == result["baseline_score"]
    value, catalog = fixture()
    value["minimum_air_concentration_mg_m3"] = 1e6
    result = optimize(value, catalog)
    assert result["status"] == "search_incomplete"
    assert result["recipe"] == [] and result["closest_candidate"] == []


@pytest.mark.parametrize("bad_times", [None, 3, [True, 15]])
def test_malformed_times_are_validation_errors_not_server_errors(bad_times):
    value = data()
    value["times_minutes"] = bad_times
    with pytest.raises(ValueError):
        LotionSimulationRequest.model_validate(value)


def test_core_and_registry_usd_estimates_join_without_currency_conversion():
    value, catalog = fixture()
    catalog.ingredients[1] = replace(catalog.ingredients[1], currency="USD_estimate_not_supplier_quote")
    result = optimize(value, catalog)
    assert result["profile_target_met"]
    assert result["preparation"]["price_currency"] == "USD"
    assert len(result["preparation"]["price_source_labels"]) == 2
    assert not result["preparation"]["live_supplier_price_verified"]


@pytest.mark.parametrize("currency", ["KRW", "EUR", "USD_cents", "unverified_unknown"])
def test_actual_currency_or_unit_mismatch_is_still_rejected(currency):
    value, catalog = fixture()
    catalog.ingredients[1] = replace(catalog.ingredients[1], currency=currency)
    with pytest.raises(ValueError, match="mixed units"):
        optimize(value, catalog)


def test_attainability_diagnostic_distinguishes_missing_axes_from_search_timeout(monkeypatch):
    value, catalog = fixture()
    value["brief"] = "rose scent"
    result = optimize(value, catalog)
    assert result["attainability"]["numeric_model_score_upper_percent"] < .03
    assert result["attainability"]["requested_target_excluded_by_numeric_bound"]
    assert not result["attainability"]["formal_certificate"]
    from types import SimpleNamespace
    import fragrance_ai.recommender.lotion_optimizer as module
    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: SimpleNamespace(status=1, success=False))
    result = optimize(value, catalog)
    assert result["attainability"]["search_incomplete"]
    assert result["attainability"]["numeric_model_score_upper_percent"] == 100
    assert not result["attainability"]["requested_target_excluded_by_numeric_bound"]
