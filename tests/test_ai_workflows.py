from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform.ai_extensions import register_ai_extensions, composition_distance, evidence_summary, display_timeline
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_ai_extensions import Formula


@pytest.fixture
def workflow():
    app = FastAPI()
    catalog = IngredientCatalog.load_builtin()
    ids = [item.ingredient_id for item in catalog.ingredients[:3]]
    calls, rates = [], []
    def generate(request, response, **kwargs):
        calls.append((request, kwargs))
        selected = next(identifier for identifier in ids if identifier not in kwargs.get("explicit_bans", []))
        return {"brief": {"constraints": request.model_dump()}, "formula_id": selected,
            "recipe": [{"ingredient_id": selected, "concentrate_percent": 100, "availability": .8}],
            "full_profile_target_met": True, "calculated_profile_similarity": 96.,
            "estimated_concentrate_cost_per_kg": 50., "regulatory": {}, "temporal_profile": [],
            "deployment": {"wheel_sha256": "test-runtime"}}
    register_ai_extensions(app, Formula, catalog, generate, lambda: rates.append(1))
    with TestClient(app) as client:
        yield client, calls, rates, ids


def test_distinct_alternative_and_server_side_comparison(workflow):
    client, calls, rates, ids = workflow
    request = {"formula": {"brief": "woody scent"}}
    first = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
    result = client.post("/v1/formulas/alternatives", json={**request, "previous_candidate_ids": [first["candidate_id"]]})
    assert result.status_code == 200
    second = result.json()["candidates"][0]
    assert result.json()["status"] == "distinct_target_met"
    assert second["formula_id"] != first["formula_id"]
    assert calls[1][1]["explicit_bans"] == [ids[0]]
    compared = client.post("/v1/formulas/compare", json={"candidate_ids": [first["candidate_id"], second["candidate_id"]]})
    assert compared.status_code == 200
    assert compared.json()["pairs"][0]["composition_distance"] == 1
    assert compared.json()["new_inference_count"] == 0 and len(calls) == 2
    assert len(rates) == 3


def test_duplicate_is_not_returned_and_next_variation_can_find_a_third_composition(workflow):
    client, calls, _, _ = workflow
    request = {"formula": {"brief": "woody scent"}}
    first = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
    second = client.post("/v1/formulas/alternatives", json={**request, "previous_candidate_ids": [first["candidate_id"]]}).json()["candidates"][0]
    query = {**request, "previous_candidate_ids": [first["candidate_id"], second["candidate_id"]]}
    duplicate = client.post("/v1/formulas/alternatives", json=query).json()
    assert duplicate["status"] == "insufficient_diversity" and not duplicate["candidates"]
    third = client.post("/v1/formulas/alternatives", json={**query, "variation_index": duplicate["next_variation_index"]}).json()
    assert third["status"] == "distinct_target_met"
    assert third["candidates"][0]["formula_id"] not in {first["formula_id"], second["formula_id"]}
    assert len(calls) == 4


def test_explicitly_requested_material_is_not_removed_to_force_diversity(workflow):
    client, _, _, _ = workflow
    request = {"formula": {"brief": "Dihydromyrcenol woody scent"}}
    first = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
    result = client.post("/v1/formulas/alternatives", json={**request, "previous_candidate_ids": [first["candidate_id"]]}).json()
    assert result["status"] == "diversity_options_exhausted"
    assert not result["candidates"]


def test_comparison_cannot_forge_scores_or_mix_intents(workflow):
    client, calls, _, _ = workflow
    left = client.post("/v1/formulas/evaluate", json={"formula": {"brief": "woody scent"}}).json()["candidates"][0]
    right = client.post("/v1/formulas/evaluate", json={"formula": {"brief": "floral scent"}}).json()["candidates"][0]
    assert client.post("/v1/formulas/compare", json={"candidate_ids": [left["candidate_id"], "forged"]}).status_code == 409
    assert client.post("/v1/formulas/compare", json={"candidate_ids": [left["candidate_id"], right["candidate_id"]]}).status_code == 422
    assert client.post("/v1/formulas/compare", json={"candidate_ids": [left["candidate_id"], left["candidate_id"]]}).status_code == 422
    assert len(calls) == 2


def test_different_concentrations_are_comparable_but_not_different_compositions(workflow):
    client, _, _, _ = workflow
    request = {"formula": {"brief": "woody scent"}, "concentration_range": {"minimum": 1, "maximum": 2}}
    a = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
    b = client.post("/v1/formulas/evaluate", json={**request, "scenario_index": 2}).json()["candidates"][0]
    result = client.post("/v1/formulas/compare", json={"candidate_ids": [a["candidate_id"], b["candidate_id"]]}).json()
    assert result["pairs"][0]["composition_distance"] == 0
    assert result["pairs"][0]["concentration_difference_points"] == 1
    assert client.post("/v1/formulas/alternatives", json={**request, "previous_candidate_ids": [b["candidate_id"]]}).status_code == 422


def test_relative_revision_preserves_budget_and_reaches_generator(workflow):
    client, calls, _, _ = workflow
    request = {"formula": {"brief": "woody musky scent", "max_formula_cost_per_kg": 123}, "instruction": "more woody, less musky"}
    prepared = client.post("/v1/briefs/revise", json=request)
    assert prepared.status_code == 200 and not calls
    data = prepared.json()
    assert data["request"]["formula"]["max_formula_cost_per_kg"] == 123
    assert data["adjustments"]["woody"] > 1 and data["adjustments"]["musky"] < 1
    response = client.post("/v1/formulas/revise", json=request)
    assert response.status_code == 200 and len(calls) == 1
    assert calls[0][1]["target_profile_override"]["woody"] > calls[0][1]["target_profile_override"]["musky"]
    assert calls[0][0].max_formula_cost_per_kg == 123


@pytest.mark.parametrize("instruction", ["increase budget", "opening more citrus", "dry down more citrus", "우디 제거 머스크 제거"])
def test_unhandled_or_empty_revisions_do_not_generate(workflow, instruction):
    client, calls, _, _ = workflow
    response = client.post("/v1/formulas/revise", json={"formula": {"brief": "woody musky scent"}, "instruction": instruction})
    assert response.status_code == 422
    assert not calls


def test_temporary_comparison_results_are_immutable(workflow):
    client, _, _, _ = workflow
    request = {"formula": {"brief": "woody scent"}, "concentration_range": {"minimum": 1, "maximum": 2}}
    left = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
    right = client.post("/v1/formulas/evaluate", json={**request, "scenario_index": 2}).json()["candidates"][0]
    left["target_match_score"] = 100
    result = client.post("/v1/formulas/compare", json={"candidate_ids": [left["candidate_id"], right["candidate_id"]]}).json()
    assert result["candidates"][0]["target_match_score"] == 96


def test_evidence_does_not_claim_live_supply_or_calibrated_human_accuracy():
    value = evidence_summary({"recipe": [{"availability": .9}], "vapor_pressure_coverage_percent": 20})
    assert not value["supply"]["live_inventory_verified"]
    assert not value["human_accuracy_proven_by_this_summary"]
    assert value["provided_vapor_pressure_coverage_percent"] == 20
    assert composition_distance({"a": 50, "b": 50}, {"a": 50, "c": 50}) == .5
    assert composition_distance({"a": 100.0001}, {"a": 100.}) == 0
    assert composition_distance({}, {"a": 100.}) is None


def test_timeline_marks_interpolation_and_never_extrapolates_duration():
    points = [{"minutes": 0, "scent_profile": {"citrus": 1.}, "relative_to_opening_intensity_percent": 100.},
              {"minutes": 60, "scent_profile": {"woody": 1.}, "relative_to_opening_intensity_percent": 40.}]
    result = display_timeline(points, (0, 30, 480))
    assert result[0]["status"] == "simulated_timepoint"
    assert result[1]["status"] == "display_interpolation_not_new_simulation"
    assert result[1]["scent_profile"]["citrus"] == result[1]["scent_profile"]["woody"] == .5
    assert result[1]["relative_to_opening_intensity_percent"] == 70
    assert result[2]["status"] == "outside_simulated_range" and result[2]["scent_profile"] is None


def test_reassessment_rejects_duplicate_materials_and_passes_fixed_weights(workflow):
    client, calls, _, ids = workflow
    request = {"formula": {"brief": "woody scent"}, "lines": [{"ingredient_id": ids[0], "concentrate_percent": 100}]}
    assert client.post("/v1/formulas/reassess", json=request).status_code == 200
    assert calls[0][1]["fixed_formula_weights"] == {ids[0]: 100}
    bad = {**request, "lines": [request["lines"][0], request["lines"][0]]}
    assert client.post("/v1/formulas/reassess", json=bad).status_code == 422
    assert len(calls) == 1


def test_structured_phase_and_intensity_controls_reach_generator(workflow):
    client, calls, _, _ = workflow
    controls = {"intensity_level": 4, "phase_target_profiles": {"heart": {"floral": 1}}}
    request = {"formula": {"brief": "woody scent"}, "edits": controls}
    prepared = client.post("/v1/briefs/prepare", json=request).json()
    assert prepared["intent"]["absolute_intensity_target"] == .75
    assert prepared["intent"]["phase_target_profiles"]["heart"]["floral"] == 1
    assert client.post("/v1/formulas/evaluate", json=request).status_code == 200
    assert calls[0][1]["intent_controls"] == controls


def test_actual_api_fixed_formula_round_trip_and_comparison():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    with TestClient(create_web_app(str(REGISTRY))) as client:
        request = {"formula": {"brief": "citrus, fruity scent", "max_risk_tier": 2, "enable_registry_trace_candidates": True}}
        response = client.post("/v1/formulas/evaluate", json=request)
        assert response.status_code == 200
        original = response.json()["candidates"][0]
        assert original["status"] == "target_met"
        lines = [{"ingredient_id": row["ingredient_id"], "concentrate_percent": row["concentrate_percent"]} for row in original["result"]["recipe"]]
        response = client.post("/v1/formulas/reassess", json={**request, "lines": lines})
        assert response.status_code == 200
        fixed = response.json()["candidates"][0]
        assert {row["ingredient_id"]: row["concentrate_percent"] for row in fixed["result"]["closest_candidate"]} == {row["ingredient_id"]: row["concentrate_percent"] for row in lines}
        assert not fixed["result"]["score_contract"]["full_generator_approval"]
        comparison = client.post("/v1/formulas/compare", json={"candidate_ids": [original["candidate_id"], fixed["candidate_id"]]})
        assert comparison.status_code == 200
        assert comparison.json()["pairs"][0]["composition_distance"] == 0
