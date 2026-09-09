from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.recommender.catalog import IngredientCatalog


class Formula(BaseModel):
    brief: str
    target_similarity: float = 95
    max_ingredients: int = 50000
    target_region: str = "EU"
    product_category: str = "eau_de_parfum"
    product_concentration_percent: float = 15
    max_formula_cost_per_kg: float = 180
    require_full_profile_match: bool = True
    experimental_disable_safety: bool = False


@pytest.fixture
def extension():
    app = FastAPI()
    calls = []
    def generate(request, response, **kwargs):
        calls.append((request, kwargs))
        return {"brief": {"constraints": request.model_dump()}, "formula_id": "fixture-formula",
            "recipe": [{"ingredient_id": "fixture", "concentrate_percent": 100}], "full_profile_target_met": True,
            "calculated_profile_similarity": 96., "estimated_concentrate_cost_per_kg": 50., "regulatory": {}, "temporal_profile": []}
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), generate, lambda: None)
    with TestClient(app) as client:
        yield client, calls


def body():
    return {"formula": {"brief": "green woody scent"}, "concentration_range": {"minimum": .6, "maximum": .9}}


def test_prepare_is_inference_free_and_edits_survive(extension):
    client, calls = extension
    request = body()
    request["edits"] = {"target_profile": {"woody": 2, "citrus": 1}}
    response = client.post("/v1/briefs/prepare", json=request)
    assert response.status_code == 200
    value = response.json()
    assert value["status"] == "ready" and not calls
    assert [row["product_concentration_percent"] for row in value["scenarios"]] == [.6, .75, .9]
    assert value["intent"]["target_profile"]["woody"] == pytest.approx(2/3)


def test_missing_scent_returns_a_question_instead_of_fabricating_an_intent(extension):
    client, calls = extension
    response = client.post("/v1/briefs/prepare", json={"formula": {"brief": "..."}})
    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["intent"] is None
    assert response.json()["questions"][0]["id"] == "formula.brief"
    assert not calls


@pytest.mark.parametrize("brief", ["woody scent, at least 8 hours", "우디 향, 지속성 8시간"])
def test_explicit_textual_duration_is_not_silently_ignored(extension, brief):
    client, calls = extension
    response = client.post("/v1/formulas/evaluate", json={"formula": {"brief": brief}})
    assert response.status_code == 422
    assert response.json()["detail"]["unsupported_requirements"][0]["code"] == "longevity_threshold_model_missing"
    assert not calls


def test_scenario_outputs_share_request_id_but_have_distinct_identity(extension):
    client, calls = extension
    request = body()
    left = client.post("/v1/formulas/evaluate", json=request).json()
    right = client.post("/v1/formulas/evaluate", json={**request, "scenario_index": 2}).json()
    assert len(calls) == 2
    assert left["request_id"] == right["request_id"]
    assert left["candidates"][0]["candidate_id"] != right["candidates"][0]["candidate_id"]
    assert [row[0].product_concentration_percent for row in calls] == [.6, .9]
    assert left["candidates"][0]["composition_total_percent"] == 100
    assert left["candidates"][0]["longevity_hours"] is None


@pytest.mark.parametrize("extra", [{"product_type": "body_lotion"}, {"minimum_longevity_hours": 8}])
def test_unsupported_models_stop_before_inference(extension, extra):
    client, calls = extension
    response = client.post("/v1/formulas/evaluate", json={**body(), **extra})
    assert response.status_code == 422
    assert response.json()["detail"]["unsupported_requirements"]
    assert not calls


def test_budget_missing_units_returns_questions(extension):
    client, calls = extension
    request = {**body(), "budget": {"amount": 4500, "per_volume_ml": 100}}
    result = client.post("/v1/briefs/prepare", json=request).json()
    assert result["status"] == "needs_clarification"
    assert "budget.product_density_g_ml" in [q["id"] for q in result["questions"]]
    assert client.post("/v1/formulas/evaluate", json=request).status_code == 422
    assert not calls


def test_budget_conversion_uses_density_concentration_base_and_explicit_conversion(extension):
    client, calls = extension
    request = {"formula": {"brief": "green scent", "product_concentration_percent": 1},
        "budget": {"amount": 100, "per_volume_ml": 100, "scope": "finished_product_materials",
            "base_material_cost_krw": 20, "product_density_g_ml": 1, "krw_per_catalog_price_unit": 1300,
            "price_basis_reference": "test-only explicit conversion, not market evidence", "price_as_of": "2026-09-05"}}
    response = client.post("/v1/formulas/evaluate", json=request)
    assert response.status_code == 200
    assert calls[0][0].max_formula_cost_per_kg == pytest.approx(80 / 1.3)
    assert response.json()["candidates"][0]["cost_krw"] == pytest.approx(85)


@pytest.mark.parametrize("extra", [
    {"concentration_range": {"minimum": 1, "maximum": .5}},
    {"edits": {"target_profile": {"unknown": 1}}},
    {"edits": {"target_profile": {"woody": -1}}},
    {"edits": {"excluded_ingredient_ids": ["nonexistent-material"]}},
    {"unexpected": True},
])
def test_bad_inputs_are_rejected(extension, extra):
    client, calls = extension
    assert client.post("/v1/briefs/prepare", json={**body(), **extra}).status_code == 422
    assert not calls


def test_textual_concentration_cannot_silently_override_a_range(extension):
    client, calls = extension
    request = body()
    request["formula"]["brief"] = "우디 향 농도 10%"
    assert client.post("/v1/formulas/evaluate", json=request).status_code == 422
    assert not calls


def test_structured_profile_is_passed_to_engine_not_only_displayed(extension):
    client, calls = extension
    request = {**body(), "edits": {"target_profile": {"citrus": 1}}}
    assert client.post("/v1/formulas/evaluate", json=request).status_code == 200
    assert calls[0][1]["target_profile_override"] == {"citrus": 1}


def test_actual_modal_app_exposes_additive_routes_and_real_recipe_response():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    with TestClient(create_web_app(str(REGISTRY))) as client:
        request = {"formula": {"brief": "clean scent", "max_ingredients": 12}}
        prepared = client.post("/v1/briefs/prepare", json=request)
        assert prepared.status_code == 200 and prepared.json()["status"] == "ready"
        response = client.post("/v1/formulas/evaluate", json=request)
        assert response.status_code == 200
        candidate = response.json()["candidates"][0]
        assert candidate["target_match_score"] == candidate["result"]["calculated_profile_similarity"]
        assert candidate["result"]["score_contract"]["effective_target"] == 95
        assert not candidate["result"]["recipe"]
        assert candidate["regulatory"]["schema_version"] == "regulatory-tabs-1"
        old = client.post("/v1/formulas", json=request["formula"])
        assert old.status_code == 200
        assert old.json()["formula_id"] == candidate["formula_id"]
        assert old.headers["X-Perfumery-Cache"] == "hit"
        assert "candidates" not in old.json()
