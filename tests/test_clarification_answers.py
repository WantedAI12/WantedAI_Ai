import copy

import pytest

from tests.test_ai_extensions import extension, body


def clarify(client, request, answers, request_id=None):
    prepared = client.post("/v1/briefs/prepare", json=request)
    assert prepared.status_code == 200
    return client.post("/v1/briefs/clarify", json={"request": request,
        "request_id": request_id or prepared.json()["request_id"], "answers": answers})


def test_missing_brief_answer_reaches_ready_without_inference(extension):
    client, calls = extension
    original = {"formula": {"brief": "..."}, "edits": {"intensity_level": 4,
        "phase_target_profiles": {"drydown": {"woody": 1}}}}
    before = copy.deepcopy(original)
    response = clarify(client, original, {"formula.brief": "green woody scent"})
    assert response.status_code == 200
    result = response.json()
    assert result["prepared"]["status"] == "ready"
    assert result["prepared"]["intent"]["intensity"] == "high"
    assert result["request"]["edits"] == {**original["edits"], "target_profile": None, "excluded_ingredient_ids": []}
    assert result["request"]["formula"]["require_full_profile_match"] is True
    assert not calls and original == before


def test_budget_answers_iterate_and_preserve_unanswered_questions(extension):
    client, calls = extension
    request = {**body(), "budget": {"amount": 4500, "per_volume_ml": 100}}
    prepared = client.post("/v1/briefs/prepare", json=request).json()
    scope = next(q for q in prepared["questions"] if q["id"] == "budget.scope")
    assert len(scope["options"]) == 2
    first = clarify(client, request, {"budget.scope": "finished_product_materials"}).json()
    assert first["prepared"]["status"] == "needs_clarification"
    assert "budget.base_material_cost_krw" in {q["id"] for q in first["prepared"]["questions"]}
    second = clarify(client, first["request"], {"budget.product_density_g_ml": 1,
        "budget.krw_per_catalog_price_unit": 1300, "budget.price_basis_reference": "test fixture, not a market quote",
        "budget.price_as_of": "2026-09-01", "budget.base_material_cost_krw": 100}).json()
    assert second["prepared"]["status"] == "ready"
    assert len(second["prepared"]["scenarios"]) == 3
    assert second["request"]["budget"]["base_material_cost_krw"] == 100
    assert not calls


def test_concentration_range_answer_is_not_lost(extension):
    client, calls = extension
    result = clarify(client, {"formula": {"brief": "woody scent 0.6~0.9%"}},
                     {"concentration_range": {"minimum": .6, "maximum": .9}})
    assert result.status_code == 200
    assert [r["product_concentration_percent"] for r in result.json()["prepared"]["scenarios"]] == [.6, .75, .9]
    assert not calls


def test_stale_request_id_is_rejected(extension):
    client, calls = extension
    request = {"formula": {"brief": "..."}}
    old = client.post("/v1/briefs/prepare", json=request).json()["request_id"]
    request["formula"]["product_concentration_percent"] = 10
    assert clarify(client, request, {"formula.brief": "green scent"}, old).status_code == 409
    assert not calls


@pytest.mark.parametrize("answers", [{}, {"formula.experimental_disable_safety": True},
    {"product_type": "body_lotion"}, {"formula.brief": " "}, {"formula.brief": 2},
    {"formula.brief": "x" * 2001}])
def test_bad_answer_or_policy_injection_is_rejected(extension, answers):
    client, calls = extension
    assert clarify(client, {"formula": {"brief": "..."}}, answers).status_code == 422
    assert not calls


@pytest.mark.parametrize("key,value", [("budget.product_density_g_ml", True),
    ("budget.product_density_g_ml", "1"), ("budget.product_density_g_ml", -1),
    ("budget.price_as_of", "2026-13-01"), ("budget.scope", "whatever")])
def test_answer_types_and_final_model_bounds_are_checked(extension, key, value):
    client, calls = extension
    assert clarify(client, {**body(), "budget": {"amount": 4500, "per_volume_ml": 100}}, {key: value}).status_code == 422
    assert not calls


@pytest.mark.parametrize("preferences,code", [
    ({"skin_type": "sensitive"}, "skin_compatibility_model_missing"),
    ({"ph_minimum": 5, "ph_maximum": 6}, "ph_compatibility_model_missing"),
    ({"note_transition_speed": "fast"}, "transition_speed_control_missing"),
    ({"skin_residual_required": True}, "skin_residual_model_missing"),
    ({"skin_residual_required": False}, "skin_residual_model_missing")])
def test_unsupported_preferences_are_preserved_and_block_evaluation(extension, preferences, code):
    client, calls = extension
    request = {**body(), "product_preferences": preferences}
    result = client.post("/v1/briefs/prepare", json=request).json()
    assert result["product_preferences"] == preferences
    assert result["unsupported_requirements"][0]["code"] == code
    assert client.post("/v1/formulas/evaluate", json=request).status_code == 422
    assert not calls


def test_preference_is_retained_after_answer_and_not_reported_ready(extension):
    client, calls = extension
    response = clarify(client, {"formula": {"brief": "..."}, "product_preferences": {"skin_type": "sensitive"}},
                       {"formula.brief": "green scent"})
    assert response.status_code == 200
    assert response.json()["prepared"]["status"] == "unsupported_requirements"
    assert not calls


@pytest.mark.parametrize("preferences", [{"ph_minimum": 6, "ph_maximum": 5},
    {"ph_minimum": 5}, {"ph_minimum": True, "ph_maximum": 6}, {"ph_minimum": 0, "ph_maximum": 15},
    {"skin_residual_required": "true"}, {"note_transition_speed": "instant"}])
def test_invalid_preferences_are_rejected(extension, preferences):
    client, _ = extension
    assert client.post("/v1/briefs/prepare", json={**body(), "product_preferences": preferences}).status_code == 422


def test_all_skin_context_does_not_assert_skin_compatibility(extension):
    client, _ = extension
    result = client.post("/v1/briefs/prepare", json={**body(), "product_preferences": {"skin_type": "all"}}).json()
    assert result["status"] == "ready"
    capabilities = client.get("/v1/ai/capabilities").json()["features"]
    assert capabilities["typed_clarification_answers"] is True
    assert capabilities["skin_type_compatibility"] is False
