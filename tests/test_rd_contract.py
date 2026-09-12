"""R&D contract regressions; all review/quote documents below are test fixtures."""
import copy
from datetime import date, timedelta
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.rd_evidence import EvidenceStore, EvidenceAssessment
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_ai_extensions import Formula


POLICY = {"finished_batch_mass_g": 1000., "maximum_lead_time_days": 10,
          "maximum_purchase_cost_usd": 100.}


def reviewed_request():
    return {"request": {"formula": {"brief": "green woody scent", "target_region": "EU",
        "product_category": "eau_de_parfum", "product_concentration_percent": 15.,
        "max_formula_cost_per_kg": 180.}}, "evidence_policy": dict(POLICY)}


@pytest.fixture
def data(tmp_path):
    catalog = IngredientCatalog.load_builtin()
    material = next(r for r in catalog.ingredients if r.cas_number)
    today = date.today()
    old, future = (today - timedelta(days=1)).isoformat(), (today + timedelta(days=30)).isoformat()
    record = {"ingredient_id": material.ingredient_id, "cas_number": material.cas_number,
        "region": "EU", "product_category": "eau_de_parfum", "source_reference": "fixture review",
        "document_sha256": "a" * 64, "reviewer": "fixture reviewer", "reviewed_on": old,
        "valid_until": future, "rule_version": "fixture-rules-1", "frameworks": {"IFRA": "supported", "EU_REACH": "supported"},
        "maximum_finished_product_percent": 30., "supplier": "fixture supplier", "sku": "fixture-sku",
        "quote_reference": "fixture quote", "quote_document_sha256": "b" * 64, "quoted_on": old,
        "quote_valid_until": future, "currency": "USD", "price_per_kg": 50., "available_kg": 100.,
        "minimum_order_kg": .5, "lead_time_days": 3}
    previous = {"version": "previous", "effective_on": old, "records": [copy.deepcopy(record)]}
    current = {"version": "current", "effective_on": today.isoformat(), "records": [copy.deepcopy(record)]}
    bundle = {"schema_version": "rd-evidence-1", "active_version": "current", "snapshots": [previous, current]}
    path = tmp_path / "evidence.json"

    def store(value=None):
        raw = json.dumps(value or bundle).encode()
        path.write_bytes(raw)
        return EvidenceStore(path, hashlib.sha256(raw).hexdigest())

    assessment = {"lines": [{"ingredient_id": material.ingredient_id, "concentrate_percent": 100.}],
                  "target_region": "EU", "product_category": "eau_de_parfum",
                  "product_concentration_percent": 15., "max_formula_cost_per_kg": 180., "policy": dict(POLICY)}
    return catalog, store, bundle, assessment, path


def client_for(catalog, store, *, safety=True, domain=True, real=False):
    calls = []
    app = FastAPI()
    material = next(r for r in catalog.ingredients if r.cas_number)

    def generate(request, response, **kwargs):
        calls.append(kwargs)
        if real:
            from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
            from fragrance_ai.recommender.fixed_assessment import assess_fixed_formula
            constraints = RecipeConstraints(product_category=request.product_category,
                product_concentration_percent=request.product_concentration_percent,
                max_formula_cost_per_kg=request.max_formula_cost_per_kg, target_similarity=95.)
            with NaturalLanguagePerfumeryAI(catalog=catalog, minimum_profile_target=95.) as ai:
                return assess_fixed_formula(ai, request.brief, constraints, kwargs["fixed_formula_weights"]).to_dict()
        return {"brief": {"constraints": request.model_dump()}, "formula_id": material.ingredient_id,
                "recipe": [{"ingredient_id": material.ingredient_id, "concentrate_percent": 100.}],
                "full_profile_target_met": True, "calculated_profile_similarity": 96.,
                "estimated_concentrate_cost_per_kg": 50., "temporal_profile": [],
                "safety": {"internal_gate_passed": safety}, "scientific_model_domain_passed": domain}

    register_ai_extensions(app, Formula, catalog, generate, lambda: None, evidence_store=store)
    return TestClient(app), calls


def test_required_fields_cannot_be_satisfied_by_defaults(data):
    catalog, factory, _, _, _ = data
    client, calls = client_for(catalog, factory())
    with client:
        original = {"request": {"formula": {"brief": "woody scent"}}, "evidence_policy": POLICY}
        value = client.post('/v2/briefs/prepare', json=original).json()
        assert value["status"] == "needs_input" and value["review_id"] is None
        assert "request.formula.product_category" in value["missing_fields"]
        result = client.post('/v2/formulas/evaluate', json={**original, "confirmed_review_id": "0"*64})
        assert result.status_code == 422 and not calls
        assert client.post('/v1/briefs/prepare', json=original["request"]).json()["status"] == "ready"


def test_review_binding_and_missing_evidence_block_before_inference(data):
    catalog, factory, _, _, _ = data
    for store in (factory(), EvidenceStore()):
        client, calls = client_for(catalog, store)
        with client:
            request = reviewed_request()
            reviewed = client.post('/v2/briefs/prepare', json=request).json()
            assert reviewed["prepared"]["interpretation"]["uncertainty"]["calibrated"] is False
            altered = copy.deepcopy(request)
            altered["evidence_policy"]["maximum_purchase_cost_usd"] = 1.
            assert client.post('/v2/formulas/evaluate', json={**altered,
                "confirmed_review_id": reviewed["review_id"]}).status_code == 409
            assert not calls
            if store.bundle is None:
                response = client.post('/v2/formulas/evaluate', json={**request, "confirmed_review_id": reviewed["review_id"]})
                assert response.status_code == 422 and not calls


@pytest.mark.parametrize("safety,domain,accepted", [(True, True, True), (False, True, False), (True, False, False)])
def test_external_evidence_never_overrides_existing_gates(data, safety, domain, accepted):
    catalog, factory, _, _, _ = data
    client, calls = client_for(catalog, factory(), safety=safety, domain=domain)
    with client:
        request = reviewed_request()
        reviewed = client.post('/v2/briefs/prepare', json=request).json()
        response = client.post('/v2/formulas/evaluate', json={**request, "confirmed_review_id": reviewed["review_id"]})
        assert response.status_code == 200, response.text
        result = response.json()
        assert bool(result["candidates"]) == accepted and len(calls) == 1
        candidate = (result["candidates"] or result["diagnostic_candidates"])[0]
        assert candidate["recommendation_allowed"] == accepted
        assert not result["manufacturing_approval"] and not result["state_changed"]


@pytest.mark.parametrize("change,reason", [
    ({"quote_valid_until": (date.today()-timedelta(days=1)).isoformat()}, "quote_not_current"),
    ({"valid_until": (date.today()-timedelta(days=1)).isoformat()}, "regulatory_evidence_not_current"),
    ({"available_kg": .2}, "insufficient_recorded_stock"),
    ({"lead_time_days": 11}, "lead_time_exceeded"),
    ({"frameworks": {"IFRA": "supported"}}, "framework_not_supported:EU_REACH"),
    ({"cas_number": "wrong"}, "material_identity_mismatch"),
    ({"maximum_finished_product_percent": 10.}, "reviewed_use_limit_exceeded"),
    ({"price_per_kg": 301.}, "ingredient_price_limit_exceeded"),
])
def test_changed_evidence_is_reported_with_values_and_blockers(data, change, reason):
    catalog, factory, bundle, request, _ = data
    bundle["snapshots"][1]["records"][0].update(change)
    client, calls = client_for(catalog, factory(bundle))
    with client:
        response = client.post('/v2/formulas/change-impact', json={**request, "previous_evidence_version": "previous"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["before"]["gate_passed"] and not result["after"]["gate_passed"]
        assert result["review_required"] and result["affected_material_count"] == 1
        assert reason in result["changes"][0]["current_blockers"]
        assert result["changes"][0]["changes"] and not calls and not result["state_changed"]


def test_moq_and_missing_identity_or_scope_are_fail_closed(data):
    catalog, factory, _, request, _ = data
    store = factory()
    result = store.assess(EvidenceAssessment(**request), catalog)
    assert result["materials"][0]["required_kg"] == pytest.approx(.15)
    assert result["materials"][0]["purchase_kg"] == .5
    assert result["purchase_cost_usd"] == 25.
    for patch in ({"target_region": "KR"}, {"product_category": "body_lotion"},
                  {"lines": [{"ingredient_id": "unknown", "concentrate_percent": 100.}]}):
        result = store.assess(EvidenceAssessment(**{**request, **patch}), catalog)
        assert not result["gate_passed"] and result["quoted_formula_cost_usd_per_kg"] is None


def test_bundle_hash_duplicates_and_request_injection_are_rejected(data):
    catalog, factory, bundle, request, path = data
    store = factory()
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        store.assess(EvidenceAssessment(**request), catalog)
    with pytest.raises(ValueError, match="together"):
        EvidenceStore(path)
    bundle["snapshots"][1]["records"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        factory(bundle)
    with pytest.raises(ValueError):
        EvidenceAssessment(**{**request, "lines": request["lines"] * 2})
    with pytest.raises(ValueError):
        EvidenceAssessment(**{**request, "lines": [{**request["lines"][0], "concentrate_percent": float('nan')}]})
    with pytest.raises(ValueError):
        EvidenceAssessment(**{**request, "approved": True})


def test_fixed_weights_require_their_own_review_and_actual_engine_runs(data):
    catalog, factory, bundle, _, _ = data
    weights = {"dihydromyrcenol": 25., "hedione": 35., "linalyl_acetate": 20., "phenethyl_alcohol": 20.}
    materials = {r.ingredient_id: r for r in catalog.ingredients}
    for snapshot in bundle["snapshots"]:
        template = snapshot["records"][0]
        snapshot["records"] = [{**template, "ingredient_id": key, "cas_number": materials[key].cas_number} for key in weights]
    client, calls = client_for(catalog, factory(bundle), real=True)
    with client:
        request = {**reviewed_request(), "lines": [
            {"ingredient_id": key, "concentrate_percent": value} for key, value in weights.items()]}
        reviewed = client.post('/v2/briefs/prepare', json=request).json()
        changed = copy.deepcopy(request)
        changed["lines"][0]["concentrate_percent"] -= 1.
        changed["lines"][1]["concentrate_percent"] += 1.
        assert client.post('/v2/formulas/reassess', json={**changed,
            "confirmed_review_id": reviewed["review_id"]}).status_code == 409
        assert not calls
        response = client.post('/v2/formulas/reassess', json={**request, "confirmed_review_id": reviewed["review_id"]})
        assert response.status_code == 200, response.text
        result = response.json()
        candidate = (result["candidates"] or result["diagnostic_candidates"])[0]
        assert candidate["result"]["scientific_model_version"]
        assert candidate["result"]["score_contract"]["full_generator_approval"] is False
        assert {r["ingredient_id"]: r["concentrate_percent"] for r in candidate["result"]["closest_candidate"]} == weights
        assert len(calls) == 1


@pytest.mark.parametrize("brief,field", [("woody scent concentration 3%", "product_concentration_percent"),
                                         ("우디 향 한국 판매", "target_region")])
def test_explicit_conditions_conflicting_with_text_require_review(data, brief, field):
    catalog, factory, _, _, _ = data
    client, calls = client_for(catalog, factory())
    with client:
        request = reviewed_request()
        request["request"]["formula"]["brief"] = brief
        result = client.post('/v2/briefs/prepare', json=request).json()
        assert result["status"] == "needs_clarification" and result["review_id"] is None
        assert "request.formula." + field in result["conflicting_fields"] and not calls
