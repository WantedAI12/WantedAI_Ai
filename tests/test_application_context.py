from datetime import date, timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform.application_context import ApplicationContext, assess_application_context
from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_ai_extensions import Formula


def context_data():
    return {"emulsion_type": "oil_in_water", "base_components": [{"name": "water", "mass_percent": 80, "role": "water"},
        {"name": "test oil phase", "mass_percent": 20, "role": "oil"}], "product_density_g_ml": 1.,
        "application_mass_mg_cm2": 2., "temperature_c": 25., "relative_humidity_percent": 50., "substrate": "skin_model",
        "fragrance_concentration_percent": 1., "formula_reference": "test-formula-not-commercial",
        "release_series": {"source_reference": "test fixture only", "source_date": "2026-09-01", "source_kind": "simulated",
            "signal_unit": "instrument_units", "points": [{"minutes": 0, "signal": 100}, {"minutes": 60, "signal": 50}]}}


def test_context_validation_is_not_model_qualification():
    result = assess_application_context(ApplicationContext.model_validate(context_data()))
    assert result["status"] == "ready_for_model_data_review"
    assert not result["source_verified"] and not result["body_lotion_prediction_enabled"]
    assert not result["human_detection_calibration_available"]
    assert result["caller_declared_source_kind"] == "simulated"
    missing = assess_application_context(ApplicationContext())
    assert missing["status"] == "needs_input" and "base_components" in missing["missing_fields"]


@pytest.mark.parametrize("kind", ["sum", "duplicate", "time", "future", "boolean"])
def test_invalid_context_is_rejected(kind):
    data = context_data()
    if kind == "sum": data["base_components"][0]["mass_percent"] = 70
    if kind == "duplicate": data["base_components"][1]["name"] = "water"
    if kind == "time": data["release_series"]["points"][1]["minutes"] = 0
    if kind == "future": data["release_series"]["source_date"] = str(date.today() + timedelta(days=1))
    if kind == "boolean": data["temperature_c"] = True
    with pytest.raises(ValueError): ApplicationContext.model_validate(data)


def test_api_model_persistence_gates_uncertain_results_and_reuses_existing_request_contract():
    app, calls = FastAPI(), []
    def generate(request, response, **kwargs):
        calls.append(request)
        return {"brief": {"constraints": request.model_dump()}, "formula_id": "test", "full_profile_target_met": True,
            "recipe": [{"ingredient_id": "test", "concentrate_percent": 100}], "calculated_profile_similarity": 96,
            "estimated_concentrate_cost_per_kg": 50,
            "temporal_profile": [{"minutes": 0, "relative_to_opening_intensity_percent": 100},
                                 {"minutes": 60, "relative_to_opening_intensity_percent": 70},
                                 {"minutes": 120, "relative_to_opening_intensity_percent": 20}]}
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), generate, lambda: None)
    with TestClient(app) as client:
        request = {"formula": {"brief": "woody scent"}, "model_persistence": {"relative_threshold_percent": 50., "required_duration_minutes": 90.}}
        result = client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]
        assert result["status"] == "candidate_only"
        assert result["model_persistence"]["meets_requirement"] is None
        assert result["longevity_hours"] is None
        request["model_persistence"]["required_duration_minutes"] = 50.
        assert client.post("/v1/formulas/evaluate", json=request).json()["candidates"][0]["status"] == "target_met"
        before = len(calls)
        blocked = client.post("/v1/formulas/evaluate", json={"formula": {"brief": "woody scent"}, "application_context": context_data()})
        assert blocked.status_code == 422 and len(calls) == before
        validated = client.post("/v1/applications/validate", json=context_data())
        assert validated.status_code == 200 and not validated.json()["body_lotion_prediction_enabled"]


def test_real_api_computes_model_persistence_without_repeating_inference_for_threshold_changes():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    with TestClient(create_web_app(str(REGISTRY))) as client:
        request = {"formula": {"brief": "clean scent", "max_ingredients": 12},
                   "model_persistence": {"relative_threshold_percent": 20.}}
        first = client.post("/v1/formulas/evaluate", json=request)
        assert first.status_code == 200
        initial = first.json()["candidates"][0]
        assert initial["model_persistence"]["status"] in {"crossing_estimated", "right_censored", "below_threshold_at_start"}
        assert initial["model_persistence"]["human_detection_guaranteed"] is False
        assert initial["longevity_hours"] is None
        request["model_persistence"]["relative_threshold_percent"] = 10.
        second = client.post("/v1/formulas/evaluate", json=request)
        assert second.status_code == 200 and second.headers["X-Perfumery-Cache"] == "hit"
        assert second.json()["candidates"][0]["target_match_score"] == initial["target_match_score"]
