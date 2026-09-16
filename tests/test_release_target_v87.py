"""Target defaults and numeric passthrough; fixtures are not model evaluations."""

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import Field

from fragrance_ai.platform import ai_extensions
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest
from tests.test_ai_extensions import Formula
from tests.test_lotion_v21 import fixture as lotion_fixture


class Formula90(Formula):
    target_similarity: float = Field(default=90., gt=0, le=100)


@pytest.fixture
def client90(monkeypatch):
    app = FastAPI()
    _, catalog = lotion_fixture()

    def formula(request, response, **kwargs):
        return {"brief": {"constraints": request.model_dump()}, "formula_id": "fixture-only",
                "recipe": [], "closest_candidate": [{"ingredient_id": "fixture-citrus", "concentrate_percent": 100}],
                "full_profile_target_met": False, "calculated_profile_similarity": 88., "confidence": None}

    def lotion(request, *args, **kwargs):
        return {"target_similarity": request.target_similarity, "effective_target": request.target_similarity,
                "calculated_profile_similarity": 88., "confidence": None,
                "recipe": [], "profile_target_met": False, "status": "fixture_only"}

    monkeypatch.setattr(ai_extensions, "estimate_lotion_recipe", lotion)
    monkeypatch.setattr(ai_extensions, "optimize_lotion", lotion)
    monkeypatch.setattr(ai_extensions, "prepare_lotion_optimization", lambda *a, **k: (lotion(*a, **k), None, None))
    ai_extensions.register_ai_extensions(app, Formula90, catalog, formula, lambda: None,
                                        minimum_profile_target=90.)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("target,expected", [(None, 90.), (90., 90.), (95., 95.)])
def test_prepare_and_evaluate_do_not_raise_target_or_replace_score(client90, target, expected):
    formula = {"brief": "citrus woody scent"}
    if target is not None:
        formula["target_similarity"] = target
    request = {"formula": formula}
    prepared = client90.post("/v1/briefs/prepare", json=request)
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["effective_target"] == expected
    response = client90.post("/v1/formulas/evaluate", json=request)
    assert response.status_code == 200, response.text
    candidate = response.json()["candidates"][0]
    assert candidate["target_match_score"] == 88.
    assert candidate["status"] == "candidate_only"
    assert candidate["result"]["calculated_profile_similarity"] == 88.
    assert candidate["result"]["confidence"] is None


@pytest.mark.parametrize("operation", ["design", "prepare", "optimize"])
@pytest.mark.parametrize("target,expected", [(None, 90.), (90., 90.), (95., 95.)])
def test_all_lotion_defaults_match_served_schema(client90, operation, target, expected):
    request = {"brief": "citrus woody scent"} if operation == "design" else lotion_fixture()[0]
    if target is not None:
        request["target_similarity"] = target
    path = "/v1/applications/body-lotion/" + operation
    schema = client90.get("/openapi.json").json()
    ref = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert schema["components"]["schemas"][ref.rsplit("/", 1)[1]]["properties"]["target_similarity"]["default"] == 90.
    response = client90.post(path, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["effective_target"] == expected
    assert response.json()["calculated_profile_similarity"] == 88.
    assert response.json()["confidence"] is None
    assert response.json()["profile_target_met"] is False


def test_shared_lotion_defaults_are_not_mutated(client90):
    assert LotionEstimateRequest.model_fields["target_similarity"].default == 95.
    assert LotionOptimizationRequest.model_fields["target_similarity"].default == 95.


def test_release_entry_explicitly_selects_90_without_rewriting_response(monkeypatch):
    from deploy import target_runtime_v87
    calls, sentinel = [], object()
    def create(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(target_runtime_v87, "create_local_app", create)
    assert target_runtime_v87.create_release_app(registry_path="test-only.db") is sentinel
    assert calls == [{"language_backend": None, "registry_path": "test-only.db", "minimum_profile_target": 90.}]


def test_modal_entry_uses_new_bundle_and_target_factory_with_existing_auth():
    path = Path(__file__).resolve().parents[1] / "deploy/modal_release_v87.py"
    source = path.read_text(encoding="utf8")
    tree = ast.parse(source)
    imports = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "deploy.target_runtime_v87" in imports
    assert "deploy.runtime_release_v87" in imports
    assert "create_local_app(" not in source
    assert "package-05-target90/wheel/" in source
    release = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "release_app")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "create_release_app" for node in ast.walk(release))
    web = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "web")
    auth = next(node for node in web.decorator_list if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "asgi_app")
    assert {item.arg: ast.literal_eval(item.value) for item in auth.keywords} == {"requires_proxy_auth": True}
    assert "modal.App('perfumery-ai-core')" in source


@pytest.mark.parametrize("target", [89., 101., None, True, "90", float("nan"), float("inf")])
def test_invalid_default_fails_before_runtime_construction(target):
    from deploy.web_app import create_web_app
    with pytest.raises(ValueError, match="API default target"):
        create_web_app(minimum_profile_target=target)
    with pytest.raises(ValueError, match="API default target"):
        ai_extensions.register_ai_extensions(FastAPI(), Formula90, IngredientCatalog([]), None, None,
                                            minimum_profile_target=target)
