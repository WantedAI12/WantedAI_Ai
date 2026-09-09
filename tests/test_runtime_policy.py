"""Strict-95 policy, complete catalog routing, large counts and queued drift."""

from dataclasses import replace
from datetime import date
import hashlib
import json
from types import SimpleNamespace

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.api import TokenAuthorizer, create_app
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog
from fragrance_ai.platform.store import SqliteWorkspaceStore
from fragrance_ai.platform.workspace import FormulaWorkspaceService, QueuedRuntimeMismatch, constraints_from_payload
from fragrance_ai.recommender.catalog import IngredientCatalog, HistoricalReferenceCorpus
from fragrance_ai.recommender.models import MAX_FORMULA_INGREDIENTS
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer
from fragrance_ai.recommender.registry_activation import RegistryActivationReport, write_runtime_catalog
from fragrance_ai.recommender.runtime import MANIFEST_ENV, MANIFEST_HASH_ENV, RuntimeAIFactory, _source_snapshot, _data_snapshot, load_configured_catalog
from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION
from tests._http_client import TestClient


def _extra_catalog():
    catalog = IngredientCatalog.load_builtin()
    extra = replace(catalog.ingredients[0], ingredient_id="test_additional_material", name="Test additional material", aliases=(),
                    formulation_ready=True, blocked=False, max_concentrate_percent=100., risk_tier=1)
    return IngredientCatalog([*catalog.ingredients, extra], catalog.metadata)


def test_strict_95_floor_cannot_be_lowered_by_request_and_does_not_mutate_it(tmp_path):
    request = RecipeConstraints(target_similarity=50, simulation_draws=64, physics_search_population=1)
    with NaturalLanguagePerfumeryAI(minimum_profile_target=95, allow_experimental_safety=False,
            corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("clean scent", request, as_of=date(2026, 9, 5))
        assert ai.require_full_profile_match
        assert request.target_similarity == 50
        assert result.brief.constraints.target_similarity == 95
        assert result.score_contract["runtime_minimum_profile_target"] == 95
        assert result.similarity_score == result.calculated_profile_similarity
        assert not result.full_profile_target_met and not result.recipe
        assert result.closest_candidate
        assert not result.human_similarity_90_claim_authorized
        with pytest.raises(ValueError, match="does not allow"):
            ai.create_recipe("rose scent", replace(request, experimental_disable_safety=True, enable_registry_trace_candidates=True))


@pytest.mark.parametrize("target", [True, float("nan"), 0, 101, "95"])
def test_bad_quality_floor_is_rejected(target):
    with pytest.raises(ValueError, match="minimum_profile_target"):
        NaturalLanguagePerfumeryAI(minimum_profile_target=target)


def test_factory_workspace_and_worker_instance_share_effective_catalog(tmp_path):
    factory = RuntimeAIFactory(catalog=_extra_catalog())
    store = SqliteWorkspaceStore(tmp_path / "shared.db")
    workspace = FormulaWorkspaceService(store=store, ai_factory=factory)
    with factory() as ai:
        worker = FormulaWorkspaceService(store=store, ai_factory=factory, ai_instance=ai)
        assert worker.catalog.ingredients == workspace.catalog.ingredients == ai.catalog.ingredients
        assert any(row["ingredient_id"] == "test_additional_material" for row in workspace.catalog_payload()["ingredients"])
        assert workspace.runtime_contract == worker.runtime_contract == ai.runtime_contract
        assert ai.minimum_profile_target == 95 and ai.require_full_profile_match
    store.close()


def test_manifest_loader_is_hash_source_and_path_bound(tmp_path, monkeypatch):
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    monkeypatch.delenv(MANIFEST_HASH_ENV, raising=False)
    catalog = _extra_catalog()
    blob = tmp_path / "catalog.gz"
    report = RegistryActivationReport("a" * 64, 1, 0, 1, 0, 0, len(catalog.ingredients), 0, 0, 0)
    digest = write_runtime_catalog(blob, catalog, report, {}, wheel_sha256="b" * 64)
    manifest = {"odor_integrity_version": ODOR_INTEGRITY_VERSION, "runtime_source_sha256": _source_snapshot(), "runtime_data_sha256": _data_snapshot(),
        "runtime_catalog": {"path": blob.name, "sha256": digest, "wheel_sha256": "b" * 64, "registry_sha256": "a" * 64}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    restored, bound = load_configured_catalog(path, expected.upper())
    assert restored.ingredients == catalog.ingredients and bound == expected
    with pytest.raises(ValueError, match="both manifest"):
        load_configured_catalog(path)
    with pytest.raises(ValueError, match="hash"):
        load_configured_catalog(path, "f" * 64)
    manifest["runtime_source_sha256"] = {}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different package"):
        load_configured_catalog(path, hashlib.sha256(path.read_bytes()).hexdigest())
    manifest["runtime_source_sha256"] = _source_snapshot()
    manifest["runtime_data_sha256"]["fragrance_ai/data/scientific_properties.db"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="physical-property assets"):
        load_configured_catalog(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_production_cannot_downgrade_to_lower_target(monkeypatch):
    monkeypatch.setenv("PERFUMERY_AI_ENV", "production")
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    monkeypatch.delenv(MANIFEST_HASH_ENV, raising=False)
    with pytest.raises(ValueError, match="at least the 95"):
        RuntimeAIFactory.from_environment(minimum_profile_target=90)
    with pytest.raises(ValueError, match="explicitly pinned"):
        RuntimeAIFactory.from_environment()
    monkeypatch.setattr("fragrance_ai.recommender.runtime.load_configured_catalog", lambda *a: (IngredientCatalog.load_builtin(), "a" * 64))
    factory = RuntimeAIFactory.from_environment()
    assert not factory.allow_experimental_safety


def test_queue_policy_mismatch_and_missing_policy_fail_before_inference(tmp_path):
    factory = RuntimeAIFactory(catalog=_extra_catalog())
    store = SqliteWorkspaceStore(tmp_path / "queue.db")
    workspace = FormulaWorkspaceService(store=store, ai_factory=factory)
    for contract in (None, {**factory.runtime_contract, "minimum_profile_target": 90},
                     {**factory.runtime_contract, "material_snapshot_sha256": "f" * 64}):
        job = SimpleNamespace(payload={"runtime_contract": contract}, kind="recipe.generate")
        with pytest.raises(QueuedRuntimeMismatch):
            workspace.process_job(job)
    store.close()


def test_runtime_policy_binds_model_assets_and_detects_live_snapshot_changes(monkeypatch):
    factory = RuntimeAIFactory(catalog=_extra_catalog())
    assert factory.runtime_contract["runtime_data_sha256"]
    data = _data_snapshot()
    changed = {**data, "fragrance_ai/data/scientific_properties.db": "0" * 64}
    monkeypatch.setattr("fragrance_ai.recommender.runtime._data_snapshot", lambda *a: changed)
    other = RuntimeAIFactory(catalog=_extra_catalog())
    assert other.runtime_contract["runtime_data_sha256"] != factory.runtime_contract["runtime_data_sha256"]
    monkeypatch.setattr("fragrance_ai.recommender.runtime._runtime_metadata_snapshot", lambda: ("changed",))
    with pytest.raises(ValueError, match="snapshot changed"):
        factory()


def test_api_catalog_and_enqueue_include_the_same_verified_runtime_policy(tmp_path):
    factory = RuntimeAIFactory(catalog=_extra_catalog())
    store = SqliteWorkspaceStore(tmp_path / "api.db")
    auth = TokenAuthorizer.from_plaintext({"test-only": ("a", "formulator", "t")})
    audit = AppendOnlyAuditLog(tmp_path / "audit.db", signing_key=b"a" * 32)
    app = create_app(ai_factory=factory, authorizer=auth, audit_log=audit, workspace_store=store, enable_ui=False)
    headers = {"Authorization": "Bearer test-only", "X-Tenant-ID": "t"}
    with TestClient(app) as client:
        catalog = client.get("/v1/catalog", headers=headers)
        assert catalog.status_code == 200
        assert any(row["ingredient_id"] == "test_additional_material" for row in catalog.json()["ingredients"])
        project = client.post("/v1/projects", headers=headers, json={"name": "policy test"}).json()
        queued = client.post("/v1/jobs/recipes", headers=headers,
            json={"project_id": project["project_id"], "brief": "rose scent", "constraints": {"target_similarity": 50, "max_ingredients": 100}})
        assert queued.status_code == 202
        assert queued.json()["payload"]["runtime_contract"] == catalog.json()["runtime_contract"] == factory.runtime_contract
    store.close()
    audit.close()


def test_sdk_workspace_large_maximums_and_manual_editor_are_not_capped_at_30_or_50(tmp_path):
    for count in (51, 500, MAX_FORMULA_INGREDIENTS):
        constraints = constraints_from_payload({"max_ingredients": count, "physics_search_population": 7})
        NaturalLanguagePerfumeryAI._validate_constraints(constraints)
    with pytest.raises(ValueError):
        constraints_from_payload({"max_ingredients": MAX_FORMULA_INGREDIENTS + 1})
    template = IngredientCatalog.load_builtin().ingredients[0]
    items = [replace(template, ingredient_id=f"test_{i}", name=f"test material {i}", aliases=(), formulation_ready=True,
        blocked=False, max_concentrate_percent=100., risk_tier=1) for i in range(40)]
    store = SqliteWorkspaceStore(tmp_path / "edit.db")
    workspace = FormulaWorkspaceService(store=store, ai_factory=lambda: None, catalog=IngredientCatalog(items))
    payload = workspace._manual_payload({"brief": {"constraints": {"max_risk_tier": 1}}},
        [{"ingredient_id": item.ingredient_id, "concentrate_percent": 2.5} for item in items])
    assert len(payload["recipe"]) == 40
    assert sum(row["concentrate_percent"] for row in payload["recipe"]) == 100
    assert payload["manual_edit_requires_recalculation"] and not payload["full_profile_target_met"]
    store.close()


def test_large_maximum_is_not_a_request_for_a_quadratic_all_material_seed():
    template = IngredientCatalog.load_builtin().ingredients[0]
    items = [replace(template, ingredient_id=f"seed_{i}", name=f"seed material {i}", aliases=(), formulation_ready=True,
        blocked=False, max_concentrate_percent=100., pyramid=("top", "heart", "base")[i % 3],
        profile={("citrus", "floral", "woody")[i % 3]: 1.}) for i in range(300)]
    brief = NaturalLanguageBriefParser(IngredientCatalog(items)).parse("citrus floral woody scent", RecipeConstraints(max_ingredients=MAX_FORMULA_INGREDIENTS))
    selected = ConstrainedFormulaOptimizer()._select_candidates(items, brief)
    assert len(selected) <= 12 + len(brief.desired_dimensions)
    assert len(items) == 300 and brief.constraints.max_ingredients == MAX_FORMULA_INGREDIENTS
