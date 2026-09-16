import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("modal", reason="Modal CLI dependency is isolated from core runtime")

from deploy.web_app import REGISTRY_SHA256, create_web_app
from fragrance_ai.recommender import runtime
from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry
from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION
from fragrance_ai.recommender.registry_activation import write_runtime_catalog
from tests.test_registry_activation import REGISTRY_ONLY_ACTIVE, _activated_catalog


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REMOTE_EVIDENCE = ROOT / "benchmarks" / "modal_temporal_evolution_v3.json"
RELEASE_MANIFEST = ROOT / "dist" / "temporal-evolution-v3" / "release_manifest.json"


@pytest.fixture(scope="module")
def runtime_bundle(tmp_path_factory):
    """Real current wheel/loader; only already-published registry inputs.

    The private V32 source-enriched catalog is not a clean-checkout dependency.
    This fixture's coverage is deliberately distinguished from that deployment.
    """
    root = tmp_path_factory.mktemp("published-registry-runtime")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(root / "wheel")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    wheel = next((root / "wheel").glob("*.whl"))
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    catalog, report = _activated_catalog()
    with IndustrialIngredientRegistry(REGISTRY) as registry:
        stats = registry.stats()
    blob = root / "runtime_catalog.json.gz"
    catalog_sha = write_runtime_catalog(blob, catalog, report, stats, wheel_sha256=wheel_sha)
    manifest = {
        "odor_integrity_version": ODOR_INTEGRITY_VERSION,
        "runtime_source_sha256": runtime._source_snapshot(),
        "runtime_data_sha256": runtime._data_snapshot(),
        "runtime_catalog": {"path": blob.name, "sha256": catalog_sha,
                            "wheel_sha256": wheel_sha, "registry_sha256": REGISTRY_SHA256},
    }
    path = root / "catalog_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "wheel": wheel, "wheel_sha256": wheel_sha, "manifest": manifest}


@pytest.fixture
def api_runtime(runtime_bundle, monkeypatch):
    # Developer profiles and service secrets must not select a different model
    # or catalog when these same tests are run locally and on GitHub.
    for name in list(__import__("os").environ):
        if name.startswith("PERFUMERY_AI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PERFUMERY_AI_LOCAL_PROFILE", "disabled")

    def create(**kwargs):
        return create_web_app(
            str(REGISTRY), catalog_manifest_path=runtime_bundle["path"],
            catalog_manifest_sha256=runtime_bundle["sha256"], **kwargs,
        )

    return create


def test_modal_cpu_app_health_catalog_and_formula(api_runtime, runtime_bundle):
    with TestClient(api_runtime(allow_legacy_research=True)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": runtime_bundle["wheel_sha256"],
            "registry_sha256": REGISTRY_SHA256,
        }

        catalog = client.get("/v1/catalog")
        assert catalog.status_code == 200
        assert catalog.json()["reference_molecules"] == 29_240
        assert catalog.json()["safety_screened"] == 29_240
        assert catalog.json()["reference_molecules_connected"] == 29_240
        assert catalog.json()["conditional_trace_candidates_active"] == REGISTRY_ONLY_ACTIVE
        assert catalog.json()["formulation_ready"] == REGISTRY_ONLY_ACTIVE + 34
        assert catalog.json()["experimental_formula_candidates"] == REGISTRY_ONLY_ACTIVE + 34
        assert catalog.json()["connected_catalog_rows"] == 29_259
        assert catalog.json()["integrity_counts"]["reported_odorless"] == 318
        assert catalog.json()["runtime_binding"]["catalog_manifest_sha256"] == runtime_bundle["sha256"]

        response = client.post(
            "/v1/formulas",
            json={
                "brief": "clean fresh citrus woody musk",
                "require_full_profile_match": False,
                "target_similarity": 90,
                "max_risk_tier": 1,
                "target_region": "EU",
                "product_category": "eau_de_parfum",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["recipe"]
        assert payload["safety"]["internal_gate_passed"] is True
        assert payload["deployment"]["provider"] == "modal"
        assert payload["deployment"]["gpu_required"] is False
        assert payload["deployment"]["registry_connected_total"] == 29_240
        assert payload["deployment"]["registry_conditional_trace_active"] == REGISTRY_ONLY_ACTIVE
        assert payload["temporal_timepoints_minutes"] == [0, 15, 60, 240, 480]
        assert [point["phase"] for point in payload["temporal_profile"]] == [
            "opening",
            "opening",
            "heart",
            "heart",
            "drydown",
        ]
        assert payload["ingredient_temporal_profile"]
        assert payload["ingredient_temporal_profile"][0]["points"][0][
            "application_surface_remaining_fraction_percent"
        ] == 100.0
        assert payload["temporal_concentration_basis"] == (
            "first_order_application_surface_evaporation_proxy"
        )

        expanded = client.post(
            "/v1/formulas",
            json={
                "brief": "smoky leathery woody dry fragrance",
                "require_full_profile_match": False,
                "experimental_disable_safety": True,
                "max_risk_tier": 2,
                "enable_registry_trace_candidates": True,
                "target_similarity": 50,
                "target_region": "EU",
                "product_category": "eau_de_parfum",
            },
        )
        assert expanded.status_code == 200
        expanded_payload = expanded.json()
        assert expanded_payload["status"] == "experimental_registry_candidate"
        assert any(
            line["data_source"]
            == "industrial-registry-public-descriptor-conditional-v3"
            for line in expanded_payload["recipe"]
        )
        assert expanded_payload["safety"]["status"] == "experimental_safety_disabled"


def test_modal_request_schema_rejects_expansion_and_invalid_risk(api_runtime):
    with TestClient(api_runtime()) as client:
        invalid = client.post(
            "/v1/formulas",
            json={
                "brief": "clean woody",
                "max_risk_tier": 3,
                "allow_rare": True,
            },
        )
    assert invalid.status_code == 422


def test_actual_formula_stream_matches_json_and_audit_routes_do_not_run_ai(api_runtime, monkeypatch):
    from fragrance_ai import NaturalLanguagePerfumeryAI
    from tests.test_audit_reporting_api import history, parse_sse
    body = {"brief": "clean fresh citrus woody musk", "require_full_profile_match": True}
    with TestClient(api_runtime()) as client:
        streamed = client.post("/v1/formulas/stream", json=body)
        assert streamed.status_code == 200
        events = parse_sse(streamed.text)
        stages = [row["data"]["stage"] for row in events if row["event"] == "progress"]
        assert stages == ["RECEIVED", "INGREDIENT_SCREENING", "SAFETY_CHECK", "RATIO_OPTIMIZATION", "TEMPORAL_PROFILE", "DONE"]
        result = next(row["data"] for row in events if row["event"] == "result")
        ordinary = client.post("/v1/formulas", json=body)
        assert ordinary.status_code == 200 and ordinary.json() == result
        assert ordinary.headers["X-Perfumery-Cache"] == "hit"
        assert result["score_contract"]["runtime_minimum_profile_target"] == 95
        if not result["full_profile_target_met"]:
            assert result['target_match_met'] is False
            assert result['target_match_score'] == result['calculated_profile_similarity']
            assert result['recipe_delivery']['approval_inferred'] is False
            if result['recipe_delivery']['returned']:
                assert result['recipe'] == result['closest_candidate']
                assert result['recipe_delivery']['status'] == 'target_not_met'
            else:
                assert result['recipe'] == []
        monkeypatch.setattr(NaturalLanguagePerfumeryAI, "create_recipe", lambda *a, **kw: pytest.fail("audit invoked recipe inference"))
        before = client.app.state.formula_cache.stats()["computations"]
        for path in ("/v1/reports/audit", "/v1/audit-logs/stream", "/v1/reports/audit/stream"):
            assert client.post(path, json=history()).status_code == 200
        assert client.app.state.formula_cache.stats()["computations"] == before


def test_modal_strict_profile_gate_does_not_fabricate_scores_and_separates_cached_modes(api_runtime):
    with TestClient(api_runtime(allow_legacy_research=True)) as client:
        body = {"brief": "clean scent", "require_full_profile_match": False, "target_similarity": 95}
        ordinary = client.post("/v1/formulas", json=body)
        strict = client.post("/v1/formulas", json={**body, "require_full_profile_match": True})
        assert ordinary.status_code == strict.status_code == 200
        before, after = ordinary.json(), strict.json()
        assert before["calculated_profile_similarity"] < 90
        assert after["calculated_profile_similarity"] < 90
        assert after["score_contract"]["runtime_minimum_profile_target"] == 95
        assert after["brief"]["constraints"]["target_similarity"] == 95
        assert not after["full_profile_target_met"]
        assert after["closest_candidate"]
        assert after['recipe'] == after['closest_candidate']
        assert after['recipe_delivery']['status'] == 'target_not_met'
        assert not after['recipe_delivery']['approval_inferred']
        assert after['assessment_status'] == 'no_safe_match'
        assert after['target_match_score'] == after['calculated_profile_similarity']
        assert after['target_match_met'] is False
        assert strict.headers["X-Perfumery-Cache"] == "miss"
        assert client.post("/v1/formulas", json=body).json() == before


def test_full_registry_release_evidence_matches_sealed_artifacts():
    evidence = json.loads(REMOTE_EVIDENCE.read_text(encoding="utf-8"))
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    wheel = RELEASE_MANIFEST.parent / release["wheel"]["path"]
    sbom = RELEASE_MANIFEST.parent / release["supply_chain"]["sbom_path"]
    policy = RELEASE_MANIFEST.parent / release["supply_chain"]["release_policy_path"]
    wheel_verify = (
        RELEASE_MANIFEST.parent
        / release["supply_chain"]["wheel_install_verify_path"]
    )

    assert evidence["coverage"]["reference_molecules_connected"] == 29_240
    assert evidence["coverage"]["unlinked_registry_candidates_active"] == 29_212
    assert evidence["remote_checks"]["unauthenticated_status"] == 401
    assert evidence["remote_checks"]["timepoints_minutes"] == [0, 15, 60, 240, 480]
    assert evidence["remote_checks"]["scent_dimensions_per_timepoint"] == 19
    assert evidence["remote_checks"]["remaining_concentration_monotonic"] is True
    assert evidence["remote_checks"]["headspace_and_odor_contribution_sums_100"] is True
    assert release["wheel"]["bytes"] == wheel.stat().st_size
    assert release["wheel"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert release["registry"]["sha256"] == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    assert release["deployment"]["remote_smoke_passed"] is True
    assert release["temporal_output"]["model_version"] == (
        "headspace-olfactory-twin-2.2"
    )
    assert release["deployment"]["remote_evidence_sha256"] == hashlib.sha256(
        REMOTE_EVIDENCE.read_bytes()
    ).hexdigest()
    assert release["supply_chain"]["sbom_sha256"] == hashlib.sha256(sbom.read_bytes()).hexdigest()
    assert release["supply_chain"]["release_policy_sha256"] == hashlib.sha256(
        policy.read_bytes()
    ).hexdigest()
    assert release["supply_chain"]["release_policy_passed"] is True
    assert release["supply_chain"]["wheel_install_verify_sha256"] == hashlib.sha256(
        wheel_verify.read_bytes()
    ).hexdigest()
    assert release["supply_chain"]["wheel_install_verify_passed"] is True


def test_deployed_v4_has_matching_quality_and_artifact_evidence():
    # The deployed release is immutable evidence. A locally prepared candidate
    # is checked separately and must not be presented as remotely deployed.
    deployed_root = ROOT / "dist" / "performance-quality-v4"
    release = json.loads((deployed_root / "release_manifest.json").read_text(encoding="utf-8"))
    deployed_wheel = deployed_root / release["wheel"]["path"]
    deployed_hash = release["wheel"]["sha256"]
    deployment = release["deployment"]
    evidence_path = deployed_root / deployment["remote_evidence_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert deployed_hash == "f78a5c6bb60ebbdcc92314e58d12aae8f10d168f2be55fb908971a2780d9048a"
    assert hashlib.sha256(deployed_wheel.read_bytes()).hexdigest() == deployed_hash
    assert evidence["wheel_sha256"] == deployed_hash
    assert evidence["passed"] is True
    assert evidence["excluded_material_absent"] is True
    assert evidence["english_max_six_actual_lines"] <= 6
    assert evidence["korean_max_six_actual_lines"] <= 6
    assert evidence["formula_requests_passed"] == 9
    assert evidence["temporary_verification_token_deleted"] is True
    assert deployment["remote_evidence_sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    for label in ("sbom", "release_policy", "wheel_install_verify"):
        artifact = release["supply_chain"][label]
        path = deployed_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_prepared_candidate_wheel_and_catalog_are_bound(runtime_bundle):
    from scripts.build_odor_integrity_catalog import verify_wheel_sources
    assert hashlib.sha256(runtime_bundle["wheel"].read_bytes()).hexdigest() == runtime_bundle["wheel_sha256"]
    assert verify_wheel_sources(runtime_bundle["wheel"]) == runtime_bundle["manifest"]["runtime_source_sha256"]
    loaded = runtime.load_verified_catalog_bundle(runtime_bundle["path"], runtime_bundle["sha256"])
    catalog, activation, stats = loaded["catalog"], loaded["activation_report"], loaded["registry_stats"]
    assert len(catalog.ingredients) == 29_259
    assert activation.reference_molecules_connected == 29_240
    assert activation.active_odorant_candidates == REGISTRY_ONLY_ACTIVE + 34
    assert activation.conditional_trace_candidates_active == REGISTRY_ONLY_ACTIVE
    assert stats["reference_molecules"] == 29_240
    with pytest.raises(ValueError, match="hash"):
        runtime.load_verified_catalog_bundle(runtime_bundle["path"], "0" * 64)


def test_actual_api_coalesces_requests_and_bypasses_mutable_state(monkeypatch, api_runtime):
    from concurrent.futures import ThreadPoolExecutor
    from fragrance_ai import NaturalLanguagePerfumeryAI
    import threading
    for name in list(__import__("os").environ):
        if name.startswith("PERFUMERY_AI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PERFUMERY_AI_LOCAL_PROFILE", "disabled")
    calls = []
    lock = threading.Lock()
    actual = NaturalLanguagePerfumeryAI.create_recipe
    def counted(self, *args, **kwargs):
        with lock:
            calls.append(kwargs["as_of"])
        return actual(self, *args, **kwargs)
    monkeypatch.setattr(NaturalLanguagePerfumeryAI, "create_recipe", counted)
    app = api_runtime()
    body = {"brief":"clean fresh citrus woody musk"}
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _:client.post("/v1/formulas",json=body),range(4)))
        assert all(response.status_code == 200 for response in responses)
        assert len(calls) == 1
        assert all(response.json() == responses[0].json() for response in responses)
        assert client.get("/health").status_code == 200
        assert client.post("/v1/formulas",json=body).headers["X-Perfumery-Cache"] == "hit"
        changed = client.post("/v1/formulas",json={**body,"max_ingredients":6})
        assert changed.status_code == 200
        assert len(calls) == 2
        monkeypatch.setenv("PERFUMERY_AI_OPTIONAL_STATE_TEST","enabled")
        for _ in range(2):
            response = client.post("/v1/formulas",json=body)
            assert response.status_code == 200
            assert response.headers["X-Perfumery-Cache"] == "bypass"
        assert len(calls) == 4


def test_phase_api_returns_recipes_with_unchanged_default_threshold(api_runtime):
    with TestClient(api_runtime(allow_legacy_research=True)) as client:
        first = client.post("/v1/formulas", json={"brief":"opening clean fresh citrus, drydown woody musk", "require_full_profile_match":False, "target_similarity":90})
        reverse = client.post("/v1/formulas", json={"brief":"opening woody musk, drydown clean fresh citrus", "require_full_profile_match":False, "target_similarity":90})
    for response in (first, reverse):
        assert response.status_code == 200
        payload = response.json()
        assert payload["recipe"]
        assert payload["brief"]["constraints"]["target_similarity"] == 90.0
        assert payload["similarity_score"] >= 90.0
        assert payload["brief"]["phase_target_profiles"]
    assert first.json()["formula_id"] != reverse.json()["formula_id"]


def test_shared_factory_95_point_gate_is_independent_of_recipe_delivery(api_runtime):
    with TestClient(api_runtime()) as client:
        response = client.post("/v1/formulas", json={"brief": "clean scent"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["brief"]["constraints"]["target_similarity"] == 95
        assert payload["brief"]["constraints"]["simulation_draws"] == 200
        assert not payload["brief"]["constraints"]["experimental_disable_safety"]
        assert payload["score_contract"]["strict_full_profile_gate"]
        assert payload['recipe'] == payload['closest_candidate'] and payload['recipe']
        assert payload['recipe_delivery']['status'] == 'target_not_met'
        assert not payload['recipe_delivery']['approval_inferred']
        assert payload['target_match_met'] is False
        assert payload["similarity_score"] == payload["calculated_profile_similarity"] < 95
        downgraded = client.post("/v1/formulas", json={"brief": "rose scent", "require_full_profile_match": False,
            "target_similarity": 50, "experimental_disable_safety": True, "enable_registry_trace_candidates": True})
        assert downgraded.status_code == 422
