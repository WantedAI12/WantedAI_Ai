import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("modal", reason="Modal CLI dependency is isolated from core runtime")

from deploy.modal_app import REGISTRY_SHA256, WHEEL_SHA256, create_web_app


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REMOTE_EVIDENCE = ROOT / "benchmarks" / "modal_temporal_evolution_v3.json"
RELEASE_MANIFEST = ROOT / "dist" / "temporal-evolution-v3" / "release_manifest.json"


def test_modal_cpu_app_health_catalog_and_formula():
    with TestClient(create_web_app(str(REGISTRY))) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": WHEEL_SHA256,
            "registry_sha256": REGISTRY_SHA256,
        }

        catalog = client.get("/v1/catalog")
        assert catalog.status_code == 200
        assert catalog.json()["reference_molecules"] == 29_240
        assert catalog.json()["safety_screened"] == 29_240
        assert catalog.json()["reference_molecules_connected"] == 29_240
        assert catalog.json()["conditional_trace_candidates_active"] == 29_212
        assert catalog.json()["formulation_ready"] == 29_246
        assert catalog.json()["experimental_formula_candidates"] == 29_259

        response = client.post(
            "/v1/formulas",
            json={
                "brief": "clean fresh citrus woody musk",
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
        assert payload["deployment"]["registry_conditional_trace_active"] == 29_212
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
            == "industrial-registry-public-descriptor-conditional-v2"
            for line in expanded_payload["recipe"]
        )
        assert expanded_payload["safety"]["status"] == "experimental_safety_disabled"


def test_modal_request_schema_rejects_expansion_and_invalid_risk():
    with TestClient(create_web_app(str(REGISTRY))) as client:
        invalid = client.post(
            "/v1/formulas",
            json={
                "brief": "clean woody",
                "max_risk_tier": 3,
                "allow_rare": True,
            },
        )
    assert invalid.status_code == 422


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
