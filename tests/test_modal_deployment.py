import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("modal", reason="Modal CLI dependency is isolated from core runtime")

from deploy.modal_app import REGISTRY_SHA256, WHEEL_SHA256, create_web_app


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REMOTE_EVIDENCE = ROOT / "benchmarks" / "modal_full_registry_activation_v1.json"
RELEASE_MANIFEST = ROOT / "dist" / "full-registry-activation-v1" / "release_manifest.json"


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
        assert catalog.json()["conditional_trace_candidates_active"] == 637
        assert catalog.json()["formulation_ready"] == 671

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
        assert payload["deployment"]["registry_conditional_trace_active"] == 637

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
            == "industrial-registry-public-descriptor-conditional-v1"
            for line in expanded_payload["recipe"]
        )


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
    assert evidence["coverage"]["conditional_trace_candidates_active"] == 637
    assert evidence["remote_checks"]["expanded_manufacturing_ready"] is False
    assert evidence["remote_checks"]["unauthenticated_status"] == 401
    assert release["wheel"]["bytes"] == wheel.stat().st_size
    assert release["wheel"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert release["registry"]["sha256"] == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    assert release["deployment"]["remote_smoke_passed"] is True
    assert release["supply_chain"]["sbom_sha256"] == hashlib.sha256(sbom.read_bytes()).hexdigest()
    assert release["supply_chain"]["release_policy_sha256"] == hashlib.sha256(
        policy.read_bytes()
    ).hexdigest()
    assert release["supply_chain"]["release_policy_passed"] is True
    assert release["supply_chain"]["wheel_install_verify_sha256"] == hashlib.sha256(
        wheel_verify.read_bytes()
    ).hexdigest()
    assert release["supply_chain"]["wheel_install_verify_passed"] is True
