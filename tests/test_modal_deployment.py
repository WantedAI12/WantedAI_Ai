from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("modal", reason="Modal CLI dependency is isolated from core runtime")

from deploy.modal_app import REGISTRY_SHA256, WHEEL_SHA256, create_web_app


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"


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
