import base64
import hashlib
import json
from datetime import date
from importlib import resources

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    signing_payload,
)
from fragrance_ai.recommender.concentration_response import (
    CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
    ConcentrationResponseTrustDecision,
    FrozenConcentrationResponse,
)
from fragrance_ai.recommender.physsim import ConcentrationAwarePhysSim
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI


def test_concentration_response_artifact_is_release_gated_and_hashed():
    data = resources.files("fragrance_ai").joinpath("data")
    manifest = json.loads(
        data.joinpath("concentration_response_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    payload = data.joinpath(manifest["runtime_file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["runtime_sha256"]
    runtime = json.loads(payload)
    assert runtime["allow_pickle"] is False
    assert runtime["source_training_artifact_required_at_runtime"] is False
    assert manifest["distribution_contract"]["source_model_packaged"] is False
    assert manifest["distribution_contract"]["pickle_deserialization_allowed"] is False
    assert manifest["release_gate"]["passed"] is True
    assert all(manifest["release_gate"]["checks"].values())
    assert manifest["algorithm"] == "concentration_only_ridge"
    assert manifest["structure_specific_weight"] == 0.0


def test_measured_global_response_increases_across_measured_dilution_range():
    model = FrozenConcentrationResponse()
    low, low_domain = model.intensity(0.0001)
    high, high_domain = model.intensity(0.1)
    assert low_domain and high_domain
    assert high > low
    assert model.approved_primary_score_weight == 0.0


def test_only_an_independently_signed_authorization_can_enable_primary_weight():
    with pytest.raises(TypeError):
        ConcentrationResponseTrustDecision(
            model_sha256="a" * 64,
            manifest_sha256="b" * 64,
            approved_primary_score_weight=0.05,
            authorization_artifact_id="forged",
            signer_id="forged",
        )
    data = resources.files("fragrance_ai").joinpath("data")
    manifest_bytes = data.joinpath("concentration_response_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    model_bytes = data.joinpath(manifest["runtime_file"]).read_bytes()
    private_key = Ed25519PrivateKey.generate()
    public_key = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )
    scope = {
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "algorithm": manifest["algorithm"],
        "approved_primary_score_weight": 0.05,
    }
    envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": "concentration-release-1",
        "artifact_type": CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
        "signer_id": "independent-model-reviewer",
        "signer_role": "model_release_approver",
        "scope": scope,
        "issued_at": "2026-07-01T00:00:00+00:00",
        "expires_at": "2027-07-01T00:00:00+00:00",
        "artifact_hashes": {
            "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
            "model": hashlib.sha256(model_bytes).hexdigest(),
        },
    }
    envelope["signature"] = base64.b64encode(
        private_key.sign(signing_payload(envelope))
    ).decode("ascii")
    decision = ConcentrationResponseTrustDecision.from_signed_authorization(
        {
            "signers": {
                "independent-model-reviewer": {
                    "public_key": public_key,
                    "roles": ["model_release_approver"],
                    "artifact_types": [
                        CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE
                    ],
                }
            }
        },
        envelope,
        as_of=date(2026, 7, 28),
    )
    assert FrozenConcentrationResponse().approved_primary_score_weight == 0.0
    assert FrozenConcentrationResponse(decision).approved_primary_score_weight == 0.05

    ai = NaturalLanguagePerfumeryAI()
    recipe = ai.create_recipe("깨끗하고 시원한 시트러스 우디", as_of=date(2026, 7, 28))
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    trusted = ConcentrationAwarePhysSim(
        concentration_response=FrozenConcentrationResponse(decision)
    ).evaluate(
        recipe.closest_candidate,
        ingredients,
        recipe.brief,
        ai.scientific_store,
        reference_target_lines=recipe.closest_candidate,
    )
    assert trusted.concentration_response_status == "validation_gated"
    assert trusted.concentration_response_applied_weight == 0.05
    assert "concentration_response_authorized_for_primary_score" in trusted.flags

    expired_envelope = dict(envelope)
    expired_envelope["artifact_id"] = "concentration-release-expired"
    expired_envelope["expires_at"] = "2026-07-02T00:00:00+00:00"
    expired_envelope["signature"] = base64.b64encode(
        private_key.sign(signing_payload(expired_envelope))
    ).decode("ascii")
    expired_decision = ConcentrationResponseTrustDecision.from_signed_authorization(
        {
            "signers": {
                "independent-model-reviewer": {
                    "public_key": public_key,
                    "roles": ["model_release_approver"],
                    "artifact_types": [
                        CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE
                    ],
                }
            }
        },
        expired_envelope,
        as_of=date(2026, 7, 1),
    )
    assert FrozenConcentrationResponse(expired_decision).approved_primary_score_weight == 0.0
