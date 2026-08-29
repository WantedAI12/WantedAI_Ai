"""Fail-closed tests for signed quality and sensory evidence."""

from __future__ import annotations

import base64
from datetime import date

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    signing_payload,
    sha256_file,
)
from fragrance_ai.recommender.manufacturing import REQUIRED_STABILITY_TESTS
from fragrance_ai.recommender.quality import (
    QUALITY_ARTIFACT_TYPE,
    QualityEvidenceStore,
)
from fragrance_ai.recommender.sensory import (
    SENSORY_ARTIFACT_TYPE,
    SensoryEvaluationStore,
)


AS_OF = date(2026, 7, 28)


def _signer(role: str, artifact_type: str):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    root = {
        "signers": {
            "independent-lab": {
                "public_key": public,
                "roles": [role],
                "artifact_types": [artifact_type],
            }
        }
    }
    return private, root


def _envelope(private, artifact_type: str, scope: dict, artifact_paths: dict, artifact_id: str, role: str):
    envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "signer_id": "independent-lab",
        "signer_role": role,
        "scope": scope,
        "issued_at": "2026-07-01T00:00:00+00:00",
        "expires_at": "2027-07-01T00:00:00+00:00",
        "artifact_hashes": {key: sha256_file(path) for key, path in artifact_paths.items()},
    }
    envelope["signature"] = base64.b64encode(private.sign(signing_payload(envelope))).decode()
    return envelope


def test_quality_legacy_rows_are_audit_only_and_verified_reports_rehash(tmp_path):
    private, root = _signer("quality_laboratory", QUALITY_ARTIFACT_TYPE)
    store = QualityEvidenceStore(trusted_signers=root)
    scope = "sha256:" + "a" * 64
    store.record(scope, REQUIRED_STABILITY_TESTS[0], "pass", AS_OF, "old-protocol", "old-report")
    assert store.assess(scope, as_of=AS_OF).status == "legacy_unverified"
    assert not store.assess(scope, as_of=AS_OF).passed

    report_paths = []
    for index, test_name in enumerate(REQUIRED_STABILITY_TESTS):
        protocol = tmp_path / f"{index}-protocol.pdf"
        report = tmp_path / f"{index}-report.pdf"
        protocol.write_bytes(f"protocol/{test_name}".encode())
        report.write_bytes(f"report/{test_name}".encode())
        report_paths.append(report)
        envelope = _envelope(
            private,
            QUALITY_ARTIFACT_TYPE,
            {
                "release_spec_id": scope,
                "test_name": test_name,
                "result": "pass",
                "completed_on": AS_OF.isoformat(),
            },
            {"protocol": protocol, "report": report},
            f"quality-{index}",
            "quality_laboratory",
        )
        store.record_verified(scope, test_name, "pass", AS_OF, protocol, report, envelope, as_of=AS_OF)

    assert store.assess(scope, as_of=AS_OF).passed
    report_paths[0].write_bytes(b"changed after signature")
    assessment = store.assess(scope, as_of=AS_OF)
    assert not assessment.passed
    assert assessment.status == "verified_evidence_invalid"
    assert assessment.verification_failures


def test_quality_rejects_future_unknown_and_revoked_artifacts(tmp_path):
    private, root = _signer("quality_laboratory", QUALITY_ARTIFACT_TYPE)
    store = QualityEvidenceStore(trusted_signers=root)
    scope = "sha256:" + "b" * 64
    protocol = tmp_path / "protocol.pdf"
    report = tmp_path / "report.pdf"
    protocol.write_bytes(b"protocol")
    report.write_bytes(b"report")
    envelope = _envelope(
        private, QUALITY_ARTIFACT_TYPE,
        {"release_spec_id": scope, "test_name": REQUIRED_STABILITY_TESTS[0], "result": "pass", "completed_on": AS_OF.isoformat()},
        {"protocol": protocol, "report": report}, "revoke-me", "quality_laboratory",
    )
    future = dict(envelope)
    future["issued_at"] = "2028-01-01T00:00:00+00:00"
    future["expires_at"] = "2029-01-01T00:00:00+00:00"
    future["signature"] = base64.b64encode(private.sign(signing_payload(future))).decode()
    with pytest.raises(ValueError, match="not yet effective"):
        store.record_verified(scope, REQUIRED_STABILITY_TESTS[0], "pass", AS_OF, protocol, report, future, as_of=AS_OF)
    unknown = dict(envelope)
    unknown["signer_id"] = "not-allowlisted"
    unknown["signature"] = base64.b64encode(private.sign(signing_payload(unknown))).decode()
    with pytest.raises(ValueError, match="allowlist"):
        store.record_verified(scope, REQUIRED_STABILITY_TESTS[0], "pass", AS_OF, protocol, report, unknown, as_of=AS_OF)
    store.record_verified(scope, REQUIRED_STABILITY_TESTS[0], "pass", AS_OF, protocol, report, envelope, as_of=AS_OF)
    store.revoke("revoke-me", AS_OF, "certificate withdrawn")
    assert store.assess(scope, as_of=AS_OF).status == "verified_evidence_invalid"


def test_sensory_direct_rows_are_legacy_and_signed_study_is_rechecked(tmp_path):
    private, root = _signer("sensory_laboratory", SENSORY_ARTIFACT_TYPE)
    store = SensoryEvaluationStore(trusted_signers=root)
    formula_a = "sha256:" + "c" * 64
    formula_b = "sha256:" + "d" * 64
    store.register_formula(formula_a, "clean citrus", 96.0, [])
    store.register_formula(formula_b, "clean citrus", 80.0, [])
    mapping = store.create_study("clean citrus", [formula_a, formula_b], "legacy-protocol", seed=1)
    study_id = mapping.pop("study_id")
    code_a = next(code for code, formula in mapping.items() if formula == formula_a)
    store.record_evaluation(study_id, code_a, "P-legacy", 99.0, 80.0, expert=True)
    legacy = store.formula_evidence(formula_a, min_panelists=1, min_experts=1, as_of=AS_OF)
    assert legacy.status == "legacy_unverified"
    assert not legacy.passed

    data = tmp_path / "signed-study.csv"
    rows = ["study_id,blind_code,panelist_id,similarity,liking,expert"]
    rows.extend(f"{study_id},{code_a},P-{index},96,80,{str(index < 3).lower()}" for index in range(12))
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")
    envelope = _envelope(
        private, SENSORY_ARTIFACT_TYPE,
        {"study_id": study_id, "formula_ids": sorted([formula_a, formula_b])},
        {"study_data": data}, "study-verified-1", "sensory_laboratory",
    )
    assert store.import_verified(study_id, data, envelope, as_of=AS_OF) == 12
    evidence = store.formula_evidence(formula_a, min_panelists=12, min_experts=3, as_of=AS_OF)
    assert evidence.passed
    assert evidence.status == "verified_passed"
    assert evidence.unique_panelists == 12
    assert evidence.expert_panelists == 3
    assert evidence.lower_confidence_bound_95 == 96.0
    expert_shortfall = store.formula_evidence(
        formula_a, min_panelists=12, min_experts=4, as_of=AS_OF
    )
    assert not expert_shortfall.passed
    assert expert_shortfall.status == "insufficient_verified_experts"
    data.write_text(data.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    invalid = store.formula_evidence(formula_a, min_panelists=1, min_experts=1, as_of=AS_OF)
    assert not invalid.passed
    assert invalid.verification_failures


def test_sensory_scope_and_expired_artifacts_fail_closed(tmp_path):
    private, root = _signer("sensory_laboratory", SENSORY_ARTIFACT_TYPE)
    store = SensoryEvaluationStore(trusted_signers=root)
    formula_a = "sha256:" + "e" * 64
    formula_b = "sha256:" + "f" * 64
    for formula in (formula_a, formula_b):
        store.register_formula(formula, "brief", 90, [])
    mapping = store.create_study("brief", [formula_a, formula_b], "protocol", seed=3)
    study_id = mapping.pop("study_id")
    code_a = next(code for code, formula in mapping.items() if formula == formula_a)
    data = tmp_path / "study.csv"
    data.write_text(
        f"study_id,blind_code,panelist_id,similarity,liking\n{study_id},{code_a},P-1,95,80\n",
        encoding="utf-8",
    )
    envelope = _envelope(
        private, SENSORY_ARTIFACT_TYPE,
        {"study_id": study_id, "formula_ids": [formula_a]},
        {"study_data": data}, "wrong-scope", "sensory_laboratory",
    )
    with pytest.raises(ValueError, match="formula_ids"):
        store.import_verified(study_id, data, envelope, as_of=AS_OF)
    expired = _envelope(
        private, SENSORY_ARTIFACT_TYPE,
        {"study_id": study_id, "formula_ids": sorted([formula_a, formula_b])},
        {"study_data": data}, "expired-study", "sensory_laboratory",
    )
    expired["expires_at"] = "2026-07-02T00:00:00+00:00"
    expired["signature"] = base64.b64encode(private.sign(signing_payload(expired))).decode()
    with pytest.raises(ValueError, match="expired"):
        store.import_verified(study_id, data, expired, as_of=AS_OF)
