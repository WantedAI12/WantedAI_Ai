from __future__ import annotations

import base64
import hashlib
from datetime import date
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.models import RecipeConstraints, RecipeLine
from fragrance_ai.recommender.release import (
    CommercialReleaseStore,
    RegulatorySignoff,
    VerifiedRegulatorySignoff,
    signing_payload,
)
from fragrance_ai.recommender.release_spec import ReleaseSpec, sha256_file
from fragrance_ai.recommender.supplier import SupplierMaterial, SupplierRegistry


AS_OF = date(2026, 7, 28)


def _offer(lot: str = "LOT-A") -> SupplierMaterial:
    return SupplierMaterial(
        ingredient_id="test_material",
        supplier="Verified Supplier",
        sku="SKU-100",
        cas_number=None,
        price_per_kg=20.0,
        currency="USD",
        moq_kg=1.0,
        in_stock=True,
        lead_time_days=7,
        density_g_ml=0.9,
        active_strength_percent=80.0,
        carrier="TEC",
        regions=("EU",),
        ifra_amendment="51",
        ifra_certificate_valid_until="2027-07-28",
        sds_valid_until="2027-07-28",
        coa_available=True,
        allergen_statement_valid_until="2027-07-28",
        allergen_fractions={"Linalool": 0.0},
        lot_number=lot,
    )


def _scope(tmp_path, lot: str = "LOT-A") -> ReleaseSpec:
    document_paths = {}
    for name in ("coa", "sds", "ifra_certificate", "allergen_statement"):
        path = tmp_path / f"{lot}-{name}.txt"
        path.write_text(f"{lot}:{name}", encoding="utf-8")
        document_paths[name] = {"path": str(path)}
    constraints = RecipeConstraints(
        product_concentration_percent=15.0,
        product_category="eau_de_parfum",
        target_region="EU",
        commercial_product_base_id="BASE-ETHANOL-01",
        commercial_packaging_id="GLASS-50ML-01",
        commercial_rule_pack_version="ifra-51.0.0",
        commercial_data_version="catalog-2026-07-28",
        commercial_model_version="physsim-1.0.0",
        commercial_supplier_evidence={
            "test_material": {
                "supplier": "Verified Supplier",
                "sku": "SKU-100",
                "lot_number": lot,
                "documents": document_paths,
            }
        },
    )
    lines = [
        RecipeLine(
            ingredient_id="test_material",
            name="Test material",
            pyramid="heart",
            concentrate_percent=100.0,
            finished_product_percent=15.0,
            volume_ml_for_batch=None,
            price_per_kg=20.0,
            availability=1.0,
            risk_tier=0,
            reason="test",
            active_strength_percent=80.0,
            carrier="TEC",
        )
    ]
    return ReleaseSpec.build(
        lines,
        constraints,
        SupplierRegistry([_offer(lot)]),
        rule_pack_version=constraints.commercial_rule_pack_version,
        data_version=constraints.commercial_data_version,
        model_version=constraints.commercial_model_version,
    )


def _signed(scope: ReleaseSpec, report, private: Ed25519PrivateKey) -> VerifiedRegulatorySignoff:
    unsigned = VerifiedRegulatorySignoff(
        release_spec_id=scope.release_spec_id,
        market_region="EU",
        approver_role="Responsible Person",
        organization="Independent Regulatory Lab",
        signer_id="lab-01",
        approved_on="2026-07-27",
        valid_until="2027-07-27",
        report_ref="SAFETY-REPORT-001",
        report_path=str(report),
        report_sha256=sha256_file(report),
        signature="",
    )
    signature = base64.b64encode(private.sign(signing_payload(unsigned))).decode("ascii")
    return VerifiedRegulatorySignoff(**{**unsigned.__dict__, "signature": signature})


def _store(tmp_path, private: Ed25519PrivateKey) -> CommercialReleaseStore:
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return CommercialReleaseStore(tmp_path / "release.db", {"lab-01": public})


def test_only_real_report_bytes_and_allowlisted_ed25519_signature_can_pass(tmp_path):
    scope = _scope(tmp_path)
    report = tmp_path / "report.pdf"
    report.write_bytes(b"actual signed report bytes")
    private = Ed25519PrivateKey.generate()
    store = _store(tmp_path, private)
    signoff = _signed(scope, report, private)
    store.record_verified(scope, signoff)
    assessment = store.assess_scope(scope, AS_OF)
    assert assessment.passed
    assert assessment.scope_verified

    forged_hash = VerifiedRegulatorySignoff(
        **{**signoff.__dict__, "report_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="report_sha256 does not match report bytes"):
        store.record_verified(scope, forged_hash)

    other_private = Ed25519PrivateKey.generate()
    forged_signature = _signed(scope, report, other_private)
    with pytest.raises(ValueError, match="signature verification failed"):
        store.record_verified(scope, forged_signature)


def test_wrong_scope_changed_lot_and_future_or_revoked_approval_fail_closed(tmp_path):
    scope_a = _scope(tmp_path, "LOT-A")
    report = tmp_path / "report.pdf"
    report.write_bytes(b"report")
    private = Ed25519PrivateKey.generate()
    store = _store(tmp_path, private)
    signoff = _signed(scope_a, report, private)
    store.record_verified(scope_a, signoff)

    scope_b = _scope(tmp_path, "LOT-B")
    assert scope_a.release_spec_id != scope_b.release_spec_id
    changed_lot = store.assess_scope(scope_b, AS_OF)
    assert not changed_lot.passed
    assert changed_lot.status == "external_regulatory_signoff_missing"

    future_unsigned = VerifiedRegulatorySignoff(
        **{**signoff.__dict__, "approved_on": "2026-07-29", "valid_until": "2027-07-29"}
    )
    future = VerifiedRegulatorySignoff(
        **{
            **future_unsigned.__dict__,
            "signature": base64.b64encode(private.sign(signing_payload(future_unsigned))).decode("ascii"),
        }
    )
    store.record_verified(scope_a, future)
    not_effective = store.assess_scope(scope_a, AS_OF)
    assert not not_effective.passed
    assert not_effective.status == "external_regulatory_signoff_invalid"

    # Restore the current approval before testing explicit revocation.
    store.record_verified(scope_a, signoff)

    store.revoke(scope_a.release_spec_id, signoff.report_sha256, AS_OF, "lot recall")
    revoked = store.assess_scope(scope_a, AS_OF)
    assert not revoked.passed
    assert revoked.status == "external_regulatory_signoff_revoked"


def test_report_or_supplier_document_mutation_and_legacy_api_cannot_approve(tmp_path):
    scope = _scope(tmp_path)
    report = tmp_path / "report.pdf"
    report.write_bytes(b"unchanged at signing")
    private = Ed25519PrivateKey.generate()
    store = _store(tmp_path, private)
    signoff = _signed(scope, report, private)
    store.record_verified(scope, signoff)

    report.unlink()
    missing_report = store.assess_scope(scope, AS_OF)
    assert not missing_report.passed
    assert missing_report.status == "external_regulatory_signoff_invalid"

    # The old claimed-hash API is audit-only, even for a syntactically valid
    # formula fingerprint and report digest.
    legacy_formula = "sha256:" + "a" * 64
    store.record(
        RegulatorySignoff(
            legacy_formula, "EU", "Responsible Person", "Lab", "2026-07-01",
            "2027-07-01", "legacy", hashlib.sha256(b"claimed").hexdigest(),
        )
    )
    legacy = store.assess(legacy_formula, "EU", AS_OF)
    assert not legacy.passed
    assert legacy.status == "legacy_formula_fingerprint_not_approvable"

    # Supplier certificate bytes are re-hashed immediately before release.
    document_path = Path(dict(scope.document_paths)["test_material.coa"])
    document_path.write_text("changed COA", encoding="utf-8")
    modified_scope = store.assess_scope(scope, AS_OF)
    assert not modified_scope.passed
    assert modified_scope.status == "release_scope_unverifiable"
