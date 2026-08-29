import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    EvidenceTrustRoot,
    sha256_file,
    signing_payload,
)
from fragrance_ai.recommender.industrial_catalog import (
    INDUSTRIAL_REGISTRY_SCHEMA,
    SAFETY_DOSSIER_ARTIFACT_TYPE,
    IndustrialIngredientRegistry,
)
from fragrance_ai.recommender.models import RecipeConstraints
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REPORT = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.json"
BUILDER = ROOT / "scripts" / "build_industrial_ingredient_registry_v1.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_industrial_registry_counts_and_source_binding():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == INDUSTRIAL_REGISTRY_SCHEMA
    assert report["status"] == "industrial_scale_reference_registry_built"
    assert report["database"]["sha256"] == _sha256(DATABASE)
    assert report["database"]["bytes"] == DATABASE.stat().st_size
    assert report["implementation"]["script_sha256"] == _sha256(BUILDER)
    assert report["counts"]["reference_molecules"] == 29_240
    assert report["counts"]["source_links"] == 36_327
    assert report["counts"]["descriptor_assertions"] == 71_458
    assert report["counts"]["molecules_with_descriptors"] == 28_549
    assert report["counts"]["catalog_materials"] == 47
    assert report["counts"]["prototype_safe_active"] == 29
    assert report["counts"]["prototype_conditional_active"] == 5
    assert report["counts"]["prototype_active_total"] == 34
    assert report["counts"]["safety_screened"] == 29_240
    assert report["counts"]["promotion_candidates_total"] == 29_212
    assert report["counts"]["promotion_evidence_pending"] == 20_757
    assert report["counts"]["promotion_structural_review_required"] == 8_455
    assert report["counts"]["high_priority_candidates"] == 856
    assert report["counts"]["invalid_cas_identifiers_rejected"] == 1
    assert report["tier_contract"]["reference_molecules_are_formula_eligible"] is False
    assert report["tier_contract"]["all_reference_molecules_have_safety_screening"]
    assert report["tier_contract"]["all_unlinked_molecules_have_promotion_path"]
    assert "formulation_spec" in report["tier_contract"]["required_promotion_evidence"]
    assert report["tier_contract"]["runtime_auto_activation"]["enabled"] is True
    assert report["tier_contract"]["qualified_or_commercial_materials"] == 0


def test_industrial_registry_search_preserves_formulation_tiers():
    with IndustrialIngredientRegistry(DATABASE) as registry:
        assert registry.stats() == {
            "reference_molecules": 29_240,
            "source_links": 36_327,
            "descriptor_assertions": 71_458,
            "formulation_materials": 47,
            "active_safe": 29,
            "active_conditional": 5,
            "prototype_active_total": 34,
            "molecularly_linked_materials": 28,
            "safety_screened": 29_240,
            "screening_evidence_pending": 20_785,
            "structural_review_required": 8_455,
            "molecules_with_cas": 3_787,
            "ifra_reference_molecules": 1_060,
            "promotion_candidates_total": 29_212,
            "promotion_evidence_pending": 20_757,
            "promotion_structural_review_required": 8_455,
            "high_priority_candidates": 856,
        }
        hedione = registry.search("hedione", limit=5)
        assert hedione[0].formulation_material_id == "hedione"
        assert hedione[0].formulation_tier == "prototype_safe_active"
        woody = registry.search("woody", limit=50, formulation_only=True)
        assert woody
        assert all(item.formulation_material_id is not None for item in woody)
        assert all(item.formulation_tier != "reference_only" for item in woody)
        broad = registry.search("woody", limit=500)
        assert len(broad) > len(woody)
        assert any(item.formulation_tier == "reference_only" for item in broad)
        cas_hit = registry.search("93-15-2", limit=5)
        assert cas_hit[0].preferred_name == "methyl eugenol"
        assert cas_hit[0].cas_number == "93-15-2"
        assert cas_hit[0].formulation_tier == "reference_only"
        patchouli = registry.search("patchouli", formulation_only=True)
        assert patchouli[0].formulation_tier == "prototype_conditional_active"
        assert patchouli[0].formulation_material_id == "patchouli_oil"
        candidates = registry.promotion_candidates(limit=20)
        assert len(candidates) == 20
        assert all(
            item["promotion_status"] == "evidence_pending" for item in candidates
        )
        structural = registry.promotion_candidates(
            limit=20, status="structural_review_required"
        )
        assert len(structural) == 20
        assert all(
            item["promotion_status"] == "structural_review_required"
            for item in structural
        )


def test_industrial_registry_rejects_invalid_search_controls():
    with IndustrialIngredientRegistry(DATABASE) as registry:
        with pytest.raises(ValueError, match="query is required"):
            registry.search("---")
        with pytest.raises(ValueError, match="between 1 and 500"):
            registry.search("woody", limit=0)
        with pytest.raises(ValueError, match="between 1 and 1000"):
            registry.promotion_candidates(limit=0)
        with pytest.raises(ValueError, match="unsupported promotion candidate status"):
            registry.promotion_candidates(status="safe")


def test_every_reference_has_screening_and_fail_closed_promotion_path():
    with IndustrialIngredientRegistry(DATABASE) as registry:
        evidence_candidate = registry.promotion_candidates(
            limit=1, status="evidence_pending"
        )[0]
        screening = registry.safety_screening(evidence_candidate["registry_id"])
        assert screening is not None
        assert screening.structural_alerts == ()

        incomplete = registry.evaluate_safety_promotion(
            screening.registry_id,
            (),
            target_tier="prototype_safe_active",
        )
        assert incomplete.eligible_tier is None
        assert "low_risk_signoff" in incomplete.missing_evidence
        assert incomplete.decision_reason == "missing_required_evidence"

        labels = {*screening.required_evidence, "low_risk_signoff"}
        complete_but_unsigned = registry.evaluate_safety_promotion(
            screening.registry_id,
            labels,
            target_tier="prototype_safe_active",
        )
        assert complete_but_unsigned.dossier_complete is True
        assert complete_but_unsigned.independent_signature_verified is False
        assert complete_but_unsigned.eligible_tier is None
        assert complete_but_unsigned.decision_reason == "independent_signature_required"

        structural_candidate = registry.promotion_candidates(
            limit=1, status="structural_review_required"
        )[0]
        structural_screening = registry.safety_screening(
            structural_candidate["registry_id"]
        )
        assert structural_screening is not None
        structural_decision = registry.evaluate_safety_promotion(
            structural_screening.registry_id,
            structural_screening.required_evidence,
        )
        assert "structural_review_signoff" in structural_decision.missing_evidence
        assert (
            structural_decision.blocking_alerts
            == structural_screening.structural_alerts
        )

        lyral = registry.search("lyral", formulation_only=True)[0]
        with pytest.raises(ValueError, match="policy-blocked"):
            registry.evaluate_safety_promotion(lyral.registry_id, labels)


def test_signed_scoped_dossier_can_authorize_safe_tier(tmp_path: Path):
    moment = datetime(2026, 8, 29, tzinfo=timezone.utc)
    with IndustrialIngredientRegistry(DATABASE) as registry:
        candidate = registry.promotion_candidates(limit=1, status="evidence_pending")[0]
        ingredient = registry.get(candidate["registry_id"])
        screening = registry.safety_screening(candidate["registry_id"])
        assert ingredient is not None and screening is not None
        required = {*screening.required_evidence, "low_risk_signoff"}
        artifact_paths = {}
        for label in sorted(required):
            path = tmp_path / f"{label}.json"
            path.write_text(json.dumps({"label": label}), encoding="utf-8")
            artifact_paths[label] = path

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        trust_root = EvidenceTrustRoot(
            {
                "signers": {
                    "independent-safety-lab": {
                        "public_key": public_key.hex(),
                        "roles": ["toxicologist"],
                        "artifact_types": [SAFETY_DOSSIER_ARTIFACT_TYPE],
                    }
                }
            }
        )
        scope = {
            "registry_schema": INDUSTRIAL_REGISTRY_SCHEMA,
            "registry_sha256": registry.sha256,
            "registry_id": ingredient.registry_id,
            "canonical_smiles": ingredient.canonical_smiles,
            "target_tier": "prototype_safe_active",
            "market": "KR",
            "product_category": "fine_fragrance",
        }
        envelope = {
            "schema": ARTIFACT_SIGNATURE_SCHEMA,
            "artifact_id": "test-safety-dossier-1",
            "artifact_type": SAFETY_DOSSIER_ARTIFACT_TYPE,
            "signer_id": "independent-safety-lab",
            "signer_role": "toxicologist",
            "scope": scope,
            "issued_at": "2026-08-01T00:00:00+00:00",
            "expires_at": "2027-08-01T00:00:00+00:00",
            "artifact_hashes": {
                label: sha256_file(path) for label, path in artifact_paths.items()
            },
        }
        envelope["signature"] = base64.b64encode(
            private_key.sign(signing_payload(envelope))
        ).decode("ascii")

        decision = registry.verify_safety_promotion(
            ingredient.registry_id,
            target_tier="prototype_safe_active",
            market="KR",
            product_category="fine_fragrance",
            envelope=envelope,
            artifact_paths=artifact_paths,
            trust_root=trust_root,
            as_of=moment,
        )
        assert decision.dossier_complete is True
        assert decision.independent_signature_verified is True
        assert decision.eligible_tier == "prototype_safe_active"
        assert decision.decision_reason == "signed_dossier_verified"

        wrong_scope = dict(scope)
        wrong_scope["registry_sha256"] = "0" * 64
        replay = {**envelope, "scope": wrong_scope}
        replay["signature"] = base64.b64encode(
            private_key.sign(signing_payload(replay))
        ).decode("ascii")
        with pytest.raises(ValueError, match="registry_sha256"):
            registry.verify_safety_promotion(
                ingredient.registry_id,
                target_tier="prototype_safe_active",
                market="KR",
                product_category="fine_fragrance",
                envelope=replay,
                artifact_paths=artifact_paths,
                trust_root=trust_root,
                as_of=moment,
            )


def test_explicit_risk_two_request_uses_conditional_materials_with_caps():
    constraints = RecipeConstraints(
        max_risk_tier=2,
        min_availability=0.75,
        target_similarity=70,
        max_ingredients=20,
        simulation_draws=64,
        physics_search_population=2,
        minimum_realism_score=50,
    )
    with NaturalLanguagePerfumeryAI() as ai:
        result = ai.create_recipe(
            "lavender patchouli sweet orange cedarwood aromatic woody fragrance",
            constraints,
        )
    assert result.status == "prototype_ready"
    conditional = {
        line.ingredient_id: line for line in result.recipe if line.risk_tier == 2
    }
    assert {
        "methyl_ionone_gamma",
        "sweet_orange_oil",
        "lavender_oil",
        "cedarwood_virginia",
    }.issubset(conditional)
    assert all(line.concentrate_percent <= 5.0 for line in conditional.values())

    default_constraints = RecipeConstraints(
        max_risk_tier=1,
        min_availability=0.75,
        target_similarity=70,
        max_ingredients=20,
        simulation_draws=64,
        physics_search_population=2,
        minimum_realism_score=50,
    )
    with NaturalLanguagePerfumeryAI() as ai:
        default_result = ai.create_recipe(
            "lavender patchouli sweet orange cedarwood aromatic woody fragrance",
            default_constraints,
        )
    assert all(line.risk_tier <= 1 for line in default_result.recipe)
