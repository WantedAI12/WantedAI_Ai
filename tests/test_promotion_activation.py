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
    signing_payload,
)
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.industrial_catalog import (
    INDUSTRIAL_REGISTRY_SCHEMA,
    SAFETY_DOSSIER_ARTIFACT_TYPE,
    IndustrialIngredientRegistry,
)
from fragrance_ai.recommender.models import RecipeConstraints
from fragrance_ai.recommender.promotion_activation import (
    FORMULATION_SPEC_SCHEMA,
    PROMOTION_DIRECTORY_ENV,
    PROMOTION_REGISTRY_ENV,
    PROMOTION_TRUST_ROOT_ENV,
    PromotionActivationBundle,
)
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
AS_OF = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_promotion_package(
    tmp_path: Path,
    *,
    target_tier: str = "prototype_safe_active",
    market: str = "EU",
    product_category: str = "eau_de_parfum",
    aliases: list[str] | None = None,
) -> tuple[Path, EvidenceTrustRoot, Path, dict]:
    promotion_root = tmp_path / "promotions"
    package = promotion_root / "verified-smoky-material"
    package.mkdir(parents=True)
    with IndustrialIngredientRegistry(DATABASE) as registry:
        reference = registry.search("93-15-2", limit=1)[0]
        screening = registry.safety_screening(reference.registry_id)
        assert screening is not None
        required = set(screening.required_evidence)
        required.add("formulation_spec")
        if target_tier == "prototype_safe_active":
            required.add("low_risk_signoff")
        if screening.structural_alerts:
            required.add("structural_review_signoff")
        scope = {
            "registry_schema": INDUSTRIAL_REGISTRY_SCHEMA,
            "registry_sha256": registry.sha256,
            "registry_id": reference.registry_id,
            "canonical_smiles": reference.canonical_smiles,
            "target_tier": target_tier,
            "market": market,
            "product_category": product_category,
        }
        ingredient_id = "industrial_" + reference.registry_id.removeprefix("mol:")
        formulation_spec = {
            "schema": FORMULATION_SPEC_SCHEMA,
            "registry_id": reference.registry_id,
            "canonical_smiles": reference.canonical_smiles,
            "target_tier": target_tier,
            "market": market,
            "product_category": product_category,
            "ingredient_id": ingredient_id,
            "name": "Verified Smoky Material",
            "aliases": aliases or ["verified smoky material", "검증 스모키 원료"],
            "pyramid": "base",
            "profile": {"smoky": 1.0, "woody": 0.55, "spicy": 0.25},
            "rarity": "common",
            "odor_impact": 2.5,
            "max_concentrate_percent": 5.0,
            "oxidation_risk": "low",
            "discoloration_risk": "low",
            "shelf_life_months": 24,
            "eu_allergens": ["Test Allergen"],
            "solubility": ["ethanol"],
            "supplier_material": {
                "supplier": "Independent Test Supplier",
                "sku": "VERIFIED-SMOKY-001",
                "cas_number": reference.cas_number,
                "price_per_kg": 42.0,
                "currency": "USD",
                "moq_kg": 1.0,
                "in_stock": True,
                "lead_time_days": 7,
                "density_g_ml": 0.98,
                "active_strength_percent": 100.0,
                "carrier": None,
                "regions": [market],
                "ifra_amendment": "51",
                "ifra_certificate_valid_until": "2027-08-01",
                "sds_valid_until": "2027-08-01",
                "coa_available": True,
                "allergen_statement_valid_until": "2027-08-01",
                "allergen_fractions": {"Test Allergen": 0.001},
                "lot_number": "TEST-LOT-001",
            },
        }
        artifact_files = {}
        artifact_hashes = {}
        for label in sorted(required):
            path = package / f"{label}.json"
            payload = (
                formulation_spec
                if label == "formulation_spec"
                else {"schema": "test-evidence/v1", "label": label}
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            artifact_files[label] = path.name
            artifact_hashes[label] = _sha256(path)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    policy = {
        "signers": {
            "independent-safety-lab": {
                "public_key": public_key.hex(),
                "roles": ["toxicologist"],
                "artifact_types": [SAFETY_DOSSIER_ARTIFACT_TYPE],
            }
        }
    }
    trust_root = EvidenceTrustRoot(policy)
    envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": "verified-smoky-material-v1",
        "artifact_type": SAFETY_DOSSIER_ARTIFACT_TYPE,
        "signer_id": "independent-safety-lab",
        "signer_role": "toxicologist",
        "scope": scope,
        "issued_at": "2026-08-01T00:00:00+00:00",
        "expires_at": "2027-08-01T00:00:00+00:00",
        "artifact_files": artifact_files,
        "artifact_hashes": artifact_hashes,
    }
    envelope["signature"] = base64.b64encode(
        private_key.sign(signing_payload(envelope))
    ).decode("ascii")
    envelope_path = package / "envelope.json"
    envelope_path.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return promotion_root, trust_root, envelope_path, policy


def _recipe_constraints(**overrides) -> RecipeConstraints:
    values = {
        "target_region": "EU",
        "product_category": "eau_de_parfum",
        "max_risk_tier": 1,
        "target_similarity": 70,
        "max_ingredients": 20,
        "simulation_draws": 64,
        "physics_search_population": 2,
        "minimum_realism_score": 50,
    }
    values.update(overrides)
    return RecipeConstraints(**values)


def test_signed_safety_pass_is_automatically_used_by_recipe_service(
    tmp_path: Path,
):
    promotion_root, trust_root, _, _ = _signed_promotion_package(tmp_path)
    bundle = PromotionActivationBundle.load(
        registry_path=DATABASE,
        promotion_directory=promotion_root,
        trust_root=trust_root,
        as_of=AS_OF,
    )
    assert len(bundle.promotions) == 1
    promoted = bundle.promotions[0].ingredient
    assert promoted.formulation_ready is True
    assert promoted.risk_tier == 1
    assert promoted.approved_formulation_scopes == ("EU|eau_de_parfum",)
    assert promoted.carrier is None

    with NaturalLanguagePerfumeryAI(promotion_bundle=bundle) as ai:
        assert ai.catalog.stats()["formulation_ready"] == 35
        assert ai.catalog.stats()["signed_promotions_active"] == 1
        result = ai.create_recipe(
            "verified smoky material with smoky dry woods and a fresh opening",
            _recipe_constraints(),
            as_of=AS_OF.date(),
        )
        expired_result = ai.create_recipe(
            "verified smoky material with smoky dry woods and a fresh opening",
            _recipe_constraints(),
            as_of=datetime(2027, 8, 1, tzinfo=timezone.utc).date(),
        )
    lines = {line.ingredient_id: line for line in result.recipe}
    assert promoted.ingredient_id in lines
    assert lines[promoted.ingredient_id].concentrate_percent <= 5.0
    assert lines[promoted.ingredient_id].data_source == (
        "signed-industrial-promotion:verified-smoky-material-v1"
    )
    assert lines[promoted.ingredient_id].approved_formulation_scopes == (
        "EU|eau_de_parfum",
    )
    assert lines[promoted.ingredient_id].promotion_artifact_id == (
        "verified-smoky-material-v1"
    )
    assert result.safety.internal_gate_passed is True
    assert all(
        line.ingredient_id != promoted.ingredient_id for line in expired_result.recipe
    )
    assert expired_result.rejected_candidate_counts["signed_promotion_expired"] == 1


def test_default_service_loads_configured_promotions_and_enforces_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    promotion_root, _, _, policy = _signed_promotion_package(tmp_path)
    trust_path = tmp_path / "trust-root.json"
    trust_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setenv(PROMOTION_REGISTRY_ENV, str(DATABASE))
    monkeypatch.setenv(PROMOTION_DIRECTORY_ENV, str(promotion_root))
    monkeypatch.setenv(PROMOTION_TRUST_ROOT_ENV, str(trust_path))

    with NaturalLanguagePerfumeryAI() as ai:
        promoted = next(
            item for item in ai.catalog.ingredients if item.approved_formulation_scopes
        )
        assert ai.catalog.stats()["formulation_ready"] == 35
        allowed = ai.create_recipe(
            "verified smoky material smoky woods with a fresh opening",
            _recipe_constraints(),
            as_of=AS_OF.date(),
        )
        blocked_scope = ai.create_recipe(
            "verified smoky material smoky woods with a fresh opening",
            _recipe_constraints(target_region="KR"),
            as_of=AS_OF.date(),
        )
    assert any(line.ingredient_id == promoted.ingredient_id for line in allowed.recipe)
    assert all(
        line.ingredient_id != promoted.ingredient_id for line in blocked_scope.recipe
    )
    assert blocked_scope.rejected_candidate_counts["signed_promotion_scope"] == 1


def test_tampered_formulation_spec_and_partial_environment_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    promotion_root, trust_root, envelope_path, _ = _signed_promotion_package(tmp_path)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    spec_path = envelope_path.parent / envelope["artifact_files"]["formulation_spec"]
    spec_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence artifact bytes changed"):
        PromotionActivationBundle.load(
            registry_path=DATABASE,
            promotion_directory=promotion_root,
            trust_root=trust_root,
            as_of=AS_OF,
        )

    monkeypatch.setenv(PROMOTION_REGISTRY_ENV, str(DATABASE))
    monkeypatch.delenv(PROMOTION_DIRECTORY_ENV, raising=False)
    monkeypatch.delenv(PROMOTION_TRUST_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="configuration is incomplete"):
        NaturalLanguagePerfumeryAI()


def test_promoted_alias_cannot_hijack_an_existing_catalog_material(tmp_path: Path):
    promotion_root, trust_root, _, _ = _signed_promotion_package(
        tmp_path,
        aliases=["hedione"],
    )
    bundle = PromotionActivationBundle.load(
        registry_path=DATABASE,
        promotion_directory=promotion_root,
        trust_root=trust_root,
        as_of=AS_OF,
    )
    with pytest.raises(ValueError, match="alias collision"):
        bundle.merge_catalog(IngredientCatalog.load_builtin())


def test_signed_conditional_promotion_requires_explicit_risk_two(tmp_path: Path):
    promotion_root, trust_root, _, _ = _signed_promotion_package(
        tmp_path,
        target_tier="prototype_conditional_active",
    )
    bundle = PromotionActivationBundle.load(
        registry_path=DATABASE,
        promotion_directory=promotion_root,
        trust_root=trust_root,
        as_of=AS_OF,
    )
    promoted = bundle.promotions[0].ingredient
    assert promoted.risk_tier == 2
    with NaturalLanguagePerfumeryAI(promotion_bundle=bundle) as ai:
        default_result = ai.create_recipe(
            "verified smoky material smoky woods with a fresh opening",
            _recipe_constraints(),
            as_of=AS_OF.date(),
        )
        explicit_result = ai.create_recipe(
            "verified smoky material smoky woods with a fresh opening",
            _recipe_constraints(max_risk_tier=2),
            as_of=AS_OF.date(),
        )
    assert all(
        line.ingredient_id != promoted.ingredient_id for line in default_result.recipe
    )
    assert default_result.rejected_candidate_counts["risk_tier"] >= 1
    assert any(
        line.ingredient_id == promoted.ingredient_id for line in explicit_result.recipe
    )
