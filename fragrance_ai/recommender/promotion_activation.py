"""Automatic, fail-closed activation of independently approved ingredients.

Each promotion package contains a signed envelope, real evidence files, and a
machine-readable formulation specification.  A reference molecule enters the
runtime formula pool only after the industrial registry verifies the exact
registry bytes, molecule, market, category, target tier, evidence hashes, and
an allowlisted Ed25519 signature.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_trust import EvidenceTrustRoot
from .catalog import IngredientCatalog
from .industrial_catalog import IndustrialIngredientRegistry
from .models import Ingredient, PYRAMID_LEVELS, RecipeConstraints, SCENT_DIMENSIONS
from .supplier import SupplierMaterial, SupplierRegistry


FORMULATION_SPEC_SCHEMA = "perfumery-promoted-ingredient-formulation/v1"
PROMOTION_REGISTRY_ENV = "PERFUMERY_AI_INDUSTRIAL_REGISTRY_DB"
PROMOTION_DIRECTORY_ENV = "PERFUMERY_AI_PROMOTION_DIRECTORY"
PROMOTION_TRUST_ROOT_ENV = "PERFUMERY_AI_PROMOTION_TRUST_ROOT"
MAX_PROMOTION_PACKAGES = 50_000
MAX_JSON_BYTES = 1_000_000
SUPPORTED_PRODUCT_CATEGORIES = frozenset(
    {
        "eau_de_parfum",
        "eau_de_toilette",
        "eau_de_cologne",
        "face_cream",
        "face_toner",
        "mouthwash",
        "shampoo",
        "body_wash",
        "candle",
        "room_spray",
        "diffuser",
    }
)


def formulation_scope_key(market: str, product_category: str) -> str:
    normalized_market = str(market).strip().upper()
    normalized_category = str(product_category).strip().casefold()
    if not normalized_market or not normalized_category:
        raise ValueError("promotion market and product category are required")
    if "|" in normalized_market or "|" in normalized_category:
        raise ValueError("promotion scope values cannot contain '|'")
    return f"{normalized_market}|{normalized_category}"


def formulation_scope_allows(
    approved_scopes: tuple[str, ...],
    market: str,
    product_category: str,
) -> bool:
    requested_market, requested_category = formulation_scope_key(
        market, product_category
    ).split("|", 1)
    for scope in approved_scopes:
        approved_market, approved_category = scope.split("|", 1)
        if approved_category == requested_category and approved_market in {
            requested_market,
            "GLOBAL",
        }:
            return True
    return False


def _required_text(value: Any, name: str, *, maximum: int = 300) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    result = " ".join(value.strip().split())
    if not result or len(result) > maximum:
        raise ValueError(f"{name} must contain 1 to {maximum} characters")
    return result


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _valid_cas_number(value: str) -> bool:
    match = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", value)
    if match is None:
        return False
    body = match.group(1) + match.group(2)
    checksum = (
        sum(
            multiplier * int(digit)
            for multiplier, digit in enumerate(reversed(body), start=1)
        )
        % 10
    )
    return checksum == int(match.group(3))


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{name} exceeds the {MAX_JSON_BYTES}-byte limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _artifact_paths(
    package_directory: Path,
    envelope: Mapping[str, Any],
) -> dict[str, Path]:
    files = envelope.get("artifact_files")
    hashes = envelope.get("artifact_hashes")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError(
            "promotion envelope requires artifact_files and artifact_hashes"
        )
    if set(files) != set(hashes):
        raise ValueError("promotion artifact file labels do not match signed hashes")
    resolved: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for raw_label, raw_relative in files.items():
        label = _required_text(raw_label, "artifact label", maximum=100)
        if not isinstance(raw_relative, str) or not raw_relative.strip():
            raise ValueError(f"artifact_files.{label} must be a relative path")
        relative = Path(raw_relative)
        if relative.is_absolute():
            raise ValueError(f"artifact_files.{label} must be relative")
        unresolved_path = package_directory / relative
        if unresolved_path.is_symlink():
            raise ValueError(f"artifact_files.{label} cannot be a symlink")
        path = unresolved_path.resolve(strict=True)
        try:
            path.relative_to(package_directory)
        except ValueError as error:
            raise ValueError(f"artifact_files.{label} escapes its package") from error
        if not path.is_file() or path in seen_paths:
            raise ValueError(f"artifact_files.{label} must identify a unique file")
        if label in resolved:
            raise ValueError(
                "promotion artifact labels must be unique after normalization"
            )
        resolved[label] = path
        seen_paths.add(path)
    return resolved


def _text_tuple(
    value: Any,
    name: str,
    *,
    maximum_items: int = 64,
    maximum_length: int = 200,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{name} must be a list with at most {maximum_items} items")
    return tuple(
        _required_text(item, f"{name} item", maximum=maximum_length) for item in value
    )


def _formulation_profile(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("formulation_spec.profile must be a nonempty object")
    unknown = sorted(set(value) - set(SCENT_DIMENSIONS))
    if unknown:
        raise ValueError(
            "formulation profile has unknown dimensions: " + ", ".join(unknown)
        )
    profile = {
        str(name): _finite_float(
            amount,
            f"formulation_spec.profile.{name}",
            minimum=0.0,
            maximum=1.0,
        )
        for name, amount in value.items()
    }
    if sum(profile.values()) <= 0:
        raise ValueError("formulation profile must contain positive scent mass")
    return profile


@dataclass(frozen=True)
class ActivatedIngredientPromotion:
    registry_id: str
    target_tier: str
    market: str
    product_category: str
    artifact_id: str
    signer_id: str
    ingredient: Ingredient
    supplier_material: SupplierMaterial


@dataclass(frozen=True)
class PromotionActivationBundle:
    promotions: tuple[ActivatedIngredientPromotion, ...] = ()
    registry_sha256: str | None = None

    @property
    def ingredients(self) -> tuple[Ingredient, ...]:
        return tuple(item.ingredient for item in self.promotions)

    @property
    def supplier_materials(self) -> tuple[SupplierMaterial, ...]:
        return tuple(item.supplier_material for item in self.promotions)

    def merge_catalog(self, catalog: IngredientCatalog) -> IngredientCatalog:
        return catalog.with_promoted_ingredients(
            self.ingredients,
            metadata={
                "signed_promotions_active": len(self.promotions),
                "promotion_registry_sha256": self.registry_sha256 or "",
            },
        )

    def merge_supplier_registry(self, registry: SupplierRegistry) -> SupplierRegistry:
        if not self.supplier_materials:
            return registry
        existing = {(item.supplier, item.sku) for item in registry.records}
        for item in self.supplier_materials:
            key = (item.supplier, item.sku)
            if key in existing:
                raise ValueError(
                    f"signed promotion duplicates supplier SKU: {item.supplier}/{item.sku}"
                )
            existing.add(key)
        metadata = dict(registry.metadata)
        metadata["signed_promotion_records"] = len(self.supplier_materials)
        return SupplierRegistry(
            [*registry.records, *self.supplier_materials],
            metadata,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        as_of: date | datetime | None = None,
    ) -> "PromotionActivationBundle":
        values = {
            PROMOTION_REGISTRY_ENV: os.environ.get(PROMOTION_REGISTRY_ENV, "").strip(),
            PROMOTION_DIRECTORY_ENV: os.environ.get(
                PROMOTION_DIRECTORY_ENV, ""
            ).strip(),
            PROMOTION_TRUST_ROOT_ENV: os.environ.get(
                PROMOTION_TRUST_ROOT_ENV, ""
            ).strip(),
        }
        configured = [name for name, value in values.items() if value]
        if not configured:
            return cls()
        if len(configured) != len(values):
            missing = sorted(name for name, value in values.items() if not value)
            raise RuntimeError(
                "signed ingredient promotion configuration is incomplete: "
                + ", ".join(missing)
            )
        trust_root = EvidenceTrustRoot.from_json_file(values[PROMOTION_TRUST_ROOT_ENV])
        return cls.load(
            registry_path=values[PROMOTION_REGISTRY_ENV],
            promotion_directory=values[PROMOTION_DIRECTORY_ENV],
            trust_root=trust_root,
            as_of=as_of,
        )

    @classmethod
    def load(
        cls,
        *,
        registry_path: str | Path,
        promotion_directory: str | Path,
        trust_root: EvidenceTrustRoot,
        as_of: date | datetime | None = None,
    ) -> "PromotionActivationBundle":
        root = Path(promotion_directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("promotion directory must be a directory")
        package_directories = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if (root / "envelope.json").is_file():
            package_directories.insert(0, root)
        if len(package_directories) > MAX_PROMOTION_PACKAGES:
            raise ValueError("promotion directory contains too many packages")
        promotions: list[ActivatedIngredientPromotion] = []
        registry_ids: set[str] = set()
        with IndustrialIngredientRegistry(registry_path) as registry:
            for unresolved_package_directory in package_directories:
                if unresolved_package_directory.is_symlink():
                    raise ValueError("promotion package directories cannot be symlinks")
                package_directory = unresolved_package_directory.resolve(strict=True)
                try:
                    package_directory.relative_to(root)
                except ValueError as error:
                    raise ValueError("promotion package escapes its root") from error
                envelope_path = package_directory / "envelope.json"
                if not envelope_path.is_file() or envelope_path.is_symlink():
                    raise ValueError(
                        f"promotion package is missing envelope.json: {package_directory.name}"
                    )
                envelope = _read_json_object(envelope_path, "promotion envelope")
                scope = envelope.get("scope")
                if not isinstance(scope, Mapping):
                    raise ValueError("promotion envelope scope must be an object")
                registry_id = _required_text(
                    scope.get("registry_id"), "scope.registry_id", maximum=100
                )
                target_tier = _required_text(
                    scope.get("target_tier"), "scope.target_tier", maximum=100
                )
                market = _required_text(
                    scope.get("market"), "scope.market", maximum=40
                ).upper()
                product_category = _required_text(
                    scope.get("product_category"),
                    "scope.product_category",
                    maximum=80,
                ).casefold()
                if product_category not in SUPPORTED_PRODUCT_CATEGORIES:
                    raise ValueError("promotion product category is unsupported")
                if registry_id in registry_ids:
                    raise ValueError(
                        f"duplicate signed promotion for registry molecule: {registry_id}"
                    )
                artifact_paths = _artifact_paths(package_directory, envelope)
                if "formulation_spec" not in artifact_paths:
                    raise ValueError(
                        "signed promotion requires formulation_spec evidence"
                    )
                decision = registry.verify_safety_promotion(
                    registry_id,
                    target_tier=target_tier,
                    market=market,
                    product_category=product_category,
                    envelope=envelope,
                    artifact_paths=artifact_paths,
                    trust_root=trust_root,
                    as_of=as_of,
                )
                if (
                    decision.eligible_tier != target_tier
                    or not decision.artifact_id
                    or not decision.signer_id
                    or not decision.issued_at
                    or not decision.expires_at
                ):
                    raise ValueError(
                        "promotion dossier did not authorize a formulation tier"
                    )
                reference = registry.get(registry_id)
                if reference is None or reference.formulation_tier != "reference_only":
                    raise ValueError(
                        "automatic promotion is only supported for reference-only molecules"
                    )
                spec = _read_json_object(
                    artifact_paths["formulation_spec"], "formulation_spec"
                )
                promotion = _activation_from_spec(
                    spec=spec,
                    reference=reference,
                    registry_sha256=registry.sha256,
                    registry_id=registry_id,
                    target_tier=target_tier,
                    market=market,
                    product_category=product_category,
                    artifact_id=decision.artifact_id,
                    signer_id=decision.signer_id,
                    issued_at=decision.issued_at,
                    expires_at=decision.expires_at,
                    source_file=str(artifact_paths["formulation_spec"]),
                    as_of=as_of,
                )
                promotions.append(promotion)
                registry_ids.add(registry_id)
            return cls(tuple(promotions), registry.sha256)


def _activation_from_spec(
    *,
    spec: Mapping[str, Any],
    reference: Any,
    registry_sha256: str,
    registry_id: str,
    target_tier: str,
    market: str,
    product_category: str,
    artifact_id: str,
    signer_id: str,
    issued_at: str,
    expires_at: str,
    source_file: str,
    as_of: date | datetime | None,
) -> ActivatedIngredientPromotion:
    if spec.get("schema") != FORMULATION_SPEC_SCHEMA:
        raise ValueError("unsupported promoted ingredient formulation spec")
    expected = {
        "registry_id": registry_id,
        "canonical_smiles": reference.canonical_smiles,
        "target_tier": target_tier,
        "market": market,
        "product_category": product_category,
    }
    for name, value in expected.items():
        if spec.get(name) != value:
            raise ValueError(f"formulation_spec scope mismatch for {name}")
    ingredient_id = "industrial_" + registry_id.removeprefix("mol:")
    if spec.get("ingredient_id") not in {None, ingredient_id}:
        raise ValueError("formulation_spec ingredient_id is not registry-derived")
    name = _required_text(spec.get("name"), "formulation_spec.name")
    aliases = _text_tuple(spec.get("aliases", []), "formulation_spec.aliases")
    pyramid = _required_text(
        spec.get("pyramid"), "formulation_spec.pyramid", maximum=20
    ).casefold()
    if pyramid not in PYRAMID_LEVELS:
        raise ValueError("formulation_spec.pyramid is unsupported")
    profile = _formulation_profile(spec.get("profile"))
    rarity = _required_text(
        spec.get("rarity"), "formulation_spec.rarity", maximum=20
    ).casefold()
    if rarity not in {"common", "standard"}:
        raise ValueError("signed promotions cannot activate rare materials")
    max_concentrate_percent = _finite_float(
        spec.get("max_concentrate_percent"),
        "formulation_spec.max_concentrate_percent",
        minimum=0.000001,
        maximum=100.0,
    )
    odor_impact = _finite_float(
        spec.get("odor_impact"),
        "formulation_spec.odor_impact",
        minimum=0.000001,
        maximum=100.0,
    )
    oxidation_risk = _required_text(
        spec.get("oxidation_risk"), "formulation_spec.oxidation_risk", maximum=40
    ).casefold()
    discoloration_risk = _required_text(
        spec.get("discoloration_risk"),
        "formulation_spec.discoloration_risk",
        maximum=40,
    ).casefold()
    if oxidation_risk in {"unknown", "unverified"} or discoloration_risk in {
        "unknown",
        "unverified",
    }:
        raise ValueError("signed promotion requires verified stability risks")
    shelf_life_months = spec.get("shelf_life_months")
    if isinstance(shelf_life_months, bool) or not isinstance(shelf_life_months, int):
        raise ValueError("formulation_spec.shelf_life_months must be an integer")
    if not 1 <= shelf_life_months <= 120:
        raise ValueError("formulation_spec.shelf_life_months must be between 1 and 120")
    eu_allergens = _text_tuple(spec.get("eu_allergens", []), "eu_allergens")
    solubility = _text_tuple(spec.get("solubility", []), "solubility")
    supplier_payload = spec.get("supplier_material")
    if not isinstance(supplier_payload, Mapping):
        raise ValueError("formulation_spec.supplier_material must be an object")
    supplier_mapping = dict(supplier_payload)
    for field, minimum, maximum in (
        ("price_per_kg", 0.000001, 300.0),
        ("moq_kg", 0.0, 5.0),
        ("density_g_ml", 0.000001, 10.0),
        ("active_strength_percent", 0.000001, 100.0),
    ):
        supplier_mapping[field] = _finite_float(
            supplier_mapping.get(field),
            f"supplier_material.{field}",
            minimum=minimum,
            maximum=maximum,
        )
    lead_time_days = supplier_mapping.get("lead_time_days")
    if isinstance(lead_time_days, bool) or not isinstance(lead_time_days, int):
        raise ValueError("supplier_material.lead_time_days must be an integer")
    if not 0 <= lead_time_days <= 30:
        raise ValueError("supplier_material.lead_time_days must be between 0 and 30")
    for field in ("in_stock", "coa_available"):
        if not isinstance(supplier_mapping.get(field), bool):
            raise ValueError(f"supplier_material.{field} must be boolean")
    raw_regions = supplier_mapping.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("supplier_material.regions must be a nonempty list")
    supplier_mapping["regions"] = [
        _required_text(item, "supplier_material region", maximum=40).upper()
        for item in raw_regions
    ]
    if not isinstance(supplier_mapping.get("allergen_fractions"), Mapping):
        raise ValueError("supplier_material.allergen_fractions must be an object")
    if supplier_mapping.get("carrier") is None:
        supplier_mapping["carrier"] = ""
    spec_cas = _required_text(
        supplier_mapping.get("cas_number"),
        "supplier_material.cas_number",
        maximum=30,
    )
    if not _valid_cas_number(spec_cas):
        raise ValueError("supplier material CAS number failed checksum validation")
    if reference.cas_number and reference.cas_number != spec_cas:
        raise ValueError("supplier material CAS does not match the industrial registry")
    supplier_mapping.update(
        {
            "ingredient_id": ingredient_id,
            "cas_number": spec_cas,
            "source_file": source_file,
        }
    )
    supplier = SupplierMaterial.from_mapping(supplier_mapping)
    supplier.validate()
    _required_text(supplier.supplier, "supplier_material.supplier", maximum=200)
    _required_text(supplier.sku, "supplier_material.sku", maximum=200)
    _required_text(supplier.lot_number, "supplier_material.lot_number", maximum=200)
    if re.fullmatch(r"[A-Z]{3}", supplier.currency) is None:
        raise ValueError("supplier_material.currency must be a three-letter code")
    if market not in supplier.regions and "GLOBAL" not in supplier.regions:
        raise ValueError("signed supplier offer does not cover the promotion market")
    if not supplier.in_stock:
        raise ValueError("signed promotion cannot activate an out-of-stock material")
    positive_allergens = {
        name for name, fraction in supplier.allergen_fractions.items() if fraction > 0
    }
    if not positive_allergens.issubset(set(eu_allergens)):
        raise ValueError("formulation spec omits quantified supplier allergens")
    moment = as_of
    if moment is None:
        moment = datetime.now(timezone.utc)
    if isinstance(moment, datetime):
        assessment_date = moment.date()
    else:
        assessment_date = moment
    constraints = RecipeConstraints(
        target_region=market,
        product_category=product_category,
    )
    assessment = SupplierRegistry([supplier]).assess_offer(
        supplier, constraints, assessment_date
    )
    if not assessment.qualified:
        details = sorted({*assessment.reasons, *assessment.missing_documents})
        raise ValueError(
            "signed supplier material is not qualified: " + "; ".join(details)
        )
    risk_tier = 1 if target_tier == "prototype_safe_active" else 2
    issued_date = datetime.fromisoformat(issued_at).date().isoformat()
    ingredient = Ingredient(
        ingredient_id=ingredient_id,
        name=name,
        aliases=aliases,
        cas_number=spec_cas,
        pyramid=pyramid,
        profile=profile,
        price_per_kg=supplier.price_per_kg,
        availability=1.0,
        rarity=rarity,
        risk_tier=risk_tier,
        odor_impact=odor_impact,
        max_concentrate_percent=max_concentrate_percent,
        formulation_ready=True,
        blocked=False,
        eu_allergens=eu_allergens,
        data_source=f"signed-industrial-promotion:{artifact_id}",
        currency=supplier.currency,
        density_g_ml=supplier.density_g_ml,
        active_strength_percent=supplier.active_strength_percent,
        carrier=supplier.carrier,
        solubility=solubility,
        oxidation_risk=oxidation_risk,
        discoloration_risk=discoloration_risk,
        shelf_life_months=shelf_life_months,
        data_verified_on=issued_date,
        approved_formulation_scopes=(formulation_scope_key(market, product_category),),
        approval_expires_at=expires_at,
        promotion_artifact_id=artifact_id,
        promotion_registry_sha256=registry_sha256,
    )
    return ActivatedIngredientPromotion(
        registry_id=registry_id,
        target_tier=target_tier,
        market=market,
        product_category=product_category,
        artifact_id=artifact_id,
        signer_id=signer_id,
        ingredient=ingredient,
        supplier_material=supplier,
    )
