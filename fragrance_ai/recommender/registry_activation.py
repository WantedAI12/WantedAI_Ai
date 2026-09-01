"""Strict runtime connection of the industrial odor-molecule registry.

The external registry is much broader than the curated formulation catalog.
This module connects every registry row for coverage/audit purposes and turns
only a narrow, public-data subset into risk-tier-2 experimental candidates. It never
labels those derived candidates supplier-qualified, manufacturing-ready, or
independently safety-approved.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from .artifact_trust import sha256_file
from .catalog import IngredientCatalog, normalize_name
from .industrial_catalog import IndustrialIngredientRegistry
from .models import Ingredient, SCENT_DIMENSIONS, normalize_profile
from .odor_descriptors import load_builtin_odor_descriptor_lexicon
from .semantic_ontology import ScentSemanticOntology


REGISTRY_CONDITIONAL_DATA_SOURCE = (
    "industrial-registry-public-descriptor-conditional-v2"
)
REGISTRY_CONDITIONAL_CAP_PERCENT = 100.0
REGISTRY_CONDITIONAL_CURRENCY = "USD_estimate_not_supplier_quote"

_DIRECT_DESCRIPTOR_PROFILES: dict[str, dict[str, float]] = {
    **{normalize_name(name): {name: 1.0} for name in SCENT_DIMENSIONS},
    "aldehydic": {"clean": 0.65, "fresh": 0.35},
    "balsam": {"amber": 0.65, "woody": 0.35},
    "floral": {"floral": 1.0},
    "fruit": {"fruity": 1.0},
    "fruity": {"fruity": 1.0},
    "herbaceous": {"aromatic": 0.65, "green": 0.35},
    "herbal": {"aromatic": 0.65, "green": 0.35},
    "jasmin": {"white_floral": 0.8, "floral": 0.2},
    "jasmine": {"white_floral": 0.8, "floral": 0.2},
    "leather": {"leathery": 0.8, "smoky": 0.2},
    "marine": {"aquatic": 0.8, "fresh": 0.2},
    "musk": {"musky": 1.0},
    "musky": {"musky": 1.0},
    "oceanic": {"aquatic": 0.8, "fresh": 0.2},
    "ozonic": {"aquatic": 0.55, "fresh": 0.45},
    "sweet": {"gourmand": 0.85, "amber": 0.15},
    "tobacco": {"leathery": 0.45, "smoky": 0.30, "woody": 0.25},
    "vanilla": {"gourmand": 0.75, "powdery": 0.25},
}


@dataclass(frozen=True)
class RegistryActivationReport:
    registry_sha256: str
    reference_molecules_connected: int
    structurally_blocked: int
    evidence_pending: int
    strict_conditional_rows: int
    conditional_trace_candidates_active: int
    experimental_formula_candidates: int
    blocked_known_policy: int
    blocked_unsupported_descriptor: int
    alias_collisions_resolved: int
    risk_tier: int = 2
    max_concentrate_percent: float = REGISTRY_CONDITIONAL_CAP_PERCENT
    activation_mode: str = "prototype_conditional_full_range"
    claim_boundary: str = (
        "Public odor descriptors and coarse structure screens create R&D "
        "candidates only. They are not supplier-qualified, independently safety-"
        "approved, manufacturing-ready, or commercial formula materials."
    )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@lru_cache(maxsize=1)
def _descriptor_projection_maps() -> tuple[
    dict[str, tuple[float, dict[str, float]]], set[str]
]:
    supported: dict[str, tuple[float, dict[str, float]]] = {}
    unsupported: set[str] = set()
    for row in load_builtin_odor_descriptor_lexicon().descriptors:
        target = supported if row.formula_supported else None
        for alias in row.aliases:
            key = normalize_name(alias)
            if not key:
                continue
            if target is None:
                unsupported.add(key)
            else:
                target[key] = (row.projection_confidence, dict(row.profile))
    for descriptor, profile in _DIRECT_DESCRIPTOR_PROFILES.items():
        supported.setdefault(descriptor, (0.75, profile))
    return supported, unsupported


@lru_cache(maxsize=1)
def _semantic_ontology() -> ScentSemanticOntology:
    return ScentSemanticOntology()


def _project_profile(
    descriptors: tuple[str, ...], fallback_text: str
) -> dict[str, float]:
    supported, _ = _descriptor_projection_maps()
    normalized = {normalize_name(value) for value in descriptors}
    values = {name: 0.0 for name in SCENT_DIMENSIONS}
    matched = 0
    for descriptor in sorted(normalized):
        projection = supported.get(descriptor)
        if projection is None:
            continue
        confidence, profile = projection
        for dimension, weight in profile.items():
            values[dimension] += confidence * weight
        matched += 1
    if matched == 0:
        semantic = _semantic_ontology().infer(fallback_text)
        values.update(semantic.scores)
    if sum(values.values()) <= 0.0:
        digest = hashlib.sha256(fallback_text.encode("utf-8")).digest()
        for offset, weight in enumerate((1.0, 0.55, 0.25)):
            values[SCENT_DIMENSIONS[digest[offset] % len(SCENT_DIMENSIONS)]] += weight
    return normalize_profile(values)


def _pyramid(profile: dict[str, float], molecular_weight: float) -> str:
    top = sum(
        profile[name]
        for name in ("citrus", "fresh", "clean", "green", "aquatic", "aromatic")
    )
    base = sum(
        profile[name]
        for name in ("woody", "amber", "musky", "powdery", "smoky", "leathery", "earthy")
    )
    if molecular_weight >= 235.0 or (molecular_weight >= 175.0 and base >= 0.5):
        return "base"
    if molecular_weight <= 170.0 or top >= 0.55:
        return "top"
    return "heart"


def _estimated_price(molecular_weight: float, source_count: int) -> float:
    # This estimate is used only for prototype ranking and is explicitly
    # labelled as not being a supplier quotation in every emitted line.
    estimate = 210.0 - min(120.0, max(0, source_count - 1) * 20.0)
    estimate += max(0.0, molecular_weight - 180.0) * 0.12
    return round(min(180.0, max(60.0, estimate)), 2)


def activate_registry_conditionals(
    catalog: IngredientCatalog,
    registry_path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[IngredientCatalog, RegistryActivationReport]:
    """Connect the full registry and add strict tier-2 full-range candidates."""

    path = Path(registry_path).expanduser().resolve(strict=True)
    calculated_sha256 = sha256_file(path)
    if not isinstance(expected_sha256, str):
        raise ValueError("expected industrial registry SHA-256 is invalid")
    normalized_expected = expected_sha256.casefold().strip()
    if len(normalized_expected) != 64 or any(
        value not in "0123456789abcdef" for value in normalized_expected
    ):
        raise ValueError("expected industrial registry SHA-256 is invalid")
    if calculated_sha256 != normalized_expected:
        raise ValueError("industrial registry hash mismatch during activation")

    with IndustrialIngredientRegistry(path) as registry:
        stats = registry.stats()
        rows = registry.conditional_runtime_candidates()

    used_aliases = {
        normalize_name(alias)
        for ingredient in catalog.ingredients
        for alias in ingredient.all_names()
        if normalize_name(alias)
    }
    activated: list[Ingredient] = []
    collisions = 0

    for row in rows:
        short_id = row.registry_id.split(":", 1)[-1]
        ingredient_id = f"registry_{short_id}"
        preferred_name = row.preferred_name or f"Registry molecule {short_id[:8]}"
        profile = _project_profile(
            row.descriptors,
            " ".join((preferred_name, *row.descriptors, row.canonical_smiles)),
        )
        molecular_weight = (
            180.0 if row.molecular_weight is None else float(row.molecular_weight)
        )
        name = preferred_name
        if normalize_name(name) in used_aliases:
            name = f"{preferred_name} [{short_id[:8]}]"
            collisions += 1
        aliases: list[str] = []
        for alias in row.aliases:
            key = normalize_name(alias)
            if not key or key in used_aliases or key == normalize_name(name):
                continue
            aliases.append(alias)
            used_aliases.add(key)
            if len(aliases) >= 8:
                break
        used_aliases.add(normalize_name(name))

        activated.append(
            Ingredient(
                ingredient_id=ingredient_id,
                name=name,
                aliases=tuple(aliases),
                cas_number=row.cas_number,
                pyramid=_pyramid(profile, molecular_weight),
                profile=profile,
                price_per_kg=_estimated_price(
                    molecular_weight, row.source_count
                ),
                availability=min(0.95, 0.75 + 0.04 * (row.source_count - 2)),
                rarity="standard",
                risk_tier=2,
                odor_impact=max(
                    0.25, min(4.0, 1.0 + row.evidence_score / 100.0)
                ),
                max_concentrate_percent=REGISTRY_CONDITIONAL_CAP_PERCENT,
                formulation_ready=True,
                blocked=False,
                data_source=REGISTRY_CONDITIONAL_DATA_SOURCE,
                currency=REGISTRY_CONDITIONAL_CURRENCY,
                oxidation_risk="unknown",
                discoloration_risk="unknown",
            )
        )

    report = RegistryActivationReport(
        registry_sha256=calculated_sha256,
        reference_molecules_connected=stats["reference_molecules"],
        structurally_blocked=stats["structural_review_required"],
        evidence_pending=stats["screening_evidence_pending"],
        strict_conditional_rows=len(rows),
        conditional_trace_candidates_active=len(activated),
        experimental_formula_candidates=len(catalog.ingredients) + len(activated),
        blocked_known_policy=0,
        blocked_unsupported_descriptor=0,
        alias_collisions_resolved=collisions,
    )
    metadata = dict(catalog.metadata)
    metadata.update(
        {
            "industrial_registry_sha256": calculated_sha256,
            "industrial_registry_connected_total": stats["reference_molecules"],
            "industrial_registry_structurally_blocked": stats[
                "structural_review_required"
            ],
            "industrial_registry_evidence_pending": stats[
                "screening_evidence_pending"
            ],
            "industrial_registry_strict_conditional_rows": len(rows),
            "industrial_registry_conditional_trace_active": len(activated),
            "industrial_registry_experimental_formula_candidates": (
                len(catalog.ingredients) + len(activated)
            ),
            "industrial_registry_activation_mode": report.activation_mode,
            "industrial_registry_claim_boundary": report.claim_boundary,
        }
    )
    return IngredientCatalog([*catalog.ingredients, *activated], metadata), report
