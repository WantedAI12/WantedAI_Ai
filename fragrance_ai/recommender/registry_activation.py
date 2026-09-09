"""Strict runtime connection of the industrial odor-molecule registry.

The external registry is much broader than the curated formulation catalog.
This module connects every registry row for coverage/audit purposes and turns
only a narrow, public-data subset into risk-tier-2 experimental candidates. It never
labels those derived candidates supplier-qualified, manufacturing-ready, or
independently safety-approved.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import get_origin, get_type_hints

from .artifact_trust import sha256_file
from .catalog import IngredientCatalog, normalize_name
from .industrial_catalog import IndustrialIngredientRegistry
from .models import Ingredient, SCENT_DIMENSIONS
from .odor_integrity import (
    ODOR_INTEGRITY_VERSION, POSITIVE_ODOR_STATUS, REGISTRY_ODOR_SOURCE,
    assess_odor_assertions, is_registry_material, registry_odor_rejection,
    LEGACY_ODOR_PROJECTION, resolved_projection_version,
)


REGISTRY_CONDITIONAL_DATA_SOURCE = REGISTRY_ODOR_SOURCE
REGISTRY_CONDITIONAL_CAP_PERCENT = 100.0
REGISTRY_CONDITIONAL_CURRENCY = "USD_estimate_not_supplier_quote"
RUNTIME_CATALOG_SCHEMA = "perfumery-runtime-catalog/v2"
LEGACY_RUNTIME_CATALOG_SCHEMA = "perfumery-runtime-catalog/v1"

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
    odor_integrity_version: str = ""
    active_odorant_candidates: int = 0
    integrity_counts: dict[str, int] = field(default_factory=dict)
    connected_catalog_rows: int = 0

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def write_runtime_catalog(
    path: str | Path, catalog: IngredientCatalog, report: RegistryActivationReport,
    registry_stats: dict, *, wheel_sha256: str,
) -> str:
    """Precompute public registry activation once for an exact deployment wheel."""
    active = sum(item.formulation_ready and not item.blocked for item in catalog.ingredients)
    report = replace(report, connected_catalog_rows=len(catalog.ingredients),
                     experimental_formula_candidates=active, active_odorant_candidates=active)
    payload = {
        "schema": RUNTIME_CATALOG_SCHEMA,
        "wheel_sha256": wheel_sha256,
        "registry_sha256": report.registry_sha256,
        "ingredients": [asdict(item) for item in catalog.ingredients],
        "metadata": catalog.metadata,
        "activation_report": report.to_dict(),
        "registry_stats": registry_stats,
    }
    for item in catalog.ingredients:
        if item.formulation_ready and (reason := registry_odor_rejection(item)):
            raise ValueError(f"cannot serialize unverified active odorant: {item.ingredient_id}: {reason}")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = gzip.compress(raw, mtime=0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("runtime catalog output already exists")
    target.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()


def load_runtime_catalog(
    path: str | Path, *, expected_sha256: str,
    expected_wheel_sha256: str, expected_registry_sha256: str,
) -> tuple[IngredientCatalog, RegistryActivationReport, dict]:
    raw = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("runtime catalog hash mismatch")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
        decoded = archive.read(128 * 1024 * 1024 + 1)
    if len(decoded) > 128 * 1024 * 1024:
        raise ValueError("runtime catalog exceeds size limit")
    payload = json.loads(decoded)
    if payload.get("schema") not in {RUNTIME_CATALOG_SCHEMA, LEGACY_RUNTIME_CATALOG_SCHEMA}:
        raise ValueError("runtime catalog schema mismatch")
    if payload.get("wheel_sha256") != expected_wheel_sha256 or payload.get("registry_sha256") != expected_registry_sha256:
        raise ValueError("runtime catalog artifact binding mismatch")
    tuple_fields = {name for name, hint in get_type_hints(Ingredient).items() if get_origin(hint) is tuple}
    ingredients = [Ingredient(**{key: tuple(value) if key in tuple_fields else value for key, value in item.items()}) for item in payload["ingredients"]]
    report = RegistryActivationReport(**payload["activation_report"])
    connected = report.experimental_formula_candidates if payload["schema"] == LEGACY_RUNTIME_CATALOG_SCHEMA else report.connected_catalog_rows
    if report.registry_sha256 != expected_registry_sha256 or connected != len(ingredients):
        raise ValueError("runtime catalog coverage binding mismatch")
    metadata = dict(payload["metadata"])
    if payload["schema"] == LEGACY_RUNTIME_CATALOG_SCHEMA:
        quarantined = sum(is_registry_material(item) for item in ingredients)
        ingredients = [replace(item, profile={}, formulation_ready=False, blocked=True,
                               blocked_reason="legacy registry odor profile lacks explicit odor evidence",
                               odor_evidence_status="legacy_unverified_odor_profile")
                       if is_registry_material(item) else item for item in ingredients]
        report = replace(report, conditional_trace_candidates_active=0, strict_conditional_rows=0,
                         active_odorant_candidates=sum(item.formulation_ready and not item.blocked for item in ingredients),
                         experimental_formula_candidates=sum(item.formulation_ready and not item.blocked for item in ingredients),
                         connected_catalog_rows=len(ingredients),
                         odor_integrity_version=ODOR_INTEGRITY_VERSION,
                         integrity_counts={"legacy_registry_rows_quarantined": quarantined})
        metadata.update(industrial_registry_legacy_quarantined=quarantined, industrial_registry_conditional_trace_active=0,
                        industrial_registry_strict_conditional_rows=0,
                        industrial_registry_reference_only_rows=quarantined,
                        industrial_registry_active_odorants=report.active_odorant_candidates,
                        industrial_registry_experimental_formula_candidates=report.active_odorant_candidates,
                        industrial_registry_connected_catalog_rows=len(ingredients),
                        industrial_registry_integrity_version=ODOR_INTEGRITY_VERSION)
    else:
        active = sum(item.formulation_ready and not item.blocked for item in ingredients)
        if report.experimental_formula_candidates != active or report.active_odorant_candidates != active:
            raise ValueError("runtime active-candidate count mismatch")
        for item in ingredients:
            if item.formulation_ready and (reason := registry_odor_rejection(item)):
                raise ValueError(f"runtime odor-evidence binding invalid: {item.ingredient_id}: {reason}")
            if is_registry_material(item) and item.odor_registry_sha256 != expected_registry_sha256:
                raise ValueError("runtime odor-registry binding mismatch")
    catalog = IngredientCatalog(ingredients, metadata)
    return catalog, report, payload["registry_stats"]


def _project_profile(
    descriptors: tuple[str, ...], fallback_text: str
) -> dict[str, float]:
    # Compatibility helper: fallback text is deliberately not scent evidence.
    _, items = assess_odor_assertions(tuple("goodscents:" + normalize_name(value) for value in descriptors))
    return dict(items)


def _calculated_structure_properties(smiles):
    """Builder-only descriptors; never invent vapor pressure or odor threshold."""
    try:
        import rdkit
    except ImportError as error:
        raise RuntimeError("Registry construction requires perfumery-ai-core[registry-build]; inference can use a precomputed runtime catalog without RDKit") from error
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return {}, "invalid_or_multicomponent_structure"
    if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
        return {}, "nonorganic_structure"
    if any(atom.GetFormalCharge() for atom in molecule.GetAtoms()):
        return {}, "charged_structure_requires_review"
    values = {"molecular_weight": float(Descriptors.MolWt(molecule)), "xlogp": float(Crippen.MolLogP(molecule)),
              "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)), "hbond_donors": int(Lipinski.NumHDonors(molecule)),
              "hbond_acceptors": int(Lipinski.NumHAcceptors(molecule)), "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule))}
    return values, "rdkit-" + rdkit.__version__ + "-calculated-not-measured"


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
    verified_odor_assertions: dict[str, tuple[str, ...]] | None = None,
    verified_odor_refs: dict[str, tuple[str, ...]] | None = None,
    odor_projection_version: str = LEGACY_ODOR_PROJECTION,
) -> tuple[IngredientCatalog, RegistryActivationReport]:
    """Connect the full registry and add strict tier-2 full-range candidates."""

    odor_projection_version = resolved_projection_version(odor_projection_version)
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
    integrity_counts: dict[str, int] = {}
    blocked_cas = {item.cas_number for item in catalog.ingredients if item.blocked and item.cas_number}
    blocked_names = {normalize_name(name) for item in catalog.ingredients if item.blocked for name in item.all_names()}

    for row in rows:
        short_id = row.registry_id.split(":", 1)[-1]
        ingredient_id = f"registry_{short_id}"
        preferred_name = row.preferred_name or f"Registry molecule {short_id[:8]}"
        assertions = tuple(verified_odor_assertions.get(row.registry_id, ())) if verified_odor_assertions is not None else row.odor_assertions
        refs = tuple((verified_odor_refs or {}).get(row.registry_id, ())) if verified_odor_assertions is not None else row.odor_evidence_refs
        odor_status, projected = assess_odor_assertions(assertions, odor_projection_version)
        profile = dict(projected)
        properties, properties_version = {}, ""
        alerts = row.structural_alerts
        if row.cas_number in blocked_cas or any(normalize_name(value) in blocked_names for value in (preferred_name, *row.aliases)):
            odor_status = "known_catalog_policy_blocked"
        elif odor_status == POSITIVE_ODOR_STATUS and alerts:
            odor_status = "structural_review_required"
        elif odor_status == POSITIVE_ODOR_STATUS:
            properties, properties_version = _calculated_structure_properties(row.canonical_smiles)
            if not properties:
                odor_status = properties_version
                alerts = (*alerts, properties_version)
            elif row.molecular_weight is not None and abs(properties["molecular_weight"] - row.molecular_weight) > max(.2, .01 * row.molecular_weight):
                odor_status = "structure_mass_identity_mismatch"
                alerts = (*alerts, odor_status)
        integrity_counts[odor_status] = integrity_counts.get(odor_status, 0) + 1
        eligible = odor_status == POSITIVE_ODOR_STATUS
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
                pyramid=_pyramid({name: profile.get(name, 0.) for name in SCENT_DIMENSIONS}, molecular_weight),
                profile=profile,
                price_per_kg=_estimated_price(
                    molecular_weight, row.source_count
                ),
                availability=min(0.95, 0.75 + 0.04 * (row.source_count - 2)),
                rarity="standard",
                risk_tier=2,
                odor_impact=1.0,  # Neutral prior: publication count is not odor strength.
                max_concentrate_percent=REGISTRY_CONDITIONAL_CAP_PERCENT,
                formulation_ready=eligible,
                blocked=not eligible,
                blocked_reason=None if eligible else "registry integrity: " + odor_status,
                data_source=REGISTRY_CONDITIONAL_DATA_SOURCE,
                currency=REGISTRY_CONDITIONAL_CURRENCY,
                oxidation_risk="unknown",
                discoloration_risk="unknown",
                odor_integrity_version=ODOR_INTEGRITY_VERSION,
                odor_projection_version=odor_projection_version,
                odor_evidence_status=odor_status,
                odor_assertions=assertions,
                odor_evidence_refs=refs,
                odor_registry_sha256=calculated_sha256,
                registry_structural_alerts=alerts,
                structure_smiles=row.canonical_smiles,
                structure_properties=properties,
                structure_properties_version=properties_version,
            )
        )
        if eligible and (reason := registry_odor_rejection(activated[-1])):
            integrity_counts[POSITIVE_ODOR_STATUS] -= 1
            integrity_counts[reason] = integrity_counts.get(reason, 0) + 1
            activated[-1] = replace(activated[-1], formulation_ready=False, blocked=True, blocked_reason=reason, odor_evidence_status=reason)

    report = RegistryActivationReport(
        registry_sha256=calculated_sha256,
        reference_molecules_connected=stats["reference_molecules"],
        structurally_blocked=stats["structural_review_required"],
        evidence_pending=stats["screening_evidence_pending"],
        strict_conditional_rows=sum(item.formulation_ready for item in activated),
        conditional_trace_candidates_active=sum(item.formulation_ready for item in activated),
        experimental_formula_candidates=sum(item.formulation_ready and not item.blocked for item in [*catalog.ingredients, *activated]),
        blocked_known_policy=integrity_counts.get("known_catalog_policy_blocked", 0),
        blocked_unsupported_descriptor=sum(count for status, count in integrity_counts.items() if status != POSITIVE_ODOR_STATUS),
        alias_collisions_resolved=collisions,
        odor_integrity_version=ODOR_INTEGRITY_VERSION,
        active_odorant_candidates=sum(item.formulation_ready and not item.blocked for item in [*catalog.ingredients, *activated]),
        integrity_counts=integrity_counts,
        connected_catalog_rows=len(catalog.ingredients) + len(activated),
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
            "industrial_registry_strict_conditional_rows": report.strict_conditional_rows,
            "industrial_registry_conditional_trace_active": report.conditional_trace_candidates_active,
            "industrial_registry_experimental_formula_candidates": report.experimental_formula_candidates,
            "industrial_registry_activation_mode": report.activation_mode,
            "industrial_registry_claim_boundary": report.claim_boundary,
            "industrial_registry_active_odorants": report.active_odorant_candidates,
            "industrial_registry_reference_only_rows": len(activated) - report.conditional_trace_candidates_active,
            "industrial_registry_integrity_version": ODOR_INTEGRITY_VERSION,
            "industrial_registry_calculated_structure_records": sum(bool(item.structure_properties) and item.formulation_ready for item in activated),
            "industrial_registry_connected_catalog_rows": report.connected_catalog_rows,
            "industrial_registry_procurement_status": "engineering_estimates_not_supplier_quotes_or_verified_inventory",
        }
    )
    return IngredientCatalog([*catalog.ingredients, *activated], metadata), report
