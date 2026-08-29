"""Verified base and packaging evidence for manufacturing readiness.

The recipe engine may always calculate a mass balance.  It must not turn that
arithmetic into a lab/manufacturing readiness claim unless the exact product
base and packaging system have current, byte-verifiable technical evidence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from .models import Ingredient, RecipeConstraints, RecipeLine


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"evidence is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_tags(values: Iterable[str]) -> frozenset[str]:
    return frozenset(str(value).strip().casefold() for value in values if str(value).strip())


@dataclass(frozen=True)
class TechnicalEvidence:
    """A technical document that is re-hashed whenever it is relied upon."""

    path: str
    sha256: str
    issued_on: date
    expires_on: date
    evidence_type: str

    def validate(self, *, as_of: date) -> list[str]:
        failures: list[str] = []
        expected = self.sha256.strip().lower()
        if not _SHA256.fullmatch(expected):
            failures.append(f"{self.evidence_type}:invalid_sha256")
            return failures
        if self.issued_on > as_of:
            failures.append(f"{self.evidence_type}:future_issued_on")
        if self.expires_on < as_of:
            failures.append(f"{self.evidence_type}:expired")
        try:
            actual = _sha256_file(self.path)
        except (OSError, ValueError):
            failures.append(f"{self.evidence_type}:document_unavailable")
        else:
            if actual != expected:
                failures.append(f"{self.evidence_type}:document_hash_mismatch")
        return failures


@dataclass(frozen=True)
class ProductBaseProfile:
    """Versioned physical/process profile for the exact finished-product base."""

    base_id: str
    version: str
    compatible_product_categories: tuple[str, ...]
    solvent_system: tuple[str, ...]
    density_g_ml: float
    flash_point_c: float
    maximum_process_temperature_c: float
    evidence: tuple[TechnicalEvidence, ...]


@dataclass(frozen=True)
class PackagingProfile:
    """Versioned packaging system with solvent and category compatibility."""

    packaging_id: str
    version: str
    compatible_product_categories: tuple[str, ...]
    compatible_solvent_systems: tuple[str, ...]
    evidence: tuple[TechnicalEvidence, ...]


@dataclass(frozen=True)
class ManufacturingReadiness:
    ready_for_lab_trial: bool
    ready_for_manufacture: bool
    blockers: tuple[str, ...]
    base_profile_version: str = ""
    packaging_profile_version: str = ""


class ManufacturingProfileRegistry:
    """In-memory registry whose evidence bytes remain the source of truth."""

    def __init__(
        self,
        *,
        base_profiles: Iterable[ProductBaseProfile] = (),
        packaging_profiles: Iterable[PackagingProfile] = (),
    ) -> None:
        self._bases = self._index(
            ((item.base_id, item) for item in base_profiles), "base_id"
        )
        self._packaging = self._index(
            ((item.packaging_id, item) for item in packaging_profiles),
            "packaging_id",
        )

    @staticmethod
    def _index(items: Iterable[tuple[str, object]], field: str) -> dict[str, object]:
        indexed: dict[str, object] = {}
        for raw_key, value in items:
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{field} cannot be blank")
            if key in indexed:
                raise ValueError(f"duplicate {field}: {key}")
            indexed[key] = value
        return indexed

    def assess(
        self,
        *,
        lines: Iterable[RecipeLine],
        ingredients_by_id: dict[str, Ingredient],
        constraints: RecipeConstraints,
        as_of: date,
        stability_passed: bool,
    ) -> ManufacturingReadiness:
        blockers: list[str] = []
        base_id = constraints.commercial_product_base_id.strip()
        packaging_id = constraints.commercial_packaging_id.strip()
        base = self._bases.get(base_id) if base_id else None
        packaging = self._packaging.get(packaging_id) if packaging_id else None

        if not base_id:
            blockers.append("product_base_id_missing")
        elif base is None:
            blockers.append("product_base_profile_not_registered")
        if not packaging_id:
            blockers.append("packaging_id_missing")
        elif packaging is None:
            blockers.append("packaging_profile_not_registered")

        category = constraints.product_category.strip().casefold()
        if isinstance(base, ProductBaseProfile):
            if not base.version.strip():
                blockers.append("product_base_version_missing")
            if base.density_g_ml <= 0:
                blockers.append("product_base_density_invalid")
            if base.flash_point_c <= -273.15:
                blockers.append("product_base_flash_point_invalid")
            if base.maximum_process_temperature_c >= base.flash_point_c:
                blockers.append("process_temperature_not_below_flash_point")
            if category not in _normalized_tags(base.compatible_product_categories):
                blockers.append("product_base_category_incompatible")
            if not base.solvent_system:
                blockers.append("product_base_solvent_system_missing")
            if not base.evidence:
                blockers.append("product_base_evidence_missing")
            for document in base.evidence:
                blockers.extend(document.validate(as_of=as_of))

        if isinstance(packaging, PackagingProfile):
            if not packaging.version.strip():
                blockers.append("packaging_version_missing")
            if category not in _normalized_tags(packaging.compatible_product_categories):
                blockers.append("packaging_category_incompatible")
            if not packaging.evidence:
                blockers.append("packaging_evidence_missing")
            for document in packaging.evidence:
                blockers.extend(document.validate(as_of=as_of))

        if isinstance(base, ProductBaseProfile) and isinstance(packaging, PackagingProfile):
            base_solvents = _normalized_tags(base.solvent_system)
            packaging_solvents = _normalized_tags(packaging.compatible_solvent_systems)
            if not base_solvents or not base_solvents <= packaging_solvents:
                blockers.append("packaging_solvent_system_incompatible")

        selected_lines = list(lines)
        if not selected_lines:
            blockers.append("formula_lines_missing")
        base_solvents = (
            _normalized_tags(base.solvent_system)
            if isinstance(base, ProductBaseProfile)
            else frozenset()
        )
        for line in selected_lines:
            ingredient = ingredients_by_id.get(line.ingredient_id)
            if ingredient is None:
                blockers.append(f"{line.ingredient_id}:ingredient_profile_missing")
                continue
            # Exact lot strength is already bound by the release scope.  These
            # physical fields are separately required for a viable batch.
            if ingredient.density_g_ml is None or ingredient.density_g_ml <= 0:
                blockers.append(f"{line.ingredient_id}:density_unverified")
            solubility = _normalized_tags(ingredient.solubility)
            if not solubility:
                blockers.append(f"{line.ingredient_id}:solubility_unverified")
            elif base_solvents and not (solubility & base_solvents):
                blockers.append(f"{line.ingredient_id}:base_solubility_incompatible")
            if ingredient.oxidation_risk == "unknown":
                blockers.append(f"{line.ingredient_id}:oxidation_risk_unverified")
            if ingredient.discoloration_risk == "unknown":
                blockers.append(f"{line.ingredient_id}:discoloration_risk_unverified")
            if ingredient.shelf_life_months is None:
                blockers.append(f"{line.ingredient_id}:shelf_life_unverified")

        unique = tuple(sorted(set(blockers)))
        lab_ready = not unique
        manufacture_ready = lab_ready and stability_passed
        if lab_ready and not stability_passed:
            unique = ("finished_product_stability_not_passed",)
        return ManufacturingReadiness(
            ready_for_lab_trial=lab_ready,
            ready_for_manufacture=manufacture_ready,
            blockers=unique,
            base_profile_version=(base.version if isinstance(base, ProductBaseProfile) else ""),
            packaging_profile_version=(
                packaging.version if isinstance(packaging, PackagingProfile) else ""
            ),
        )
