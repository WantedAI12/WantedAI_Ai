"""Externally evidenced reference targets for formula-to-formula simulation.

Text-only requests have no measurable molecular target.  This module provides
an explicit path for an independently supplied reference formula or GC-MS
composition. The engine has no text-to-target scoring fallback, and no
reference data are bundled or inferred by this module.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from .models import Ingredient, RecipeConstraints, RecipeLine


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_COMPOSITION_BASES = frozenset(
    {"verified_formula", "quantitative_gc_ms", "quantitative_headspace_gc_ms"}
)


def _digest(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"reference evidence is not a regular file: {source}")
    result = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class ReferenceEvidence:
    evidence_type: str
    path: str
    sha256: str
    issued_on: date
    expires_on: date | None = None

    def verify(self, *, as_of: date) -> str:
        expected = self.sha256.strip().lower()
        if not _SHA256.fullmatch(expected):
            raise ValueError(f"{self.evidence_type}: invalid SHA-256")
        if self.issued_on > as_of:
            raise ValueError(f"{self.evidence_type}: evidence is future-dated")
        if self.expires_on is not None and self.expires_on < as_of:
            raise ValueError(f"{self.evidence_type}: evidence is expired")
        if _digest(self.path) != expected:
            raise ValueError(f"{self.evidence_type}: evidence bytes changed")
        return expected


@dataclass(frozen=True)
class ReferenceComponent:
    ingredient_id: str
    concentrate_percent: float


@dataclass(frozen=True)
class ReferenceTarget:
    target_id: str
    version: str
    composition_basis: str
    product_category: str
    product_concentration_percent: float
    matrix_id: str
    components: tuple[ReferenceComponent, ...]
    evidence: tuple[ReferenceEvidence, ...]


@dataclass(frozen=True)
class ResolvedReferenceTarget:
    target_id: str
    version: str
    composition_basis: str
    matrix_id: str
    lines: tuple[RecipeLine, ...]
    evidence_sha256: tuple[str, ...]


class ReferenceTargetStore:
    """Validated reference targets with no text-to-formula fallback."""

    def __init__(self, targets: Iterable[ReferenceTarget] = ()) -> None:
        self._targets: dict[str, ReferenceTarget] = {}
        for target in targets:
            target_id = target.target_id.strip()
            if not target_id:
                raise ValueError("reference target_id cannot be blank")
            if target_id in self._targets:
                raise ValueError(f"duplicate reference target_id: {target_id}")
            self._targets[target_id] = target

    def resolve(
        self,
        target_id: str,
        *,
        ingredients: dict[str, Ingredient],
        constraints: RecipeConstraints,
        as_of: date,
    ) -> ResolvedReferenceTarget:
        requested = target_id.strip()
        if not requested:
            raise ValueError("reference_target_id is required")
        target = self._targets.get(requested)
        if target is None:
            raise ValueError("reference target is not registered")
        if not target.version.strip():
            raise ValueError("reference target version is required")
        if target.composition_basis not in ALLOWED_COMPOSITION_BASES:
            raise ValueError("reference target composition basis is not quantitative")
        if (
            target.product_category.casefold()
            != constraints.product_category.casefold()
        ):
            raise ValueError("reference target product category does not match request")
        target_concentration = float(target.product_concentration_percent)
        request_concentration = float(constraints.product_concentration_percent)
        if (
            not math.isfinite(target_concentration)
            or not 0.0 < target_concentration <= 100.0
        ):
            raise ValueError("reference target product concentration is invalid")
        if not math.isfinite(request_concentration):
            raise ValueError("request product concentration is invalid")
        if abs(target_concentration - request_concentration) > 1e-6:
            raise ValueError(
                "reference target product concentration does not match request"
            )
        requested_matrix = constraints.commercial_product_base_id.strip()
        if not requested_matrix:
            raise ValueError(
                "reference comparison requires commercial_product_base_id for matrix matching"
            )
        if requested_matrix != target.matrix_id:
            raise ValueError(
                "reference target matrix does not match candidate product base"
            )
        if not target.evidence:
            raise ValueError("reference target has no source evidence")
        evidence_types = {item.evidence_type for item in target.evidence}
        if "composition" not in evidence_types:
            raise ValueError("reference target requires composition evidence")
        evidence_sha = tuple(
            sorted(item.verify(as_of=as_of) for item in target.evidence)
        )

        if not target.components:
            raise ValueError("reference target composition is empty")
        percentages: list[float] = []
        for component in target.components:
            try:
                percent = float(component.concentrate_percent)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "reference component percentages must be numeric"
                ) from error
            if not math.isfinite(percent) or percent <= 0.0:
                raise ValueError(
                    "reference component percentages must be finite and positive"
                )
            percentages.append(percent)
        total = sum(percentages)
        if abs(total - 100.0) > 0.01:
            raise ValueError("reference target composition must sum to 100%")
        seen: set[str] = set()
        lines: list[RecipeLine] = []
        for component, percent in zip(target.components, percentages):
            if not isinstance(component.ingredient_id, str):
                raise ValueError("reference component ingredient_id must be text")
            ingredient_id = component.ingredient_id.strip()
            if not ingredient_id:
                raise ValueError("reference component ingredient_id cannot be blank")
            if ingredient_id in seen:
                raise ValueError(f"duplicate reference component: {ingredient_id}")
            seen.add(ingredient_id)
            ingredient = ingredients.get(ingredient_id)
            if ingredient is None:
                raise ValueError(
                    f"reference component is not mapped to the active catalog: {ingredient_id}"
                )
            lines.append(
                RecipeLine(
                    ingredient_id=ingredient_id,
                    name=ingredient.name,
                    pyramid=ingredient.pyramid,
                    concentrate_percent=percent,
                    finished_product_percent=(percent * target_concentration / 100.0),
                    volume_ml_for_batch=None,
                    price_per_kg=ingredient.price_per_kg,
                    availability=ingredient.availability,
                    risk_tier=ingredient.risk_tier,
                    reason=(
                        f"evidenced reference target {target.target_id}@{target.version}"
                    ),
                    active_material_percent=(
                        percent * ingredient.active_strength_percent / 100.0
                    ),
                    density_g_ml=ingredient.density_g_ml,
                    active_strength_percent=ingredient.active_strength_percent,
                    carrier=ingredient.carrier,
                    data_source=f"reference-target:{target.composition_basis}",
                    approved_formulation_scopes=ingredient.approved_formulation_scopes,
                    approval_expires_at=ingredient.approval_expires_at,
                    promotion_artifact_id=ingredient.promotion_artifact_id,
                )
            )
        return ResolvedReferenceTarget(
            target_id=target.target_id,
            version=target.version,
            composition_basis=target.composition_basis,
            matrix_id=target.matrix_id,
            lines=tuple(sorted(lines, key=lambda item: item.ingredient_id)),
            evidence_sha256=evidence_sha,
        )
