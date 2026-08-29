"""Validated fine-grained odor words and coarse formulation projections."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .models import SCENT_DIMENSIONS


DESCRIPTOR_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "odor_descriptor_projections.json"
)


@dataclass(frozen=True)
class OdorDescriptorProjection:
    descriptor: str
    aliases: tuple[str, ...]
    formula_supported: bool
    projection_confidence: float
    profile: dict[str, float]
    unsupported_reason: str = ""


@dataclass(frozen=True)
class OdorDescriptorLexicon:
    version: str
    claim_boundary: str
    descriptors: tuple[OdorDescriptorProjection, ...]


@lru_cache(maxsize=1)
def load_builtin_odor_descriptor_lexicon() -> OdorDescriptorLexicon:
    payload = json.loads(DESCRIPTOR_DATA_PATH.read_text(encoding="utf-8"))
    version = payload.get("projection_version")
    boundary = payload.get("claim_boundary")
    rows = payload.get("descriptors")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("odor descriptor projection version is missing")
    if not isinstance(boundary, str) or not boundary.strip():
        raise ValueError("odor descriptor claim boundary is missing")
    if not isinstance(rows, list) or not rows:
        raise ValueError("odor descriptor projections are missing")

    dimensions = set(SCENT_DIMENSIONS)
    canonical_seen: set[str] = set()
    aliases_seen: set[str] = set()
    descriptors: list[OdorDescriptorProjection] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("odor descriptor projection must be an object")
        descriptor = str(row.get("descriptor", "")).casefold().strip()
        aliases = tuple(
            str(value).casefold().strip() for value in row.get("aliases", [])
        )
        supported = row.get("formula_supported")
        confidence = float(row.get("projection_confidence", -1.0))
        raw_profile = row.get("profile")
        reason = str(row.get("unsupported_reason", "")).strip()
        if not descriptor or descriptor in canonical_seen:
            raise ValueError(f"duplicate or empty odor descriptor: {descriptor!r}")
        if (
            not aliases
            or descriptor not in aliases
            or any(not alias for alias in aliases)
        ):
            raise ValueError(f"invalid aliases for odor descriptor: {descriptor}")
        duplicate_aliases = aliases_seen & set(aliases)
        if duplicate_aliases:
            raise ValueError(
                f"odor descriptor aliases are ambiguous: {sorted(duplicate_aliases)}"
            )
        if not isinstance(supported, bool):
            raise ValueError(f"formula support flag is invalid: {descriptor}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"projection confidence is invalid: {descriptor}")
        if not isinstance(raw_profile, dict):
            raise ValueError(f"projection profile is invalid: {descriptor}")
        profile = {str(name): float(value) for name, value in raw_profile.items()}
        if set(profile) - dimensions:
            raise ValueError(f"projection has unknown dimensions: {descriptor}")
        if any(not math.isfinite(value) or value <= 0.0 for value in profile.values()):
            raise ValueError(f"projection has invalid weights: {descriptor}")
        if supported:
            if not profile or not math.isclose(
                sum(profile.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"supported projection must sum to one: {descriptor}")
            if reason:
                raise ValueError(
                    f"supported projection cannot have unsupported reason: {descriptor}"
                )
        elif profile or not reason:
            raise ValueError(
                f"unsupported projection must be empty and explained: {descriptor}"
            )
        canonical_seen.add(descriptor)
        aliases_seen.update(aliases)
        descriptors.append(
            OdorDescriptorProjection(
                descriptor=descriptor,
                aliases=aliases,
                formula_supported=supported,
                projection_confidence=confidence,
                profile=profile,
                unsupported_reason=reason,
            )
        )
    return OdorDescriptorLexicon(
        version=version,
        claim_boundary=boundary,
        descriptors=tuple(descriptors),
    )
