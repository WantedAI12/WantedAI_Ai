"""Ingredient catalog and historical fragrance reference corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .data_hub import NonHumanDataHub
from .models import Ingredient, PYRAMID_LEVELS, SCENT_DIMENSIONS


def normalize_name(value: str) -> str:
    """Normalize names without discarding non-Latin letters or Hangul."""
    return "".join(character for character in value.casefold() if character.isalnum())


def find_text_spans(text: str, alias: str) -> list[tuple[int, int]]:
    """Return case-insensitive alias spans without Latin substring collisions.

    ASCII boundaries prevent short material aliases such as ``PEA`` from
    matching ``pear`` or ``peach``. Korean aliases intentionally allow attached
    particles. Whitespace inside multiword aliases remains flexible.
    """

    lowered = text.casefold()
    normalized_alias = alias.casefold().strip()
    if not normalized_alias:
        return []
    pieces = re.split(r"\s+", normalized_alias)
    expression = r"\s+".join(re.escape(piece) for piece in pieces)
    if normalized_alias[0].isascii() and normalized_alias[0].isalnum():
        expression = rf"(?<![0-9a-z]){expression}"
    if normalized_alias[-1].isascii() and normalized_alias[-1].isalnum():
        expression = rf"{expression}(?![0-9a-z])"
    return [match.span() for match in re.finditer(expression, lowered)]


@dataclass(frozen=True)
class IngredientMention:
    ingredient: Ingredient
    alias: str
    start: int
    end: int


class IngredientCatalog:
    """Separates reference-only global names from formulation-ready materials."""

    def __init__(self, ingredients: Iterable[Ingredient], metadata: dict | None = None):
        self.ingredients = list(ingredients)
        self.metadata = metadata or {}
        self._alias_index: dict[str, Ingredient] = {}
        for ingredient in self.ingredients:
            for name in ingredient.all_names():
                key = normalize_name(name)
                if key:
                    self._alias_index[key] = ingredient
        self._validate()

    @classmethod
    def load_builtin(cls) -> "IngredientCatalog":
        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "safe_ingredient_catalog.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        ingredients = []
        for item in payload["ingredients"]:
            ingredients.append(
                Ingredient(
                    ingredient_id=item["ingredient_id"],
                    name=item["name"],
                    aliases=tuple(item.get("aliases", [])),
                    cas_number=item.get("cas_number"),
                    pyramid=item["pyramid"],
                    profile=item["profile"],
                    price_per_kg=float(item["price_per_kg"]),
                    availability=float(item["availability"]),
                    rarity=item["rarity"],
                    risk_tier=int(item["risk_tier"]),
                    odor_impact=float(item.get("odor_impact", 1.0)),
                    max_concentrate_percent=float(item["max_concentrate_percent"]),
                    formulation_ready=bool(item["formulation_ready"]),
                    blocked=bool(item.get("blocked", False)),
                    blocked_reason=item.get("blocked_reason"),
                    eu_allergens=tuple(item.get("eu_allergens", [])),
                    data_source=item.get("data_source", "curated-workspace-v1"),
                    currency=item.get("currency", "USD_estimate"),
                    density_g_ml=(
                        float(item["density_g_ml"])
                        if item.get("density_g_ml") is not None
                        else None
                    ),
                    active_strength_percent=float(
                        item.get("active_strength_percent", 100.0)
                    ),
                    carrier=item.get("carrier"),
                    solubility=tuple(item.get("solubility", [])),
                    oxidation_risk=item.get("oxidation_risk", "unknown"),
                    discoloration_risk=item.get("discoloration_risk", "unknown"),
                    shelf_life_months=(
                        int(item["shelf_life_months"])
                        if item.get("shelf_life_months") is not None
                        else None
                    ),
                    data_verified_on=item.get("data_verified_on"),
                )
            )
        return cls(ingredients, payload.get("metadata", {}))

    def _validate(self) -> None:
        ids: set[str] = set()
        aliases: dict[str, str] = {}
        for ingredient in self.ingredients:
            if ingredient.ingredient_id in ids:
                raise ValueError(f"Duplicate ingredient id: {ingredient.ingredient_id}")
            ids.add(ingredient.ingredient_id)
            if ingredient.pyramid not in PYRAMID_LEVELS:
                raise ValueError(f"Invalid pyramid level: {ingredient.pyramid}")
            unknown = set(ingredient.profile) - set(SCENT_DIMENSIONS)
            if unknown:
                raise ValueError(
                    f"Unknown scent dimensions for {ingredient.name}: {unknown}"
                )
            if ingredient.formulation_ready and sum(ingredient.profile.values()) <= 0:
                raise ValueError(f"Missing profile for {ingredient.name}")
            if not 0 <= ingredient.availability <= 1:
                raise ValueError(f"Invalid availability for {ingredient.name}")
            if not 0 < ingredient.active_strength_percent <= 100:
                raise ValueError(f"Invalid active strength for {ingredient.name}")
            if ingredient.density_g_ml is not None and ingredient.density_g_ml <= 0:
                raise ValueError(f"Invalid density for {ingredient.name}")
            if ingredient.approved_formulation_scopes and (
                not ingredient.promotion_artifact_id
                or not ingredient.promotion_registry_sha256
                or not ingredient.approval_expires_at
            ):
                raise ValueError(
                    f"Incomplete signed-promotion provenance for {ingredient.name}"
                )
            if ingredient.approved_formulation_scopes:
                try:
                    expires = datetime.fromisoformat(ingredient.approval_expires_at)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid promotion expiration for {ingredient.name}"
                    ) from error
                if expires.tzinfo is None:
                    raise ValueError(
                        f"Promotion expiration must include timezone for {ingredient.name}"
                    )
                digest = ingredient.promotion_registry_sha256 or ""
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise ValueError(
                        f"Invalid promotion registry hash for {ingredient.name}"
                    )
            for scope in ingredient.approved_formulation_scopes:
                if not isinstance(scope, str) or scope.count("|") != 1:
                    raise ValueError(
                        f"Invalid formulation approval scope for {ingredient.name}"
                    )
                market, category = scope.split("|", 1)
                if not market or not category:
                    raise ValueError(
                        f"Empty formulation approval scope for {ingredient.name}"
                    )
            for alias in ingredient.all_names():
                key = normalize_name(alias)
                owner = aliases.get(key)
                if key and owner is not None and owner != ingredient.ingredient_id:
                    raise ValueError(
                        f"Ingredient alias collision between {owner} and "
                        f"{ingredient.ingredient_id}: {alias}"
                    )
                if key:
                    aliases[key] = ingredient.ingredient_id

    def with_promoted_ingredients(
        self,
        ingredients: Iterable[Ingredient],
        *,
        metadata: dict | None = None,
    ) -> "IngredientCatalog":
        promoted = list(ingredients)
        if not promoted:
            return self
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata or {})
        merged_metadata["signed_promotions_active"] = len(promoted)
        return IngredientCatalog([*self.ingredients, *promoted], merged_metadata)

    def lookup(self, name: str) -> Ingredient | None:
        return self._alias_index.get(normalize_name(name))

    def mentioned_ingredients(self, text: str) -> list[Ingredient]:
        matches: list[Ingredient] = []
        seen: set[str] = set()
        for mention in self.mentioned_ingredient_spans(text):
            ingredient = mention.ingredient
            if ingredient.ingredient_id in seen:
                continue
            matches.append(ingredient)
            seen.add(ingredient.ingredient_id)
        return matches

    def mentioned_ingredient_spans(self, text: str) -> list[IngredientMention]:
        raw: list[IngredientMention] = []
        normalized_text = normalize_name(text)
        for ingredient in self.ingredients:
            for alias in sorted(ingredient.all_names(), key=len, reverse=True):
                normalized_alias = normalize_name(alias)
                if len(normalized_alias) < 2 or normalized_alias not in normalized_text:
                    continue
                for start, end in find_text_spans(text, alias):
                    raw.append(IngredientMention(ingredient, alias, start, end))
        raw.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias))
        selected: list[IngredientMention] = []
        for mention in raw:
            if any(
                mention.start < existing.end and existing.start < mention.end
                for existing in selected
            ):
                continue
            selected.append(mention)
        return selected

    def formulation_candidates(self) -> list[Ingredient]:
        return [item for item in self.ingredients if item.formulation_ready]

    @staticmethod
    def capability_report(
        ingredients: Iterable[Ingredient],
        *,
        minimum_strength: float = 0.35,
    ) -> dict[str, object]:
        """Report the expressive rank of the exact safe candidate pool.

        This is intentionally calculated after price, risk, rarity and supply
        screening.  A large global name list must not make the active formula
        space look more expressive than the materials that can actually be
        selected for the request.
        """

        rows = list(ingredients)
        if not 0.0 < minimum_strength <= 1.0:
            raise ValueError("minimum scent-dimension strength must be in (0, 1]")
        matrix = np.asarray(
            [
                [
                    max(0.0, float(item.profile.get(name, 0.0)))
                    for name in SCENT_DIMENSIONS
                ]
                for item in rows
            ],
            dtype=float,
        )
        rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
        strong_counts = {
            name: sum(
                float(item.profile.get(name, 0.0)) >= minimum_strength for item in rows
            )
            for name in SCENT_DIMENSIONS
        }
        maximum_strength = {
            name: max(
                (float(item.profile.get(name, 0.0)) for item in rows),
                default=0.0,
            )
            for name in SCENT_DIMENSIONS
        }
        return {
            "candidate_count": len(rows),
            "profile_rank": rank,
            "profile_dimension_count": len(SCENT_DIMENSIONS),
            "full_rank": rank == len(SCENT_DIMENSIONS),
            "minimum_strong_material_threshold": minimum_strength,
            "strong_material_counts": strong_counts,
            "maximum_dimension_strength": maximum_strength,
            "unsupported_dimensions": sorted(
                name for name, count in strong_counts.items() if count == 0
            ),
        }

    def stats(self) -> dict[str, int | str]:
        stats: dict[str, int | str] = {
            "embedded_total": len(self.ingredients),
            "formulation_ready": len(self.formulation_candidates()),
            "reference_only_or_blocked": len(self.ingredients)
            - len(self.formulation_candidates()),
            "signed_promotions_active": sum(
                bool(item.approved_formulation_scopes) for item in self.ingredients
            ),
            "ifra_transparency_2025_reference_count": int(
                self.metadata.get("ifra_transparency_2025_reference_count", 3691)
            ),
            "catalog_version": str(self.metadata.get("catalog_version", "unknown")),
        }
        stats.update(
            {
                key: value
                for key, value in self.metadata.items()
                if key.startswith("industrial_registry_")
                and isinstance(value, (int, str))
                and not isinstance(value, bool)
            }
        )
        return stats

    @staticmethod
    def read_ifra_transparency_csv(path: str | Path) -> list[dict[str, str]]:
        """Read an official/user-provided IFRA export as reference-only records.

        The transparency list contains names/CAS identifiers, not enough data for
        formulation. Records returned here are deliberately not promoted into the
        formulation-ready catalog.
        """
        records: list[dict[str, str]] = []
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                lowered = {
                    str(key).casefold(): str(value).strip()
                    for key, value in row.items()
                }
                name = lowered.get("principal name") or lowered.get("name") or ""
                cas = (
                    lowered.get("cas n°")
                    or lowered.get("cas number")
                    or lowered.get("cas")
                    or ""
                )
                if name:
                    records.append(
                        {"name": name, "cas_number": cas, "formulation_ready": "false"}
                    )
        return records


class HistoricalReferenceCorpus:
    """Uses the workspace's 10k-fragrance corpus only as a compatibility signal."""

    def __init__(
        self,
        path: str | Path | None = None,
        data_hub: NonHumanDataHub | None = None,
    ):
        self.path = (
            Path(path)
            if path
            else (
                Path(__file__).resolve().parent.parent
                / "data"
                / "reference_fragrances.db"
            )
        )
        self.perfume_sets: dict[str, set[int]] = defaultdict(set)
        self.perfume_notes_by_id: dict[int, set[str]] = defaultdict(set)
        self.perfume_metadata: dict[int, dict[str, str | int | None]] = {}
        self.total_perfumes = 0
        self.total_note_rows = 0
        self.base_reference_perfumes = 0
        self.base_reference_note_rows = 0
        self.sha256 = ""
        self.provenance_status = "unverified_workspace_reference"
        self.molecular_composition_status = "not_available_note_names_only"
        self.molecular_composition_claim_boundary = (
            "The historical corpus and data-hub references contain note names, not verified "
            "mass fractions or a GC-MS molecular composition. They cannot be used as an exact "
            "target formula or as olfactory-equivalence evidence."
        )
        self.hub_reference_formulas = 0
        self.hub_reference_note_rows = 0
        if self.path.exists():
            self._load()
            self.base_reference_perfumes = self.total_perfumes
            self.base_reference_note_rows = self.total_note_rows
        if data_hub is not None:
            self._load_hub_references(data_hub)

    def _load(self) -> None:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        uri = self.path.resolve().as_uri() + "?mode=ro"
        # sqlite3.Connection.__exit__ only commits/rolls back; it does not
        # close the handle. ``closing`` prevents descriptor accumulation.
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            self.total_perfumes = int(
                connection.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0]
            )
            metadata_rows = connection.execute(
                "SELECT id, brand, name, year, gender, style FROM perfumes"
            ).fetchall()
            for perfume_id, brand, name, year, gender, style in metadata_rows:
                self.perfume_metadata[int(perfume_id)] = {
                    "perfume_id": int(perfume_id),
                    "brand": str(brand),
                    "name": str(name),
                    "year": int(year) if year is not None else None,
                    "gender": str(gender) if gender is not None else None,
                    "style": str(style) if style is not None else None,
                }
            rows = connection.execute(
                "SELECT perfume_id, ingredient_name FROM perfume_notes"
            ).fetchall()
        self.total_note_rows = len(rows)
        for perfume_id, ingredient_name in rows:
            normalized = normalize_name(str(ingredient_name))
            identifier = int(perfume_id)
            self.perfume_sets[normalized].add(identifier)
            self.perfume_notes_by_id[identifier].add(normalized)

    def _load_hub_references(self, data_hub: NonHumanDataHub) -> None:
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for (
            source_id,
            formula_ref,
            material_name,
            formula_name,
        ) in data_hub.additional_reference_formulas():
            item = grouped.setdefault(
                (source_id, formula_ref), {"name": formula_name, "notes": set()}
            )
            notes = item["notes"]
            assert isinstance(notes, set)
            note = normalize_name(material_name)
            if note:
                notes.add(note)
        next_identifier = -1
        for (source_id, formula_ref), item in sorted(grouped.items()):
            notes = item["notes"]
            assert isinstance(notes, set)
            if not notes:
                continue
            while next_identifier in self.perfume_notes_by_id:
                next_identifier -= 1
            self.perfume_notes_by_id[next_identifier] = set(notes)
            self.perfume_metadata[next_identifier] = {
                "perfume_id": next_identifier,
                "brand": "nonhuman_data_hub",
                "name": str(item["name"]),
                "year": None,
                "gender": None,
                "style": f"reference_only:{source_id}:{formula_ref}",
            }
            for note in notes:
                self.perfume_sets[note].add(next_identifier)
            self.hub_reference_formulas += 1
            self.hub_reference_note_rows += len(notes)
            next_identifier -= 1
        self.total_perfumes += self.hub_reference_formulas
        self.total_note_rows += self.hub_reference_note_rows

    def frequency(self, ingredient_name: str) -> float:
        if self.total_perfumes <= 0:
            return 0.0
        return (
            len(self.perfume_sets.get(normalize_name(ingredient_name), set()))
            / self.total_perfumes
        )

    def pair_support(self, first: str, second: str) -> float:
        left = self.perfume_sets.get(normalize_name(first), set())
        right = self.perfume_sets.get(normalize_name(second), set())
        if not left or not right:
            return 0.5
        return len(left & right) / max(1, min(len(left), len(right)))

    def recipe_support(self, names: list[str]) -> float:
        scores = [
            self.pair_support(names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        return sum(scores) / len(scores) if scores else 0.5

    def nearest_references(self, names: Iterable[str], limit: int = 5) -> list[dict]:
        """Return historical products sharing note names, never a sensory claim."""
        query = {normalize_name(name) for name in names if normalize_name(name)}
        if not query:
            return []
        ranked: list[tuple[float, int]] = []
        for perfume_id, note_names in self.perfume_notes_by_id.items():
            overlap = len(query & note_names)
            if not overlap:
                continue
            union = len(query | note_names)
            ranked.append((overlap / max(1, union), perfume_id))
        ranked.sort(reverse=True)
        output: list[dict] = []
        for score, perfume_id in ranked[:limit]:
            metadata = dict(
                self.perfume_metadata.get(perfume_id, {"perfume_id": perfume_id})
            )
            metadata["note_overlap_score"] = round(score * 100.0, 4)
            metadata["reference_only"] = True
            output.append(metadata)
        return output

    def stats(self) -> dict[str, int | str]:
        return {
            "reference_perfumes": self.base_reference_perfumes,
            "reference_note_rows": self.base_reference_note_rows,
            "combined_reference_formulas": self.total_perfumes,
            "combined_reference_note_rows": self.total_note_rows,
            "unique_reference_material_names": len(self.perfume_sets),
            "reference_corpus_sha256": self.sha256,
            "reference_corpus_provenance": self.provenance_status,
            "hub_reference_formulas_used": self.hub_reference_formulas,
            "hub_reference_note_rows_used": self.hub_reference_note_rows,
            "reference_molecular_composition_status": self.molecular_composition_status,
        }
