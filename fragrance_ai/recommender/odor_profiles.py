"""Observed ingredient odor profiles and catalog overlays.

The optimizer can use these profiles when operators import real panel or
instrument-linked observations. Without observations it keeps the curated
engineering profile and explicitly reports the coverage gap.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any

from .catalog import IngredientCatalog
from .models import Ingredient, SCENT_DIMENSIONS, normalize_profile
from .sqlite_lifecycle import SQLiteConnectionOwner


@dataclass(frozen=True)
class OdorProfileOverlay:
    ingredient_id: str
    profile: dict[str, float]
    observation_count: int
    panelist_count: int
    median_dilution_percent: float
    confidence: float


class OdorProfileStore(SQLiteConnectionOwner):
    """SQLite store for panel-derived ingredient odor descriptors."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS odor_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_id TEXT NOT NULL,
                panelist_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                dilution_percent REAL NOT NULL,
                descriptor_json TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                observed_on TEXT NOT NULL,
                perceived_intensity REAL,
                replicate_id TEXT NOT NULL DEFAULT 'primary'
            )
            """
        )
        columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(odor_observations)")
        }
        if "perceived_intensity" not in columns:
            self.connection.execute(
                "ALTER TABLE odor_observations ADD COLUMN perceived_intensity REAL"
            )
        if "replicate_id" not in columns:
            self.connection.execute(
                "ALTER TABLE odor_observations ADD COLUMN replicate_id TEXT NOT NULL DEFAULT 'primary'"
            )
        self.connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_odor_observation
            ON odor_observations
            (ingredient_id, panelist_id, batch_id, dilution_percent, observed_on, replicate_id)"""
        )
        self.connection.commit()

    @staticmethod
    def _descriptors(values: dict[str, Any]) -> dict[str, float]:
        unknown = set(values) - set(SCENT_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown odor dimensions: {sorted(unknown)}")
        cleaned = {key: float(values.get(key, 0.0)) for key in SCENT_DIMENSIONS}
        if any(value < 0 or value > 1 for value in cleaned.values()):
            raise ValueError("odor descriptor scores must be between 0 and 1")
        if sum(cleaned.values()) <= 0:
            raise ValueError("at least one odor descriptor must be positive")
        return normalize_profile(cleaned)

    def record(
        self,
        ingredient_id: str,
        panelist_id: str,
        batch_id: str,
        dilution_percent: float,
        descriptor_scores: dict[str, Any],
        source_ref: str,
        observed_on: date | None = None,
        perceived_intensity: float | None = None,
        replicate_id: str = "primary",
    ) -> None:
        if not ingredient_id.strip() or not panelist_id.strip() or not batch_id.strip():
            raise ValueError("ingredient_id, panelist_id, and batch_id are required")
        if dilution_percent <= 0 or dilution_percent > 100:
            raise ValueError("dilution_percent must be in (0, 100]")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if perceived_intensity is not None and not 0 <= perceived_intensity <= 100:
            raise ValueError("perceived_intensity must be between 0 and 100")
        if not replicate_id.strip():
            raise ValueError("replicate_id is required")
        descriptors = self._descriptors(descriptor_scores)
        self.connection.execute(
            """
            INSERT INTO odor_observations
            (ingredient_id, panelist_id, batch_id, dilution_percent,
             descriptor_json, source_ref, observed_on, perceived_intensity, replicate_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ingredient_id,
                panelist_id,
                batch_id,
                float(dilution_percent),
                json.dumps(descriptors, sort_keys=True),
                source_ref,
                (observed_on or date.today()).isoformat(),
                perceived_intensity,
                replicate_id,
            ),
        )
        self.connection.commit()

    def import_csv(self, path: str | Path) -> int:
        source = Path(path)
        count = 0
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                descriptor_json = row.get("descriptor_json", "")
                if not descriptor_json:
                    raise ValueError("descriptor_json is required in every odor row")
                self.record(
                    row.get("ingredient_id", ""),
                    row.get("panelist_id", ""),
                    row.get("batch_id", ""),
                    float(row.get("dilution_percent", 0)),
                    json.loads(descriptor_json),
                    row.get("source_ref", str(source)),
                    date.fromisoformat(row["observed_on"]) if row.get("observed_on") else None,
                    float(row["perceived_intensity"])
                    if row.get("perceived_intensity") not in (None, "")
                    else None,
                    row.get("replicate_id", "primary"),
                )
                count += 1
        return count

    def overlays(
        self,
        min_observations: int = 3,
        min_panelists: int = 3,
    ) -> dict[str, OdorProfileOverlay]:
        rows = self.connection.execute(
            """
            SELECT ingredient_id, panelist_id, dilution_percent, descriptor_json
            FROM odor_observations ORDER BY ingredient_id, observation_id
            """
        ).fetchall()
        grouped: dict[str, list[tuple[str, float, dict[str, float]]]] = {}
        for ingredient_id, panelist, dilution, descriptor_json in rows:
            grouped.setdefault(str(ingredient_id), []).append(
                (str(panelist), float(dilution), json.loads(descriptor_json))
            )
        output: dict[str, OdorProfileOverlay] = {}
        for ingredient_id, observations in grouped.items():
            dilution_counts = Counter(round(row[1], 2) for row in observations)
            reference_dilution, _ = dilution_counts.most_common(1)[0]
            observations = [
                row for row in observations if abs(row[1] - reference_dilution) <= 0.01
            ]
            panelists = {row[0] for row in observations}
            if len(observations) < min_observations or len(panelists) < min_panelists:
                continue
            panel_profiles: dict[str, dict[str, float]] = {}
            for panelist in panelists:
                rows_for_panel = [row for row in observations if row[0] == panelist]
                panel_profiles[panelist] = {
                    dimension: mean(row[2].get(dimension, 0.0) for row in rows_for_panel)
                    for dimension in SCENT_DIMENSIONS
                }
            profile = {
                dimension: median(values[dimension] for values in panel_profiles.values())
                for dimension in SCENT_DIMENSIONS
            }
            confidence = min(1.0, len(panelists) / 12.0)
            output[ingredient_id] = OdorProfileOverlay(
                ingredient_id=ingredient_id,
                profile=normalize_profile(profile),
                observation_count=len(observations),
                panelist_count=len(panelists),
                median_dilution_percent=sorted(row[1] for row in observations)[len(observations) // 2],
                confidence=round(confidence, 4),
            )
        return output

    def apply_to_catalog(
        self,
        catalog: IngredientCatalog,
        min_observations: int = 3,
        min_panelists: int = 3,
    ) -> IngredientCatalog:
        overlays = self.overlays(min_observations, min_panelists)
        updated: list[Ingredient] = []
        for ingredient in catalog.ingredients:
            overlay = overlays.get(ingredient.ingredient_id)
            if overlay is None:
                updated.append(ingredient)
                continue
            updated.append(
                replace(
                    ingredient,
                    profile=overlay.profile,
                    data_source=(
                        f"odor-observed:{overlay.panelist_count}p/{overlay.observation_count}n"
                    ),
                    data_verified_on=date.today().isoformat(),
                )
            )
        metadata = dict(catalog.metadata)
        metadata["odor_profile_observed_ingredients"] = len(overlays)
        metadata["odor_profile_min_observations"] = min_observations
        metadata["odor_profile_min_panelists"] = min_panelists
        return IngredientCatalog(updated, metadata)

    def stats(self, catalog: IngredientCatalog | None = None) -> dict[str, int | float]:
        observations = int(
            self.connection.execute("SELECT COUNT(*) FROM odor_observations").fetchone()[0]
        )
        ingredients = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT ingredient_id) FROM odor_observations"
            ).fetchone()[0]
        )
        if catalog:
            known_ids = {item.ingredient_id for item in catalog.ingredients}
            ingredients = len(
                known_ids
                & {
                    str(row[0])
                    for row in self.connection.execute(
                        "SELECT DISTINCT ingredient_id FROM odor_observations"
                    ).fetchall()
                }
            )
        catalog_total = len(catalog.ingredients) if catalog else 0
        return {
            "odor_observation_count": observations,
            "odor_observed_ingredients": ingredients,
            "odor_profile_coverage_percent": round(
                ingredients / max(1, catalog_total) * 100.0, 4
            ),
        }

    def close(self) -> None:
        self.connection.close()
