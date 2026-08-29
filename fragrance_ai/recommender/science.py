"""Physicochemical and temporal digital twin for fragrance formulas.

The model separates measured molecular properties from inferred fallbacks.  It
uses concentration-response saturation, first-order evaporation, and bounded
mixture suppression to estimate how the scent profile changes over time.  The
result is an engineering digital twin, not a replacement for human panels.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np

from .models import Ingredient, RecipeLine, ScentBrief, SCENT_DIMENSIONS, profile_vector
from .optimizer import semantic_brief_similarity
from .sqlite_lifecycle import SQLiteConnectionOwner


SCIENTIFIC_MODEL_VERSION = "headspace-olfactory-twin-2.1"
TIMEPOINTS_MINUTES = (0, 15, 60, 240, 480)
TIMEPOINT_WEIGHTS = np.asarray((0.25, 0.25, 0.20, 0.18, 0.12), dtype=float)
ATMOSPHERIC_PRESSURE_PA = 101_325.0
ETHANOL_MOLECULAR_WEIGHT = 46.06844
DEFAULT_MONTE_CARLO_DRAWS = 256


@dataclass(frozen=True)
class MolecularProperties:
    ingredient_id: str
    cas_number: str | None
    molecular_weight: float
    xlogp: float | None
    tpsa: float | None
    hbond_donors: int | None
    hbond_acceptors: int | None
    rotatable_bonds: int | None
    complexity: float | None
    vapor_pressure_pa_25c: float | None
    boiling_point_c: float | None
    odor_threshold_ppm: float | None
    source_ref: str
    verified_on: str


@dataclass(frozen=True)
class TemporalPoint:
    minutes: int
    similarity: float
    dominant_dimensions: tuple[str, ...]
    total_relative_intensity: float
    similarity_p05: float = 0.0
    similarity_p95: float = 0.0


@dataclass(frozen=True)
class ScientificTwinResult:
    status: str
    model_version: str
    scientific_data_coverage_percent: float
    molecular_descriptor_coverage_percent: float
    temporal_similarity_mean: float
    minimum_temporal_similarity: float
    temporal_points: tuple[TemporalPoint, ...]
    confidence: str
    flags: tuple[str, ...]
    vapor_pressure_coverage_percent: float = 0.0
    odor_threshold_coverage_percent: float = 0.0
    model_applicability_percent: float = 0.0
    temporal_similarity_p05: float = 0.0
    temporal_similarity_p95: float = 0.0
    minimum_temporal_similarity_p05: float = 0.0
    simulation_only_approved: bool = False
    monte_carlo_draws: int = 0
    model_domain_passed: bool = False
    uncertainty_kind: str = "prior_propagation_not_calibrated_prediction_error"


@dataclass(frozen=True)
class _PreparedMaterial:
    line: RecipeLine
    ingredient: Ingredient
    properties: MolecularProperties | None
    mole_fraction: float
    vapor_pressure_pa: float
    vapor_log_sigma: float
    odor_threshold_ppm: float
    threshold_log_sigma: float
    activity_coefficient: float


class ScientificPropertyStore(SQLiteConnectionOwner):
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS molecular_properties (
                ingredient_id TEXT PRIMARY KEY,
                cas_number TEXT,
                molecular_weight REAL NOT NULL,
                xlogp REAL,
                tpsa REAL,
                hbond_donors INTEGER,
                hbond_acceptors INTEGER,
                rotatable_bonds INTEGER,
                complexity REAL,
                vapor_pressure_pa_25c REAL,
                boiling_point_c REAL,
                odor_threshold_ppm REAL,
                source_ref TEXT NOT NULL,
                verified_on TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @classmethod
    def load_builtin(cls) -> "ScientificPropertyStore":
        path = (
            Path(__file__).resolve().parent.parent / "data" / "scientific_properties.db"
        )
        return cls(path if path.exists() else ":memory:")

    @staticmethod
    def _float(value: str | float | None) -> float | None:
        return None if value in (None, "") else float(value)

    @staticmethod
    def _int(value: str | int | None) -> int | None:
        return None if value in (None, "") else int(value)

    @staticmethod
    def _validate_properties(properties: MolecularProperties) -> None:
        if not isinstance(properties.ingredient_id, str) or not properties.ingredient_id:
            raise ValueError("ingredient_id is required")
        try:
            molecular_weight = float(properties.molecular_weight)
        except (TypeError, ValueError) as error:
            raise ValueError("molecular_weight must be numeric") from error
        if isinstance(properties.molecular_weight, bool) or not (
            math.isfinite(molecular_weight) and molecular_weight > 0.0
        ):
            raise ValueError("molecular_weight must be finite and positive")
        optional_values: dict[str, float] = {}
        for name in (
            "xlogp",
            "tpsa",
            "complexity",
            "vapor_pressure_pa_25c",
            "boiling_point_c",
            "odor_threshold_ppm",
        ):
            value = getattr(properties, name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric when provided") from error
            if isinstance(value, bool) or not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite when provided")
            optional_values[name] = numeric
        if optional_values.get("tpsa", 0.0) < 0:
            raise ValueError("tpsa cannot be negative")
        if optional_values.get("complexity", 0.0) < 0:
            raise ValueError("complexity cannot be negative")
        for name in ("hbond_donors", "hbond_acceptors", "rotatable_bonds"):
            value = getattr(properties, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if optional_values.get("vapor_pressure_pa_25c", 0.0) < 0:
            raise ValueError("vapor pressure cannot be negative")
        if (
            properties.odor_threshold_ppm is not None
            and optional_values["odor_threshold_ppm"] <= 0
        ):
            raise ValueError("odor threshold must be positive")
        if (
            not isinstance(properties.source_ref, str)
            or not properties.source_ref
            or not isinstance(properties.verified_on, str)
            or not properties.verified_on
        ):
            raise ValueError("source_ref and verified_on are required")

    def upsert(self, properties: MolecularProperties) -> None:
        self._validate_properties(properties)
        self.connection.execute(
            """INSERT OR REPLACE INTO molecular_properties VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                properties.ingredient_id,
                properties.cas_number,
                properties.molecular_weight,
                properties.xlogp,
                properties.tpsa,
                properties.hbond_donors,
                properties.hbond_acceptors,
                properties.rotatable_bonds,
                properties.complexity,
                properties.vapor_pressure_pa_25c,
                properties.boiling_point_c,
                properties.odor_threshold_ppm,
                properties.source_ref,
                properties.verified_on,
            ),
        )
        self.connection.commit()

    def import_csv(
        self,
        path: str | Path,
        allowed_ingredient_ids: Collection[str] | None = None,
    ) -> int:
        rows: list[MolecularProperties] = []
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    MolecularProperties(
                        ingredient_id=row["ingredient_id"],
                        cas_number=row.get("cas_number") or None,
                        molecular_weight=float(row["molecular_weight"]),
                        xlogp=self._float(row.get("xlogp")),
                        tpsa=self._float(row.get("tpsa")),
                        hbond_donors=self._int(row.get("hbond_donors")),
                        hbond_acceptors=self._int(row.get("hbond_acceptors")),
                        rotatable_bonds=self._int(row.get("rotatable_bonds")),
                        complexity=self._float(row.get("complexity")),
                        vapor_pressure_pa_25c=self._float(
                            row.get("vapor_pressure_pa_25c")
                        ),
                        boiling_point_c=self._float(row.get("boiling_point_c")),
                        odor_threshold_ppm=self._float(row.get("odor_threshold_ppm")),
                        source_ref=row["source_ref"],
                        verified_on=row["verified_on"],
                    )
                )
        allowed = (
            set(allowed_ingredient_ids) if allowed_ingredient_ids is not None else None
        )
        for item in rows:
            self._validate_properties(item)
            if allowed is not None and item.ingredient_id not in allowed:
                raise ValueError(f"unknown ingredient id: {item.ingredient_id}")
        with self.connection:
            for item in rows:
                self.connection.execute(
                    """INSERT OR REPLACE INTO molecular_properties VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(item.__dict__.values()),
                )
        return len(rows)

    def get(self, ingredient_id: str) -> MolecularProperties | None:
        row = self.connection.execute(
            "SELECT * FROM molecular_properties WHERE ingredient_id = ?",
            (ingredient_id,),
        ).fetchone()
        return MolecularProperties(*row) if row else None

    def stats(self) -> dict[str, int]:
        return {
            "scientific_property_records": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM molecular_properties"
                ).fetchone()[0]
            )
        }

    def close(self) -> None:
        self.connection.close()


class TemporalMixtureSimulator:
    """Physics-informed headspace and olfactory-mixture digital twin.

    The model uses an ideal hydroalcoholic matrix as its explicit reference
    system.  It converts formula mass fractions to approximate liquid mole
    fractions, applies Raoult-like headspace partitioning, divides gas-phase
    concentration by the odor threshold (odor activity value), then applies a
    saturating response and competitive mixture suppression.  Missing physical
    values are sampled from deliberately wide priors and reduce applicability.
    """

    VAPOR_PRESSURE_PRIOR_PA = {"top": 20.0, "heart": 0.8, "base": 0.03}
    THRESHOLD_PRIOR_PPM = {"top": 0.5, "heart": 0.05, "base": 0.005}
    PYRAMID_PERSISTENCE = {"top": 0.55, "heart": 1.0, "base": 2.2}

    @staticmethod
    def _air_to_receptor_transport(props: MolecularProperties | None) -> float:
        """Bounded mucosa/receptor-access prior from molecular descriptors."""
        if props is None:
            return 0.8
        mass_term = math.exp(-max(0.0, props.molecular_weight - 150.0) / 600.0)
        polar_term = (
            math.exp(-max(0.0, props.tpsa) / 350.0) if props.tpsa is not None else 1.0
        )
        lipophilic_term = (
            math.exp(-max(0.0, abs(props.xlogp - 2.0) - 2.0) * 0.08)
            if props.xlogp is not None
            else 1.0
        )
        return max(0.30, min(1.0, mass_term * polar_term * lipophilic_term))

    @staticmethod
    def _boiling_point_to_vapor_pressure(boiling_point_c: float) -> float:
        """Estimate vapor pressure at 25 C with Trouton's-rule enthalpy.

        The estimate is only a fallback for a measured normal boiling point and
        receives substantially wider uncertainty than a measured vapor pressure.
        """
        boiling_k = boiling_point_c + 273.15
        ambient_k = 298.15
        if boiling_k <= 0:
            return 1e-8
        if boiling_k <= ambient_k:
            return ATMOSPHERIC_PRESSURE_PA
        enthalpy_j_mol = 88.0 * boiling_k
        log_pressure = math.log(ATMOSPHERIC_PRESSURE_PA) - (
            enthalpy_j_mol / 8.314462618
        ) * (1.0 / ambient_k - 1.0 / boiling_k)
        return max(1e-8, min(ATMOSPHERIC_PRESSURE_PA, math.exp(log_pressure)))

    @classmethod
    def _vapor_pressure_prior(
        cls, ingredient: Ingredient, props: MolecularProperties | None
    ) -> tuple[float, float]:
        if props and props.vapor_pressure_pa_25c is not None:
            return max(1e-8, props.vapor_pressure_pa_25c), 0.20
        if props and props.boiling_point_c is not None:
            return cls._boiling_point_to_vapor_pressure(props.boiling_point_c), 0.55
        return cls.VAPOR_PRESSURE_PRIOR_PA[ingredient.pyramid], 1.25

    @classmethod
    def _threshold_prior(
        cls, ingredient: Ingredient, props: MolecularProperties | None
    ) -> tuple[float, float]:
        if props and props.odor_threshold_ppm is not None:
            return max(1e-9, props.odor_threshold_ppm), 0.35
        impact = max(0.05, ingredient.odor_impact)
        return max(1e-6, cls.THRESHOLD_PRIOR_PPM[ingredient.pyramid] / impact), 1.60

    @classmethod
    def _half_life_minutes(
        cls,
        ingredient: Ingredient,
        props: MolecularProperties | None,
        vapor_pressure_pa: float | None = None,
    ) -> float:
        pressure = vapor_pressure_pa
        if pressure is None:
            pressure = cls._vapor_pressure_prior(ingredient, props)[0]
        molecular_weight = props.molecular_weight if props else 180.0
        base = 360.0 * math.sqrt(max(1.0, molecular_weight) / 160.0)
        volatility = 1.0 + 0.8 * max(1e-8, pressure) ** 0.35
        half_life = base * cls.PYRAMID_PERSISTENCE[ingredient.pyramid] / volatility
        return max(5.0, min(1440.0, half_life))

    @classmethod
    def _initial_activation(
        cls, line: RecipeLine, ingredient: Ingredient, props: MolecularProperties | None
    ) -> float:
        """Compatibility helper using the same OAV saturation as the full twin."""
        vapor_pressure, _ = cls._vapor_pressure_prior(ingredient, props)
        threshold, _ = cls._threshold_prior(ingredient, props)
        molecular_weight = props.molecular_weight if props else 180.0
        active_mass = (
            line.finished_product_percent * line.active_strength_percent / 100.0
        )
        odorant_moles = max(1e-12, active_mass / molecular_weight)
        base_moles = (
            max(0.0, 100.0 - line.finished_product_percent) / ETHANOL_MOLECULAR_WEIGHT
        )
        liquid_fraction = odorant_moles / max(1e-12, odorant_moles + base_moles)
        gas_ppm = (
            liquid_fraction * vapor_pressure / ATMOSPHERIC_PRESSURE_PA * 1_000_000.0
        )
        oav = gas_ppm * cls._air_to_receptor_transport(props) / threshold
        powered = max(1e-12, oav) ** 0.55
        return powered / (1.0 + powered)

    @staticmethod
    def _interaction_matrix(prepared: list[_PreparedMaterial]) -> np.ndarray:
        profiles = np.asarray(
            [item.ingredient.vector() for item in prepared], dtype=float
        )
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        normalized = profiles / np.maximum(norms, 1e-12)
        profile_overlap = np.clip(normalized @ normalized.T, 0.0, 1.0)
        chemical_overlap = np.zeros_like(profile_overlap)
        for left, left_item in enumerate(prepared):
            for right, right_item in enumerate(prepared):
                if left == right:
                    continue
                left_props = left_item.properties
                right_props = right_item.properties
                if left_props and right_props:
                    mass_similarity = math.exp(
                        -abs(left_props.molecular_weight - right_props.molecular_weight)
                        / 300.0
                    )
                    if left_props.xlogp is not None and right_props.xlogp is not None:
                        polarity_similarity = math.exp(
                            -abs(left_props.xlogp - right_props.xlogp) / 3.0
                        )
                    else:
                        polarity_similarity = 0.5
                    chemical_overlap[left, right] = (
                        mass_similarity * polarity_similarity
                    )
                else:
                    chemical_overlap[left, right] = 0.5
        interaction = 0.70 * profile_overlap + 0.30 * chemical_overlap
        np.fill_diagonal(interaction, 0.0)
        return np.clip(interaction, 0.0, 1.0)

    def ingredient_perceptual_factors(
        self,
        ingredients: Collection[Ingredient],
        store: ScientificPropertyStore,
        product_concentration_percent: float,
        timepoint_weights: Collection[float] | None = None,
    ) -> dict[str, float]:
        """Return fast headspace/persistence gains for formula optimization.

        Each material is screened at 1% of fragrance concentrate in the same
        hydroalcoholic reference matrix.  The factors are median-normalized and
        bounded; the full nonlinear Monte Carlo twin remains the final judge.
        """
        concentration = float(product_concentration_percent)
        if not math.isfinite(concentration) or not 0.0 < concentration <= 100.0:
            raise ValueError("product_concentration_percent must be in (0, 100]")
        if timepoint_weights is None:
            temporal_weights = TIMEPOINT_WEIGHTS
        else:
            temporal_weights = np.asarray(tuple(timepoint_weights), dtype=float)
            if temporal_weights.shape != TIMEPOINT_WEIGHTS.shape:
                raise ValueError(
                    f"timepoint_weights must contain {len(TIMEPOINT_WEIGHTS)} values"
                )
            if (
                not np.all(np.isfinite(temporal_weights))
                or np.any(temporal_weights < 0)
                or float(temporal_weights.sum()) <= 0
            ):
                raise ValueError(
                    "timepoint_weights must be nonnegative with positive sum"
                )
            temporal_weights = temporal_weights / temporal_weights.sum()

        raw: dict[str, float] = {}
        finished_percent = concentration / 100.0
        for ingredient in ingredients:
            props = store.get(ingredient.ingredient_id)
            molecular_weight = props.molecular_weight if props else 180.0
            active_mass = finished_percent * ingredient.active_strength_percent / 100.0
            odorant_moles = active_mass / max(1e-9, molecular_weight)
            base_moles = max(0.0, 100.0 - finished_percent) / ETHANOL_MOLECULAR_WEIGHT
            liquid_fraction = odorant_moles / max(1e-12, odorant_moles + base_moles)
            vapor_pressure, _ = self._vapor_pressure_prior(ingredient, props)
            threshold, _ = self._threshold_prior(ingredient, props)
            xlogp = props.xlogp if props and props.xlogp is not None else 2.0
            activity_coefficient = max(0.50, min(3.0, math.exp(0.18 * (xlogp - 2.0))))
            gas_ppm = (
                liquid_fraction
                * activity_coefficient
                * vapor_pressure
                / ATMOSPHERIC_PRESSURE_PA
                * 1_000_000.0
            )
            odor_activity = max(1e-12, gas_ppm / threshold)
            activation = odor_activity**0.55 / (1.0 + odor_activity**0.55)
            activation *= self._air_to_receptor_transport(props)
            half_life = self._half_life_minutes(ingredient, props, vapor_pressure)
            persistence = float(
                np.sum(
                    temporal_weights
                    * np.power(
                        0.5, np.asarray(TIMEPOINTS_MINUTES, dtype=float) / half_life
                    )
                )
            )
            raw[ingredient.ingredient_id] = max(1e-9, activation * persistence)
        if not raw:
            return {}
        median = float(np.median(list(raw.values())))
        median = max(1e-9, median)
        return {
            ingredient_id: max(0.15, min(8.0, value / median))
            for ingredient_id, value in raw.items()
        }

    def _prepare(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        store: ScientificPropertyStore,
    ) -> list[_PreparedMaterial]:
        raw: list[tuple[RecipeLine, Ingredient, MolecularProperties | None, float]] = []
        total_finished = sum(max(0.0, line.finished_product_percent) for line in lines)
        base_moles = max(0.0, 100.0 - total_finished) / ETHANOL_MOLECULAR_WEIGHT
        odorant_moles = 0.0
        for line in lines:
            ingredient = ingredients[line.ingredient_id]
            props = store.get(line.ingredient_id)
            molecular_weight = props.molecular_weight if props else 180.0
            active_mass = (
                max(0.0, line.finished_product_percent)
                * max(0.0, line.active_strength_percent)
                / 100.0
            )
            moles = active_mass / max(1e-9, molecular_weight)
            raw.append((line, ingredient, props, moles))
            odorant_moles += moles
        total_moles = max(1e-12, base_moles + odorant_moles)
        prepared: list[_PreparedMaterial] = []
        for line, ingredient, props, moles in raw:
            vapor_pressure, vapor_sigma = self._vapor_pressure_prior(ingredient, props)
            threshold, threshold_sigma = self._threshold_prior(ingredient, props)
            xlogp = props.xlogp if props and props.xlogp is not None else 2.0
            activity = max(0.50, min(3.0, math.exp(0.18 * (xlogp - 2.0))))
            prepared.append(
                _PreparedMaterial(
                    line=line,
                    ingredient=ingredient,
                    properties=props,
                    mole_fraction=moles / total_moles,
                    vapor_pressure_pa=vapor_pressure,
                    vapor_log_sigma=vapor_sigma,
                    odor_threshold_ppm=threshold,
                    threshold_log_sigma=threshold_sigma,
                    activity_coefficient=activity,
                )
            )
        return prepared

    def evaluate(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        brief: ScentBrief,
        store: ScientificPropertyStore,
        draws: int = DEFAULT_MONTE_CARLO_DRAWS,
        seed: int | None = None,
    ) -> ScientificTwinResult:
        if not lines:
            return ScientificTwinResult(
                "insufficient_data",
                SCIENTIFIC_MODEL_VERSION,
                0.0,
                0.0,
                0.0,
                0.0,
                (),
                "none",
                ("empty_formula",),
                monte_carlo_draws=0,
            )
        if not 64 <= draws <= 100_000:
            raise ValueError(
                "scientific Monte Carlo draws must be between 64 and 100000"
            )
        target = profile_vector(brief.target_profile)
        prepared = self._prepare(lines, ingredients, store)
        if seed is None:
            canonical = "|".join(
                f"{item.line.ingredient_id}:{item.line.finished_product_percent:.8f}"
                for item in sorted(
                    prepared, key=lambda current: current.line.ingredient_id
                )
            )
            seed = int(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)

        importance = np.asarray(
            [
                max(1e-9, item.line.concentrate_percent * item.ingredient.odor_impact)
                for item in prepared
            ],
            dtype=float,
        )
        importance /= importance.sum()
        has_molecule = np.asarray(
            [item.properties is not None for item in prepared], dtype=float
        )
        has_vapor = np.asarray(
            [
                bool(
                    item.properties
                    and (
                        item.properties.vapor_pressure_pa_25c is not None
                        or item.properties.boiling_point_c is not None
                    )
                )
                for item in prepared
            ],
            dtype=float,
        )
        has_threshold = np.asarray(
            [
                bool(item.properties and item.properties.odor_threshold_ppm is not None)
                for item in prepared
            ],
            dtype=float,
        )
        has_complete = has_vapor * has_threshold
        molecular_coverage = float(importance @ has_molecule) * 100.0
        vapor_coverage = float(importance @ has_vapor) * 100.0
        threshold_coverage = float(importance @ has_threshold) * 100.0
        complete_coverage = float(importance @ has_complete) * 100.0
        # Applicability is now direct-evidence coverage only. Priors still let
        # the diagnostic run, but no longer manufacture domain confidence.
        applicability = (
            molecular_coverage * 0.40
            + vapor_coverage * 0.30
            + threshold_coverage * 0.30
        )

        interaction = self._interaction_matrix(prepared)
        profiles = np.asarray(
            [item.ingredient.vector() for item in prepared], dtype=float
        )
        time_similarities = np.zeros((draws, len(TIMEPOINTS_MINUTES)), dtype=float)
        time_intensities = np.zeros_like(time_similarities)
        time_mixtures = np.zeros(
            (draws, len(TIMEPOINTS_MINUTES), len(SCENT_DIMENSIONS))
        )

        for draw_index in range(draws):
            sampled_vapor = np.asarray(
                [
                    item.vapor_pressure_pa
                    * math.exp(rng.normal(0.0, item.vapor_log_sigma))
                    for item in prepared
                ],
                dtype=float,
            )
            sampled_threshold = np.asarray(
                [
                    item.odor_threshold_ppm
                    * math.exp(rng.normal(0.0, item.threshold_log_sigma))
                    for item in prepared
                ],
                dtype=float,
            )
            sampled_activity = np.asarray(
                [
                    item.activity_coefficient
                    * math.exp(rng.normal(0.0, 0.18 if item.properties else 0.50))
                    for item in prepared
                ],
                dtype=float,
            )
            gas_ppm = (
                np.asarray([item.mole_fraction for item in prepared]) * sampled_activity
            )
            gas_ppm *= sampled_vapor / ATMOSPHERIC_PRESSURE_PA * 1_000_000.0
            odor_activity = np.maximum(
                1e-12, gas_ppm / np.maximum(sampled_threshold, 1e-12)
            )
            activation = odor_activity**0.55
            activation /= 1.0 + activation
            activation *= np.asarray(
                [self._air_to_receptor_transport(item.properties) for item in prepared]
            )
            half_lives = np.asarray(
                [
                    self._half_life_minutes(item.ingredient, item.properties, pressure)
                    for item, pressure in zip(prepared, sampled_vapor)
                ]
            )
            suppression_strength = float(np.clip(rng.normal(0.20, 0.05), 0.08, 0.40))
            for time_index, minutes in enumerate(TIMEPOINTS_MINUTES):
                current = activation * np.power(0.5, minutes / half_lives)
                suppressed = current / (
                    1.0 + suppression_strength * (interaction @ current)
                )
                mixture = suppressed @ profiles
                if mixture.sum() > 0:
                    mixture /= mixture.sum()
                time_mixtures[draw_index, time_index] = mixture
                time_intensities[draw_index, time_index] = suppressed.sum()
                time_similarities[draw_index, time_index] = semantic_brief_similarity(
                    target, mixture, brief.desired_dimensions, brief.avoided_dimensions
                )

        temporal: list[TemporalPoint] = []
        for time_index, minutes in enumerate(TIMEPOINTS_MINUTES):
            similarity_values = time_similarities[:, time_index]
            mixture = np.mean(time_mixtures[:, time_index, :], axis=0)
            dominant = tuple(
                SCENT_DIMENSIONS[index]
                for index in np.argsort(-mixture)[:3]
                if mixture[index] > 0
            )
            temporal.append(
                TemporalPoint(
                    minutes=minutes,
                    similarity=round(float(np.mean(similarity_values)), 4),
                    dominant_dimensions=dominant,
                    total_relative_intensity=round(
                        float(np.mean(time_intensities[:, time_index])), 6
                    ),
                    similarity_p05=round(float(np.percentile(similarity_values, 5)), 4),
                    similarity_p95=round(
                        float(np.percentile(similarity_values, 95)), 4
                    ),
                )
            )
        per_draw_mean = time_similarities @ TIMEPOINT_WEIGHTS
        per_draw_minimum = np.min(time_similarities, axis=1)
        mean_similarity = float(np.mean(per_draw_mean))
        temporal_p05 = float(np.percentile(per_draw_mean, 5))
        temporal_p95 = float(np.percentile(per_draw_mean, 95))
        minimum = float(np.mean(per_draw_minimum))
        minimum_p05 = float(np.percentile(per_draw_minimum, 5))
        uncertainty_width = temporal_p95 - temporal_p05
        required_applicability = float(
            getattr(brief.constraints, "simulation_min_applicability_percent", 70.0)
        )
        max_uncertainty = float(
            getattr(brief.constraints, "simulation_max_uncertainty_width", 12.0)
        )
        model_domain_passed = bool(
            applicability >= required_applicability
            and uncertainty_width <= max_uncertainty
        )

        flags: list[str] = [
            "simulation_only_not_measured_human_olfactory_accuracy",
            "ideal_hydroalcoholic_matrix_assumption",
            "nonadditive_competitive_mixture_model",
            "applicability_uses_direct_property_evidence_only",
            "monte_carlo_interval_is_prior_propagation_not_empirical_error_coverage",
        ]
        if vapor_coverage < 100.0:
            flags.append("vapor_pressure_or_boiling_point_prior_sampled")
        if threshold_coverage < 100.0:
            flags.append("odor_threshold_prior_sampled")
        if molecular_coverage < 100.0:
            flags.append("molecular_descriptor_prior_sampled")
        if applicability < required_applicability:
            flags.append("outside_configured_model_applicability")
        if uncertainty_width > max_uncertainty:
            flags.append("simulation_uncertainty_too_wide")
        if temporal_p05 < brief.constraints.target_similarity:
            flags.append("temporal_similarity_lower_bound_below_target")
        status = (
            "measured_property_twin"
            if complete_coverage >= 80.0
            else "physics_informed_twin"
            if molecular_coverage >= 50.0
            else "inferred_twin"
        )
        confidence = (
            "high_nonhuman_evidence"
            if applicability >= 80.0 and uncertainty_width <= 8.0
            else "bounded_nonhuman_evidence"
            if applicability >= required_applicability
            and uncertainty_width <= max_uncertainty
            else "insufficient_nonhuman_evidence"
        )
        return ScientificTwinResult(
            status=status,
            model_version=SCIENTIFIC_MODEL_VERSION,
            scientific_data_coverage_percent=round(complete_coverage, 4),
            molecular_descriptor_coverage_percent=round(molecular_coverage, 4),
            temporal_similarity_mean=round(mean_similarity, 4),
            minimum_temporal_similarity=round(minimum, 4),
            temporal_points=tuple(temporal),
            confidence=confidence,
            flags=tuple(flags),
            vapor_pressure_coverage_percent=round(vapor_coverage, 4),
            odor_threshold_coverage_percent=round(threshold_coverage, 4),
            model_applicability_percent=round(applicability, 4),
            temporal_similarity_p05=round(temporal_p05, 4),
            temporal_similarity_p95=round(temporal_p95, 4),
            minimum_temporal_similarity_p05=round(minimum_p05, 4),
            simulation_only_approved=False,
            monte_carlo_draws=draws,
            model_domain_passed=model_domain_passed,
        )
