"""Concentration-aware adaptation of the PhysSim-Core mixture model.

The original PhysSim R2 architecture learns abstract mass, charge, radius,
position, and velocity variables from RDKit descriptors.  The package carries
the two frozen project checkpoints plus a hash-bound, portable NumPy export.
The portable path reproduces inference without deserializing model objects; it
does not turn the historical training endpoint into perfume sensory truth.

Instead, it preserves the paper's useful mixture-level inductive bias while
grounding the initial state in the properties already available to the
perfumery digital twin.  Formula concentration, vapor pressure, odor
threshold, persistence, and active strength determine each particle's
time-dependent weight.  The three R2 core interactions then relax a bounded
descriptor field before a candidate formula is compared with an independently
evidenced quantitative reference composition.  Text-only requests have no
molecular target and therefore abstain instead of comparing with a formula made
by the same heuristic that generated the candidate.

The deterministic branch is a non-human ranking feature.  When the packaged
and validation-gated R2 checkpoint is available, its centered residual is
ensembled without treating the two branches as the same absolute scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Collection

import numpy as np

from .catalog import normalize_name
from .models import (
    Ingredient,
    RecipeLine,
    ScentBrief,
    profile_vector,
)
from .optimizer import cosine_similarity_percent
from .physsim_checkpoint import FrozenR2PhysSim
from .concentration_response import (
    FrozenConcentrationResponse,
    concentration_response_from_environment,
)
from .science import (
    ATMOSPHERIC_PRESSURE_PA,
    ETHANOL_MOLECULAR_WEIGHT,
    MolecularProperties,
    ScientificPropertyStore,
    TIMEPOINTS_MINUTES,
    TIMEPOINT_WEIGHTS,
    TemporalMixtureSimulator,
)


PHYSSIM_MODEL_VERSION = "concentration-headspace-physsim-core-1.1"


@dataclass(frozen=True)
class PhysSimTemporalPoint:
    minutes: int
    similarity: float
    perceptual_profile_similarity: float
    latent_trajectory_similarity: float


@dataclass(frozen=True)
class PhysSimResult:
    status: str
    model_version: str
    similarity: float
    temporal_similarity_mean: float
    minimum_temporal_similarity: float
    descriptor_coverage_percent: float
    vapor_pressure_coverage_percent: float
    odor_threshold_coverage_percent: float
    model_applicability_percent: float
    target_ingredient_ids: tuple[str, ...]
    temporal_points: tuple[PhysSimTemporalPoint, ...]
    flags: tuple[str, ...]
    deterministic_similarity: float = 0.0
    learned_r2_status: str = "not_run"
    learned_r2_similarity: float | None = None
    learned_r2_applicability_percent: float = 0.0
    learned_r2_candidate_structure_coverage_percent: float = 0.0
    learned_r2_target_structure_coverage_percent: float = 0.0
    learned_r2_descriptor_domain_coverage_percent: float = 0.0
    learned_r2_approved_weight: float = 0.0
    learned_r2_applied_weight: float = 0.0
    learned_r2_neutral_similarity_percent: float = 50.0
    learned_r2_centered_score_adjustment: float = 0.0
    learned_r2_checkpoint_sha256: str = ""
    learned_r2_member_predictions: tuple[float, ...] = ()
    learned_r2_member_disagreement_percent: float = 0.0
    learned_r2_prediction_interval_lower_percent: float = 0.0
    learned_r2_prediction_interval_upper_percent: float = 0.0
    learned_r2_ensemble_manifest_sha256: str = ""
    concentration_response_status: str = "not_run"
    concentration_response_similarity: float | None = None
    concentration_response_coverage_percent: float = 0.0
    concentration_response_applied_weight: float = 0.0
    comparison_target_status: str = "not_assessed"
    comparison_authorized: bool = False


@dataclass(frozen=True)
class _ParticleField:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    radii: np.ndarray
    pooling_weights: np.ndarray
    profiles: np.ndarray
    descriptor_coverage: float
    vapor_coverage: float
    threshold_coverage: float


@dataclass(frozen=True)
class _FieldFingerprint:
    vector: np.ndarray
    profile: np.ndarray
    chemical_centroid: np.ndarray


class ConcentrationAwarePhysSim:
    """Property-grounded PhysSim-Core adaptation for fragrance formulas."""

    # Dimensionless priors reported for the paper's trained core model.  They
    # are used as bounded interaction scales, not as literal physical constants.
    ATTRACTION_SCALE = 0.999
    CHARGE_SCALE = 0.947
    LJ_WELL_DEPTH = 0.481
    MASS_DECAY = 0.405
    VELOCITY_LIMIT = 1.182
    SOFT_CORE_DELTA = 0.5
    STEPS = 16
    DT = 0.1 / STEPS
    CHEMICAL_DIMENSIONS = 7

    def __init__(
        self,
        *,
        concentration_response: FrozenConcentrationResponse | None = None,
    ) -> None:
        self.headspace = TemporalMixtureSimulator()
        self.frozen_r2 = FrozenR2PhysSim()
        # The default adapter has no external model-release authorization, so
        # it remains a diagnostic curve and receives zero primary-score weight.
        self.concentration_response = (
            concentration_response or concentration_response_from_environment()
        )

    @staticmethod
    def _bounded_distribution(
        scores: np.ndarray, caps: np.ndarray, total: float
    ) -> np.ndarray:
        """Allocate a target pyramid total without exceeding material caps."""
        if len(scores) == 0 or total <= 0:
            return np.zeros_like(scores)
        positive = np.maximum(scores, 1e-9)
        if float(caps.sum()) + 1e-8 < total:
            # Target prototypes are diagnostics only, but must still be built
            # from a feasible amount of safe material.
            total = float(caps.sum())
        result = np.zeros_like(positive)
        active = np.ones(len(positive), dtype=bool)
        remaining = float(total)
        for _ in range(len(positive) + 1):
            if remaining <= 1e-9 or not np.any(active):
                break
            active_scores = positive[active]
            proposal = remaining * active_scores / active_scores.sum()
            active_indices = np.where(active)[0]
            saturated = False
            for index, value in zip(active_indices, proposal):
                room = max(0.0, float(caps[index] - result[index]))
                if value >= room - 1e-10:
                    result[index] += room
                    remaining -= room
                    active[index] = False
                    saturated = True
            if not saturated:
                result[active_indices] += proposal
                remaining = 0.0
                break
        return result

    def build_target_formula(
        self,
        brief: ScentBrief,
        ingredients: Collection[Ingredient],
    ) -> list[RecipeLine]:
        """Construct a diagnostic prototype that is never used as score truth.

        Kept for formula-space debugging and compatibility only. ``evaluate``
        refuses to use this self-generated material set as a comparison target.
        """
        target = profile_vector(brief.target_profile)
        avoided = set(brief.avoided_dimensions)
        bans = {
            normalize_name(value)
            for value in (
                *brief.constraints.explicit_bans,
                *brief.excluded_ingredients,
            )
            if value
        }
        eligible = [
            item
            for item in ingredients
            if item.formulation_ready
            and not item.blocked
            and item.risk_tier <= brief.constraints.max_risk_tier
            and item.price_per_kg <= brief.constraints.max_ingredient_price_per_kg
            and item.availability >= brief.constraints.min_availability
            and (brief.constraints.allow_rare or item.rarity != "rare")
            and not (
                {
                    normalize_name(value)
                    for value in (*item.all_names(), item.ingredient_id)
                }
                & bans
            )
        ]
        lines: list[RecipeLine] = []
        for pyramid, pyramid_total in brief.pyramid_ratios.items():
            group = [item for item in eligible if item.pyramid == pyramid]
            if not group:
                continue

            def score(item: Ingredient) -> float:
                vector = item.vector()
                scent_match = cosine_similarity_percent(vector, target) / 100.0
                identity = max(
                    (item.profile.get(name, 0.0) for name in brief.desired_dimensions),
                    default=0.0,
                )
                avoided_mass = sum(item.profile.get(name, 0.0) for name in avoided)
                affordability = 1.0 - min(
                    1.0,
                    item.price_per_kg
                    / max(1.0, brief.constraints.max_ingredient_price_per_kg),
                )
                return max(
                    0.001,
                    scent_match * 0.68
                    + identity * 0.20
                    + item.availability * 0.08
                    + affordability * 0.04
                    - avoided_mass * 0.35,
                )

            ranked = sorted(
                group, key=lambda item: (score(item), item.ingredient_id), reverse=True
            )
            selected: list[Ingredient] = []
            for item in ranked:
                selected.append(item)
                if (
                    len(selected) >= 4
                    and sum(current.as_supplied_cap_percent() for current in selected)
                    >= pyramid_total
                ):
                    break
            scores = np.asarray([score(item) for item in selected], dtype=float)
            caps = np.asarray(
                [item.as_supplied_cap_percent() for item in selected], dtype=float
            )
            weights = self._bounded_distribution(scores, caps, float(pyramid_total))
            for weight, ingredient in zip(weights, selected):
                if weight <= 1e-8:
                    continue
                finished_percent = (
                    weight * brief.constraints.product_concentration_percent / 100.0
                )
                lines.append(
                    RecipeLine(
                        ingredient_id=ingredient.ingredient_id,
                        name=ingredient.name,
                        pyramid=ingredient.pyramid,
                        concentrate_percent=float(weight),
                        finished_product_percent=float(finished_percent),
                        volume_ml_for_batch=None,
                        price_per_kg=ingredient.price_per_kg,
                        availability=ingredient.availability,
                        risk_tier=ingredient.risk_tier,
                        reason="natural-language PhysSim target prototype",
                        active_material_percent=(
                            float(weight) * ingredient.active_strength_percent / 100.0
                        ),
                        active_strength_percent=ingredient.active_strength_percent,
                        data_source=ingredient.data_source,
                        approved_formulation_scopes=(
                            ingredient.approved_formulation_scopes
                        ),
                        approval_expires_at=ingredient.approval_expires_at,
                        promotion_artifact_id=ingredient.promotion_artifact_id,
                    )
                )
        return sorted(lines, key=lambda line: line.ingredient_id)

    @staticmethod
    def _descriptor_values(
        ingredient: Ingredient, properties: MolecularProperties | None
    ) -> tuple[np.ndarray, float, float, float, float, float]:
        molecular_weight = properties.molecular_weight if properties else 180.0
        xlogp = properties.xlogp if properties and properties.xlogp is not None else 2.0
        tpsa = properties.tpsa if properties and properties.tpsa is not None else 35.0
        donors = (
            float(properties.hbond_donors)
            if properties and properties.hbond_donors is not None
            else 0.5
        )
        acceptors = (
            float(properties.hbond_acceptors)
            if properties and properties.hbond_acceptors is not None
            else 2.0
        )
        rotatable = (
            float(properties.rotatable_bonds)
            if properties and properties.rotatable_bonds is not None
            else 3.0
        )
        complexity = (
            properties.complexity
            if properties and properties.complexity is not None
            else 150.0
        )
        chemical = np.asarray(
            [
                math.tanh(math.log(max(1.0, molecular_weight) / 150.0)),
                math.tanh((xlogp - 2.0) / 3.0),
                math.tanh((tpsa - 35.0) / 80.0),
                math.tanh((donors - 0.5) / 2.0),
                math.tanh((acceptors - 2.0) / 4.0),
                math.tanh((rotatable - 3.0) / 6.0),
                math.tanh((math.log1p(max(0.0, complexity)) - 5.0) / 2.0),
            ],
            dtype=float,
        )
        present = 0
        if properties:
            present = 1 + sum(
                value is not None
                for value in (
                    properties.xlogp,
                    properties.tpsa,
                    properties.hbond_donors,
                    properties.hbond_acceptors,
                    properties.rotatable_bonds,
                    properties.complexity,
                )
            )
        descriptor_coverage = present / 7.0
        charge = math.tanh(
            (tpsa / 90.0 + acceptors * 0.08 - donors * 0.04 - xlogp / 5.0) - 0.25
        )
        radius = float(
            np.clip(
                0.45
                + 0.30 * math.sqrt(max(1.0, molecular_weight) / 300.0)
                + 0.12 * math.tanh(complexity / 400.0),
                0.45,
                1.10,
            )
        )
        return chemical, descriptor_coverage, charge, radius, molecular_weight, xlogp

    def _particle_field(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        store: ScientificPropertyStore,
        minutes: int,
    ) -> _ParticleField:
        ordered = sorted(lines, key=lambda line: line.ingredient_id)
        if not ordered:
            raise ValueError("PhysSim formula cannot be empty")
        total_finished = sum(
            max(0.0, line.finished_product_percent) for line in ordered
        )
        base_moles = max(0.0, 100.0 - total_finished) / ETHANOL_MOLECULAR_WEIGHT
        rows: list[
            tuple[RecipeLine, Ingredient, MolecularProperties | None, float, float]
        ] = []
        odorant_moles = 0.0
        for line in ordered:
            ingredient = ingredients[line.ingredient_id]
            properties = store.get(line.ingredient_id)
            molecular_weight = properties.molecular_weight if properties else 180.0
            active_mass = (
                max(0.0, line.finished_product_percent)
                * max(0.0, line.active_strength_percent)
                / 100.0
            )
            moles = active_mass / max(1e-9, molecular_weight)
            rows.append((line, ingredient, properties, molecular_weight, moles))
            odorant_moles += moles
        total_moles = max(1e-12, base_moles + odorant_moles)

        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        raw_weights: list[float] = []
        charges: list[float] = []
        radii: list[float] = []
        profiles: list[np.ndarray] = []
        descriptor_coverages: list[float] = []
        vapor_evidence: list[float] = []
        threshold_evidence: list[float] = []
        concentration_weights: list[float] = []

        for line, ingredient, properties, _, moles in rows:
            chemical, coverage, charge, radius, _, _ = self._descriptor_values(
                ingredient, properties
            )
            profile = ingredient.vector()
            position = np.concatenate((chemical, np.sqrt(np.maximum(profile, 0.0))))
            vapor_pressure, _ = self.headspace._vapor_pressure_prior(
                ingredient, properties
            )
            odor_threshold, _ = self.headspace._threshold_prior(ingredient, properties)
            liquid_fraction = moles / total_moles
            xlogp = (
                properties.xlogp if properties and properties.xlogp is not None else 2.0
            )
            activity = max(0.50, min(3.0, math.exp(0.18 * (xlogp - 2.0))))
            gas_ppm = (
                liquid_fraction
                * activity
                * vapor_pressure
                / ATMOSPHERIC_PRESSURE_PA
                * 1_000_000.0
            )
            transport = self.headspace._air_to_receptor_transport(properties)
            odor_activity = max(1e-12, gas_ppm * transport / odor_threshold)
            half_life = self.headspace._half_life_minutes(
                ingredient, properties, vapor_pressure
            )
            persistence = 0.5 ** (minutes / half_life)
            # log1p retains concentration sensitivity while preventing a
            # single ultra-low threshold material from dominating the field.
            perceived_weight = math.log1p(odor_activity) * persistence
            perceived_weight *= max(0.05, ingredient.odor_impact)
            raw_weights.append(max(1e-12, perceived_weight))
            volatility = math.tanh(math.log10(max(1e-8, vapor_pressure) + 1.0) / 3.0)
            velocity = np.zeros_like(position)
            velocity[self.CHEMICAL_DIMENSIONS :] = profile * volatility * 0.08
            positions.append(position)
            velocities.append(velocity)
            charges.append(charge)
            radii.append(radius)
            profiles.append(profile)
            descriptor_coverages.append(coverage)
            vapor_evidence.append(
                float(
                    bool(
                        properties
                        and (
                            properties.vapor_pressure_pa_25c is not None
                            or properties.boiling_point_c is not None
                        )
                    )
                )
            )
            threshold_evidence.append(
                float(bool(properties and properties.odor_threshold_ppm is not None))
            )
            concentration_weights.append(max(1e-12, line.concentrate_percent))

        pooling = np.asarray(raw_weights, dtype=float)
        pooling /= max(1e-12, float(pooling.sum()))
        importance = np.asarray(concentration_weights, dtype=float)
        importance /= max(1e-12, float(importance.sum()))
        # Keep particle inertia away from zero while preserving headspace rank.
        masses = 0.35 + pooling * len(pooling) * 0.85
        return _ParticleField(
            positions=np.asarray(positions, dtype=float),
            velocities=np.asarray(velocities, dtype=float),
            masses=masses,
            charges=np.asarray(charges, dtype=float),
            radii=np.asarray(radii, dtype=float),
            pooling_weights=pooling,
            profiles=np.asarray(profiles, dtype=float),
            descriptor_coverage=float(importance @ np.asarray(descriptor_coverages))
            * 100.0,
            vapor_coverage=float(importance @ np.asarray(vapor_evidence)) * 100.0,
            threshold_coverage=float(importance @ np.asarray(threshold_evidence))
            * 100.0,
        )

    def _relax(self, field: _ParticleField) -> _FieldFingerprint:
        positions = field.positions.copy()
        velocities = field.velocities.copy()
        masses = field.masses.copy()
        initial_masses = masses.copy()
        trajectory = [positions.copy()]
        speeds: list[np.ndarray] = []
        count, dimensions = positions.shape
        if count > 1:
            diagonal_mask = 1.0 - np.eye(count, dtype=float)
            for _ in range(self.STEPS):
                difference = positions[:, None, :] - positions[None, :, :]
                euclidean = np.linalg.norm(difference, axis=-1)
                direction = difference / np.maximum(euclidean[..., None], 1e-12)
                normalized_distance = euclidean / math.sqrt(max(1, dimensions))
                distance = np.sqrt(normalized_distance**2 + self.SOFT_CORE_DELTA**2)
                mass_pair = masses[:, None] * masses[None, :]
                attraction = (
                    -self.ATTRACTION_SCALE * mass_pair / np.maximum(distance**2, 1e-12)
                )[..., None] * direction
                charge_pair = field.charges[:, None] * field.charges[None, :]
                charge_force = (
                    self.CHARGE_SCALE * charge_pair / np.maximum(distance**2, 1e-12)
                )[..., None] * direction
                radius_pair = (field.radii[:, None] + field.radii[None, :]) / 2.0
                ratio = np.clip(radius_pair / np.maximum(distance, 1e-12), 0.05, 1.35)
                ratio_six = ratio**6
                lj_scalar = (
                    24.0
                    * self.LJ_WELL_DEPTH
                    / np.maximum(distance, 1e-12)
                    * (2.0 * ratio_six**2 - ratio_six)
                )
                lj_force = lj_scalar[..., None] * direction
                forces = (attraction + charge_force + lj_force) * diagonal_mask[
                    ..., None
                ]
                force_norm = np.linalg.norm(forces, axis=-1, keepdims=True)
                forces *= np.minimum(1.0, 4.0 / np.maximum(force_norm, 1e-12))
                acceleration = forces.sum(axis=1) / np.maximum(masses[:, None], 0.10)
                velocities += acceleration * self.DT
                speed = np.linalg.norm(velocities, axis=-1)
                velocity_scale = np.minimum(
                    1.0, self.VELOCITY_LIMIT / np.maximum(speed, 1e-12)
                )
                velocities *= velocity_scale[:, None]
                positions += velocities * self.DT
                masses *= np.exp(-self.MASS_DECAY * self.DT / (masses**2 + 0.05))
                positions = np.nan_to_num(positions, nan=0.0, posinf=4.0, neginf=-4.0)
                velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
                trajectory.append(positions.copy())
                speeds.append(np.linalg.norm(velocities, axis=-1))
        else:
            speeds.append(np.linalg.norm(velocities, axis=-1))

        weights = field.pooling_weights
        centroid = weights @ positions
        variance = weights @ ((positions - centroid) ** 2)
        trajectory_array = np.asarray(trajectory, dtype=float)
        trajectory_centroids = np.einsum("n,tnd->td", weights, trajectory_array)
        trajectory_variance = float(np.var(trajectory_centroids, axis=0).mean())
        speed_array = np.asarray(speeds, dtype=float)
        scalar_features = np.asarray(
            [
                trajectory_variance,
                float(np.max(speed_array)),
                float(np.var(speed_array)),
                float(weights @ masses),
                float(weights @ (masses / np.maximum(initial_masses, 1e-12))),
                float(weights @ np.abs(field.charges)),
                float(weights @ field.radii),
            ],
            dtype=float,
        )
        fingerprint = np.concatenate(
            (centroid, np.sqrt(np.maximum(variance, 0.0)), scalar_features)
        )
        profile = weights @ field.profiles
        if profile.sum() > 0:
            profile /= profile.sum()
        return _FieldFingerprint(
            vector=fingerprint,
            profile=profile,
            chemical_centroid=centroid[: self.CHEMICAL_DIMENSIONS],
        )

    @staticmethod
    def _distance_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape or left.size == 0:
            return 0.0
        if np.array_equal(left, right):
            return 100.0
        scale = max(
            0.15,
            (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
            / (2.0 * math.sqrt(left.size)),
        )
        distance = float(np.linalg.norm(left - right)) / (math.sqrt(left.size) * scale)
        return max(0.0, min(100.0, 100.0 * math.exp(-0.90 * distance)))

    def compare(
        self,
        left_lines: list[RecipeLine],
        right_lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        store: ScientificPropertyStore,
    ) -> PhysSimResult:
        if not left_lines or not right_lines:
            return PhysSimResult(
                status="insufficient_data",
                model_version=PHYSSIM_MODEL_VERSION,
                similarity=0.0,
                temporal_similarity_mean=0.0,
                minimum_temporal_similarity=0.0,
                descriptor_coverage_percent=0.0,
                vapor_pressure_coverage_percent=0.0,
                odor_threshold_coverage_percent=0.0,
                model_applicability_percent=0.0,
                target_ingredient_ids=tuple(
                    sorted({line.ingredient_id for line in right_lines})
                ),
                temporal_points=(),
                flags=("empty_formula",),
            )

        points: list[PhysSimTemporalPoint] = []
        descriptor_coverages: list[float] = []
        vapor_coverages: list[float] = []
        threshold_coverages: list[float] = []
        similarities: list[float] = []
        for minutes in TIMEPOINTS_MINUTES:
            left_field = self._particle_field(left_lines, ingredients, store, minutes)
            right_field = self._particle_field(right_lines, ingredients, store, minutes)
            left = self._relax(left_field)
            right = self._relax(right_field)
            profile_similarity = cosine_similarity_percent(left.profile, right.profile)
            trajectory_similarity = self._distance_similarity(left.vector, right.vector)
            chemical_similarity = self._distance_similarity(
                left.chemical_centroid, right.chemical_centroid
            )
            similarity = (
                profile_similarity * 0.70
                + trajectory_similarity * 0.20
                + chemical_similarity * 0.10
            )
            similarities.append(similarity)
            points.append(
                PhysSimTemporalPoint(
                    minutes=minutes,
                    similarity=round(similarity, 4),
                    perceptual_profile_similarity=round(profile_similarity, 4),
                    latent_trajectory_similarity=round(trajectory_similarity, 4),
                )
            )
            descriptor_coverages.append(
                (left_field.descriptor_coverage + right_field.descriptor_coverage) / 2.0
            )
            vapor_coverages.append(
                (left_field.vapor_coverage + right_field.vapor_coverage) / 2.0
            )
            threshold_coverages.append(
                (left_field.threshold_coverage + right_field.threshold_coverage) / 2.0
            )

        descriptor_coverage = float(np.mean(descriptor_coverages))
        vapor_coverage = float(np.mean(vapor_coverages))
        threshold_coverage = float(np.mean(threshold_coverages))
        # Concentration and curated odor profiles are complete for every
        # formula line. Direct physical evidence is reported separately.
        applicability = (
            descriptor_coverage * 0.55
            + vapor_coverage * 0.15
            + threshold_coverage * 0.10
            + 20.0
        )
        temporal_mean = float(np.asarray(similarities) @ TIMEPOINT_WEIGHTS)
        minimum = float(np.min(similarities))
        flags = [
            "adapted_from_physsim_r2_core",
            "all_interactions_use_soft_core_distance",
            "concentration_headspace_and_persistence_weighted",
            "deterministic_nonhuman_mixture_ranking_feature",
            "deterministic_branch_is_not_the_trained_r2_neural_checkpoint",
            "not_measured_human_olfactory_accuracy",
        ]
        if descriptor_coverage < 100.0:
            flags.append("molecular_descriptor_priors_used")
        if vapor_coverage < 100.0:
            flags.append("vapor_pressure_or_boiling_point_priors_used")
        if threshold_coverage < 100.0:
            flags.append("odor_threshold_priors_used")
        status = (
            "property_grounded"
            if descriptor_coverage >= 70.0
            else "partially_inferred"
            if descriptor_coverage >= 40.0
            else "mostly_inferred"
        )
        return PhysSimResult(
            status=status,
            model_version=PHYSSIM_MODEL_VERSION,
            similarity=round(temporal_mean, 4),
            temporal_similarity_mean=round(temporal_mean, 4),
            minimum_temporal_similarity=round(minimum, 4),
            descriptor_coverage_percent=round(descriptor_coverage, 4),
            vapor_pressure_coverage_percent=round(vapor_coverage, 4),
            odor_threshold_coverage_percent=round(threshold_coverage, 4),
            model_applicability_percent=round(applicability, 4),
            target_ingredient_ids=tuple(
                sorted({line.ingredient_id for line in right_lines})
            ),
            temporal_points=tuple(points),
            flags=tuple(flags),
        )

    def evaluate(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        brief: ScentBrief,
        store: ScientificPropertyStore,
        *,
        reference_target_lines: list[RecipeLine] | None = None,
    ) -> PhysSimResult:
        if reference_target_lines is not None and not reference_target_lines:
            raise ValueError("reference target lines cannot be empty")
        if reference_target_lines is None:
            fields = [
                self._particle_field(lines, ingredients, store, minutes)
                for minutes in TIMEPOINTS_MINUTES
            ]
            descriptor_coverage = min(item.descriptor_coverage for item in fields)
            vapor_coverage = min(item.vapor_coverage for item in fields)
            threshold_coverage = min(item.threshold_coverage for item in fields)
            applicability = (
                descriptor_coverage * 0.40
                + vapor_coverage * 0.30
                + threshold_coverage * 0.30
            )
            candidate_concentration = self.concentration_response.formula_profile(
                lines, ingredients
            )
            flags = [
                "no_evidenced_reference_composition_target",
                "text_brief_is_not_a_molecular_comparison_target",
                "self_generated_target_scoring_disabled",
                "candidate_formula_characterized_without_similarity_claim",
                *candidate_concentration.flags,
            ]
            return PhysSimResult(
                status="target_unavailable",
                model_version=PHYSSIM_MODEL_VERSION,
                similarity=0.0,
                temporal_similarity_mean=0.0,
                minimum_temporal_similarity=0.0,
                descriptor_coverage_percent=round(descriptor_coverage, 4),
                vapor_pressure_coverage_percent=round(vapor_coverage, 4),
                odor_threshold_coverage_percent=round(threshold_coverage, 4),
                model_applicability_percent=round(applicability, 4),
                target_ingredient_ids=(),
                temporal_points=(),
                flags=tuple(dict.fromkeys(flags)),
                concentration_response_status=(
                    "disabled_for_ablation"
                    if not brief.constraints.enable_concentration_response
                    else candidate_concentration.status
                ),
                concentration_response_coverage_percent=(
                    candidate_concentration.evidence_coverage_percent
                ),
                learned_r2_status="target_unavailable",
                comparison_target_status="absent",
                comparison_authorized=False,
            )
        target = reference_target_lines
        deterministic = self.compare(lines, target, ingredients, store)
        candidate_concentration = self.concentration_response.formula_profile(
            lines, ingredients
        )
        target_concentration = self.concentration_response.formula_profile(
            target, ingredients
        )
        concentration_similarity = None
        concentration_weight = 0.0
        primary_similarity = deterministic.similarity
        concentration_profiles_available = candidate_concentration.status in {
            "validation_gated",
            "validation_gated_diagnostic_only",
        } and target_concentration.status in {
            "validation_gated",
            "validation_gated_diagnostic_only",
        }
        if concentration_profiles_available:
            # Preserve the measured mono-molecule dilution comparison as a
            # diagnostic even when no independent model-release signature has
            # authorized it to influence the primary mixture score.
            concentration_similarity = cosine_similarity_percent(
                candidate_concentration.profile,
                target_concentration.profile,
            )
        if (
            brief.constraints.enable_concentration_response
            and concentration_profiles_available
            and candidate_concentration.approved_primary_score_weight > 0.0
            and target_concentration.approved_primary_score_weight > 0.0
        ):
            # Limited because the calibration target is mono-molecule intensity,
            # not mixture similarity. It nevertheless lets measured dilution
            # response participate directly in candidate selection only after
            # an independently signed deployment authorization.
            concentration_weight = min(
                candidate_concentration.approved_primary_score_weight,
                target_concentration.approved_primary_score_weight,
            )
            primary_similarity = (
                (1.0 - concentration_weight) * deterministic.similarity
                + concentration_weight * concentration_similarity
            )
        learned = self.frozen_r2.evaluate(lines, target)
        learned_similarity = learned.similarity
        weight = (
            learned.applied_primary_score_weight
            if brief.constraints.enable_learned_r2
            else 0.0
        )
        ensemble_similarity = primary_similarity
        if learned_similarity is not None and weight > 0:
            ensemble_similarity = primary_similarity + learned.centered_score_adjustment
        flags = [
            *deterministic.flags,
            *candidate_concentration.flags,
            *target_concentration.flags,
            *learned.flags,
        ]
        flags.append("evidenced_reference_composition_target")
        if weight > 0:
            flags.append("validation_gated_r2_weight_applied")
        elif not brief.constraints.enable_learned_r2:
            flags.append("learned_r2_disabled_for_ablation")
        if not brief.constraints.enable_concentration_response:
            flags.append("concentration_response_disabled_for_ablation")
        elif concentration_profiles_available and concentration_weight <= 0.0:
            flags.append(
                "concentration_response_independent_authorization_missing_weight_zero"
            )
        return replace(
            deterministic,
            model_version=(
                f"{PHYSSIM_MODEL_VERSION}+jcim-r2-checkpoint"
                if learned.status != "unavailable"
                else PHYSSIM_MODEL_VERSION
            ),
            similarity=round(ensemble_similarity, 4),
            deterministic_similarity=deterministic.similarity,
            learned_r2_status=learned.status,
            learned_r2_similarity=learned_similarity,
            learned_r2_applicability_percent=learned.applicability_percent,
            learned_r2_candidate_structure_coverage_percent=(
                learned.candidate_structure_coverage_percent
            ),
            learned_r2_target_structure_coverage_percent=(
                learned.target_structure_coverage_percent
            ),
            learned_r2_descriptor_domain_coverage_percent=(
                learned.descriptor_domain_coverage_percent
            ),
            learned_r2_approved_weight=learned.approved_primary_score_weight,
            learned_r2_applied_weight=learned.applied_primary_score_weight,
            learned_r2_neutral_similarity_percent=(learned.neutral_similarity_percent),
            learned_r2_centered_score_adjustment=(learned.centered_score_adjustment),
            learned_r2_checkpoint_sha256=learned.checkpoint_sha256,
            learned_r2_member_predictions=learned.member_predictions_percent,
            learned_r2_member_disagreement_percent=learned.member_disagreement_percent,
            learned_r2_prediction_interval_lower_percent=(
                learned.prediction_interval_lower_percent
            ),
            learned_r2_prediction_interval_upper_percent=(
                learned.prediction_interval_upper_percent
            ),
            learned_r2_ensemble_manifest_sha256=learned.ensemble_manifest_sha256,
            concentration_response_status=(
                "validation_gated"
                if concentration_weight > 0
                else (
                    "disabled_for_ablation"
                    if not brief.constraints.enable_concentration_response
                    else candidate_concentration.status
                )
            ),
            concentration_response_similarity=(
                round(concentration_similarity, 4)
                if concentration_similarity is not None
                else None
            ),
            concentration_response_coverage_percent=round(
                min(
                    candidate_concentration.evidence_coverage_percent,
                    target_concentration.evidence_coverage_percent,
                ),
                4,
            ),
            concentration_response_applied_weight=concentration_weight,
            comparison_target_status="evidenced_quantitative_composition",
            comparison_authorized=True,
            flags=tuple(dict.fromkeys(flags)),
        )
