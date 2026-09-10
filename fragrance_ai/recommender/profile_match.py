"""Complete model-profile agreement, deliberately separate from human similarity.

Every represented dimension participates. Extra notes cannot disappear by
slicing to the requested dimensions. Inputs are normalized independently, so
arbitrary descriptor-weight scale is not confused with profile agreement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Mapping, Sequence

import numpy as np

from .models import RecipeResult, SCENT_DIMENSIONS

PROFILE_MATCH_VERSION = "full-model-profile-agreement-1.0"
PROFILE_MATCH_KIND = "full_model_profile_agreement_not_human_similarity"


@dataclass(frozen=True)
class ProfileMatch:
    status: str
    score: float | None
    cosine_score: float | None
    overlap_score: float | None
    total_variation: float | None
    avoided_mass: float | None
    target_profile: dict[str, float]
    predicted_profile: dict[str, float]
    excess: dict[str, float]
    deficit: dict[str, float]
    dimensions: tuple[str, ...]
    version: str = PROFILE_MATCH_VERSION
    score_kind: str = PROFILE_MATCH_KIND
    actual_human_similarity_measured: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _vector(values, dimensions: tuple[str, ...]) -> np.ndarray:
    if isinstance(values, Mapping):
        unknown = set(values) - set(dimensions)
        if unknown:
            raise ValueError(f"unknown profile dimensions: {sorted(unknown)}")
        result = np.asarray([values.get(name, 0.0) for name in dimensions], dtype=float)
    else:
        result = np.asarray(values, dtype=float)
    if result.shape != (len(dimensions),) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("profile must have finite, nonnegative values in the declared dimensions")
    return result


def _normalize(value: np.ndarray) -> np.ndarray | None:
    largest = float(value.max())
    if largest <= 0:
        return None
    scaled = value / largest
    return scaled / scaled.sum()


def compare_profiles(target, predicted, *, avoided: Sequence[str] = (),
                     dimensions: Sequence[str] = SCENT_DIMENSIONS) -> ProfileMatch:
    """Minimum of full-vector cosine, probability-mass overlap and avoidance.

    A 90-point result implies <= 0.1 total variation *in these normalized
    model vectors*. Normalization removes absolute intensity; no calibrated
    human-perception accuracy or error probability follows from this bound.
    """
    names = tuple(dimensions)
    if not names or len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("profile dimensions must be unique nonempty names")
    if any(name not in names for name in avoided):
        raise ValueError("unknown avoided profile dimension")
    a, b = _vector(target, names), _vector(predicted, names)
    normalized_a, normalized_b = _normalize(a), _normalize(b)
    if normalized_a is None or normalized_b is None:
        return ProfileMatch("undefined_empty_profile", None, None, None, None, None,
                            dict(zip(names, a.tolist())), dict(zip(names, b.tolist())), {}, {}, names)
    a, b = normalized_a, normalized_b
    total_variation = float(0.5 * np.abs(a - b).sum())
    overlap = 100.0 * float(np.minimum(a, b).sum())
    cosine = 100.0 * float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    avoided_mass = float(sum(b[names.index(name)] for name in set(avoided)))
    score = float(np.clip(min(cosine, overlap, 100.0 * (1.0 - avoided_mass)), 0.0, 100.0))
    return ProfileMatch(
        "computed_model_profile_only", score, min(100.0, cosine), min(100.0, overlap),
        total_variation, avoided_mass, dict(zip(names, a.tolist())), dict(zip(names, b.tolist())),
        {name: float(delta) for name, delta in zip(names, b - a) if delta > 1e-12},
        {name: float(delta) for name, delta in zip(names, a - b) if delta > 1e-12}, names,
    )


def full_profile_similarity(target, predicted, desired_dimensions=(), avoided_dimensions=()) -> float:
    """Optimizer-compatible score. Desired dimensions never mask other axes."""
    a, b = _vector(target, SCENT_DIMENSIONS), _vector(predicted, SCENT_DIMENSIONS)
    a, b = _normalize(a), _normalize(b)
    if a is None or b is None:
        return 0.0
    cosine = 100.0 * float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    overlap = 100.0 * float(np.minimum(a, b).sum())
    if any(name not in SCENT_DIMENSIONS for name in avoided_dimensions):
        raise ValueError("unknown avoided profile dimension")
    avoidance = 100.0 * (1.0 - sum(float(b[SCENT_DIMENSIONS.index(name)]) for name in set(avoided_dimensions)))
    return float(np.clip(min(cosine, overlap, avoidance), 0.0, 100.0))


def assess_recipe_profiles(brief, achieved_profile: Mapping, temporal_points: Sequence[Mapping],
                           time_weights: Sequence[float]) -> dict:
    from .intent_controls import representation_contract
    nominal = compare_profiles(brief.target_profile, achieved_profile, avoided=brief.avoided_dimensions)
    points, temporal_score = [], None
    if temporal_points:
        weights = np.array(time_weights, dtype=float, copy=True)
        if weights.shape != (len(temporal_points),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.max() <= 0:
            raise ValueError("invalid temporal profile weights")
        weights = _normalize(weights)
        for point, weight in zip(temporal_points, weights):
            target = point.get("target_profile")
            if target is None:
                target = brief.target_profile
            avoided = sorted(set(brief.avoided_dimensions) | set(brief.phase_avoided_dimensions.get(point.get("phase", ""), [])))
            assessment = compare_profiles(target, point.get("scent_profile", {}), avoided=avoided)
            points.append({"minutes": point.get("minutes"), "phase": point.get("phase"), "weight": float(weight), **assessment.to_dict()})
        if all(point["score"] is not None for point in points if point["weight"] > 0):
            temporal_score = float(sum(point["score"] * point["weight"] for point in points if point["weight"] > 0))
    score = nominal.score
    if temporal_points:
        score = None if score is None or temporal_score is None else min(score, temporal_score)
    target = float(brief.constraints.target_similarity)
    return {"version": PROFILE_MATCH_VERSION, "score_kind": PROFILE_MATCH_KIND,
            "target_representation": representation_contract(brief),
            "score": score, "target": target, "target_met": score is not None and score + 1e-8 >= target,
            "nominal": nominal.to_dict(), "temporal": points, "temporal_mean_score": temporal_score,
            "temporal_minimum_score": min((point["score"] for point in points if point["weight"] > 0), default=None) if temporal_score is not None else None,
            "score_aggregation": "minimum_of_nominal_and_complete_time_weighted_profile_scores",
            "uncertainty_interval": None, "uncertainty_kind": "point_estimate_not_calibrated_human_error_interval",
            "target_source": "natural_language_model_profile_not_measured_reference_odor",
            "actual_human_similarity_measured": False}


@dataclass
class FullProfileRecipeResult(RecipeResult):
    calculated_profile_similarity: float | None = None
    full_profile_target_met: bool = False
    legacy_preference_score: float = 0.0
    full_profile_assessment: dict = field(default_factory=dict)
    score_contract: dict = field(default_factory=dict)
    perception_guidance: dict | None = None

    def to_dict(self) -> dict:
        result = super().to_dict()
        if self.perception_guidance is None:
            result.pop("perception_guidance", None)
        return result


def attach_profile_assessment(result: RecipeResult, assessment: dict, *, strict: bool) -> RecipeResult:
    output = FullProfileRecipeResult(
        **{item.name: getattr(result, item.name) for item in fields(RecipeResult)},
        calculated_profile_similarity=assessment["score"], full_profile_target_met=assessment["target_met"],
        legacy_preference_score=result.similarity_score, full_profile_assessment=assessment,
        perception_guidance=getattr(result, "perception_guidance", None),
        score_contract={"version": "2.0", "primary_profile_score_field": "calculated_profile_similarity",
                        "product_model": {"product": "perfume", "prediction_model": "perfume_temporal_mixture",
                            "requested_product": result.brief.constraints.product_category,
                            "scope": "perfume_or_fragrance_concentrate_not_finished_lotion",
                            "body_lotion_matrix_evaluated": False, "cross_product_scores_comparable": False},
                        "compatibility_score_field": "legacy_preference_score",
                        "legacy_similarity_field_retained": not strict,
                        "strict_full_profile_gate": strict, "actual_human_90_proven_by_this_score": False},
    )
    boundary = (
        "전체 19차원 모델 향 프로필 판정에는 calculated_profile_similarity와 full_profile_target_met를 사용합니다. "
        "호환 모드의 similarity_score는 기존 선호 점수이며, 두 점수 모두 실제 후각 정확도가 아닙니다."
    )
    output.limitations = [*result.limitations]
    if boundary not in output.limitations:
        output.limitations.append(boundary)
    if strict:
        output.similarity_score = assessment["score"] if assessment["score"] is not None else 0.0
        output.raw_similarity_score = output.similarity_score
        output.similarity_kind = PROFILE_MATCH_KIND
        if not assessment["target_met"]:
            output.recipe = []
            output.status = "no_safe_match"
            output.simulation_only_approved = False
            rendered = "계산 불가" if assessment["score"] is None else f"{assessment['score']:.2f}점"
            explanation = f"전체 향 프로필 {rendered}: 요청 기준 {assessment['target']:.2f}점 미충족. closest_candidate는 미승인 후보입니다."
            output.message = explanation if result.recipe else f"{result.message}; {explanation}"
            if output.manufacturing_plan is not None:
                output.manufacturing_plan = replace(
                    output.manufacturing_plan, ready_for_lab_trial=False, ready_for_manufacture=False,
                    readiness_blockers=[*output.manufacturing_plan.readiness_blockers, explanation],
                )
            if output.perception_guidance is not None:
                output.perception_guidance = {**output.perception_guidance, "recipe_returned": False, "recipe_changed": False,
                                             "status": "candidate_only_no_full_profile_match"}
    return output
