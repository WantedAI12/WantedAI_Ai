"""Constrained scent-profile optimizer for affordable and available materials."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .catalog import HistoricalReferenceCorpus
from .models import (
    Ingredient,
    RecipeLine,
    ScentBrief,
    SCENT_DIMENSIONS,
    profile_vector,
)


class NoFeasibleFormula(RuntimeError):
    pass


def cosine_similarity_percent(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return max(0.0, min(100.0, float(np.dot(left, right) / denominator * 100.0)))


def semantic_brief_similarity(
    target: np.ndarray,
    achieved: np.ndarray,
    desired_dimensions: list[str],
    avoided_dimensions: list[str],
) -> float:
    """Score fulfillment of stated preferences, ignoring unspecified support notes.

    This is a model profile score, not a human sensory-panel accuracy metric.
    It rewards the shape and coverage of requested dimensions and penalizes only
    dimensions the user explicitly asked to avoid.
    """
    desired_indices = [
        SCENT_DIMENSIONS.index(name)
        for name in desired_dimensions
        if name in SCENT_DIMENSIONS
    ]
    if not desired_indices:
        desired_indices = [index for index, value in enumerate(target) if value > 0]
    if not desired_indices:
        return 0.0

    target_slice = target[desired_indices]
    achieved_slice = achieved[desired_indices]
    shape = cosine_similarity_percent(target_slice, achieved_slice) / 100.0

    desired_mass = float(achieved_slice.sum())
    # A structurally complete perfume must devote material to top/heart/base
    # support even when the brief names only one or two facets.  Requiring most
    # of the normalized profile to sit in named facets incorrectly rejects
    # otherwise close formulas, so coverage is calibrated to that constraint.
    coverage_target = min(0.57, 0.32 + 0.025 * len(desired_indices))
    coverage = min(1.0, desired_mass / coverage_target)

    avoided_indices = [
        SCENT_DIMENSIONS.index(name)
        for name in avoided_dimensions
        if name in SCENT_DIMENSIONS
    ]
    avoided_mass = float(achieved[avoided_indices].sum()) if avoided_indices else 0.0
    avoidance = max(0.0, 1.0 - min(1.0, avoided_mass * 3.0))

    return max(
        0.0, min(100.0, (shape * 0.70 + coverage * 0.25 + avoidance * 0.05) * 100.0)
    )


def _project_capped_simplex(
    values: np.ndarray, caps: np.ndarray, total: float
) -> np.ndarray:
    if total < -1e-9 or float(caps.sum()) + 1e-7 < total:
        raise NoFeasibleFormula(
            f"농도 한도 합계 {caps.sum():.2f}%로 목표 {total:.2f}%를 채울 수 없습니다."
        )
    if total <= 0:
        return np.zeros_like(values)

    if math.isclose(float(caps.sum()), total, abs_tol=1e-10):
        return caps.copy()

    # The Euclidean projection is clip(values - theta, 0, caps).  Its sum is
    # piecewise linear with breakpoints at values-caps and values, so theta can
    # be solved exactly on one of at most 2N-1 intervals.  This replaces the
    # former 48-step bisection inside every optimizer iteration.
    breakpoints = np.unique(np.concatenate((values - caps, values)))
    result: np.ndarray | None = None
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        middle = (float(lower) + float(upper)) / 2.0
        capped = middle <= values - caps
        active = (~capped) & (middle < values)
        active_count = int(active.sum())
        if not active_count:
            continue
        theta = (
            float(caps[capped].sum()) + float(values[active].sum()) - total
        ) / active_count
        if float(lower) - 1e-12 <= theta <= float(upper) + 1e-12:
            result = np.clip(values - theta, 0.0, caps)
            break
    if result is None:  # Defensive numerical fallback for degenerate inputs.
        low = float(np.min(values - caps)) - total
        high = float(np.max(values)) + total
        for _ in range(48):
            middle = (low + high) / 2.0
            projected = np.clip(values - middle, 0.0, caps)
            if projected.sum() > total:
                low = middle
            else:
                high = middle
        result = np.clip(values - high, 0.0, caps)

    residual = total - float(result.sum())
    if residual > 1e-7:
        for index in np.argsort(-(caps - result)):
            addition = min(residual, float(caps[index] - result[index]))
            result[index] += addition
            residual -= addition
            if residual <= 1e-7:
                break
    elif residual < -1e-7:
        for index in np.argsort(-result):
            removal = min(-residual, float(result[index]))
            result[index] -= removal
            residual += removal
            if residual >= -1e-7:
                break
    return result


class ConstrainedFormulaOptimizer:
    def __init__(self, corpus: HistoricalReferenceCorpus | None = None):
        self.corpus = corpus or HistoricalReferenceCorpus()

    def _ingredient_score(
        self, ingredient: Ingredient, target: np.ndarray, max_price: float
    ) -> float:
        scent_match = cosine_similarity_percent(ingredient.vector(), target) / 100.0
        prevalence = min(1.0, self.corpus.frequency(ingredient.name) * 20.0)
        affordability = 1.0 - min(1.0, ingredient.price_per_kg / max(1.0, max_price))
        return (
            scent_match * 0.78
            + ingredient.availability * 0.12
            + affordability * 0.07
            + prevalence * 0.03
        )

    def _select_candidates(
        self, candidates: list[Ingredient], brief: ScentBrief
    ) -> list[Ingredient]:
        target = profile_vector(brief.target_profile)
        selected: list[Ingredient] = []
        per_group = max(3, brief.constraints.max_ingredients // 3)

        for pyramid, target_total in brief.pyramid_ratios.items():
            group = [item for item in candidates if item.pyramid == pyramid]
            ranked = sorted(
                group,
                key=lambda item: self._ingredient_score(
                    item, target, brief.constraints.max_ingredient_price_per_kg
                ),
                reverse=True,
            )
            chosen = ranked[:per_group]
            next_index = per_group
            while (
                sum(item.as_supplied_cap_percent() for item in chosen) + 1e-7
                < target_total
            ):
                if next_index >= len(ranked):
                    raise NoFeasibleFormula(
                        f"{pyramid} 노트의 안전 농도 용량이 부족합니다."
                    )
                chosen.append(ranked[next_index])
                next_index += 1
            selected.extend(chosen)

        # A multi-facet brief can contain a lower-volume but identity-defining
        # dimension (for example vanilla/gourmand beside clean musk). Pure
        # whole-vector ranking can omit that material entirely, so explicitly
        # retain one strong representative for every requested dimension.
        for dimension in brief.desired_dimensions:
            representatives = [
                item for item in candidates if item.profile.get(dimension, 0.0) >= 0.35
            ]
            if not representatives:
                continue
            representative = max(
                representatives,
                key=lambda item: (
                    item.profile.get(dimension, 0.0) * 0.70
                    + self._ingredient_score(
                        item, target, brief.constraints.max_ingredient_price_per_kg
                    )
                    * 0.30
                ),
            )
            if representative not in selected:
                selected.append(representative)

        if len(selected) > brief.constraints.max_ingredients:
            # Keep required pyramid capacity while removing the lowest scoring extras.
            removable = sorted(
                selected,
                key=lambda item: self._ingredient_score(
                    item, target, brief.constraints.max_ingredient_price_per_kg
                ),
            )
            for candidate in removable:
                if len(selected) <= brief.constraints.max_ingredients:
                    break
                uniquely_required = any(
                    candidate.profile.get(dimension, 0.0) >= 0.35
                    and not any(
                        other != candidate and other.profile.get(dimension, 0.0) >= 0.35
                        for other in selected
                    )
                    for dimension in brief.desired_dimensions
                )
                if uniquely_required:
                    continue
                same_group = [
                    item for item in selected if item.pyramid == candidate.pyramid
                ]
                required = brief.pyramid_ratios[candidate.pyramid]
                remaining_capacity = sum(
                    item.as_supplied_cap_percent()
                    for item in same_group
                    if item != candidate
                )
                if remaining_capacity >= required:
                    selected.remove(candidate)
        if len(selected) > brief.constraints.max_ingredients:
            raise NoFeasibleFormula(
                "요청한 향 구조와 안전 농도 상한을 max_ingredients 안에서 동시에 충족할 수 없습니다."
            )
        return selected

    @staticmethod
    def _achieved_profile(
        weights: np.ndarray,
        ingredients: list[Ingredient],
        perceptual_factors: dict[str, float] | None = None,
    ) -> np.ndarray:
        matrix = np.vstack(
            [
                item.vector()
                * (
                    perceptual_factors.get(item.ingredient_id, item.odor_impact)
                    if perceptual_factors is not None
                    else item.odor_impact
                )
                * item.active_strength_percent
                / 100.0
                for item in ingredients
            ]
        )
        raw = (weights / 100.0) @ matrix
        total = float(raw.sum())
        return raw / total if total > 0 else raw

    def optimize(
        self,
        candidates: list[Ingredient],
        brief: ScentBrief,
        perceptual_factors: dict[str, float] | None = None,
        formula_objective: Callable[[np.ndarray, list[Ingredient]], float]
        | None = None,
    ) -> tuple[list[RecipeLine], float, dict[str, float], float, float]:
        ingredients = self._select_candidates(candidates, brief)
        target = profile_vector(brief.target_profile)
        ingredient_vectors = np.vstack([item.vector() for item in ingredients])
        structural_factors = np.asarray(
            [
                item.odor_impact * item.active_strength_percent / 100.0
                for item in ingredients
            ]
        )
        perceptual_multiplier = np.asarray(
            [
                (
                    perceptual_factors.get(item.ingredient_id, item.odor_impact)
                    if perceptual_factors is not None
                    else item.odor_impact
                )
                * item.active_strength_percent
                / 100.0
                for item in ingredients
            ],
            dtype=float,
        )
        matrix = ingredient_vectors * perceptual_multiplier[:, None]
        structural_matrix = ingredient_vectors * structural_factors[:, None]
        caps = np.asarray(
            [item.as_supplied_cap_percent() for item in ingredients], dtype=float
        )
        prices = np.asarray([item.price_per_kg for item in ingredients], dtype=float)
        pyramid_indices = {
            pyramid: np.asarray(
                [
                    index
                    for index, item in enumerate(ingredients)
                    if item.pyramid == pyramid
                ],
                dtype=int,
            )
            for pyramid in brief.pyramid_ratios
        }

        def achieved_from(
            current: np.ndarray, source: np.ndarray = matrix
        ) -> np.ndarray:
            raw = (current / 100.0) @ source
            profile_total = float(raw.sum())
            return raw / profile_total if profile_total > 0 else raw

        pair_matrix = np.full((len(ingredients), len(ingredients)), 0.5, dtype=float)
        for left in range(len(ingredients)):
            for right in range(left + 1, len(ingredients)):
                support = self.corpus.pair_support(
                    ingredients[left].name, ingredients[right].name
                )
                pair_matrix[left, right] = support
                pair_matrix[right, left] = support

        def coherence(current: np.ndarray) -> float:
            pair_weights = np.outer(current, current)
            upper = np.triu(pair_weights * pair_matrix, 1)
            denominator = float(np.triu(pair_weights, 1).sum())
            return float(upper.sum() / denominator) if denominator else 0.5

        weights = np.zeros(len(ingredients), dtype=float)
        for pyramid, total in brief.pyramid_ratios.items():
            indices = pyramid_indices[pyramid]
            initial_scores = np.asarray(
                [
                    max(
                        0.01,
                        self._ingredient_score(
                            ingredients[index],
                            target,
                            brief.constraints.max_ingredient_price_per_kg,
                        ),
                    )
                    for index in indices
                ]
            )
            initial = initial_scores / initial_scores.sum() * total
            weights[indices] = _project_capped_simplex(initial, caps[indices], total)

        for _ in range(420):
            achieved = achieved_from(weights)
            error = achieved - target
            gradient = (matrix @ error) / 100.0
            cost_gradient = prices / max(
                1.0, brief.constraints.max_ingredient_price_per_kg
            )
            proposal = weights - 22.0 * gradient - 0.002 * cost_gradient
            for pyramid, total in brief.pyramid_ratios.items():
                indices = pyramid_indices[pyramid]
                proposal[indices] = _project_capped_simplex(
                    proposal[indices], caps[indices], total
                )
            if np.max(np.abs(proposal - weights)) < 1e-8:
                weights = proposal
                break
            weights = proposal

        def objective(current: np.ndarray) -> float:
            current_profile = achieved_from(current)
            semantic_score = semantic_brief_similarity(
                target,
                current_profile,
                brief.desired_dimensions,
                brief.avoided_dimensions,
            )
            score = semantic_score
            if formula_objective is not None:
                context_score = float(formula_objective(current, ingredients))
                if not math.isfinite(context_score):
                    raise ValueError("formula objective must return a finite score")
                context_score = max(0.0, min(100.0, context_score))
                context_weight = max(
                    0.0,
                    min(0.50, float(brief.constraints.surrogate_objective_weight)),
                )
                score = (
                    1.0 - context_weight
                ) * semantic_score + context_weight * context_score
            current_cost = float(np.dot(current / 100.0, prices))
            if current_cost > brief.constraints.max_formula_cost_per_kg:
                score -= (
                    current_cost - brief.constraints.max_formula_cost_per_kg
                ) * 0.1
            # Use historical co-occurrence only as a small plausibility tie
            # breaker. It must never overpower the explicit scent brief.
            return score + 1.5 * (coherence(current) - 0.5)

        # Coordinate refinement directly optimizes the user-facing semantic score.
        best_score = objective(weights)
        for step in (1.0, 0.25, 0.05):
            for _ in range(18):
                improved = False
                for pyramid in brief.pyramid_ratios:
                    indices = pyramid_indices[pyramid]
                    for donor in indices:
                        if weights[donor] < step:
                            continue
                        for receiver in indices:
                            if (
                                donor == receiver
                                or weights[receiver] + step > caps[receiver] + 1e-9
                            ):
                                continue
                            proposal = weights.copy()
                            proposal[donor] -= step
                            proposal[receiver] += step
                            score = objective(proposal)
                            if score > best_score + 1e-8:
                                weights = proposal
                                best_score = score
                                improved = True
                if not improved:
                    break

        # Final exact projection protects against accumulated floating-point drift.
        for pyramid, total in brief.pyramid_ratios.items():
            indices = pyramid_indices[pyramid]
            weights[indices] = _project_capped_simplex(
                weights[indices], caps[indices], total
            )

        # Optimize with headspace-aware gains, but report the established
        # structural semantic profile separately. The nonlinear scientific
        # twin supplies the physical score; mixing the two would make the
        # public semantic metric change meaning across model versions.
        achieved = achieved_from(weights, structural_matrix)
        similarity = semantic_brief_similarity(
            target,
            achieved,
            brief.desired_dimensions,
            brief.avoided_dimensions,
        )
        estimated_cost = float(np.dot(weights / 100.0, prices))

        batch_concentrate_mass_g = (
            brief.constraints.finished_batch_mass_g
            * brief.constraints.product_concentration_percent
            / 100.0
        )
        lines: list[RecipeLine] = []
        desired = set(brief.desired_dimensions)
        for weight, ingredient in sorted(
            zip(weights, ingredients), key=lambda pair: pair[0], reverse=True
        ):
            if weight < 0.000001:
                continue
            matching_dimensions = sorted(
                desired.intersection(
                    key for key, value in ingredient.profile.items() if value >= 0.35
                )
            )
            reason = (
                ", ".join(matching_dimensions[:3]) or f"{ingredient.pyramid} 구조 보완"
            )
            finished_percent = (
                weight * brief.constraints.product_concentration_percent / 100.0
            )
            mass_g = batch_concentrate_mass_g * weight / 100.0
            active_percent = weight * ingredient.active_strength_percent / 100.0
            active_mass_g = mass_g * ingredient.active_strength_percent / 100.0
            volume_ml = (
                mass_g / ingredient.density_g_ml
                if ingredient.density_g_ml is not None
                else None
            )
            lines.append(
                RecipeLine(
                    ingredient_id=ingredient.ingredient_id,
                    name=ingredient.name,
                    pyramid=ingredient.pyramid,
                    concentrate_percent=round(float(weight), 4),
                    finished_product_percent=round(float(finished_percent), 6),
                    volume_ml_for_batch=(
                        round(float(volume_ml), 4) if volume_ml else None
                    ),
                    price_per_kg=ingredient.price_per_kg,
                    availability=ingredient.availability,
                    risk_tier=ingredient.risk_tier,
                    reason=reason,
                    mass_g_for_batch=round(float(mass_g), 5),
                    active_material_percent=round(float(active_percent), 5),
                    active_mass_g_for_batch=round(float(active_mass_g), 5),
                    density_g_ml=ingredient.density_g_ml,
                    active_strength_percent=ingredient.active_strength_percent,
                    carrier=ingredient.carrier,
                    data_source=ingredient.data_source,
                    approved_formulation_scopes=ingredient.approved_formulation_scopes,
                    approval_expires_at=ingredient.approval_expires_at,
                    promotion_artifact_id=ingredient.promotion_artifact_id,
                )
            )

        support = self.corpus.recipe_support([line.name for line in lines])
        achieved_dict = {
            dimension: round(float(achieved[index]), 6)
            for index, dimension in enumerate(SCENT_DIMENSIONS)
            if achieved[index] >= 0.001
        }
        return lines, similarity, achieved_dict, estimated_cost, support
