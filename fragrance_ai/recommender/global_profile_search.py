"""Full-pool fractional-programming search over fixed ingredient profiles.

The LP maximizes a relaxation (overlap/avoidance, without cosine or mixture
physics). It proposes recipes; the unchanged complete V7 scorer and all
existing safety/evidence checks still decide whether a candidate can be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, hstack, vstack

from .models import Ingredient, PYRAMID_LEVELS, SCENT_DIMENSIONS, ScentBrief, profile_vector
from .linear_program_cache import cached_linprog


@dataclass
class PoolSolution:
    status: str
    weights_percent: dict[str, float] = field(default_factory=dict)
    relaxed_overlap_score: float | None = None
    pool_size: int = 0
    restricted_support: bool = False
    certified_overlap_upper_score: float | None = None


def fractional_dual_upper(solution, objective, inequality, limits, equality, equality_rhs, caps, gain, target):
    """Conservative Lagrangian bound, including finite transformed-variable boxes.

    A solver primal score alone is not an upper bound. Recompute a dual lower
    bound on min(-overlap), clipping inequality multipliers to their legal sign
    and bounding every reduced-cost residual over a known containing box.
    """
    positive = np.asarray(gain)[np.asarray(caps) > 0]
    if not len(positive) or np.any(positive <= 0):
        return None
    try:
        mu = np.minimum(np.asarray(solution.ineqlin.marginals, float), 0.)
        nu = np.asarray(solution.eqlin.marginals, float)
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(mu).all() or not np.isfinite(nu).all():
        return None
    residual = objective-inequality.T@mu-equality.T@nu
    scale_upper = 1./float(positive.min())
    upper = np.r_[np.asarray(caps)*scale_upper, scale_upper, np.asarray(target), 1.]
    terms = np.r_[mu*np.asarray(limits), nu*np.asarray(equality_rhs), np.minimum(residual, 0)*upper]
    if not np.isfinite(terms).all():
        return None
    lower = float(np.sum(terms, dtype=np.longdouble))
    allowance = 1e-7*(1+float(np.abs(terms).sum()))
    return min(100., max(0., -100*(lower-allowance)))


def profile_upper_bound(ingredients: list[Ingredient], target: dict[str, float]) -> dict:
    """Optimistic model bound even with unlimited material counts and caps.

    A positive weighted mixture is in the convex hull of ingredient vectors.
    Bound its overlap by both the mass on requested axes and coordinate maxima.
    Account conservatively for the legacy renderer dropping values < .001 and
    rounding retained values to 6 decimals; never use this as a human bound.
    """
    p = profile_vector(target)
    if not ingredients or p.sum() <= 0:
        return {"status": "undefined_no_pool_or_target", "upper_score": None,
                "actual_human_accuracy_bound": False, "candidate_count": len(ingredients)}
    matrix = np.asarray([item.vector() for item in ingredients])
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("invalid ingredient profile")
    maxima = matrix.max(axis=0)
    support = p > 0
    raw = min(1.0, float(matrix[:, support].sum(axis=1).max()), float(np.minimum(p, maxima).sum()))
    # Removal changes TV by at most the removed mass. Rounding adds at most
    # the sum of absolute rounding errors after normalization.
    rounding = len(SCENT_DIMENSIONS) * .5e-6
    render_allowance = 100 * (len(SCENT_DIMENSIONS) * .001 + rounding / (1 - rounding))
    return {
        "status": "optimistic_fixed_profile_convex_bound", "upper_score": min(100., 100 * raw + render_allowance),
        "unrounded_profile_upper_score": 100 * raw, "legacy_render_allowance_points": render_allowance,
        "candidate_count": len(ingredients), "bounds_include_caps_cost_pyramid": False,
        "upper_bound_is_achievable_score": False, "actual_human_accuracy_bound": False,
        "unsupported_dimensions": [d for d, value in zip(SCENT_DIMENSIONS, maxima) if value <= 0 and target.get(d, 0) > 0],
    }


def _fractional_lp(
    ingredients: list[Ingredient], brief: ScentBrief, *, factors: dict[str, float] | None = None,
    target: dict[str, float] | None = None, minimum_percent: dict[str, float] | None = None,
    pyramid_bounds: dict[str, tuple[float, float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
    diffusion_range: tuple[float, float] | None = None,
) -> PoolSolution:
    """Charnes-Cooper LP; physical weights are fractions summing to one."""
    n, dimensions = len(ingredients), len(SCENT_DIMENSIONS)
    def failed(status):
        return PoolSolution(status, pool_size=n)
    if not ingredients:
        return failed("empty_pool")
    if len({item.ingredient_id for item in ingredients}) != n:
        raise ValueError("duplicate candidate IDs")
    if any(value > 0 and key not in {item.ingredient_id for item in ingredients} for key, value in (minimum_percent or {}).items()):
        return failed("missing_required_material")
    if pyramid_bounds is not None:
        if set(pyramid_bounds) != set(PYRAMID_LEVELS) or any(
            len(band) != 2 or not all(math.isfinite(v) for v in band) or not 0 <= band[0] <= band[1] <= 100
            for band in pyramid_bounds.values()
        ):
            raise ValueError("invalid pyramid bounds")
        if sum(band[0] for band in pyramid_bounds.values()) > 100 + 1e-8 or sum(band[1] for band in pyramid_bounds.values()) < 100 - 1e-8:
            return failed("infeasible_pyramid_bounds")
    for band in (intensity_range, diffusion_range):
        if band is not None and (len(band) != 2 or not all(math.isfinite(v) for v in band) or not 0 <= band[0] <= band[1] <= 1):
            raise ValueError("invalid perceptual feature bounds")
    p = profile_vector(brief.target_profile if target is None else target)
    if p.sum() <= 0:
        return failed("undefined_target")
    gain = np.asarray([
        (factors.get(item.ingredient_id, item.odor_impact) if factors is not None else item.odor_impact)
        * item.active_strength_percent / 100.0 for item in ingredients
    ])
    vectors = np.asarray([item.vector() for item in ingredients])
    matrix = vectors * gain[:, None]
    caps = np.asarray([item.as_supplied_cap_percent() / 100. for item in ingredients])
    prices = np.asarray([item.price_per_kg for item in ingredients])
    lower = np.asarray([(minimum_percent or {}).get(item.ingredient_id, 0.) / 100. for item in ingredients])
    if any(not np.isfinite(value).all() for value in (matrix, caps, prices, lower)) or np.any(matrix < 0) or np.any(lower < 0):
        raise ValueError("LP requires finite, nonnegative profiles and limits")
    if np.any(lower > caps) or matrix.sum() <= 0:
        return failed("infeasible_limits")
    # Variables: y = w / (sum(matrix @ w)), t = 1 / sum(matrix @ w),
    # z = coordinate overlaps, u = minimum(overlap, avoidance).
    width = n + 1 + dimensions + 1
    eq_rows, eq_rhs = [], []
    eq_rows.append(hstack([csr_matrix(matrix.sum(axis=1).reshape(1, -1)), csr_matrix((1, dimensions + 2))]))
    eq_rhs.append(1.)
    if pyramid_bounds is None:
        for group, percent in brief.pyramid_ratios.items():
            indicators = np.asarray([item.pyramid == group for item in ingredients], dtype=float)
            eq_rows.append(hstack([csr_matrix(indicators.reshape(1, -1)), csr_matrix([[-percent / 100.]]), csr_matrix((1, dimensions + 1))]))
            eq_rhs.append(0.)
    else:
        # Bounds no longer imply mass conservation: sum(y) must equal t.
        eq_rows.append(hstack([csr_matrix(np.ones((1, n))), csr_matrix([[-1.]]), csr_matrix((1, dimensions + 1))]))
        eq_rhs.append(0.)
    ub_rows = [
        hstack([diags(np.ones(n)), csr_matrix(-caps.reshape(-1, 1)), csr_matrix((n, dimensions + 1))]),
        hstack([csr_matrix(-matrix.T), csr_matrix((dimensions, 1)), diags(np.ones(dimensions)), csr_matrix((dimensions, 1))]),
        csr_matrix([[*([0.] * (n + 1)), *([-1.] * dimensions), 1.]]),
        hstack([csr_matrix(prices.reshape(1, -1)), csr_matrix([[-brief.constraints.max_formula_cost_per_kg]]), csr_matrix((1, dimensions + 1))]),
    ]
    rhs = [np.zeros(n), np.zeros(dimensions), np.zeros(1), np.zeros(1)]

    def linear_band(coefficients, low, high):
        if low > 0:
            ub_rows.append(hstack([csr_matrix(-np.asarray(coefficients).reshape(1, -1)), csr_matrix([[low]]), csr_matrix((1, dimensions + 1))]))
            rhs.append(np.zeros(1))
        if high is not None:
            ub_rows.append(hstack([csr_matrix(np.asarray(coefficients).reshape(1, -1)), csr_matrix([[-high]]), csr_matrix((1, dimensions + 1))]))
            rhs.append(np.zeros(1))

    if pyramid_bounds is not None:
        for group, (low, high) in pyramid_bounds.items():
            linear_band([float(item.pyramid == group) for item in ingredients], low / 100, high / 100)
    if intensity_range is not None:
        # This is the existing intensity heuristic, not the headspace gain
        # used to propose scent-profile weights. Saturation at 1 has no upper
        # linear constraint when the requested range includes 1.
        coefficients = [item.odor_impact * item.active_strength_percent / 100 / 2.5 for item in ingredients]
        linear_band(coefficients, intensity_range[0], intensity_range[1] if intensity_range[1] < 1 else None)
    if diffusion_range is not None:
        linear_band([{"top": .9, "heart": .55, "base": .2}[item.pyramid] for item in ingredients], *diffusion_range)
    avoided = [SCENT_DIMENSIONS.index(d) for d in set(brief.avoided_dimensions)]
    if avoided:
        ub_rows.append(hstack([csr_matrix(matrix[:, avoided].sum(axis=1).reshape(1, -1)), csr_matrix((1, dimensions + 1)), csr_matrix([[1.]])]))
        rhs.append(np.ones(1))
    positive_lower = np.flatnonzero(lower > 0)
    if positive_lower.size:
        idx = np.arange(positive_lower.size)
        row = csr_matrix((-np.ones(len(idx)), (idx, positive_lower)), shape=(len(idx), n))
        ub_rows.append(hstack([row, csr_matrix(lower[positive_lower].reshape(-1, 1)), csr_matrix((len(idx), dimensions + 1))]))
        rhs.append(np.zeros(len(idx)))
    objective = np.zeros(width)
    objective[-1] = -1.
    bounds = [(0., None)] * (n + 1) + [(0., float(value)) for value in p] + [(0., 1.)]
    solution = cached_linprog(
        linprog, objective, A_ub=vstack(ub_rows, format="csr"), b_ub=np.concatenate(rhs),
        A_eq=vstack(eq_rows, format="csr"), b_eq=np.asarray(eq_rhs), bounds=bounds,
        method="highs-ds" if n > 1024 else "highs",
        # Dense presolve reductions on the 29k-column experimental registry
        # can overrun HiGHS' solve time limit. Keep large systems sparse.
        options={"time_limit": 2., "presolve": n <= 1024,
                 "primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8},
    )
    if not solution.success or solution.x is None:
        return failed({1: "solver_limit", 2: "infeasible_constraints"}.get(solution.status, "solver_failed"))
    certified_upper = fractional_dual_upper(solution, objective, vstack(ub_rows, format='csr'), np.concatenate(rhs),
        vstack(eq_rows, format='csr'), np.asarray(eq_rhs), caps, matrix.sum(axis=1), p)
    # Lexicographic fine-identity refinement. It can resolve different odors
    # sharing one coarse vector, but cannot trade away the original LP optimum.
    from .odor_expression import expression_utility
    fine, _ = expression_utility(ingredients, brief)
    if np.ptp(fine)>1e-12:
        floor = float(solution.x[-1])-1e-9
        preference = np.r_[-fine*gain, np.zeros(dimensions+2)]
        extra = csr_matrix(([ -1. ], ([0],[width-1])),shape=(1,width))
        refined = cached_linprog(linprog, preference, A_ub=vstack([*ub_rows,extra],format='csr'),
            b_ub=np.r_[np.concatenate(rhs),-floor], A_eq=vstack(eq_rows,format='csr'),
            b_eq=np.asarray(eq_rhs),bounds=bounds,method='highs-ds',
            options={'time_limit':2.,'presolve':n<=1024})
        if refined.success and refined.x is not None and refined.x[-1]>=floor-1e-9:
            solution = refined
    scale = float(solution.x[n])
    if not math.isfinite(scale) or scale <= 0:
        return failed("invalid_solver_scale")
    weights = np.maximum(0., solution.x[:n] / scale)
    group_totals = {group: sum(weights[i] for i, item in enumerate(ingredients) if item.pyramid == group) * 100
                    for group in PYRAMID_LEVELS}
    group_ok = (all(abs(group_totals[group] - percent) <= 1e-5 for group, percent in brief.pyramid_ratios.items())
                if pyramid_bounds is None else all(low - 1e-5 <= group_totals[group] <= high + 1e-5
                                                  for group, (low, high) in pyramid_bounds.items()))
    actual_intensity = min(1., sum(weights[i] * item.odor_impact * item.active_strength_percent / 100 / 2.5 for i, item in enumerate(ingredients)))
    actual_diffusion = sum(weights[i] * {"top": .9, "heart": .55, "base": .2}[item.pyramid] for i, item in enumerate(ingredients))
    features_ok = all(band is None or band[0] - 1e-7 <= value <= band[1] + 1e-7
                      for band, value in ((intensity_range, actual_intensity), (diffusion_range, actual_diffusion)))
    if (not np.isfinite(weights).all() or np.any(weights > caps + 1e-7)
            or np.any(weights < lower - 1e-7) or abs(float(weights.sum()) - 1) > 1e-7
            or float(weights @ prices) > brief.constraints.max_formula_cost_per_kg + 1e-6
            or not group_ok or not features_ok):
        return failed("solver_primal_check_failed")
    return PoolSolution(
        "relaxed_profile_optimum", {item.ingredient_id: float(weight * 100) for item, weight in zip(ingredients, weights) if weight > 1e-10},
        float(np.clip(solution.x[-1] * 100, 0, 100)), n,
        certified_overlap_upper_score=certified_upper,
    )


def optimize_full_pool(
    ingredients: list[Ingredient], brief: ScentBrief, *, factors: dict[str, float] | None = None,
    target: dict[str, float] | None = None, minimum_percent: dict[str, float] | None = None,
    pyramid_bounds: dict[str, tuple[float, float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
    diffusion_range: tuple[float, float] | None = None,
) -> PoolSolution:
    """Search the entire pool, then re-solve any cardinality-limited support.

    The unrestricted LP is not a claim that its sparse support obeys the
    caller's material count. Failed restricted problems return no candidate.
    """
    ingredients = sorted(ingredients, key=lambda item: item.ingredient_id)
    options = dict(factors=factors, target=target, minimum_percent=minimum_percent, pyramid_bounds=pyramid_bounds,
                   intensity_range=intensity_range, diffusion_range=diffusion_range)
    result = _fractional_lp(ingredients, brief, **options)
    if not result.weights_percent or len(result.weights_percent) <= brief.constraints.max_ingredients:
        return result
    required = {key for key, value in (minimum_percent or {}).items() if value > 0}
    ranked = sorted(ingredients, key=lambda item: (-result.weights_percent.get(item.ingredient_id, 0.), item.ingredient_id))
    orders = [
        ranked,
        sorted(ranked, key=lambda item: (-item.as_supplied_cap_percent(), -result.weights_percent.get(item.ingredient_id, 0.), item.ingredient_id)),
        sorted(ranked, key=lambda item: (item.price_per_kg, -item.as_supplied_cap_percent(), item.ingredient_id)),
    ]
    restricted_solutions, seen_supports = [], set()
    actual_capacity = {group: sum(result.weights_percent.get(item.ingredient_id, 0.) for item in ingredients if item.pyramid == group)
                       for group in PYRAMID_LEVELS}
    capacity_targets = [brief.pyramid_ratios] if pyramid_bounds is None else [actual_capacity, {group: band[0] for group, band in pyramid_bounds.items()}]
    for order, required_capacity in ((order, capacity) for capacity in capacity_targets for order in orders):
        chosen = [item for item in ranked if item.ingredient_id in required]
        # A low-cap LP support may need too many materials even though a
        # high-cap alternative fits the same scent within the count limit.
        for group, percent in required_capacity.items():
            for item in order:
                if sum(row.as_supplied_cap_percent() for row in chosen if row.pyramid == group) + 1e-7 >= percent:
                    break
                if item.pyramid == group and item not in chosen:
                    chosen.append(item)
        if len(chosen) > brief.constraints.max_ingredients:
            continue
        for item in ranked:
            if len(chosen) >= brief.constraints.max_ingredients:
                break
            if item not in chosen and result.weights_percent.get(item.ingredient_id, 0.) > 0:
                chosen.append(item)
        identity = tuple(sorted(item.ingredient_id for item in chosen))
        if identity in seen_supports:
            continue
        seen_supports.add(identity)
        restricted = _fractional_lp(chosen, brief, **options)
        if restricted.weights_percent:
            restricted.pool_size = len(ingredients)
            restricted.restricted_support = True
            restricted.certified_overlap_upper_score = result.certified_overlap_upper_score
            restricted_solutions.append(restricted)
    if restricted_solutions:
        return max(restricted_solutions, key=lambda row: row.relaxed_overlap_score)
    # Failure of a bounded support search is not a proof of infeasibility.
    return PoolSolution("cardinality_support_search_exhausted", relaxed_overlap_score=result.relaxed_overlap_score,
                        pool_size=len(ingredients), restricted_support=True,
                        certified_overlap_upper_score=result.certified_overlap_upper_score)
