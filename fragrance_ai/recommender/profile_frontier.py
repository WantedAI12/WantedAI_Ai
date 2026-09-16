"""Explain unreachable profile requests without weakening recipe constraints."""

from dataclasses import replace
import numpy as np


def explain_profile_frontier(items, brief, bound):
    target = float(brief.constraints.target_similarity)
    upper = bound.get("upper_score")
    report = {
        "version": "profile-frontier/v75",
        "requested_target": target,
        "current_upper_score": upper,
        "actual_request_or_material_limits_changed": False,
        "diagnostic_relaxations_are_not_recipe_proposals": True,
        "human_accuracy_bound": False,
    }
    if upper is None or bound.get("status") != "validated_fractional_cap_cost_upper":
        return {**report, "status": "no_certified_request_bound"}
    if upper + 1e-8 >= target:
        return {**report, "status": "not_ruled_out_not_guaranteed_feasible"}
    from .global_profile_search import optimize_full_pool
    from .models import MAX_FORMULA_INGREDIENTS, SCENT_DIMENSIONS

    vectors = np.array([item.vector() for item in items])
    maxima = vectors.max(0)
    report["desired_axis_catalogue_maxima"] = {
        name: float(maxima[i])
        for i, name in enumerate(SCENT_DIMENSIONS)
        if brief.target_profile.get(name, 0) > 0
    }
    report["minimum_model_score_shortfall"] = target - upper
    # Only a separate diagnostic problem is relaxed. Neither its weights nor
    # its modified records are returned to the recipe selection loop.
    diagnostic_items = [replace(item, max_concentrate_percent=100.0) for item in items]
    diagnostic_brief = replace(
        brief,
        constraints=replace(
            brief.constraints,
            max_ingredients=MAX_FORMULA_INGREDIENTS,
            max_formula_cost_per_kg=max(1.0, max(item.price_per_kg for item in items)),
        ),
    )
    solution = optimize_full_pool(
        diagnostic_items,
        diagnostic_brief,
        pyramid_bounds={group: (0.0, 100.0) for group in brief.pyramid_ratios},
    )
    relaxed = solution.certified_overlap_upper_score
    if relaxed is not None:
        relaxed = min(100.0, relaxed + bound.get("legacy_render_allowance_points", 0.0))
    report["without_dose_caps_and_budget_upper_score"] = relaxed
    report["status"] = (
        "profile_geometry_or_target_expression_bottleneck"
        if relaxed is not None and relaxed + 1e-8 < target
        else "limits_or_profile_geometry_not_fully_separated"
    )
    report["reference_profile_expansion_recommended"] = True
    report["requested_score_threshold_preserved"] = True
    return report
