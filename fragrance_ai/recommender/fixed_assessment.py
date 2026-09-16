"""Recompute a supplied R&D formula; never optimize or silently alter weights."""
from dataclasses import asdict, replace
from datetime import date
import math

import numpy as np

from .models import RecipeResult, SCENT_DIMENSIONS, normalize_profile
from .profile_match import assess_recipe_profiles, attach_profile_assessment
from .quality import formula_fingerprint
from .science import ScientificPropertyStore
from .adaptive_pyramid import actual_pyramid, explicit_pyramid
from .catalog import normalize_name


def assess_fixed_formula(ai, text, constraints, weights, *, as_of=None, target_profile_override=None, intent_controls=None):
    from .perception_runtime import assert_provider_current
    from .perception_guidance import attach_guidance
    provider = getattr(ai, 'perception_guidance', None)
    assert_provider_current(provider)
    guard = getattr(ai, "_runtime_snapshot_guard", None)
    if guard:
        guard()
    ai._validate_constraints(constraints)
    if constraints.validation_level != "prototype" or constraints.reference_target_id or constraints.experimental_disable_safety:
        raise ValueError("fixed-formula reassessment is safety-enabled prototype diagnostics only")
    if not isinstance(weights, dict) or not weights or len(weights) > constraints.max_ingredients:
        raise ValueError("fixed formula requires unique materials within the requested maximum")
    for key, value in weights.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("fixed percentages must be positive finite numbers")
    if abs(sum(weights.values()) - 100) > .001 + 1e-9:
        raise ValueError("fixed concentrate percentages must sum to 100 within the existing 0.001-point rendering tolerance; automatic normalization is disabled")
    as_of = as_of or date.today()
    brief = ai.parser.parse(text, constraints)
    if intent_controls:
        from .intent_controls import apply_intent_controls
        brief = apply_intent_controls(brief, intent_controls)
    minimum_target = 95. if ai.minimum_profile_target is None else ai.minimum_profile_target
    brief = replace(brief, constraints=replace(brief.constraints,
        target_similarity=max(minimum_target, brief.constraints.target_similarity)))
    ai._validate_constraints(brief.constraints)
    if len(weights) > brief.constraints.max_ingredients:
        raise ValueError("fixed formula exceeds the natural-language material limit")
    if target_profile_override is not None:
        if (not isinstance(target_profile_override, dict) or set(target_profile_override) - set(SCENT_DIMENSIONS)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in target_profile_override.values())
                or not math.isfinite(sum(target_profile_override.values())) or sum(target_profile_override.values()) <= 0):
            raise ValueError("invalid fixed-formula target profile")
        profile = normalize_profile(target_profile_override)
        brief = replace(brief, target_profile=profile, desired_dimensions=[key for key, value in profile.items() if value > 0])
    eligible, rejected = ai.screen.screen(ai.catalog, brief, ai.supplier_registry, as_of=as_of)
    by_id = {item.ingredient_id: item for item in eligible}
    if set(weights) - set(by_id):
        raise ValueError("fixed formula includes unknown, excluded or ineligible ingredients")
    if not {normalize_name(name) for name in brief.requested_ingredients}.issubset({normalize_name(by_id[key].name) for key in weights}):
        raise ValueError("fixed formula omits an explicitly requested ingredient")
    for key, value in weights.items():
        if value > by_id[key].as_supplied_cap_percent() + 1e-9:
            raise ValueError("fixed formula exceeds an ingredient concentration cap")
    selected = [by_id[key] for key in sorted(weights)]
    lines, _, nominal, cost, _ = ai.optimizer.variant_from_weights(selected, brief,
        np.array([weights[item.ingredient_id] for item in selected]), preserve_input_precision=True)
    if {line.ingredient_id: line.concentrate_percent for line in lines} != weights:
        raise ValueError("rendering would alter the submitted fixed weights")
    fixed_notes, _ = explicit_pyramid(text)
    actual_notes = actual_pyramid(lines)
    if any(abs(actual_notes[group] - value) > .001 for group, value in fixed_notes.items()):
        raise ValueError("fixed formula violates an explicit note percentage")
    if cost > brief.constraints.max_formula_cost_per_kg + 1e-6:
        raise ValueError("fixed formula exceeds the cost constraint")
    safety = ai.safety_gate.evaluate(lines, by_id, brief.constraints, as_of=as_of)
    if not safety.internal_gate_passed:
        raise ValueError("fixed formula failed the internal safety gate: " + "; ".join(safety.violations))
    properties = ScientificPropertyStore.with_catalog_structures(selected, ai.scientific_store.get_many(list(weights)))
    guidance_report = None
    if provider is not None:
        session = provider.begin(brief)
        prediction = session.evaluate_lines(lines, by_id)
        guidance_report = session.report(prediction, prediction, changed=False, variants=0)
        guidance_report.update(operation='fixed_formula_reassessment', weights_modified=False)
    twin = ai.temporal_simulator.evaluate(lines, by_id, brief, properties, draws=brief.constraints.simulation_draws)
    points = [asdict(point) for point in twin.temporal_points]
    assessment = assess_recipe_profiles(brief, nominal, points, ai.temporal_simulator.time_weights(brief))
    result = RecipeResult(status="fixed_formula_diagnostic", message="고정 배합의 과학 모델 점수를 재계산했습니다. 새 처방 승인이나 제조 승인이 아닙니다.",
        brief=brief, similarity_score=assessment["score"] or 0., similarity_kind="full_model_profile_agreement_not_human_similarity",
        recipe=[], closest_candidate=lines, achieved_profile=nominal, estimated_concentrate_cost_per_kg=cost,
        historical_support_score=0., catalog_stats=ai._catalog_stats(), rejected_candidate_counts=rejected, safety=safety,
        limitations=["Fixed weights were not optimized. Full generator eligibility and commercial release were not granted."],
        formula_id=formula_fingerprint(lines), scientific_twin_status=twin.status, scientific_model_version=twin.model_version,
        confidence=twin.confidence, temporal_similarity_score=twin.temporal_similarity_mean,
        minimum_temporal_similarity=twin.minimum_temporal_similarity,
        minimum_temporal_similarity_p05=twin.minimum_temporal_similarity_p05,
        scientific_data_coverage_percent=twin.scientific_data_coverage_percent,
        calculated_structure_coverage_percent=twin.calculated_structure_coverage_percent,
        molecular_descriptor_coverage_percent=twin.molecular_descriptor_coverage_percent,
        temporal_profile=points, ingredient_temporal_profile=[asdict(row) for row in twin.ingredient_temporal_profiles],
        temporal_timepoints_minutes=[point.minutes for point in twin.temporal_points],
        temporal_concentration_basis=twin.temporal_concentration_basis, temporal_model_claim_boundary=twin.temporal_model_claim_boundary,
        scientific_flags=list(twin.flags), vapor_pressure_coverage_percent=twin.vapor_pressure_coverage_percent,
        odor_threshold_coverage_percent=twin.odor_threshold_coverage_percent, model_applicability_percent=twin.model_applicability_percent,
        temporal_similarity_p05=twin.temporal_similarity_p05, temporal_similarity_p95=twin.temporal_similarity_p95,
        scientific_model_domain_passed=twin.model_domain_passed, scientific_uncertainty_kind=twin.uncertainty_kind,
        scientific_monte_carlo_draws=twin.monte_carlo_draws)
    result = attach_profile_assessment(result, assessment, strict=True)
    result.score_contract.update(effective_target=brief.constraints.target_similarity,
        assessment_kind="fixed_formula_scientific_reassessment", full_generator_approval=False,
        input_rounding_residual_percent=100. - sum(weights.values()))
    if guidance_report is not None:
        result = attach_guidance(result, guidance_report)
    assert_provider_current(provider)
    if guard:
        guard()
    return result
