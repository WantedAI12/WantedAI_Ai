"""Explicit structured controls; never round-trip edits through prose."""
from dataclasses import replace
import math
from .models import SCENT_DIMENSIONS, normalize_profile


def normalized_target(profile, *, avoided=()):
    if (not isinstance(profile, dict) or not profile or set(profile)-set(SCENT_DIMENSIONS)
            or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0 for v in profile.values())
            or not math.isfinite(sum(profile.values())) or sum(profile.values()) <= 0):
        raise ValueError('target profile requires known axes and finite positive weights')
    if any(profile.get(axis, 0.) > 0 for axis in avoided):
        raise ValueError('target profile conflicts with an explicit avoidance constraint')
    return normalize_profile(profile)


def target_controls(request):
    return {key: getattr(request, key) for key in ('target_profile','phase_target_profiles')
            if getattr(request, key, None) is not None}


def effective_phase_target(brief, phase):
    """A prohibition-only phase inherits only the remaining positive intent.

    Never invent a new positive axis when all inherited axes were prohibited.
    Explicit positive phase profiles are retained and checked for conflicts.
    """
    avoided = set(brief.avoided_dimensions) | set(brief.phase_avoided_dimensions.get(phase, []))
    explicit = brief.phase_target_profiles.get(phase)
    if explicit and sum(explicit.values()) > 0:
        return normalized_target(explicit, avoided=avoided)
    inherited = {k:v for k,v in brief.target_profile.items() if k not in avoided}
    if sum(inherited.values()) <= 0:
        raise ValueError('phase avoidance removes every positive target; provide a positive phase profile')
    return normalized_target(inherited)


def representation_contract(brief, supported_axes=None):
    from .odor_expression import brief_expression
    requested = set(brief.desired_dimensions) | set(brief.avoided_dimensions)
    for profiles in (brief.phase_target_profiles,):
        requested.update(k for p in profiles.values() for k,v in p.items() if v > 0)
    requested.update(k for values in brief.phase_avoided_dimensions.values() for k in values)
    supported = None if supported_axes is None else set(supported_axes)
    result = {'version': 'explicit-target-representation/v1', 'axes': list(SCENT_DIMENSIONS),
        'source': brief.target_profile_source, 'weight_semantics': 'normalized_model_profile_not_human_intensity',
        'zero_target_axis_semantics': 'zero_desired_model_mass_not_an_explicit_user_ban',
        'unrequested_axes_remain_in_strict_comparison': True,
        'explicit_avoidance_is_separate': True, 'normalization_removes_absolute_intensity': True,
        'learned_supported_axes': sorted(supported) if supported is not None else None,
        'learned_unmodeled_requested_axes': sorted(requested-supported) if supported is not None else None,
        'learned_coverage_is_not_primary_19_axis_score': True,
        'fine_expression':brief_expression(brief), 'human_similarity_percent': None}
    from .odor_space import configured_odor_space
    space = configured_odor_space()
    if space is not None and brief.target_profile_source != 'explicit_structured_relative_weights':
        from .lotion_reference_objective import load_configured_reference_bank
        from .odor_space import target_report
        bank = load_configured_reference_bank()
        if bank is not None and bank.odor_space is not None:
            rows = [{'phase':'overall','target_profile':brief.target_profile,'avoided':brief.avoided_dimensions}]
            result.update(hierarchical_target=target_report(bank,brief,rows), legacy_projection_only=True,
                zero_target_axis_semantics='legacy_display_zero_not_unmentioned_reference_absence',
                unrequested_axes_remain_in_strict_comparison=False)
    return result


def apply_intent_controls(brief, controls):
    if not controls:
        return brief
    if not isinstance(controls, dict) or set(controls) - {"intensity_level", "phase_target_profiles", "target_profile"}:
        raise ValueError("unknown structured intent control")
    if brief.constraints.validation_level != "prototype":
        raise ValueError("structured intent controls currently support prototype requests only")
    changes = {}
    if 'target_profile' in controls:
        profile = normalized_target(controls['target_profile'], avoided=brief.avoided_dimensions)
        changes.update(target_profile=profile, desired_dimensions=[k for k,v in profile.items() if v > 0],
                       target_profile_source='explicit_structured_relative_weights')
    if "intensity_level" in controls:
        level = controls["intensity_level"]
        if type(level) is not int or not 1 <= level <= 5:
            raise ValueError("intensity_level must be an integer from 1 to 5")
        changes.update(absolute_intensity_target=(level - 1) / 4,
                       intensity="low" if level < 3 else "high" if level > 3 else "medium")
    if "phase_target_profiles" in controls:
        profiles = controls["phase_target_profiles"]
        if not isinstance(profiles, dict) or not profiles or set(profiles) - {"opening", "heart", "drydown"}:
            raise ValueError("phase_target_profiles requires opening, heart or drydown keys")
        targets, desired = dict(brief.phase_target_profiles), dict(brief.phase_desired_dimensions)
        for phase, profile in profiles.items():
            if (not isinstance(profile, dict) or not profile or set(profile) - set(SCENT_DIMENSIONS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in profile.values())
                    or not math.isfinite(sum(profile.values())) or sum(profile.values()) <= 0):
                raise ValueError("each phase requires a finite positive scent profile")
            avoided = set(brief.avoided_dimensions) | set(brief.phase_avoided_dimensions.get(phase, []))
            if any(profile.get(axis, 0.) > 0 for axis in avoided):
                raise ValueError("phase edit conflicts with an explicit avoidance constraint")
            targets[phase] = normalize_profile(profile)
            desired[phase] = [axis for axis, value in targets[phase].items() if value > 0]
        changes.update(phase_target_profiles=targets, phase_desired_dimensions=desired)
    return replace(brief, **changes)
