"""Engineering note-allocation policy and conservative model-response guards.

The 10..80% automatic bands are search limits, not a universal perfumery
standard. User-specified percentages are exact constraints, never rescaled.
"""

from __future__ import annotations

import math
import re

from .models import PYRAMID_LEVELS

LABELS = {
    "top": r"(?:\btop(?:\s+notes?)?\b|탑\s*(?:노트)?)",
    "heart": r"(?:\b(?:heart|middle)(?:\s+notes?)?\b|(?:미들|하트)\s*(?:노트)?)",
    "base": r"(?:\bbase(?:\s+notes?)?\b|베이스\s*(?:노트)?)",
}
NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf(?:inity)?))"
PERCENTAGES = {group: re.compile(label + r"\s*[:=]?\s*" + NUMBER + r"\s*%", re.I)
               for group, label in LABELS.items()}
RATIO = re.compile(r"(?:top\s*/\s*(?:heart|middle)\s*/\s*base|탑\s*/\s*(?:미들|하트)\s*/\s*베이스)\s*[:=]?\s*"
                   + NUMBER + r"\s*/\s*" + NUMBER + r"\s*/\s*" + NUMBER, re.I)
DIFFUSION = {"top": .90, "heart": .55, "base": .20}


def explicit_pyramid(text: str) -> tuple[dict[str, float], set[tuple[int, int]]]:
    fixed, spans = {}, set()

    def add(group, raw):
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("노트 비율은 0~100%의 유한한 값이어야 합니다.")
        if group in fixed and abs(fixed[group] - value) > 1e-8:
            raise ValueError(f"{group} 노트 비율이 서로 충돌합니다.")
        fixed[group] = value

    for group, pattern in PERCENTAGES.items():
        for match in pattern.finditer(text):
            add(group, match.group(1))
            spans.add(match.span())
    for match in RATIO.finditer(text):
        for group, raw in zip(PYRAMID_LEVELS, match.groups()):
            add(group, raw)
        spans.add(match.span())
    total = sum(fixed.values())
    if total > 100 + 1e-8 or (len(fixed) == 3 and abs(total - 100) > 1e-8):
        raise ValueError("명시한 노트 비율이 100% 배합과 맞지 않습니다.")
    return fixed, spans


def apply_explicit_pyramid(inferred: dict[str, float], fixed: dict[str, float]) -> dict[str, float]:
    if not fixed:
        return dict(inferred)
    remaining = [group for group in PYRAMID_LEVELS if group not in fixed]
    result = dict(fixed)
    amount = 100 - sum(fixed.values())
    source_total = sum(inferred[group] for group in remaining)
    for group in remaining:
        result[group] = amount * (inferred[group] / source_total if source_total > 0 else 1 / len(remaining))
    return {group: result[group] for group in PYRAMID_LEVELS}


def actual_pyramid(lines) -> dict[str, float]:
    return {group: sum(line.concentrate_percent for line in lines if line.pyramid == group) for group in PYRAMID_LEVELS}


def blended_proposals(baseline_lines, proposed: dict[str, float], max_ingredients: int, modeled_ids: set[str]):
    """Convex, mass-preserving steps; never drop materials to fake a count cap."""
    baseline = {line.ingredient_id: line.concentrate_percent for line in baseline_lines if line.concentrate_percent > 0}
    identifiers = sorted(set(baseline) | set(proposed))
    can_blend = len(identifiers) <= max_ingredients and set(identifiers).issubset(modeled_ids)
    for fraction in ((.25, .5, .75, 1.) if can_blend else (1.,)):
        weights = {key: (1 - fraction) * baseline.get(key, 0.) + fraction * proposed.get(key, 0.) for key in identifiers}
        weights = {key: value for key, value in weights.items() if value > 1e-8}
        if len(weights) <= max_ingredients:
            yield fraction, weights


def intensity_and_diffusion(lines, ingredients) -> tuple[float, float]:
    impact = sum(line.concentrate_percent / 100 * ingredients[line.ingredient_id].odor_impact
                 * ingredients[line.ingredient_id].active_strength_percent / 100 for line in lines)
    diffusion = sum(line.concentrate_percent / 100 * DIFFUSION[line.pyramid] for line in lines)
    return min(1., impact / 2.5), diffusion


def prepare_adaptive_policy(brief, baseline_lines, ingredients) -> dict:
    fixed, _ = explicit_pyramid(brief.original_text)
    bands = {group: (fixed[group], fixed[group]) if group in fixed else
             (min(10., brief.pyramid_ratios[group]), max(80., brief.pyramid_ratios[group])) for group in PYRAMID_LEVELS}
    intensity, diffusion = intensity_and_diffusion(baseline_lines, ingredients)
    text = brief.original_text.casefold()
    explicit_intensity = brief.intensity != "medium" or bool(re.search(r"(?:강도|intensity)\s*\d", text))
    explicit_diffusion = abs(brief.diffusion_target - .5) > 1e-8

    def band(target, observed):
        error = abs(target - observed)
        return (max(0., target - error), min(1., target + error))

    return {
        "bounds": bands, "explicit_percentages": fixed, "inferred_pyramid": dict(brief.pyramid_ratios),
        "intensity_range": band(brief.absolute_intensity_target, intensity) if explicit_intensity else None,
        "diffusion_range": band(brief.diffusion_target, diffusion) if explicit_diffusion else None,
        "baseline_intensity": intensity, "baseline_diffusion": diffusion,
        "long_lasting": bool(re.search(r"오래\s*지속|지속력|롱래스팅|long[- ]?lasting|long[- ]lasts", text)),
        "signal_retention_floor": .90,
        "claim_boundary": "model-only preservation policy; not calibrated human intensity or duration",
    }


def check_adaptive_response(brief, baseline_assessment, assessment, baseline_twin, twin, lines, ingredients, policy) -> list[str]:
    """Reject improvement obtained by losing requested phases or odor signal."""
    violations = []
    if baseline_assessment["nominal"]["target_profile"] != assessment["nominal"]["target_profile"]:
        return ["odor_target_changed"]
    old_avoid = baseline_assessment["nominal"]["avoided_mass"]
    new_avoid = assessment["nominal"]["avoided_mass"]
    if old_avoid is not None and new_avoid is not None and new_avoid > old_avoid + 1e-6:
        violations.append("nominal_avoidance_regressed")
    intensity, diffusion = intensity_and_diffusion(lines, ingredients)
    for name, value in (("intensity", intensity), ("diffusion", diffusion)):
        interval = policy[name + "_range"]
        if interval is not None and not interval[0] - 1e-5 <= value <= interval[1] + 1e-5:
            violations.append(name + "_request_regressed")
    before, after = baseline_assessment["temporal"], assessment["temporal"]
    if len(before) != len(after) or len(before) != len(twin.temporal_points) or len(before) != len(baseline_twin.temporal_points):
        return [*violations, "time_grid_mismatch"]
    if not before:
        return [*violations, "missing_temporal_response"]

    def below(value, limit):
        return value < limit and not math.isclose(value, limit, rel_tol=1e-9, abs_tol=0.)
    positive = []
    for index, (left, right, left_twin, right_twin) in enumerate(zip(before, after, baseline_twin.temporal_points, twin.temporal_points)):
        if left["minutes"] != right["minutes"] or left["phase"] != right["phase"] or left["weight"] != right["weight"]:
            return [*violations, "time_grid_mismatch"]
        if left["target_profile"] != right["target_profile"]:
            return [*violations, "odor_target_changed"]
        if left["weight"] <= 0:
            continue
        positive.append(index)
        signal = right_twin.total_relative_intensity
        if left["score"] is None or right["score"] is None:
            return [*violations, "undefined_time_profile"]
        if (not math.isfinite(signal) or not math.isfinite(left_twin.total_relative_intensity) or signal <= 0
                or below(signal, policy["signal_retention_floor"] * left_twin.total_relative_intensity)):
            violations.append("temporal_odor_signal_lost")
        if left["phase"] in brief.phase_target_profiles and (right["score"] is None or left["score"] is None or right["score"] + 1e-6 < left["score"]):
            violations.append("explicit_phase_profile_regressed")
        if right["avoided_mass"] is not None and left["avoided_mass"] is not None and right["avoided_mass"] > left["avoided_mass"] + 1e-6:
            violations.append("phase_avoidance_regressed")
    if positive:
        old_min = min(before[i]["score"] for i in positive)
        new_min = min(after[i]["score"] for i in positive)
        if new_min + 1e-6 < old_min:
            violations.append("worst_time_profile_regressed")
    if policy["long_lasting"]:
        # Explicit persistence remains a requirement even if phase-specific
        # scent weights assign zero to the final modeled time point.
        final = max(range(len(before)), key=lambda i: before[i]["minutes"])
        left, right = baseline_twin.temporal_points[final], twin.temporal_points[final]
        values = (left.total_relative_intensity, right.total_relative_intensity,
                  left.relative_to_opening_intensity_percent, right.relative_to_opening_intensity_percent)
        if (not all(math.isfinite(value) for value in values) or right.total_relative_intensity <= 0
                or below(right.total_relative_intensity, left.total_relative_intensity)
                or below(right.relative_to_opening_intensity_percent, left.relative_to_opening_intensity_percent)):
            violations.append("requested_persistence_regressed")
    return sorted(set(violations))
