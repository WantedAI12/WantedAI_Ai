"""Engineering plausibility checks for more realistic smelling formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .catalog import HistoricalReferenceCorpus
from .models import Ingredient, RecipeLine, ScentBrief, SCENT_DIMENSIONS, profile_vector
from .optimizer import semantic_brief_similarity


@dataclass(frozen=True)
class RealismAssessment:
    score: float
    kind: str
    accord_family: str
    components: dict[str, float]
    flags: tuple[str, ...]
    confidence: str
    observed_profile_coverage_percent: float


def infer_accord_family(brief: ScentBrief) -> str:
    desired = set(brief.desired_dimensions)
    if {"citrus", "fresh"}.issubset(desired):
        return "fresh_citrus"
    if {"clean", "musky"}.issubset(desired):
        return "clean_musk"
    if {"aquatic", "fresh"}.issubset(desired):
        return "aquatic_fresh"
    if {"white_floral", "floral"}.issubset(desired):
        return "white_floral"
    if {"rose", "floral"}.issubset(desired):
        return "rose_floral"
    if {"woody", "amber"}.issubset(desired):
        return "woody_amber"
    if {"gourmand", "fruity"}.issubset(desired):
        return "fruity_gourmand"
    if {"green", "aromatic"}.issubset(desired):
        return "green_aromatic"
    if desired:
        return "_".join(sorted(desired)[:2])
    return "unknown"


def weighted_pair_support(
    lines: Iterable[RecipeLine],
    ingredients: dict[str, Ingredient],
    corpus: HistoricalReferenceCorpus,
) -> float:
    rows = list(lines)
    if len(rows) < 2:
        return 0.5
    weighted_sum = 0.0
    total_weight = 0.0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            weight = max(0.0, left.concentrate_percent * right.concentrate_percent)
            weighted_sum += weight * corpus.pair_support(left.name, right.name)
            total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.5


def weighted_pair_support_from_arrays(
    weights: np.ndarray,
    ingredients: list[Ingredient],
    corpus: HistoricalReferenceCorpus,
) -> float:
    total = 0.0
    score = 0.0
    for i, left in enumerate(ingredients):
        for j in range(i + 1, len(ingredients)):
            weight = float(max(0.0, weights[i] * weights[j]))
            total += weight
            score += weight * corpus.pair_support(left.name, ingredients[j].name)
    return score / total if total else 0.5


def assess_realism(
    lines: list[RecipeLine],
    ingredients: dict[str, Ingredient],
    brief: ScentBrief,
    corpus: HistoricalReferenceCorpus,
) -> RealismAssessment:
    target = profile_vector(brief.target_profile)
    achieved = np.zeros(len(SCENT_DIMENSIONS), dtype=float)
    impact_values: list[float] = []
    for line in lines:
        ingredient = ingredients[line.ingredient_id]
        contribution = (
            line.concentrate_percent
            * ingredient.odor_impact
            * ingredient.active_strength_percent
            / 100.0
        )
        achieved += contribution * ingredient.vector()
        impact_values.append(max(0.0, contribution))
    if achieved.sum() > 0:
        achieved /= achieved.sum()
    semantic = semantic_brief_similarity(
        target, achieved, brief.desired_dimensions, brief.avoided_dimensions
    )
    coherence = weighted_pair_support(lines, ingredients, corpus) * 100.0
    total_impact = sum(impact_values)
    max_share = max(impact_values, default=0.0) / max(1e-9, total_impact)
    balance = 100.0
    if max_share > 0.65:
        balance -= min(70.0, (max_share - 0.65) * 220.0)
    elif max_share < 0.08 and len(lines) > 8:
        balance -= 15.0
    if not 4 <= len(lines) <= 10:
        balance -= 8.0
    balance = max(0.0, balance)
    sourcing = sum(item.availability for item in ingredients.values() if item.ingredient_id in {line.ingredient_id for line in lines}) / max(1, len(lines)) * 100.0
    observed = sum(
        1 for line in lines if ingredients[line.ingredient_id].data_source.startswith("odor-observed:")
    )
    observed_coverage = observed / max(1, len(lines)) * 100.0
    evidence = 35.0 + observed_coverage * 0.65
    score = semantic * 0.55 + coherence * 0.20 + balance * 0.15 + sourcing * 0.05 + evidence * 0.05
    flags: list[str] = []
    if observed == 0:
        flags.append("원료 실측 관능 프로필이 없어 큐레이션 프로필을 사용했습니다.")
    elif observed < len(lines):
        flags.append("일부 원료만 실측 관능 프로필로 보정되었습니다.")
    if coherence < 35.0:
        flags.append("과거 조합 참고 DB에서 낮은 동시출현 조합입니다.")
    if max_share > 0.65:
        flags.append("고충격 원료 하나의 지배 가능성이 높아 시향 확인이 필요합니다.")
    confidence = "observed_profile_supported" if observed_coverage >= 70 else (
        "mixed_profile" if observed_coverage > 0 else "heuristic_only"
    )
    return RealismAssessment(
        score=round(max(0.0, min(100.0, score)), 4),
        kind="engineering_plausibility_not_sensory_accuracy",
        accord_family=infer_accord_family(brief),
        components={
            "semantic_profile": round(semantic, 4),
            "historical_pair_coherence": round(coherence, 4),
            "impact_balance": round(balance, 4),
            "sourcing_plausibility": round(sourcing, 4),
            "odor_profile_evidence": round(evidence, 4),
        },
        flags=tuple(flags),
        confidence=confidence,
        observed_profile_coverage_percent=round(observed_coverage, 4),
    )
