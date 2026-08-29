"""Deterministic engineering diagnostic for pre-lab recipe iteration.

This module is deliberately isolated from human evaluation storage.  It helps
rank and debug formulas before a panel is available; it never turns synthetic
draws into human evidence or a regulatory claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .catalog import HistoricalReferenceCorpus
from .models import Ingredient, RecipeLine, ScentBrief, SCENT_DIMENSIONS
from .optimizer import semantic_brief_similarity
from .realism import assess_realism, weighted_pair_support


@dataclass(frozen=True)
class SimulatedSensoryResult:
    score: float
    mean: float
    standard_deviation: float
    p05: float
    p95: float
    draws: int
    status: str
    confidence: str
    components: dict[str, float]
    flags: tuple[str, ...]
    seed: int


class SimulatedSensoryEngine:
    """Estimate a synthetic panel distribution from the structured formula."""

    def evaluate(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
        brief: ScentBrief,
        corpus: HistoricalReferenceCorpus,
        target: float | None = None,
        draws: int | None = None,
        seed: int | None = None,
        calibration: object | None = None,
        scientific_twin: object | None = None,
        physsim: object | None = None,
        target_evidenced: bool = False,
    ) -> SimulatedSensoryResult:
        target = float(
            target if target is not None else brief.constraints.target_similarity
        )
        draws = int(draws if draws is not None else 200)
        if not 0 < target <= 100:
            raise ValueError("simulation target must be in (0, 100]")
        if not 30 <= draws <= 100_000:
            raise ValueError("synthetic simulation draws must be between 30 and 100000")
        if not lines:
            return SimulatedSensoryResult(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                draws,
                "proxy_fail",
                "not_run",
                {},
                ("empty_formula",),
                seed or 0,
            )

        source = seed
        if source is None:
            canonical = "|".join(
                f"{line.ingredient_id}:{line.concentrate_percent:.4f}"
                for line in sorted(lines, key=lambda item: item.ingredient_id)
            )
            source = int(hashlib.sha256(canonical.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(source)

        realism = assess_realism(lines, ingredients, brief, corpus)
        target_vector = np.asarray(
            [brief.target_profile.get(key, 0.0) for key in SCENT_DIMENSIONS],
            dtype=float,
        )
        # Semantic score is calculated through the same profile path used by
        # the optimizer, while the following release vector catches formulas
        # whose pyramid percentages do not translate into perceived release.
        achieved = np.zeros_like(target_vector)
        release = {"top": 0.0, "heart": 0.0, "base": 0.0}
        for line in lines:
            ingredient = ingredients[line.ingredient_id]
            impact = (
                line.concentrate_percent
                * ingredient.odor_impact
                * ingredient.active_strength_percent
                / 100.0
            )
            # Match the optimizer's normalized ingredient vector; using raw
            # descriptor sums here would make the simulator disagree with the
            # formula that it is supposed to evaluate.
            profile = ingredient.vector()
            achieved += impact * profile
            release[line.pyramid] += impact
        if achieved.sum() > 0:
            achieved /= achieved.sum()
        semantic = semantic_brief_similarity(
            target_vector,
            achieved,
            brief.desired_dimensions,
            brief.avoided_dimensions,
        )
        release_total = sum(release.values()) or 1.0
        release_vector = np.asarray(
            [release[level] / release_total for level in ("top", "heart", "base")]
        )
        requested_release = np.asarray(
            [brief.pyramid_ratios[level] / 100.0 for level in ("top", "heart", "base")]
        )
        release_score = max(
            0.0, 100.0 - float(np.abs(release_vector - requested_release).sum()) * 80.0
        )
        coherence = weighted_pair_support(lines, ingredients, corpus) * 100.0
        # The historical corpus is sparse for many modern material names. A
        # missing/zero co-occurrence is therefore treated as neutral evidence,
        # not as proof that the accord smells bad.
        effective_coherence = max(50.0, coherence)

        trusted_calibration = (
            calibration is not None
            and callable(getattr(calibration, "is_trusted", None))
            and calibration.is_trusted()
        )
        if trusted_calibration:
            calibrated = calibration.predict(semantic)
            semantic_component = float(
                calibrated if calibrated is not None else semantic
            )
            confidence = "calibrated_human_proxy"
        else:
            semantic_component = semantic
            confidence = "synthetic_structural_proxy"
        physics_available = scientific_twin is not None and hasattr(
            scientific_twin, "temporal_similarity_mean"
        )
        physsim_available = (
            physsim is not None
            and hasattr(physsim, "similarity")
            and bool(getattr(physsim, "comparison_authorized", False))
            and float(getattr(physsim, "model_applicability_percent", 0.0))
            >= brief.constraints.physsim_min_applicability_percent
        )
        evidenced_reference_similarity = (
            float(getattr(physsim, "similarity", 0.0))
            if physsim_available
            else 0.0
        )
        if physics_available:
            temporal_mean = float(scientific_twin.temporal_similarity_mean)
            temporal_floor = float(scientific_twin.minimum_temporal_similarity)
            applicability = float(scientific_twin.model_applicability_percent)
            temporal_width = max(
                0.0,
                float(scientific_twin.temporal_similarity_p95)
                - float(scientific_twin.temporal_similarity_p05),
            )
            evidence_penalty = max(0.0, 70.0 - applicability) * 0.10
            if physsim_available:
                physsim_similarity = float(physsim.similarity)
                learned_r2_similarity = getattr(physsim, "learned_r2_similarity", None)
                physsim_primary_weight = float(
                    getattr(physsim, "learned_r2_applied_weight", 0.0)
                )
                learned_r2_centered_adjustment = float(
                    getattr(physsim, "learned_r2_centered_score_adjustment", 0.0)
                )
            else:
                physsim_similarity = 0.0
                learned_r2_similarity = None
                physsim_primary_weight = 0.0
                learned_r2_centered_adjustment = 0.0
            base_score_without_learned_r2 = (
                semantic_component * 0.52
                + temporal_mean * 0.38
                + temporal_floor * 0.03
                + realism.score * 0.04
                + release_score * 0.02
                + effective_coherence * 0.01
                - evidence_penalty
            )
            # R2 labels and the text/headspace proxy do not share an absolute
            # intercept. Apply only the validation-gated residual around the
            # Snitz training-label mean, instead of averaging unlike scales.
            if learned_r2_similarity is not None and physsim_primary_weight > 0:
                base_score = (
                    base_score_without_learned_r2 + learned_r2_centered_adjustment
                )
            else:
                base_score = base_score_without_learned_r2
            noise_sd = max(
                1.0,
                0.75 + temporal_width * 0.20 + max(0.0, 80.0 - applicability) * 0.025,
            )
            # Preserve the public confidence label for API compatibility.
            # PhysSim participation is exposed through components and flags.
            confidence = (
                "physics_informed_nonhuman_proxy"
                if bool(getattr(scientific_twin, "model_domain_passed", False))
                else "insufficient_nonhuman_evidence"
            )
        else:
            temporal_mean = semantic_component
            temporal_floor = semantic_component
            applicability = 0.0
            temporal_width = 0.0
            physsim_similarity = 0.0
            learned_r2_similarity = None
            physsim_primary_weight = 0.0
            learned_r2_centered_adjustment = 0.0
            base_score = (
                semantic_component * 0.92
                + realism.score * 0.05
                + release_score * 0.02
                + effective_coherence * 0.01
            )
            noise_sd = 1.5 + max(0.0, 75.0 - realism.score) * 0.035
        values = np.clip(rng.normal(base_score, noise_sd, draws), 0.0, 100.0)
        mean_score = float(values.mean())
        p05 = float(np.percentile(values, 5))
        p95 = float(np.percentile(values, 95))
        flags = list(realism.flags)
        flags.extend(
            [
                "synthetic_draws_are_not_human_panel_records",
                "actual_olfactory_accuracy_is_not_identifiable_without_measurement",
                "synthetic_quantiles_are_prior_propagation_not_empirical_error_coverage",
            ]
        )
        if physics_available:
            flags.extend(scientific_twin.flags)
            if not bool(getattr(scientific_twin, "model_domain_passed", False)):
                flags.append("physics_informed_approval_gate_failed")
        if physsim_available:
            flags.extend(getattr(physsim, "flags", ()))
            flags.append("physsim_used_as_independent_candidate_ranking_signal")
            if physsim_primary_weight > 0:
                flags.append("validation_gated_physsim_r2_primary_weight_applied")
        elif physsim is not None:
            flags.append("physsim_outside_applicability_not_scored")
        if coherence < 25.0:
            flags.append("historical_pair_corpus_sparse_or_low; neutral_prior_used")
        physics_gate = not physics_available or bool(
            getattr(scientific_twin, "model_domain_passed", False)
        )
        if not target_evidenced:
            status = "diagnostic_only"
            confidence = "unvalidated_text_target_diagnostic"
            flags.extend(
                [
                    "no_evidenced_olfactory_target",
                    "diagnostic_score_cannot_approve_similarity_threshold",
                ]
            )
        else:
            reference_gate = True
            if not physsim_available:
                reference_gate = False
                flags.append(
                    "evidenced_reference_physsim_outside_applicability"
                )
            elif evidenced_reference_similarity + 1e-8 < target:
                reference_gate = False
                flags.append("evidenced_reference_physsim_below_target")
            status = (
                "evidenced_nonhuman_pass"
                if p05 >= target and physics_gate and reference_gate
                else "evidenced_nonhuman_fail"
            )
        return SimulatedSensoryResult(
            score=round(max(0.0, min(100.0, mean_score)), 4),
            mean=round(mean_score, 4),
            standard_deviation=round(float(values.std(ddof=1)), 4),
            p05=round(p05, 4),
            p95=round(p95, 4),
            draws=draws,
            status=status,
            confidence=confidence,
            components={
                "semantic_component": round(semantic_component, 4),
                "realism_component": realism.score,
                "release_balance_component": round(release_score, 4),
                "historical_coherence_raw": round(coherence, 4),
                "historical_coherence_component": round(effective_coherence, 4),
                "headspace_temporal_mean": round(temporal_mean, 4),
                "headspace_temporal_floor": round(temporal_floor, 4),
                "model_applicability_percent": round(applicability, 4),
                "headspace_uncertainty_width": round(temporal_width, 4),
                "physsim_similarity_component": round(physsim_similarity, 4),
                "physsim_applicability_percent": round(
                    float(getattr(physsim, "model_applicability_percent", 0.0)), 4
                ),
                "physsim_applied": float(physsim_available),
                "physsim_primary_score_weight": round(physsim_primary_weight, 4),
                "physsim_learned_r2_similarity": round(
                    float(learned_r2_similarity or 0.0), 4
                ),
                "physsim_learned_r2_centered_adjustment": round(
                    learned_r2_centered_adjustment, 4
                ),
                "noise_standard_deviation": round(noise_sd, 4),
            },
            flags=tuple(flags),
            seed=int(source),
        )
