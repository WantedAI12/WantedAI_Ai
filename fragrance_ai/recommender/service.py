"""End-to-end natural-language perfumery recipe service."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import date
from types import TracebackType
from typing import Callable, Self

import numpy as np

from .brief_parser import NaturalLanguageBriefParser
from .catalog import HistoricalReferenceCorpus, IngredientCatalog
from .data_hub import NonHumanDataHub
from .manufacturing import ManufacturingPlanner
from .manufacturing_profiles import ManufacturingProfileRegistry
from .human_calibration import HumanMixtureCalibration
from .models import (
    RecipeConstraints,
    RecipeResult,
    SCENT_DIMENSIONS,
    SafetyReport,
    ScentBrief,
    normalize_profile,
)
from .optimizer import ConstrainedFormulaOptimizer, NoFeasibleFormula
from .odor_profiles import OdorProfileStore
from .quality import QualityEvidenceStore, formula_fingerprint
from .promotion_activation import PromotionActivationBundle
from .perception_guidance import PerceptionGuidance, attach_guidance
from .profile_match import assess_recipe_profiles, attach_profile_assessment, full_profile_similarity, profile_search_assessment
from .global_profile_search import optimize_full_pool, profile_upper_bound
from .linear_program_cache import linear_program_request
from .adaptive_pyramid import actual_pyramid, blended_proposals, check_adaptive_response, prepare_adaptive_policy
from .dose_refinement import dose_refinement_proposals, full_inferred_note_policy
from .search_budget import governed, exhausted as search_budget_exhausted
from .failure_recovery import failed_only_recovery, perfume_recovery_available, recovery_active
from .registry_activation import REGISTRY_CONDITIONAL_DATA_SOURCE
from .odor_integrity import registry_odor_rejection
from .release_spec import ReleaseSpec
from .realism import assess_realism
from .reference_targets import ReferenceTargetStore, ResolvedReferenceTarget
from .safety import (
    PRODUCT_CATEGORY_MAP,
    VALIDATION_LEVELS,
    CandidateSafetyScreen,
    FormulaSafetyGate,
)
from .sensory import CalibrationArtifact, SensoryEvaluationStore
from .simulation import SimulatedSensoryEngine
from .science import ScientificPropertyStore, TemporalMixtureSimulator
from .physsim import ConcentrationAwarePhysSim
from .concentration_response import FrozenConcentrationResponse
from .release import CommercialReleaseStore
from .supplier import SupplierRegistry


LIMITATIONS = [
    "90점은 구조·물성·헤드스페이스 기반 비인간 시뮬레이션 점수이며 실제 인간 후각 90%의 증명이 아닙니다.",
    "처방 후보는 내장 검토 원료와 서명된 안전·공급 dossier를 통과한 자동 승격 원료로 제한됩니다.",
    "산업 레지스트리 실험 후보의 가격·가용성은 순위 계산용 추정치이며 실제 견적·재고가 아닙니다. 공급 적격성은 별도의 검증 자료로 판정합니다.",
    "상용 출시는 안정성, 용기 적합성, 실제 배치, 시장별 전문가 검토와 서명된 외부 승인이 별도로 필요합니다.",
    "과거 향수 DB의 노트 정보는 참고 신호이며 측정된 분자 조성으로 취급하지 않습니다.",
    "물리 모델은 불완전한 물성·역치 자료와 모델 가정을 포함하므로 실제 헤드스페이스 측정을 대체하지 않습니다.",
    "학습된 R2 체크포인트는 분자 혼합물 유사도 프록시이며 완성 처방의 인간 관능 검증값이 아닙니다.",
    "출처·라이선스·측정 조건이 불명확한 데이터는 물성·규제·공급 사실로 승격하지 않습니다.",
    "산업 레지스트리 조건부 실험 후보는 공개 냄새 기술 기반 R&D 가설이며 독립 안전·공급사 승인 원료가 아닙니다.",
]


REFERENCE_TERMS = {
    "citrus": ("bergamot", "lemon", "orange", "grapefruit", "lime"),
    "fresh": ("mint", "lemon", "aldehyde"),
    "clean": ("musk", "aldehyde", "soap"),
    "green": ("galbanum", "grass", "leaf", "violet leaf"),
    "aquatic": ("marine", "ozone", "water"),
    "floral": ("rose", "jasmine", "floral"),
    "rose": ("rose",),
    "white_floral": ("jasmine", "gardenia", "tuberose"),
    "fruity": ("peach", "pear", "apple", "berry"),
    "spicy": ("pepper", "ginger", "cardamom"),
    "aromatic": ("lavender", "rosemary", "herbal"),
    "woody": ("cedar", "sandalwood", "woods"),
    "amber": ("amber", "benzoin", "resin"),
    "musky": ("musk",),
    "gourmand": ("vanilla", "caramel", "chocolate"),
    "powdery": ("powder", "iris"),
    "smoky": ("smoke", "incense"),
    "leathery": ("leather", "suede"),
    "earthy": ("moss", "patchouli", "earth"),
}


def _olfactory_validation_status(
    sensory: object | None,
    fallback_status: str,
    requested_target_similarity: float,
) -> str:
    """Name verified panel outcomes without inflating the tested threshold."""

    if sensory is None:
        return fallback_status
    passed = bool(getattr(sensory, "passed", False))
    lower_bound = getattr(sensory, "lower_confidence_bound_95", None)
    if passed:
        if (
            requested_target_similarity >= 90.0
            and lower_bound is not None
            and float(lower_bound) >= 90.0
        ):
            return "human_validated_90"
        return "human_validated_requested_target"
    if getattr(sensory, "status", "") == "below_target":
        return "human_below_requested_target"
    if int(getattr(sensory, "unique_panelists", 0)) > 0:
        return "human_evidence_insufficient"
    return fallback_status


class NaturalLanguagePerfumeryAI:
    def __init__(
        self,
        catalog: IngredientCatalog | None = None,
        corpus: HistoricalReferenceCorpus | None = None,
        supplier_registry: SupplierRegistry | None = None,
        sensory_store: SensoryEvaluationStore | None = None,
        quality_store: QualityEvidenceStore | None = None,
        calibration: CalibrationArtifact | None = None,
        odor_store: OdorProfileStore | None = None,
        scientific_store: ScientificPropertyStore | None = None,
        release_store: CommercialReleaseStore | None = None,
        data_hub: NonHumanDataHub | None = None,
        manufacturing_profile_registry: ManufacturingProfileRegistry | None = None,
        reference_target_store: ReferenceTargetStore | None = None,
        human_mixture_calibration: HumanMixtureCalibration | None = None,
        concentration_response: FrozenConcentrationResponse | None = None,
        promotion_bundle: PromotionActivationBundle | None = None,
        perception_guidance: PerceptionGuidance | None = None,
        require_full_profile_match: bool = False,
        enable_full_pool_search: bool = True,
        enable_adaptive_pyramid: bool = True,
        enable_dose_refinement: bool = True,
        minimum_profile_target: float | None = None,
        allow_experimental_safety: bool = True,
    ):
        if not isinstance(require_full_profile_match, bool):
            raise ValueError("require_full_profile_match must be boolean")
        if minimum_profile_target is not None and (
            isinstance(minimum_profile_target, bool) or not isinstance(minimum_profile_target, (int, float))
            or not math.isfinite(minimum_profile_target) or not 0 < minimum_profile_target <= 100
        ):
            raise ValueError("minimum_profile_target must be finite and in (0, 100]")
        if not isinstance(allow_experimental_safety, bool):
            raise ValueError("allow_experimental_safety must be boolean")
        self.minimum_profile_target = minimum_profile_target
        self.allow_experimental_safety = allow_experimental_safety
        self.require_full_profile_match = require_full_profile_match or minimum_profile_target is not None
        if not isinstance(enable_full_pool_search, bool):
            raise ValueError("enable_full_pool_search must be boolean")
        self.enable_full_pool_search = enable_full_pool_search
        if not isinstance(enable_adaptive_pyramid, bool):
            raise ValueError("enable_adaptive_pyramid must be boolean")
        self.enable_adaptive_pyramid = enable_adaptive_pyramid
        if not isinstance(enable_dose_refinement, bool):
            raise ValueError("enable_dose_refinement must be boolean")
        self.enable_dose_refinement = enable_dose_refinement
        from .perception_runtime import configured_perception, assert_provider_product
        self.perception_guidance = perception_guidance if perception_guidance is not None else configured_perception()
        assert_provider_product(self.perception_guidance, 'perfume')
        self.promotion_bundle = (
            promotion_bundle
            if promotion_bundle is not None
            else PromotionActivationBundle.from_environment()
        )
        if catalog is None:
            from .runtime import load_configured_catalog
            base_catalog, self.local_catalog_manifest_sha256 = load_configured_catalog()
            if self.local_catalog_manifest_sha256 is not None:
                # The local SDK must not silently use a weaker policy than the
                # API/factory just because it was constructed directly.
                self.minimum_profile_target = 95. if minimum_profile_target is None else minimum_profile_target
                self.require_full_profile_match = True
        else:
            base_catalog = catalog
            self.local_catalog_manifest_sha256 = None
        base_catalog = self.promotion_bundle.merge_catalog(base_catalog)
        self.odor_store = odor_store
        self.catalog = (
            odor_store.apply_to_catalog(base_catalog) if odor_store else base_catalog
        )
        invalid_active = {item.ingredient_id: reason for item in self.catalog.ingredients
                          if (item.formulation_ready or not item.blocked) and (reason := registry_odor_rejection(item))}
        if invalid_active:
            repaired = [replace(item, formulation_ready=False, blocked=True, blocked_reason=invalid_active[item.ingredient_id],
                                profile={} if "legacy" in invalid_active[item.ingredient_id] else item.profile)
                        if item.ingredient_id in invalid_active else item for item in self.catalog.ingredients]
            self.catalog = IngredientCatalog(repaired, {**self.catalog.metadata, "industrial_registry_in_memory_quarantined": len(invalid_active)})
        self._owned_resources: list[object] = []
        self._closed = False
        self.data_hub = data_hub or NonHumanDataHub()
        if data_hub is None:
            self._owned_resources.append(self.data_hub)
        self.reference_target_store = reference_target_store
        self.human_mixture_calibration = (
            human_mixture_calibration or HumanMixtureCalibration()
        )
        self.corpus = corpus or HistoricalReferenceCorpus(data_hub=self.data_hub)
        base_supplier_registry = supplier_registry or SupplierRegistry.load_builtin()
        self.supplier_registry = self.promotion_bundle.merge_supplier_registry(
            base_supplier_registry
        )
        self.sensory_store = sensory_store
        self.quality_store = quality_store
        self.calibration = calibration
        self.parser = NaturalLanguageBriefParser(self.catalog)
        self.screen = CandidateSafetyScreen()
        self.optimizer = ConstrainedFormulaOptimizer(self.corpus)
        self.safety_gate = FormulaSafetyGate(self.supplier_registry)
        self.manufacturing_planner = ManufacturingPlanner(
            manufacturing_profile_registry
        )
        self.simulation_engine = SimulatedSensoryEngine()
        self.scientific_store = (
            scientific_store or ScientificPropertyStore.load_builtin()
        )
        if scientific_store is None:
            self._owned_resources.append(self.scientific_store)
        self.temporal_simulator = TemporalMixtureSimulator(reference_provider=self.perception_guidance)
        self.physsim_engine = ConcentrationAwarePhysSim(
            concentration_response=concentration_response
        )
        self.release_store = release_store or CommercialReleaseStore()
        if release_store is None:
            self._owned_resources.append(self.release_store)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Close only repositories created and owned by this service."""
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def _catalog_stats(self) -> dict:
        return {
            **self.catalog.stats(),
            **self.corpus.stats(),
            **self.supplier_registry.stats(),
            **(
                self.odor_store.stats(self.catalog)
                if self.odor_store
                else {
                    "odor_observation_count": 0,
                    "odor_observed_ingredients": 0,
                    "odor_profile_coverage_percent": 0.0,
                }
            ),
            **self.scientific_store.stats(),
            **self.data_hub.stats(),
        }

    def _blocked_result(
        self,
        brief: ScentBrief,
        message: str,
        rejected: dict[str, int],
        as_of: date,
    ) -> RecipeResult:
        safety = SafetyReport(
            internal_gate_passed=False,
            status="blocked",
            active_ifra_amendment=self.safety_gate.ACTIVE_IFRA_LABEL,
            standards_checked_on=self.safety_gate.REVIEWED_ON.isoformat(),
            standards_review_due=self.safety_gate.REVIEW_DUE.isoformat(),
            regulatory_data_complete=False,
            manufacturing_ready=False,
            violations=[message],
            warnings=[],
            eu_label_declarations=[],
            target_region=brief.constraints.target_region.upper(),
            product_category=brief.constraints.product_category,
            validation_level=brief.constraints.validation_level,
            audit_id=self.safety_gate._audit_id([], brief.constraints, as_of),
        )
        return RecipeResult(
            status="no_safe_match",
            message=message,
            brief=brief,
            similarity_score=0.0,
            similarity_kind="semantic_profile_match_not_human_panel_accuracy",
            recipe=[],
            closest_candidate=[],
            achieved_profile={},
            estimated_concentrate_cost_per_kg=0.0,
            historical_support_score=0.0,
            catalog_stats=self._catalog_stats(),
            rejected_candidate_counts=rejected,
            safety=safety,
            limitations=LIMITATIONS.copy(),
            sensory_validation_status="not_tested",
            historical_reference_matches=[],
            reference_molecular_composition_status=(
                self.corpus.molecular_composition_status
            ),
            reference_molecular_composition_claim_boundary=(
                self.corpus.molecular_composition_claim_boundary
            ),
            olfactory_validation_status="not_tested",
        )

    @staticmethod
    def _validate_constraints(constraints: RecipeConstraints) -> None:
        """Reject invalid request controls instead of silently clipping them."""

        def number(name: str, value: object) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"{name} must be finite")
            return parsed

        def integer(name: str, value: object, minimum: int, maximum: int) -> int:
            parsed = number(name, value)
            if not parsed.is_integer() or not minimum <= parsed <= maximum:
                raise ValueError(
                    f"{name} must be an integer between {minimum} and {maximum}"
                )
            return int(parsed)

        for name in (
            "allow_rare",
            "require_simulation_pass",
            "enable_semantic_ontology",
            "enable_concentration_response",
            "enable_learned_r2",
            "enable_registry_trace_candidates",
            "experimental_disable_safety",
            "require_evidenced_olfactory_target",
            "require_catalog_dimension_support",
        ):
            if not isinstance(getattr(constraints, name), bool):
                raise ValueError(f"{name} must be boolean")

        bounded_percentages = {
            "target_similarity": constraints.target_similarity,
            "product_concentration_percent": constraints.product_concentration_percent,
            "minimum_realism_score": constraints.minimum_realism_score,
            "simulation_min_applicability_percent": (
                constraints.simulation_min_applicability_percent
            ),
            "simulation_max_uncertainty_width": (
                constraints.simulation_max_uncertainty_width
            ),
            "physsim_min_applicability_percent": (
                constraints.physsim_min_applicability_percent
            ),
            "commercial_min_scientific_coverage_percent": (
                constraints.commercial_min_scientific_coverage_percent
            ),
            "commercial_min_temporal_similarity": (
                constraints.commercial_min_temporal_similarity
            ),
        }
        for name, value in bounded_percentages.items():
            numeric = number(name, value)
            if name in {"target_similarity", "product_concentration_percent"}:
                valid = 0.0 < numeric <= 100.0
            else:
                valid = 0.0 <= numeric <= 100.0
            if not valid:
                raise ValueError(
                    f"{name} must be within the supported percentage range"
                )
        integer("simulation_draws", constraints.simulation_draws, 64, 100_000)
        from .models import MAX_FORMULA_INGREDIENTS
        integer("max_ingredients", constraints.max_ingredients, 3, MAX_FORMULA_INGREDIENTS)
        integer(
            "physics_search_population",
            constraints.physics_search_population,
            1,
            7,
        )
        integer("max_risk_tier", constraints.max_risk_tier, 0, 3)
        integer(
            "max_supplier_lead_time_days",
            constraints.max_supplier_lead_time_days,
            0,
            3_650,
        )
        minimum_panelists = integer(
            "min_panelists", constraints.min_panelists, 1, 10_000
        )
        minimum_experts = integer(
            "min_expert_panelists", constraints.min_expert_panelists, 0, 10_000
        )
        if minimum_experts > minimum_panelists:
            raise ValueError("min_expert_panelists cannot exceed min_panelists")
        dimension_strength = number(
            "minimum_dimension_material_strength",
            constraints.minimum_dimension_material_strength,
        )
        if not 0.0 < dimension_strength <= 1.0:
            raise ValueError("minimum_dimension_material_strength must be in (0, 1]")
        objective_weight = number(
            "surrogate_objective_weight", constraints.surrogate_objective_weight
        )
        if not 0.0 <= objective_weight <= 0.5:
            raise ValueError("surrogate_objective_weight must be between 0 and 0.5")
        positive_controls = {
            "max_ingredient_price_per_kg": constraints.max_ingredient_price_per_kg,
            "max_formula_cost_per_kg": constraints.max_formula_cost_per_kg,
            "finished_volume_ml": constraints.finished_volume_ml,
            "finished_batch_mass_g": constraints.finished_batch_mass_g,
        }
        for name, value in positive_controls.items():
            if number(name, value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if number("max_supplier_moq_kg", constraints.max_supplier_moq_kg) < 0.0:
            raise ValueError("max_supplier_moq_kg must be nonnegative")
        if not 0.0 <= number("min_availability", constraints.min_availability) <= 1.0:
            raise ValueError("min_availability must be between 0 and 1")
        if (
            constraints.product_density_g_ml is not None
            and number("product_density_g_ml", constraints.product_density_g_ml) <= 0.0
        ):
            raise ValueError("product_density_g_ml must be positive")
        if (
            not isinstance(constraints.validation_level, str)
            or constraints.validation_level not in VALIDATION_LEVELS
        ):
            raise ValueError("unsupported validation_level")
        if (
            not isinstance(constraints.product_category, str)
            or constraints.product_category not in PRODUCT_CATEGORY_MAP
        ):
            raise ValueError("unsupported product_category")
        if (
            not isinstance(constraints.target_region, str)
            or not constraints.target_region.strip()
            or len(constraints.target_region) > 32
        ):
            raise ValueError(
                "target_region must be non-empty text of at most 32 characters"
            )
        if not isinstance(constraints.explicit_bans, (set, frozenset)) or not all(
            isinstance(item, str) and len(item) <= 256
            for item in constraints.explicit_bans
        ):
            raise ValueError("explicit_bans must be a set of ingredient names")
        if not isinstance(constraints.commercial_supplier_evidence, dict):
            raise ValueError("commercial_supplier_evidence must be an object")
        if constraints.experimental_disable_safety and (
            not constraints.enable_registry_trace_candidates
            or constraints.validation_level != "prototype"
        ):
            raise ValueError(
                "experimental_disable_safety requires prototype registry mode"
            )
        for name in (
            "reference_target_id",
            "commercial_product_base_id",
            "commercial_packaging_id",
            "commercial_rule_pack_version",
            "commercial_data_version",
            "commercial_model_version",
        ):
            value = getattr(constraints, name)
            if not isinstance(value, str) or len(value) > 256:
                raise ValueError(f"{name} must be text of at most 256 characters")

    @failed_only_recovery('perfume', eligibility=perfume_recovery_available)
    @linear_program_request
    @governed('perfume')
    def create_recipe(
        self,
        natural_language_brief: str,
        constraints: RecipeConstraints | None = None,
        as_of: date | None = None,
        *,
        target_profile_override: dict[str, float] | None = None,
        intent_controls: dict | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RecipeResult:
        snapshot_guard = getattr(self, "_runtime_snapshot_guard", None)
        from .perception_runtime import assert_provider_current
        assert_provider_current(self.perception_guidance)
        if snapshot_guard is not None:
            snapshot_guard()
        if constraints is not None:
            self._validate_constraints(constraints)
        requested = self.parser.parse(natural_language_brief, constraints).constraints
        automatic = bool(self.require_full_profile_match and requested.max_ingredients > 12
                         and requested.validation_level == "prototype" and not requested.reference_target_id)
        if recovery_active() and target_profile_override is None:
            from .hierarchical_perfume import active as hierarchical_active
            if hierarchical_active(self.perception_guidance, self.parser.parse(natural_language_brief, requested)):
                automatic = False
        budgets = []
        support_results = []

        def run_budget(limit):
            budgets.append(limit)
            return self._create_recipe_impl(natural_language_brief,
                replace(requested, max_ingredients=limit), as_of, target_profile_override=target_profile_override,
                **({"intent_controls": intent_controls} if intent_controls else {}),
                **({"progress_callback": progress_callback} if progress_callback is not None else {}))

        def assessed(candidate):
            return assess_recipe_profiles(candidate.brief, candidate.achieved_profile, candidate.temporal_profile,
                                          self.temporal_simulator.time_weights(candidate.brief))

        def record(candidate):
            assessment = assessed(candidate)
            support_results.append({"max_ingredients": budgets[-1], "score": assessment["score"],
                                    "target_met": assessment["target_met"], "usable_recipe": bool(candidate.recipe)})
            return assessment

        def rank(candidate, assessment):
            score = assessment['score']
            if score is None:
                score = assessment.get('partial_profile_score')
            return (bool(candidate.recipe), -1. if score is None else score)

        result = run_budget(12 if automatic else requested.max_ingredients)
        best = record(result)
        if automatic:
            # The small-support trajectory is an incumbent, not a restriction
            # on the eligible pool. Keep its winner when wider searches regress.
            latest = result
            limits = () if getattr(result, '_hierarchical_intent', None) is not None else (min(24, requested.max_ingredients), requested.max_ingredients)
            for limit in dict.fromkeys(limits):
                if search_budget_exhausted():
                    break
                if best["target_met"] and result.recipe:
                    break
                # Preserve the complete V15 24-first path when the small
                # search fails; only the full-limit retry uses its old bound.
                if limit > 24:
                    bound = getattr(latest, "_full_profile_search", {}).get("full_pool_search", {}).get("bound", {}).get("upper_score")
                    if bound is not None and bound + 1e-8 < latest.brief.constraints.target_similarity:
                        break
                alternative = run_budget(limit)
                second = record(alternative)
                latest = alternative
                if rank(alternative, second) > rank(result, best):
                    result, best = alternative, second
            # Search budgets are not user constraints. Restore the actual
            # maximum; every candidate satisfied a tighter or identical cap.
            result.brief = replace(result.brief, constraints=replace(result.brief.constraints,
                                    max_ingredients=requested.max_ingredients))
        if (self.perception_guidance is not None and getattr(result, 'perception_guidance', None) is None
                and getattr(result, '_hierarchical_intent', None) is None):
            session = self.perception_guidance.begin(result.brief)
            report = session.report(None, None, changed=False, variants=0)
            report.update(status='not_evaluated_no_eligible_candidate', operation='create_recipe', recipe_returned=False)
            result = attach_guidance(result, report)
        if progress_callback is not None:
            progress_callback("TEMPORAL_PROFILE")
        assessment = assess_recipe_profiles(
            result.brief, result.achieved_profile, result.temporal_profile,
            self.temporal_simulator.time_weights(result.brief),
        )
        assessment["search"] = getattr(result, "_full_profile_search", {})
        assessment['search']['score_scope'] = assessment.get('score_scope', 'complete_requested_profile')
        if getattr(result, '_hierarchical_intent', None) is not None:
            assessment.update(score=None, target_met=False, target_representation=result._hierarchical_intent,
                score_kind='hierarchical_source_reference_unresolved', version='hierarchical-perfume-reference/v77')
        if result.brief.constraints.reference_target_id.strip():
            assessment.update(score=None, target_met=False,
                              scope="explicit_reference_uses_existing_reference_comparison_contract")
        output = attach_profile_assessment(result, assessment, strict=self.require_full_profile_match)
        output.score_contract.update(runtime_minimum_profile_target=self.minimum_profile_target,
                                     effective_target=output.brief.constraints.target_similarity,
                                     search_support_budgets=budgets,
                                     search_support_results=support_results,
                                     legacy_preference_screen_target=self._preference_screen_target(output.brief.constraints),
                                     actual_human_similarity_proven_by_this_score=False)
        from .odor_expression import recipe_expression
        output.score_contract['fine_odor_expression'] = recipe_expression(
            output.recipe or output.closest_candidate,output.brief,self.catalog)
        if snapshot_guard is not None:
            snapshot_guard()
        assert_provider_current(self.perception_guidance)
        return output

    def _preference_screen_target(self, constraints):
        # The compatibility preference proxy is not the full-profile score.
        # Raising the requested full-profile goal must not erase useful search
        # intermediates. Final full-profile acceptance still uses the exact
        # requested goal, and evidenced/qualified flows keep their old gates.
        if self.require_full_profile_match and constraints.validation_level == "prototype" and not constraints.reference_target_id:
            bank = getattr(self.perception_guidance, 'complete_reference_bank', None)
            if bank is not None and getattr(bank, 'odor_space', None) is not None:
                # The legacy preference remains a diagnostic/seed heuristic;
                # the new full-reference final threshold is unchanged.
                return 0.
            return min(90., constraints.target_similarity)
        return constraints.target_similarity

    def _create_recipe_impl(
        self,
        natural_language_brief: str,
        constraints: RecipeConstraints | None = None,
        as_of: date | None = None,
        *,
        target_profile_override: dict[str, float] | None = None,
        intent_controls: dict | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RecipeResult:
        as_of = as_of or date.today()
        if constraints is not None:
            self._validate_constraints(constraints)
        brief = self.parser.parse(natural_language_brief, constraints)
        if intent_controls:
            from .intent_controls import apply_intent_controls
            brief = apply_intent_controls(brief, intent_controls)
        self._validate_constraints(brief.constraints)
        if not self.allow_experimental_safety and brief.constraints.experimental_disable_safety:
            raise ValueError("runtime policy does not allow experimental safety overrides")
        if self.minimum_profile_target is not None:
            brief = replace(brief, constraints=replace(brief.constraints,
                target_similarity=max(brief.constraints.target_similarity, self.minimum_profile_target)))
        if self.perception_guidance is not None and brief.constraints.validation_level != "prototype":
            raise ValueError("experimental perception guidance is restricted to prototype research")
        if target_profile_override is not None:
            if not isinstance(target_profile_override, dict):
                raise ValueError("target profile override must be an object")
            if not all(isinstance(key, str) for key in target_profile_override):
                raise ValueError("target profile override dimension names must be text")
            unknown_dimensions = sorted(
                set(target_profile_override) - set(SCENT_DIMENSIONS)
            )
            if unknown_dimensions:
                raise ValueError(
                    "target profile override contains unknown dimensions: "
                    + ", ".join(unknown_dimensions)
                )
            for dimension, value in target_profile_override.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"target profile override value for {dimension} must be "
                        "finite and nonnegative"
                    )
            target_profile = normalize_profile(target_profile_override)
            if sum(target_profile.values()) <= 0:
                raise ValueError(
                    "target profile override must contain positive scent mass"
                )
            if any(target_profile.get(axis, 0.) > 0 for axis in brief.avoided_dimensions):
                raise ValueError('target profile conflicts with an explicit avoidance constraint')
            brief = replace(
                brief,
                target_profile=target_profile,
                target_profile_source='explicit_structured_relative_weights',
                desired_dimensions=sorted(
                    dimension
                    for dimension, value in target_profile.items()
                    if value >= 0.01
                ),
                avoided_dimensions=sorted(brief.avoided_dimensions),
            )
        if progress_callback is not None:
            progress_callback("INGREDIENT_SCREENING")
        from .hierarchical_perfume import active as hierarchical_active, target_rows
        if hierarchical_active(self.perception_guidance, brief):
            from .odor_space import target_report
            intent = target_report(self.perception_guidance.complete_reference_bank, brief, target_rows(brief))
            if not intent['searchable']:
                blocked = self._blocked_result(brief, '세부 향의 정량 참조가 부족합니다: '+', '.join(intent['unsupported']), {}, as_of)
                blocked._hierarchical_intent = intent
                return blocked
        candidates, rejected = self.screen.screen(
            self.catalog,
            brief,
            supplier_registry=self.supplier_registry,
            as_of=as_of,
        )

        if progress_callback is not None:
            progress_callback("SAFETY_CHECK")

        for pyramid in brief.pyramid_ratios:
            if brief.pyramid_ratios[pyramid] > 0 and not any(item.pyramid == pyramid for item in candidates):
                level_hint = (
                    " 검증 단계에서는 실제 공급사 문서가 연결된 원료를 먼저 등록해야 합니다."
                    if brief.constraints.validation_level != "prototype"
                    else ""
                )
                return self._blocked_result(
                    brief,
                    f"안전·가격·공급 조건을 만족하는 {pyramid} 원료가 없습니다.{level_hint}",
                    rejected,
                    as_of,
                )

        capability = self.catalog.capability_report(
            candidates,
            minimum_strength=brief.constraints.minimum_dimension_material_strength,
        )
        unsupported_requested = sorted(
            dimension
            for dimension in brief.desired_dimensions
            if int(capability["strong_material_counts"].get(dimension, 0)) == 0
        )
        if (
            unsupported_requested
            and brief.constraints.require_catalog_dimension_support
        ):
            return self._blocked_result(
                brief,
                "현재 안전·가격·공급 조건의 원료 공간에서 충분한 표현력이 없는 향 축: "
                + ", ".join(unsupported_requested),
                rejected,
                as_of,
            )

        candidate_map = {item.ingredient_id: item for item in candidates}
        ingredient_map = {item.ingredient_id: item for item in self.catalog.ingredients}
        ingredient_map.update(candidate_map)
        reference_target: ResolvedReferenceTarget | None = None
        if brief.constraints.reference_target_id.strip():
            if self.reference_target_store is None:
                return self._blocked_result(
                    brief,
                    "요청한 기준 향 조성 저장소가 연결되지 않아 비교를 중단했습니다.",
                    rejected,
                    as_of,
                )
            try:
                reference_target = self.reference_target_store.resolve(
                    brief.constraints.reference_target_id,
                    ingredients=ingredient_map,
                    constraints=brief.constraints,
                    as_of=as_of,
                )
            except (OSError, ValueError) as error:
                return self._blocked_result(
                    brief,
                    f"기준 향 조성 증거를 검증할 수 없습니다: {error}",
                    rejected,
                    as_of,
                )
        elif brief.constraints.require_evidenced_olfactory_target:
            return self._blocked_result(
                brief,
                "실측 후각 목표가 필수이지만 검증된 정량 기준 향 조성이 지정되지 않았습니다.",
                rejected,
                as_of,
            )

        guide = self.perception_guidance.begin(brief) if self.perception_guidance is not None else None
        guide_sets = []
        guidance_variants_added = 0
        full_profile_variants_added = 0
        phase_context_matrices = {}
        response_index = {}
        nominal_responses = None
        time_targets = self.temporal_simulator.targets_by_time(brief)
        time_weights = self.temporal_simulator.time_weights(brief)

        def formula_context_objective(weights, selected_ingredients) -> float:
            if reference_target is not None:
                candidate_weights = {
                    item.ingredient_id: max(0.0, float(weight))
                    for item, weight in zip(selected_ingredients, weights)
                    if weight > 0.0
                }
                target_weights = {
                    line.ingredient_id: max(0.0, line.concentrate_percent)
                    for line in reference_target.lines
                }
                identifiers = set(candidate_weights).union(target_weights)
                numerator = sum(
                    min(candidate_weights.get(key, 0.0), target_weights.get(key, 0.0))
                    for key in identifiers
                )
                denominator = sum(
                    max(candidate_weights.get(key, 0.0), target_weights.get(key, 0.0))
                    for key in identifiers
                )
                return 100.0 * numerator / max(denominator, 1e-12)

            if brief.phase_target_profiles:
                key = tuple(item.ingredient_id for item in selected_ingredients)
                if key not in phase_context_matrices:
                    rows = np.asarray([response_index[identifier] for identifier in key])
                    vectors = np.asarray([item.vector() for item in selected_ingredients])
                    phase_context_matrices[key] = nominal_responses[rows, :, None] * vectors[:, None, :]
                matrix = phase_context_matrices[key]
                profiles = np.einsum("i,itd->td", weights, matrix)
                profiles /= np.maximum(1e-12, profiles.sum(axis=1, keepdims=True))
                return sum(float(weight) * self.temporal_simulator.temporal_target_similarity(target, profiles[index], desired, avoided)
                           for index, ((target, desired, avoided), weight) in enumerate(zip(time_targets, time_weights))
                           if weight > 0)

            impact = sum(
                max(0.0, float(weight))
                / 100.0
                * item.odor_impact
                * item.active_strength_percent
                / 100.0
                for item, weight in zip(selected_ingredients, weights)
            )
            perceived_intensity = min(1.0, impact / 2.5)
            diffusion = sum(
                max(0.0, float(weight))
                / 100.0
                * {"top": 0.90, "heart": 0.55, "base": 0.20}[item.pyramid]
                for item, weight in zip(selected_ingredients, weights)
            )
            error = 0.70 * abs(
                perceived_intensity - brief.absolute_intensity_target
            ) + 0.30 * abs(diffusion - brief.diffusion_target)
            return max(0.0, 100.0 * (1.0 - error))

        def internally_eligible(variant, context_brief=None) -> bool:
            context_brief = context_brief or brief
            if retired_blends.reject_lines(variant[0]):
                return False
            if variant[1] + 1e-8 < self._preference_screen_target(context_brief.constraints):
                return False
            if variant[3] > context_brief.constraints.max_formula_cost_per_kg:
                return False
            if assess_realism(variant[0], ingredient_map, context_brief, self.corpus).score < context_brief.constraints.minimum_realism_score:
                return False
            if not context_brief.constraints.experimental_disable_safety and not self.safety_gate.evaluate(
                variant[0], ingredient_map, context_brief.constraints, as_of=as_of
            ).internal_gate_passed:
                return False
            return True

        if progress_callback is not None:
            progress_callback("RATIO_OPTIMIZATION")
        from .retired_blends import RetiredBlendFilter
        retired_blends = RetiredBlendFilter(
            'perfume' if brief.constraints.product_category in
            ('eau_de_parfum', 'eau_de_toilette', 'eau_de_cologne') else 'other',
            brief.constraints.product_concentration_percent)
        try:
            selected_candidates = self.optimizer._select_candidates(candidates, brief)
            property_ids = [item.ingredient_id for item in candidates]
            if reference_target is not None:
                property_ids.extend(line.ingredient_id for line in reference_target.lines)
            scientific_properties = self.scientific_store.get_many(property_ids)
            scientific_properties = ScientificPropertyStore.with_catalog_structures(candidates, scientific_properties)
            if guide is not None and hasattr(guide, 'bind_properties'):
                guide.bind_properties(scientific_properties)
            response_inputs = self.temporal_simulator.prepare_response_inputs(candidates, scientific_properties)
            response_index = {identifier: index for index, identifier in enumerate(response_inputs.identifiers)}
            response_matrices = {}

            def responses_at(concentration):
                if concentration not in response_matrices:
                    response_matrices[concentration] = self.temporal_simulator.ingredient_response_matrix(response_inputs, concentration)
                return response_matrices[concentration]

            nominal_responses = responses_at(brief.constraints.product_concentration_percent)
            # Build a deterministic candidate population under balanced, early,
            # middle, late, low-dilution and high-dilution headspace objectives.
            # The expensive twin and learned R2 score every unique member, so
            # physics now guides formula selection rather than merely annotating
            # a single semantic solution after optimization.
            requested_temporal = (
                brief.temporal_emphasis.get("opening", 0.25) * 0.65,
                brief.temporal_emphasis.get("opening", 0.25) * 0.35,
                brief.temporal_emphasis.get("heart", 0.40),
                brief.temporal_emphasis.get("drydown", 0.35) * 0.40,
                brief.temporal_emphasis.get("drydown", 0.35) * 0.60,
            )
            requested_total = sum(requested_temporal) or 1.0
            requested_temporal = tuple(
                value / requested_total for value in requested_temporal
            )
            if brief.phase_target_profiles:
                requested_temporal = tuple(time_weights)
            search_specs = (
                ("requested", 1.0, requested_temporal),
                ("balanced", 1.0, None),
                ("early", 1.0, (0.70, 0.20, 0.08, 0.02, 0.00)),
                ("middle", 1.0, (0.05, 0.25, 0.45, 0.20, 0.05)),
                ("late", 1.0, (0.00, 0.03, 0.12, 0.35, 0.50)),
                ("low_dilution", 0.50, None),
                ("high_dilution", 1.50, None),
            )
            requested_population = min(
                len(search_specs), int(brief.constraints.physics_search_population)
            )
            formula_variants = []
            variant_fingerprints: set[str] = set()

            def add_variant(variant):
                if retired_blends.reject_lines(variant[0]):
                    return False
                fingerprint = formula_fingerprint(variant[0])
                if fingerprint not in variant_fingerprints:
                    variant_fingerprints.add(fingerprint)
                    formula_variants.append(variant)
                    return True
                return False

            for _, concentration_scale, temporal_weights in search_specs[
                :requested_population
            ]:
                perceptual_factors = self.temporal_simulator.factors_from_responses(
                    response_inputs.identifiers,
                    responses_at(max(0.1, brief.constraints.product_concentration_percent * concentration_scale)),
                    temporal_weights,
                )
                add_variant(
                    self.optimizer._optimize_selected(
                        selected_candidates,
                        brief,
                        perceptual_factors=perceptual_factors,
                        formula_objective=formula_context_objective,
                    )
                )
            add_variant(
                self.optimizer._optimize_selected(
                    selected_candidates,
                    brief,
                    formula_objective=formula_context_objective,
                )
            )
            baseline_variant_count = len(formula_variants)
            replacement_sets = self.optimizer.replacement_selections(
                candidates, selected_candidates, brief,
                limit=6 if brief.phase_target_profiles else 3,
            )
            nominal_factors = self.temporal_simulator.factors_from_responses(
                response_inputs.identifiers, nominal_responses, requested_temporal,
            )
            for replacement in replacement_sets:
                try:
                    replacement_variant = self.optimizer._optimize_selected(
                        replacement, brief, perceptual_factors=nominal_factors,
                        formula_objective=formula_context_objective,
                    )
                except NoFeasibleFormula:
                    continue
                if replacement_variant[1] + 1e-8 < self._preference_screen_target(brief.constraints):
                    continue
                if replacement_variant[3] > brief.constraints.max_formula_cost_per_kg:
                    continue
                if assess_realism(replacement_variant[0], ingredient_map, brief, self.corpus).score < brief.constraints.minimum_realism_score:
                    continue
                if not brief.constraints.experimental_disable_safety and not self.safety_gate.evaluate(replacement_variant[0], ingredient_map, brief.constraints, as_of=as_of).internal_gate_passed:
                    continue
                add_variant(replacement_variant)
            unguided_variant_count = len(formula_variants)
            unguided_swap_count = len(formula_variants) - baseline_variant_count
            if guide is not None and guide.enabled:
                supported = [item for item in candidates if guide.supports(item)]
                required_ids = {
                    item.ingredient_id for name in brief.requested_ingredients
                    if (item := self.catalog.lookup(name)) is not None
                }
                if required_ids.issubset({item.ingredient_id for item in supported}):
                    try:
                        guided_selection = self.optimizer._select_candidates(supported, brief)
                        proposed_sets = [guided_selection, *self.optimizer.replacement_selections(
                            supported, guided_selection, brief, limit=3
                        )]
                        if all(guide.supports(item) for item in selected_candidates):
                            proposed_sets.insert(0, selected_candidates)
                        set_ids = set()
                        for selection in proposed_sets:
                            identity = tuple(sorted(item.ingredient_id for item in selection))
                            if identity not in set_ids:
                                set_ids.add(identity)
                                guide_sets.append(selection)
                    except NoFeasibleFormula:
                        # Unsupported naturals/insufficient modeled pyramid
                        # capacity cannot be replaced by zero odor vectors.
                        guide_sets = []
                guided_brief = replace(brief, constraints=replace(
                    brief.constraints, surrogate_objective_weight=guide.weight
                ))

                def learned_objective(weights, ingredients):
                    value = guide.score(weights, ingredients)
                    # Invalid proposed formulas are not given the high legacy
                    # context score as a way to escape the learned objective.
                    return 0.0 if value is None else value

                for selection in guide_sets:
                    try:
                        variant = self.optimizer._optimize_selected(
                            selection, guided_brief, formula_objective=learned_objective
                        )
                    except NoFeasibleFormula:
                        continue
                    if not required_ids.issubset({line.ingredient_id for line in variant[0]}):
                        continue
                    if internally_eligible(variant):
                        guidance_variants_added += int(add_variant(variant))
            legacy_search_count = len(formula_variants)
            if reference_target is None:
                # Keep every compatibility candidate, and add full-vector
                # refinements. No goal profile is reconstructed from a candidate.
                full_sets = [selected_candidates, *replacement_sets]
                for selection in full_sets:
                    try:
                        refined = self.optimizer._optimize_selected(
                            selection, brief, perceptual_factors=nominal_factors,
                            formula_objective=formula_context_objective,
                            profile_scorer=full_profile_similarity,
                        )
                    except NoFeasibleFormula:
                        continue
                    if internally_eligible(refined):
                        full_profile_variants_added += int(add_variant(refined))
        except NoFeasibleFormula as error:
            return self._blocked_result(brief, str(error), rejected, as_of)
        if not formula_variants:
            return self._blocked_result(brief, "제외된 배합 외의 후보를 찾지 못했습니다.", rejected, as_of)
        evaluated_variants = []
        screening_draws = min(64, max(30, brief.constraints.simulation_draws))
        for variant in formula_variants:
            variant_lines = variant[0]
            variant_twin = self.temporal_simulator.evaluate(
                variant_lines,
                ingredient_map,
                brief,
                scientific_properties,
                draws=screening_draws,
            )
            variant_physsim = self.physsim_engine.evaluate(
                variant_lines,
                ingredient_map,
                brief,
                scientific_properties,
                reference_target_lines=(
                    list(reference_target.lines) if reference_target else None
                ),
            )
            variant_simulation = self.simulation_engine.evaluate(
                variant_lines,
                ingredient_map,
                brief,
                self.corpus,
                target=brief.constraints.target_similarity,
                draws=screening_draws,
                calibration=self.calibration,
                scientific_twin=variant_twin,
                physsim=variant_physsim,
                target_evidenced=reference_target is not None,
            )
            evaluated_variants.append(
                (variant, variant_twin, variant_physsim, variant_simulation)
            )

        def physics_objective(item) -> float:
            _, twin_result, physsim_result, simulation_result = item
            if brief.constraints.experimental_disable_safety:
                return (
                    0.75 * item[0][1]
                    + 0.20 * twin_result.temporal_similarity_p05
                    + 0.05
                    * simulation_result.components.get(
                        "release_balance_component", 0.0
                    )
                )
            if not physsim_result.comparison_authorized:
                return (
                    0.55 * twin_result.temporal_similarity_p05
                    + 0.35 * item[0][1]
                    + 0.10
                    * simulation_result.components.get("release_balance_component", 0.0)
                )
            return (
                0.40 * physsim_result.similarity
                + 0.35 * simulation_result.p05
                + 0.25 * twin_result.temporal_similarity_p05
            )

        original_choice = max(
            evaluated_variants[:unguided_variant_count] or evaluated_variants,
            key=lambda item: (
                item[3].status == "evidenced_nonhuman_pass",
                physics_objective(item),
                item[0][1],
            ),
        )
        chosen = original_choice
        baseline_guidance = selected_guidance = None
        if guide is not None and guide.enabled:
            baseline_guidance = guide.evaluate_lines(original_choice[0][0], ingredient_map)
            eligible = []
            required_ids = {
                item.ingredient_id for name in brief.requested_ingredients
                if (item := self.catalog.lookup(name)) is not None
            }
            for item in evaluated_variants[:legacy_search_count]:
                if not internally_eligible(item[0]):
                    continue
                if not required_ids.issubset({line.ingredient_id for line in item[0][0]}):
                    continue
                exact = guide.evaluate_lines(item[0][0], ingredient_map)
                if exact is None:
                    continue
                if baseline_guidance is not None and exact["score"] + 1e-8 < baseline_guidance["score"]:
                    continue
                # A research prior cannot erase existing requested-target,
                # safety or evidence gates. Bound legacy proxy regression too.
                if not getattr(guide, 'authoritative_profile_objective', False) and physics_objective(item) < physics_objective(original_choice) - 3.0:
                    continue
                if original_choice[3].status == "evidenced_nonhuman_pass" and item[3].status != "evidenced_nonhuman_pass":
                    continue
                eligible.append((item, exact))
            if eligible:
                chosen, selected_guidance = max(eligible, key=lambda pair: (
                    (1.0 - guide.weight) * physics_objective(pair[0]) + guide.weight * pair[1]["score"],
                    pair[0][0][1],
                ))
        before_full_choice = chosen
        full_before = full_after = None
        if reference_target is None:
            def complete_score(item):
                assessment = profile_search_assessment(
                    brief, item[0][2], [asdict(point) for point in item[1].temporal_points], time_weights
                )
                return assessment["score"]

            full_before = complete_score(chosen)
            guide_before = guide.evaluate_lines(chosen[0][0], ingredient_map) if guide is not None and guide.enabled else None
            legacy_was_eligible = internally_eligible(chosen[0])
            named_ids = {item.ingredient_id for name in brief.requested_ingredients
                         if (item := self.catalog.lookup(name)) is not None}
            preserved_named_ids = named_ids & {line.ingredient_id for line in chosen[0][0]}
            ranked = []
            for item in evaluated_variants:
                if not internally_eligible(item[0]):
                    continue
                if not preserved_named_ids.issubset({line.ingredient_id for line in item[0][0]}):
                    continue
                if chosen[3].status == "evidenced_nonhuman_pass" and item[3].status != "evidenced_nonhuman_pass":
                    continue
                score = complete_score(item)
                if score is None or (not self.require_full_profile_match and not legacy_was_eligible and score + 1e-8 < brief.constraints.target_similarity):
                    continue
                if guide_before is not None:
                    candidate_guide = guide.evaluate_lines(item[0][0], ingredient_map)
                    if candidate_guide is None or candidate_guide["score"] + 1e-8 < guide_before["score"]:
                        continue
                ranked.append((score, item))
            if ranked:
                full_after, chosen = max(ranked, key=lambda pair: (pair[0], physics_objective(pair[1]), pair[1][0][1]))
            else:
                full_after = full_before
        screening_full_before, screening_full_after = full_before, full_after
        if brief.constraints.simulation_draws > screening_draws:
            def final_evaluation(item):
                variant, _, variant_physsim, _ = item
                twin = self.temporal_simulator.evaluate(
                    variant[0], ingredient_map, brief, scientific_properties,
                    draws=brief.constraints.simulation_draws,
                )
                simulated = self.simulation_engine.evaluate(
                    variant[0], ingredient_map, brief, self.corpus,
                    target=brief.constraints.target_similarity,
                    draws=brief.constraints.simulation_draws,
                    calibration=self.calibration, scientific_twin=twin,
                    physsim=variant_physsim, target_evidenced=reference_target is not None,
                )
                return variant, twin, variant_physsim, simulated

            chosen = final_evaluation(chosen)
            if reference_target is None:
                # Screening means use fewer prior draws. Verify the improvement
                # at the actual output draw count, never compare 64 with 200.
                if formula_fingerprint(before_full_choice[0][0]) != formula_fingerprint(chosen[0][0]):
                    final_baseline = final_evaluation(before_full_choice)
                    full_before, full_after = complete_score(final_baseline), complete_score(chosen)
                    if full_after is None or (full_before is not None and full_after + 1e-8 < full_before):
                        chosen, full_after = final_baseline, full_before
                else:
                    full_before = full_after = complete_score(chosen)
        # V7's final-draw choice remains the baseline. New full-pool solutions
        # are evaluated at that same final draw count, never promoted by an LP
        # bound or a shorter screening run.
        pool_diagnostic = {"enabled": self.enable_full_pool_search, "status": "not_applicable", "attempts": []}
        pool_variants_evaluated = 0
        pool_sets_evaluated = set()
        dose_seeds = []
        if reference_target is None:
            current_full = complete_score(chosen)
            optimistic_bound = profile_upper_bound(candidates, brief.target_profile)
            if getattr(guide, 'authoritative_profile_objective', False):
                optimistic_bound = {'status':'not_certified_for_full_reference_objective', 'upper_score':100.,
                    'legacy_render_allowance_points':0., 'legacy_19_axis_bound_not_applied':True}
            if (self.enable_full_pool_search and self.require_full_profile_match and current_full is not None
                    and not getattr(guide, 'authoritative_profile_objective', False)
                    and current_full+1e-8 < brief.constraints.target_similarity):
                # A positive structural mixture must obey this cap/cost hull,
                # even if inferred note bands and cardinality are removed.
                # The recomputed dual is a bound, not a favorable LP proposal.
                from .models import MAX_FORMULA_INGREDIENTS
                relaxed_brief = replace(brief, constraints=replace(brief.constraints, max_ingredients=MAX_FORMULA_INGREDIENTS))
                relaxed = optimize_full_pool(candidates, relaxed_brief,
                    pyramid_bounds={group: (0., 100.) for group in brief.pyramid_ratios})
                if relaxed.certified_overlap_upper_score is not None:
                    tighter = min(optimistic_bound['upper_score'], relaxed.certified_overlap_upper_score+
                                  optimistic_bound['legacy_render_allowance_points'])
                    optimistic_bound.update(status='validated_fractional_cap_cost_upper', upper_score=tighter,
                        certified_unrendered_upper_score=relaxed.certified_overlap_upper_score,
                        bounds_include_caps_cost_pyramid='caps_cost_only_pyramid_and_cardinality_relaxed',
                        certificate_method='recomputed_lagrangian_bound_with_reduced_cost_box_residuals',
                        numerical_optimum_is_not_achievability=True)
            pool_diagnostic.update(
                candidate_count=len(candidates), bound=optimistic_bound,
                baseline_v7_score=current_full, selected_score=current_full, candidate_changed=False,
                baseline_v7_formula_id=formula_fingerprint(chosen[0][0]),
                selected_formula_id=formula_fingerprint(chosen[0][0]),
                requested_target_ruled_out_by_bound=(optimistic_bound["upper_score"] is not None
                                                    and optimistic_bound["upper_score"] + 1e-8 < brief.constraints.target_similarity),
            )
            if (self.require_full_profile_match and pool_diagnostic['requested_target_ruled_out_by_bound']
                    and getattr(getattr(guide,'provider',None),'core',None) is not None
                    and getattr(guide.provider.core,'version',None) in ('shared-formulation-core/v75','shared-formulation-core/v76')):
                from .profile_frontier import explain_profile_frontier
                pool_diagnostic['profile_frontier']=explain_profile_frontier(candidates,brief,optimistic_bound)
            if self.enable_full_pool_search and current_full is not None and current_full + 1e-8 < brief.constraints.target_similarity:
                pool_diagnostic["status"] = "evaluated_entire_screened_pool"
                minimums = {
                    line.ingredient_id: line.concentrate_percent for line in chosen[0][0]
                    if line.ingredient_id in preserved_named_ids
                }
                # Risk-tier permission does not require retaining an automatic
                # ingredient choice. All reviewed UPPER dose caps still apply;
                # unnamed materials can be reduced or replaced to fit the brief.
                contexts = [("structural", None, brief.target_profile), ("headspace", nominal_factors, brief.target_profile)]
                if brief.phase_target_profiles:
                    for phase, profile in brief.phase_target_profiles.items():
                        if sum(profile.values()) > 0:
                            phase_weights = tuple(float(point.phase == phase) for point in chosen[1].temporal_points)
                            contexts.append((phase, self.temporal_simulator.factors_from_responses(
                                response_inputs.identifiers, nominal_responses, phase_weights,
                            ), profile))
                else:
                    contexts.append(("drydown", self.temporal_simulator.factors_from_responses(
                        response_inputs.identifiers, nominal_responses, (0., 0., 0., .4, .6),
                    ), brief.target_profile))
                prior_guide = guide.evaluate_lines(chosen[0][0], ingredient_map) if guide is not None and guide.enabled else None
                new_fingerprints = {formula_fingerprint(chosen[0][0])}
                for context, factors, target in contexts:
                    proposal = optimize_full_pool(candidates, brief, factors=factors, target=target, minimum_percent=minimums)
                    attempt = {"context": context, "status": proposal.status, "relaxed_overlap_score": proposal.relaxed_overlap_score,
                               "restricted_support": proposal.restricted_support, "ingredient_count": len(proposal.weights_percent)}
                    pool_diagnostic["attempts"].append(attempt)
                    if not proposal.weights_percent:
                        continue
                    dose_seeds.append(proposal.weights_percent)
                    used = [ingredient_map[key] for key in proposal.weights_percent]
                    variant = self.optimizer.variant_from_weights(used, brief, np.asarray(list(proposal.weights_percent.values())))
                    if (any(line.concentrate_percent <= 0 for line in variant[0])
                            or abs(sum(line.concentrate_percent for line in variant[0]) - 100.) > .001
                            or not internally_eligible(variant)):
                        attempt["status"] = "rejected_existing_recipe_constraints"
                        continue
                    identity = formula_fingerprint(variant[0])
                    if identity in new_fingerprints:
                        attempt["status"] = "duplicate_recipe"
                        continue
                    new_fingerprints.add(identity)
                    twin = self.temporal_simulator.evaluate(variant[0], ingredient_map, brief, scientific_properties, draws=brief.constraints.simulation_draws)
                    pool_variants_evaluated += 1
                    pool_sets_evaluated.add(tuple(sorted(line.ingredient_id for line in variant[0])))
                    comparison = profile_search_assessment(brief, variant[2], [asdict(point) for point in twin.temporal_points], time_weights)["score"]
                    attempt["full_profile_score"] = comparison
                    if (comparison is None or comparison <= current_full + 1e-8
                            or (not self.require_full_profile_match and not legacy_was_eligible and comparison + 1e-8 < brief.constraints.target_similarity)):
                        attempt["status"] = "no_verified_full_profile_improvement"
                        continue
                    if prior_guide is not None:
                        exact_guide = guide.evaluate_lines(variant[0], ingredient_map)
                        if exact_guide is None or exact_guide["score"] + 1e-8 < prior_guide["score"]:
                            attempt["status"] = "rejected_existing_perception_guidance"
                            continue
                    variant_physsim = self.physsim_engine.evaluate(variant[0], ingredient_map, brief, scientific_properties)
                    variant_simulation = self.simulation_engine.evaluate(
                        variant[0], ingredient_map, brief, self.corpus, target=brief.constraints.target_similarity,
                        draws=brief.constraints.simulation_draws, calibration=self.calibration,
                        scientific_twin=twin, physsim=variant_physsim, target_evidenced=False,
                    )
                    if chosen[3].status == "evidenced_nonhuman_pass" and variant_simulation.status != "evidenced_nonhuman_pass":
                        attempt["status"] = "rejected_existing_evidence_gate"
                        continue
                    chosen = (variant, twin, variant_physsim, variant_simulation)
                    current_full = comparison
                    full_after = comparison
                    if prior_guide is not None:
                        prior_guide = exact_guide
                    pool_diagnostic.update(selected_score=comparison, candidate_changed=True, selected_formula_id=identity)
                    attempt["status"] = "selected_full_profile_improvement"
            elif self.enable_full_pool_search:
                pool_diagnostic["status"] = "undefined_complete_target" if current_full is None else "requested_target_already_met"
            else:
                pool_diagnostic["status"] = "disabled_for_control_comparison"
        # Preserve V8's completed result as the control. Note allocation is
        # inferred policy, while the odor target and user constraints stay fixed.
        adaptive = {"enabled": self.enable_adaptive_pyramid and self.enable_full_pool_search, "status": "not_applicable", "attempts": []}
        if reference_target is None:
            baseline_v8 = chosen
            baseline_assessment = profile_search_assessment(brief, chosen[0][2], [asdict(p) for p in chosen[1].temporal_points], time_weights)
            adaptive.update(baseline_v8_score=baseline_assessment["score"], selected_score=baseline_assessment["score"],
                            baseline_v8_formula_id=formula_fingerprint(chosen[0][0]), candidate_changed=False,
                            original_inferred_pyramid=dict(brief.pyramid_ratios), selected_pyramid=actual_pyramid(chosen[0][0]))
            if not adaptive["enabled"]:
                adaptive["status"] = "disabled_for_control_comparison"
            elif brief.constraints.validation_level != "prototype":
                adaptive["status"] = "qualified_release_scope_preserved"
            elif baseline_assessment["score"] is None:
                adaptive["status"] = "undefined_complete_target"
            elif baseline_assessment["score"] + 1e-8 >= brief.constraints.target_similarity:
                adaptive["status"] = "requested_target_already_met"
            else:
                policy = prepare_adaptive_policy(brief, chosen[0][0], ingredient_map)
                adaptive["policy"] = policy
                if all(low == high for low, high in policy["bounds"].values()):
                    adaptive["status"] = "explicit_pyramid_locked"
                else:
                    adaptive["status"] = "evaluated_bounded_note_allocations"
                    source_brief = brief
                    modeled = [item for item in candidates if item.vector().sum() > 0]
                    minimums = {line.ingredient_id: line.concentrate_percent for line in chosen[0][0]
                                if line.ingredient_id in preserved_named_ids}
                    adaptive_contexts = [("structural", None, source_brief.target_profile),
                                         ("headspace", nominal_factors, source_brief.target_profile)]
                    if source_brief.phase_target_profiles:
                        for phase, target in source_brief.phase_target_profiles.items():
                            if sum(target.values()) > 0:
                                weights_by_time = tuple(float(point.phase == phase) for point in chosen[1].temporal_points)
                                adaptive_contexts.append((phase, self.temporal_simulator.factors_from_responses(
                                    response_inputs.identifiers, nominal_responses, weights_by_time), target))
                    else:
                        adaptive_contexts.append(("drydown", self.temporal_simulator.factors_from_responses(
                            response_inputs.identifiers, nominal_responses, (0., 0., 0., .4, .6)), source_brief.target_profile))
                    seen_adaptive = {formula_fingerprint(chosen[0][0])}
                    current_adaptive_score = baseline_assessment["score"]
                    prior_guide = guide.evaluate_lines(chosen[0][0], ingredient_map) if guide is not None and guide.enabled else None
                    for context, factors, target in adaptive_contexts:
                        proposal = optimize_full_pool(modeled, source_brief, factors=factors, target=target, minimum_percent=minimums,
                                                      pyramid_bounds=policy["bounds"], intensity_range=policy["intensity_range"],
                                                      diffusion_range=policy["diffusion_range"])
                        attempt = {"context": "adaptive_" + context, "status": proposal.status,
                                   "relaxed_overlap_score": proposal.relaxed_overlap_score}
                        if not proposal.weights_percent:
                            adaptive["attempts"].append(attempt)
                            pool_diagnostic["attempts"].append(attempt)
                            continue
                        if context == "structural":
                            dose_seeds.insert(0, proposal.weights_percent)
                        else:
                            dose_seeds.append(proposal.weights_percent)
                        for fraction, mixture_weights in blended_proposals(
                            baseline_v8[0][0], proposal.weights_percent, source_brief.constraints.max_ingredients,
                            {item.ingredient_id for item in modeled},
                        ):
                            attempt = {"context": "adaptive_" + context, "status": "proposed", "blend_fraction": fraction,
                                       "endpoint_lp_overlap_score": proposal.relaxed_overlap_score}
                            adaptive["attempts"].append(attempt)
                            pool_diagnostic["attempts"].append(attempt)
                            used = [ingredient_map[key] for key in mixture_weights]
                            variant = self.optimizer.variant_from_weights(used, source_brief, np.asarray(list(mixture_weights.values())))
                            ratios = actual_pyramid(variant[0])
                            actual_cost = sum(line.concentrate_percent / 100 * line.price_per_kg for line in variant[0])
                            if (not internally_eligible(variant) or any(line.concentrate_percent <= 0 for line in variant[0])
                                    or abs(sum(ratios.values()) - 100) > .001
                                    or actual_cost > source_brief.constraints.max_formula_cost_per_kg + 1e-6
                                    or any(not low - .001 <= ratios[group] <= high + .001 for group, (low, high) in policy["bounds"].items())):
                                attempt["status"] = "rejected_existing_or_allocation_constraints"
                                continue
                            identity = formula_fingerprint(variant[0])
                            if identity in seen_adaptive:
                                attempt["status"] = "duplicate_recipe"
                                continue
                            seen_adaptive.add(identity)
                            candidate_brief = replace(source_brief, pyramid_ratios=ratios)
                            twin = self.temporal_simulator.evaluate(variant[0], ingredient_map, candidate_brief, scientific_properties,
                                                                    draws=source_brief.constraints.simulation_draws)
                            pool_variants_evaluated += 1
                            pool_sets_evaluated.add(tuple(sorted(line.ingredient_id for line in variant[0])))
                            assessment = profile_search_assessment(candidate_brief, variant[2], [asdict(p) for p in twin.temporal_points], time_weights)
                            score = assessment["score"]
                            attempt.update(full_profile_score=score, pyramid_ratios=ratios)
                            if score is None or score <= current_adaptive_score + 1e-8 or (not self.require_full_profile_match and not legacy_was_eligible and score + 1e-8 < source_brief.constraints.target_similarity):
                                attempt["status"] = "no_verified_full_profile_improvement"
                                continue
                            violations = check_adaptive_response(source_brief, baseline_assessment, assessment,
                                                                 baseline_v8[1], twin, variant[0], ingredient_map, policy)
                            if violations:
                                attempt.update(status="rejected_response_preservation", violations=violations)
                                continue
                            if prior_guide is not None:
                                exact = guide.evaluate_lines(variant[0], ingredient_map)
                                if exact is None or exact["score"] + 1e-8 < prior_guide["score"]:
                                    attempt["status"] = "rejected_existing_perception_guidance"
                                    continue
                            candidate_physsim = self.physsim_engine.evaluate(variant[0], ingredient_map, candidate_brief, scientific_properties)
                            candidate_simulation = self.simulation_engine.evaluate(variant[0], ingredient_map, candidate_brief, self.corpus,
                                target=source_brief.constraints.target_similarity, draws=source_brief.constraints.simulation_draws,
                                calibration=self.calibration, scientific_twin=twin, physsim=candidate_physsim, target_evidenced=False)
                            if chosen[3].status == "evidenced_nonhuman_pass" and candidate_simulation.status != "evidenced_nonhuman_pass":
                                attempt["status"] = "rejected_existing_evidence_gate"
                                continue
                            chosen, brief = (variant, twin, candidate_physsim, candidate_simulation), candidate_brief
                            current_adaptive_score = full_after = score
                            if prior_guide is not None:
                                prior_guide = exact
                            adaptive.update(candidate_changed=True, selected_score=score, selected_pyramid=ratios)
                            pool_diagnostic.update(selected_score=score, candidate_changed=True, selected_formula_id=identity)
                            attempt["status"] = "selected_adaptive_improvement"
                            if score + 1e-8 >= source_brief.constraints.target_similarity:
                                break
                        if current_adaptive_score + 1e-8 >= source_brief.constraints.target_similarity:
                            break
        pool_diagnostic["adaptive_pyramid"] = adaptive
        dose = {"enabled": bool(self.enable_dose_refinement and self.enable_adaptive_pyramid and self.enable_full_pool_search),
                "status": "not_applicable", "attempts": []}
        if reference_target is None:
            dose_baseline, dose_brief = chosen, brief
            dose_assessment = profile_search_assessment(brief, chosen[0][2], [asdict(p) for p in chosen[1].temporal_points], time_weights)
            dose_score = dose_assessment["score"]
            dose.update(baseline_v9_score=dose_score, selected_score=dose_score, candidate_changed=False,
                        baseline_v9_formula_id=formula_fingerprint(chosen[0][0]),
                        original_pyramid=dict(brief.pyramid_ratios), selected_pyramid=actual_pyramid(chosen[0][0]))
            if not dose["enabled"]:
                dose["status"] = "disabled_for_control_comparison"
            elif brief.constraints.validation_level != "prototype":
                dose["status"] = "qualified_release_scope_preserved"
            elif dose_score is None:
                dose.update(status="undefined_complete_target", missing_positive_phase_targets=sorted({
                    point["phase"] for point in dose_assessment["temporal"]
                    if point["weight"] > 0 and not sum(point["target_profile"].values())
                }))
            elif dose_score + 1e-8 >= brief.constraints.target_similarity:
                dose["status"] = "requested_target_already_met"
            else:
                dose["status"] = "evaluated_actual_dose_joint_time_proposals"
                policy = prepare_adaptive_policy(brief, chosen[0][0], ingredient_map)
                minimums = {line.ingredient_id: line.concentrate_percent for line in chosen[0][0]
                            if line.ingredient_id in preserved_named_ids}
                seen_dose = {formula_fingerprint(chosen[0][0])}
                prior_guide = guide.evaluate_lines(chosen[0][0], ingredient_map) if guide is not None and guide.enabled else None
                modeled_ids = {item.ingredient_id for item in candidates if item.vector().sum() > 0}
                dose['autoregressive_refinement'] = {}
                for proposal in dose_refinement_proposals(candidates, dose_brief, scientific_properties, dose_baseline[0][0], policy, dose_seeds, minimums,
                                                         current_lines=lambda: chosen[0][0], current_score=lambda: dose_score,
                                                         current_feedback=lambda: profile_search_assessment(dose_brief, chosen[0][2], [asdict(p) for p in chosen[1].temporal_points], time_weights),
                                                         autoregressive_diagnostics=dose['autoregressive_refinement'],
                                                         guidance=guide, minimum_guidance=prior_guide['score'] if prior_guide is not None else None):
                    neural_usage=(dose['autoregressive_refinement'].get('neural_usage') if proposal.get('replacement_mode')
                        in ('trusted_neural_feedback','trained_autoregressive_decoder','reference_neural_feedback','reference_autoregressive_decoder',
                            'learned_seed_actual_physics_polish') else None)
                    allocation_bounds = (full_inferred_note_policy(policy)["bounds"] if proposal.get("allocation_mode") == "inferred_full_range" else policy["bounds"])
                    from .adaptive_pyramid import refinement_blends
                    blends = list(refinement_blends(dose_baseline[0][0], proposal, brief.constraints.max_ingredients, modeled_ids))
                    for fraction, weights in reversed(blends):
                        attempt = {key: value for key, value in proposal.items() if key not in ("weights_percent","anchor_weights_percent")}
                        attempt.update(context="actual_dose_joint_time", blend_fraction=fraction, status="proposed")
                        dose["attempts"].append(attempt)
                        pool_diagnostic["attempts"].append(attempt)
                        variant = self.optimizer.variant_from_weights([ingredient_map[key] for key in weights], dose_brief, np.asarray(list(weights.values())))
                        ratios = actual_pyramid(variant[0])
                        context_brief = replace(dose_brief, pyramid_ratios=ratios)
                        if (not internally_eligible(variant, context_brief) or any(line.concentrate_percent <= 0 for line in variant[0])
                                or abs(sum(ratios.values()) - 100) > .001
                                or sum(line.concentrate_percent / 100 * line.price_per_kg for line in variant[0]) > brief.constraints.max_formula_cost_per_kg + 1e-6
                                or any(not low - .001 <= ratios[group] <= high + .001 for group, (low, high) in allocation_bounds.items())):
                            attempt["status"] = "rejected_existing_or_allocation_constraints"
                            if proposal.get('replacement_mode','').startswith('source-fixed-physical-inverse/'):
                                reality = assess_realism(variant[0], ingredient_map, context_brief, self.corpus)
                                gate = self.safety_gate.evaluate(variant[0], ingredient_map, context_brief.constraints, as_of=as_of)
                                attempt['constraint_rejection'] = {
                                    'legacy_preference':variant[1],
                                    'legacy_preference_minimum':self._preference_screen_target(context_brief.constraints),
                                    'realism_score':reality.score,'realism_minimum':context_brief.constraints.minimum_realism_score,
                                    'realism_components':reality.components,'safety_violations':list(gate.violations),
                                    'zero_dose_rows':sum(line.concentrate_percent<=0 for line in variant[0]),
                                    'mass_total':sum(ratios.values()),'formula_cost':variant[3]}
                            continue
                        identity = formula_fingerprint(variant[0])
                        if identity in seen_dose:
                            attempt["status"] = "duplicate_recipe"
                            continue
                        seen_dose.add(identity)
                        twin = self.temporal_simulator.evaluate(variant[0], ingredient_map, context_brief, scientific_properties, draws=brief.constraints.simulation_draws)
                        if neural_usage is not None:
                            neural_usage['actual_physics_evaluations']+=1
                        pool_variants_evaluated += 1
                        pool_sets_evaluated.add(tuple(sorted(line.ingredient_id for line in variant[0])))
                        assessment = profile_search_assessment(context_brief, variant[2], [asdict(p) for p in twin.temporal_points], time_weights)
                        score = assessment["score"]
                        attempt.update(full_profile_score=score, pyramid_ratios=ratios)
                        if score is None or score <= dose_score + 1e-8 or (not self.require_full_profile_match and not legacy_was_eligible and score + 1e-8 < brief.constraints.target_similarity):
                            attempt["status"] = "no_verified_full_profile_improvement"
                            continue
                        violations = check_adaptive_response(dose_brief, dose_assessment, assessment, dose_baseline[1], twin, variant[0], ingredient_map, policy)
                        if violations:
                            attempt.update(status="rejected_response_preservation", violations=violations)
                            continue
                        exact = guide.evaluate_lines(variant[0], ingredient_map) if prior_guide is not None else None
                        if prior_guide is not None and (exact is None or exact["score"] + 1e-8 < prior_guide["score"]):
                            attempt["status"] = "rejected_existing_perception_guidance"
                            continue
                        candidate_physsim = self.physsim_engine.evaluate(variant[0], ingredient_map, context_brief, scientific_properties)
                        candidate_simulation = self.simulation_engine.evaluate(variant[0], ingredient_map, context_brief, self.corpus,
                            target=brief.constraints.target_similarity, draws=brief.constraints.simulation_draws,
                            calibration=self.calibration, scientific_twin=twin, physsim=candidate_physsim, target_evidenced=False)
                        if chosen[3].status == "evidenced_nonhuman_pass" and candidate_simulation.status != "evidenced_nonhuman_pass":
                            attempt["status"] = "rejected_existing_evidence_gate"
                            continue
                        chosen, brief = (variant, twin, candidate_physsim, candidate_simulation), context_brief
                        dose_score = full_after = score
                        if prior_guide is not None:
                            prior_guide = exact
                        dose.update(candidate_changed=True, selected_score=score, selected_formula_id=identity, selected_pyramid=ratios)
                        pool_diagnostic.update(selected_score=score, candidate_changed=True, selected_formula_id=identity)
                        attempt["status"] = "selected_actual_dose_improvement"
                        if neural_usage is not None:
                            neural_usage['adopted_improvements']+=1
                        if score + 1e-8 >= brief.constraints.target_similarity:
                            break
                    if dose_score + 1e-8 >= brief.constraints.target_similarity:
                        break
        pool_diagnostic["dose_refinement"] = dose
        selected_variant, scientific_twin, physsim, simulation = chosen
        if guide is not None:
            selected_guidance = guide.evaluate_lines(selected_variant[0], ingredient_map)
        lines, raw_similarity, achieved, cost, support = selected_variant
        human_calibration = self.human_mixture_calibration.compare(
            lines,
            list(reference_target.lines) if reference_target else None,
            matrix_id=(reference_target.matrix_id if reference_target else ""),
            product_concentration_percent=brief.constraints.product_concentration_percent,
        )
        safety = self.safety_gate.evaluate(
            lines,
            ingredient_map,
            brief.constraints,
            as_of=as_of,
        )
        experimental_safety_disabled = bool(
            brief.constraints.experimental_disable_safety
            and brief.constraints.enable_registry_trace_candidates
            and brief.constraints.validation_level == "prototype"
        )
        if experimental_safety_disabled and not any(registry_odor_rejection(ingredient_map[line.ingredient_id]) for line in lines):
            safety = replace(
                safety,
                internal_gate_passed=True,
                status="experimental_safety_disabled",
                regulatory_data_complete=False,
                manufacturing_ready=False,
                violations=[],
                warnings=[
                    *safety.warnings,
                    "실험 모드 요청으로 내부 안전·규제 승인 차단을 사용하지 않았습니다.",
                ],
            )
        formula_id = formula_fingerprint(lines)
        # A formula-only fingerprint is deliberately insufficient for a
        # commercial approval.  Build the full product/supplier-lot/version
        # scope and fail closed when any real document is absent or changes.
        release_spec: ReleaseSpec | None = None
        try:
            release_spec = ReleaseSpec.build(
                lines,
                brief.constraints,
                self.supplier_registry,
                rule_pack_version=brief.constraints.commercial_rule_pack_version,
                data_version=brief.constraints.commercial_data_version,
                model_version=brief.constraints.commercial_model_version,
                as_of=as_of,
            )
            release_assessment = self.release_store.assess_scope(release_spec, as_of)
        except (OSError, ValueError) as error:
            # Keep the detailed cause in the evidence assessment rather than
            # attempting a legacy formula-ID lookup.
            from .release import ReleaseEvidenceAssessment

            release_assessment = ReleaseEvidenceAssessment(
                False,
                "release_scope_incomplete",
                0,
                ("verified canonical commercial release scope",),
                False,
                (str(error),),
            )
        realism = assess_realism(lines, ingredient_map, brief, self.corpus)
        reference_terms = [line.name for line in lines] + brief.requested_ingredients
        for dimension in brief.desired_dimensions:
            reference_terms.extend(REFERENCE_TERMS.get(dimension, ()))
        historical_references = self.corpus.nearest_references(reference_terms, limit=5)

        level = brief.constraints.validation_level
        evidence_scope_id = (
            release_spec.release_spec_id
            if level in {"qualified", "commercial"} and release_spec
            else formula_id
        )
        quality = (
            self.quality_store.assess(evidence_scope_id, as_of=as_of)
            if self.quality_store
            else None
        )
        manufacturing = self.manufacturing_planner.build(
            lines,
            ingredient_map,
            brief.constraints,
            stability_passed=bool(quality and quality.passed),
            as_of=as_of,
        )

        sensory = (
            self.sensory_store.formula_evidence(
                evidence_scope_id,
                brief.constraints.target_similarity,
                brief.constraints.min_panelists,
                brief.constraints.min_expert_panelists,
                as_of=as_of,
            )
            if self.sensory_store
            else None
        )
        if self.sensory_store:
            self.sensory_store.register_formula(
                evidence_scope_id,
                brief.original_text,
                raw_similarity,
                [
                    {
                        "ingredient_id": line.ingredient_id,
                        "concentrate_percent": line.concentrate_percent,
                    }
                    for line in lines
                ],
            )

        # Do not expose a panel-calibrated-looking value when the artifact
        # fails its integrity or statistical gate.
        calibrated = (
            self.calibration.predict(raw_similarity)
            if self.calibration and self.calibration.is_trusted()
            else None
        )
        cost_ok = bool(
            experimental_safety_disabled
            or cost <= brief.constraints.max_formula_cost_per_kg
        )
        semantic_ok = raw_similarity + 1e-8 >= self._preference_screen_target(brief.constraints)
        sensory_ok = bool(sensory and sensory.passed)
        quality_ok = bool(quality and quality.passed)
        release_ok = bool(
            release_spec is not None
            and release_assessment.passed
            and release_assessment.scope_verified
        )
        science_ok = (
            scientific_twin.scientific_data_coverage_percent
            >= brief.constraints.commercial_min_scientific_coverage_percent
            and scientific_twin.minimum_temporal_similarity
            >= brief.constraints.commercial_min_temporal_similarity
            and scientific_twin.model_domain_passed
        )
        realism_ok = realism.score + 1e-8 >= brief.constraints.minimum_realism_score
        simulation_ok = bool(
            reference_target is not None
            and simulation.status == "evidenced_nonhuman_pass"
        )
        approved = (
            safety.internal_gate_passed and cost_ok and semantic_ok and realism_ok
        )
        if brief.constraints.require_simulation_pass:
            approved = approved and simulation_ok
        if level in {"qualified", "commercial"}:
            # Human/quality evidence for a market-bound result is accepted only
            # against the complete product, supplier-lot, document and
            # model/rule/data scope—not a reusable formula-only fingerprint.
            approved = (
                approved
                and release_spec is not None
                and release_ok
                and sensory_ok
                and manufacturing.ready_for_lab_trial
            )
        if level == "commercial":
            approved = approved and quality_ok and science_ok

        uses_registry_conditionals = any(
            line.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE for line in lines
        )

        if approved and level == "commercial":
            recipe = lines
            if (
                release_assessment.passed
                and release_assessment.scope_verified
                and manufacturing.ready_for_manufacture
            ):
                status = "manufacturing_ready"
                message = "과학·공급사·인간 관능·품질 증거와 목표 시장의 외부 규제 서명이 모두 처방 지문에 연결됐습니다."
                safety = replace(
                    safety,
                    status="commercial_release_gate_passed",
                    manufacturing_ready=True,
                    regulatory_data_complete=True,
                )
            else:
                status = "commercial_evidence_ready"
                message = "과학 물성, 공급사 문서, 인간 관능, 안정성·포장·파일럿 내부 증거가 준비됐습니다. 외부 규제 책임자의 시장별 서명 전에는 제조·판매 승인 상태가 아닙니다."
                safety = replace(
                    safety,
                    status=(
                        "manufacturing_readiness_incomplete"
                        if release_assessment.passed
                        and release_assessment.scope_verified
                        else "external_regulatory_signoff_required"
                    ),
                    manufacturing_ready=False,
                    regulatory_data_complete=bool(
                        release_assessment.passed and release_assessment.scope_verified
                    ),
                )
        elif approved and level == "qualified":
            status = "lab_validated"
            message = "공급사 문서와 블라인드 관능 근거가 확인된 랩 검증 처방입니다. 상업 생산 전 안정성·포장·파일럿 시험이 필요합니다."
            recipe = lines
            safety = replace(safety, status="lab_validated")
        elif approved:
            if uses_registry_conditionals:
                status = "experimental_registry_candidate"
                message = (
                    "전체 산업 레지스트리에서 선별한 조건부 실험 원료가 포함된 "
                    "R&D 가설 처방입니다. 독립 안전·공급사·규제 문서가 연결되기 "
                    "전에는 prototype_ready 또는 제조 후보로 승격되지 않습니다."
                )
            else:
                status = "prototype_ready"
                message = (
                    "안전·가격·공급·의미 프로필 조건을 충족한 R&D 처방입니다. "
                    "정량 기준 향이 없어 실제 후각 유사도는 기권 처리되었습니다."
                    if reference_target is None
                    else (
                        f"검증된 정량 기준 향에 대한 비인간 모델 5% 하한 "
                        f"{simulation.p05:.2f}점의 R&D 처방입니다. "
                        "인간 후각 90% 승인과는 별개입니다."
                    )
                )
            recipe = lines
        else:
            status = "no_safe_match"
            reasons: list[str] = []
            if not semantic_ok:
                reasons.append(
                    f"의미 프로필 {raw_similarity:.2f}%가 기준 "
                    f"{brief.constraints.target_similarity:.2f}% 미만"
                )
            if not cost_ok:
                reasons.append(
                    f"예상 원가 {cost:.2f}/kg가 한도 "
                    f"{brief.constraints.max_formula_cost_per_kg:.2f}/kg 초과"
                )
            if not safety.internal_gate_passed:
                reasons.append("안전·규제·공급사 증빙 게이트 미통과")
            if not realism_ok:
                reasons.append(
                    f"현실성 점수 {realism.score:.2f}%가 기준 "
                    f"{brief.constraints.minimum_realism_score:.2f}% 미만"
                )
            if brief.constraints.require_simulation_pass and not simulation_ok:
                if reference_target is None:
                    reasons.append("검증된 정량 기준 향이 없어 시뮬레이션 승인을 기권")
                elif simulation.p05 < brief.constraints.target_similarity:
                    reasons.append(
                        f"시뮬레이션 5% 하한 {simulation.p05:.2f}%가 기준 "
                        f"{brief.constraints.target_similarity:.2f}% 미만"
                    )
                else:
                    reasons.append(
                        "헤드스페이스 트윈 승인 실패: "
                        f"적용 가능성 {scientific_twin.model_applicability_percent:.2f}%, "
                        f"시간가중 5% 하한 {scientific_twin.temporal_similarity_p05:.2f}%, "
                        f"최저 시간점 5% 하한 {scientific_twin.minimum_temporal_similarity_p05:.2f}%"
                    )
            if level in {"qualified", "commercial"} and not sensory_ok:
                sensory_status = sensory.status if sensory else "not_tested"
                reasons.append(f"블라인드 관능 근거 미달: {sensory_status}")
            if level in {"qualified", "commercial"} and not release_ok:
                reasons.append(
                    f"서명된 제품·공급사 로트 출시 범위 미충족: "
                    f"{release_assessment.status}"
                )
            if level == "commercial" and not quality_ok:
                reasons.append("안정성·포장·파일럿 품질 시험 미완료")
            if level == "commercial" and not science_ok:
                reasons.append(
                    "과학 물성 커버리지 또는 시간축 최소 유사도가 상용 기준 미달"
                )
            message = "; ".join(reasons) or "승인 조건을 충족하지 못했습니다."
            recipe = []

        # Replace extraction-era mojibake with the gate that actually
        # determined this result. Machine-readable details remain adjacent.
        if status == "manufacturing_ready":
            message = (
                "서명된 출시 범위, 공급사 로트 문서, 검증된 품질 시험, "
                "제품 베이스·포장 적합성과 외부 규제 승인이 모두 확인되었습니다."
            )
        elif status == "commercial_evidence_ready":
            missing = [
                *release_assessment.missing_evidence,
                *manufacturing.readiness_blockers,
            ]
            message = (
                "상용 후보 계산은 완료됐지만 제조 출시는 차단되어 있습니다. "
                + ("미충족 항목: " + ", ".join(sorted(set(missing))) if missing else "")
            ).strip()
        elif status == "lab_validated":
            message = (
                "서명 검증된 관능 근거와 공급 문서가 확인된 실험 후보입니다. "
                "상용 제조 승인은 별도 출시 범위와 품질·규제 증거가 필요합니다."
            )
        elif status == "prototype_ready":
            message = (
                "안전·가격·공급·의미 조건을 충족한 R&D 후보입니다. "
                "실측 기준 향이 없어 후각 유사도 점수는 생성하지 않았습니다."
                if reference_target is None
                else (
                    f"정량 기준 향 대비 비인간 시뮬레이션 5% 하한 "
                    f"{simulation.p05:.2f}점의 R&D 후보입니다."
                )
            )
        elif status == "experimental_registry_candidate":
            message = (
                "전체 산업 레지스트리에서 선별한 risk-tier-2 실험 후보가 "
                "포함된 R&D 가설 처방입니다. 공개 냄새 기술과 구조 스크리닝은 "
                "독립 안전·공급사·규제 승인을 대체하지 않습니다."
            )
        elif status == "no_safe_match":
            clean_reasons: list[str] = []
            if not semantic_ok:
                clean_reasons.append(
                    f"의미 프로필 {raw_similarity:.2f}점이 기준 "
                    f"{brief.constraints.target_similarity:.2f}점 미만"
                )
            if not cost_ok:
                clean_reasons.append(
                    f"예상 원가 {cost:.2f}/kg이 상한 "
                    f"{brief.constraints.max_formula_cost_per_kg:.2f}/kg 초과"
                )
            if not safety.internal_gate_passed:
                clean_reasons.append("안전·규제·공급 증거 게이트 미통과")
            if not realism_ok:
                clean_reasons.append(
                    f"공학적 현실성 {realism.score:.2f}점이 기준 "
                    f"{brief.constraints.minimum_realism_score:.2f}점 미만"
                )
            if brief.constraints.require_simulation_pass and not simulation_ok:
                clean_reasons.append("비인간 시뮬레이션 게이트 미통과")
            if level in {"qualified", "commercial"} and not sensory_ok:
                clean_reasons.append(
                    f"서명 검증된 관능 증거 미충족: "
                    f"{sensory.status if sensory else 'not_tested'}"
                )
            if level in {"qualified", "commercial"} and not release_ok:
                clean_reasons.append(
                    "독립 서명된 제품·공급사 로트 출시 범위 미충족: "
                    f"{release_assessment.status}"
                )
            if level == "commercial" and not quality_ok:
                clean_reasons.append(
                    f"서명 검증된 품질·안정성 증거 미충족: "
                    f"{quality.status if quality else 'not_tested'}"
                )
            if level == "commercial" and not science_ok:
                clean_reasons.append("상용 과학 커버리지·시간축 기준 미충족")
            message = "; ".join(clean_reasons) or "승인 조건을 충족하지 못했습니다."

        result = RecipeResult(
            status=status,
            message=message,
            brief=brief,
            similarity_score=round(raw_similarity, 4),
            similarity_kind="semantic_profile_match_not_human_panel_accuracy",
            recipe=recipe,
            closest_candidate=lines,
            achieved_profile=achieved,
            estimated_concentrate_cost_per_kg=round(cost, 4),
            historical_support_score=round(support * 100.0, 4),
            catalog_stats=self._catalog_stats(),
            rejected_candidate_counts=rejected,
            safety=safety,
            limitations=LIMITATIONS.copy(),
            formula_id=formula_id,
            raw_similarity_score=round(raw_similarity, 4),
            sensory_similarity_score=(
                sensory.mean_similarity if sensory else calibrated
            ),
            sensory_panel_size=(sensory.unique_panelists if sensory else 0),
            sensory_validation_status=(sensory.status if sensory else "not_tested"),
            manufacturing_plan=manufacturing,
            realism_score=realism.score,
            realism_kind=realism.kind,
            accord_family=realism.accord_family,
            odor_profile_coverage_percent=realism.observed_profile_coverage_percent,
            confidence=realism.confidence,
            realism_components=realism.components,
            realism_flags=list(realism.flags),
            historical_reference_matches=historical_references,
            reference_molecular_composition_status=(
                "verified_reference_target_composition"
                if reference_target
                else self.corpus.molecular_composition_status
            ),
            reference_molecular_composition_claim_boundary=(
                (
                    "The PhysSim target is bound to re-hashed quantitative "
                    "composition evidence; it is still not human sensory accuracy."
                )
                if reference_target
                else self.corpus.molecular_composition_claim_boundary
            ),
            reference_target_id=(
                reference_target.target_id if reference_target else ""
            ),
            reference_target_version=(
                reference_target.version if reference_target else ""
            ),
            reference_target_composition_basis=(
                reference_target.composition_basis if reference_target else ""
            ),
            reference_target_evidence_sha256=(
                list(reference_target.evidence_sha256) if reference_target else []
            ),
            reference_target_comparison_kind=(
                "evidenced_composition_physsim_target"
                if reference_target
                else "no_evidenced_target"
            ),
            simulated_similarity_score=simulation.mean,
            simulation_status=simulation.status,
            simulation_confidence=simulation.confidence,
            simulation_draws=simulation.draws,
            simulation_components=simulation.components,
            simulation_flags=list(simulation.flags),
            simulation_p05=simulation.p05,
            simulation_p95=simulation.p95,
            scientific_twin_status=scientific_twin.status,
            scientific_model_version=scientific_twin.model_version,
            scientific_data_coverage_percent=scientific_twin.scientific_data_coverage_percent,
            calculated_structure_coverage_percent=scientific_twin.calculated_structure_coverage_percent,
            molecular_descriptor_coverage_percent=scientific_twin.molecular_descriptor_coverage_percent,
            temporal_similarity_score=scientific_twin.temporal_similarity_mean,
            minimum_temporal_similarity=scientific_twin.minimum_temporal_similarity,
            temporal_profile=[
                asdict(point) for point in scientific_twin.temporal_points
            ],
            ingredient_temporal_profile=[
                asdict(profile)
                for profile in scientific_twin.ingredient_temporal_profiles
            ],
            temporal_timepoints_minutes=[
                point.minutes for point in scientific_twin.temporal_points
            ],
            temporal_concentration_basis=(
                scientific_twin.temporal_concentration_basis
            ),
            temporal_model_claim_boundary=(
                scientific_twin.temporal_model_claim_boundary
            ),
            scientific_flags=list(scientific_twin.flags),
            vapor_pressure_coverage_percent=scientific_twin.vapor_pressure_coverage_percent,
            odor_threshold_coverage_percent=scientific_twin.odor_threshold_coverage_percent,
            model_applicability_percent=scientific_twin.model_applicability_percent,
            temporal_similarity_p05=scientific_twin.temporal_similarity_p05,
            temporal_similarity_p95=scientific_twin.temporal_similarity_p95,
            minimum_temporal_similarity_p05=scientific_twin.minimum_temporal_similarity_p05,
            simulation_only_approved=scientific_twin.simulation_only_approved,
            scientific_monte_carlo_draws=scientific_twin.monte_carlo_draws,
            scientific_model_domain_passed=scientific_twin.model_domain_passed,
            scientific_uncertainty_kind=scientific_twin.uncertainty_kind,
            scientific_sampling_version=scientific_twin.uncertainty_sampling_version,
            physsim_status=physsim.status,
            physsim_model_version=physsim.model_version,
            physsim_similarity_score=physsim.similarity,
            physsim_minimum_temporal_similarity=physsim.minimum_temporal_similarity,
            physsim_descriptor_coverage_percent=physsim.descriptor_coverage_percent,
            physsim_vapor_pressure_coverage_percent=physsim.vapor_pressure_coverage_percent,
            physsim_odor_threshold_coverage_percent=physsim.odor_threshold_coverage_percent,
            physsim_applicability_percent=physsim.model_applicability_percent,
            physsim_target_ingredient_ids=list(physsim.target_ingredient_ids),
            physsim_temporal_profile=[
                asdict(point) for point in physsim.temporal_points
            ],
            physsim_flags=list(physsim.flags),
            physsim_comparison_target_status=physsim.comparison_target_status,
            physsim_comparison_authorized=physsim.comparison_authorized,
            physsim_deterministic_similarity_score=physsim.deterministic_similarity,
            physsim_learned_r2_status=physsim.learned_r2_status,
            physsim_learned_r2_similarity_score=physsim.learned_r2_similarity,
            physsim_learned_r2_applicability_percent=physsim.learned_r2_applicability_percent,
            physsim_learned_r2_candidate_structure_coverage_percent=(
                physsim.learned_r2_candidate_structure_coverage_percent
            ),
            physsim_learned_r2_target_structure_coverage_percent=(
                physsim.learned_r2_target_structure_coverage_percent
            ),
            physsim_learned_r2_descriptor_domain_coverage_percent=(
                physsim.learned_r2_descriptor_domain_coverage_percent
            ),
            physsim_learned_r2_approved_weight=physsim.learned_r2_approved_weight,
            physsim_learned_r2_applied_weight=physsim.learned_r2_applied_weight,
            physsim_learned_r2_neutral_similarity_percent=(
                physsim.learned_r2_neutral_similarity_percent
            ),
            physsim_learned_r2_centered_score_adjustment=(
                physsim.learned_r2_centered_score_adjustment
            ),
            physsim_learned_r2_checkpoint_sha256=physsim.learned_r2_checkpoint_sha256,
            physsim_learned_r2_member_predictions=list(
                physsim.learned_r2_member_predictions
            ),
            physsim_learned_r2_member_disagreement_percent=(
                physsim.learned_r2_member_disagreement_percent
            ),
            physsim_learned_r2_prediction_interval_lower_percent=(
                physsim.learned_r2_prediction_interval_lower_percent
            ),
            physsim_learned_r2_prediction_interval_upper_percent=(
                physsim.learned_r2_prediction_interval_upper_percent
            ),
            physsim_learned_r2_ensemble_manifest_sha256=(
                physsim.learned_r2_ensemble_manifest_sha256
            ),
            concentration_response_status=physsim.concentration_response_status,
            concentration_response_similarity_score=(
                physsim.concentration_response_similarity
            ),
            concentration_response_coverage_percent=(
                physsim.concentration_response_coverage_percent
            ),
            concentration_response_applied_weight=(
                physsim.concentration_response_applied_weight
            ),
            olfactory_validation_status=_olfactory_validation_status(
                sensory,
                human_calibration.status,
                brief.constraints.target_similarity,
            ),
            actual_olfactory_similarity_score=(
                sensory.mean_similarity if sensory else None
            ),
            actual_olfactory_lower_bound_95=(
                sensory.lower_confidence_bound_95 if sensory else None
            ),
            release_evidence_status=release_assessment.status,
            external_regulatory_signoff_valid=release_assessment.passed,
            release_spec_id=(release_spec.release_spec_id if release_spec else ""),
            release_scope_verified=release_assessment.scope_verified,
            evidence_scope_id=evidence_scope_id,
            candidate_variants_evaluated=len(evaluated_variants) + pool_variants_evaluated,
            ingredient_sets_evaluated=1 + len(replacement_sets) + len(guide_sets) + len(pool_sets_evaluated),
            ingredient_swaps_evaluated=unguided_swap_count,
            physics_guided_search=len(evaluated_variants) > 1,
            physics_search_objective=round(
                physics_objective(
                    (selected_variant, scientific_twin, physsim, simulation)
                ),
                4,
            ),
            catalog_profile_rank=int(capability["profile_rank"]),
            catalog_profile_dimension_count=int(capability["profile_dimension_count"]),
            catalog_unsupported_dimensions=list(capability["unsupported_dimensions"]),
            perceptual_prediction_status=human_calibration.status,
            human_discrimination_probability=(
                human_calibration.discrimination_probability
            ),
            human_discrimination_lower_95=human_calibration.lower_95,
            human_discrimination_upper_95=human_calibration.upper_95,
            human_calibration_applicability_percent=(
                human_calibration.applicability_percent
            ),
            human_calibration_artifact_id=human_calibration.artifact_id,
            human_calibration_flags=list(human_calibration.flags),
            human_similarity_90_claim_authorized=(
                human_calibration.similarity_90_claim_authorized
            ),
        )
        if guide is not None:
            guidance_report = guide.report(
                baseline_guidance, selected_guidance,
                changed=formula_fingerprint(original_choice[0][0]) != formula_fingerprint(selected_variant[0]),
                variants=guidance_variants_added,
            )
            guidance_report["recipe_returned"] = bool(result.recipe)
            guidance_report["candidate_changed"] = guidance_report["recipe_changed"]
            if not result.recipe:
                guidance_report["recipe_changed"] = False
                guidance_report["status"] = "candidate_only_no_approved_recipe"
            result = attach_guidance(result, guidance_report)
            result.limitations.append(
                "관능모델 가이드는 원료 선택·배합비 검색에 사용한 연구용 가산 혼합 추정입니다. "
                "지정 용매 시나리오는 실제 제품 베이스 검증이 아니며, 점수는 실제 향 유사도/90% 인증이 아닙니다."
            )
        result._full_profile_search = {
            "enabled": reference_target is None,
            "additional_variants": full_profile_variants_added,
            "baseline_score": full_before, "selected_score": full_after,
            "screening_baseline_score": screening_full_before, "screening_selected_score": screening_full_after,
            "screening_draws": screening_draws, "final_comparison_draws": scientific_twin.monte_carlo_draws,
            "candidate_changed": formula_fingerprint(before_full_choice[0][0]) != formula_fingerprint(selected_variant[0]),
            "legacy_request_gate_preserved": True,
            "not_human_perception_validation": True,
            "full_pool_search": pool_diagnostic,
        }
        if retired_blends.rejected:
            result._full_profile_search['retired_blends'] = retired_blends.report()
        return result

    def create_recipe_with_target_profile(
        self,
        natural_language_brief: str,
        constraints: RecipeConstraints,
        target_profile: dict[str, float],
    ) -> RecipeResult:
        """Generate against an explicit profile produced by a versioned edit."""

        return self.create_recipe(
            natural_language_brief,
            constraints,
            target_profile_override=target_profile,
        )
