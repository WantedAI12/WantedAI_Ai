"""End-to-end natural-language perfumery recipe service."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import date
from types import TracebackType
from typing import Self

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
from .registry_activation import REGISTRY_CONDITIONAL_DATA_SOURCE
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
    "가격·재고·COA·SDS·IFRA 증명과 규제 값은 검증된 공급사 원료 자료가 연결된 경우에만 사용합니다.",
    "상용 출시는 안정성, 용기 적합성, 실제 배치, 시장별 전문가 검토와 서명된 외부 승인이 별도로 필요합니다.",
    "과거 향수 DB의 노트 정보는 참고 신호이며 측정된 분자 조성으로 취급하지 않습니다.",
    "물리 모델은 불완전한 물성·역치 자료와 모델 가정을 포함하므로 실제 헤드스페이스 측정을 대체하지 않습니다.",
    "학습된 R2 체크포인트는 분자 혼합물 유사도 프록시이며 완성 처방의 인간 관능 검증값이 아닙니다.",
    "출처·라이선스·측정 조건이 불명확한 데이터는 물성·규제·공급 사실로 승격하지 않습니다.",
    "산업 레지스트리 조건부 trace 후보는 공개 냄새 기술 기반 R&D 가설이며 독립 안전·공급사 승인 원료가 아닙니다.",
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
    ):
        self.promotion_bundle = (
            promotion_bundle
            if promotion_bundle is not None
            else PromotionActivationBundle.from_environment()
        )
        base_catalog = catalog or IngredientCatalog.load_builtin()
        base_catalog = self.promotion_bundle.merge_catalog(base_catalog)
        self.odor_store = odor_store
        self.catalog = (
            odor_store.apply_to_catalog(base_catalog) if odor_store else base_catalog
        )
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
        self.temporal_simulator = TemporalMixtureSimulator()
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
        integer("max_ingredients", constraints.max_ingredients, 3, 50)
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

    def create_recipe(
        self,
        natural_language_brief: str,
        constraints: RecipeConstraints | None = None,
        as_of: date | None = None,
        *,
        target_profile_override: dict[str, float] | None = None,
    ) -> RecipeResult:
        as_of = as_of or date.today()
        if constraints is not None:
            self._validate_constraints(constraints)
        brief = self.parser.parse(natural_language_brief, constraints)
        self._validate_constraints(brief.constraints)
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
            brief = replace(
                brief,
                target_profile=target_profile,
                desired_dimensions=sorted(
                    dimension
                    for dimension, value in target_profile.items()
                    if value >= 0.01
                ),
                avoided_dimensions=sorted(
                    dimension
                    for dimension in brief.avoided_dimensions
                    if target_profile.get(dimension, 0.0) <= 1e-12
                ),
            )
        candidates, rejected = self.screen.screen(
            self.catalog,
            brief,
            supplier_registry=self.supplier_registry,
            as_of=as_of,
        )

        for pyramid in brief.pyramid_ratios:
            if not any(item.pyramid == pyramid for item in candidates):
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

        try:
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
                fingerprint = formula_fingerprint(variant[0])
                if fingerprint not in variant_fingerprints:
                    variant_fingerprints.add(fingerprint)
                    formula_variants.append(variant)

            for _, concentration_scale, temporal_weights in search_specs[
                :requested_population
            ]:
                perceptual_factors = (
                    self.temporal_simulator.ingredient_perceptual_factors(
                        candidates,
                        self.scientific_store,
                        max(
                            0.1,
                            brief.constraints.product_concentration_percent
                            * concentration_scale,
                        ),
                        timepoint_weights=temporal_weights,
                    )
                )
                add_variant(
                    self.optimizer.optimize(
                        candidates,
                        brief,
                        perceptual_factors=perceptual_factors,
                        formula_objective=formula_context_objective,
                    )
                )
            add_variant(
                self.optimizer.optimize(
                    candidates,
                    brief,
                    formula_objective=formula_context_objective,
                )
            )
        except NoFeasibleFormula as error:
            return self._blocked_result(brief, str(error), rejected, as_of)
        evaluated_variants = []
        screening_draws = min(64, max(30, brief.constraints.simulation_draws))
        for variant in formula_variants:
            variant_lines = variant[0]
            variant_twin = self.temporal_simulator.evaluate(
                variant_lines,
                ingredient_map,
                brief,
                self.scientific_store,
                draws=screening_draws,
            )
            variant_physsim = self.physsim_engine.evaluate(
                variant_lines,
                ingredient_map,
                brief,
                self.scientific_store,
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

        selected_variant, scientific_twin, physsim, simulation = max(
            evaluated_variants,
            key=lambda item: (
                item[3].status == "evidenced_nonhuman_pass",
                physics_objective(item),
                item[0][1],
            ),
        )
        if brief.constraints.simulation_draws > screening_draws:
            selected_lines = selected_variant[0]
            scientific_twin = self.temporal_simulator.evaluate(
                selected_lines,
                ingredient_map,
                brief,
                self.scientific_store,
                draws=brief.constraints.simulation_draws,
            )
            simulation = self.simulation_engine.evaluate(
                selected_lines,
                ingredient_map,
                brief,
                self.corpus,
                target=brief.constraints.target_similarity,
                draws=brief.constraints.simulation_draws,
                calibration=self.calibration,
                scientific_twin=scientific_twin,
                physsim=physsim,
                target_evidenced=reference_target is not None,
            )
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
        cost_ok = cost <= brief.constraints.max_formula_cost_per_kg
        semantic_ok = raw_similarity + 1e-8 >= brief.constraints.target_similarity
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
                    "전체 산업 레지스트리에서 선별한 조건부 trace 원료가 포함된 "
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
                "전체 산업 레지스트리에서 선별한 risk-tier-2 trace 후보가 "
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

        return RecipeResult(
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
            molecular_descriptor_coverage_percent=scientific_twin.molecular_descriptor_coverage_percent,
            temporal_similarity_score=scientific_twin.temporal_similarity_mean,
            minimum_temporal_similarity=scientific_twin.minimum_temporal_similarity,
            temporal_profile=[
                asdict(point) for point in scientific_twin.temporal_points
            ],
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
            candidate_variants_evaluated=len(evaluated_variants),
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
