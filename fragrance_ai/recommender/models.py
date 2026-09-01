"""Shared data models for the constrained perfumery recommender."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


SCENT_DIMENSIONS: tuple[str, ...] = (
    "citrus",
    "fresh",
    "clean",
    "green",
    "aquatic",
    "floral",
    "rose",
    "white_floral",
    "fruity",
    "spicy",
    "aromatic",
    "woody",
    "amber",
    "musky",
    "gourmand",
    "powdery",
    "smoky",
    "leathery",
    "earthy",
)

# These perceptual controls are deliberately kept separate from the relative
# odor-family simplex above.  Normalizing them together would erase absolute
# intensity and conflate trigeminal/texture judgements with odor identity.
TEXTURE_DIMENSIONS: tuple[str, ...] = (
    "transparent",
    "dense",
    "dry",
    "creamy",
    "soft",
)
TRIGEMINAL_DIMENSIONS: tuple[str, ...] = (
    "cooling",
    "warming",
    "tingling",
)
TEMPORAL_DIMENSIONS: tuple[str, ...] = ("opening", "heart", "drydown")

PYRAMID_LEVELS: tuple[str, ...] = ("top", "heart", "base")


def normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(profile.get(key, 0.0))) for key in SCENT_DIMENSIONS}
    total = sum(cleaned.values())
    if total <= 0:
        return cleaned
    return {key: value / total for key, value in cleaned.items()}


def profile_vector(profile: dict[str, float]) -> np.ndarray:
    normalized = normalize_profile(profile)
    return np.asarray([normalized[key] for key in SCENT_DIMENSIONS], dtype=float)


@dataclass(frozen=True)
class Ingredient:
    ingredient_id: str
    name: str
    aliases: tuple[str, ...]
    cas_number: str | None
    pyramid: str
    profile: dict[str, float]
    price_per_kg: float
    availability: float
    rarity: str
    risk_tier: int
    odor_impact: float
    max_concentrate_percent: float
    formulation_ready: bool
    blocked: bool = False
    blocked_reason: str | None = None
    eu_allergens: tuple[str, ...] = ()
    data_source: str = "curated-workspace-v1"
    currency: str = "USD_estimate"
    density_g_ml: float | None = None
    active_strength_percent: float = 100.0
    carrier: str | None = None
    solubility: tuple[str, ...] = ()
    oxidation_risk: str = "unknown"
    discoloration_risk: str = "unknown"
    shelf_life_months: int | None = None
    data_verified_on: str | None = None
    approved_formulation_scopes: tuple[str, ...] = ()
    approval_expires_at: str | None = None
    promotion_artifact_id: str | None = None
    promotion_registry_sha256: str | None = None

    def vector(self) -> np.ndarray:
        return profile_vector(self.profile)

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def as_supplied_cap_percent(self) -> float:
        """Convert an active-material cap to an as-supplied material cap."""
        strength = max(0.000001, self.active_strength_percent / 100.0)
        return min(100.0, self.max_concentrate_percent / strength)


@dataclass
class RecipeConstraints:
    max_ingredient_price_per_kg: float = 300.0
    max_formula_cost_per_kg: float = 180.0
    min_availability: float = 0.75
    max_risk_tier: int = 1
    target_similarity: float = 90.0
    product_concentration_percent: float = 15.0
    finished_volume_ml: float = 50.0
    max_ingredients: int = 12
    allow_rare: bool = False
    explicit_bans: set[str] = field(default_factory=set)
    validation_level: str = "prototype"
    target_region: str = "EU"
    product_category: str = "eau_de_parfum"
    finished_batch_mass_g: float = 50.0
    product_density_g_ml: float | None = None
    max_supplier_lead_time_days: int = 30
    max_supplier_moq_kg: float = 5.0
    min_panelists: int = 12
    min_expert_panelists: int = 3
    minimum_realism_score: float = 65.0
    simulation_draws: int = 200
    # Synthetic draws are a diagnostic, not evidence of human similarity.
    # A caller can explicitly require the evidenced-reference comparison gate,
    # but text-only requests must remain able to return an unvalidated R&D
    # candidate instead of turning an internal proxy into a 90% claim.
    require_simulation_pass: bool = False
    simulation_min_applicability_percent: float = 55.0
    simulation_max_uncertainty_width: float = 12.0
    physsim_min_applicability_percent: float = 55.0
    commercial_min_scientific_coverage_percent: float = 80.0
    commercial_min_temporal_similarity: float = 90.0
    physics_search_population: int = 6
    enable_semantic_ontology: bool = True
    enable_concentration_response: bool = True
    enable_learned_r2: bool = True
    enable_registry_trace_candidates: bool = False
    reference_target_id: str = ""
    require_evidenced_olfactory_target: bool = False
    require_catalog_dimension_support: bool = True
    minimum_dimension_material_strength: float = 0.35
    surrogate_objective_weight: float = 0.25
    # These fields are intentionally blank by default.  A commercial approval
    # must name the exact product base, packaging, artifact versions and
    # supplier-lot documents; defaults would permit an approval to drift.
    commercial_product_base_id: str = ""
    commercial_packaging_id: str = ""
    commercial_rule_pack_version: str = ""
    commercial_data_version: str = ""
    commercial_model_version: str = ""
    commercial_supplier_evidence: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass
class ScentBrief:
    original_text: str
    target_profile: dict[str, float]
    desired_dimensions: list[str]
    avoided_dimensions: list[str]
    requested_ingredients: list[str]
    excluded_ingredients: list[str]
    intensity: str
    pyramid_ratios: dict[str, float]
    constraints: RecipeConstraints
    semantic_backend: str = "keyword_rules"
    semantic_confidence: float = 0.0
    ontology_version: str = ""
    ontology_concepts: list[str] = field(default_factory=list)
    recognized_descriptors: list[str] = field(default_factory=list)
    avoided_descriptors: list[str] = field(default_factory=list)
    descriptor_projection_version: str = ""
    descriptor_projection_claim_boundary: str = ""
    absolute_intensity_target: float = 0.50
    diffusion_target: float = 0.50
    texture_profile: dict[str, float] = field(default_factory=dict)
    trigeminal_profile: dict[str, float] = field(default_factory=dict)
    temporal_emphasis: dict[str, float] = field(
        default_factory=lambda: {"opening": 0.25, "heart": 0.40, "drydown": 0.35}
    )
    perceptual_intent_version: str = "perceptual-intent-1.0"


@dataclass
class RecipeLine:
    ingredient_id: str
    name: str
    pyramid: str
    concentrate_percent: float
    finished_product_percent: float
    volume_ml_for_batch: float | None
    price_per_kg: float
    availability: float
    risk_tier: int
    reason: str
    mass_g_for_batch: float = 0.0
    active_material_percent: float = 0.0
    active_mass_g_for_batch: float = 0.0
    density_g_ml: float | None = None
    active_strength_percent: float = 100.0
    carrier: str | None = None
    data_source: str = "curated-workspace-v1"
    approved_formulation_scopes: tuple[str, ...] = ()
    approval_expires_at: str | None = None
    promotion_artifact_id: str | None = None


@dataclass
class SafetyReport:
    internal_gate_passed: bool
    status: str
    active_ifra_amendment: str
    standards_checked_on: str
    standards_review_due: str
    regulatory_data_complete: bool
    manufacturing_ready: bool
    violations: list[str]
    warnings: list[str]
    eu_label_declarations: list[str]
    potential_eu_allergens: list[str] = field(default_factory=list)
    allergen_quantification_complete: bool = False
    evidence_coverage_percent: float = 0.0
    missing_documents: list[str] = field(default_factory=list)
    target_region: str = "EU"
    product_category: str = "eau_de_parfum"
    validation_level: str = "prototype"
    audit_id: str = ""
    internal_evidence_complete: bool = False


@dataclass
class ManufacturingLine:
    ingredient_id: str
    name: str
    as_supplied_mass_g: float
    active_material_mass_g: float
    carrier_mass_g: float
    estimated_volume_ml: float | None
    active_strength_percent: float
    recommended_weighing_tolerance_g: float


@dataclass
class ManufacturingPlan:
    basis: str
    finished_batch_mass_g: float
    fragrance_concentrate_mass_g: float
    product_base_mass_g: float
    total_as_supplied_material_mass_g: float
    recommended_balance_readability_g: float
    lines: list[ManufacturingLine]
    process_steps: list[str]
    required_stability_tests: list[str]
    stability_status: str
    ready_for_lab_trial: bool
    warnings: list[str]
    ready_for_manufacture: bool = False
    readiness_blockers: list[str] = field(default_factory=list)
    product_base_profile_version: str = ""
    packaging_profile_version: str = ""


@dataclass
class RecipeResult:
    status: str
    message: str
    brief: ScentBrief
    similarity_score: float
    similarity_kind: str
    recipe: list[RecipeLine]
    closest_candidate: list[RecipeLine]
    achieved_profile: dict[str, float]
    estimated_concentrate_cost_per_kg: float
    historical_support_score: float
    catalog_stats: dict[str, Any]
    rejected_candidate_counts: dict[str, int]
    safety: SafetyReport
    limitations: list[str]
    formula_id: str = ""
    raw_similarity_score: float = 0.0
    sensory_similarity_score: float | None = None
    sensory_panel_size: int = 0
    sensory_validation_status: str = "not_tested"
    manufacturing_plan: ManufacturingPlan | None = None
    realism_score: float = 0.0
    realism_kind: str = "engineering_plausibility_not_sensory_accuracy"
    accord_family: str = "unknown"
    odor_profile_coverage_percent: float = 0.0
    confidence: str = "heuristic"
    realism_components: dict[str, float] = field(default_factory=dict)
    realism_flags: list[str] = field(default_factory=list)
    historical_reference_matches: list[dict[str, Any]] = field(default_factory=list)
    reference_molecular_composition_status: str = "not_available"
    reference_molecular_composition_claim_boundary: str = (
        "Historical note co-occurrence is not a measured molecular formula."
    )
    reference_target_id: str = ""
    reference_target_version: str = ""
    reference_target_composition_basis: str = ""
    reference_target_evidence_sha256: list[str] = field(default_factory=list)
    reference_target_comparison_kind: str = "no_evidenced_target"
    simulated_similarity_score: float = 0.0
    simulation_status: str = "not_run"
    simulation_confidence: str = "not_run"
    simulation_draws: int = 0
    simulation_components: dict[str, float] = field(default_factory=dict)
    simulation_flags: list[str] = field(default_factory=list)
    simulation_p05: float = 0.0
    simulation_p95: float = 0.0
    scientific_twin_status: str = "not_run"
    scientific_model_version: str = ""
    scientific_data_coverage_percent: float = 0.0
    molecular_descriptor_coverage_percent: float = 0.0
    temporal_similarity_score: float = 0.0
    minimum_temporal_similarity: float = 0.0
    temporal_profile: list[dict[str, Any]] = field(default_factory=list)
    scientific_flags: list[str] = field(default_factory=list)
    vapor_pressure_coverage_percent: float = 0.0
    odor_threshold_coverage_percent: float = 0.0
    model_applicability_percent: float = 0.0
    temporal_similarity_p05: float = 0.0
    temporal_similarity_p95: float = 0.0
    minimum_temporal_similarity_p05: float = 0.0
    simulation_only_approved: bool = False
    scientific_monte_carlo_draws: int = 0
    scientific_model_domain_passed: bool = False
    scientific_uncertainty_kind: str = ""
    physsim_status: str = "not_run"
    physsim_model_version: str = ""
    physsim_similarity_score: float = 0.0
    physsim_minimum_temporal_similarity: float = 0.0
    physsim_descriptor_coverage_percent: float = 0.0
    physsim_vapor_pressure_coverage_percent: float = 0.0
    physsim_odor_threshold_coverage_percent: float = 0.0
    physsim_applicability_percent: float = 0.0
    physsim_target_ingredient_ids: list[str] = field(default_factory=list)
    physsim_temporal_profile: list[dict[str, Any]] = field(default_factory=list)
    physsim_flags: list[str] = field(default_factory=list)
    physsim_comparison_target_status: str = "not_assessed"
    physsim_comparison_authorized: bool = False
    physsim_deterministic_similarity_score: float = 0.0
    physsim_learned_r2_status: str = "not_run"
    physsim_learned_r2_similarity_score: float | None = None
    physsim_learned_r2_applicability_percent: float = 0.0
    physsim_learned_r2_candidate_structure_coverage_percent: float = 0.0
    physsim_learned_r2_target_structure_coverage_percent: float = 0.0
    physsim_learned_r2_descriptor_domain_coverage_percent: float = 0.0
    physsim_learned_r2_approved_weight: float = 0.0
    physsim_learned_r2_applied_weight: float = 0.0
    physsim_learned_r2_neutral_similarity_percent: float = 50.0
    physsim_learned_r2_centered_score_adjustment: float = 0.0
    physsim_learned_r2_checkpoint_sha256: str = ""
    physsim_learned_r2_member_predictions: list[float] = field(default_factory=list)
    physsim_learned_r2_member_disagreement_percent: float = 0.0
    physsim_learned_r2_prediction_interval_lower_percent: float = 0.0
    physsim_learned_r2_prediction_interval_upper_percent: float = 0.0
    physsim_learned_r2_ensemble_manifest_sha256: str = ""
    concentration_response_status: str = "not_run"
    concentration_response_similarity_score: float | None = None
    concentration_response_coverage_percent: float = 0.0
    concentration_response_applied_weight: float = 0.0
    olfactory_validation_status: str = "abstained_no_evidenced_target"
    actual_olfactory_similarity_score: float | None = None
    actual_olfactory_lower_bound_95: float | None = None
    release_evidence_status: str = "not_assessed"
    external_regulatory_signoff_valid: bool = False
    release_spec_id: str = ""
    release_scope_verified: bool = False
    evidence_scope_id: str = ""
    candidate_variants_evaluated: int = 0
    physics_guided_search: bool = False
    physics_search_objective: float = 0.0
    catalog_profile_rank: int = 0
    catalog_profile_dimension_count: int = 0
    catalog_unsupported_dimensions: list[str] = field(default_factory=list)
    perceptual_prediction_status: str = "abstained_no_evidenced_target"
    human_discrimination_probability: float | None = None
    human_discrimination_lower_95: float | None = None
    human_discrimination_upper_95: float | None = None
    human_calibration_applicability_percent: float = 0.0
    human_calibration_artifact_id: str = ""
    human_calibration_flags: list[str] = field(default_factory=list)
    human_similarity_90_claim_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["brief"]["constraints"]["explicit_bans"] = sorted(
            self.brief.constraints.explicit_bans
        )
        return result
