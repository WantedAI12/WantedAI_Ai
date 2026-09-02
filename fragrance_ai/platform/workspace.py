"""Project, formula-version, revision, accord, and comparison workflows."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Iterator

from ..recommender.brief_parser import apply_relative_revision_profile
from ..recommender.catalog import IngredientCatalog
from ..recommender.models import RecipeConstraints, normalize_profile, profile_vector
from ..recommender.optimizer import cosine_similarity_percent
from ..recommender.safety import PRODUCT_CATEGORY_MAP, VALIDATION_LEVELS
from .store import WorkspaceStore, _bounded_text


_CONSTRAINT_FIELDS = {item.name for item in fields(RecipeConstraints)}
_METRICS = (
    "similarity_score",
    "simulated_similarity_score",
    "simulation_p05",
    "realism_score",
    "estimated_concentrate_cost_per_kg",
    "scientific_data_coverage_percent",
)
_BOOLEAN_CONSTRAINTS = (
    "allow_rare",
    "require_simulation_pass",
    "enable_semantic_ontology",
    "enable_concentration_response",
    "enable_learned_r2",
    "enable_registry_trace_candidates",
    "experimental_disable_safety",
    "require_evidenced_olfactory_target",
    "require_catalog_dimension_support",
)


def _number_in_range(
    value: object,
    field: str,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    lower_ok = parsed >= minimum if minimum_inclusive else parsed > minimum
    if not lower_ok or parsed > maximum:
        operator = "at least" if minimum_inclusive else "greater than"
        raise ValueError(
            f"{field} must be {operator} {minimum:g} and at most {maximum:g}"
        )
    if integer and not parsed.is_integer():
        raise ValueError(f"{field} must be an integer")
    return parsed


def _validate_constraints(result: RecipeConstraints) -> None:
    for field in _BOOLEAN_CONSTRAINTS:
        if not isinstance(getattr(result, field), bool):
            raise ValueError(f"{field} must be boolean")

    numeric_ranges = {
        "max_ingredient_price_per_kg": (0, 10_000_000, False, True),
        "max_formula_cost_per_kg": (0, 10_000_000, False, True),
        "min_availability": (0, 1, False, True),
        "max_risk_tier": (0, 3, True, True),
        "target_similarity": (0, 100, False, False),
        "product_concentration_percent": (0, 100, False, False),
        "finished_volume_ml": (0, 100_000, False, False),
        "max_ingredients": (3, 30, True, True),
        "finished_batch_mass_g": (0, 10_000, False, False),
        "max_supplier_lead_time_days": (0, 3_650, True, True),
        "max_supplier_moq_kg": (0, 1_000_000, False, True),
        "min_panelists": (1, 10_000, True, True),
        "min_expert_panelists": (0, 10_000, True, True),
        "minimum_realism_score": (0, 100, False, True),
        "simulation_draws": (64, 2_000, True, True),
        "simulation_min_applicability_percent": (0, 100, False, True),
        "simulation_max_uncertainty_width": (0, 100, False, True),
        "physsim_min_applicability_percent": (0, 100, False, True),
        "commercial_min_scientific_coverage_percent": (0, 100, False, True),
        "commercial_min_temporal_similarity": (0, 100, False, True),
        "physics_search_population": (2, 6, True, True),
        "minimum_dimension_material_strength": (0, 1, False, False),
        "surrogate_objective_weight": (0, 0.5, False, True),
    }
    for field, (minimum, maximum, integer, inclusive) in numeric_ranges.items():
        _number_in_range(
            getattr(result, field),
            field,
            minimum,
            maximum,
            integer=integer,
            minimum_inclusive=inclusive,
        )
    if result.product_density_g_ml is not None:
        _number_in_range(
            result.product_density_g_ml,
            "product_density_g_ml",
            0,
            100,
            minimum_inclusive=False,
        )
    if int(result.min_expert_panelists) > int(result.min_panelists):
        raise ValueError("min_expert_panelists cannot exceed min_panelists")
    if result.validation_level not in VALIDATION_LEVELS:
        raise ValueError(
            "validation_level must be one of " + ", ".join(sorted(VALIDATION_LEVELS))
        )
    if result.product_category not in PRODUCT_CATEGORY_MAP:
        raise ValueError("unsupported product_category")
    if (
        not isinstance(result.target_region, str)
        or not result.target_region.strip()
        or len(result.target_region) > 32
    ):
        raise ValueError(
            "target_region must be non-empty text of at most 32 characters"
        )
    if not isinstance(result.commercial_supplier_evidence, dict):
        raise ValueError("commercial_supplier_evidence must be an object")
    for field in (
        "reference_target_id",
        "commercial_product_base_id",
        "commercial_packaging_id",
        "commercial_rule_pack_version",
        "commercial_data_version",
        "commercial_model_version",
    ):
        value = getattr(result, field)
        if not isinstance(value, str) or len(value) > 256:
            raise ValueError(f"{field} must be text of at most 256 characters")


def constraints_from_payload(
    payload: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
) -> RecipeConstraints:
    if base is not None and not isinstance(base, dict):
        raise ValueError("base constraints must be an object")
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("constraints must be an object")
    raw = dict(base or {})
    raw.update(payload or {})
    unknown = sorted(set(raw) - _CONSTRAINT_FIELDS)
    if unknown:
        raise ValueError("unknown recipe constraints: " + ", ".join(unknown))
    if "explicit_bans" in raw:
        bans = raw["explicit_bans"]
        if not isinstance(bans, (list, tuple, set)) or not all(
            isinstance(item, str) for item in bans
        ):
            raise ValueError("explicit_bans must be a list of ingredient names")
        raw["explicit_bans"] = set(bans)
    result = RecipeConstraints(**raw)
    _validate_constraints(result)
    return result


class FormulaWorkspaceService:
    """Orchestrates the AI core without granting storage cross-tenant access."""

    def __init__(
        self,
        *,
        store: WorkspaceStore,
        ai_factory: Callable[[], Any],
        catalog: IngredientCatalog | None = None,
        ai_instance: Any | None = None,
    ):
        self.store = store
        self.ai_factory = ai_factory
        self.ai_instance = ai_instance
        self.catalog = catalog or IngredientCatalog.load_builtin()
        self._ingredients = {
            item.ingredient_id: item for item in self.catalog.ingredients
        }

    @contextmanager
    def _ai_instance(self) -> Iterator[Any]:
        """Bound heavyweight model repositories to one workspace operation."""
        if self.ai_instance is not None:
            yield self.ai_instance
            return
        ai = self.ai_factory()
        try:
            yield ai
        finally:
            close = getattr(ai, "close", None)
            if callable(close):
                close()

    def catalog_payload(self) -> dict[str, Any]:
        ingredients = []
        for item in self.catalog.ingredients:
            if not item.formulation_ready or item.blocked:
                continue
            ingredients.append(
                {
                    "ingredient_id": item.ingredient_id,
                    "name": item.name,
                    "pyramid": item.pyramid,
                    "profile": item.profile,
                    "price_per_kg": item.price_per_kg,
                    "availability": item.availability,
                    "risk_tier": item.risk_tier,
                    "max_concentrate_percent": item.as_supplied_cap_percent(),
                    "active_strength_percent": item.active_strength_percent,
                    "odor_impact": item.odor_impact,
                    "density_g_ml": item.density_g_ml,
                    "carrier": item.carrier,
                }
            )
        return {
            "catalog_version": self.catalog.stats()["catalog_version"],
            "ingredients": ingredients,
        }

    def generate_formula(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        project_id: str,
        brief: str,
        constraints: dict[str, Any] | None,
        name: str,
        kind: str = "formula",
        source_job_id: str | None = None,
        before_persist: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        clean_brief = _bounded_text(brief, "brief", 4000)
        recipe_constraints = constraints_from_payload(constraints)
        with self._ai_instance() as ai:
            result = ai.create_recipe(clean_brief, recipe_constraints).to_dict()
        formula = None
        if result.get("recipe"):
            if before_persist is not None:
                before_persist()
            formula = self.store.create_formula(
                tenant_id=tenant_id,
                project_id=project_id,
                name=name,
                kind=kind,
                payload=result,
                actor_id=actor_id,
                change_note=f"Generated from brief: {clean_brief[:240]}",
                source_job_id=source_job_id,
            )
        return {
            "result": result,
            "workspace_formula": formula.to_dict() if formula else None,
        }

    def revise_formula(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        project_id: str,
        formula_id: str,
        base_version_id: str,
        instruction: str,
        constraints: dict[str, Any] | None = None,
        source_job_id: str | None = None,
        before_persist: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        base = self.store.get_formula_version(
            tenant_id=tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            version_id=base_version_id,
        )
        if base is None:
            raise KeyError("formula version not found")
        clean_instruction = _bounded_text(instruction, "instruction", 2000)
        base_brief = str(base.payload.get("brief", {}).get("original_text", "")).strip()
        if not base_brief:
            raise ValueError("base formula has no reproducible natural-language brief")
        base_constraints = base.payload.get("brief", {}).get("constraints", {})
        if not isinstance(base_constraints, dict):
            base_constraints = {}
        recipe_constraints = constraints_from_payload(
            constraints,
            base=base_constraints,
        )
        revised_brief = f"{base_brief}\nRevision request: {clean_instruction}"
        base_profile = base.payload.get("achieved_profile")
        profile_source = "achieved_profile"
        if not isinstance(base_profile, dict) or not base_profile:
            base_profile = base.payload.get("brief", {}).get("target_profile", {})
            profile_source = "target_profile"
        revised_profile, adjustments = apply_relative_revision_profile(
            base_profile if isinstance(base_profile, dict) else {},
            clean_instruction,
        )
        with self._ai_instance() as ai:
            contextual_generator = getattr(
                ai, "create_recipe_with_target_profile", None
            )
            if adjustments and callable(contextual_generator):
                generated = contextual_generator(
                    revised_brief,
                    recipe_constraints,
                    revised_profile,
                )
            else:
                generated = ai.create_recipe(revised_brief, recipe_constraints)
        result = generated.to_dict()
        result["revision_context"] = {
            "base_version_id": base_version_id,
            "instruction": clean_instruction,
            "mode": (
                "relative_profile_edit" if adjustments else "semantic_regeneration"
            ),
            "profile_source": profile_source,
            "adjustment_multipliers": adjustments,
            "target_profile_before": base_profile,
            "target_profile_after": revised_profile if adjustments else None,
        }
        version = None
        if result.get("recipe"):
            if before_persist is not None:
                before_persist()
            version = self.store.append_formula_version(
                tenant_id=tenant_id,
                project_id=project_id,
                formula_id=formula_id,
                expected_parent_version_id=base_version_id,
                change_kind="natural_language_revision",
                change_note=clean_instruction,
                payload=result,
                actor_id=actor_id,
                source_job_id=source_job_id,
            )
        return {
            "result": result,
            "workspace_version": version.to_dict() if version else None,
        }

    def create_accord(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        project_id: str,
        brief: str,
        constraints: dict[str, Any] | None,
        name: str,
        source_job_id: str | None = None,
        before_persist: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        raw = dict(constraints or {})
        raw.setdefault("max_ingredients", 8)
        raw.setdefault("finished_batch_mass_g", 10.0)
        return self.generate_formula(
            tenant_id=tenant_id,
            actor_id=actor_id,
            project_id=project_id,
            brief=f"Accord: {brief}",
            constraints=raw,
            name=name,
            kind="accord",
            source_job_id=source_job_id,
            before_persist=before_persist,
        )

    def manual_edit(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        project_id: str,
        formula_id: str,
        base_version_id: str,
        lines: list[dict[str, Any]],
        change_note: str,
    ) -> dict[str, Any]:
        base = self.store.get_formula_version(
            tenant_id=tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            version_id=base_version_id,
        )
        if base is None:
            raise KeyError("formula version not found")
        edited = self._manual_payload(base.payload, lines)
        version = self.store.append_formula_version(
            tenant_id=tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            expected_parent_version_id=base_version_id,
            change_kind="manual_edit",
            change_note=_bounded_text(change_note, "change_note", 2000, empty=True),
            payload=edited,
            actor_id=actor_id,
        )
        return version.to_dict()

    def _manual_payload(
        self,
        base_payload: dict[str, Any],
        lines: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(lines, list) or not 1 <= len(lines) <= 30:
            raise ValueError("manual formula must contain between 1 and 30 lines")
        parsed: list[tuple[Any, float]] = []
        seen: set[str] = set()
        for line in lines:
            if not isinstance(line, dict):
                raise ValueError("each formula line must be an object")
            ingredient_id = line.get("ingredient_id")
            if not isinstance(ingredient_id, str) or ingredient_id in seen:
                raise ValueError("ingredient IDs must be unique strings")
            ingredient = self._ingredients.get(ingredient_id)
            if (
                ingredient is None
                or not ingredient.formulation_ready
                or ingredient.blocked
            ):
                raise ValueError(
                    f"ingredient is not formulation-ready: {ingredient_id}"
                )
            value = line.get("concentrate_percent")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("concentrate_percent must be numeric")
            percent = float(value)
            if not math.isfinite(percent) or percent <= 0:
                raise ValueError("concentrate_percent must be finite and positive")
            if percent > ingredient.as_supplied_cap_percent() + 1e-8:
                raise ValueError(f"ingredient cap exceeded: {ingredient.name}")
            seen.add(ingredient_id)
            parsed.append((ingredient, percent))
        total = sum(percent for _, percent in parsed)
        if not math.isclose(total, 100.0, abs_tol=0.02):
            raise ValueError("manual formula concentrate percentages must sum to 100")
        scale = 100.0 / total
        normalized: list[tuple[Any, float]] = []
        for ingredient, percent in parsed:
            adjusted = percent * scale
            if adjusted > ingredient.as_supplied_cap_percent() + 1e-8:
                raise ValueError(
                    f"normalizing to 100% would exceed ingredient cap: {ingredient.name}"
                )
            normalized.append((ingredient, round(adjusted, 6)))
        residual = round(100.0 - sum(percent for _, percent in normalized), 6)
        if residual:
            for index in range(len(normalized) - 1, -1, -1):
                ingredient, percent = normalized[index]
                adjusted = round(percent + residual, 6)
                if 0 < adjusted <= ingredient.as_supplied_cap_percent() + 1e-8:
                    normalized[index] = (ingredient, adjusted)
                    break
            else:  # pragma: no cover - bounded rounding residual invariant
                raise ValueError("manual formula could not be normalized exactly")
        parsed = normalized

        payload = copy.deepcopy(base_payload)
        constraints = payload.get("brief", {}).get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        max_risk = int(constraints.get("max_risk_tier", 1))
        product_concentration = float(
            constraints.get("product_concentration_percent", 15.0)
        )
        batch_mass = float(constraints.get("finished_batch_mass_g", 50.0))
        concentrate_mass = batch_mass * product_concentration / 100.0
        profile_accumulator: dict[str, float] = {}
        recipe_lines: list[dict[str, Any]] = []
        cost = 0.0
        for ingredient, percent in parsed:
            if ingredient.risk_tier > max_risk:
                raise ValueError(
                    f"ingredient risk tier exceeds formula constraint: {ingredient.name}"
                )
            supplied_mass = concentrate_mass * percent / 100.0
            active_mass = supplied_mass * ingredient.active_strength_percent / 100.0
            estimated_volume = (
                supplied_mass / ingredient.density_g_ml
                if ingredient.density_g_ml and ingredient.density_g_ml > 0
                else None
            )
            active_percent = percent * ingredient.active_strength_percent / 100.0
            for dimension, intensity in ingredient.profile.items():
                profile_accumulator[dimension] = profile_accumulator.get(
                    dimension, 0.0
                ) + (percent * ingredient.odor_impact * float(intensity))
            cost += percent / 100.0 * ingredient.price_per_kg
            recipe_lines.append(
                {
                    "ingredient_id": ingredient.ingredient_id,
                    "name": ingredient.name,
                    "pyramid": ingredient.pyramid,
                    "concentrate_percent": round(percent, 6),
                    "finished_product_percent": round(
                        percent * product_concentration / 100.0, 6
                    ),
                    "volume_ml_for_batch": (
                        round(estimated_volume, 6)
                        if estimated_volume is not None
                        else None
                    ),
                    "price_per_kg": ingredient.price_per_kg,
                    "availability": ingredient.availability,
                    "risk_tier": ingredient.risk_tier,
                    "reason": "manual visual formula edit",
                    "mass_g_for_batch": round(supplied_mass, 6),
                    "active_material_percent": round(active_percent, 6),
                    "active_mass_g_for_batch": round(active_mass, 6),
                    "density_g_ml": ingredient.density_g_ml,
                    "active_strength_percent": ingredient.active_strength_percent,
                    "carrier": ingredient.carrier,
                    "data_source": ingredient.data_source,
                }
            )
        achieved = normalize_profile(profile_accumulator)
        target = payload.get("brief", {}).get("target_profile", {})
        similarity = (
            cosine_similarity_percent(profile_vector(target), profile_vector(achieved))
            if isinstance(target, dict) and target
            else 0.0
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    (item["ingredient_id"], item["concentrate_percent"])
                    for item in recipe_lines
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload.update(
            {
                "status": "draft_manual_edit",
                "message": (
                    "Manual formula edit saved. Safety, physics, simulation, quality, "
                    "and release evidence must be recalculated before approval."
                ),
                "formula_id": "sha256:" + fingerprint,
                "recipe": recipe_lines,
                "closest_candidate": recipe_lines,
                "achieved_profile": achieved,
                "similarity_score": round(similarity, 4),
                "raw_similarity_score": round(similarity, 4),
                "similarity_kind": "manual_draft_semantic_profile_only",
                "estimated_concentrate_cost_per_kg": round(cost, 4),
                "historical_support_score": 0.0,
                "historical_reference_matches": [],
                "reference_target_comparison_kind": (
                    "not_recomputed_after_manual_edit"
                ),
                "simulated_similarity_score": 0.0,
                "simulation_status": "not_run_after_manual_edit",
                "simulation_confidence": "invalidated_by_formula_change",
                "simulation_draws": 0,
                "simulation_components": {},
                "simulation_flags": ["manual_edit_invalidated_previous_simulation"],
                "simulation_p05": 0.0,
                "simulation_p95": 0.0,
                "simulation_only_approved": False,
                "sensory_similarity_score": None,
                "sensory_panel_size": 0,
                "sensory_validation_status": "invalidated_by_formula_change",
                "realism_score": 0.0,
                "realism_kind": "not_run_after_manual_edit",
                "accord_family": "unvalidated_manual_draft",
                "odor_profile_coverage_percent": 0.0,
                "confidence": "invalidated_by_formula_change",
                "realism_components": {},
                "realism_flags": [
                    "manual_edit_invalidated_previous_realism_assessment"
                ],
                "scientific_twin_status": "not_run_after_manual_edit",
                "scientific_model_version": "",
                "scientific_data_coverage_percent": 0.0,
                "molecular_descriptor_coverage_percent": 0.0,
                "temporal_similarity_score": 0.0,
                "minimum_temporal_similarity": 0.0,
                "temporal_profile": [],
                "ingredient_temporal_profile": [],
                "temporal_timepoints_minutes": [],
                "temporal_concentration_basis": "",
                "temporal_model_claim_boundary": (
                    "Manual formula edit invalidated the previous temporal model."
                ),
                "scientific_flags": [
                    "manual_edit_invalidated_previous_scientific_twin"
                ],
                "vapor_pressure_coverage_percent": 0.0,
                "odor_threshold_coverage_percent": 0.0,
                "model_applicability_percent": 0.0,
                "temporal_similarity_p05": 0.0,
                "temporal_similarity_p95": 0.0,
                "minimum_temporal_similarity_p05": 0.0,
                "scientific_monte_carlo_draws": 0,
                "scientific_model_domain_passed": False,
                "scientific_uncertainty_kind": "invalidated_by_formula_change",
                "physsim_status": "not_run_after_manual_edit",
                "physsim_model_version": "",
                "physsim_similarity_score": 0.0,
                "physsim_minimum_temporal_similarity": 0.0,
                "physsim_descriptor_coverage_percent": 0.0,
                "physsim_vapor_pressure_coverage_percent": 0.0,
                "physsim_odor_threshold_coverage_percent": 0.0,
                "physsim_applicability_percent": 0.0,
                "physsim_target_ingredient_ids": [],
                "physsim_temporal_profile": [],
                "physsim_flags": ["manual_edit_invalidated_previous_physsim"],
                "physsim_comparison_target_status": "not_assessed_after_manual_edit",
                "physsim_comparison_authorized": False,
                "physsim_deterministic_similarity_score": 0.0,
                "physsim_learned_r2_status": "not_run_after_manual_edit",
                "physsim_learned_r2_similarity_score": None,
                "physsim_learned_r2_applicability_percent": 0.0,
                "physsim_learned_r2_candidate_structure_coverage_percent": 0.0,
                "physsim_learned_r2_target_structure_coverage_percent": 0.0,
                "physsim_learned_r2_descriptor_domain_coverage_percent": 0.0,
                "physsim_learned_r2_approved_weight": 0.0,
                "physsim_learned_r2_applied_weight": 0.0,
                "physsim_learned_r2_centered_score_adjustment": 0.0,
                "physsim_learned_r2_checkpoint_sha256": "",
                "physsim_learned_r2_member_predictions": [],
                "physsim_learned_r2_member_disagreement_percent": 0.0,
                "physsim_learned_r2_prediction_interval_lower_percent": 0.0,
                "physsim_learned_r2_prediction_interval_upper_percent": 0.0,
                "physsim_learned_r2_ensemble_manifest_sha256": "",
                "concentration_response_status": "not_run_after_manual_edit",
                "concentration_response_similarity_score": None,
                "concentration_response_coverage_percent": 0.0,
                "concentration_response_applied_weight": 0.0,
                "olfactory_validation_status": "manual_draft_unvalidated",
                "actual_olfactory_similarity_score": None,
                "actual_olfactory_lower_bound_95": None,
                "perceptual_prediction_status": "manual_draft_unvalidated",
                "human_discrimination_probability": None,
                "human_discrimination_lower_95": None,
                "human_discrimination_upper_95": None,
                "human_calibration_applicability_percent": 0.0,
                "human_calibration_artifact_id": "",
                "human_calibration_flags": [
                    "manual_edit_invalidated_previous_human_calibration"
                ],
                "human_similarity_90_claim_authorized": False,
                "release_evidence_status": "invalidated_by_formula_change",
                "external_regulatory_signoff_valid": False,
                "release_spec_id": "",
                "release_scope_verified": False,
                "evidence_scope_id": "",
                "manufacturing_plan": None,
                "candidate_variants_evaluated": 0,
                "physics_guided_search": False,
                "physics_search_objective": 0.0,
                "manual_edit_requires_recalculation": True,
            }
        )
        safety = payload.get("safety")
        if isinstance(safety, dict):
            warning = (
                "Manual edit invalidated the previous safety and release assessment."
            )
            safety.update(
                {
                    "internal_gate_passed": False,
                    "status": "requires_revalidation_after_manual_edit",
                    "regulatory_data_complete": False,
                    "manufacturing_ready": False,
                    "violations": [warning],
                    "warnings": [warning],
                    "eu_label_declarations": [],
                    "potential_eu_allergens": [],
                    "allergen_quantification_complete": False,
                    "evidence_coverage_percent": 0.0,
                    "missing_documents": [
                        "safety reassessment after manual formula edit"
                    ],
                    "audit_id": "",
                    "internal_evidence_complete": False,
                }
            )
        return payload

    def compare_versions(
        self,
        *,
        tenant_id: str,
        project_id: str,
        formula_id: str,
        left_version_id: str,
        right_version_id: str,
    ) -> dict[str, Any]:
        left = self.store.get_formula_version(
            tenant_id=tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            version_id=left_version_id,
        )
        right = self.store.get_formula_version(
            tenant_id=tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            version_id=right_version_id,
        )
        if left is None or right is None:
            raise KeyError("formula version not found")
        left_lines = self._line_map(left.payload)
        right_lines = self._line_map(right.payload)
        changes = []
        for ingredient_id in sorted(set(left_lines) | set(right_lines)):
            before = left_lines.get(ingredient_id, {})
            after = right_lines.get(ingredient_id, {})
            before_percent = float(before.get("concentrate_percent", 0.0))
            after_percent = float(after.get("concentrate_percent", 0.0))
            if math.isclose(before_percent, after_percent, abs_tol=1e-8):
                continue
            changes.append(
                {
                    "ingredient_id": ingredient_id,
                    "name": after.get("name") or before.get("name") or ingredient_id,
                    "before_percent": before_percent,
                    "after_percent": after_percent,
                    "delta_percent": round(after_percent - before_percent, 6),
                }
            )
        metric_changes = {}
        for metric in _METRICS:
            before = left.payload.get(metric)
            after = right.payload.get(metric)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                metric_changes[metric] = {
                    "before": float(before),
                    "after": float(after),
                    "delta": round(float(after) - float(before), 6),
                }
        left_profile = left.payload.get("achieved_profile", {})
        right_profile = right.payload.get("achieved_profile", {})
        dimensions = sorted(
            set(left_profile if isinstance(left_profile, dict) else {})
            | set(right_profile if isinstance(right_profile, dict) else {})
        )
        profile_delta = {
            dimension: round(
                float(right_profile.get(dimension, 0.0))
                - float(left_profile.get(dimension, 0.0)),
                6,
            )
            for dimension in dimensions
            if not math.isclose(
                float(right_profile.get(dimension, 0.0)),
                float(left_profile.get(dimension, 0.0)),
                abs_tol=1e-10,
            )
        }
        return {
            "formula_id": formula_id,
            "left": left.to_dict(include_payload=False),
            "right": right.to_dict(include_payload=False),
            "ingredient_changes": changes,
            "metric_changes": metric_changes,
            "profile_delta": profile_delta,
        }

    @staticmethod
    def _line_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lines = payload.get("recipe") or payload.get("closest_candidate") or []
        return {
            str(line["ingredient_id"]): line
            for line in lines
            if isinstance(line, dict) and isinstance(line.get("ingredient_id"), str)
        }

    def process_job(
        self,
        job,
        *,
        before_persist: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        payload = job.payload
        if job.kind == "recipe.generate":
            return self.generate_formula(
                tenant_id=job.tenant_id,
                actor_id=job.created_by,
                project_id=str(payload["project_id"]),
                brief=str(payload["brief"]),
                constraints=payload.get("constraints"),
                name=str(payload.get("name") or "Generated formula"),
                source_job_id=job.job_id,
                before_persist=before_persist,
            )
        if job.kind == "accord.generate":
            return self.create_accord(
                tenant_id=job.tenant_id,
                actor_id=job.created_by,
                project_id=str(payload["project_id"]),
                brief=str(payload["brief"]),
                constraints=payload.get("constraints"),
                name=str(payload.get("name") or "Generated accord"),
                source_job_id=job.job_id,
                before_persist=before_persist,
            )
        if job.kind == "formula.revise":
            return self.revise_formula(
                tenant_id=job.tenant_id,
                actor_id=job.created_by,
                project_id=str(payload["project_id"]),
                formula_id=str(payload["formula_id"]),
                base_version_id=str(payload["base_version_id"]),
                instruction=str(payload["instruction"]),
                constraints=payload.get("constraints"),
                source_job_id=job.job_id,
                before_persist=before_persist,
            )
        raise ValueError("unsupported job kind")
