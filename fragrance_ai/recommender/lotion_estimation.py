"""Explicit engineering estimates for a reference lotion, without external services.

No constants here are represented as measured CCT/Simulgel transport data.
The scenario envelope is a sensitivity check, not a statistical confidence band.
"""
from collections import Counter
from copy import deepcopy
from datetime import date
import math
from typing import Annotated, Literal

from pydantic import Field, field_validator

from ..platform.application_context import ApplicationContext, assess_application_context
from ..platform.lotion_inputs import LotionInput, LotionTargetInput, LotionSimulationRequest, LotionOptimizationRequest, TransitionSchedule
from ..platform.formulation_inputs import StorageConditions, SkinExposureConditions, LotionBaseDesignOptions
from ..platform.lotion_reference import lotion_reference
from ..platform.process_inputs import ManufacturingProcess
from .formulation_workflow import formulation_workflow
from .brief_parser import NaturalLanguageBriefParser
from .catalog import IngredientCatalog
from .lotion_optimizer import optimize_lotion
from .lotion_coverage import lotion_profile_coverage
from .models import RecipeConstraints
from .safety import CandidateSafetyScreen
from .science import ScientificPropertyStore, TemporalMixtureSimulator
from .formulation_science import permeability_estimates, storage_retention, skin_exposure_report
from .intent_controls import apply_intent_controls, target_controls
from .failure_recovery import failed_only_recovery


ESTIMATION_VERSION = "reference-lotion-engineering-v2"
R = 8.314462618
TEMPERATURE_K = 298.15
OIL_EFFECTIVE_MW = 500.  # engineering CCT mixture surrogate, not a measured lot
OIL_DENSITY = .95       # g/mL engineering surrogate


class LotionDoseTrials(LotionInput):
    """Explicit experimental design space, not a recommended safe use range."""
    concentrations_percent: list[Annotated[float, Field(ge=.01, le=3.)]] = Field(min_length=1, max_length=4)

    @field_validator('concentrations_percent', mode='before')
    @classmethod
    def reject_boolean_doses(cls, values):
        if isinstance(values, (list, tuple)) and any(isinstance(x, bool) for x in values):
            raise ValueError('dose candidates cannot be booleans')
        return values

    @field_validator('concentrations_percent')
    @classmethod
    def unique_doses(cls, values):
        if len(set(values)) != len(values):
            raise ValueError('dose candidates must be unique')
        return values


class LotionEstimateRequest(LotionTargetInput):
    brief: str = Field(min_length=3, max_length=2000)
    reference_id: Literal["LC-CCT-OW-01"] = "LC-CCT-OW-01"
    fragrance_concentration_percent: float = Field(default=.5, ge=.01, le=3.)
    application_mass_mg_cm2: float = Field(default=2., ge=.1, le=10.)
    substrate: Literal["skin_model", "inert_surface"] = "skin_model"
    target_similarity: float = Field(default=95., ge=90., le=100.)
    search_goal: Literal["maximize", "reach_target"] = "reach_target"
    registry_pool: Literal["core", "conditional_research"] = Field(default="conditional_research",
        description="Optional; omission uses the extended screened registry. Explicit core selects the former limited pool; risk and safety constraints still apply.")
    max_risk_tier: int = Field(default=1, strict=True, ge=1, le=2)
    max_formula_cost_per_kg: float = Field(default=180., gt=0, le=1e6)
    max_ingredient_price_per_kg: float = Field(default=300., gt=0, le=1e6)
    min_availability: float = Field(default=.75, ge=0, le=1)
    excluded_ingredient_ids: list[str] = Field(default_factory=list, max_length=50000)
    storage: StorageConditions | None = None
    skin_exposure: SkinExposureConditions = Field(default_factory=SkinExposureConditions)
    transition_schedule: TransitionSchedule | None = None
    base_design: LotionBaseDesignOptions | None = None
    dose_trials: LotionDoseTrials | None = None
    process: ManufacturingProcess | None = None


def build_estimated_lotion_inputs(request, catalog, parser=None, *, oil_base_percent=10., _request_snapshot=None):
    if not math.isfinite(oil_base_percent) or not 5 <= oil_base_percent <= 30:
        raise ValueError("oil fraction outside the selected engineering design range")
    constraints = RecipeConstraints(product_category="body_lotion", target_similarity=request.target_similarity,
        product_concentration_percent=request.fragrance_concentration_percent,
        max_risk_tier=request.max_risk_tier, enable_registry_trace_candidates=request.registry_pool == "conditional_research",
        max_formula_cost_per_kg=request.max_formula_cost_per_kg, max_ingredient_price_per_kg=request.max_ingredient_price_per_kg,
        min_availability=request.min_availability, explicit_bans=set(request.excluded_ingredient_ids))
    if set(request.excluded_ingredient_ids) - {i.ingredient_id for i in catalog.ingredients}:
        raise ValueError("unknown excluded ingredient ID")
    snapshot, today = _request_snapshot, date.today()
    controls = target_controls(request)
    reusable = (snapshot is not None and snapshot.get('catalog') is catalog
        and snapshot.get('parser') is parser and snapshot.get('text') == request.brief
        and snapshot.get('constraints') == constraints and snapshot.get('day') == today
        and snapshot.get('intent_controls', {}) == controls and 'properties' in snapshot)
    if reusable:
        brief, eligible, properties = snapshot['brief'], snapshot['eligible'], snapshot['properties']
        rejected = dict(snapshot['rejected'])
    else:
        defaults = deepcopy(constraints)
        brief = (parser or NaturalLanguageBriefParser(catalog)).parse(request.brief, constraints)
        brief = apply_intent_controls(brief, controls)
        raw_eligible, rejected = CandidateSafetyScreen().screen(catalog, brief, as_of=today)
        eligible = tuple(i for i in raw_eligible if abs(i.active_strength_percent-100) < 1e-8 and i.vector().sum() > 0)
        if not eligible:
            raise ValueError("no eligible candidates for estimated lotion design")
        # Request-local only: oil-dependent QSPR/storage/scenarios below stay fresh.
        with ScientificPropertyStore.load_builtin() as store:
            properties = ScientificPropertyStore.with_catalog_structures(eligible, store.get_many([i.ingredient_id for i in eligible]))
        if snapshot is not None:
            snapshot.clear()
            snapshot.update(catalog=catalog, parser=parser, text=request.brief, constraints=defaults,
                day=today, brief=brief, raw_eligible=tuple(raw_eligible), rejected=dict(rejected),
                eligible=eligible, properties=properties)
            snapshot['intent_controls'] = deepcopy(controls)
    context = ApplicationContext.model_validate({**lotion_reference(request.reference_id)["application_context"],
        "product_density_g_ml": 1., "application_mass_mg_cm2": request.application_mass_mg_cm2,
        "temperature_c": 25., "relative_humidity_percent": 50., "substrate": request.substrate,
        "fragrance_concentration_percent": request.fragrance_concentration_percent})
    if oil_base_percent != 10.:
        data = context.model_dump(mode="json")
        for component in data["base_components"]:
            if component["role"] == "oil":
                component["mass_percent"] = oil_base_percent
            elif component["role"] == "water":
                component["mass_percent"] = 94.-oil_base_percent
        data["formula_reference"] += f"; unqualified engineering oil/water variant: oil={oil_base_percent:g}% of fragrance-free base"
        context = ApplicationContext.model_validate(data)
    context_id = assess_application_context(context)["context_id"]
    counts, base_parameters, kinetic_assessments = Counter(), [], {}
    kinetics = {row.ingredient_id: row for row in request.storage.kinetics} if request.storage else {}
    if set(kinetics) - {i.ingredient_id for i in catalog.ingredients}:
        raise ValueError("unknown hydrolysis ingredient ID")
    # Same effective volumes as the fixed reference base, before drying.
    aqueous_volume = sum(c.mass_percent / (1.26 if c.role == "humectant" else 1.)
                         for c in context.base_components if c.role != "oil")
    lipid_volume = sum(c.mass_percent / OIL_DENSITY for c in context.base_components if c.role == "oil")
    for item in eligible:
        prop = properties.get(item.ingredient_id)
        if prop and ';identity-joined-evidence:' in prop.source_ref:
            counts['identity_joined_property_records'] += 1
        mw = prop.molecular_weight if prop else 180.
        logp = prop.xlogp if prop and prop.xlogp is not None else 2.
        counts["molecular_weight_available" if prop else "molecular_weight_prior"] += 1
        counts["xlogp_available" if prop and prop.xlogp is not None else "xlogp_prior"] += 1
        counts["provided_vapor_pressure" if prop and prop.vapor_pressure_pa_25c is not None
               else "boiling_point_vapor_estimate" if prop and prop.boiling_point_c is not None else "note_class_vapor_prior"] += 1
        pressure, sigma = TemporalMixtureSimulator._vapor_pressure_prior(item, prop)
        threshold_ppm, _ = TemporalMixtureSimulator._threshold_prior(item, prop)
        counts["provided_odor_threshold" if prop and prop.odor_threshold_ppm is not None else "odor_threshold_prior"] += 1
        # Octanol/water is a proxy, explicitly NOT measured CCT/water partition.
        if logp < -3 or logp > 6:
            counts["xlogp_outside_selected_proxy_range"] += 1
        bounded_logp = max(-3., min(6., logp))
        lipid_water = 10**bounded_logp
        # Raoult ideal-dilute oil surrogate + ideal gas law. g/mL -> g/m3.
        air_lipid = pressure * OIL_EFFECTIVE_MW / (R*TEMPERATURE_K*OIL_DENSITY*1e6)
        air_water = max(1e-12, air_lipid*lipid_water)
        # Potts-Guy aqueous QSPR in cm/h -> cm/min. Not lotion/skin qualification.
        permeability = permeability_estimates(mw, logp)
        if request.substrate != "inert_surface" and permeability["cm_min"] is None:
            counts["excluded_outside_skin_qspr_domain"] += 1
            continue
        skin = 0. if request.substrate == "inert_surface" else permeability["cm_min"]
        reaction = {}
        if request.storage:
            if item.ingredient_id not in kinetics:
                counts["missing_hydrolysis_kinetics"] += 1
                if request.storage.minimum_parent_retention_percent is not None:
                    continue
            else:
                retention = storage_retention(kinetics[item.ingredient_id], request.storage,
                                             aqueous_volume / (aqueous_volume + lipid_water * lipid_volume))
                kinetic_assessments[item.ingredient_id] = retention
                worst_retention = storage_retention(kinetics[item.ingredient_id], request.storage,
                    aqueous_volume / (aqueous_volume + lipid_water * 10**(-.5) * lipid_volume))
                if (request.storage.minimum_parent_retention_percent is not None and
                        worst_retention["parent_fraction"] * 100 < request.storage.minimum_parent_retention_percent):
                    counts["excluded_by_parent_retention_constraint"] += 1
                    continue
                reaction = {"initial_parent_fraction": retention["parent_fraction"],
                            "aqueous_hydrolysis_per_min": retention["aqueous_rate_per_min"],
                            "reaction_source_reference": retention["source_reference"]}
        base_parameters.append((item, {"ingredient_id": item.ingredient_id, "concentrate_percent": 100./len(eligible),
            "lipid_water_partition": lipid_water, "air_water_partition": air_water,
            "gas_transfer_cm_min": .1, "skin_permeability_cm_min": skin,
            "odor_threshold_mg_m3": max(1e-12, threshold_ppm*mw/(R*TEMPERATURE_K/101.325)),
            "source_reference": ESTIMATION_VERSION + "; " + (prop.source_ref if prop else "catalog-note-and-impact-priors"),
            "source_date": str(date.today()), "source_kind": "estimated", **reaction}, sigma))
    if not base_parameters:
        raise ValueError("no candidates with sufficient physical data satisfying the requested formulation constraints")
    for _, parameters, _ in base_parameters:
        parameters["concentrate_percent"] = 100. / len(base_parameters)
    phases = [{"name": c.name, "phase": "lipid" if c.role == "oil" else "aqueous",
               "density_g_ml": OIL_DENSITY if c.role == "oil" else 1.26 if c.role == "humectant" else 1.,
               "source_kind": "estimated", "source_reference": "effective phase allocation; not constituent analysis"}
              for c in context.base_components]
    scenarios = []
    for direction, name in ((0, "central"), (1, "opening_biased_fast_release"), (-1, "base_biased_slow_release")):
        materials = []
        for item, central, sigma in base_parameters:
            # Deliberately deterministic stress directions, not Monte Carlo quantiles.
            contrast = 1 if item.pyramid == "top" else -1 if item.pyramid == "base" else .25
            factor = math.exp(direction*contrast*max(.5, min(1.25, sigma)))
            partition_factor = 10**(-direction*.5)
            air_water = central["air_water_partition"]*factor*partition_factor
            if air_water < 1e-12:
                counts["air_partition_numerical_floor_scenario_values"] += 1
            reaction = {}
            if item.ingredient_id in kinetics:
                retention = storage_retention(kinetics[item.ingredient_id], request.storage,
                    aqueous_volume / (aqueous_volume + central["lipid_water_partition"] * partition_factor * lipid_volume))
                reaction["initial_parent_fraction"] = retention["parent_fraction"]
            materials.append({**central, **reaction, "air_water_partition": max(1e-12, air_water),
                "lipid_water_partition": central["lipid_water_partition"]*partition_factor,
                "gas_transfer_cm_min": .1*2**direction,
                "odor_threshold_mg_m3": central["odor_threshold_mg_m3"]/factor,
                "source_reference": central["source_reference"] + ";scenario=" + name})
        times = [0.,15.,60.,240.,480.]
        if request.transition_schedule:
            schedule = request.transition_schedule
            times = sorted(set(times + [schedule.opening_until_minutes, schedule.heart_until_minutes,
                                       min(1440., schedule.heart_until_minutes + 60.)]))
        scenarios.append(LotionSimulationRequest.model_validate({"application_context": context,
            "parameter_context_id": context_id, "phase_components": phases, "materials": materials,
            "water_loss_per_min": .015*2**direction, "retained_water_fraction": .1,
            "water_loss_source_reference": "engineering drying assumption at 25 C and RH50; not measured",
            "headspace_height_cm": 1., "air_exchange_per_min": 1.,
            "times_minutes": times, "integration_step_minutes": 1.,
            "transport_mode": "bidirectional_air", "coefficient_scope": "dilute_fixed_base",
            "coefficient_scope_reference": ESTIMATION_VERSION + "; weak-interaction approximation, not measured validity",
            "profile_weighting": "odor_activity"}))
    inputs = LotionOptimizationRequest(simulation=scenarios[0], brief=request.brief,
        evaluation_mode=request.evaluation_mode,
        **controls,
        target_similarity=request.target_similarity, search_goal=request.search_goal, minimum_air_concentration_mg_m3=1e-12,
        max_formula_cost_per_kg=request.max_formula_cost_per_kg, max_ingredient_price_per_kg=request.max_ingredient_price_per_kg,
        min_availability=request.min_availability, max_risk_tier=request.max_risk_tier,
        registry_pool=request.registry_pool, excluded_ingredient_ids=request.excluded_ingredient_ids,
        transition_schedule=request.transition_schedule,
        maximum_modeled_uptake_mg_cm2=request.skin_exposure.maximum_modeled_uptake_mg_cm2)
    from .physical_evidence import evidence_contract
    return inputs, scenarios[1:], {"estimation_version": ESTIMATION_VERSION, "candidate_count": len(base_parameters),
        "physical_evidence_connection": evidence_contract(),
        "base_variant": {"oil_percent_of_fragrance_free_base": oil_base_percent,
            "is_reference_composition": oil_base_percent == 10., "stability_verified": False,
            "scope": "fixed_reference" if oil_base_percent == 10. else "unqualified_engineering_variant"},
        "storage": request.storage.model_dump(mode="json") if request.storage else None,
        "storage_kinetic_assessments": kinetic_assessments,
        "transition_schedule": request.transition_schedule.model_dump() if request.transition_schedule else None,
        "coverage": dict(counts), "rejected_candidate_counts": rejected,
        "scenario_names": ["central", "opening_biased_fast_release", "base_biased_slow_release"],
        "application_context": context.model_dump(mode="json"), "external_api_calls": 0,
        "assumptions": {"oil_effective_molecular_weight_g_mol": OIL_EFFECTIVE_MW, "oil_density_g_ml": OIL_DENSITY,
            "other_blends": "effective nonvolatile aqueous volume; actual micellar allocation unresolved",
            "phase_partition": "octanol/water proxy for CCT/water, not a measurement",
            "gas_transfer_cm_min": [.05,.1,.2], "water_loss_per_min": [.0075,.015,.03],
            "headspace_height_cm": 1., "air_exchange_per_min": 1., "retained_water_fraction": .1},
        "evidence_kind": "physics_informed_engineering_estimates_not_measured_lotion_coefficients",
        "uncertainty_kind": "three_prespecified_sensitivity_scenarios_not_confidence_interval"}


def _estimate_single_base(request, catalog, parser=None, *, oil_base_percent=10., target_only=False, improve_score_from=None,
                          _request_snapshot=None, _shape_predictor=None, _incumbent_recipe=None):
    inputs, scenarios, provenance = build_estimated_lotion_inputs(request, catalog, parser,
        oil_base_percent=oil_base_percent, _request_snapshot=_request_snapshot)
    result = optimize_lotion(inputs, catalog, parser, transport_scenarios=scenarios, target_only=target_only,
        improve_score_from=improve_score_from, reuse_transport_basis=True, _request_snapshot=_request_snapshot,
        _transport_only=_shape_predictor is None,
        perception_guidance=_shape_predictor.provider if _shape_predictor is not None else None,
        _shape_predictor=_shape_predictor, _incumbent_recipe=_incumbent_recipe)
    return {**result, "schema_version": "estimated-lotion-design-1", "estimation": provenance,
        "skin_exposure": skin_exposure_report(result, request.skin_exposure, catalog),
        "candidate_recipe": result.get("recipe") or result.get("closest_candidate") or [],
        "candidate_use": "unvalidated_research_formula_not_manufacturing_instruction",
        "human_similarity_percent": None, "manufacturing_approved": False,
        "all_user_requirements_verified": False}


@failed_only_recovery('body_lotion')
def estimate_lotion_recipe(request, catalog, parser=None, *, perception_guidance=None, use_configured_perception=True,
                          _incumbent_recipe=None):
    from .lotion_perception import attach_lotion_perception, resolve_provider
    if not isinstance(use_configured_perception, bool):
        raise ValueError('use_configured_perception must be boolean')
    if request.process is not None:
        context = {**lotion_reference(request.reference_id)['application_context'],
                   'fragrance_concentration_percent': request.fragrance_concentration_percent}
        plan = formulation_workflow({'product_type': 'body_lotion', 'application_context': context,
                                     'process': request.process})
        if plan['status'] == 'process_conflict':
            raise ValueError('manufacturing process conflict: ' + ', '.join(
                row['code'] for row in plan['checks'] if row['severity'] == 'conflict'))
    provider = resolve_provider(perception_guidance) if (use_configured_perception or perception_guidance is not None) else None
    result = _estimate_lotion_recipe(request, catalog, parser, perception_guidance=provider,
                                    _incumbent_recipe=_incumbent_recipe)
    if request.dose_trials is not None:
        baseline, selected = result, request.fragrance_concentration_percent
        def record(value, dose):
            return {'fragrance_concentration_percent': dose, 'score': value.get('score'),
                'profile_target_met': bool(value.get('profile_target_met')), 'status': value['status'],
                'solver_calls': value.get('base_design', {}).get('total_solver_calls', value.get('solver_calls', 0)),
                'formula_id': value.get('formula_id')}
        trials, errors = [record(result, selected)], []
        for dose in sorted(request.dose_trials.concentrations_percent):
            if dose == request.fragrance_concentration_percent:
                continue
            # Rebuild context IDs, safety screening, transport and finished-product
            # amounts at each dose. Never rescale an old result or reuse its score.
            candidate_request = LotionEstimateRequest.model_validate({**request.model_dump(),
                'dose_trials': None, 'fragrance_concentration_percent': dose})
            try:
                candidate = _estimate_lotion_recipe(candidate_request, catalog, parser, perception_guidance=provider)
            except ValueError as error:
                errors.append({'fragrance_concentration_percent': dose, 'error': str(error)})
                continue
            trials.append(record(candidate, dose))
            candidate_score, score = candidate.get('score'), result.get('score')
            if candidate_score is not None and (score is None or candidate_score > score+1e-8
                    or (candidate_score >= score and dose < selected)):
                result, selected = candidate, dose
        result = {**result, 'dose_design': {'method': 'explicit_finished_product_dose_ladder/v1',
            'fixed_dose_percent': request.fragrance_concentration_percent,
            'fixed_dose_score': baseline.get('score'), 'selected_dose_percent': selected,
            'evaluated_variants': trials, 'variant_errors': errors,
            'total_solver_calls': sum(row['solver_calls'] for row in trials),
            'selection_rule': 'no_score_regression_then_lower_dose_on_exact_tie',
            'stability_verified': False, 'base_odor_measured': False,
            'human_intensity_calibrated': False, 'manufacturing_approved': False}}
    # Bind to the selected oil AND dose variant, never the initial reference.
    result['formulation_workflow'] = formulation_workflow({'product_type': 'body_lotion',
        'application_context': result.get('estimation', {}).get('application_context'),
        'process': request.process or ManufacturingProcess()})
    return result if 'perception_model' in result else attach_lotion_perception(result, catalog, provider)


def _estimate_lotion_recipe(request, catalog, parser=None, *, perception_guidance=None, _incumbent_recipe=None):
    snapshot = {}
    predictor = None
    if perception_guidance is not None:
        from .lotion_perception import make_lotion_shape_predictor
        predictor = make_lotion_shape_predictor(perception_guidance)
    baseline = _estimate_single_base(request, catalog, parser, _request_snapshot=snapshot, _shape_predictor=predictor,
                                     _incumbent_recipe=_incumbent_recipe)
    if request.base_design is None or baseline.get('status') == 'insufficient_observed_target_coverage':
        return baseline
    best, selected = baseline, 10.
    def record(result, oil):
        return {"oil_base_percent": oil, "score": result.get("score"),
                "profile_target_met": bool(result.get("profile_target_met")),
                "solver_calls": result.get("solver_calls", 0), "status": result["status"],
                "incumbent_reuse": result.get('incumbent_reuse')}
    variants = [record(baseline, 10.)]
    errors = []
    coverage = None
    if request.storage is None and baseline.get('perceptual_evaluation'):
        coverage = baseline['perceptual_evaluation'].get('profile_coverage')
    if (not baseline.get('profile_target_met') and baseline.get('preparation', {}).get('evaluation_targets')
            and 'perceptual_evaluation' not in baseline):
        coverage_catalog, pool_scope = catalog, 'all_catalog_including_ineligible'
        covered_ids = set(baseline['preparation'].get('candidate_ids', []))
        # Without storage filtering, changing only oil/water cannot change the
        # screened, structure/QSPR-covered pool. With storage it CAN change
        # retention eligibility, so retain the broader catalog upper bound.
        if request.storage is None and covered_ids:
            coverage_catalog = IngredientCatalog([i for i in catalog.ingredients if i.ingredient_id in covered_ids])
            pool_scope = 'screened_transport_covered_pool_invariant_under_oil_water_change'
        coverage = lotion_profile_coverage(coverage_catalog, baseline['preparation']['evaluation_targets'], request.target_similarity)
        coverage['profile_pool_scope'] = pool_scope
    excluded = bool(coverage and coverage['target_excluded'])
    if not baseline.get("profile_target_met") and not excluded:
        for oil in request.base_design.additional_oil_base_percents:
            try:
                candidate = _estimate_single_base(request, catalog, parser, oil_base_percent=oil, target_only=True,
                    improve_score_from=best.get('score'), _request_snapshot=snapshot, _shape_predictor=predictor,
                    _incumbent_recipe=best.get('recipe') or best.get('closest_candidate'))
            except ValueError as error:
                errors.append({"oil_base_percent": oil, "error": str(error)})
                continue
            variants.append(record(candidate, oil))
            if candidate.get("score") is not None and (best.get("score") is None or candidate["score"] > best["score"]):
                best, selected = candidate, oil
            if best.get("profile_target_met"):
                break
    adaptive_trials = []
    # Search between the caller's successfully evaluated endpoints, without
    # widening the oil/water envelope or changing the fixed reference.
    if not best.get('profile_target_met') and not excluded:
        attempted = {row['oil_base_percent'] for row in variants} | {row['oil_base_percent'] for row in errors}
        for _ in range(request.base_design.adaptive_oil_refinement_steps):
            valid_variants = sorted((row for row in variants if row.get('score') is not None), key=lambda row: row['oil_base_percent'])
            intervals = []
            for left, right in zip(valid_variants, valid_variants[1:]):
                width = right['oil_base_percent']-left['oil_base_percent']
                midpoint = (left['oil_base_percent']+right['oil_base_percent'])/2.
                if width >= .25 and midpoint not in attempted:
                    intervals.append((max(left['score'], right['score']), width, -midpoint, midpoint))
            if not intervals:
                break
            oil = max(intervals)[-1]
            attempted.add(oil)
            adaptive_trials.append(oil)
            try:
                candidate = _estimate_single_base(request, catalog, parser, oil_base_percent=oil, target_only=True,
                    improve_score_from=best.get('score'), _request_snapshot=snapshot, _shape_predictor=predictor,
                    _incumbent_recipe=best.get('recipe') or best.get('closest_candidate'))
            except ValueError as error:
                errors.append({'oil_base_percent': oil, 'error': str(error)})
                continue
            variants.append(record(candidate, oil))
            if candidate.get('score') is not None and (best.get('score') is None or candidate['score'] > best['score']):
                best, selected = candidate, oil
            if best.get('profile_target_met'):
                break
    context = best["estimation"]["application_context"]
    return {**best, "base_design": {
        "mode": "opt_in_oil_water_codesign", "selected_oil_base_percent": selected,
        "fixed_reference_score": baseline.get("score"),
        "fixed_reference_passed95": bool(baseline.get("profile_target_met")),
        "evaluated_variants": variants, "variant_errors": errors,
        "adaptive_oil_trials": adaptive_trials,
        "adaptive_oil_refinement_steps": request.base_design.adaptive_oil_refinement_steps,
        "profile_coverage": coverage,
        "additional_variants_skipped": 'fixed_profile_upper_below_target' if excluded else None,
        "total_solver_calls": sum(v["solver_calls"] for v in variants),
        "reference_formula_modified": selected != 10., "stability_verified": False,
        "manufacturing_approved": False, "full_base_design_optimized": False,
        "search_scope": "reference, explicit oil/water ratios and bounded midpoint refinement; selected evaluation contract and all transport scenarios retained",
        "finished_product_base": [{"name": c["name"], "role": c["role"],
            "finished_product_percent": c["mass_percent"]*(1-context["fragrance_concentration_percent"]/100.)}
            for c in context["base_components"]]}}
