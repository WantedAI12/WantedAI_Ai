"""Additive, stateless AI I/O for intent review and concentration scenarios.

Each evaluate call performs one inference. The backend can queue scenario calls
independently; this module does not start an unbounded synchronous batch.
"""
from datetime import date
import hashlib
import json
import math
import re
import threading
from typing import Any, Literal

from fastapi import HTTPException, Response, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..recommender.brief_parser import NaturalLanguageBriefParser, BriefParseError, UnsupportedOdorDescriptorError, apply_relative_revision_profile, PHASE_MARKER
from ..recommender.models import RecipeConstraints, SCENT_DIMENSIONS
from ..recommender.runtime_cache import InferenceCache, InferenceBusy
from ..recommender.catalog import normalize_name
from ..recommender.intent_controls import apply_intent_controls, representation_contract
from ..recommender.persistence import model_persistence
from .application_context import ApplicationContext, assess_application_context
from .process_inputs import ManufacturingProcess, WorkflowRequest
from ..recommender.formulation_workflow import formulation_workflow, knowledge_contract
from .lotion_inputs import LotionSimulationRequest, LotionOptimizationRequest
from .lotion_reference import REFERENCE_ID, lotion_reference
from .clarification import ProductPreferences, apply_question_answers, unsupported_preferences
from .rd_api import interpretation, register_rd_api
from .rd_evidence import EvidenceStore
from .operation_contracts import operation_contracts
from ..recommender.lotion import simulate_lotion
from ..recommender.lotion_optimizer import prepare_lotion_optimization, optimize_lotion
from ..recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
from ..recommender.compact_language import AssistantRequest, assistant_reply, validate_proposal
from ..recommender.perception_runtime import configured_perception, model_contract, assert_provider_current, assert_provider_product, environment_snapshot
from ..recommender.lotion_evaluation import evaluation_contract


def composition_distance(left, right):
    """Total variation of concentrate composition, not scent dissimilarity."""
    a, b = sum(left.values()), sum(right.values())
    if a <= 0 or b <= 0:
        return None
    return min(1., sum(abs(left.get(key, 0.) / a - right.get(key, 0.) / b) for key in set(left) | set(right)) / 2.)


def display_timeline(points, minutes=(0, 30, 240, 480)):
    """Display-only interpolation; never invent extra simulation observations."""
    ordered = sorted(points, key=lambda row: row["minutes"])
    output = []
    for minute in minutes:
        if not ordered or minute < ordered[0]["minutes"] or minute > ordered[-1]["minutes"]:
            output.append({"minutes": minute, "status": "outside_simulated_range", "scent_profile": None,
                           "relative_to_opening_intensity_percent": None})
            continue
        exact = next((row for row in ordered if row["minutes"] == minute), None)
        if exact is not None:
            profile = dict(exact.get("scent_profile") or {})
            intensity = exact.get("relative_to_opening_intensity_percent")
            bracket, status = [minute, minute], "simulated_timepoint"
        else:
            left = max((row for row in ordered if row["minutes"] < minute), key=lambda row: row["minutes"])
            right = min((row for row in ordered if row["minutes"] > minute), key=lambda row: row["minutes"])
            fraction = (minute - left["minutes"]) / (right["minutes"] - left["minutes"])
            profile = {axis: (1 - fraction) * left.get("scent_profile", {}).get(axis, 0.)
                       + fraction * right.get("scent_profile", {}).get(axis, 0.) for axis in SCENT_DIMENSIONS}
            a, b = left.get("relative_to_opening_intensity_percent"), right.get("relative_to_opening_intensity_percent")
            intensity = None if a is None or b is None else (1 - fraction) * a + fraction * b
            bracket, status = [left["minutes"], right["minutes"]], "display_interpolation_not_new_simulation"
        output.append({"minutes": minute, "status": status, "source_minutes": bracket,
                       "scent_profile": profile, "relative_to_opening_intensity_percent": intensity,
                       "dominant_dimensions": sorted((axis for axis in profile if profile[axis] > 0), key=lambda axis: (-profile[axis], axis))[:3]})
    return output


def evidence_summary(payload):
    safety = payload.get("safety") or {}
    lines = payload.get("recipe") or payload.get("closest_candidate") or []
    return {"model_version": payload.get("scientific_model_version"),
        "catalog_snapshot": payload.get("catalog_stats", {}),
        "material_count": len(lines), "reference_match_count": len(payload.get("historical_reference_matches") or []),
        "provided_vapor_pressure_coverage_percent": payload.get("vapor_pressure_coverage_percent"),
        "odor_threshold_coverage_percent": payload.get("odor_threshold_coverage_percent"),
        "calculated_structure_coverage_percent": payload.get("calculated_structure_coverage_percent"),
        "model_applicability_percent": payload.get("model_applicability_percent"),
        "model_domain_gate_passed": payload.get("scientific_model_domain_passed", False),
        "uncertainty_kind": payload.get("scientific_uncertainty_kind", "not_available"),
        "temporal_model_quantiles": {"p05": payload.get("temporal_similarity_p05"), "p95": payload.get("temporal_similarity_p95")},
        "human_accuracy_proven_by_this_summary": False,
        "supply": {"availability_estimate_minimum": min((row.get("availability", 0.) for row in lines), default=None),
                   "live_inventory_verified": False, "supplier_evidence_coverage_percent": safety.get("evidence_coverage_percent"),
                   "missing_documents": safety.get("missing_documents", [])},
        "scientific_flags": payload.get("scientific_flags", []), "limitations": payload.get("limitations", [])}


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExpressionInterpretRequest(StrictInput):
    text: str = Field(min_length=1,max_length=4000)


class ExpressionPredictRequest(StrictInput):
    ingredient_ids: list[str] = Field(min_length=1,max_length=128)
    top_k: int = Field(default=12,ge=1,le=100)
    brief: str | None = Field(default=None,max_length=4000)


class ConcentrationRange(StrictInput):
    minimum: float = Field(gt=0, le=30)
    maximum: float = Field(gt=0, le=30)
    basis: Literal["w/w_percent"] = "w/w_percent"

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("concentration minimum must not exceed maximum")
        return self


class PriceBudget(StrictInput):
    amount: float = Field(gt=0, le=1e9)
    currency: Literal["KRW"] = "KRW"
    per_volume_ml: float = Field(gt=0, le=1e6)
    product_density_g_ml: float | None = Field(default=None, gt=0, le=10)
    scope: Literal["fragrance_only", "finished_product_materials"] | None = None
    base_material_cost_krw: float | None = Field(default=None, ge=0, le=1e9)
    krw_per_catalog_price_unit: float | None = Field(default=None, gt=0, le=1e7)
    price_basis_reference: str | None = Field(default=None, min_length=1, max_length=500)
    price_as_of: date | None = None


class IntentEdits(StrictInput):
    target_profile: dict[str, float] | None = None
    excluded_ingredient_ids: list[str] = Field(default_factory=list, max_length=500)
    intensity_level: int | None = Field(default=None, strict=True, ge=1, le=5)
    phase_target_profiles: dict[str, dict[str, float]] | None = None

    @model_validator(mode="after")
    def profile_valid(self):
        if self.target_profile is not None:
            values = self.target_profile
            if not values or set(values) - set(SCENT_DIMENSIONS) or any(v < 0 for v in values.values()) or not math.isfinite(sum(values.values())) or sum(values.values()) <= 0:
                raise ValueError("target_profile requires known scent axes and positive total weight")
        return self


class ModelPersistenceRequest(StrictInput):
    basis: Literal["relative_model_intensity"] = "relative_model_intensity"
    relative_threshold_percent: float = Field(strict=True, gt=0, le=100)
    required_duration_minutes: float | None = Field(default=None, strict=True, gt=0, le=43200)


def register_ai_extensions(app, formula_type, catalog, generate_formula, rate_limit, *, language_backend=None, perception_guidance=None, lotion_perception_guidance=None, stock_mixture_predictor=None, catalog_contract=None, runtime_guard=None, unified_product_predictor=None, evidence_store=None):
    """Use the legacy generator, authentication boundary, cache and safety gates."""
    evidence_store = evidence_store if evidence_store is not None else EvidenceStore.configured()
    class ExtendedRequest(StrictInput):
        formula: formula_type
        product_type: str | None = Field(default=None, min_length=1, max_length=80)
        concentration_range: ConcentrationRange | None = None
        budget: PriceBudget | None = None
        minimum_longevity_hours: float | None = Field(default=None, gt=0, le=720)
        model_persistence: ModelPersistenceRequest | None = None
        application_context: ApplicationContext | None = None
        process: ManufacturingProcess | None = None
        product_preferences: ProductPreferences = Field(default_factory=ProductPreferences)
        edits: IntentEdits = Field(default_factory=IntentEdits)
        scenario_index: int = Field(default=0, ge=0, le=2)

    class ClarificationRequest(StrictInput):
        request: ExtendedRequest
        request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
        answers: dict[str, Any]

    class AlternativeRequest(ExtendedRequest):
        previous_candidate_ids: list[str] = Field(min_length=1, max_length=12)
        variation_index: int = Field(default=0, ge=0, le=49999)
        minimum_composition_distance: float = Field(default=.02, gt=0, le=1)

    class RevisionRequest(ExtendedRequest):
        instruction: str = Field(min_length=1, max_length=1000)

    class ComparisonRequest(StrictInput):
        candidate_ids: list[str] = Field(min_length=2, max_length=10)

    class FixedLine(StrictInput):
        ingredient_id: str = Field(min_length=1, max_length=300)
        concentrate_percent: float = Field(ge=.0001, le=100)

    class FixedRequest(ExtendedRequest):
        lines: list[FixedLine] = Field(min_length=1, max_length=50000)

    class StockLine(StrictInput):
        ingredient_id: str = Field(min_length=1, max_length=300)
        stock_dilution: float = Field(strict=True, gt=0, le=1)
        solvent: Literal['nt','pg','dep','paraffin oil','mineral oil','90% ethanol','99% ethanol']
        relative_volume: float | None = Field(default=None, strict=True, ge=0, le=1e9)
        supplied_mass_g: float | None = Field(default=None, strict=True, ge=0, le=1e9)
        stock_density_g_ml: float | None = Field(default=None, strict=True, ge=.01, le=30)

    class StockRequest(StrictInput):
        application_domain: Literal['stock_aliquot_assay'] = 'stock_aliquot_assay'
        basis: Literal['relative_volume','supplied_mass']
        components: list[StockLine] = Field(min_length=1, max_length=50000)

        @model_validator(mode='after')
        def complete_basis(self):
            for row in self.components:
                if self.basis == 'relative_volume':
                    if row.relative_volume is None or row.supplied_mass_g is not None or row.stock_density_g_ml is not None:
                        raise ValueError('volume basis requires only relative_volume')
                elif row.supplied_mass_g is None or row.stock_density_g_ml is None or row.relative_volume is not None:
                    raise ValueError('mass basis requires stock mass and diluted-stock density')
            return self

    parser = NaturalLanguageBriefParser(catalog)
    perception_provider = perception_guidance if perception_guidance is not None else configured_perception()
    from ..recommender.lotion_perception import resolve_provider
    lotion_provider = resolve_provider(lotion_perception_guidance)
    assert_provider_product(perception_provider, 'perfume')
    assert_provider_product(lotion_provider, 'body_lotion')
    if stock_mixture_predictor is not None:
        from ..recommender.frozen_stock_assay import FrozenStockAssay
        if not isinstance(stock_mixture_predictor, FrozenStockAssay) and stock_mixture_predictor.provider is not perception_provider:
            raise ValueError('stock mixture requires the same bound component provider')
        stock_mixture_predictor.assert_current()
    if lotion_provider is not None and lotion_provider is perception_provider:
        raise ValueError('perfume and lotion require independent provider instances')
    lotion_environment = environment_snapshot('body_lotion')
    from ..recommender.lotion_surrogate import configured_lotion_surrogate, predict_release
    release_model = configured_lotion_surrogate()
    from ..recommender.lotion_reference_objective import load_configured_reference_bank
    target_reference_bank = load_configured_reference_bank()
    from ..recommender.unified_product import configured_unified_product
    unified_predictor = (unified_product_predictor if unified_product_predictor is not None else
                         configured_unified_product(catalog, component_provider=perception_provider))
    from ..recommender.fine_odor_model import configured_fine_odor
    from ..recommender.odor_expression import registry, expression_contract, parse_expression, summarize_expression
    fine_model = configured_fine_odor()

    def assert_lotion_current():
        if runtime_guard is not None:
            runtime_guard()
        if lotion_environment != environment_snapshot('body_lotion'):
            raise ValueError('lotion model policy changed; reload lotion API')
        assert_provider_product(lotion_provider, 'body_lotion')
        if release_model is not None:
            release_model.assert_current()
        if target_reference_bank is not None:
            target_reference_bank.assert_current()
        if unified_predictor is not None:
            unified_predictor.assert_current()

    lotion_contract = {**evaluation_contract(), 'component_model': model_contract(lotion_provider)}
    lotion_contract['trained_release_model'] = release_model.contract() if release_model is not None else None
    lotion_contract['observed_target_reference'] = target_reference_bank.contract() if target_reference_bank is not None else None
    lotion_contract['supported_evaluation_modes'] = ['auto', 'legacy_profile', 'observed_reference']
    if catalog_contract is not None:
        lotion_contract['catalog_snapshot'] = dict(catalog_contract)
    language_slot = threading.BoundedSemaphore(1)
    language_cache = InferenceCache(ttl_seconds=120., max_entries=64, max_bytes=1024 * 1024,
                                    max_pending=1, max_followers=8, wait_seconds=150.)
    app.state.language_cache = language_cache

    @app.post('/v1/formulation-workflows/plan')
    def plan_formulation(request: WorkflowRequest):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        return formulation_workflow(request)

    from .emulsion_inputs import EmulsionRequest

    @app.post('/v1/formulation-workflows/emulsion-prediction')
    def predict_emulsion(request: EmulsionRequest):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        from ..recommender.formulation_core import configured_formulation_core
        core = configured_formulation_core()
        if core is None:
            raise HTTPException(status_code=503, detail='shared formulation checkpoint not configured')
        return core.emulsion(request.raw_rows())

    @app.post('/v1/ai/assistant')
    def assistant(request: AssistantRequest, response: Response):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        if not language_slot.acquire(blocking=False):
            raise HTTPException(status_code=503, detail='assistant busy; retry', headers={'Retry-After': '2'})
        try:
            calls, cache_status = 0, 'not_used'

            def cached_language(message):
                nonlocal cache_status
                contract = getattr(language_backend, 'contract', None)
                identity = contract() if callable(contract) else None
                key = hashlib.sha256(json.dumps({'message': message, 'backend': identity},
                    sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

                def compute():
                    nonlocal calls, cache_status
                    calls += 1
                    cache_status = 'error'
                    # Invalid output and transient model failures never enter
                    # the cache. The existing parser fallback still handles them.
                    proposal = validate_proposal(language_backend(message)).model_dump()
                    if callable(contract) and contract() != identity:
                        raise ValueError('language model changed during inference')
                    return proposal

                proposal, cache_status = language_cache.run(key, compute)
                return proposal

            result = assistant_reply(request, parser, cached_language if language_backend is not None else None)
            if runtime_guard is not None:
                runtime_guard()
            response.headers['X-Perfumery-Language-Cache'] = cache_status
            response.headers['X-Perfumery-LLM-Calls'] = str(calls)
            return result
        finally:
            language_slot.release()

    materials_by_id = {item.ingredient_id:item for item in catalog.ingredients}
    known_ids = set(materials_by_id)
    name_to_ids = {}
    for item in catalog.ingredients:
        name_to_ids.setdefault(normalize_name(item.name), set()).add(item.ingredient_id)
    # Same authentication boundary as generation. This is a bounded, temporary
    # comparison cache, not persistence or a separate tenant database.
    candidate_cache = InferenceCache(ttl_seconds=1800., max_entries=128, max_bytes=8 * 1024 * 1024)
    lotion_cache = InferenceCache(ttl_seconds=1800., max_entries=128, max_bytes=8 * 1024 * 1024)
    stock_cache = InferenceCache(ttl_seconds=1800., max_entries=16, max_bytes=2 * 1024 * 1024)
    unified_cache = InferenceCache(ttl_seconds=1800., max_entries=16, max_bytes=8 * 1024 * 1024)

    from .unified_product_inputs import UnifiedProductContext, UnifiedProductRequest, context_id

    @app.get('/v1/odor-expressions')
    def list_expressions(q: str = Query(default='',max_length=100), offset: int = Query(default=0,ge=0),
                         limit: int = Query(default=50,ge=1,le=100)):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        query = q.casefold().strip()
        rows = [r for r in registry()['rows'].values() if not query or
                any(query in alias for alias in r['aliases']) or query in r['id']]
        modeled = set(fine_model.endpoints) if fine_model is not None else set()
        return {'contract':expression_contract(),'total':len(rows),'offset':offset,
            'items':[{**row,'prediction_status':'model_connected' if row['id'] in modeled else 'vocabulary_only'}
                     for row in rows[offset:offset+limit]]}

    @app.post('/v1/odor-expressions/interpret')
    def interpret_expression(request: ExpressionInterpretRequest):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        return {**parse_expression(request.text),'contract':expression_contract(),
                'recipe_generated':False,'unknown_text_guaranteed_understood':False}

    @app.post('/v1/odor-expressions/predict')
    def predict_expression(request: ExpressionPredictRequest):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        if fine_model is None:
            raise HTTPException(status_code=503,detail='fine odor expression model is not configured')
        if len(set(request.ingredient_ids)) != len(request.ingredient_ids) or set(request.ingredient_ids)-known_ids:
            raise HTTPException(status_code=422,detail='ingredient IDs must be unique and present in this catalog')
        items = [materials_by_id[k] for k in request.ingredient_ids]
        values,evidence = fine_model.materials(items)
        intent = parse_expression(request.brief) if request.brief else None
        requested = set(intent['wanted'])|set(intent['avoided']) if intent else ()
        return {'model':fine_model.contract(),'intent':intent,'materials':[
            {'ingredient_id':item.ingredient_id,'evidence':proof,
             **summarize_expression(None if proof['status']=='missing_or_multicomponent_structure' else row,
                 fine_model.endpoints,top_k=request.top_k,requested=requested)}
            for item,row,proof in zip(items,values,evidence)],'manufacturing_approved':False}

    @app.post('/v1/applications/unified/context')
    def prepare_unified_context(context: UnifiedProductContext):
        rate_limit()
        if runtime_guard is not None:
            runtime_guard()
        return {'parameter_context_id': context_id(context), 'context': context.model_dump(mode='json'),
                'coefficient_sources_verified': False, 'kinetics_required_for_every_stage': True}

    @app.post('/v1/applications/unified/predict')
    def predict_unified(request: UnifiedProductRequest, response: Response):
        rate_limit()
        assert_lotion_current()
        if unified_predictor is None:
            raise HTTPException(status_code=503, detail='unified product model is not configured')
        identity = hashlib.sha256(json.dumps({'request': request.model_dump(mode='json'),
            'model': unified_predictor.contract(), 'catalog': catalog_contract},
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        try:
            value, status = unified_cache.run(identity, lambda: unified_predictor.predict(request))
            assert_lotion_current()
            response.headers['X-Perfumery-Unified-Cache'] = status
            return value
        except (InferenceBusy, TimeoutError) as error:
            raise HTTPException(status_code=503, detail='unified inference busy; retry', headers={'Retry-After': '5'}) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post('/v1/formulations/stock-mixture/predict')
    def predict_stock_mixture(request: StockRequest, response: Response):
        rate_limit()
        if stock_mixture_predictor is None:
            raise HTTPException(status_code=503, detail='stock mixture model is not configured')
        try:
            if runtime_guard is not None:
                runtime_guard()
            stock_mixture_predictor.assert_current()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        unknown = sorted({r.ingredient_id for r in request.components}-known_ids)
        if unknown:
            raise HTTPException(status_code=422, detail={'unknown_ingredient_ids':unknown})
        identity = hashlib.sha256(json.dumps({'request':request.model_dump(mode='json'),
            'model':stock_mixture_predictor.sha256,'component':stock_mixture_predictor.component_sha256},
            sort_keys=True,allow_nan=False).encode()).hexdigest()
        def compute():
            from ..recommender.stock_mixture import StockAliquot,StockMass
            if request.basis == 'supplied_mass':
                return stock_mixture_predictor.predict_masses([StockMass(materials_by_id[r.ingredient_id],
                    r.stock_dilution,r.supplied_mass_g,r.stock_density_g_ml,r.solvent) for r in request.components])
            return stock_mixture_predictor.predict([StockAliquot(materials_by_id[r.ingredient_id],
                r.stock_dilution,r.relative_volume,r.solvent) for r in request.components])
        try:
            def checked_compute():
                value = compute()
                stock_mixture_predictor.assert_current()
                if runtime_guard is not None:
                    runtime_guard()
                return value
            value,status = stock_cache.run(identity,checked_compute)
            stock_mixture_predictor.assert_current()
            if runtime_guard is not None:
                runtime_guard()
            response.headers['X-Perfumery-Stock-Cache'] = status
            return value
        except (InferenceBusy,TimeoutError) as error:
            raise HTTPException(status_code=503,detail='stock mixture inference busy; retry',headers={'Retry-After':'5'}) from error
        except ValueError as error:
            raise HTTPException(status_code=422,detail=str(error)) from error

    def cache_run(identity, compute):
        try:
            return candidate_cache.run(identity, compute)[0]
        except (InferenceBusy, TimeoutError) as error:
            raise HTTPException(status_code=503, detail="comparison cache busy; retry", headers={"Retry-After": "5"}) from error

    def remember(candidate, context, scenario, diversity_exclusions=()):
        lines = candidate["result"].get("recipe") or candidate["result"].get("closest_candidate") or []
        record = {key: value for key, value in candidate.items() if key not in {"candidate_id", "result"}}
        record.update(context_id=context, scenario_index=scenario,
                      composition={row["ingredient_id"]: row["concentrate_percent"] for row in lines},
                      diversity_exclusions=list(diversity_exclusions))
        identity = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
        candidate["candidate_id"] = identity
        record["candidate_id"] = identity
        cache_run(identity, lambda: record)

    def recall(identity):
        def missing():
            raise ValueError("candidate expired or belongs to another runtime; regenerate it with its original request")
        try:
            return cache_run(identity, missing)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    supported_products = {
        "eau_de_parfum", "eau_de_toilette", "eau_de_cologne", "shampoo", "body_wash", "candle", "room_spray", "diffuser"}

    @app.get("/v1/ai/capabilities")
    def capabilities():
        assert_provider_current(perception_provider)
        assert_lotion_current()
        language_contract = getattr(language_backend, 'contract', None)
        from ..recommender.formulation_core import configured_formulation_core
        shared = configured_formulation_core()
        return {"schema_version": "ai-capabilities-1", "supported_product_codes": sorted(supported_products),
            "integration_contract": operation_contracts(app, supported_products,
                unified_available=unified_predictor is not None,
                evidence_configured=evidence_store.bundle is not None),
            "language_model": language_contract() if language_contract is not None else {
                'backend': 'injected' if language_backend is not None else 'deterministic_parser'},
            "perception_model": model_contract(perception_provider),
            "stock_mixture_model": (stock_mixture_predictor.contract()
                                    if stock_mixture_predictor is not None else None),
            "unified_product_model": unified_predictor.contract() if unified_predictor is not None else None,
            "odor_expression":{**expression_contract(),'prediction_model':fine_model.contract() if fine_model else None},
            "formulation_knowledge": knowledge_contract(),
            "shared_formulation_model": shared.contract() if shared is not None else None,
            "product_models": {
                "perfume": {"product": "perfume", "component_model": model_contract(perception_provider),
                            "prediction_model": "perfume_temporal_mixture", "cross_product_scores_comparable": False,
                            "unified_conditional_prediction_available": unified_predictor is not None},
                "body_lotion": lotion_contract,
                "body_wash": {"product": "body_wash", "unified_conditional_prediction_available": unified_predictor is not None,
                              "supplied_rinse_and_stage_coefficients_required": True, "matrix_calibrated": False}},
            "features": {"intent_preparation": True, "clarification_questions": True,
                "sourced_formulation_workflows": True, "process_condition_conflict_checks": True,
                "finished_lotion_batch_mass_balance": True,
                "bounded_assistant": True, "quantized_language_backend": language_backend is not None,
                "shared_product_transport_and_odor_backbone": unified_predictor is not None,
                "body_wash_supplied_dilution_rinse_trajectory": unified_predictor is not None,
                "typed_clarification_answers": True, "product_preference_support_reporting": True,
                "skin_type_compatibility": False, "ph_compatibility": False,
                "direct_note_transition_speed_control": False,
                "concentration_scenarios": True, "distinct_composition_alternatives": True,
                "server_generated_comparison": True, "global_relative_revision": True,
                "fixed_formula_scientific_reassessment": True, "evidence_summary": True,
                "portable_rd_snapshot_comparison": True, "rd_typed_clarification_answers": True,
                "saved_candidate_intent_revision": True,
                "structured_phase_profiles": True, "five_level_intensity_input": True,
                "model_relative_persistence": True, "application_context_validation": True,
                "body_lotion_research_transport_simulation": True,
                "body_lotion_trained_release_surrogate": release_model is not None,
                "body_lotion_observed_reference_design": target_reference_bank is not None,
                "body_lotion_reference_composition": True, "body_lotion_split_base_components": True,
                "body_lotion_engineering_estimate_design": True,
                "body_lotion_learned_temporal_shape": lotion_provider is not None,
                "body_lotion_learned_optimization": lotion_provider is not None,
                "explicit_stock_mixture_prediction": stock_mixture_predictor is not None,
                "body_lotion_oil_water_codesign": True,
                "body_lotion_timed_phase_targets": True, "body_lotion_modeled_uptake_constraint": True,
                "body_lotion_supplied_hydrolysis_kinetics": True,
                "body_lotion_bidirectional_air_transport": True, "body_lotion_fixed_base_inverse_design": True,
                "phase_specific_relative_revision": False, "body_lotion_matrix_model": False,
                "minimum_detectable_longevity_guarantee": False, "live_supplier_inventory": False,
                "complete_regulatory_certification": False},
            "limits": {"new_inferences_per_alternative_call": 1, "comparison_candidates": 10,
                       "comparison_cache_seconds": 1800, "comparison_cache_entries": 128},
            "scope": "software capabilities, not manufacturing approval or measured human accuracy"}

    @app.post("/v1/applications/validate")
    def validate_application(context: ApplicationContext):
        rate_limit()
        return assess_application_context(context)

    @app.get("/v1/applications/body-lotion/references")
    def list_lotion_references():
        rate_limit()
        return {"schema_version": "lotion-references-1", "references": [lotion_reference()]}

    @app.get("/v1/applications/body-lotion/references/{reference_id}")
    def get_lotion_reference(reference_id: str):
        rate_limit()
        if reference_id != REFERENCE_ID:
            raise HTTPException(status_code=404, detail="unknown lotion reference")
        return lotion_reference(reference_id)

    def lotion_run(request, operation, response, compute):
        assert_lotion_current()
        # Reuse the existing bounded CPU/cache mechanism. Include the date so
        # screening expiry cannot be bypassed by a result cached yesterday.
        identity = "lotion:" + operation + ":" + date.today().isoformat() + ":" + hashlib.sha256(
            json.dumps({'request': request.model_dump(mode='json'), 'product_model': lotion_contract},
                       sort_keys=True, allow_nan=False).encode()).hexdigest()
        try:
            payload, status = lotion_cache.run(identity, compute)
            if lotion_provider is not None and 'perception_model' not in payload:
                payload = {**payload, 'perception_model': {**model_contract(lotion_provider),
                    'operation': 'lotion_' + operation, 'applied': False, 'full_model_application': False,
                    'status': 'not_evaluated_preparation_only', 'search_weight': 0.}}
            payload = {**payload, 'product_model': {**payload.get('product_model', evaluation_contract()),
                                                   'component_model': model_contract(lotion_provider)}}
            if catalog_contract is not None:
                payload['catalog_snapshot'] = dict(catalog_contract)
            assert_lotion_current()
            response.headers["X-Perfumery-Lotion-Cache"] = status
            return payload
        except (InferenceBusy, TimeoutError) as error:
            raise HTTPException(status_code=503, detail="lotion inference busy; retry", headers={"Retry-After": "5"}) from error

    @app.post("/v1/applications/body-lotion/simulate")
    def simulate_application(request: LotionSimulationRequest, response: Response):
        rate_limit()
        try:
            return lotion_run(request, "simulate", response, lambda: simulate_lotion(request, catalog, perception_guidance=lotion_provider))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/applications/body-lotion/predict-release")
    def fast_release_prediction(request: LotionSimulationRequest, response: Response):
        rate_limit()
        if release_model is None:
            raise HTTPException(status_code=503, detail='trained lotion release model is not configured')
        def compute():
            value = predict_release(request, catalog, model=release_model)
            if lotion_provider is not None and 'temporal_profile' in value:
                from ..recommender.lotion_perception import attach_lotion_perception
                value = attach_lotion_perception(value, catalog, lotion_provider)
            return value
        try:
            return lotion_run(request, 'trained-release', response, compute)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/applications/body-lotion/design")
    def design_lotion(request: LotionEstimateRequest, response: Response):
        rate_limit()
        try:
            return lotion_run(request, "estimated-design-v2", response,
                              lambda: estimate_lotion_recipe(request, catalog, parser, perception_guidance=lotion_provider))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/applications/body-lotion/prepare")
    def prepare_lotion(request: LotionOptimizationRequest, response: Response):
        rate_limit()
        try:
            return lotion_run(request, "prepare", response, lambda: prepare_lotion_optimization(request, catalog, parser)[0])
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/applications/body-lotion/optimize")
    def generate_lotion(request: LotionOptimizationRequest, response: Response):
        rate_limit()
        try:
            return lotion_run(request, "optimize", response, lambda: optimize_lotion(request, catalog, parser, perception_guidance=lotion_provider))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def prepare(request):
        value = request.formula
        digest = hashlib.sha256(json.dumps(request.model_dump(mode="json", exclude={"scenario_index"}),
            sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if not value.require_full_profile_match or value.experimental_disable_safety:
            raise ValueError("extended workflows require the strict quality and safety gates")
        if set(request.edits.excluded_ingredient_ids) - known_ids:
            raise ValueError("unknown excluded ingredient ID")
        constraints = RecipeConstraints(target_similarity=max(95., value.target_similarity), max_ingredients=value.max_ingredients,
            target_region=value.target_region, product_category=value.product_category,
            product_concentration_percent=value.product_concentration_percent,
            explicit_bans=set(request.edits.excluded_ingredient_ids))
        try:
            brief = parser.parse(value.brief.strip(), constraints)
            brief = apply_intent_controls(brief, controls_for(request))
        except BriefParseError as error:
            unsupported = isinstance(error, UnsupportedOdorDescriptorError)
            pending = unsupported_preferences(request.product_preferences)
            if unsupported:
                pending.append({"code": "unsupported_odor_descriptor", "field": "formula.brief", "message": str(error)})
            return {"schema_version": "ai-brief-1", "request_id": digest,
                "interpretation": interpretation(unsupported=unsupported),
                "status": "unsupported_requirements" if pending else "needs_clarification",
                "original_text": value.brief, "intent": None, "effective_target": max(95., value.target_similarity),
                "questions": [] if unsupported else [{"id": "formula.brief", "question": str(error), "input_type": "text", "required": True}],
                "unsupported_requirements": pending,
                "product_preferences": request.product_preferences.model_dump(mode="json", exclude_none=True),
                "scenarios": [], "budget_basis": "not_evaluated", "scope": "intent requires user review"}
        profile = request.edits.target_profile
        if profile is not None:
            brief = apply_intent_controls(brief, {'target_profile': profile})
        questions, unsupported = [], unsupported_preferences(request.product_preferences)

        def question(key, label, input_type):
            row = {"id": key, "question": label, "input_type": input_type, "required": True}
            if input_type == "budget_scope":
                row["options"] = [{"value": "fragrance_only", "label": "향료비만"},
                                  {"value": "finished_product_materials", "label": "베이스 포함 전체 원료비"}]
            if input_type == "percent_range":
                row["unit"] = "w/w_percent"
            questions.append(row)

        context = request.application_context
        if context and request.product_type and context.product_type != request.product_type:
            raise ValueError("application_context and product_type disagree")
        product = request.product_type or (context.product_type if context else value.product_category)
        if not request.product_type and re.search(r"바디\s*로션|body\s*lotion", value.brief, re.I):
            product = "body_lotion"
        if product not in supported_products:
            unsupported.append({"code": "unsupported_product_model", "field": "product_type", "value": product,
                                "message": "이 제품의 용도·매질 모델이 연결되지 않았습니다. 향수로 대체 계산하지 않습니다."})
        workflow = formulation_workflow({'product_type': 'body_lotion' if product == 'body_lotion' else 'perfume',
            'application_context': context,
            'process': request.process or ManufacturingProcess()}) if product in {
                'eau_de_parfum', 'eau_de_toilette', 'eau_de_cologne', 'body_lotion'} else None
        if request.process is not None and workflow is None:
            unsupported.append({'code': 'process_product_not_covered', 'field': 'process',
                'message': '해당 제품의 제조 공정 지식은 연결되지 않았습니다.'})
        if request.process is not None and workflow and workflow['status'] == 'process_conflict':
            unsupported.append({'code': 'manufacturing_process_conflict', 'field': 'process',
                'message': '입력한 공정 조건이 연결된 원료·제형 조건과 충돌합니다.',
                'checks': [row for row in workflow['checks'] if row['severity'] == 'conflict']})
        duration = request.minimum_longevity_hours
        duration_text = re.search(r"(\d+(?:\.\d+)?)\s*(?:시간|hours?|h)\s*(?:이상|지속|lasting|minimum)", value.brief, re.I)
        duration_text = duration_text or re.search(r"(?:at\s+least|minimum|>=|≥|지속(?:성|력)?)\s*(\d+(?:\.\d+)?)\s*(?:시간|hours?|h)", value.brief, re.I)
        if duration is not None or duration_text:
            unsupported.append({"code": "longevity_threshold_model_missing", "field": "minimum_longevity_hours",
                "message": "시간별 향 프로필은 계산하지만 완제품 감지 지속시간의 최소 보장은 아직 지원하지 않습니다."})
        budget = request.budget
        if budget:
            for field, label, kind in (
                ("scope", "예산은 향료비만인가요, 베이스를 포함한 전체 원료비인가요?", "budget_scope"),
                ("product_density_g_ml", "완제품 밀도(g/mL)를 입력해 주세요.", "number"),
                ("krw_per_catalog_price_unit", "카탈로그 가격 단위당 원화 환산값을 입력해 주세요.", "number"),
                ("price_basis_reference", "가격 단위와 환산값의 근거를 입력해 주세요.", "text"),
                ("price_as_of", "가격 기준일을 입력해 주세요.", "date"),
            ):
                if getattr(budget, field) is None:
                    question("budget." + field, label, kind)
            if budget.scope == "finished_product_materials" and budget.base_material_cost_krw is None:
                question("budget.base_material_cost_krw", "동일 기준 용량의 향료 외 원료비를 입력해 주세요.", "number")
            if budget.price_as_of and budget.price_as_of > date.today():
                raise ValueError("price_as_of cannot be in the future")
        elif re.search(r"₩|원\s*(?:이하|미만|/)|\d\s*원|KRW", value.brief, re.I):
            question("budget", "문장의 예산을 금액·통화·기준 용량으로 확인해 주세요.", "budget")
        if request.concentration_range:
            low, high = request.concentration_range.minimum, request.concentration_range.maximum
            points = list(dict.fromkeys((low, (low + high) / 2, high)))
        else:
            points = [value.product_concentration_percent]
            if re.search(r"\d\s*(?:%\s*)?[~～–]\s*\d", value.brief):
                question("concentration_range", "농도 범위의 최소·최대값을 확인해 주세요.", "percent_range")
        explicit_concentration = re.search(r"(?:향료\s*)?농도\s*(\d+(?:\.\d+)?)\s*%", value.brief)
        if explicit_concentration:
            fixed = float(explicit_concentration.group(1))
            if not 0 < fixed <= 30:
                raise ValueError("natural-language concentration must be in (0, 30]")
            if request.concentration_range and not low <= fixed <= high:
                unsupported.append({"code": "conflicting_concentration", "field": "concentration_range",
                                    "message": "문장에 지정한 농도가 입력 범위 밖입니다."})
            else:
                points = [fixed]
        scenarios = []
        for index, concentration in enumerate(points):
            limit = value.max_formula_cost_per_kg
            if budget and not any(row["id"].startswith("budget") for row in questions):
                base_cost = budget.base_material_cost_krw if budget.scope == "finished_product_materials" else 0.
                available = budget.amount - base_cost
                if available <= 0:
                    unsupported.append({"code": "budget_exhausted_by_base", "field": "budget", "message": "베이스 비용이 예산 이상입니다."})
                    continue
                mass = budget.per_volume_ml * budget.product_density_g_ml / 1000 * concentration / 100
                if not math.isfinite(mass) or mass <= 0:
                    raise ValueError("budget mass is outside representable precision")
                limit = min(limit, available / (mass * budget.krw_per_catalog_price_unit))
            scenarios.append({"index": index, "product_concentration_percent": concentration,
                              "max_formula_cost_per_kg": limit, "product_category": product})
        return {"schema_version": "ai-brief-1", "request_id": digest,
            "interpretation": interpretation(brief),
            "parsed_conditions": {"target_region": brief.constraints.target_region,
                                  "product_concentration_percent": brief.constraints.product_concentration_percent},
            "status": "unsupported_requirements" if unsupported else "needs_clarification" if questions else "ready",
            "original_text": value.brief, "intent": {"target_profile": brief.target_profile,
                "representation": representation_contract(brief),
                "desired_dimensions": brief.desired_dimensions, "avoided_dimensions": brief.avoided_dimensions,
                "phase_target_profiles": brief.phase_target_profiles, "intensity": brief.intensity,
                "phase_avoided_dimensions": brief.phase_avoided_dimensions,
                "absolute_intensity_target": brief.absolute_intensity_target,
                "requested_ingredient_ids": sorted({identifier for name in brief.requested_ingredients for identifier in name_to_ids.get(normalize_name(name), ())}),
                "semantic_backend": brief.semantic_backend, "recognized_descriptors": brief.recognized_descriptors},
            "questions": questions, "unsupported_requirements": unsupported,
            "product_preferences": request.product_preferences.model_dump(mode="json", exclude_none=True),
            "application_context": assess_application_context(context) if context else None,
            "formulation_workflow": workflow,
            "model_persistence_requirement": request.model_persistence.model_dump() if request.model_persistence else None,
            "effective_target": max(95., value.target_similarity), "scenarios": scenarios,
            "budget_basis": "caller_supplied_estimate_not_verified_market_quote" if budget else "catalog_price_units_per_kg",
            "scope": "structured review, not unrestricted language understanding; scenarios vary concentration, not guaranteed distinct accords"}

    @app.post("/v1/briefs/prepare")
    def prepare_endpoint(request: ExtendedRequest):
        rate_limit()
        try:
            return prepare(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/briefs/clarify")
    def clarify_endpoint(value: ClarificationRequest):
        rate_limit()
        try:
            previous = prepare(value.request)
            if previous["request_id"] != value.request_id:
                raise HTTPException(status_code=409, detail="request changed; prepare the current request again")
            updated = apply_question_answers(value.request.model_dump(mode="json"), previous["questions"], value.answers)
            request = ExtendedRequest.model_validate(updated)
            return {"schema_version": "ai-clarification-1", "previous_request_id": previous["request_id"],
                    "applied_answer_ids": sorted(value.answers), "request": request.model_dump(mode="json"),
                    "prepared": prepare(request), "execution_mode": "no_recipe_inference",
                    "scope": "stateless_answer_application_backend_owns_session_and_storage"}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def controls_for(request):
        return {key: getattr(request.edits, key) for key in ("intensity_level", "phase_target_profiles")
                if getattr(request.edits, key) is not None}

    def evaluate_core(request, response, *, cache_candidate=True, fixed_formula_weights=None):
        try:
            prepared = prepare(request)
            if prepared["status"] != "ready":
                raise HTTPException(status_code=422, detail=prepared)
            if request.scenario_index >= len(prepared["scenarios"]):
                raise ValueError("scenario_index does not exist in this concentration plan")
            scenario = prepared["scenarios"][request.scenario_index]
            values = request.formula.model_dump()
            values.update({key: scenario[key] for key in ("product_concentration_percent", "max_formula_cost_per_kg", "product_category")})
            # Revalidate edits instead of trusting a client-supplied prepared response.
            formula = formula_type.model_validate(values)
            payload = generate_formula(formula, response, target_profile_override=request.edits.target_profile,
                                       explicit_bans=request.edits.excluded_ingredient_ids, rate_limited=True,
                                       **({"intent_controls": controls_for(request)} if controls_for(request) else {}),
                                       **({"fixed_formula_weights": fixed_formula_weights} if fixed_formula_weights is not None else {}))
            effective = payload["brief"]["constraints"]
            if (effective["product_category"] != scenario["product_category"]
                    or abs(effective["product_concentration_percent"] - scenario["product_concentration_percent"]) > 1e-9):
                raise ValueError("natural-language product/concentration conflicts with the structured scenario")
            persistence = None
            if request.model_persistence:
                persistence = model_persistence(payload.get("temporal_profile", []), request.model_persistence.relative_threshold_percent,
                                                request.model_persistence.required_duration_minutes)
            persistence_required = bool(request.model_persistence and request.model_persistence.required_duration_minutes is not None)
            persistence_met = not persistence_required or persistence["meets_requirement"] is True
            accepted = bool(payload.get("recipe")) and bool(payload.get("full_profile_target_met")) and persistence_met
            lines = payload.get("recipe") or payload.get("closest_candidate") or []
            cost = None
            if request.budget and lines:
                budget = request.budget
                fragrance_kg = budget.per_volume_ml * budget.product_density_g_ml / 1000 * scenario["product_concentration_percent"] / 100
                cost = payload["estimated_concentrate_cost_per_kg"] * fragrance_kg * budget.krw_per_catalog_price_unit
                if budget.scope == "finished_product_materials":
                    cost += budget.base_material_cost_krw
                if accepted and cost > budget.amount + 1e-6:
                    raise ValueError("computed candidate exceeds the explicit budget")
            runtime = payload.get("deployment") or {}
            identity = {"request_id": prepared["request_id"], "scenario_index": request.scenario_index,
                        "formula_id": payload.get("formula_id", ""), "runtime": runtime}
            candidate = {"candidate_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "formula_id": payload.get("formula_id"), "status": "target_met" if accepted else "candidate_only" if lines else "no_candidate",
                "assessment_kind": payload.get("score_contract", {}).get("assessment_kind", "generated_candidate"),
                "target_match_score": payload.get("calculated_profile_similarity"), "target_match_unit": "model_points_0_100",
                "product_concentration_percent": scenario["product_concentration_percent"], "concentration_basis": "w/w_percent",
                "composition_total_percent": sum(row["concentrate_percent"] for row in lines), "material_count": len(lines),
                "cost_krw": cost, "cost_per_volume_ml": request.budget.per_volume_ml if request.budget else None,
                "cost_basis": prepared["budget_basis"], "longevity_hours": None,
                "longevity_status": "not_estimated", "temporal_profile": payload.get("temporal_profile", []),
                "model_persistence": persistence,
                "selection_gates": {"profile_target_met": bool(payload.get("full_profile_target_met")),
                    "model_persistence_required": persistence_required,
                    "model_persistence_requirement_met": persistence["meets_requirement"] if persistence_required else None},
                "display_timeline": display_timeline(payload.get("temporal_profile", [])),
                "regulatory": payload.get("regulatory"), "evidence": evidence_summary(payload), "runtime": runtime, "result": payload}
            candidate['formulation_workflow'] = prepared['formulation_workflow']
            if cache_candidate:
                remember(candidate, prepared["request_id"], request.scenario_index)
            return {"schema_version": "ai-candidates-1", "request_id": prepared["request_id"],
                "scenario_index": request.scenario_index, "scenario_count": len(prepared["scenarios"]),
                "runtime": runtime, "candidates": [candidate], "execution_mode": "one_scenario_per_call_backend_may_queue_remaining"}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/formulas/evaluate")
    def evaluate_endpoint(request: ExtendedRequest, response: Response):
        rate_limit()
        return evaluate_core(request, response)

    @app.post("/v1/formulas/reassess")
    def reassess(request: FixedRequest, response: Response):
        rate_limit()
        weights = {row.ingredient_id: row.concentrate_percent for row in request.lines}
        if len(weights) != len(request.lines):
            raise HTTPException(status_code=422, detail="fixed formula ingredient IDs must be unique")
        base = ExtendedRequest.model_validate(request.model_dump(exclude={"lines"}))
        return evaluate_core(base, response, fixed_formula_weights=weights)

    @app.post("/v1/formulas/alternatives")
    def alternatives(request: AlternativeRequest, response: Response):
        rate_limit()
        base = ExtendedRequest.model_validate(request.model_dump(exclude={"previous_candidate_ids", "variation_index", "minimum_composition_distance"}))
        try:
            prepared = prepare(base)
            if prepared["status"] != "ready":
                raise HTTPException(status_code=422, detail=prepared)
            if len(set(request.previous_candidate_ids)) != len(request.previous_candidate_ids):
                raise ValueError("previous_candidate_ids must be unique")
            previous = [recall(identity) for identity in request.previous_candidate_ids]
            if any(not row["composition"] for row in previous):
                raise ValueError("alternatives require previous candidates with a composition")
            if any(row["context_id"] != prepared["request_id"] or row["scenario_index"] != base.scenario_index for row in previous):
                raise ValueError("alternatives require the same original request and concentration scenario")
            protected = set(prepared["intent"]["requested_ingredient_ids"]) | set(base.edits.excluded_ingredient_ids)
            weights = {}
            for row in previous:
                for identifier, value in row["composition"].items():
                    if identifier not in protected:
                        weights[identifier] = max(weights.get(identifier, 0.), value)
            anchors = sorted(weights, key=lambda identifier: (-weights[identifier], identifier))
            if request.variation_index >= len(anchors):
                return {"status": "diversity_options_exhausted", "request_id": prepared["request_id"], "candidates": [], "available_variations": len(anchors)}
            excluded = anchors[:request.variation_index + 1]
            edited = base.model_dump(mode="json")
            edited["edits"]["excluded_ingredient_ids"] = sorted(set(base.edits.excluded_ingredient_ids) | set(excluded))
            alternative = evaluate_core(ExtendedRequest.model_validate(edited), response, cache_candidate=False)
            candidate = alternative["candidates"][0]
            lines = candidate["result"].get("recipe") or candidate["result"].get("closest_candidate") or []
            composition = {row["ingredient_id"]: row["concentrate_percent"] for row in lines}
            distances = [composition_distance(composition, row["composition"]) for row in previous]
            distinct = bool(composition) and min(distances) + 1e-9 >= request.minimum_composition_distance
            accepted = candidate["status"] == "target_met" and distinct
            candidate["diversity"] = {"composition_distances": distances, "metric": "concentrate_total_variation_not_odor_distance",
                "excluded_for_this_search": excluded, "minimum_distance": request.minimum_composition_distance}
            if accepted:
                remember(candidate, prepared["request_id"], base.scenario_index, excluded)
            return {"schema_version": "ai-alternatives-1", "request_id": prepared["request_id"],
                "status": "distinct_target_met" if accepted else "target_not_met" if candidate["status"] != "target_met" else "insufficient_diversity",
                "candidates": [candidate] if accepted else [], "diagnostic_candidate": None if accepted else candidate,
                "next_variation_index": request.variation_index + 1 if request.variation_index + 1 < len(anchors) else None,
                "available_variations": len(anchors), "execution_mode": "one_new_inference_per_call"}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/formulas/compare")
    def compare(request: ComparisonRequest):
        rate_limit()
        if len(set(request.candidate_ids)) != len(request.candidate_ids):
            raise HTTPException(status_code=422, detail="candidate IDs must be unique")
        rows = [recall(identity) for identity in request.candidate_ids]
        if len({row["context_id"] for row in rows}) != 1 or len({json.dumps(row["runtime"], sort_keys=True) for row in rows}) != 1:
            raise HTTPException(status_code=422, detail="comparison requires the same original request and runtime")
        pairs = [{"left": left["candidate_id"], "right": right["candidate_id"],
                  "composition_distance": composition_distance(left["composition"], right["composition"]),
                  "concentration_difference_points": abs(left["product_concentration_percent"] - right["product_concentration_percent"])}
                 for i, left in enumerate(rows) for right in rows[i + 1:]]
        return {"schema_version": "ai-comparison-1", "candidates": rows, "pairs": pairs,
                "source": "server_generated_cached_results", "new_inference_count": 0,
                "cache_scope": "this_runtime_only_30_minutes_maximum_not_durable_storage"}

    def revise_request(request):
        if re.search(r"\d\s*%|퍼센트|percentage", request.instruction, re.I):
            raise ValueError("quantitative edits must use edits.target_profile; relative text edits are qualitative")
        if PHASE_MARKER.search(request.instruction) or re.search(r"첫\s*향|중간\s*향|잔\s*향", request.instruction, re.I):
            raise ValueError("phase-specific relative edits are not supported by this endpoint")
        base = ExtendedRequest.model_validate(request.model_dump(exclude={"instruction"}))
        prepared = prepare(base)
        if prepared["status"] != "ready":
            raise HTTPException(status_code=422, detail=prepared)
        revised, adjustments = apply_relative_revision_profile(prepared["intent"]["target_profile"], request.instruction)
        if not adjustments:
            raise ValueError("no supported relative scent edit found; constraints are not changed by revision text")
        edited = base.model_dump(mode="json")
        edited["edits"]["target_profile"] = revised
        return ExtendedRequest.model_validate(edited), prepared, adjustments

    @app.post("/v1/briefs/revise")
    def revise_intent(request: RevisionRequest):
        rate_limit()
        try:
            edited, previous, adjustments = revise_request(request)
            return {"prepared": prepare(edited), "request": edited.model_dump(mode="json"), "previous_request_id": previous["request_id"],
                    "adjustments": adjustments, "scope": "global_scent_axes_only_other_constraints_unchanged"}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/formulas/revise")
    def revise_formula(request: RevisionRequest, response: Response):
        rate_limit()
        try:
            edited, previous, adjustments = revise_request(request)
            result = evaluate_core(edited, response)
            result["revision"] = {"previous_request_id": previous["request_id"], "adjustments": adjustments,
                                  "scope": "regenerated_and_reassessed_against_edited_intent_not_manual_weight_validation"}
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def rd_runtime_contract():
        from ..recommender.science import SCIENTIFIC_MODEL_VERSION
        if runtime_guard:
            runtime_guard()
        current = capabilities()
        return {"catalog": catalog_contract or {}, "scientific_model_version": SCIENTIFIC_MODEL_VERSION,
                "perception_model": current["perception_model"],
                "product_models": current["product_models"],
                "odor_expression": current["odor_expression"]}

    register_rd_api(app, ExtendedRequest, prepare, evaluate_core, catalog, rate_limit,
                    rd_runtime_contract, evidence_store=evidence_store, product_codes=supported_products,
                    revise=lambda base, instruction: revise_request(RevisionRequest.model_validate(
                        {**base.model_dump(mode="json"), "instruction": instruction})))
