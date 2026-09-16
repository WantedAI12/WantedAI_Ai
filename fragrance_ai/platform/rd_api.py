"""Strict R&D request review and evidence APIs, additive to the v1 contract."""
from datetime import date
from typing import Literal

from fastapi import HTTPException, Response, Query
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .rd_evidence import (Input, EvidencePolicy, EvidenceAssessment, FormulaLine,
                          EvidenceStore, content_id)
from .rd_clarification import REQUIRED_FORMULA_FIELDS, answer_rd_questions, question_for
from .rd_policy import ReviewEvidencePolicy, resolve_policy, policy_question
from .rd_snapshots import (RevisionSource, SnapshotComparison, SnapshotRevision,
                           compare_snapshots, composition_diff, line_map)


class ChangeImpactResponse(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    schema_version: Literal['rd-change-impact-1']
    status: Literal['comparison_completed']
    comparison_kind: Literal['public_sources', 'operator_evidence']
    diagnostic_only: bool
    before: dict[str, JsonValue]
    after: dict[str, JsonValue]
    changes: list[dict[str, JsonValue]]
    affected_material_count: int = Field(ge=0)
    review_required: bool
    state_changed: Literal[False]
    manufacturing_approval: Literal[False]
    scope: str
    result_id: str
    contract: dict[str, JsonValue]


class MissingEvidenceDetail(BaseModel):
    code: Literal['EVIDENCE_SNAPSHOTS_MISSING']
    status: Literal['abstained']
    message: str
    operator_bundle_registered: Literal[False]
    public_sources_registered: Literal[False]
    state_changed: Literal[False]


class ChangeImpactErrorResponse(BaseModel):
    detail: MissingEvidenceDetail | str | list[dict[str, JsonValue]]


def interpretation(brief=None, *, unsupported=False):
    """Expose parser evidence without inventing calibrated OOD probabilities."""
    return {
        "schema_version": "intent-reliability-1",
        "confidence": {"value": getattr(brief, "semantic_confidence", None),
                       "kind": "parser_heuristic_not_calibrated_probability",
                       "backend": getattr(brief, "semantic_backend", None)},
        "ood": {"status": "outside_supported_vocabulary" if unsupported else
                          "not_assessable" if brief is None else "no_known_vocabulary_violation",
                "method": "supported_descriptor_and_requirement_checks",
                "probability": None, "full_text_understanding_verified": False},
        "uncertainty": {"calibrated": False, "interval": None,
                        "reason": "no_calibrated_natural_language_error_model"},
    }


def register_rd_api(app, request_type, prepare, evaluate, catalog, rate_limit,
                    runtime_contract, *, evidence_store=None, product_codes=(), revise=None):
    store = evidence_store if evidence_store is not None else EvidenceStore.configured()

    class PrepareRequest(Input):
        request: request_type
        evidence_policy: ReviewEvidencePolicy
        lines: list[FormulaLine] | None = Field(default=None, min_length=1, max_length=50000)
        revision: RevisionSource | None = None
        diagnostic_only: bool = False

    class ClarifyRequest(PrepareRequest):
        prepared_result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
        answers: dict[str, JsonValue]

    class EvaluateRequest(PrepareRequest):
        confirmed_review_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    class ReassessRequest(EvaluateRequest):
        lines: list[FormulaLine] = Field(min_length=1, max_length=50000)

    class ChangeRequest(EvidenceAssessment):
        previous_evidence_version: str = Field(min_length=1, max_length=200)

    def current_contract():
        store.assert_current()
        return {"api_version": "rd-api-2", "runtime": runtime_contract(),
                "evidence": store.contract(), "evaluation_date": date.today().isoformat()}

    def review(value):
        prepared = prepare(value.request)
        policy, missing_policy, policy_context = resolve_policy(value.evidence_policy, value.diagnostic_only)
        # Explicit submission is distinct from a default silently filled by pydantic.
        required = tuple(REQUIRED_FORMULA_FIELDS)
        missing = [key for key in required if key not in value.request.formula.model_fields_set]
        conflicts = []
        formula = value.request.formula
        parsed = prepared.get("parsed_conditions", {})
        if parsed.get("target_region", formula.target_region) != formula.target_region:
            conflicts.append("target_region")
        if (value.request.concentration_range is None and
                parsed.get("product_concentration_percent", formula.product_concentration_percent) != formula.product_concentration_percent):
            conflicts.append("product_concentration_percent")
        if any(row["product_category"] != formula.product_category for row in prepared.get("scenarios", [])):
            conflicts.append("product_category")
        if value.lines:
            line_map([line.model_dump() for line in value.lines])
            if policy is not None:
                EvidenceAssessment(lines=value.lines, target_region=formula.target_region,
                    product_category=formula.product_category,
                    product_concentration_percent=formula.product_concentration_percent,
                    max_formula_cost_per_kg=formula.max_formula_cost_per_kg, policy=policy)
        contract = current_contract()
        reviewed = {"request": value.request.model_dump(mode="json"),
                    "evidence_policy": (policy or value.evidence_policy).model_dump(mode="json"),
                    "lines": [line.model_dump() for line in value.lines] if value.lines else None,
                    "prepared": prepared, "contract": contract}
        if value.diagnostic_only:
            reviewed['diagnostic_only'] = True
        if policy_context is not None:
            reviewed['evidence_policy_context'] = policy_context
        if value.revision is not None:
            reviewed["revision"] = value.revision.model_dump(mode="json")
        operational_policy_missing = missing_policy if not value.diagnostic_only else []
        status = "needs_input" if missing or operational_policy_missing else "needs_clarification" if conflicts else prepared["status"]
        result = {"schema_version": "rd-brief-2", "status": status,
                  "missing_fields": ["request.formula." + key for key in missing] +
                                    ['evidence_policy.' + key for key in operational_policy_missing],
                  "conflicting_fields": ["request.formula." + key for key in conflicts],
                  "questions": [{**row, "field": "request." + row["id"],
                                 "reason_code": row.get("reason_code", "input_clarification_required")}
                                for row in prepared.get("questions", [])] + [
                      question_for(key, conflict=key in conflicts, product_codes=product_codes)
                                for key in dict.fromkeys(missing + conflicts)] +
                               [policy_question(key) for key in operational_policy_missing],
                  "prepared": prepared, "reviewed_request": reviewed["request"],
                  "reviewed_lines": reviewed["lines"],
                  "evidence_policy": reviewed["evidence_policy"], "contract": contract,
                  "review_id": content_id(reviewed) if status == "ready" else None,
                  "confirmation_required": True, "recipe_generated": False}
        if value.revision is not None:
            result["revision"] = reviewed["revision"]
        if value.diagnostic_only:
            result['diagnostic_only'] = True
        if policy_context is not None:
            result['evidence_policy_context'] = policy_context
        result["result_id"] = content_id(result)
        return result

    @app.post("/v2/briefs/prepare")
    def prepare_v2(value: PrepareRequest):
        rate_limit()
        try:
            return review(value)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v2/briefs/clarify")
    def clarify_v2(value: ClarifyRequest):
        rate_limit()
        try:
            previous = review(value)
            if previous["result_id"] != value.prepared_result_id:
                raise HTTPException(status_code=409, detail="input, runtime, date or evidence changed; prepare again")
            # exclude_unset is essential: an unanswered missing value must not
            # become explicitly confirmed through model_dump's default values.
            original = value.model_dump(mode="json", exclude_unset=True,
                                        exclude={"answers", "prepared_result_id"})
            updated = answer_rd_questions(original, previous["questions"], value.answers)
            parsed = PrepareRequest.model_validate(updated)
            return {"schema_version": "rd-clarification-2", "previous_result_id": previous["result_id"],
                    "applied_answer_ids": sorted(value.answers), "request": updated,
                    "prepared": review(parsed), "new_inference_count": 0, "state_changed": False}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v2/formulas/compare")
    def compare_v2(value: SnapshotComparison):
        rate_limit()
        try:
            return compare_snapshots(value, current_contract())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v2/briefs/revise")
    def revise_v2(value: SnapshotRevision):
        rate_limit()
        try:
            if revise is None:
                raise ValueError("intent revision is not configured")
            source = value.source
            composition = source.composition()
            if not composition:
                raise ValueError("revision requires a source candidate with a formula")
            snapshot = source.evaluation["input_snapshot"]
            base = request_type.model_validate(snapshot["request"])
            edited, _, adjustments = revise(base, value.instruction)
            updated = {"request": edited.model_dump(mode="json"),
                       "evidence_policy": snapshot["evidence_policy"],
                       "revision": {"parent_result_id": source.evaluation["result_id"],
                           "parent_candidate_id": source.candidate_id,
                           "parent_backend_version_id": source.backend_version_id,
                           "parent_lines": [{"ingredient_id": key, "concentrate_percent": weight}
                                            for key, weight in composition.items()],
                           "instruction": value.instruction}}
            if source.evaluation.get('diagnostic_only') is True:
                updated['diagnostic_only'] = True
                # Diagnostic placeholders are not confirmed purchasing limits.
                context = snapshot.get('evidence_policy_context') or {}
                updated['evidence_policy'] = {key: item for key, item in snapshot['evidence_policy'].items()
                    if key not in context.get('defaulted_fields', {})}
            parsed = PrepareRequest.model_validate(updated)
            return {"schema_version": "rd-revision-2", "request": parsed.model_dump(mode="json", exclude_none=True),
                    "prepared": review(parsed), "adjustments": adjustments,
                    "new_inference_count": 0, "state_changed": False,
                    "next_operation": "/v2/formulas/evaluate",
                    "scope": "confirm_new_intent_then_regenerate_no_parent_approval_inherited"}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def run(value, response, lines=None):
        try:
            checked = review(value)
            if checked["status"] != "ready":
                raise HTTPException(status_code=422, detail=checked)
            if checked["review_id"] != value.confirmed_review_id:
                raise HTTPException(status_code=409, detail="input, runtime, date or evidence changed; review again")
            if store.bundle is None and not (value.diagnostic_only and store.public_store is not None):
                raise HTTPException(status_code=422, detail={"status": "abstained",
                    "reason": "registered_regulatory_and_supply_evidence_missing", "candidates": []})
            base = value.request
            if base.scenario_index >= len(checked["prepared"]["scenarios"]):
                raise ValueError("scenario_index does not exist")
            scenario = checked["prepared"]["scenarios"][base.scenario_index]
            policy = EvidencePolicy.model_validate(checked['evidence_policy'])

            def assessment_for(formula_lines):
                return EvidenceAssessment(lines=formula_lines, target_region=base.formula.target_region,
                    product_category=scenario["product_category"],
                    product_concentration_percent=scenario["product_concentration_percent"],
                    max_formula_cost_per_kg=scenario["max_formula_cost_per_kg"],
                    max_ingredient_price_per_kg=getattr(base.formula, "max_ingredient_price_per_kg", 300.),
                    policy=policy)

            fixed = None
            if lines is not None:
                assessment = assessment_for(lines)
                preflight = store.assess(assessment, catalog)
                if not preflight["gate_passed"] and not (value.diagnostic_only and store.public_store is not None):
                    raise HTTPException(status_code=422, detail={"status": "abstained", "candidates": [],
                                                               "evidence_assessment": preflight})
                fixed = {line.ingredient_id: line.concentrate_percent for line in lines}
            # v2 diagnostics never enter the legacy candidate cache.
            output = evaluate(base, response, cache_candidate=False, fixed_formula_weights=fixed)
            accepted, diagnostics = [], []
            for candidate in output["candidates"]:
                payload = candidate["result"]
                composition = payload.get("recipe") or payload.get("closest_candidate") or []
                assessed = store.assess(assessment_for([
                    {"ingredient_id": r["ingredient_id"], "concentrate_percent": r["concentrate_percent"]}
                    for r in composition]), catalog) if composition else None
                gates = {"registered_evidence": bool(assessed and assessed["gate_passed"]),
                         "existing_safety": (payload.get("safety") or {}).get("internal_gate_passed") is True,
                         "scientific_domain": payload.get("scientific_model_domain_passed") is True,
                         "profile_and_persistence": candidate["status"] == "target_met"}
                passed = all(gates.values()) and not value.diagnostic_only
                if assessed is not None:
                    from ..recommender.regulatory_status import regulatory_summary
                    payload['regulatory'] = regulatory_summary(payload, evidence_assessment=assessed)
                candidate.update(status="ready_for_review" if passed else "abstained",
                                 rd_gates=gates, evidence_assessment=assessed,
                                 recommendation_allowed=passed)
                # A changed input/evidence/runtime binding creates a new candidate identity.
                candidate["candidate_id"] = content_id({"review_id": checked["review_id"],
                    "formula_id": candidate["formula_id"], "assessment": assessed})
                (accepted if passed else diagnostics).append(candidate)
            if current_contract() != checked["contract"]:
                raise HTTPException(status_code=409, detail="runtime or evidence changed during evaluation; review again")
            result = {"schema_version": "rd-candidates-2", "review_id": checked["review_id"],
                      "contract": checked["contract"], "status": "ready_for_review" if accepted else "abstained",
                      "candidates": accepted, "diagnostic_candidates": diagnostics,
                      "state_changed": False, "manufacturing_approval": False}
            if value.diagnostic_only:
                result['diagnostic_only'] = True
                result['scope'] = 'explicit_public_source_diagnostic_not_operational_recommendation'
            result["input_snapshot"] = {"request": checked["reviewed_request"],
                "evidence_policy": checked["evidence_policy"], "scenario": scenario,
                "intent": checked["prepared"]["intent"], "reviewed_lines": checked["reviewed_lines"]}
            if 'evidence_policy_context' in checked:
                result['evidence_policy_context'] = checked['evidence_policy_context']
                result['input_snapshot']['evidence_policy_context'] = checked['evidence_policy_context']
            if value.revision is not None:
                result["revision"] = value.revision.model_dump(mode="json")
                before = {line.ingredient_id: line.concentrate_percent for line in value.revision.parent_lines}
                for candidate in accepted + diagnostics:
                    after = candidate["result"].get("recipe") or candidate["result"].get("closest_candidate") or []
                    candidate["revision_composition_diff"] = composition_diff(before,
                        {r["ingredient_id"]: r["concentrate_percent"] for r in after}) if after else None
            result["result_id"] = content_id(result)
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v2/formulas/evaluate")
    def evaluate_v2(value: EvaluateRequest, response: Response):
        rate_limit()
        if value.lines is not None:
            raise HTTPException(status_code=422, detail="use /v2/formulas/reassess for a reviewed fixed formula")
        return run(value, response)

    @app.post("/v2/formulas/reassess")
    def reassess_v2(value: ReassessRequest, response: Response):
        rate_limit()
        return run(value, response, value.lines)

    @app.post("/v2/formulas/assess-evidence")
    def evidence_v2(value: EvidenceAssessment):
        rate_limit()
        try:
            return store.assess(value, catalog)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get('/v2/evidence/status')
    def evidence_status():
        rate_limit()
        store.assert_current()
        coverage = store.public_store.coverage(catalog) if store.public_store else None
        return {'schema_version': 'rd-evidence-status-1', 'contract': store.contract(),
                'operator_bundle_registered': store.bundle is not None,
                'public_sources_registered': store.public_store is not None,
                'public_source_diagnostics_available': store.public_store is not None,
                'default_operational_gate_bypassed': False,
                'coverage':{key:value for key,value in coverage.items() if key != 'rows'} if coverage else None}

    @app.get('/v2/evidence/versions')
    def evidence_versions():
        rate_limit()
        store.assert_current()
        contract = store.contract()
        if store.bundle is not None:
            active = store.bundle.active_version
            versions = [s.version for s in store.bundle.snapshots if s.effective_on <= date.today()]
            kind = 'operator_evidence'
        else:
            public = contract.get('public_sources') or {}
            active = public.get('version')
            versions = [*public.get('previous_versions',[]), *([active] if active else [])]
            kind = 'public_sources'
        previous = [v for v in versions if v != active]
        return {'schema_version':'rd-evidence-versions-1','comparison_kind':kind,
                'active_version':active,'previous_versions':previous,
                'change_impact_available':bool(active and previous),
                'operator_bundle_registered':store.bundle is not None,
                'state_changed':False}

    @app.get('/v2/evidence/coverage')
    def evidence_coverage(offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
        rate_limit()
        store.assert_current()
        if store.public_store is None:
            raise HTTPException(status_code=422, detail='public source inventory is not configured')
        coverage = store.public_store.coverage(catalog)
        return {**{key:value for key,value in coverage.items() if key != 'rows'},
                'offset':offset,'limit':limit,'items':coverage['rows'][offset:offset+limit],
                'has_more':offset+limit<len(coverage['rows'])}

    @app.post("/v2/formulas/change-impact", response_model=ChangeImpactResponse,
              responses={422:{'model':ChangeImpactErrorResponse,
                  'description':'Missing evidence is abstained; invalid input/version also remains HTTP 422.'}})
    def impact_v2(value: ChangeRequest):
        rate_limit()
        if store.bundle is None and store.public_store is None:
            raise HTTPException(status_code=422,detail={
                'code':'EVIDENCE_SNAPSHOTS_MISSING','status':'abstained',
                'message':'change impact requires registered evidence snapshots',
                'operator_bundle_registered':False,'public_sources_registered':False,
                'state_changed':False})
        try:
            request = EvidenceAssessment.model_validate(value.model_dump(exclude={"previous_evidence_version"}))
            result = store.change_impact(request, catalog, value.previous_evidence_version)
            public = result['scope'].startswith('public_observation_')
            result.update(status='comparison_completed', comparison_kind='public_sources' if public else 'operator_evidence',
                          diagnostic_only=public, manufacturing_approval=False)
            result["contract"] = current_contract()
            result["result_id"] = content_id({k: v for k, v in result.items() if k != "result_id"})
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
