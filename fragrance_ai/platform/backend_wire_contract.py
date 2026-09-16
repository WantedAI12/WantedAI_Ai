"""Wire-level recipe delivery, goal agreement and nullable confidence."""
from copy import deepcopy
import math
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat
from ..recommender.confidence_contract import confidence_fields

CONFIDENCE_CONTRACT='nullable-number-v1'
RECIPE_DELIVERY_CONTRACT='best-available-recipe/v1'


class RecipeDelivery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    contract: Literal['best-available-recipe/v1'] = RECIPE_DELIVERY_CONTRACT
    status: Literal['target_met', 'target_not_met', 'target_unverified', 'blocked', 'unavailable']
    source: Literal['recipe', 'closest_candidate', 'none']
    returned: bool
    approval_inferred: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)


class FormulaGenerationResponse(BaseModel):
    model_config=ConfigDict(extra='allow',allow_inf_nan=False)
    confidence: StrictFloat | None = Field(description='Number in [0,1] or null; never an evidence label.',ge=0,le=1)
    confidence_kind: str
    target_match_score: StrictFloat | None = Field(default=None, ge=0, le=100,
        description='Calculated agreement of this formula with the requested scent; not model accuracy.')
    target_match_unit: Literal['model_points_0_100'] = 'model_points_0_100'
    target_match_met: bool | None = None
    model_accuracy_percent: StrictFloat | None = Field(default=None, ge=0, le=100,
        description='No validated model-wide accuracy percentage is supplied by recipe generation.')
    model_accuracy_status: Literal['separate_benchmark_required'] = 'separate_benchmark_required'
    recipe_delivery: RecipeDelivery | None = None


def _valid_lines(lines):
    if not isinstance(lines, list) or not lines:
        raise ValueError('recipe must contain a nonempty list of composition lines')
    identities, amounts = set(), []
    for row in lines:
        if not isinstance(row, dict):
            raise ValueError('invalid recipe line')
        key, amount = row.get('ingredient_id'), row.get('concentrate_percent')
        if (not isinstance(key, str) or not key or key in identities
                or isinstance(amount, bool) or not isinstance(amount, Real)
                or not math.isfinite(amount) or not 0 < amount <= 100):
            raise ValueError('recipe requires unique ingredients and finite positive amounts')
        identities.add(key)
        amounts.append(float(amount))
    if abs(math.fsum(amounts) - 100.) > .002:
        raise ValueError('recipe concentrate must sum to 100 percent')


def recipe_delivery_response(payload, *, product):
    """Expose an existing scored composition without changing any quality gate.

    This runs after search/evaluation, never inside the optimizer. Unknown or
    partially evaluated targets and internal safety/physical blocks are not
    promoted. Regulatory review remains separate from delivering composition
    data, just as for existing above-target recipes. There is no score floor.
    """
    if product not in ('perfume', 'body_lotion'):
        raise ValueError('unsupported recipe delivery product')
    result = dict(payload)
    score_key, met_key = (('calculated_profile_similarity', 'full_profile_target_met')
                          if product == 'perfume' else ('score', 'profile_target_met'))
    score, met = payload.get(score_key), payload.get(met_key)
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('target match must be null or finite model points in [0,100]')
        score = float(score)
    if met is not None and type(met) is not bool:
        raise ValueError('target match decision must be boolean or null')
    result.update(target_match_score=score, target_match_unit='model_points_0_100', target_match_met=met,
                  model_accuracy_percent=None, model_accuracy_status='separate_benchmark_required')

    original = payload.get('recipe') or []
    closest = payload.get('closest_candidate') or []
    blockers = []
    source = 'recipe' if original else 'none'
    if original:
        _valid_lines(original)
    elif closest:
        _valid_lines(closest)
        if score is None:
            blockers.append('full_target_score_unavailable')
        scope = payload.get('score_scope') or (payload.get('full_profile_assessment') or {}).get('score_scope')
        if scope is not None and scope != 'complete_requested_profile':
            blockers.append('incomplete_target_scope')
        allowed = ('no_safe_match', 'candidate_only') if product == 'perfume' else ('research_candidate_only',)
        if payload.get('status') not in allowed:
            blockers.append('result_not_a_scored_quality_candidate')
        safety = payload.get('safety') or {}
        if product == 'perfume' and safety.get('internal_gate_passed') is not True:
            blockers.append('internal_safety_gate_not_passed')
        if safety.get('violations') or (safety.get('ifra_screen') or {}).get('compliant') is False:
            blockers.append('existing_safety_violation')
        evaluation = payload.get('perceptual_evaluation') or {}
        if evaluation.get('physical_presence_passed') is False:
            blockers.append('physical_presence_not_passed')
        if not blockers:
            # Copy only these small line dictionaries, not the full simulation.
            # The original candidate and cached numerical result stay intact.
            result['recipe'] = deepcopy(closest)
            source = 'closest_candidate'
            if product == 'perfume' and payload.get('status') == 'no_safe_match':
                result['assessment_status'] = payload['status']
                result['status'] = 'recipe_generated_target_not_met'
    provided = bool(result.get('recipe'))
    status = ('target_met' if met is True else 'target_not_met' if met is False and score is not None
              else 'target_unverified') if provided else ('blocked' if blockers else 'unavailable')
    result['recipe_delivery'] = RecipeDelivery(status=status, source=source, returned=provided,
        blockers=blockers).model_dump(mode='json')
    return result


def normalize_formula_response(payload):
    result=dict(payload)
    fields=confidence_fields(result.get('confidence'))
    result['confidence']=fields['confidence']
    if not isinstance(result.get('confidence_kind'),str) or not result['confidence_kind']:
        result['confidence_kind']=fields['confidence_kind']
    if 'simulation_confidence' in result:
        simulation = confidence_fields(result['simulation_confidence'])
        result['simulation_confidence'] = simulation['confidence']
        result.setdefault('simulation_confidence_kind', simulation['confidence_kind'])
    FormulaGenerationResponse.model_validate(result)
    return result
