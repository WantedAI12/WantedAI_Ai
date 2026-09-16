"""Recipe presence must not falsify scores, quality decisions or R&D gates."""
from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform import ai_extensions
from fragrance_ai.platform.backend_wire_contract import (
    FormulaGenerationResponse, RECIPE_DELIVERY_CONTRACT, recipe_delivery_response)
from tests.test_ai_extensions import Formula
from tests.test_audit_reporting_api import parse_sse
from tests.test_rd_contract import data as data, POLICY


def payload(product='perfume', score=85.56):
    value = {'recipe': [], 'closest_candidate': [{'ingredient_id': 'linalyl_acetate',
        'concentrate_percent': 100., 'finished_product_percent': 15.}],
        'status': 'no_safe_match' if product == 'perfume' else 'research_candidate_only',
        'safety': {'internal_gate_passed': True, 'violations': []},
        'manufacturing_approved': False, 'human_similarity_percent': None,
        'confidence': None, 'confidence_kind': 'heuristic_only',
        'score_scope': 'complete_requested_profile'}
    value['calculated_profile_similarity' if product == 'perfume' else 'score'] = score
    value['full_profile_target_met' if product == 'perfume' else 'profile_target_met'] = False
    return value


@pytest.mark.parametrize('product', ['perfume', 'body_lotion'])
@pytest.mark.parametrize('score', [0., 79., 80., 85.56, 89.999])
def test_below_target_recipe_is_delivered_with_the_unmodified_score(product, score):
    original = payload(product, score)
    before = deepcopy(original)
    value = recipe_delivery_response(original, product=product)
    assert value['recipe'] == original['closest_candidate']
    assert value['target_match_score'] == score
    assert value['target_match_met'] is False
    assert value['model_accuracy_percent'] is None
    assert value['recipe_delivery']['status'] == 'target_not_met'
    assert value['recipe_delivery']['source'] == 'closest_candidate'
    assert value['recipe_delivery']['approval_inferred'] is False
    changed = {'recipe', 'status'} if product == 'perfume' else {'recipe'}
    assert all(value[k] == v for k, v in before.items() if k not in changed)
    if product == 'perfume':
        assert value['status'] == 'recipe_generated_target_not_met'
        assert value['assessment_status'] == before['status']
    assert original == before
    value['recipe'][0]['concentrate_percent'] = 99.
    assert original == before


@pytest.mark.parametrize('product', ['perfume', 'body_lotion'])
def test_existing_passed_results_are_preserved(product):
    original = payload(product, 96.)
    original['recipe'], original['closest_candidate'] = original['closest_candidate'], []
    original['full_profile_target_met' if product == 'perfume' else 'profile_target_met'] = True
    value = recipe_delivery_response(original, product=product)
    assert all(value[k] == v for k, v in original.items())
    assert value['recipe_delivery']['status'] == 'target_met'
    assert value['recipe_delivery']['source'] == 'recipe'


@pytest.mark.parametrize('product', ['perfume', 'body_lotion'])
@pytest.mark.parametrize('reason', ['missing_score', 'partial_score', 'safety', 'unknown_status', 'physical'])
def test_missing_evidence_and_existing_hard_blocks_are_not_promoted(product, reason):
    original = payload(product)
    if reason == 'missing_score':
        original['calculated_profile_similarity' if product == 'perfume' else 'score'] = None
    elif reason == 'partial_score':
        original['score_scope'] = 'resolved_positive_intent_only'
    elif reason == 'safety':
        original['safety']['violations'] = ['fixture restriction']
    elif reason == 'physical':
        original['perceptual_evaluation'] = {'physical_presence_passed': False}
    else:
        original['status'] = 'unsupported_requirements'
    value = recipe_delivery_response(original, product=product)
    assert not value['recipe']
    assert value['closest_candidate'] == original['closest_candidate']
    assert value['recipe_delivery']['status'] == 'blocked'
    assert value['recipe_delivery']['blockers']


@pytest.mark.parametrize('product', ['perfume', 'body_lotion'])
def test_delivering_composition_does_not_clear_regulatory_review_blocks(product):
    original = payload(product)
    original['regulatory'] = {'status': 'blocked', 'tabs': [{'id': 'IFRA', 'status': 'blocked'}]}
    value = recipe_delivery_response(original, product=product)
    assert value['recipe'] == original['closest_candidate']
    assert value['regulatory'] == original['regulatory']
    assert value['recipe_delivery']['approval_inferred'] is False
    assert value['manufacturing_approved'] is False


@pytest.mark.parametrize('invalid', [True, '90', float('nan'), float('inf'), -1., 101.])
def test_invalid_scores_are_rejected_instead_of_being_repaired(invalid):
    with pytest.raises(ValueError, match='target match'):
        recipe_delivery_response(payload(score=invalid), product='perfume')


@pytest.mark.parametrize('invalid', ['duplicate', 'sum', 'negative', 'nan', 'boolean'])
def test_invalid_compositions_cannot_be_exposed(invalid):
    value = payload()
    if invalid == 'duplicate':
        value['closest_candidate'] *= 2
    else:
        value['closest_candidate'][0]['concentrate_percent'] = {
            'sum': 99., 'negative': -100., 'nan': float('nan'), 'boolean': True}[invalid]
    with pytest.raises(ValueError):
        recipe_delivery_response(value, product='perfume')


def test_unknown_model_accuracy_stays_null_and_schema_is_numeric():
    value = recipe_delivery_response(payload(), product='perfume')
    parsed = FormulaGenerationResponse.model_validate(value)
    assert parsed.target_match_score == 85.56 and parsed.model_accuracy_percent is None
    schema = FormulaGenerationResponse.model_json_schema()['properties']
    assert schema['target_match_score']['anyOf'][0]['type'] == 'number'
    assert schema['model_accuracy_percent']['anyOf'][0]['type'] == 'number'
    assert schema['model_accuracy_percent']['anyOf'][1]['type'] == 'null'


def test_lotion_design_and_optimize_deliver_the_same_scored_candidate(monkeypatch):
    from tests.test_lotion_v21 import fixture
    request, catalog = fixture()
    ingredient = catalog.ingredients[0].ingredient_id
    calls = []
    def generate(*a, **kw):
        calls.append(1)
        value = payload('body_lotion')
        value['closest_candidate'][0]['ingredient_id'] = ingredient
        return value
    monkeypatch.setattr(ai_extensions, 'estimate_lotion_recipe', generate)
    monkeypatch.setattr(ai_extensions, 'optimize_lotion', generate)
    app = FastAPI()
    ai_extensions.register_ai_extensions(app, Formula, catalog, None, lambda: None)
    with TestClient(app) as client:
        for path, body in (('design', {'brief': 'woody scent'}), ('optimize', request)):
            response = client.post('/v1/applications/body-lotion/' + path, json=body)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['recipe'] and result['target_match_score'] == result['score'] == 85.56
            assert not result['profile_target_met'] and not result['manufacturing_approved']
            assert response.headers['X-Perfumery-Recipe-Contract'] == RECIPE_DELIVERY_CONTRACT
    assert len(calls) == 2


def test_extended_and_rd_workflows_do_not_accept_a_below_target_delivered_recipe(data):
    catalog, factory, _, _, _ = data
    item = next(row for row in catalog.ingredients if row.cas_number)
    def generate(request, response, **kw):
        value = payload()
        value.update(brief={'constraints': request.model_dump()}, formula_id='fixture-only',
                     estimated_concentrate_cost_per_kg=50., temporal_profile=[], scientific_model_domain_passed=True)
        value['closest_candidate'][0]['ingredient_id'] = item.ingredient_id
        return value
    app = FastAPI()
    ai_extensions.register_ai_extensions(app, Formula, catalog, generate, lambda: None, evidence_store=factory())
    formula = {'brief': 'woody scent', 'product_category': 'eau_de_parfum', 'target_region': 'EU',
               'product_concentration_percent': 15., 'max_formula_cost_per_kg': 180.}
    with TestClient(app) as client:
        result = client.post('/v1/formulas/evaluate', json={'formula': formula}).json()['candidates'][0]
        assert result['result']['recipe'] and result['status'] == 'candidate_only'
        assert not result['selection_gates']['profile_target_met']
        request = {'request': {'formula': formula}, 'evidence_policy': POLICY}
        review = client.post('/v2/briefs/prepare', json=request).json()
        response = client.post('/v2/formulas/evaluate', json={**request, 'confirmed_review_id': review['review_id']})
        assert response.status_code == 200, response.text
        value = response.json()
        assert not value['candidates'] and value['status'] == 'abstained'
        diagnostic = value['diagnostic_candidates'][0]
        assert diagnostic['result']['recipe']
        assert not diagnostic['recommendation_allowed'] and not diagnostic['rd_gates']['profile_and_persistence']


def test_sse_preserves_the_delivered_recipe_and_score_verbatim():
    from deploy.formula_stream import register_formula_stream
    value = recipe_delivery_response(payload(), product='perfume')
    def generate(request, response, **kw):
        return value
    app = FastAPI()
    register_formula_stream(app, Formula, generate, lambda: None, lambda: None)
    with TestClient(app) as client:
        rows = parse_sse(client.post('/v1/formulas/stream', json={'brief': 'woody scent'}).text)
    assert next(row['data'] for row in rows if row['event'] == 'result') == value
