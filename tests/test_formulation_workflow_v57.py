from copy import deepcopy
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform.lotion_reference import lotion_reference
from fragrance_ai.platform.process_inputs import ManufacturingProcess, WorkflowRequest
from fragrance_ai.recommender import formulation_workflow as workflow
from fragrance_ai.recommender.compact_language import AssistantRequest, assistant_reply
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
from scripts.export_formulation_curriculum import curriculum


def context(oil=10., dose=.5):
    value = lotion_reference()['application_context']
    value['fragrance_concentration_percent'] = dose
    value['base_components'][0]['mass_percent'] = 94 - oil
    value['base_components'][2]['mass_percent'] = oil
    return value


def plan(base=None, **process):
    return workflow.formulation_workflow({'product_type': 'body_lotion',
        'application_context': context() if base is None else base, 'process': process})


def codes(value):
    return {row['code'] for row in value['checks']}


def test_source_integrity_and_no_training_claim():
    contract = workflow.knowledge_contract()
    assert contract['source_count'] == 6
    assert contract['sha256'] == hashlib.sha256((workflow.DATA / workflow.ASSET).read_bytes()).hexdigest()
    assert not contract['quantitative_model_weights_retrained']
    assert not contract['language_model_weights_retrained']
    assert not contract['legacy_master_perfumer_scores_used']
    assert contract['runtime_external_api_calls'] == 0
    value = workflow.formulation_workflow({'product_type': 'perfume'})
    assert value['trial_policy']['fixed_universal_note_percentages'] is None
    assert value['sensory_score_adjustment'] == 0 and value['human_similarity_percent'] is None
    value['sources'][0]['summary'] = 'MUTATED'
    assert 'MUTATED' not in json.dumps(workflow.formulation_workflow({'product_type': 'perfume'}))


def test_corrupt_source_fails_closed(tmp_path, monkeypatch):
    (tmp_path / workflow.ASSET).write_text('{}', encoding='utf-8')
    (tmp_path / 'data_manifest.json').write_text(json.dumps({'assets': {workflow.ASSET: {'sha256': '0' * 64}}}), encoding='utf-8')
    workflow._knowledge.cache_clear()
    monkeypatch.setattr(workflow, 'DATA', tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        workflow.knowledge_contract()
    workflow._knowledge.cache_clear()


@pytest.mark.parametrize('field,value', [('batch_mass_g', True), ('mixing_minutes', False),
    ('measured_ph', float('nan')), ('peak_temperature_c', float('inf')), ('batch_mass_g', 0),
    ('batch_mass_g', -1), ('mixing_minutes', 0), ('method', 'magic'), ('rpm', 100)])
def test_invalid_process_inputs(field, value):
    with pytest.raises(ValueError):
        ManufacturingProcess.model_validate({field: value})


def test_cross_domain_and_temperature_contract():
    with pytest.raises(ValueError):
        WorkflowRequest(product_type='perfume', application_context=context())
    with pytest.raises(ValueError):
        WorkflowRequest(product_type='perfume', process={'method': 'hot'})
    with pytest.raises(ValueError):
        ManufacturingProcess(peak_temperature_c=20, fragrance_addition_temperature_c=40)


@pytest.mark.parametrize('oil,dose,mass', [(10., .5, 400.), (20., 1., 2000.), (30., .25, 10.)])
def test_actual_base_dose_mass_balance(oil, dose, mass):
    base = context(oil, dose)
    before = deepcopy(base)
    value = plan(base, batch_mass_g=mass)
    assert base == before
    assert value['selected_method'] == 'cold'
    assert value['reference_parameters']['heating_temperature_c'] is None
    assert value['reference_parameters']['mixing_minutes'] == 4
    assert value['reference_parameters']['batch_mass_g'] == 400
    batch = value['batch']
    assert batch['total_percent'] == pytest.approx(100)
    assert batch['total_mass_g'] == pytest.approx(mass)
    lines = {row['role']: row for row in batch['lines']}
    assert lines['oil']['mass_g'] == pytest.approx(mass * oil / 100 * (1 - dose / 100))
    assert lines['fragrance']['mass_g'] == pytest.approx(mass * dose / 100)
    assert lines['preservative']['phase'] == 'A' and lines['fragrance']['phase'] == 'B'
    assert not value['manufacturing_approved'] and not value['scale_up_validated']


def test_application_temperature_is_not_manufacturing_temperature():
    base = context()
    base['temperature_c'] = 70
    value = plan(base)
    assert value['process']['peak_temperature_c'] is None
    assert 'unvalidated_cold_process_temperature' not in codes(value)


@pytest.mark.parametrize('change', ['reference_only', 'water_in_oil', 'wrong_role', 'excess_emulsifier'])
def test_reference_name_does_not_authorize_a_procedure(change):
    base = context()
    if change == 'reference_only':
        base['base_components'][3]['name'] = 'Unknown emulsifier'
    elif change == 'water_in_oil':
        base['emulsion_type'] = 'water_in_oil'
    elif change == 'wrong_role':
        base['base_components'][3]['role'] = 'oil'
    else:
        base['base_components'][0]['mass_percent'] -= 50
        base['base_components'][3]['mass_percent'] += 50
    value = plan(base)
    assert value['selected_method'] is None and value['reference_parameters'] is None
    assert value['status'] == 'needs_matrix_data'
    assert all(row['phase'] is None for row in value['batch']['lines'])


def test_exact_alias_duplicates_not_silently_merged():
    base = context()
    base['base_components'][0]['mass_percent'] -= 1
    base['base_components'].append({'name': 'water', 'mass_percent': 1, 'role': 'water'})
    value = plan(base)
    assert value['status'] == 'process_conflict'
    assert 'duplicate_component_identity' in codes(value)


def test_cold_method_and_mixer_conflicts():
    value = plan(method='hot', mixer_kind='manual')
    assert value['status'] == 'process_conflict'
    assert {'process_method_conflict', 'mixer_conflict'} <= codes(value)
    assert value['process_constraints_satisfied'] is False


def test_electrolyte_guard_survives_unknown_matrix():
    base = context()
    base['base_components'][0]['mass_percent'] -= 1
    base['base_components'].append({'name': 'Sodium Lactate', 'mass_percent': 1, 'role': 'humectant'})
    value = plan(base)
    assert value['selected_method'] is None
    assert 'simulgel_electrolyte_conflict' in codes(value)
    assert value['status'] == 'process_conflict'


def test_component_ph_range_never_becomes_finished_product_approval():
    within = plan(measured_ph=5.5)
    assert 'pe9010_ph_outside_supplier_range' not in codes(within)
    assert 'preservation_not_proven' in codes(within)
    assert within['process_constraints_satisfied'] is None
    assert not within['manufacturing_approved']
    outside = plan(measured_ph=2, peak_temperature_c=90)
    assert {'pe9010_ph_outside_supplier_range', 'pe9010_temperature_outside_guidance'} <= codes(outside)
    base = context()
    base['base_components'][-1]['name'] = 'Other preservative'
    unknown = plan(base, measured_ph=2, peak_temperature_c=90)
    assert not any(code.startswith('pe9010') for code in codes(unknown))


def test_hot_reference_does_not_invent_fragrance_cooldown_temperature():
    base = {'emulsion_type': 'oil_in_water', 'fragrance_concentration_percent': .5, 'base_components': []}
    for key, percent in workflow._HOT_SOURCE_PERCENT.items():
        base['base_components'].append({'name': workflow._ALIASES[key][0],
            'mass_percent': percent / 99.5 * 100, 'role': workflow._HOT_ROLES[key]})
    value = plan(base, method='hot', peak_temperature_c=72, batch_mass_g=500)
    assert value['selected_method'] == 'hot'
    assert value['reference_parameters']['phase_temperature_range_c'] == [70, 75]
    assert value['reference_parameters']['fragrance_addition_temperature_c'] is None
    assert 'fragrance_addition_condition_missing' in codes(value)
    assert value['batch']['lines'][-1]['phase'] == 'C'
    assert 'temperature_outside_reference' in codes(plan(base, peak_temperature_c=90))


@pytest.mark.parametrize('message,product', [('향수 조향 방법 알려줘', 'perfume'),
    ('로션 제조 과정 알려줘', 'body_lotion'), ('How to formulate perfume?', 'perfume')])
def test_procedure_questions_are_grounded_without_llm_or_parser(message, product):
    def forbidden(*args):
        raise AssertionError('procedure answer must not call LLM')
    value = assistant_reply(AssistantRequest(message=message), None, forbidden)
    assert value['source'] == 'sourced_formulation_knowledge'
    assert value['llm_calls_maximum'] == 0 and not value['formula_generated']
    assert value['formulation_workflow']['product_type'] == product and value['sources']


@pytest.mark.parametrize('message', ['달지 않은 우디 향수로 만들어줘', '조향 방법을 적용해서 달지 않은 우디 향수 만들어줘'])
def test_normal_odor_request_keeps_exact_single_language_call(message):
    calls = []
    def backend(text):
        calls.append(text)
        return {'desired': ['woody'], 'avoided': ['gourmand'], 'product': 'perfume', 'clarification': 'none'}
    value = assistant_reply(AssistantRequest(message=message), None, backend)
    assert len(calls) == 1 and value['source'] == 'quantized_language_model'
    assert value['intent_proposal']['avoided'] == ['gourmand']


def test_new_api_and_existing_prepare_keep_generation_separate():
    from tests.test_ai_extensions import Formula
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    from fragrance_ai.recommender.catalog import IngredientCatalog
    app, calls = FastAPI(), []
    def forbidden(*args, **kwargs):
        raise AssertionError('no inference expected')
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), forbidden, lambda: calls.append('rate'))
    with TestClient(app) as client:
        response = client.post('/v1/formulation-workflows/plan', json={'product_type': 'body_lotion',
            'application_context': context(), 'process': {'batch_mass_g': 400}})
        assert response.status_code == 200 and response.json() == plan(batch_mass_g=400)
        prepared = client.post('/v1/briefs/prepare', json={'formula': {'brief': 'woody scent'}}).json()
        assert prepared['status'] == 'ready' and prepared['formulation_workflow']['product_type'] == 'perfume'
        candle = client.post('/v1/briefs/prepare', json={'formula': {'brief': 'woody scent', 'product_category': 'candle'}}).json()
        assert candle['formulation_workflow'] is None  # No perfume procedure for a candle.
        assert len(calls) == 3
        assert client.post('/v1/formulation-workflows/plan', json={'product_type': 'body_lotion', 'process': {'batch_mass_g': True}}).status_code == 422
        assert client.get('/v1/ai/capabilities').json()['formulation_knowledge']['source_count'] == 6


@pytest.mark.parametrize('proposal', [
    {'desired': [], 'avoided': [], 'product': 'unspecified', 'clarification': 'scent'},
    {'desired': ['woody', 'gourmand'], 'avoided': [], 'product': 'perfume', 'clarification': 'none'},
    {'desired': ['woody'], 'avoided': ['gourmand'], 'product': 'body_lotion', 'clarification': 'none'}])
def test_explicit_odor_constraints_recover_from_valid_but_wrong_language_output(proposal):
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.catalog import IngredientCatalog
    calls = []
    def backend(text):
        calls.append(text)
        return proposal
    value = assistant_reply(AssistantRequest(message='달지 않은 우디 향수'),
        NaturalLanguageBriefParser(IngredientCatalog.load_builtin()), backend)
    assert len(calls) == 1
    assert value['source'] == 'deterministic_grounding_after_model_mismatch'
    assert value['intent_proposal']['desired'] == ['woody']
    assert value['intent_proposal']['avoided'] == ['gourmand']
    assert value['intent_proposal']['product'] == 'perfume'
    assert value['grounding']['original_model_proposal']['desired'] == proposal['desired']
    assert not value['formula_generated'] and value['requires_confirmation']


def test_lotion_conflict_stops_before_optimization(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('expensive optimizer must not be entered')
    monkeypatch.setattr('fragrance_ai.recommender.lotion_estimation._estimate_lotion_recipe', forbidden)
    with pytest.raises(ValueError, match='process_method_conflict'):
        estimate_lotion_recipe(LotionEstimateRequest(brief='woody scent', process={'method': 'hot'}), None,
                               use_configured_perception=False)


def test_multiple_product_mentions_do_not_override_correct_language_result():
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.catalog import IngredientCatalog
    value = assistant_reply(AssistantRequest(message='로션 말고 우디 향수'),
        NaturalLanguageBriefParser(IngredientCatalog.load_builtin()),
        lambda _: {'desired': ['woody'], 'avoided': [], 'product': 'perfume', 'clarification': 'none'})
    assert value['intent_proposal']['product'] == 'perfume'
    assert value['source'] == 'quantized_language_model'


def test_selected_oil_and_dose_are_bound_to_workflow_not_baseline(monkeypatch):
    def estimate(request, *a, **k):
        dose = request.fragrance_concentration_percent
        return {'status': 'research_candidate_only', 'score': 81 if dose == .25 else 80,
            'profile_target_met': False, 'solver_calls': 1, 'perception_model': {},
            'estimation': {'application_context': context(20, dose)}, 'recipe': [], 'closest_candidate': []}
    monkeypatch.setattr('fragrance_ai.recommender.lotion_estimation._estimate_lotion_recipe', estimate)
    request = LotionEstimateRequest(brief='woody scent', dose_trials={'concentrations_percent': [.25]}, process={'batch_mass_g': 400})
    result = estimate_lotion_recipe(request, None, use_configured_perception=False)
    assert result['score'] == 81 and not result['profile_target_met'] and result['recipe'] == []
    batch = result['formulation_workflow']['batch']
    assert batch['lines'][2]['mass_g'] == pytest.approx(400 * .2 * .9975)
    assert batch['lines'][-1]['mass_g'] == 1
    assert request.fragrance_concentration_percent == .5


def test_instruction_curriculum_is_not_measured_or_blind_test_data():
    rows = list(curriculum())
    assert len(rows) == 10 and len({row['id'] for row in rows}) == 10
    assert all(not row['numeric_sensory_training_eligible'] and not row['holdout_evaluation_eligible'] for row in rows)
    assert all(row['knowledge_sha256'] == workflow.knowledge_contract()['sha256'] for row in rows)
    assert next(row for row in rows if row['id'] == 'wrong_hot_method')['output']['status'] == 'process_conflict'
