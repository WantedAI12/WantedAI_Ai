from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.lotion_optimizer import _optimize_lotion_transport
from fragrance_ai.recommender.lotion_reference_objective import ObservedReferenceBank, VERSION, normalize
from fragrance_ai.recommender.models import RecipeConstraints
from tests.test_lotion_v21 import fixture


@pytest.fixture
def bank(tmp_path):
    endpoints = ['FRUITY,CITRUS', 'LEMON', 'WOODY, RESINOUS', 'SOAPY', 'MUSK']
    profiles = {'citrus': [[.5,.2,.15,.1,.05]]*2, 'lemon': [[.2,.55,.05,.15,.05]]*2,
        'woody': [[.1,.1,.6,.1,.1]]*2, 'clean': [[.05,.05,.1,.6,.2]]*2,
        'musky': [[.05,.05,.1,.2,.6]]*2}
    path = tmp_path/'references.json'
    path.write_text(json.dumps({'schema': VERSION, 'endpoints': endpoints, 'profiles': profiles,
        'background': [[.2]*5]*2, 'concept_metadata': {}, 'parent_atlas_sha256': 'a'*64,
        'recipe_outcomes_used': False}))
    return ObservedReferenceBank(path, hashlib.sha256(path.read_bytes()).hexdigest())


def query(bank, text):
    _, catalog = fixture()
    brief = NaturalLanguageBriefParser(catalog).parse(text, RecipeConstraints(product_category='body_lotion'))
    rows = [{'phase': 'heart', 'target_profile': brief.target_profile, 'avoided': brief.avoided_dimensions}]
    return brief, bank.targets(brief, rows)


def test_target_contains_observed_companion_notes(bank):
    _, (targets, unsupported) = query(bank, 'clean scent')
    assert not unsupported
    target = targets[0]
    assert target['profiles'][0, bank.endpoints.index('MUSK')] > 0
    actual = bank.compare(target, target['profiles'][0], 0)
    assert actual['score'] == pytest.approx(100)
    assert actual['reference_more_specific_than_background']


def test_wrong_and_generic_scents_are_not_approved(bank):
    _, (targets, _) = query(bank, 'clean scent')
    target = targets[0]
    wrong = bank.compare(target, bank.profiles['woody'][0], 0)
    average = bank.compare(target, bank.background[0], 0)
    assert wrong['score'] < 95
    assert not average['reference_more_specific_than_background']


def test_fine_lemon_is_not_collapsed_into_citrus(bank):
    _, (targets, unsupported) = query(bank, 'lemon scent')
    assert not unsupported
    assert targets[0]['concepts'] == {'lemon': 1.}
    np.testing.assert_allclose(targets[0]['profiles'], bank.profiles['lemon'])


def test_unavailable_semantics_are_retained(bank):
    _, (_, unsupported) = query(bank, 'aquatic clean scent')
    assert 'aquatic' in unsupported


def test_fine_terms_preserve_unequal_relative_weights(bank):
    brief, _ = query(bank, 'lemon cedar scent')
    bank.profiles['cedar'] = bank.profiles['woody'].copy()
    brief = replace(brief, target_profile={'citrus': .9, 'woody': .1})
    targets, unsupported = bank.targets(brief, [{'phase': 'heart', 'target_profile': brief.target_profile, 'avoided': []}])
    assert not unsupported
    assert targets[0]['concepts'] == pytest.approx({'lemon': .9, 'cedar': .1})


def test_two_details_sharing_one_family_do_not_depend_on_sort_order(bank):
    brief, _ = query(bank, 'lemon orange scent')
    bank.profiles['orange'] = bank.profiles['citrus'].copy()
    targets, unsupported = bank.targets(brief, [{'phase': 'heart', 'target_profile': {'citrus': 1.}, 'avoided': []}])
    assert not unsupported
    assert targets[0]['concepts'] == pytest.approx({'lemon': .5, 'orange': .5})


def test_explicit_prohibition_remains_separate(bank):
    _, (targets, unsupported) = query(bank, 'clean scent without musk')
    assert not unsupported
    assert 'MUSK' in targets[0]['avoided']
    assert targets[0]['profiles'][0, -1] == 0
    assert bank.compare(targets[0], [0,0,0,0,1], 0)['score'] == 0


def test_phase_details_do_not_leak(bank):
    _, catalog = fixture()
    brief = NaturalLanguageBriefParser(catalog).parse('opening lemon, drydown woody', RecipeConstraints())
    targets, unsupported = bank.targets(brief, [
        {'phase': phase, 'target_profile': brief.phase_target_profiles[phase], 'avoided': []}
        for phase in ('opening','drydown')])
    assert not unsupported
    assert 'lemon' in targets[0]['concepts'] and 'lemon' not in targets[1]['concepts']


def test_independent_target_and_artifact_drift(bank):
    _, (a, _) = query(bank, 'clean scent')
    _, (b, _) = query(bank, 'clean scent')
    np.testing.assert_array_equal(a[0]['profiles'], b[0]['profiles'])
    bank.path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        bank.assert_current()


@pytest.mark.parametrize('x', [[0,0], [-1,2], [float('nan'),1], [float('inf'),0]])
def test_invalid_reference_is_never_normalized_to_success(x):
    with pytest.raises(ValueError):
        normalize(x)


def fixture_predictor(bank, catalog, *, missing=False):
    values = {i.ingredient_id: bank.profiles['citrus' if j == 0 else 'woody']
              for j,i in enumerate(catalog.ingredients)}
    return SimpleNamespace(provider=SimpleNamespace(endpoints=bank.endpoints, component_model_sha256='a'*64),
        prefetch=lambda items: None,
        shape=lambda item: None if missing else values[item.ingredient_id])


def test_primary_search_uses_finished_mixture_and_preserves_legacy_score(bank, monkeypatch):
    value, catalog = fixture()
    value['brief'] = 'citrus scent'
    value['evaluation_mode'] = 'observed_reference'
    for m in value['simulation']['materials']:
        m['odor_threshold_mg_m3'] = 1e-8
    request = LotionOptimizationRequest.model_validate(value)
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r,p: bank)
    result = _optimize_lotion_transport(request, catalog, _shape_predictor=fixture_predictor(bank,catalog))
    assert result['profile_target_met']
    assert result['score'] >= 95
    assert result['perceptual_evaluation']['full_endpoint_count'] == 5
    assert result['legacy_evaluation']['used_for_recipe_selection'] is False
    assert sum(r['concentrate_percent'] for r in result['recipe']) == pytest.approx(100)
    assert result['score'] == min(r['score'] for r in result['timepoint_assessments'])


def test_tiny_airborne_mass_cannot_pass_by_shape_only(bank, monkeypatch):
    value, catalog = fixture()
    value.update(brief='citrus scent', evaluation_mode='observed_reference')
    for m in value['simulation']['materials']:
        m['odor_threshold_mg_m3'] = 1e12
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r,p: bank)
    result = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog,
        _shape_predictor=fixture_predictor(bank,catalog))
    assert not result['profile_target_met']
    assert not result['recipe']


def test_missing_full_profiles_never_receive_favorable_zero_vectors(bank, monkeypatch):
    value, catalog = fixture()
    value.update(brief='citrus scent', evaluation_mode='observed_reference')
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r,p: bank)
    result = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog,
        _shape_predictor=fixture_predictor(bank,catalog,missing=True))
    assert not result['profile_target_met'] and not result['recipe']


def test_full_pool_pricing_can_add_a_material_outside_initial_working_set():
    from scipy import sparse
    from scipy.optimize import linprog
    from fragrance_ai.recommender.lotion_reference_search import column_linprog
    n = 180
    objective = np.zeros(n)
    objective[-1] = -1
    args = dict(A_ub=sparse.csr_matrix(np.ones((1,n))),b_ub=np.ones(1),
        A_eq=sparse.csr_matrix(np.ones((1,n))),b_eq=np.ones(1),bounds=[(0,1)]*n,method='highs')
    report = dict(linear_solves=0,maximum_working_materials=0,full_pool_pricing_passes=0,
                  material_columns_priced=0,expansion_budget_reached=False)
    result = column_linprog(objective,material_count=n,initial_columns=[],column_order=list(range(n)),
                            diagnostics=report,**args)
    expected = linprog(objective,**args)
    assert result.success and expected.success
    assert result.x[-1] == pytest.approx(1)
    assert objective@result.x == pytest.approx(expected.fun)
    assert report['material_columns_priced'] == n and report['linear_solves'] >= 2


def test_restricted_infeasibility_does_not_ban_other_materials():
    from scipy import sparse
    from fragrance_ai.recommender.lotion_reference_search import column_linprog
    n = 180
    inequality = np.zeros((1,n))
    inequality[0,-1] = -1
    report = dict(linear_solves=0,maximum_working_materials=0,full_pool_pricing_passes=0,
                  material_columns_priced=0,expansion_budget_reached=False)
    result = column_linprog(np.ones(n),material_count=n,initial_columns=[],column_order=list(range(n)),
        diagnostics=report,A_ub=sparse.csr_matrix(inequality),b_ub=np.array([-.8]),
        A_eq=sparse.csr_matrix(np.ones((1,n))),b_eq=np.ones(1),bounds=[(0,1)]*n,method='highs')
    assert result.success and result.x[-1] >= .8-1e-8


def test_unscoped_brief_does_not_require_an_unchanged_8_hour_scent(bank):
    from fragrance_ai.recommender.lotion_reference_objective import exposure_groups
    brief, _ = query(bank,'clean scent')
    rows = [{'minutes':t,'phase':p} for t,p in ((15,'opening'),(60,'heart'),(240,'heart'),(480,'drydown'))]
    groups = exposure_groups(brief,rows)
    assert len(groups) == 1 and groups[0]['phase'] == 'overall'
    assert groups[0]['weights']@np.array([100,60,0,0]) >= 1
    assert groups[0]['weights']@np.array([0,0,0,0]) == 0


def test_explicit_drydown_is_not_hidden_by_a_strong_opening(bank):
    from fragrance_ai.recommender.lotion_reference_objective import exposure_groups
    brief,_ = query(bank,'opening clean, drydown woody')
    rows = [{'minutes':t,'phase':p} for t,p in ((15,'opening'),(60,'heart'),(240,'heart'),(480,'drydown'))]
    groups = exposure_groups(brief,rows)
    assert [g['phase'] for g in groups] == ['opening','drydown']
    assert groups[0]['weights']@np.array([100,60,0,0]) > 1
    assert groups[1]['weights']@np.array([100,60,0,0]) == 0


def test_whole_exposure_does_not_cherry_pick_its_best_time(bank):
    from fragrance_ai.recommender.lotion_reference_objective import exposure_groups
    brief,(target,_) = query(bank,'clean scent')
    rows = [{'minutes':t,'phase':p} for t,p in ((15,'opening'),(60,'heart'),(240,'heart'),(480,'drydown'))]
    weights = exposure_groups(brief,rows)[0]['weights']
    clean = weights@np.array([100,0,0,0])
    wrong = weights@np.array([0,0,100,100])
    mixture = (clean*bank.profiles['clean'][0]+wrong*bank.profiles['woody'][0])/(clean+wrong)
    assert bank.compare(target[0],mixture,0)['score'] < 95
