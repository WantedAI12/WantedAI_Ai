from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.corrective_profile_search import corrective_profile_search, profile_derivatives
from fragrance_ai.recommender.formulation_guidance import SharedRecipeSession
from scripts.extend_reference_annotations_v71 import annotation_routes, fit_annotation_profiles


@pytest.mark.parametrize('smooth', [0., 1e-5])
@pytest.mark.parametrize('seed', [17, 102, 981])
def test_all_scenarios_and_heads_have_correct_analytic_mass_derivatives(seed, smooth):
    rng = np.random.default_rng(seed)
    shapes = rng.random((2, 8, 6))
    shapes /= shapes.sum(-1, keepdims=True)
    targets = rng.random((2, 3, 6))
    targets /= targets.sum(-1, keepdims=True)
    responses = rng.uniform(.01, 1., (3, 8))
    weights = rng.uniform(.1, .2, 8)
    weights /= weights.sum()
    avoided = np.zeros((3, 6))
    avoided[:, 2] = 1
    background = np.full((2, 6), 1/6)
    def function(x):
        return profile_derivatives(shapes, targets, responses, x, avoided, background, smooth_epsilon=smooth)
    values, jac, contrast, contrast_jac = function(weights)
    numerical, numerical_contrast = [], []
    for i in range(len(weights)):
        step = np.zeros_like(weights)
        step[i] = 1e-7
        above, below = function(weights+step), function(weights-step)
        numerical.append((above[0]-below[0])/2e-7)
        numerical_contrast.append((above[2]-below[2])/2e-7)
    np.testing.assert_allclose(jac, np.asarray(numerical).T, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(contrast_jac, np.asarray(numerical_contrast).T, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(jac@weights, 0., atol=1e-12)
    assert len(values) == 18 and len(contrast) == 6
    if smooth:
        exact = profile_derivatives(shapes, targets, responses, weights, avoided, background)[0]
        assert np.all(values <= exact+1e-14)


def simple_problem(n=300):
    shapes = np.zeros((2, n, 4))
    shapes[:, :n//2, 0] = 1.
    shapes[:, n//2:, 1] = 1.
    target = np.zeros((2, 3, 4))
    target[:, :, :2] = .5
    seed = np.zeros(n)
    seed[0] = 1.
    return dict(shapes=shapes, wanted=target, responses=np.ones((3, n)),
        fixed=np.ones((1, n)), rhs=np.array([1.]), equality=np.ones((1, n)),
        equality_rhs=np.array([1.]), bounds=[(0., 1.)]*n, initial=seed, target=.95)


def test_correction_prices_the_entire_pool_and_preserves_mass():
    result = corrective_profile_search(**simple_problem())
    assert result.success and result.candidate_score >= .95-1e-9
    assert result.full_pool_count == 300 and result.priced_columns >= 300
    assert result.x.sum() == pytest.approx(1.)
    assert .45-1e-8 <= result.x[:150].sum() <= .55+1e-8


def test_off_notes_and_infeasible_caps_are_not_converted_to_success():
    args = simple_problem()
    args['shapes'] *= .8
    args['shapes'][:, :, 3] = .2
    result = corrective_profile_search(**args)
    assert not result.success and result.candidate_score <= .8+1e-9
    assert result.candidate_score >= .4-1e-9
    args = simple_problem()
    args['bounds'][150:] = [(0., 0.)]*150
    result = corrective_profile_search(**args)
    assert not result.success and result.candidate_score <= .5+1e-9
    assert not np.any(result.candidate[150:])


def test_work_limit_preserves_verified_incumbent():
    args = simple_problem()
    result = corrective_profile_search(**args, maximum_seconds=1e-12)
    assert not result.success and result.status == 1
    np.testing.assert_array_equal(result.candidate, args['initial'])


@pytest.mark.parametrize('avoid', [[], [1]])
def test_batched_neural_replacements_equal_each_exact_evaluation(avoid):
    rng = np.random.default_rng(7154)
    items = [SimpleNamespace(ingredient_id=str(i), active_strength_percent=100./(i+1)) for i in range(6)]
    shapes = rng.uniform(.01, 1, (6, 2, 5))
    shapes /= shapes.sum(-1, keepdims=True)
    session = SharedRecipeSession.__new__(SharedRecipeSession)
    session.enabled, session.target = True, np.array([.7, .3, 0.])
    session.projection = np.zeros((5, 3))
    session.projection[:3, :3] = np.eye(3)
    session.avoided = avoid
    session.shapes = SimpleNamespace(prefetch=lambda rows: None, shape=lambda item: shapes[int(item.ingredient_id)])
    session.grid_calls, session.exact_calls = 0, 0
    session.provider = SimpleNamespace(core=SimpleNamespace(sha256='a'*64), endpoints=list('abcde'))
    original = {'0': 60., '1': 40.}
    batch = session.replacement_scores(original, items, '0', items[2:])
    expected = [session.evaluate([60., 40.], [item, items[1]])['score'] for item in items[2:]]
    np.testing.assert_allclose(batch, expected, atol=1e-12, rtol=1e-12)


def test_annotation_fit_uses_distinct_measured_identities_not_repeated_rows():
    rows = [{'graph': key, 'applicability': np.array([.7, .2, .1]), 'use': np.array([.6, .3, .1])}
            for key in ('a', 'a', 'b', 'c')]
    profiles, metadata = fit_annotation_profiles(rows, {'a':[0], 'b':[0], 'c':[0]}, {'amber':[0], 'unknown':[1]})
    np.testing.assert_allclose(profiles['amber'], [[.7, .2, .1], [.6, .3, .1]])
    assert metadata['amber']['distinct_identity_groups'] == 3
    assert 'unknown' not in profiles
    failed, _ = fit_annotation_profiles(rows[:3], {'a':[0], 'b':[0]}, {'amber':[0]})
    assert not failed


def test_annotation_language_mapping_excludes_quality_and_weak_secondary_routes():
    registry = {'rows': {
        'resin': {'kind':'odor', 'aliases':['resinous'], 'source_terms':['resin'], 'coarse_projection':{'amber':.7,'woody':.3}},
        'low': {'kind':'quality', 'aliases':['low'], 'source_terms':['low'], 'coarse_projection':{}},
    }}
    routes = annotation_routes(['resinous','low'], registry, ['amber','woody'])
    assert routes == {'resin':[0], 'amber':[0]}


def test_complete_prior_perfume_trajectory_precedes_global_recovery(monkeypatch):
    from fragrance_ai.recommender import dose_refinement as module
    calls = []
    def preserved(*args, **kwargs):
        calls.append('old-start')
        yield {'kind':'old-1'}
        yield {'kind':'old-2'}
        calls.append('old-complete')
    def new(*args, **kwargs):
        assert calls == ['old-start', 'old-complete']
        assert kwargs['neural_rank'] is True
        yield {'kind':'new'}
    monkeypatch.setattr(module, '_preserved_dose_refinement_proposals', preserved)
    monkeypatch.setattr(module, '_bounded_dose_refinement_proposals', new)
    monkeypatch.setattr(module, 'profile_upper_bound', lambda *a: {'upper_score':99.})
    monkeypatch.setattr(module, 'full_inferred_note_policy', lambda p: p)
    brief = SimpleNamespace(constraints=SimpleNamespace(target_similarity=95.), target_profile={'woody':1.})
    rows = list(module.dose_refinement_proposals([], brief, {}, [], {}, [], {},
        current_lines=lambda: [], current_score=lambda: 70.))
    assert [r['kind'] for r in rows] == ['old-1','old-2','new']
    assert rows[-1]['prior_search_trajectory_preserved']
