from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

from fragrance_ai.recommender.lotion_evaluation import refinement_score_floors, LOTION_PROJECTION, compare_lotion_profiles
from fragrance_ai.recommender.lotion_learned_search import refine_lotion_weights, target_coefficients
from fragrance_ai.recommender.lotion_optimizer import _compact_profile_residuals
from fragrance_ai.recommender.models import profile_vector
from tests.test_lotion_v21 import fixture


@pytest.mark.parametrize('scores,target,expected', [
    ([94.,99.],95.,[94.,95.]), ([96.,99.],95.,[96.,96.]),
    ([94.,94.5,99.],95.,[94.,94.5,95.]),
    ([95.2,99.],97.,[95.2,97.]), ([95.2,99.],100.,[95.2,99.]),
    ([100.,100.],95.,[100.,100.])])
def test_only_surplus_changes_never_target_or_worst_or_failed_point(scores,target,expected):
    np.testing.assert_allclose(refinement_score_floors(scores,target),expected,atol=1e-10)


@pytest.mark.parametrize('scores,target', [([],95),([float('nan')],95),([101],95),([90],89.9),([90],True)])
def test_invalid_quality_contract_rejected(scores,target):
    with pytest.raises(ValueError):
        refinement_score_floors(scores,target)


def test_guard_invariants_over_deterministic_random_profiles():
    rng = np.random.default_rng(46)
    for _ in range(100):
        scores = rng.uniform(0,100,20)
        target = rng.uniform(95,100)
        floors = refinement_score_floors(scores,target)
        assert floors.min() == scores.min()
        np.testing.assert_array_equal(floors[scores < target],scores[scores < target])
        assert np.all(floors[scores >= target] >= target)
        assert np.all(floors <= scores)


def analytic_refinement(target):
    _,catalog = fixture()
    pool = [replace(catalog.ingredients[0],profile={'citrus':.5,'woody':.5}),
            replace(catalog.ingredients[1],profile={'citrus':.6,'woody':.4})]
    rows = [{'minutes':15.,'phase':'opening','target_profile':{'floral':.6,'citrus':.2,'woody':.2},'avoided':[]},
            {'minutes':60.,'phase':'heart','target_profile':{'citrus':.5,'woody':.5},'avoided':[]}]
    profiles = np.array([i.vector() for i in pool])
    targets = np.array([profile_vector(row['target_profile']) for row in rows])
    responses = np.ones((2,2))
    residual,sums,outside = _compact_profile_residuals(profiles,targets,responses)
    ns = sums.shape[1]
    endpoints = tuple(dict.fromkeys(name for names in LOTION_PROJECTION.values() for name in names))
    bad,good = np.zeros(len(endpoints)),np.zeros(len(endpoints))
    bad[endpoints.index('Earthy')] = 1.
    for name,weight in [('Floral',.6),('Citrus',.2),('Woody',.2)]:
        good[endpoints.index(name)] = weight
    shapes = {pool[0].ingredient_id:np.tile(bad,(3,1)),pool[1].ingredient_id:np.tile(good,(3,1))}
    predictor = SimpleNamespace(provider=SimpleNamespace(endpoints=endpoints),prefetch=lambda items:None,
                                shape=lambda item:shapes[item.ingredient_id])
    def assess(weights):
        checks = [compare_lotion_profiles(row['target_profile'],weights@profiles).to_dict() for row in rows]
        return min(row['score'] for row in checks),checks
    def valid(weights):
        return bool(np.isfinite(weights).all() and np.all(weights >= -1e-10)
                    and np.all(weights <= 1+1e-10) and abs(weights.sum()-1) <= 1e-8)
    baseline = np.array([.8,.2])
    best,report = refine_lotion_weights(predictor=predictor,pool=pool,responses=responses,target_rows=rows,
        incumbent=baseline,residual_rows=residual,slack_sums=sums,outside=outside,avoid_vectors=np.zeros_like(targets),
        profiles=profiles,fixed_rows=sparse.csr_matrix((0,2+ns)),fixed_rhs=np.array([]),
        a_eq=sparse.csr_matrix([np.r_[np.ones(2),np.zeros(ns)]]),eq_rhs=np.array([1.]),
        bounds=[(0.,1.)]*2+[(0.,None)]*ns,prices=np.ones(2),valid=valid,assessments=assess,target_score=target)
    return best,report,assess(baseline),assess(best)


def test_real_solver_uses_surplus_without_losing_any_passed_point():
    best,report,before,after = analytic_refinement(95.)
    assert before[0] == pytest.approx(40.) and after[0] == pytest.approx(40.)
    assert before[1][1]['score'] == pytest.approx(98.)
    assert after[1][1]['score'] >= 95.-1e-8
    assert best[1] == pytest.approx(.5,abs=1e-6)
    assert report['selected_affinity'] > report['baseline_affinity']+10.
    assert report['recipe_changed'] and not report['acceptance_threshold_modified']
    assert report['required_strict_scores'] == pytest.approx([40.,95.])


def test_stricter_user_target_does_not_surrender_previously_failed_score():
    best,report,before,after = analytic_refinement(99.)
    assert before[0] == pytest.approx(after[0])
    assert after[1][1]['score'] >= before[1][1]['score']-1e-8
    assert best == pytest.approx([.8,.2],abs=1e-6)
    assert report['required_strict_scores'] == pytest.approx([40.,98.])


def test_partial_target_keeps_unsupported_mass_instead_of_rescaling_to_full_credit():
    rows = [{'target_profile':{'clean':.9,'woody':.1},'avoided':[]}]
    coefficients = target_coefficients(rows,('Woody','Floral'))
    np.testing.assert_array_equal(coefficients,[[.1,0.]])
    assert float(coefficients.max()) == .1
    with pytest.raises(ValueError,match='avoidance'):
        target_coefficients([{'target_profile':{'woody':1.},'avoided':['clean']}],('Woody',))


def test_partial_target_reaches_actual_recipe_search_and_preserves_all_strict_axes(monkeypatch):
    from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
    from tests.test_lotion_learned_search_v36 import setup
    request,catalog,p,_ = setup(monkeypatch)
    request.brief = 'clean citrus scent'
    baseline = optimize_lotion(request,catalog,_transport_only=True)
    result = optimize_lotion(request,catalog,perception_guidance=p)
    report = result['learned_optimization']
    assert report['partial_target_guidance'] and not report['all_target_axes_modeled']
    assert report['unsupported_target_axes'] == ['clean']
    assert report['solver_calls'] > 0 and report['recipe_changed']
    assert report['fresh_transport_verified']
    assert result['score']+1e-8 >= baseline['score']
    assert result['profile_target_met'] == baseline['profile_target_met'] == False
    assert result['recipe'] == []  # partial guidance does not turn failure into pass
    assert result['preparation']['intent'] == baseline['preparation']['intent']
    assert result['perception_model']['partial_target_guidance']


def test_fractional_iterations_cannot_accumulate_the_score_tolerance(monkeypatch):
    from fragrance_ai.recommender import lotion_learned_search as search
    # A deterministic proposal advances learned affinity but loses 3e-8
    # native points over the whole segment. Per-step-only guards incorrectly
    # accept several individually sub-tolerance steps with a larger total loss.
    def solve(solver, objective, **kwargs):
        return SimpleNamespace(status=0, success=True, x=np.r_[0.,1.,np.zeros(len(objective)-2)])
    monkeypatch.setattr(search, 'conditioned_linprog', solve)
    best, report = search.fractional_refinement(best=np.array([1.,0.]),
        learned=lambda w: np.full((3,1), w[1]), affinity=np.tile([0.,1.],(3,1,1)),
        responses=np.ones((1,2)), fixed=sparse.csr_matrix((0,2)), rhs=np.array([]),
        affinity_rows=lambda levels: sparse.csr_matrix((3,2)), residual_rows=sparse.csr_matrix((0,2)),
        slack_sums=sparse.csr_matrix((1,0)), outside=np.zeros((1,2)),
        avoid_vectors=np.zeros((1,1)), profiles=np.ones((2,1)),
        a_eq=sparse.csr_matrix([[1.,1.]]), eq_rhs=np.array([1.]), bounds=[(0.,1.)]*2,
        prices=np.ones(2), valid=lambda w: True,
        assessments=lambda w: (90.-3e-8*w[1],[{'score':90.-3e-8*w[1]}]))
    assert report['accepted_steps'] >= 2
    assert .25 <= best[1] <= 1./3.+1e-7
    assert 90.-3e-8*best[1]+1e-8 >= 90.
