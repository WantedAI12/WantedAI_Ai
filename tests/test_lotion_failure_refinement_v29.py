import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace

from fragrance_ai.recommender import lotion_coverage, lotion_estimation
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture
from tests.test_lotion_base_design_v28 import result


def test_overlap_dual_finds_joint_limit_missed_by_coordinate_maxima():
    # Coordinate maxima incorrectly suggest 100% of a half/half target.
    profiles = np.array([[.6,0,.4], [0,.6,.4]])
    assert lotion_coverage.profile_overlap_upper(profiles, [.5,.5,0]) == pytest.approx(60., abs=2e-6)


def test_dual_bound_contains_all_random_convex_mixture_scores():
    rng = np.random.default_rng(29)
    matrix = rng.dirichlet(np.ones(19), 50)
    target = rng.dirichlet(np.ones(19))
    upper = lotion_coverage.profile_overlap_upper(matrix, target)
    mixtures = rng.dirichlet(np.ones(50), 200) @ matrix
    assert np.all(100*np.minimum(mixtures, target).sum(axis=1) <= upper)


def test_failed_numerical_bound_never_excludes(monkeypatch):
    monkeypatch.setattr(lotion_coverage, 'linprog', lambda *a, **k: SimpleNamespace(x=None))
    assert lotion_coverage.profile_overlap_upper([[.2,.8]], [1.,0]) == 100.


def test_bad_solver_dual_is_clipped_and_recomputed_not_trusted(monkeypatch):
    monkeypatch.setattr(lotion_coverage, 'linprog', lambda *a, **k: SimpleNamespace(x=np.array([-4.,8.,-999.])))
    upper = lotion_coverage.profile_overlap_upper([[.2,.8]], [1.,0])
    assert upper >= 20.


def test_refinement_improves_without_lowering_approval_threshold():
    value, catalog = fixture()
    catalog = type(catalog)([replace(i, profile={'citrus':.94, 'woody':.06}) for i in catalog.ingredients])
    value['brief'] = 'citrus scent'
    r = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, target_only=True, improve_score_from=90.)
    assert r['score'] == pytest.approx(94.)
    assert not r['profile_target_met'] and not r['recipe'] and r['closest_candidate']
    assert r['preparation']['effective_target'] == 95.
    assert r['solver_calls'] == 6


def test_foreign_base_incumbent_is_not_reported_as_feasible_lower_bound():
    value, catalog = fixture()
    catalog = type(catalog)([replace(i, profile={'citrus':.5, 'woody':.5}) for i in catalog.ingredients])
    value['brief'] = 'citrus scent'
    r = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, target_only=True, improve_score_from=94.)
    assert r['attainability']['numeric_overlap_lower_percent'] <= r['score'] + 1e-6


def test_proven_coverage_limit_skips_variants_without_changing_baseline(monkeypatch):
    baseline = result(80.)
    baseline['preparation'] = {'evaluation_targets':[{'target_profile':{'citrus':1.}, 'phase':'opening', 'minutes':15.}]}
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return baseline
    monkeypatch.setattr(lotion_estimation, '_estimate_single_base', run)
    monkeypatch.setattr(lotion_estimation, 'lotion_profile_coverage', lambda *a: {'target_excluded':True})
    r = lotion_estimation.estimate_lotion_recipe(lotion_estimation.LotionEstimateRequest(brief='citrus scent', base_design={}), None)
    assert len(calls) == 1 and r['score'] == 80.
    assert r['base_design']['additional_variants_skipped'] == 'fixed_profile_upper_below_target'
