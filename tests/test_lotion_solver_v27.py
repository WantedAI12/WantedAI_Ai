"""Exact reformulation and numerical recovery, without relaxing scoring."""
from types import SimpleNamespace
import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender import lotion_optimizer as module
from tests.test_lotion_v21 import fixture


@pytest.mark.parametrize('active', [1, 2, 7, 19])
def test_compact_residual_is_exact_full_19_axis_l1(active):
    rng = np.random.default_rng(2700+active)
    profiles = rng.dirichlet(np.ones(19), 31)
    targets = np.zeros((6, 19))
    for target in targets:
        indices = rng.choice(19, active, replace=False)
        target[indices] = rng.dirichlet(np.ones(active))
    responses = np.exp(rng.normal(size=(6, 31)))
    weights = rng.dirichlet(np.ones(31))
    rows, sums, outside = module._compact_profile_residuals(profiles, targets, responses)
    times, axes = np.nonzero(targets > 0)
    unnormalized = (responses * weights) @ profiles
    residual = unnormalized - targets * (responses @ weights)[:, None]
    slack = np.abs(residual[times, axes])
    np.testing.assert_allclose(sums @ slack + outside @ weights, np.abs(residual).sum(axis=1), atol=1e-13)
    assert np.max(rows @ np.r_[weights, slack]) < 1e-13
    assert len(slack) == 6*active


@pytest.mark.parametrize('failure', [1, 4])
def test_numerical_failure_retries_same_problem_with_other_algorithm(monkeypatch, failure):
    value, catalog = fixture()
    value['search_goal'] = 'reach_target'
    original = module.linprog
    arguments = []
    def solve(*args, **kwargs):
        arguments.append(kwargs)
        return SimpleNamespace(status=failure) if len(arguments) == 1 else original(*args, **kwargs)
    monkeypatch.setattr(module, 'linprog', solve)
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result['profile_target_met'] and result['score'] >= 95
    assert result['solver_recovery_calls'] == 1 and not result['search_incomplete']
    assert arguments[1]['method'] == 'highs-ipm'
    assert arguments[0]['options']['small_matrix_value'] == 1e-12
    assert arguments[1]['options']['small_matrix_value'] == 1e-12
    for name in ('A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds'):
        assert arguments[0][name] is arguments[1][name]


def test_unresolved_recovery_is_not_reported_as_infeasibility(monkeypatch):
    value, catalog = fixture()
    # Isolate unresolved LP recovery. V44's independent accord search can
    # legitimately find a passing recipe even when every LP call fails.
    monkeypatch.setattr('fragrance_ai.recommender.accord_trials.accord_trials', lambda *a, **kw: ())
    monkeypatch.setattr(module, 'linprog', lambda *a, **kw: SimpleNamespace(status=4))
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result['search_incomplete'] and not result['profile_target_met']
    assert not result['attainability']['requested_target_excluded_by_numeric_bound']


@pytest.mark.parametrize('threshold_ratio', [1e8, 1e9, 1e10])
def test_small_response_columns_remain_available_to_optimizer(threshold_ratio):
    value, catalog = fixture()
    value['search_goal'] = 'reach_target'
    value['simulation']['materials'][0]['odor_threshold_mg_m3'] = 1/threshold_ratio
    value['simulation']['materials'][1]['odor_threshold_mg_m3'] = 1.
    value['simulation']['profile_weighting'] = 'odor_activity'
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result['profile_target_met'] and result['score'] >= 95
    assert len(result['recipe']) == 2
