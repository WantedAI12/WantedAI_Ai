"""Recovery of observed numerical failures; no relaxed recipe acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender import lotion_optimizer as module
from tests.test_lotion_v21 import fixture


def test_unused_recovery_is_available_below_the_requested_target(monkeypatch):
    value, catalog = fixture()
    value['search_goal'] = 'maximize'
    original, calls = module.linprog, []
    def solve(*args, **kwargs):
        calls.append(kwargs['method'])
        # solve(0) succeeds; the next midpoint and its ordinary algorithm
        # fallback encounter the same numerical failure seen in real data.
        if len(calls) in (2, 3):
            return SimpleNamespace(status=4, success=False, x=None)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'linprog', solve)
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, _transport_only=True)
    assert result['solver_conditioned_mode']
    assert result['solver_conditioning_calls'] >= 1
    assert result['profile_target_met'] and result['score'] >= 95.
    assert not result['search_incomplete']
    assert result['preparation']['effective_target'] == 95.


def test_verified_scaling_is_reused_instead_of_exhausting_retry_quota():
    value, catalog = fixture()
    value['search_goal'] = 'maximize'
    value['simulation']['profile_weighting'] = 'odor_activity'
    value['simulation']['materials'][0]['odor_threshold_mg_m3'] = 1e-12
    value['simulation']['materials'][1]['odor_threshold_mg_m3'] = 1.
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, _transport_only=True)
    assert result['solver_conditioned_mode']
    assert result['solver_conditioning_calls'] > 2
    assert result['solver_recovery_calls'] <= 2
    assert not result['search_incomplete']
    assert result['score'] >= 95. and result['profile_target_met']
    assert all(row['overlap_score']+1e-7 >= result['attainability']['numeric_overlap_lower_percent']
               for row in result['timepoint_assessments'])


def test_invalid_scaled_success_never_promotes_the_mode_or_recipe(monkeypatch):
    value, catalog = fixture()
    value['search_goal'] = 'reach_target'
    monkeypatch.setattr('fragrance_ai.recommender.accord_trials.accord_trials', lambda *a, **kw: ())
    def invalid(objective, **kwargs):
        x = np.zeros(len(objective))
        x[0] = 1.  # Valid mass, but not the requested balanced scent.
        return SimpleNamespace(status=0, success=True, x=x)
    monkeypatch.setattr(module, 'linprog', invalid)
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, _transport_only=True)
    assert not result['profile_target_met'] and result['score'] < 95.
    assert not result['solver_conditioned_mode']
    assert result['search_incomplete']
    assert result['recipe'] == []


@pytest.mark.parametrize('failure', [0, 1, 4])
def test_independent_accord_recovery_still_requires_actual_passing_profile(monkeypatch, failure):
    value, catalog = fixture()
    def failed(objective, **kwargs):
        if failure:
            return SimpleNamespace(status=failure, success=False, x=None)
        x = np.zeros(len(objective))
        x[0] = 1.
        return SimpleNamespace(status=0, success=True, x=x)
    monkeypatch.setattr(module, 'linprog', failed)
    result = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, _transport_only=True)
    assert result['search_incomplete']  # Do not erase the unresolved LP status.
    assert not result['solver_conditioned_mode']
    assert result['accord_refinement']['accepted'] > 0
    assert result['profile_target_met'] and result['recipe']
    assert result['score'] >= 95. and result['preparation']['effective_target'] == 95.
    assert min(row['score'] for row in result['timepoint_assessments']) >= 95.
    assert not result['attainability']['requested_target_excluded_by_numeric_bound']
