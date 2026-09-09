from dataclasses import replace
import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender import lotion_optimizer as module, lotion_basis_cache
from fragrance_ai.recommender.lotion_numerics import response_variable_scales
from fragrance_ai.recommender.runtime_cache import InferenceCache
from tests.test_lotion_v21 import fixture


def test_exact_target_baseline_is_not_lost_between_bisection_steps():
    value, catalog = fixture()
    value.update(brief='citrus scent', search_goal='reach_target')
    catalog = type(catalog)([replace(catalog.ingredients[0], max_concentrate_percent=95., price_per_kg=100.),
        replace(catalog.ingredients[1], price_per_kg=1.)])
    r = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert r['profile_target_met'] and r['score'] == pytest.approx(95., abs=1e-8)
    assert r['solver_calls'] == 2 and r['attainability']['numeric_overlap_lower_percent'] == 95.


def test_extreme_threshold_ratio_recovers_valid_recipe_without_false_lower_bound():
    value, catalog = fixture()
    value['search_goal'] = 'reach_target'
    value['simulation']['profile_weighting'] = 'odor_activity'
    value['simulation']['materials'][0]['odor_threshold_mg_m3'] = 1e-12
    value['simulation']['materials'][1]['odor_threshold_mg_m3'] = 1.
    r = module.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert r['profile_target_met'] and r['score'] >= 95.
    assert r['solver_conditioning_calls'] >= 1
    assert all(a['overlap_score']+1e-7 >= r['attainability']['numeric_overlap_lower_percent'] for a in r['timepoint_assessments'])
    assert r['recipe'][0]['concentrate_percent'] < 1e-8


def test_coordinate_scales_keep_zero_columns_and_bound_inverse():
    scales = response_variable_scales(np.array([[1., 1e-100, 0.], [.2,1e-90,0.]]), 4)
    assert scales[2] == 1. and np.all(scales >= 1e-12) and np.all(scales <= 1.)


def fresh_cache(monkeypatch):
    monkeypatch.setattr(lotion_basis_cache, '_CACHE', InferenceCache(max_entries=8,max_bytes=8*1024*1024))


def test_basis_cache_reuses_physics_but_recalculates_final_recipe(monkeypatch):
    fresh_cache(monkeypatch)
    value, catalog = fixture()
    request = LotionOptimizationRequest.model_validate(value)
    original, calls = module.simulate_lotion, []
    def simulate(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'simulate_lotion', simulate)
    a = module.optimize_lotion(request,catalog,reuse_transport_basis=True)
    assert len(calls) == 2 and a['basis_cache_status'] == 'miss'
    b = module.optimize_lotion(request,catalog,reuse_transport_basis=True)
    assert len(calls) == 3 and b['transport_simulation_calls'] == 1 and b['reused_basis_simulations'] == 1
    assert a['formula_id'] == b['formula_id'] and a['recipe'] == b['recipe'] and a['score'] == b['score']
    b['recipe'][0]['concentrate_percent'] = -1
    c = module.optimize_lotion(request,catalog,reuse_transport_basis=True)
    assert c['recipe'] == a['recipe']


def test_changed_intent_reuses_physics_not_score(monkeypatch):
    fresh_cache(monkeypatch)
    value, catalog = fixture()
    a = module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    value['brief'] = 'woody scent'
    b = module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    assert b['basis_cache_status'] == 'hit' and a['recipe'] != b['recipe']
    assert b['score'] >= 95.


def test_coefficient_and_profile_changes_invalidate_physics_cache(monkeypatch):
    fresh_cache(monkeypatch)
    value, catalog = fixture()
    module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    value['simulation']['materials'][0]['gas_transfer_cm_min'] *= 2
    b = module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    assert b['basis_cache_status'] == 'miss'
    catalog = type(catalog)([replace(catalog.ingredients[0],profile={'citrus':.9,'floral':.1}),catalog.ingredients[1]])
    c = module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    assert c['basis_cache_status'] == 'miss'


def test_failed_basis_is_not_cached(monkeypatch):
    fresh_cache(monkeypatch)
    value, catalog = fixture()
    original = module.simulate_lotion
    def fail(*a, **kw): raise ValueError('synthetic failed transport')
    monkeypatch.setattr(module,'simulate_lotion',fail)
    with pytest.raises(ValueError):
        module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    monkeypatch.setattr(module,'simulate_lotion',original)
    r = module.optimize_lotion(LotionOptimizationRequest.model_validate(value),catalog,reuse_transport_basis=True)
    assert r['basis_cache_status'] == 'miss'
