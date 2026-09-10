from dataclasses import replace
from types import SimpleNamespace

import pytest

from fragrance_ai.recommender.lotion_incumbent import incumbent_weights
from fragrance_ai.recommender import lotion_optimizer as module
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture


@pytest.mark.parametrize('lines,status', [
    ([{'ingredient_id': 'unknown', 'concentrate_percent': 100}], 'ingredient_not_in_current_pool'),
    ([{'ingredient_id': 'fixture-citrus', 'concentrate_percent': True}], 'invalid_concentration'),
    ([{'ingredient_id': 'fixture-citrus', 'concentrate_percent': float('nan')}], 'invalid_concentration'),
    ([{'ingredient_id': 'fixture-citrus', 'concentrate_percent': 90}], 'concentrations_do_not_sum_to_100'),
    ([{'ingredient_id': 'fixture-citrus', 'concentrate_percent': 50}]*2, 'duplicate_ingredient'),
])
def test_invalid_incumbent_is_not_repaired_into_a_different_formula(lines, status):
    _, catalog = fixture()
    weights, actual = incumbent_weights(lines, catalog.ingredients)
    assert weights is None and actual == status


def test_timeout_keeps_freshly_verified_seed_not_uniform_baseline(monkeypatch):
    data, catalog = fixture()
    data['brief'] = 'woody scent'
    # Test incumbent retention independently of V44's successful accord trials.
    monkeypatch.setattr('fragrance_ai.recommender.accord_trials.accord_trials', lambda *a, **kw: ())
    monkeypatch.setattr(module, 'linprog', lambda *a, **kw: SimpleNamespace(status=1, success=False, x=None))
    request = LotionOptimizationRequest.model_validate(data)
    seed = [{'ingredient_id': 'fixture-wood', 'concentrate_percent': 100.}]
    before = module.optimize_lotion(request, catalog, target_only=True, _transport_only=True)
    after = module.optimize_lotion(request, catalog, target_only=True, _transport_only=True, _incumbent_recipe=seed)
    assert after['score'] > before['score']
    assert after['score'] == pytest.approx(100.) and after['profile_target_met']
    assert after['incumbent_reuse']['used']
    assert after['incumbent_reuse']['old_score_reused'] is False
    assert after['transport_simulation_calls'] == before['transport_simulation_calls']


def test_current_cap_rejects_seed_even_if_previous_recipe_was_accepted(monkeypatch):
    data, catalog = fixture()
    data['brief'] = 'woody scent'
    catalog = IngredientCatalog([catalog.ingredients[0], replace(catalog.ingredients[1], max_concentrate_percent=20.)])
    monkeypatch.setattr(module, 'linprog', lambda *a, **kw: SimpleNamespace(status=1, success=False, x=None))
    r = module.optimize_lotion(LotionOptimizationRequest.model_validate(data), catalog, target_only=True,
        _transport_only=True, _incumbent_recipe=[{'ingredient_id': 'fixture-wood', 'concentrate_percent': 100.}])
    assert r['incumbent_reuse']['status'] == 'rejected_by_current_constraints'
    assert not r['incumbent_reuse']['used'] and not r['profile_target_met']


def test_base_variants_receive_current_best_formula(monkeypatch):
    from fragrance_ai.recommender import lotion_estimation as estimation
    from tests.test_lotion_base_design_v28 import result
    calls = []
    def run(*a, **kw):
        calls.append(kw.get('_incumbent_recipe'))
        r = result(90+len(calls)/10)
        r['closest_candidate'] = [{'ingredient_id': str(len(calls)), 'concentrate_percent': 100.}]
        return r
    monkeypatch.setattr(estimation, '_estimate_single_base', run)
    estimation.estimate_lotion_recipe(estimation.LotionEstimateRequest(brief='clean woody',
        base_design={'adaptive_oil_refinement_steps': 0}), None)
    assert calls == [None, [{'ingredient_id':'1', 'concentrate_percent':100.}],
                     [{'ingredient_id':'2', 'concentrate_percent':100.}]]
