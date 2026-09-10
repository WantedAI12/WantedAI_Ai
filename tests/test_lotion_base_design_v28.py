from copy import deepcopy
from dataclasses import replace

import pytest

from fragrance_ai.platform.formulation_inputs import LotionBaseDesignOptions
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender import lotion_estimation as module
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from tests.test_lotion_v21 import fixture


@pytest.mark.parametrize('values', [[4], [31], [10], [20,20], [True], [float('nan')], [], [5,15,20,30]])
def test_invalid_base_design_options(values):
    with pytest.raises(ValueError):
        LotionBaseDesignOptions(additional_oil_base_percents=values)


def test_variant_changes_context_and_volumes_not_request_or_reference():
    catalog = IngredientCatalog.load_builtin()
    request = module.LotionEstimateRequest(brief='citrus woody scent')
    original = request.model_dump()
    fixed, _, _ = module.build_estimated_lotion_inputs(request, catalog)
    changed, scenarios, evidence = module.build_estimated_lotion_inputs(request, catalog, oil_base_percent=30)
    base = {c.role:c.mass_percent for c in changed.simulation.application_context.base_components}
    assert base == {'water':64, 'oil':30, 'humectant':3, 'emulsifier':2, 'preservative':1}
    assert changed.simulation.parameter_context_id != fixed.simulation.parameter_context_id
    assert all(s.parameter_context_id == changed.simulation.parameter_context_id for s in scenarios)
    assert not evidence['base_variant']['stability_verified']
    assert request.model_dump() == original


def test_target_only_does_not_spend_maximization_budget_on_failed_variant():
    value, catalog = fixture()
    value['brief'] = 'rose scent'
    result = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, target_only=True)
    assert result['solver_calls'] == 2 and not result['profile_target_met']
    assert result['recipe'] == []


def test_target_only_retries_exact_95_without_requiring_clearance():
    value, catalog = fixture()
    # Force a search: the supplied 90/10 baseline violates the 70% cap, while
    # a 50/50 feasible mixture still reaches exactly 95.1 rather than 95.2.
    catalog = IngredientCatalog([replace(i, profile={'citrus':.951, 'woody':.049}, max_concentrate_percent=70.) for i in catalog.ingredients])
    value['brief'] = 'citrus scent'
    r = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog, target_only=True)
    assert r['profile_target_met'] and r['score'] == pytest.approx(95.1)
    assert r['solver_calls'] == 2


def result(score):
    return {'status':'research_profile_target_met' if score >= 95 else 'research_candidate_only',
        'score':score, 'profile_target_met':score >= 95, 'solver_calls':1,
        'estimation':{'application_context':{'fragrance_concentration_percent':.5,
            'base_components':[{'name':'water','role':'water','mass_percent':90},
                               {'name':'oil','role':'oil','mass_percent':10}]}}}


def test_existing_pass_returns_unchanged_without_extra_optimization(monkeypatch):
    fixed = result(98)
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(fixed)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody', base_design={}), None)
    assert len(calls) == 1 and r['score'] == 98
    assert r['base_design']['selected_oil_base_percent'] == 10
    assert not r['base_design']['reference_formula_modified']
    assert sum(c['finished_product_percent'] for c in r['base_design']['finished_product_base']) == pytest.approx(99.5)


def test_variant_search_never_replaces_baseline_with_worse_score(monkeypatch):
    def run(*args, **kwargs):
        return result(90 if kwargs.get('oil_base_percent',10) == 10 else 80)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody', base_design={}), None)
    assert r['score'] == 90 and r['base_design']['selected_oil_base_percent'] == 10
    assert len(r['base_design']['evaluated_variants']) == 6
    assert all(10 < oil < 30 for oil in r['base_design']['adaptive_oil_trials'])


def test_first_passing_variant_is_selected_and_remaining_not_run(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        oil = kwargs.get('oil_base_percent',10)
        calls.append(oil)
        return result(94 if oil == 10 else 96)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody', base_design={}), None)
    assert calls == [10,20] and r['profile_target_met']
    assert r['base_design']['reference_formula_modified']
    assert r['base_design']['fixed_reference_score'] == 94


def test_bad_variant_preserves_baseline_and_reports_failure(monkeypatch):
    def run(*args, **kwargs):
        if 'oil_base_percent' in kwargs:
            raise ValueError('synthetic variant failure')
        return result(90)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody', base_design={}), None)
    assert r['score'] == 90 and len(r['base_design']['variant_errors']) == 2
