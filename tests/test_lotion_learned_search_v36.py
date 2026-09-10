from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.recommender.perception_guidance import PerceptionSearchSession
from tests.test_lotion_v21 import fixture
from tests.test_perception_guidance import provider


def setup(monkeypatch):
    value, catalog = fixture()
    catalog = IngredientCatalog([replace(i, profile={'citrus': 1.}) for i in catalog.ingredients])
    value['brief'] = 'citrus scent'
    p = provider({i.ingredient_id: ('CCO', None) for i in catalog.ingredients})
    good = catalog.ingredients[1].ingredient_id
    calls = []
    def predict(self, item, doses):
        calls.append(item.ingredient_id)
        matrix = np.zeros((3, len(p.endpoints)))
        matrix[:, p.endpoints.index('Citrus' if item.ingredient_id == good else 'Woody')] = 1.
        return matrix, [{'basis': 'synthetic_test'}]*3
    monkeypatch.setattr(PerceptionSearchSession, '_predict', predict)
    return LotionOptimizationRequest.model_validate(value), catalog, p, calls


def test_learned_model_actually_changes_weights_without_quality_or_cost_loss(monkeypatch):
    request, catalog, p, calls = setup(monkeypatch)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    r = result['learned_optimization']
    assert r['recipe_changed'] and r['selected_affinity'] > r['baseline_affinity']+50
    assert r['fresh_transport_verified']
    assert np.all(np.asarray(r['selected_reference_affinities'])+1e-7 >= np.asarray(r['baseline_reference_affinities']))
    assert result['recipe'] != baseline['recipe']
    assert result['score']+1e-8 >= baseline['score']
    assert result['estimated_concentrate_cost_per_kg'] <= baseline['estimated_concentrate_cost_per_kg']+1e-8
    assert all(a['score']+1e-8 >= b['score'] for a,b in zip(result['timepoint_assessments'], baseline['timepoint_assessments']))
    assert result['perception_model']['recipe_weights_modified']
    assert len(calls) == len(set(calls)) == 2  # search and final reporting reuse
    assert result['transport_simulation_calls'] == baseline['transport_simulation_calls']
    assert result['human_similarity_percent'] is None


def test_more_expensive_learned_favorite_cannot_break_user_budget(monkeypatch):
    request, catalog, p, _ = setup(monkeypatch)
    catalog = IngredientCatalog([catalog.ingredients[0], replace(catalog.ingredients[1], price_per_kg=100.)])
    request.max_formula_cost_per_kg = 60.
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['estimated_concentrate_cost_per_kg'] <= 60.+1e-8
    assert result['learned_optimization']['cost_change_per_kg'] == pytest.approx(
        result['estimated_concentrate_cost_per_kg']-baseline['estimated_concentrate_cost_per_kg'])
    assert result['score']+1e-8 >= baseline['score']


def test_missing_molecule_is_penalized_not_removed_from_pool(monkeypatch):
    request, catalog, p, _ = setup(monkeypatch)
    p.structures.pop(catalog.ingredients[0].ingredient_id)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    r = result['learned_optimization']
    assert r['candidate_count'] == 2
    assert r['unmapped_candidate_ids'] == [catalog.ingredients[0].ingredient_id]
    assert r['baseline_affinity'] < 0
    assert r['recipe_changed']


def test_unknown_axes_do_not_fake_learned_objective(monkeypatch):
    request, catalog, p, calls = setup(monkeypatch)
    request.brief = 'clean scent'
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['learned_optimization']['status'] == 'unsupported_target_axes'
    assert 'clean' in result['learned_optimization']['unsupported_target_axes']
    assert result['learned_optimization']['solver_calls'] == 0


def test_solver_timeout_keeps_incumbent(monkeypatch):
    from fragrance_ai.recommender import lotion_learned_search
    request, catalog, p, _ = setup(monkeypatch)
    monkeypatch.setattr(lotion_learned_search, 'linprog', lambda *a, **kw: SimpleNamespace(status=1))
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['recipe'] == baseline['recipe']
    assert result['learned_optimization']['solver_incomplete']
    assert not result['learned_optimization']['recipe_changed']


def test_natural_language_design_retains_model_across_preparation_cache_reset(monkeypatch):
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    _, catalog, p, calls = setup(monkeypatch)
    result = estimate_lotion_recipe(LotionEstimateRequest(brief='citrus scent'), catalog, perception_guidance=p)
    assert result['learned_optimization']['enabled']
    assert result['perception_model']['optimization_guidance_evaluated']
    assert len(calls) == len(set(calls)) == 2


def test_fresh_transport_learned_regression_rolls_back_weights(monkeypatch):
    from fragrance_ai.recommender import lotion_learned_search
    request, catalog, p, _ = setup(monkeypatch)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    monkeypatch.setattr(lotion_learned_search, 'fresh_curve_affinity',
        lambda results, rows, *a: np.full((3, len(rows)), -1.))
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['recipe'] == baseline['recipe']
    assert result['learned_optimization']['status'] == 'fresh_transport_guard_rejected'
    assert not result['learned_optimization']['recipe_changed']
    assert result['learned_optimization']['cost_change_per_kg'] == 0


def test_learned_favorite_respects_material_cap(monkeypatch):
    request, catalog, p, _ = setup(monkeypatch)
    catalog = IngredientCatalog([catalog.ingredients[0], replace(catalog.ingredients[1], max_concentrate_percent=50.)])
    result = optimize_lotion(request, catalog, perception_guidance=p)
    good_weight = next(row['concentrate_percent'] for row in result['recipe'] if row['ingredient_id'] == catalog.ingredients[1].ingredient_id)
    assert good_weight <= 50.+1e-8
    assert result['learned_optimization']['recipe_changed']
