"""Analytic scent-balance counterexamples, not measured lotion accuracy."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.recommender.lotion_perception import LotionShapePredictor
from fragrance_ai.recommender.lotion_profile_balance import prepare_balance, profile_values
from fragrance_ai.recommender.perception_guidance import PerceptionSearchSession
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture
from tests.test_perception_guidance import provider


def balanced_setup(monkeypatch):
    request, catalog = fixture()
    catalog = IngredientCatalog([replace(i, profile={'citrus': .5, 'woody': .5}) for i in catalog.ingredients])
    p = provider({i.ingredient_id: ('CCO', None) for i in catalog.ingredients})
    p.component_model_sha256 = 'synthetic-balanced-shape-test'
    def predict(self, item, doses):
        out = np.zeros((3, len(p.endpoints)))
        out[:, p.endpoints.index('Citrus' if item.ingredient_id.endswith('citrus') else 'Woody')] = 1.
        return out, [{'basis': 'synthetic_analytic_counterexample'}]*3
    monkeypatch.setattr(PerceptionSearchSession, '_predict', predict)
    return LotionOptimizationRequest.model_validate(request), catalog, p


def test_equal_affinity_does_not_justify_wrong_note_ratios(monkeypatch):
    request, catalog, p = balanced_setup(monkeypatch)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    report = result['learned_optimization']
    balance = report['profile_balance']
    assert report['baseline_affinity'] == pytest.approx(100.)
    assert report['selected_affinity'] == pytest.approx(100.)
    # The physical incumbent retains the fixture's 90/10 recipe: its overlap
    # with the requested 50/50 profile is 0.5 + 0.1, despite affinity=1.
    assert balance['baseline_score'] == pytest.approx(60.)
    assert balance['selected_score'] == pytest.approx(100., abs=1e-6)
    assert report['recipe_changed'] and report['fresh_transport_verified']
    assert balance['fresh_transport_verified']
    assert len(result['recipe']) == 2
    assert [row['concentrate_percent'] for row in result['recipe']] == pytest.approx([50., 50.], abs=1e-6)
    assert result['score'] == pytest.approx(100.)
    assert result['human_similarity_percent'] is None


def test_lotion_partition_changes_balance_recipe(monkeypatch):
    request, catalog, p = balanced_setup(monkeypatch)
    baseline = optimize_lotion(request, catalog, perception_guidance=p)
    request.simulation.materials[1].lipid_water_partition = 100.
    changed = optimize_lotion(request, catalog, perception_guidance=p)
    weights = {row['ingredient_id']: row['concentrate_percent'] for row in changed['recipe']}
    assert weights['fixture-wood'] > 70.
    assert changed['recipe'] != baseline['recipe']
    assert changed['learned_optimization']['profile_balance']['selected_score'] > 95.
    assert changed['learned_optimization']['fresh_transport_verified']


@pytest.mark.parametrize('guard', ['cost', 'cap'])
def test_balance_improvement_keeps_user_constraints(monkeypatch, guard):
    request, catalog, p = balanced_setup(monkeypatch)
    if guard == 'cost':
        catalog = IngredientCatalog([catalog.ingredients[0], replace(catalog.ingredients[1], price_per_kg=100.)])
        request.max_formula_cost_per_kg = 60.
    else:
        catalog = IngredientCatalog([catalog.ingredients[0], replace(catalog.ingredients[1], max_concentrate_percent=25.)])
    result = optimize_lotion(request, catalog, perception_guidance=p)
    report = result['learned_optimization']['profile_balance']
    assert report['recipe_changed'] and report['selected_score'] > report['baseline_score']
    assert result['estimated_concentrate_cost_per_kg'] <= request.max_formula_cost_per_kg+1e-7
    if guard == 'cap':
        assert next(r['concentrate_percent'] for r in result['recipe'] if r['ingredient_id']=='fixture-wood') <= 25.+1e-7


def test_unmapped_endpoint_mass_is_not_renormalized_away(monkeypatch):
    request, catalog, p = balanced_setup(monkeypatch)
    predictor = LotionShapePredictor(p)
    for i in catalog.ingredients:
        shape = np.zeros((3, len(p.endpoints)))
        shape[:, p.endpoints.index('Citrus')] = .25
        shape[:, p.endpoints.index('Woody')] = .25
        from fragrance_ai.recommender.lotion_evaluation import LOTION_PROJECTION
        mapped = {name for names in LOTION_PROJECTION.values() for name in names}
        unknown = next(name for name in p.endpoints if name not in mapped)
        shape[:, p.endpoints.index(unknown)] = .5
        predictor.shapes[i.ingredient_id] = shape
    shapes, targets, avoided = prepare_balance(predictor, catalog.ingredients,
        [{'target_profile': {'citrus': .5, 'woody': .5}, 'avoided': []}])
    score = profile_values(shapes, targets, avoided, np.ones((1, 2)), np.array([.5, .5]))
    assert score == pytest.approx(np.full((3, 1), .5))


def test_balance_timeout_preserves_prior_recipe(monkeypatch):
    from fragrance_ai.recommender import lotion_profile_balance
    request, catalog, p = balanced_setup(monkeypatch)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    monkeypatch.setattr(lotion_profile_balance, 'linprog', lambda *a, **kw: SimpleNamespace(status=1))
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['recipe'] == baseline['recipe']
    assert result['learned_optimization']['solver_incomplete']
    assert not result['learned_optimization']['recipe_changed']


def test_final_curves_can_reject_balanced_proposal(monkeypatch):
    from fragrance_ai.recommender import lotion_profile_balance
    request, catalog, p = balanced_setup(monkeypatch)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    monkeypatch.setattr(lotion_profile_balance, 'fresh_curve_profile_match',
                        lambda results, rows, *a: np.zeros((3, len(rows))))
    result = optimize_lotion(request, catalog, perception_guidance=p)
    assert result['recipe'] == baseline['recipe']
    assert result['learned_optimization']['status'] == 'fresh_transport_guard_rejected'
    assert not result['learned_optimization']['profile_balance']['recipe_changed']


def test_unknown_molecule_keeps_its_mass_and_conservative_avoidance(monkeypatch):
    _, catalog, p = balanced_setup(monkeypatch)
    predictor = LotionShapePredictor(p)
    predictor.shapes[catalog.ingredients[0].ingredient_id] = None
    rows = [{'target_profile': {'woody': 1.}, 'avoided': ['floral']}]
    shapes, targets, avoided = prepare_balance(predictor, catalog.ingredients, rows)
    values = profile_values(shapes, targets, avoided, np.ones((1, 2)), np.array([.4, .6]))
    assert values == pytest.approx(np.full((3, 1), .6))
    assert shapes[:, 0, -1] == pytest.approx(np.ones(3))


def test_unsupported_target_cannot_be_removed_by_projection(monkeypatch):
    _, catalog, p = balanced_setup(monkeypatch)
    with pytest.raises(ValueError, match='unsupported'):
        prepare_balance(LotionShapePredictor(p), catalog.ingredients,
                        [{'target_profile': {'clean': .9, 'woody': .1}, 'avoided': []}])
