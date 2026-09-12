"""Solver/inference regression examples, not measured sensory accuracy."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

from fragrance_ai.recommender.lotion_reference_search import column_linprog
from tests import test_odor_expression_v61 as odor_fixtures

tiny_model = odor_fixtures.tiny_model


def diagnostics():
    return dict(linear_solves=0, maximum_working_materials=0,
                full_pool_pricing_passes=0, material_columns_priced=0,
                expansion_budget_reached=False)


@pytest.mark.parametrize('failure', ['cosine', 'background', 'indistinguishable'])
def test_reference_search_optimizes_every_final_shape_condition(monkeypatch, failure):
    from tests.test_lotion_v21 import fixture
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.lotion_reference_objective import ObservedReferenceBank
    from fragrance_ai.recommender.lotion_optimizer import _optimize_lotion_transport
    from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
    value, catalog = fixture()
    catalog = IngredientCatalog([replace(catalog.ingredients[0], price_per_kg=10.),
                                 replace(catalog.ingredients[1], price_per_kg=100.)])
    if failure == 'cosine':
        target = np.r_[np.ones(140)/140, np.zeros(6)]
        bad = .96*target
        bad[-1] = .04
        good = .96*target
        good[-6:] = .04/6
        background = np.eye(146)[-1]
    else:
        target = np.array([.5, .2, .15, .1, .05])
        background = target.copy()
        background[0] -= .01
        background[1] += .01
        bad, good = background.copy(), target.copy()
        if failure == 'indistinguishable':
            background = target.copy()
    bank = ObservedReferenceBank.__new__(ObservedReferenceBank)
    bank.sha256, bank.parent_sha256 = 'b'*64, 'a'*64
    bank.endpoints = tuple('e'+str(i) for i in range(len(target)))
    bank.profiles = {'citrus': np.stack([target]*2)}
    bank.background = np.stack([background]*2)
    bank.assert_current = lambda: None
    predictor = SimpleNamespace(provider=SimpleNamespace(endpoints=bank.endpoints,
        component_model_sha256='a'*64), prefetch=lambda items: None,
        shape=lambda item: np.stack([bad if item.ingredient_id == 'fixture-citrus' else good]*2))
    value.update(brief='citrus scent', evaluation_mode='observed_reference')
    value['simulation']['materials'][0]['concentrate_percent'] = 99.9999
    value['simulation']['materials'][1]['concentrate_percent'] = .0001
    for material in value['simulation']['materials']:
        material['odor_threshold_mg_m3'] = 1e-8
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r, p: bank)
    monkeypatch.setattr('fragrance_ai.recommender.lotion_surrogate.attach_release_prediction', lambda actual, *a, **kw: actual)
    result = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog, _shape_predictor=predictor)
    if failure == 'indistinguishable':
        assert not result['profile_target_met'] and not result['recipe']
        assert result['search_incomplete']
        assert result['material_column_search']['maximum_separation_rounds'] <= 12
        return
    assert result['profile_target_met'], result['score']
    assert result['score'] >= 95
    assert all(row['cosine_score'] >= 95 for row in result['timepoint_assessments'])
    assert result['perceptual_evaluation']['background_discrimination_passed']
    assert sum(r['concentrate_percent'] for r in result['recipe']) == pytest.approx(100)
    assert not result['manufacturing_approved']


@pytest.mark.parametrize('n', [3830, 12000])
def test_feasibility_pricing_finds_unranked_industrial_material(n):
    matrix = sparse.csr_matrix(([-1.], ([0], [n-1])), shape=(1, n))
    args = dict(A_ub=matrix, b_ub=np.array([-.8]),
                A_eq=sparse.csr_matrix(np.ones((1, n))), b_eq=np.ones(1),
                bounds=[(0., 1.)]*n, method='highs')
    report = diagnostics()
    result = column_linprog(np.ones(n), material_count=n, initial_columns=[],
                            column_order=list(range(n)), diagnostics=report, **args)
    expected = linprog(np.ones(n), **args)
    assert expected.success and result.success
    assert result.x[-1] >= .8-1e-8
    assert result.fun == pytest.approx(expected.fun)
    assert report['phase_one_pricing_passes'] >= 1
    assert report['maximum_working_materials'] < 256
    assert report['material_columns_priced'] == n


def test_phase_one_artificial_slack_is_never_returned_as_recipe():
    n = 3830
    report = diagnostics()
    result = column_linprog(np.ones(n), material_count=n, initial_columns=[],
        column_order=list(range(n)), diagnostics=report,
        A_ub=sparse.csr_matrix(-np.ones((1, n))), b_ub=np.array([-2.]),
        A_eq=sparse.csr_matrix(np.ones((1, n))), b_eq=np.ones(1),
        bounds=[(0., 1.)]*n, method='highs')
    assert not result.success and result.x is None


def test_feasibility_pricing_preserves_auxiliary_variables_and_fixed_caps():
    n = 3830
    # The auxiliary is an existing real model variable, not Phase I's slack.
    inequality = sparse.csr_matrix(([-1., -1.], ([0, 0], [n-1, n])), shape=(1, n+1))
    equality = sparse.csr_matrix(np.r_[np.ones(n), 0.].reshape(1, -1))
    report = diagnostics()
    result = column_linprog(np.r_[np.ones(n), 0.], material_count=n, initial_columns=[],
        column_order=list(range(n)), diagnostics=report, A_ub=inequality, b_ub=np.array([-.8]),
        A_eq=equality, b_eq=np.ones(1), bounds=[(0., 1.)]*n+[(.1, .1)], method='highs')
    assert result.success and result.x.shape == (n+1,)
    assert result.x[-1] == pytest.approx(.1)
    assert result.x[n-1] >= .7-1e-8
    assert result.x[:n].sum() == pytest.approx(1.)


def test_material_batch_guards_once_at_each_boundary(tiny_model, monkeypatch):
    from fragrance_ai.recommender.catalog import IngredientCatalog
    seed = IngredientCatalog.load_builtin().ingredients[0]
    items = [replace(seed, ingredient_id=str(i), structure_smiles=f'fixture-{i}')
             for i in range(769)]
    monkeypatch.setattr('fragrance_ai.recommender.fine_odor_model.structure_features',
                        lambda graphs: np.zeros((len(graphs), 1040), np.float32))
    calls = []
    original = tiny_model.assert_current
    def guarded():
        calls.append(1)
        original()
    monkeypatch.setattr(tiny_model, 'assert_current', guarded)
    expected = tiny_model.predict_features(np.zeros((769, 1040), np.float32))
    actual, evidence = tiny_model.materials(items)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 2
    assert len(evidence) == 769
    actual[:] = 0
    np.testing.assert_array_equal(tiny_model.materials(items)[0], expected)


def test_changed_model_during_batch_does_not_publish_or_cache(tiny_model, monkeypatch):
    from fragrance_ai.recommender.catalog import IngredientCatalog
    seed = IngredientCatalog.load_builtin().ingredients[0]
    original = tiny_model.predict_features
    def changed(x):
        result = original(x)
        tiny_model.weights_path.write_bytes(b'changed during inference')
        return result
    monkeypatch.setattr(tiny_model, 'predict_features', changed)
    with pytest.raises(ValueError, match='changed'):
        tiny_model.materials([replace(seed, structure_smiles='CCO')])
    assert not tiny_model._cache


@pytest.mark.parametrize('drift', ['before', 'cached', 'during', 'model_before', 'model_cached'])
def test_stock_route_checks_catalog_around_computation_and_cache(monkeypatch, tmp_path, drift):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from tests.test_ai_extensions import Formula
    from tests.test_mixture_core_v48 import sdk_fixture
    from tests.test_perception_guidance import material
    from tests.test_stock_mixture_api_v49 import stock_body
    sdk, provider, *_ = sdk_fixture(monkeypatch, tmp_path)
    current = [True]
    def guard():
        if not current[0]:
            raise ValueError('catalog snapshot changed')
    original = sdk.predict_masses
    def compute(rows):
        result = original(rows)
        if drift == 'during':
            current[0] = False
        return result
    monkeypatch.setattr(sdk, 'predict_masses', compute)
    original_assert = sdk.assert_current
    def model_guard():
        original_assert()
        if drift.startswith('model_') and not current[0]:
            raise ValueError('model snapshot changed')
    monkeypatch.setattr(sdk, 'assert_current', model_guard)
    app = FastAPI()
    register_ai_extensions(app, Formula, IngredientCatalog([material('a'), material('b')]),
        lambda *a, **kw: pytest.fail('must not generate'), lambda: None,
        perception_guidance=provider, stock_mixture_predictor=sdk,
        runtime_guard=None if drift.startswith('model_') else guard)
    with TestClient(app) as client:
        if drift in ('cached', 'model_cached'):
            assert client.post('/v1/formulations/stock-mixture/predict', json=stock_body()).status_code == 200
        if drift != 'during':
            current[0] = False
        response = client.post('/v1/formulations/stock-mixture/predict', json=stock_body())
        assert response.status_code != 200
        assert 'snapshot changed' in response.text
