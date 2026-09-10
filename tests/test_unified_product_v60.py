"""Fixture-level contracts; real checkpoint/held-out checks have separate logs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.linalg import expm

from fragrance_ai.platform.unified_product_inputs import UnifiedProductContext, UnifiedProductRequest, context_id
from fragrance_ai.recommender.unified_transport import (
    VERSION, FEATURE_NAMES, STATE_NAMES, PRODUCTS, UnifiedTransportModel,
    baseline_kernel, constant_kernel, kernel_mask, trajectory)
from fragrance_ai.recommender.unified_product import UnifiedProductPredictor
from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION
from fragrance_ai.recommender.catalog import IngredientCatalog


@pytest.fixture
def model(tmp_path):
    # Zero-weight numerical fixture, explicitly not a trained-model benchmark.
    dims = [8, 4, 4, 4, 10]
    arrays = {'mean': np.zeros(8), 'scale': np.ones(8)}
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        arrays[f'weight_{i}'], arrays[f'bias_{i}'] = np.zeros((a, b)), np.zeros(b)
    np.savez(tmp_path/'weights.npz', **arrays)
    manifest = {'schema': VERSION, 'feature_names': list(FEATURE_NAMES), 'products': list(PRODUCTS),
        'state_names': list(STATE_NAMES), 'training_executed': True, 'accepted_for_local_inference': True,
        'label_kind': 'synthetic_transition_operators', 'architecture': {'dimensions': dims},
        'validation_selected_blend': 1., 'parent_models': {'atlas': {'sha256': 'a'*64}},
        'weights': {'path': 'weights.npz', 'sha256': hashlib.sha256((tmp_path/'weights.npz').read_bytes()).hexdigest()}}
    path = tmp_path/'model.json'; path.write_text(json.dumps(manifest))
    return UnifiedTransportModel(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('rates', [(0.,0.,0.,0.,0.), (0.,.2,.3,0.,0.), (.1,0.,0.,0.,.1),
    (0.,0.,0.,.1,.1), (.1,0.,0.,.1,0.), (1e5,1e-4,0.,.1,1.), (1.,2.,3.,4.,5.)])
def test_transition_matches_independent_exponential_including_zero_paths(rates):
    e,u,h,r,v = rates
    a = np.array([[-e-u-h,r,0,0,0], [e,-r-v,0,0,0], [u,0,0,0,0],
                  [0,v,0,0,0], [h,0,0,0,0]])
    expected = expm(a*.7)[:, :2].T
    actual = constant_kernel(*[np.array([r]) for r in rates], .7)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-8)
    np.testing.assert_allclose(actual.sum(axis=1), 1., atol=1e-14)


def test_zero_capacity_change_has_exact_analytic_identity(model):
    raw = np.array([[.3,.1,.01,.2,.4,0.,.8,.1], [.3,.1,.01,.2,.4,3.,0.,.5]])
    np.testing.assert_array_equal(model.kernel(raw), baseline_kernel(raw))
    np.testing.assert_array_equal(model.stable_kernel(raw), baseline_kernel(raw))


def test_operator_conservation_reachability_and_monotone_rollout(model):
    raw = np.array([[.3,0.,0.,.1,.2,2.,.6,.2], [.2,.1,.1,.2,0.,2.,.5,.2]])
    kernel = model.kernel(raw)
    assert np.all(kernel[~kernel_mask(raw)] == 0)
    y, _ = trajectory(raw[:, :6], raw[:, 6:], [0.,.1,.25,.5,1.], operator=model.stable_kernel)
    assert np.all(y >= 0) and np.all(np.diff(y[:, :, 2:], axis=0) >= 0)
    np.testing.assert_allclose(y.sum(axis=2), 1., atol=2e-14)


def test_display_sampling_does_not_change_trajectory(model):
    raw = np.array([[.3,.1,.01,.2,.4,2.,.8,.1]])
    a, _ = trajectory(raw[:, :6], raw[:, 6:], [0.,.1,.5,1.], operator=model.stable_kernel)
    b, _ = trajectory(raw[:, :6], raw[:, 6:], [0.,.07,.1,.2,.5,.71,1.], operator=model.stable_kernel)
    np.testing.assert_allclose(a, b[[0,2,4,6]], atol=2e-15, rtol=1e-12)


def test_model_rejects_domain_and_checkpoint_drift(model):
    with pytest.raises(ValueError, match='domain'):
        model.kernel(np.array([[1,1,1,1,1,1,.8,.8]]))
    model.weights_path.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='changed'):
        model.assert_current()


def make_predictor(model):
    from tests.test_lotion_v21 import fixture
    _, old_catalog = fixture()
    items = [replace(row, structure_smiles=graph) for row, graph in zip(old_catalog.ingredients, ('C','CC'))]
    catalog = IngredientCatalog(items)
    endpoints = tuple(sorted({s for values in ATLAS_PROJECTION.values() for s in values}))
    def predict(graphs, reference_level):
        p = np.ones((len(graphs), len(endpoints)))
        for i, graph in enumerate(graphs):
            p[i, endpoints.index('LEMON' if graph == 'C' else 'CEDARWOOD')] = 50.
        return {'applicability': p, 'use': p*.9}
    atlas = SimpleNamespace(sha256='a'*64, endpoints=endpoints, fine=None, native={}, predict=predict)
    provider = UnifiedProductPredictor(model, atlas,
        {row.ingredient_id:(graph,row.cas_number) for row, graph in zip(items, ('C','CC'))}, catalog)
    return provider


def request_payload(product='perfume'):
    context = UnifiedProductContext(product_type=product, application_mass_mg_cm2=2.,
        fragrance_concentration_percent=.5, temperature_c=25., relative_humidity_percent=50., headspace_height_cm=1.,
        stages=[{'stage_id':'initial', 'duration_minutes':1., 'formulation_reference':'synthetic unit-test carrier'},
                {'stage_id':'after', 'duration_minutes':2., 'formulation_reference':'synthetic unit-test second stage'}])
    ids = ['fixture-citrus','fixture-wood']
    components = [{'ingredient_id':key, 'concentrate_percent':50., 'odor_threshold_mg_m3':.1} for key in ids]
    coefficients = [{'ingredient_id':key, 'evaporation_per_min':e, 'uptake_per_min':.01,
        'hydrolysis_per_min':0., 'air_return_per_min':.1, 'ventilation_per_min':.2,
        'capacity_decay_per_min':.03, 'evaporating_capacity_fraction':.5, 'nonreactive_capacity_fraction':.2,
        'source_kind':'simulated', 'source_reference':'test fixture, not a measured product'}
        for key, e in zip(ids, (.3,.02))]
    stages = [{'stage_id':s.stage_id, 'coefficients':deepcopy(coefficients)} for s in context.stages]
    if product == 'body_wash':
        stages[0].update(rinse_retained_film_fractions=dict.fromkeys(ids, .1), rinse_source_reference='test rinse')
    return {'context':context.model_dump(), 'parameter_context_id':context_id(context), 'components':components,
            'stages':stages, 'times_minutes':[0.,.1,1.,2.,3.]}


@pytest.mark.parametrize('product', PRODUCTS)
def test_shared_model_all_product_paths_and_mass(model, product):
    predictor = make_predictor(model)
    result = predictor.predict(UnifiedProductRequest(**request_payload(product)))
    assert result['status'] == 'research_prediction'
    assert result['model']['checkpoint_sha256'] == model.sha256
    assert result['diagnostics']['cumulative_sinks_monotone']
    assert result['diagnostics']['mass_balance_max_abs_error_mg_cm2'] < 1e-14
    assert result['human_similarity_percent'] is None
    assert result['process_events'] == [] if product != 'body_wash' else result['process_events'][0]['washed_off_mg_cm2'] > 0


def test_rinse_changes_later_airborne_scent_not_just_a_label(model):
    predictor = make_predictor(model)
    before = predictor.predict(UnifiedProductRequest(**request_payload('body_lotion')))
    after = predictor.predict(UnifiedProductRequest(**request_payload('body_wash')))
    assert after['temporal_profile'][-1]['total_air_concentration_mg_m3'] < before['temporal_profile'][-1]['total_air_concentration_mg_m3']
    a, b = before['temporal_profile'][2]['materials'][0], after['temporal_profile'][2]['materials'][0]
    assert b['remaining_mg_cm2'] == pytest.approx(a['remaining_mg_cm2']*.1)
    assert b['headspace_mg_cm2'] == pytest.approx(a['headspace_mg_cm2'])


@pytest.mark.parametrize('failure', ['stale_context','missing_material','wash_without_rinse','perfume_with_rinse','boolean_retention'])
def test_no_cross_product_or_incomplete_process_fallback(failure):
    p = request_payload('body_wash' if failure in ('perfume_with_rinse','boolean_retention') else 'perfume')
    if failure == 'stale_context':
        p['context']['temperature_c'] = 30.
    elif failure == 'missing_material':
        p['stages'][1]['coefficients'].pop()
    elif failure == 'wash_without_rinse':
        p['context']['product_type'] = 'body_wash'
        p['parameter_context_id'] = context_id(UnifiedProductContext(**p['context']))
    elif failure == 'perfume_with_rinse':
        p['context']['product_type'] = 'perfume'
        p['parameter_context_id'] = context_id(UnifiedProductContext(**p['context']))
    else:
        p['stages'][0]['rinse_retained_film_fractions']['fixture-citrus'] = True
    with pytest.raises(ValueError):
        UnifiedProductRequest(**p)


def test_missing_graph_is_not_filled_with_zero_profile(model):
    predictor = make_predictor(model)
    predictor.known['fixture-citrus'] = replace(predictor.known['fixture-citrus'], structure_smiles='CCC')
    result = predictor.predict(UnifiedProductRequest(**request_payload()))
    assert result['status'] == 'transport_only_missing_odor_profiles'
    assert result['missing_odor_profile_ids'] == ['fixture-citrus']
    assert result['temporal_profile'][-1]['full_odor_reference_profiles'] is None


def test_api_uses_one_model_cache_without_breaking_existing_capabilities(model):
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    from tests.test_ai_extensions import Formula
    predictor = make_predictor(model)
    app = FastAPI()
    register_ai_extensions(app, Formula, predictor.catalog, lambda *a,**k:None, lambda:None,
                           unified_product_predictor=predictor)
    with TestClient(app) as client:
        p = request_payload('body_wash')
        prepared = client.post('/v1/applications/unified/context', json=p['context'])
        assert prepared.status_code == 200
        first = client.post('/v1/applications/unified/predict', json=p)
        second = client.post('/v1/applications/unified/predict', json=p)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert second.headers['X-Perfumery-Unified-Cache'] == 'hit'
        caps = client.get('/v1/ai/capabilities').json()
        assert caps['features']['body_wash_supplied_dilution_rinse_trajectory']
        assert caps['features']['intent_preparation']
