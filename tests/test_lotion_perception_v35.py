"""Bridge contract tests use labelled synthetic shapes; live model tested separately."""
from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest, LotionSimulationRequest
from fragrance_ai.recommender.lotion import _simulate_lotion_transport, simulate_lotion
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.recommender.lotion_perception import attach_lotion_perception, REFERENCE_DILUTIONS
from fragrance_ai.recommender.perception_guidance import PerceptionSearchSession
from tests.test_ai_extensions import Formula
from tests.test_lotion_v21 import fixture
from tests.test_perception_guidance import provider


@pytest.fixture
def model(monkeypatch):
    value, catalog = fixture()
    p = provider({item.ingredient_id: ('CCO', None) for item in catalog.ingredients})
    p.component_model_sha256 = 'synthetic-checkpoint-not-real'
    calls = []
    def predict(self, item, doses):
        calls.append((item.ingredient_id, doses.tolist()))
        result = np.zeros((3, len(p.endpoints)))
        name = 'Citrus' if item.ingredient_id.endswith('citrus') else 'Woody'
        # Different absolute magnitudes must disappear at shape normalization.
        result[:, p.endpoints.index(name)] = (2 if name == 'Citrus' else 200)
        return result, [{'basis': 'synthetic_unit_fixture'} for _ in doses]
    monkeypatch.setattr(PerceptionSearchSession, '_predict', predict)
    return value, catalog, p, calls


def test_transport_weighted_shape_not_stock_dose_or_double_weight(model):
    value, catalog, p, calls = model
    request = LotionSimulationRequest.model_validate(value['simulation'])
    raw = _simulate_lotion_transport(request, catalog)
    before = deepcopy(raw)
    result = attach_lotion_perception(raw, catalog, p)
    assert raw == before
    assert result['diagnostics'] == raw['diagnostics']
    assert result['human_similarity_percent'] is None
    assert result['temporal_profile'][0]['learned_perception']['status'] == 'no_airborne_material'
    for point in result['temporal_profile'][1:]:
        learned = point['learned_perception']
        masses = [r['odor_activity_proxy'] for r in point['materials']]
        assert learned['predicted_endpoint_fraction']['Citrus'] == pytest.approx(masses[0]/sum(masses))
        assert sum(learned['predicted_endpoint_fraction'].values()) == pytest.approx(1.)
    assert len(calls) == 2
    assert all(doses == list(REFERENCE_DILUTIONS) for _, doses in calls)
    contract = result['perception_model']
    assert contract['applied'] and not contract['optimization_guidance_applied']
    assert not contract['gas_to_stock_conversion_used'] and not contract['full_model_application']


def test_missing_active_graph_abstains_not_renormalized_to_perfect_match(model):
    value, catalog, p, _ = model
    p.structures.pop(catalog.ingredients[1].ingredient_id)
    result = simulate_lotion(LotionSimulationRequest.model_validate(value['simulation']), catalog, perception_guidance=p)
    row = result['temporal_profile'][-1]['learned_perception']
    assert row['status'] == 'abstained_incomplete_component_coverage'
    assert row['predicted_endpoint_fraction'] is None
    assert 0 < row['covered_transport_weight_percent'] < 100
    assert result['perception_model']['status'] == 'research_transport_shape_partial'
    assert sum(row['supported_endpoint_contribution'].values()) == pytest.approx(row['covered_transport_weight_percent']/100.)
    assert row['unknown_transport_weight_fraction'] > 0
    assert row['endpoint_fraction_bounds']['Woody']['upper'] >= row['unknown_transport_weight_fraction']


def test_scenario_cache_does_not_repeat_molecular_inference(model):
    value, catalog, p, calls = model
    raw = _simulate_lotion_transport(LotionSimulationRequest.model_validate(value['simulation']), catalog)
    r = attach_lotion_perception({'simulation': raw, 'scenario_simulations': [deepcopy(raw), deepcopy(raw)]}, catalog, p)
    assert len(calls) == r['perception_model']['component_model_calls'] == 2
    assert r['perception_model']['evaluated_timepoint_count'] == 6


def test_optimizer_does_not_change_acceptance_or_infer_entire_lp_basis(model):
    value, catalog, p, calls = model
    request = LotionOptimizationRequest.model_validate(value)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    # The full-balance stage now refines the old near-50/50 discretized
    # solution to the exact supported target; the strict gate is unchanged.
    assert [row['concentrate_percent'] for row in result['recipe']] == pytest.approx([50., 50.], abs=1e-6)
    assert result['score'] == pytest.approx(100.)
    assert result['score'] >= baseline['score'] and result['profile_target_met'] == baseline['profile_target_met']
    assert result['learned_optimization']['fresh_transport_verified']
    assert result['transport_simulation_calls'] == baseline['transport_simulation_calls']
    assert len(calls) == len(result['recipe'])


@pytest.mark.parametrize('kind', ['nan', 'negative', 'zero'])
def test_invalid_component_outputs_fail_closed(model, monkeypatch, kind):
    value, catalog, p, _ = model
    matrix = np.full((3, len(p.endpoints)), {'nan': np.nan, 'negative': -1., 'zero': 0.}[kind])
    monkeypatch.setattr(PerceptionSearchSession, '_predict', lambda *a: (matrix, [{}]*3))
    request = LotionSimulationRequest.model_validate(value['simulation'])
    if kind == 'zero':
        result = simulate_lotion(request, catalog, perception_guidance=p)
        assert not result['perception_model']['applied']
        assert result['perception_model']['component_model_calls'] == 2
    else:
        with pytest.raises(ValueError, match='invalid learned'):
            simulate_lotion(request, catalog, perception_guidance=p)


def test_api_routes_preserve_checkpoint_and_cache_payload(model):
    value, catalog, p, calls = model
    app = FastAPI()
    register_ai_extensions(app, Formula, catalog, lambda request: {}, lambda: None, lotion_perception_guidance=p)
    with TestClient(app) as client:
        features = client.get('/v1/ai/capabilities').json()['features']
        assert features['body_lotion_learned_temporal_shape'] and features['body_lotion_learned_optimization']
        prepared = client.post('/v1/applications/body-lotion/prepare', json=value)
        assert prepared.status_code == 200
        assert prepared.json()['perception_model']['status'] == 'not_evaluated_preparation_only'
        assert not calls
        for endpoint, body in [('simulate', value['simulation']), ('optimize', value)]:
            a = client.post('/v1/applications/body-lotion/'+endpoint, json=body)
            assert a.status_code == 200, a.text
            assert a.json()['perception_model']['component_model_sha256'] == p.component_model_sha256
            count = len(calls)
            b = client.post('/v1/applications/body-lotion/'+endpoint, json=body)
            assert b.status_code == 200 and b.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            assert b.json() == a.json() and len(calls) == count


def test_legacy_no_configuration_returns_original_shape(model, monkeypatch):
    from fragrance_ai.recommender.perception_runtime import PATH_ENV, HASH_ENV
    monkeypatch.delenv(PATH_ENV, raising=False)
    monkeypatch.delenv(HASH_ENV, raising=False)
    value, catalog, _, calls = model
    request = LotionSimulationRequest.model_validate(value['simulation'])
    assert simulate_lotion(request, catalog) == _simulate_lotion_transport(request, catalog)
    assert not calls


def test_invalid_open_sink_does_not_receive_learned_prediction(model):
    value, catalog, p, calls = model
    value['simulation'].update(transport_mode='open_sink', air_exchange_per_min=.001)
    result = simulate_lotion(LotionSimulationRequest.model_validate(value['simulation']), catalog, perception_guidance=p)
    assert result['status'] == 'outside_open_sink_assumption'
    assert not result['perception_model']['applied'] and not calls
    assert all(row['learned_perception']['status'] == 'abstained_invalid_transport_assumption' for row in result['temporal_profile'])
