"""Algebra, bindings and API contracts; fixture weights are NOT trained evidence."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion import _simulate_lotion_transport
from fragrance_ai.recommender.lotion_surrogate import (FEATURE_NAMES, VERSION, LotionReleaseSurrogate,
    physical_baseline, phase_volumes, request_features, predict_release, attach_release_prediction)
from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
from tests.test_lotion import data, rebind


@pytest.fixture
def model(tmp_path):
    dims = [8, 2, 2, 2, 5]
    arrays = {'mean': np.zeros(8), 'scale': np.ones(8)}
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        arrays[f'weight_{i}'], arrays[f'bias_{i}'] = np.zeros((a, b)), np.zeros(b)
    path = tmp_path/'weights.npz'
    np.savez(path, **arrays)
    manifest = {'schema': VERSION, 'training_executed': True,
        'label_kind': 'synthetic_transport', 'measured_lotion_observations': 0,
        'fixture_only_not_training_evidence': True,
        'accepted_for_local_surrogate_inference': True, 'feature_names': list(FEATURE_NAMES),
        'weights': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'architecture': {'dimensions': dims}, 'parent_models': {'atlas': {'sha256': 'a'*64}}}
    meta = tmp_path/'model.json'
    meta.write_text(json.dumps(manifest), encoding='utf-8')
    return LotionReleaseSurrogate(meta, sha256=hashlib.sha256(meta.read_bytes()).hexdigest())


def request():
    value = data()
    value.update(transport_mode='bidirectional_air', air_exchange_per_min=1.,
                 water_loss_per_min=.015, times_minutes=[0., 15., 60., 120.])
    return LotionSimulationRequest.model_validate(value)


def test_constant_capacity_baseline_and_teacher_agree():
    from scripts.train_lotion_surrogate_v58 import teacher
    raw = np.array([[.2, .3, .1, .1, 2., 0., .5, .2],
                    [30., 0., 0., 3., 1., 10., 0., .5]])
    assert physical_baseline(raw) == pytest.approx(teacher(raw, 64), abs=1e-13)


def test_small_drying_is_continuous():
    raw = np.array([[.2, .3, .1, .1, 2., 0., .5, .2]])
    constant = physical_baseline(raw)
    for tiny in (1e-12, 1e-6, 1e-4):
        raw[0, 5] = tiny
        assert physical_baseline(raw) == pytest.approx(constant, abs=max(1e-12, tiny))


def test_cpu_network_output_conserves_mass_and_zero_sinks(model):
    raw = np.array([[.2, 0., 0., .1, 2., 10., .5, .2],
                    [30., .2, .1, 3., 1., 0., .3, .6]])
    prediction = model.predict(raw)
    assert prediction.sum(axis=1) == pytest.approx(np.ones(2), abs=1e-15)
    assert np.all(prediction >= 0)
    assert prediction[0, 2] == 0 and prediction[0, 4] == 0
    assert prediction == pytest.approx(physical_baseline(raw), abs=1e-13)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_bad_features_fail_closed(model, bad):
    raw = np.array([[.2, .3, .1, .1, 2., 1., .5, .2]])
    raw[0, 0] = bad
    with pytest.raises(ValueError, match='domain'):
        model.predict(raw)


def test_impossible_capacity_rejected(model):
    with pytest.raises(ValueError, match='domain'):
        model.predict([[.2, .3, .1, .1, 2., 1., .8, .8]])


def test_hash_and_post_load_mutations_rejected(model):
    with pytest.raises(ValueError, match='hash'):
        LotionReleaseSurrogate(model.path, sha256='0'*64)
    with model.weights_path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        model.predict([[.2, .3, .1, .1, 2., 1., .5, .2]])


def test_rejected_checkpoint_not_loaded(model):
    value = deepcopy(model.manifest)
    value['accepted_for_local_surrogate_inference'] = False
    model.path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='accepted'):
        LotionReleaseSurrogate(model.path, sha256=hashlib.sha256(model.path.read_bytes()).hexdigest())


def test_volume_and_time_features_bound_to_actual_dose():
    r = request()
    fixed, water, lipid = phase_volumes(r)
    exact = _simulate_lotion_transport(r, IngredientCatalog.load_builtin())
    assert water+fixed == pytest.approx(exact['temporal_profile'][0]['aqueous_volume_ml_cm2'])
    assert lipid == pytest.approx(exact['temporal_profile'][0]['lipid_volume_ml_cm2'])
    raw = request_features(r)
    assert raw.shape == (3, 8)
    assert raw[1, :6] == pytest.approx(raw[0, :6]*4)
    changed = r.model_dump(mode='json')
    changed['application_context']['application_mass_mg_cm2'] *= 2
    rebind(changed)
    new = request_features(LotionSimulationRequest.model_validate(changed))
    assert new[:, :2] == pytest.approx(raw[:, :2]/2)
    assert new[:, 2:] == pytest.approx(raw[:, 2:])


def test_neural_response_mass_and_initial_parent_loss(model):
    r = request()
    m = r.materials[0].model_copy(update={'initial_parent_fraction': .6,
        'reaction_source_reference': 'synthetic fixture only'})
    r = r.model_copy(update={'materials': [m]})
    result = predict_release(r, IngredientCatalog.load_builtin(), model=model)
    assert result['human_similarity_percent'] is None
    assert result['training_executed'] and result['measured_lotion_observations'] == 0
    assert result['diagnostics']['mass_balance_max_abs_error_mg_cm2'] < 1e-14
    first = result['temporal_profile'][0]['materials'][0]
    assert first['remaining_mg_cm2']/first['initial_mg_cm2'] == pytest.approx(.6)
    assert first['degraded_parent_equivalent_mg_cm2']/first['initial_mg_cm2'] == pytest.approx(.4)
    assert first['headspace_mg_cm2'] == 0.


def test_unsupported_conditions_are_not_silently_extrapolated(model):
    r = request().model_copy(update={'transport_mode': 'open_sink'})
    assert predict_release(r, IngredientCatalog.load_builtin(), model=model)['status'] == 'abstained'
    r = request().model_copy(update={'times_minutes': [0., 15., 600.]})
    result = predict_release(r, IngredientCatalog.load_builtin(), model=model)
    assert result['status'] == 'abstained' and 'temporal_profile' not in result


def test_existing_numerical_result_is_unchanged(monkeypatch, model):
    import fragrance_ai.recommender.lotion_surrogate as module
    monkeypatch.setattr(module, 'configured_lotion_surrogate', lambda: model)
    r, catalog = request(), IngredientCatalog.load_builtin()
    original = _simulate_lotion_transport(r, catalog)
    combined = attach_release_prediction(original, r, catalog)
    assert {k: v for k, v in combined.items() if k != 'learned_release'} == original
    assert combined['learned_release']['same_request_solver_comparison']['state_fraction_mae'] >= 0


def test_api_trained_release_route_is_wired_without_altering_existing_route(monkeypatch, model):
    import fragrance_ai.recommender.lotion_surrogate as module
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    from tests.test_ai_extensions import Formula
    monkeypatch.setattr(module, 'configured_lotion_surrogate', lambda: model)
    app = FastAPI()
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), lambda _: {}, lambda: None)
    with TestClient(app) as client:
        caps = client.get('/v1/ai/capabilities')
        assert caps.json()['features']['body_lotion_trained_release_surrogate']
        value = request().model_dump(mode='json')
        fast = client.post('/v1/applications/body-lotion/predict-release', json=value)
        exact = client.post('/v1/applications/body-lotion/simulate', json=value)
        assert fast.status_code == exact.status_code == 200
        assert fast.json()['checkpoint_sha256'] == model.sha256
        assert exact.json()['learned_release']['checkpoint_sha256'] == model.sha256
        assert exact.json()['temporal_profile'] == _simulate_lotion_transport(request(), IngredientCatalog.load_builtin())['temporal_profile']


def test_split_generation_keeps_whole_material_and_formula_groups_disjoint():
    from scripts.train_lotion_surrogate_v58 import generate
    base = request().materials[0].model_dump(mode='json')
    source = {'materials': [{'identity_group': str(i), 'parameters': base} for i in range(40)]}
    data, oils = generate(source, 12, 580901)
    assert len(data['raw']) == 40*12*6
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not set(data['identity_group'][data['split'] == a]) & set(data['identity_group'][data['split'] == b])
        assert not set(data['formulation_id'][data['split'] == a]) & set(data['formulation_id'][data['split'] == b])
        assert not set(oils[a]) & set(oils[b])
