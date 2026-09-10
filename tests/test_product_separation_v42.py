"""Product routing, evidence isolation and unmodified numeric gates."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fragrance_ai.recommender import perception_runtime as runtime
from fragrance_ai.recommender.lotion_evaluation import phase_for_time, compare_lotion_profiles, LOTION_PROJECTION
from fragrance_ai.recommender.perception_guidance import PROJECTION
from fragrance_ai.recommender.profile_match import compare_profiles
from fragrance_ai.platform.ai_extensions import register_ai_extensions
from tests.test_ai_extensions import Formula
from tests.test_lotion_perception_v35 import model
from tests.test_perception_guidance import provider as synthetic_provider

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_model_config(monkeypatch):
    for name in (runtime.PATH_ENV, runtime.HASH_ENV, runtime.LOTION_PATH_ENV, runtime.LOTION_HASH_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PERFUMERY_AI_ENV', 'development')


def configure(monkeypatch, product):
    path = ROOT/'.benchmarks/product_runtime_v42'/f'{product}.json'
    if not path.exists():
        pytest.skip('local research manifests not distributed')
    names = (runtime.PATH_ENV, runtime.HASH_ENV) if product == 'perfume' else (runtime.LOTION_PATH_ENV, runtime.LOTION_HASH_ENV)
    monkeypatch.setenv(names[0], str(path))
    monkeypatch.setenv(names[1], hashlib.sha256(path.read_bytes()).hexdigest())
    return runtime.configured_perception(product)


def test_no_implicit_perfume_to_lotion_fallback(monkeypatch):
    perfume = configure(monkeypatch, 'perfume')
    assert perfume.runtime_product == 'perfume'
    assert runtime.configured_perception('body_lotion') is None
    with pytest.raises(ValueError, match='different product'):
        from fragrance_ai.recommender.lotion_perception import resolve_provider
        resolve_provider(perfume)


def test_real_checkpoint_instances_and_mutable_caches_are_isolated(monkeypatch):
    perfume = configure(monkeypatch, 'perfume')
    lotion = configure(monkeypatch, 'body_lotion')
    assert perfume is not lotion
    assert perfume.component_model_sha256 == lotion.component_model_sha256
    assert perfume.lotion_shape_cache is not lotion.lotion_shape_cache
    assert perfume.model is not lotion.model
    assert runtime.configured_perception('body_lotion') is lotion
    monkeypatch.delenv(runtime.LOTION_PATH_ENV)
    monkeypatch.delenv(runtime.LOTION_HASH_ENV)
    runtime.assert_provider_current(perfume)
    with pytest.raises(ValueError, match='changed'):
        runtime.assert_provider_current(lotion)


def test_manifest_cannot_route_perfume_checkpoint_as_lotion(monkeypatch):
    path = ROOT/'.benchmarks/product_runtime_v42/perfume.json'
    if not path.exists():
        pytest.skip('local research manifest unavailable')
    monkeypatch.setenv(runtime.LOTION_PATH_ENV, str(path))
    monkeypatch.setenv(runtime.LOTION_HASH_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='product mismatch'):
        runtime.configured_perception('body_lotion')


def test_lotion_model_rejected_by_perfume_sdk(monkeypatch):
    from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI
    lotion = configure(monkeypatch, 'body_lotion')
    with pytest.raises(ValueError, match='different product'):
        NaturalLanguagePerfumeryAI(perception_guidance=lotion)


def test_lotion_schedule_and_projection_do_not_import_perfume_policy(monkeypatch):
    from fragrance_ai.recommender.science import TemporalMixtureSimulator
    from fragrance_ai.platform.lotion_inputs import TransitionSchedule
    monkeypatch.setattr(TemporalMixtureSimulator, '_phase_for_time', lambda t: 'wrong_product')
    monkeypatch.setitem(PROJECTION, 'citrus', ('WrongPerfumeEndpoint',))
    assert LOTION_PROJECTION['citrus'] == ('Citrus',)
    assert [phase_for_time(t) for t in (15, 30, 240, 480)] == ['opening', 'heart', 'heart', 'drydown']
    custom = TransitionSchedule(opening_until_minutes=5, heart_until_minutes=60)
    assert [phase_for_time(t, custom) for t in (4, 15, 120)] == ['opening', 'heart', 'drydown']


def test_product_score_is_tagged_without_inflating_or_dropping_axes():
    target, prediction = {'woody': 1.}, {'woody': .7, 'musky': .3}
    shared = compare_profiles(target, prediction, avoided=['musky'])
    lotion = compare_lotion_profiles(target, prediction, avoided=['musky'])
    assert lotion.score == shared.score == pytest.approx(70.)
    assert lotion.dimensions == shared.dimensions and lotion.deficit == shared.deficit
    assert lotion.version != shared.version and lotion.score_kind != shared.score_kind


def test_api_perfume_configuration_does_not_enable_lotion_learning(model):
    value, catalog, p, calls = model
    app = FastAPI()
    register_ai_extensions(app, Formula, catalog, lambda *a: {}, lambda: None, perception_guidance=p)
    with TestClient(app) as client:
        capability = client.get('/v1/ai/capabilities').json()
        assert capability['perception_model']['configured']
        assert not capability['features']['body_lotion_learned_temporal_shape']
        result = client.post('/v1/applications/body-lotion/simulate', json=value['simulation'])
        assert result.status_code == 200, result.text
        assert not calls and not result.json()['product_model']['component_model']['configured']


def test_api_lotion_uses_its_own_provider_only_and_keeps_cache(model, monkeypatch):
    value, catalog, lotion, calls = model
    perfume = synthetic_provider()
    monkeypatch.setattr(perfume, 'begin', lambda *_: pytest.fail('perfume provider used by lotion'))
    app = FastAPI()
    register_ai_extensions(app, Formula, catalog, lambda *a: {}, lambda: None,
                           perception_guidance=perfume, lotion_perception_guidance=lotion)
    with TestClient(app) as client:
        a = client.post('/v1/applications/body-lotion/optimize', json=value)
        assert a.status_code == 200, a.text
        assert calls and a.json()['product_model']['product'] == 'body_lotion'
        count = len(calls)
        b = client.post('/v1/applications/body-lotion/optimize', json=value)
        assert b.json() == a.json() and len(calls) == count
        assert b.headers['X-Perfumery-Lotion-Cache'] == 'hit'


def test_api_rejects_same_mutable_provider_for_both_products(model):
    _, catalog, p, _ = model
    with pytest.raises(ValueError, match='independent provider'):
        register_ai_extensions(FastAPI(), Formula, catalog, lambda *a: {}, lambda: None,
                               perception_guidance=p, lotion_perception_guidance=p)


def test_lotion_environment_enable_after_start_is_not_ignored(model, monkeypatch):
    value, catalog, _, _ = model
    app = FastAPI()
    register_ai_extensions(app, Formula, catalog, lambda *a: {}, lambda: None)
    configure(monkeypatch, 'body_lotion')
    with TestClient(app) as client:
        result = client.post('/v1/applications/body-lotion/simulate', json=value['simulation'])
        assert result.status_code == 422
        assert 'policy changed' in result.json()['detail']


def test_product_benchmark_mixing_is_rejected(tmp_path):
    from scripts.compare_lotion_benchmarks import read_run
    (tmp_path/'manifest.json').write_text(json.dumps({'product': 'perfume'}))
    (tmp_path/'summary.json').write_text(json.dumps({'product': 'body_lotion'}))
    with pytest.raises(ValueError, match='cannot be combined'):
        read_run(tmp_path)
