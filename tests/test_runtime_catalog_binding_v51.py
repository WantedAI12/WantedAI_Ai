"""Real loaders and API/SDK routing on clearly synthetic catalog fixtures."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

pytest.importorskip('modal')

from deploy import web_app as modal_app
from fragrance_ai.recommender import runtime
from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION
from fragrance_ai.recommender.registry_activation import RegistryActivationReport, write_runtime_catalog
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    for name in (runtime.MANIFEST_ENV, runtime.MANIFEST_HASH_ENV,
                 'PERFUMERY_AI_PERCEPTION_MANIFEST', 'PERFUMERY_AI_PERCEPTION_MANIFEST_SHA256',
                 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST', 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST_SHA256'):
        monkeypatch.delenv(name, raising=False)
    request, catalog = fixture()
    registry = tmp_path/'synthetic-registry-placeholder'
    registry.write_bytes(b'not a real registry; metadata-only unit test')
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
    monkeypatch.setattr(modal_app, 'REGISTRY_SHA256', registry_sha)
    monkeypatch.setattr(modal_app, 'RUNTIME_CATALOG', tmp_path/'old-catalog-must-not-be-used.gz')
    monkeypatch.setattr(modal_app, 'RUNTIME_CATALOG_SHA256', 'f'*64)
    report = RegistryActivationReport(registry_sha, 2, 0, 2, 0, 0, len(catalog.ingredients), 0, 0, 0)
    blob = tmp_path/'catalog.json.gz'
    digest = write_runtime_catalog(blob, catalog, report, {'safety_screened': 2}, wheel_sha256='b'*64)
    manifest = {'odor_integrity_version': ODOR_INTEGRITY_VERSION,
        'runtime_source_sha256': runtime._source_snapshot(), 'runtime_data_sha256': runtime._data_snapshot(),
        'runtime_catalog': {'path': blob.name, 'sha256': digest, 'wheel_sha256': 'b'*64, 'registry_sha256': registry_sha}}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return request, catalog, registry, path, sha, manifest


def app_for(snapshot, **kwargs):
    _, _, registry, path, sha, _ = snapshot
    return modal_app.create_web_app(str(registry), catalog_manifest_path=path, catalog_manifest_sha256=sha, **kwargs)


def test_explicit_catalog_is_used_by_health_catalog_lotion_and_cache(snapshot, monkeypatch):
    request, catalog, _, _, sha, manifest = snapshot
    original, calls = runtime.load_runtime_catalog, []
    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'load_runtime_catalog', load)
    app = app_for(snapshot)
    assert len(calls) == 1
    sdk = optimize_lotion(LotionOptimizationRequest.model_validate(request), catalog, _transport_only=True)
    with TestClient(app) as client:
        assert client.get('/health').json()['wheel_sha256'] == 'b'*64
        binding = client.get('/v1/catalog').json()['runtime_binding']
        assert binding['catalog_manifest_sha256'] == sha
        assert binding['material_snapshot_sha256'] == runtime._material_digest(catalog)
        assert binding['catalog_sha256'] == manifest['runtime_catalog']['sha256']
        response = client.post('/v1/applications/body-lotion/optimize', json=request)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['recipe'] == sdk['recipe'] and result['score'] == sdk['score']
        assert result['catalog_snapshot'] == binding
        assert result['product_model']['component_model']['configured'] is False
        repeated = client.post('/v1/applications/body-lotion/optimize', json=request)
        assert repeated.json() == result
        assert repeated.headers['X-Perfumery-Lotion-Cache'] == 'hit'
        assert len(calls) == 1  # No catalog/database reload per API call.


def test_environment_binding_agrees_with_sdk_without_explicit_api_arguments(snapshot, monkeypatch):
    _, expected, registry, path, sha, _ = snapshot
    monkeypatch.setenv(runtime.MANIFEST_ENV, str(path))
    monkeypatch.setenv(runtime.MANIFEST_HASH_ENV, sha)
    catalog, digest = runtime.load_configured_catalog()
    assert catalog.ingredients == expected.ingredients and digest == sha
    with TestClient(modal_app.create_web_app(str(registry))) as client:
        response = client.get('/v1/catalog')
        assert response.status_code == 200
        assert response.json()['runtime_binding']['material_snapshot_sha256'] == runtime._material_digest(catalog)
        payload = {'brief': 'citrus woody scent'}
        first = client.post('/v1/formulas', json=payload)
        assert first.status_code == 200, first.text
        second = client.post('/v1/formulas', json=payload)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert second.headers['X-Perfumery-Cache'] == 'hit'
        assert second.json()['deployment']['catalog_snapshot']['catalog_manifest_sha256'] == sha


@pytest.mark.parametrize('arguments', [
    {'catalog_manifest_path': ''}, {'catalog_manifest_sha256': 'a'*64},
    {'catalog_manifest_path': 'missing.json'},
])
def test_partial_explicit_binding_does_not_borrow_environment_or_fall_back(snapshot, monkeypatch, arguments):
    _, _, registry, path, sha, _ = snapshot
    monkeypatch.setenv(runtime.MANIFEST_ENV, str(path))
    monkeypatch.setenv(runtime.MANIFEST_HASH_ENV, sha)
    with pytest.raises(ValueError, match='both manifest'):
        modal_app.create_web_app(str(registry), **arguments)


def test_ambiguous_and_invalid_trust_binding_fail_before_legacy_catalog(snapshot):
    _, _, registry, path, sha, _ = snapshot
    with pytest.raises(ValueError, match='not both'):
        app_for(snapshot, runtime_catalog_path='old-catalog.gz')
    with pytest.raises(ValueError, match='hash'):
        modal_app.create_web_app(str(registry), catalog_manifest_path=path, catalog_manifest_sha256='0'*64)


@pytest.mark.parametrize('changed', ['manifest', 'catalog', 'environment'])
def test_bound_configuration_drift_cannot_return_old_lotion_cache(snapshot, monkeypatch, changed):
    request, _, _, path, _, manifest = snapshot
    with TestClient(app_for(snapshot)) as client:
        first = client.post('/v1/applications/body-lotion/optimize', json=request)
        assert first.status_code == 200
        if changed == 'environment':
            monkeypatch.setenv(runtime.MANIFEST_HASH_ENV, '0'*64)
        else:
            target = path if changed == 'manifest' else path.parent/manifest['runtime_catalog']['path']
            target.write_bytes(target.read_bytes()+b' ')
        response = client.post('/v1/applications/body-lotion/optimize', json=request)
        assert response.status_code == 422
        assert 'reload API' in response.json()['detail']
