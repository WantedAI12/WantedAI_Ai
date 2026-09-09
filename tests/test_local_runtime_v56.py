import json
from pathlib import Path

import pytest

from fragrance_ai.recommender import local_runtime as runtime


@pytest.fixture
def profile(tmp_path,monkeypatch):
    path = tmp_path/'perfumery.local.json'
    value = {'schema':'perfumery-local-runtime/v1','scope':'local_research','lotion_reference':'atlas'}
    for name in ('catalog','perfume','body_lotion','atlas','stock_mixture'):
        value[name] = {'path': name+'.json','sha256':'a'*64}
    path.write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setenv(runtime.PROFILE_ENV,str(path))
    monkeypatch.delenv('PERFUMERY_AI_ENV',raising=False)
    return path


def test_profile_is_not_a_shared_mutable_configuration(profile):
    first = runtime.local_profile()
    first['catalog'] = ('wrong','b'*64)
    assert runtime.local_profile()['catalog'][0] == str(profile.parent/'catalog.json')


def test_partial_environment_override_is_not_filled_from_local_profile(profile,monkeypatch):
    monkeypatch.setenv('CUSTOM_PATH','explicit.json')
    monkeypatch.delenv('CUSTOM_HASH',raising=False)
    assert runtime.configured_pair('CUSTOM_PATH','CUSTOM_HASH','catalog') == ('explicit.json','')


def test_profile_drift_changes_snapshot(profile):
    before = runtime.local_snapshot()
    value = json.loads(profile.read_text())
    value['catalog']['sha256'] = 'b'*64
    profile.write_text(json.dumps(value),encoding='utf-8')
    assert runtime.local_snapshot() != before


def test_local_models_cannot_be_implicitly_promoted_to_production(profile,monkeypatch):
    monkeypatch.setenv('PERFUMERY_AI_ENV','production')
    with pytest.raises(ValueError,match='production'):
        runtime.local_profile()
    monkeypatch.delenv(runtime.PROFILE_ENV)
    assert runtime.local_profile() is None


def test_missing_explicit_profile_is_not_a_builtin_fallback(profile,monkeypatch):
    monkeypatch.setenv(runtime.PROFILE_ENV,str(profile.parent/'missing.json'))
    with pytest.raises(ValueError,match='unavailable'):
        runtime.local_profile()


@pytest.mark.parametrize('change', [('path','../outside.json'),('sha256',123),('sha256','wrong')])
def test_invalid_artifact_binding_rejected(profile,change):
    data = json.loads(profile.read_text())
    data['catalog'][change[0]] = change[1]
    profile.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError):
        runtime.local_profile()


def test_language_artifact_paths_and_hashes_are_required(tmp_path):
    from fragrance_ai.recommender.local_language import LocalLanguageBackend
    with pytest.raises(ValueError,match='path and SHA256'):
        LocalLanguageBackend({'model':{}},tmp_path)


def test_closed_lazy_language_backend_does_not_spawn(tmp_path):
    from fragrance_ai.recommender.local_language import LocalLanguageBackend
    settings = {}
    for name in ('model','executable'):
        (tmp_path/name).write_bytes(b'')
        settings[name] = {'path':name,'sha256':'a'*64}
    backend = LocalLanguageBackend(settings,tmp_path)
    backend.close()
    with pytest.raises(ValueError,match='closed'):
        backend('test')
    assert backend._process is None


def test_odor_correction_requires_parent_and_changes_snapshot(profile):
    value=json.loads(profile.read_text())
    value['odor_calibration']={'path':'correction.json','sha256':'c'*64}
    profile.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='requires a pinned'):
        runtime.local_profile()
    value['odor_expression']={'path':'expression.json','sha256':'e'*64}
    profile.write_text(json.dumps(value))
    before=runtime.local_snapshot()
    assert runtime.local_profile()['odor_calibration']==(str(profile.parent/'correction.json'),'c'*64)
    value['odor_calibration']['sha256']='d'*64
    profile.write_text(json.dumps(value))
    assert runtime.local_snapshot()!=before
