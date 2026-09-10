from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.research.kernel_profiles import fit_kernel, fit_kernel_v4, predict_component_regressor, _kernel_v4


def arrays():
    rng = np.random.default_rng(41)
    x = np.zeros((12,1105))
    x[:, :1024] = rng.integers(0,2,(12,1024))
    x[:, 1024:1095] = rng.normal(size=(12,71))
    x[:, 1095] = np.linspace(-4,0,12)
    x[:, 1096] = x[:, 1095]**2
    x[:, 1097] = 1
    return x, rng.uniform(size=(12,51))


def test_kernel_is_symmetric_psd_and_context_sensitive():
    x, _ = arrays()
    params = dict(fingerprint_weight=.25, native_weight=.25, dose_bandwidth=2.)
    k = _kernel_v4(x,x,np.ones(81),params)
    np.testing.assert_allclose(k,k.T,atol=1e-12)
    assert np.linalg.eigvalsh(k).min() > -1e-10
    shifted = x.copy()
    shifted[:,1095] -= 3
    assert np.max(abs(k-_kernel_v4(x,shifted,np.ones(81),params))) > .1


def test_prediction_roundtrip_batch_parity_and_v3_unchanged():
    import json
    x,y = arrays()
    old = fit_kernel(x,y,1.)
    before = predict_component_regressor(old,x)
    model = json.loads(json.dumps(fit_kernel_v4(x,y,1.)))
    batch = predict_component_regressor(model,x)
    scalar = np.vstack([predict_component_regressor(model,row[None,:]) for row in x])
    np.testing.assert_allclose(batch,scalar,atol=1e-12)
    np.testing.assert_array_equal(before,predict_component_regressor(old,x))
    assert np.isfinite(batch).all() and np.all(batch>=0)


@pytest.mark.parametrize('changes', [{'fingerprint_weight':1.1}, {'native_weight':-.1}, {'dose_bandwidth':0}, {'dose_bandwidth':float('nan')}])
def test_invalid_kernel_parameters_fail_closed(changes):
    x,y=arrays()
    with pytest.raises(ValueError):
        fit_kernel_v4(x,y,1.,**changes)


def test_invalid_feature_width_rejected():
    x,y=arrays()
    with pytest.raises(ValueError,match='1105'):
        fit_kernel_v4(x[:,:1100],y,1.)


def test_real_v4_manifest_loads_and_preserves_shared_cache(monkeypatch):
    from fragrance_ai.recommender.perception_runtime import PATH_ENV, HASH_ENV, configured_perception
    path=Path(__file__).resolve().parents[1]/'.benchmarks/perception_runtime_v41/manifest.json'
    if not path.exists():
        pytest.skip('local research checkpoint unavailable')
    monkeypatch.setenv('PERFUMERY_AI_ENV','development')
    monkeypatch.setenv(PATH_ENV,str(path))
    monkeypatch.setenv(HASH_ENV,hashlib.sha256(path.read_bytes()).hexdigest())
    provider=configured_perception()
    assert provider.component_model_version=='v4'
    assert provider.model['regressor']['kind']=='molecular-kernel-v4'
    assert configured_perception() is provider


def test_training_only_folds_and_mae_report_are_bound_to_same_population():
    import json
    root=Path(__file__).resolve().parents[1]
    path=root/'.benchmarks/perception_core_v4/run-01/report.json'
    if not path.exists():
        pytest.skip('local training report unavailable')
    report=json.loads(path.read_text(encoding='utf-8'))
    for fold in range(5):
        train={g for g,f in zip(report['outer_group_ids'],report['outer_fold_ids']) if f!=fold}
        test={g for g,f in zip(report['outer_group_ids'],report['outer_fold_ids']) if f==fold}
        assert not train & test
    assert report['comparison']['paired_profiles']==report['predicted_conditions']==333
    assert report['outcome_read_attempts']==[]


def test_manifest_cannot_mislabel_checkpoint_version(monkeypatch,tmp_path):
    import json
    from fragrance_ai.recommender.perception_runtime import PATH_ENV,HASH_ENV,configured_perception
    root=Path(__file__).resolve().parents[1]
    source=root/'.benchmarks/perception_runtime_v41/manifest.json'
    if not source.exists():
        pytest.skip('local model unavailable')
    document=json.loads(source.read_text(encoding='utf-8'))
    document['component_model_version']='v3'
    for key in ('base_model','component_model','registry'):
        document[key]['path']=str((source.parent/document[key]['path']).resolve())
    path=tmp_path/'mislabeled.json'
    path.write_text(json.dumps(document),encoding='utf-8')
    monkeypatch.setenv('PERFUMERY_AI_ENV','development')
    monkeypatch.setenv(PATH_ENV,str(path))
    monkeypatch.setenv(HASH_ENV,hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='versions differ'):
        configured_perception()


def test_v41_launcher_cannot_silently_disable_the_model(monkeypatch):
    from scripts.serve_perception_runtime_v41 import create_app
    from fragrance_ai.recommender.perception_runtime import PATH_ENV,HASH_ENV
    monkeypatch.setenv('PERFUMERY_AI_ENV','development')
    monkeypatch.setenv(PATH_ENV,'some-other-model')
    monkeypatch.delenv(HASH_ENV,raising=False)
    with pytest.raises(ValueError,match='configuration differs'):
        create_app()
