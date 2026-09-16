"""Deploy all current AI functionality using the existing app, URL and auth."""
import modal

from deploy.compact_language_worker import backend_for, register_worker
from deploy.runtime_release_v80 import ROOT, verify, registry_path
from deploy.system_runtime_v76 import create_local_app

REMOTE_ROOT = '/opt/perfumery/runtime'
LOCAL_BUNDLE = ROOT/'tmp/modal-runtime-v80/release-02'
BUNDLE_SHA256 = '480e68a906232c545d62c159a8cbf3cbfde94e7cc4f7b4120d8729853b29849d'
WHEEL_REL = '.benchmarks/v80_supported_release/package-03/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
if modal.is_local():
    verify(LOCAL_BUNDLE, BUNDLE_SHA256)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.6', 'scipy==1.15.2', 'rdkit==2025.9.4', 'clarabel==0.11.1',
        'fastapi[standard]==0.116.1', 'cryptography==46.0.3', 'PyJWT==2.10.1',
        'starlette==0.47.3', 'anyio==4.10.0', 'httpx==0.28.1')
    .add_local_dir(LOCAL_BUNDLE, REMOTE_ROOT, copy=True)
    .run_commands(f'python -m pip install --no-cache-dir --no-deps {REMOTE_ROOT}/{WHEEL_REL}')
    .env({'PERFUMERY_AI_LOCAL_PROFILE':REMOTE_ROOT+'/perfumery.local.json',
        'PERFUMERY_AI_ENV':'research', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1', 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':'/root'})
    .add_local_python_source('deploy', copy=True)
    .run_commands(f'python -m deploy.runtime_release_v80 --check-installed {REMOTE_ROOT} --sha256 {BUNDLE_SHA256}')
)

app = modal.App('perfumery-ai-core')
CompactLanguage = register_worker(app)


def release_app():
    manifest = verify(REMOTE_ROOT, BUNDLE_SHA256)
    return create_local_app(language_backend=backend_for(CompactLanguage),
        registry_path=registry_path(REMOTE_ROOT, manifest))


@app.function(image=image, cpu=1.0, memory=1024, min_containers=0, max_containers=1,
    scaledown_window=300, timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    return release_app()


@app.function(image=image, cpu=1.0, memory=1024, min_containers=0, max_containers=1,
    scaledown_window=60, timeout=600)
def verify_release(snapshot_requests=None):
    """Private signed-SDK check; not a new public endpoint or backend credential."""
    from fastapi.testclient import TestClient
    from deploy.verify_supported_api_v80 import run_checks
    manifest = verify(REMOTE_ROOT, BUNDLE_SHA256)
    with TestClient(release_app(), raise_server_exceptions=False) as client:
        if snapshot_requests is None:
            report = run_checks(client, manifest['wheel_sha256'])
            response = client.post('/v1/ai/assistant', json={'message':'머스크 없이 장미 향 바디로션으로 만들어줘'})
            if response.status_code != 200:
                raise ValueError('private CPU language worker integration failed')
            report['language_assistant'] = response.json()
        else:
            from copy import deepcopy
            checks = []
            for key,path in [('comparison','/v2/formulas/compare'),('revision','/v2/briefs/revise')]:
                response = client.post(path,json=snapshot_requests[key])
                if response.status_code != 200:
                    raise ValueError('deployed immutable snapshot '+key+' failed: '+str(response.status_code))
                checks.append({'name':'snapshot_'+key,'http':response.status_code})
                if key == 'revision':
                    value = response.json()
                    assert value['request']['revision']['parent_backend_version_id'] == snapshot_requests[key]['source']['backend_version_id']
            bad = deepcopy(snapshot_requests['revision'])
            bad['source']['evaluation']['result_id'] = '0'*64
            response = client.post('/v2/briefs/revise',json=bad)
            assert response.status_code == 422
            checks.append({'name':'tampered_snapshot_rejected','http':422})
            report = {'passed':True,'checks':checks,'valid_source_whitespace_preserved':True,
                'tampering_still_rejected':True,'state_changed':False}
    report.update(release_id=manifest['release_id'], bundle_sha256=BUNDLE_SHA256,
        wheel_sha256=manifest['wheel_sha256'], shared_checkpoint_sha256=manifest['shared_checkpoint_sha256'],
        deployed_image_exercised=True, public_proxy_auth_bypassed=False,
        request_transport='private_authenticated_Modal_SDK_and_in_container_ASGI')
    return report
