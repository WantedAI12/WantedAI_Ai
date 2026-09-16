"""V91 backend compatibility update; keep the existing authenticated URL."""
import modal
from deploy.compact_language_worker import backend_for, register_worker
from deploy.runtime_release_v91 import ROOT,verify,registry_path
from deploy.target_runtime_v87 import create_release_app

REMOTE_ROOT='/opt/perfumery/runtime'
LOCAL_BUNDLE=ROOT/'tmp/modal-runtime-v91/release-01'
BUNDLE_SHA256='5b00fa7445adff767f2b04b20863c5f782d07dbff0bf05f77c623fc8af7ed54d'
WHEEL_REL='.benchmarks/v91_backend_alignment/package-01/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
if modal.is_local():
    verify(LOCAL_BUNDLE,BUNDLE_SHA256)

image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.6','scipy==1.15.2','rdkit==2025.9.4','clarabel==0.11.1',
        'fastapi[standard]==0.116.1','cryptography==46.0.3','PyJWT==2.10.1',
        'starlette==0.47.3','anyio==4.10.0','httpx==0.28.1')
    .add_local_dir(LOCAL_BUNDLE,REMOTE_ROOT,copy=True)
    .run_commands(f'python -m pip install --no-cache-dir --no-deps {REMOTE_ROOT}/{WHEEL_REL}')
    .env({'PERFUMERY_AI_LOCAL_PROFILE':REMOTE_ROOT+'/perfumery.local.json','PERFUMERY_AI_ENV':'research',
          'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
          'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/root'})
    .add_local_python_source('deploy',copy=True)
    .run_commands(f'python -m deploy.runtime_release_v91 --check-installed {REMOTE_ROOT} --sha256 {BUNDLE_SHA256}'))

app=modal.App('perfumery-ai-core')
CompactLanguage=register_worker(app)


def release_app():
    manifest=verify(REMOTE_ROOT,BUNDLE_SHA256)
    return create_release_app(language_backend=backend_for(CompactLanguage),registry_path=registry_path(REMOTE_ROOT,manifest))


@app.function(image=image,cpu=1.0,memory=1024,min_containers=0,max_containers=1,scaledown_window=300,timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    return release_app()


@app.function(image=image,cpu=1.0,memory=1024,min_containers=0,max_containers=1,scaledown_window=60,timeout=600)
def verify_release(suite='contracts',case_index=None):
    from fastapi.testclient import TestClient
    from deploy.verify_supported_api_v89 import run_checks
    manifest=verify(REMOTE_ROOT,BUNDLE_SHA256)
    with TestClient(release_app(),raise_server_exceptions=False) as client:
        if suite=='backend_contracts':
            from deploy.verify_backend_v91 import run_backend_checks
            result=run_backend_checks(client,manifest['wheel_sha256'])
        elif suite=='contracts':
            result=run_checks(client,manifest['wheel_sha256'])
        else:
            from deploy.qa_release_v89 import run_suite
            result=run_suite(client,manifest['wheel_sha256'],suite,case_index)
    result.update(release_id=manifest['release_id'],wheel_sha256=manifest['wheel_sha256'],
        bundle_sha256=BUNDLE_SHA256,deployed_image_exercised=True,
        request_transport='private_authenticated_Modal_SDK_and_in_container_ASGI',
        public_authenticated_requests_tested=False,backend_proxy_token_tested=False,
        existing_backend_credentials_modified=False,new_proxy_credentials_created=False)
    return result
