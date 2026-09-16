"""Prepared V70 deployment, preserving the existing URL and proxy authentication.

Defining this entrypoint does not deploy it. A release operator must explicitly
authorize and invoke deployment after reviewing the full-scope results.
"""
import modal

from deploy.compact_language_worker import backend_for, register_worker
from deploy.runtime_release_v69 import ROOT
from deploy.runtime_release_v70 import registry_path, verify
from deploy.shared_runtime_v69 import create_release_app

REMOTE_ROOT = '/opt/perfumery/runtime'
LOCAL_BUNDLE = ROOT / 'tmp/modal-runtime-v70/release-01'
BUNDLE_SHA256 = 'f94ea18d1035fb8112f1cb21265448311eec909466aa7d4d94994325792cfb23'
WHEEL_REL = '.benchmarks/v70_repair/candidate-09/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
if modal.is_local():
    verify(LOCAL_BUNDLE, BUNDLE_SHA256)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.6', 'scipy==1.15.2', 'rdkit==2025.9.4', 'clarabel==0.11.1',
                 'fastapi[standard]==0.116.1', 'cryptography==46.0.3', 'PyJWT==2.10.1',
                 'starlette==0.47.3', 'anyio==4.10.0', 'httpx==0.28.1')
    .add_local_dir(LOCAL_BUNDLE, REMOTE_ROOT, copy=True)
    .run_commands(f'python -m pip install --no-cache-dir --no-deps {REMOTE_ROOT}/{WHEEL_REL}')
    .env({'PERFUMERY_AI_LOCAL_PROFILE': REMOTE_ROOT + '/perfumery.local.json',
          'PERFUMERY_AI_ENV': 'research', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '/root'})
    .add_local_python_source('deploy', copy=True)
    .run_commands(f'python -m deploy.runtime_release_v70 --check-installed {REMOTE_ROOT} --sha256 {BUNDLE_SHA256}')
)

app = modal.App('perfumery-ai-core')
CompactLanguage = register_worker(app)


@app.function(image=image, cpu=1.0, memory=1024, min_containers=0, max_containers=1,
              scaledown_window=300, timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    manifest = verify(REMOTE_ROOT, BUNDLE_SHA256)
    return create_release_app(language_backend=backend_for(CompactLanguage),
                              registry_path=registry_path(REMOTE_ROOT, manifest))
