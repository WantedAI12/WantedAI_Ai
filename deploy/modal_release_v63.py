"""Authenticated V63 research deployment; keeps the existing app and web URL."""
import os
from pathlib import Path

import modal

from deploy.compact_language_worker import register_worker,backend_for
from deploy.runtime_release_v63 import ROOT,WHEEL_REL,verify

REMOTE_ROOT='/opt/perfumery/runtime'
LOCAL_BUNDLE=ROOT/'tmp/modal-runtime-v63/release-01'
BUNDLE_SHA256='16bb65f75e96a2deada5ada6bf8750c99c1ddb610bc7c49d07814cf085521c7b'
if modal.is_local():
    verify(LOCAL_BUNDLE,BUNDLE_SHA256)

image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.6','scipy==1.15.2','rdkit==2025.9.4','clarabel==0.11.1',
                 'fastapi[standard]==0.116.1','cryptography==46.0.3')
    .add_local_dir(LOCAL_BUNDLE,REMOTE_ROOT,copy=True)
    .run_commands(f'python -m pip install --no-cache-dir --no-deps {REMOTE_ROOT}/{WHEEL_REL}')
    .env({'PERFUMERY_AI_LOCAL_PROFILE':REMOTE_ROOT+'/perfumery.local.json','PERFUMERY_AI_ENV':'research',
          'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    .add_local_file(ROOT/'deploy/runtime_release_v63.py','/opt/perfumery/runtime_release_v63.py',copy=True)
    .run_commands(f'python /opt/perfumery/runtime_release_v63.py --check-installed {REMOTE_ROOT}')
    .add_local_python_source('deploy'))

app=modal.App('perfumery-ai-core')
CompactLanguage=register_worker(app)


def create_release_app(*,language_backend=None):
    """Same pinned factories as the local API, explicit Linux artifact root."""
    from fragrance_ai.recommender.local_runtime import local_profile,local_lotion_provider,local_atlas_provider
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai import StockMixturePredictor
    from deploy.web_app import create_web_app
    profile=local_profile()
    if not profile or os.environ.get('PERFUMERY_AI_ENV')=='production':
        raise ValueError('explicit research profile required')
    perfume=configured_perception()
    lotion=local_lotion_provider(configured_perception('body_lotion'))
    path,digest=profile['stock_mixture']
    stock=StockMixturePredictor(perfume,path,sha256=digest,experimental=True,atlas_predictor=local_atlas_provider())
    path,digest=profile['catalog']
    root=Path(profile['profile_path']).parent
    return create_web_app(str(root/'benchmarks/industrial_ingredient_registry_v1.db'),
        perception_guidance=perfume,lotion_perception_guidance=lotion,stock_mixture_predictor=stock,
        language_backend=language_backend,catalog_manifest_path=path,catalog_manifest_sha256=digest)


@app.function(image=image,cpu=1.,memory=1024,min_containers=0,max_containers=1,
              scaledown_window=300,timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    verify(REMOTE_ROOT,BUNDLE_SHA256)
    return create_release_app(language_backend=backend_for(CompactLanguage))
