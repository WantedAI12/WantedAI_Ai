"""Local research API with the learned V4 checkpoint required, never silently off."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MANIFEST_SHA256='2c288272f1c633e3fb9a761e9b6ad6468cb2eff16a6e3e9994b6c294ebdc001d'


def create_app():
    from fragrance_ai.recommender.perception_runtime import PATH_ENV,HASH_ENV,LOTION_PATH_ENV,LOTION_HASH_ENV,configured_perception
    if os.environ.get('PERFUMERY_AI_ENV','').strip().lower()=='production':
        raise ValueError('V4 is a local research checkpoint, not a production release')
    expected={PATH_ENV:str(ROOT/'.benchmarks/perception_runtime_v41/manifest.json'), HASH_ENV:MANIFEST_SHA256,
              LOTION_PATH_ENV:str(ROOT/'.benchmarks/product_runtime_v42/body_lotion.json'),
              LOTION_HASH_ENV:'aae267ac9714e3d418208c2e819eb53d5799e116e65987ec6f06b2b7ea07ae37'}
    for name,value in expected.items():
        if os.environ.get(name) not in (None,'',value):
            raise ValueError('existing perception configuration differs; do not overwrite it')
    os.environ.update(expected)
    provider=configured_perception()
    if provider is None or provider.component_model_version!='v4':
        raise ValueError('V4 checkpoint must be enabled')
    from deploy.modal_app import create_web_app,REGISTRY
    return create_web_app(str(REGISTRY),perception_guidance=provider)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=8001)
    args=p.parse_args()
    if not 1024<=args.port<=65535:
        p.error('port must be between 1024 and 65535')
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=args.port,workers=1)


if __name__=='__main__':
    main()
