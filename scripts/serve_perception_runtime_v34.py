"""Local-only V3 component priors, independently bound to perfume and lotion."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST_SHA256 = 'a0be5c314fbac46e175d718a5e3ed8ed920e07b0e8ef8500ac8a1f1f810fbea1'


def create_app():
    if os.environ.get('PERFUMERY_AI_ENV', '').strip().lower() == 'production':
        raise ValueError('this checkpoint is a local research model, not a production release')
    from fragrance_ai.recommender.perception_runtime import PATH_ENV, HASH_ENV, LOTION_PATH_ENV, LOTION_HASH_ENV
    expected = {PATH_ENV: str(ROOT/'.benchmarks/perception_runtime_v34/manifest.json'), HASH_ENV: MANIFEST_SHA256,
                LOTION_PATH_ENV: str(ROOT/'.benchmarks/product_runtime_v42/body_lotion_v3.json'),
                LOTION_HASH_ENV: 'c5c5aa94b6ac21ffb63abfdec7c3fd36e4ec20f9abc6d3dd16a7cc8c66181133'}
    for name, value in expected.items():
        if os.environ.get(name) not in (None, '', value):
            raise ValueError('existing perception configuration differs; do not silently overwrite it')
    os.environ.update(expected)
    from deploy.modal_app import create_web_app, REGISTRY
    return create_web_app(str(REGISTRY))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8001)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error('port must be between 1024 and 65535')
    import uvicorn
    # No unauthenticated public listener. Production retains its proxy auth.
    uvicorn.run(create_app(), host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
