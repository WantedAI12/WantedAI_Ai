"""Local CPU API with pinned product models and the integrated V54 stock head."""
import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_app(*, lotion_reference=None, enable_language=True):
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.runtime import MANIFEST_ENV, MANIFEST_HASH_ENV
    from fragrance_ai.recommender.local_runtime import local_profile, configured_pair, local_lotion_provider, local_atlas_provider
    if os.environ.get('PERFUMERY_AI_ENV', '').strip().lower() == 'production':
        raise ValueError('V42 product models are research-only, not production releases')
    if lotion_reference not in (None, 'component', 'atlas'):
        raise ValueError('unknown lotion reference profile model')
    profile = local_profile()
    if profile is None:
        raise ValueError('local API requires perfumery.local.json or PERFUMERY_AI_LOCAL_PROFILE')
    perfume, lotion = configured_perception(), configured_perception('body_lotion')
    if perfume is None or lotion is None or perfume is lotion:
        raise ValueError('two independently bound product models must be enabled')
    lotion = local_lotion_provider(lotion, reference=lotion_reference)
    # An explicit stock-assay lane. Never substituted for perfume or lotion
    # release/quality scores. The cross-moment V49 candidate remains optional.
    from fragrance_ai import StockMixturePredictor
    stock_path, stock_sha = profile['stock_mixture']
    stock = StockMixturePredictor(perfume, stock_path, sha256=stock_sha, experimental=True,
                                  atlas_predictor=local_atlas_provider())
    from fragrance_ai.recommender.local_language import configured_language
    language = configured_language() if enable_language else None
    path, digest = configured_pair(MANIFEST_ENV, MANIFEST_HASH_ENV, 'catalog')
    from deploy.modal_app import create_web_app, REGISTRY
    try:
        web = create_web_app(str(REGISTRY), perception_guidance=perfume, lotion_perception_guidance=lotion,
            stock_mixture_predictor=stock, language_backend=language,
            catalog_manifest_path=path, catalog_manifest_sha256=digest)
    except BaseException:
        if language is not None:
            language.close()
        raise
    original_lifespan = web.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        try:
            async with original_lifespan(app) as state:
                yield state
        finally:
            if language is not None:
                language.close()
    web.router.lifespan_context = lifespan
    web.state.local_language_backend = language
    web.state.local_stock_mixture_predictor = stock
    web.state.local_profile_sha256 = profile['profile_sha256']
    return web


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--lotion-reference', choices=('component', 'atlas'), help='default comes from the local pinned profile')
    p.add_argument('--no-language', action='store_true', help='explicit deterministic-parser-only local session')
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error('port must be between 1024 and 65535')
    import uvicorn
    uvicorn.run(create_app(lotion_reference=args.lotion_reference, enable_language=not args.no_language), host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
