"""Local candidate factory; does not alter the deployed V69 release entrypoint."""
import os
from pathlib import Path

CORE_SHA256='bc45f11dd2e1ad5b80ecfe7126c8983d6c4e7e70b4ea53d93e86c0163f0a9dfb'


def create_local_app():
    from fragrance_ai.recommender.local_runtime import local_profile,local_lotion_provider
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.autoregressive_neural import SCHEMA
    from fragrance_ai.recommender.frozen_stock_assay import FrozenStockAssay
    from deploy.web_app import create_web_app
    profile=local_profile()
    core=configured_formulation_core()
    if (os.environ.get('PERFUMERY_AI_ENV')!='research' or profile is None or core is None
            or core.sha256!=CORE_SHA256 or core.version!=SCHEMA):
        raise ValueError('explicit tested V73 local checkpoint required')
    perfume=configured_perception()
    lotion=local_lotion_provider(configured_perception('body_lotion'))
    if perfume.core is not core or lotion.core is not core:
        raise ValueError('autoregressive products must share the same checkpoint')
    stock=FrozenStockAssay(profile)
    root=Path(profile['profile_path']).parent
    path,digest=profile['catalog']
    app=create_web_app(str(root/'benchmarks/industrial_ingredient_registry_v1.db'),
        perception_guidance=perfume,lotion_perception_guidance=lotion,stock_mixture_predictor=stock,
        catalog_manifest_path=path,catalog_manifest_sha256=digest)
    app.state.local_stock_mixture_predictor=stock
    app.state.local_profile_sha256=profile['profile_sha256']
    return app
