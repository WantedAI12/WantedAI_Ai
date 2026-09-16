"""Import-safe API factory for the single shared V69 model on Modal."""

import os
from pathlib import Path

from deploy.runtime_release_v69 import CORE_SHA256


def create_release_app(*, language_backend=None, registry_path=None):
    from fragrance_ai.recommender.local_runtime import (
        local_profile,
        local_lotion_provider,
    )
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.frozen_stock_assay import FrozenStockAssay
    from deploy.web_app import create_web_app

    profile = local_profile()
    if not profile or os.environ.get("PERFUMERY_AI_ENV") != "research":
        raise ValueError("explicit V69 research profile required")
    core = configured_formulation_core()
    if core is None or core.sha256 != CORE_SHA256:
        raise ValueError("release cannot fall back to a legacy main model")
    perfume = configured_perception()
    lotion = local_lotion_provider(configured_perception("body_lotion"))
    if perfume.core is not core or lotion.core is not core:
        raise ValueError("release product views must share the same checkpoint")
    stock = FrozenStockAssay(profile)
    path, digest = profile["catalog"]
    root = Path(profile["profile_path"]).parent
    app = create_web_app(
        str(Path(registry_path).resolve() if registry_path is not None else root / "benchmarks/industrial_ingredient_registry_v1.db"),
        perception_guidance=perfume,
        lotion_perception_guidance=lotion,
        stock_mixture_predictor=stock,
        language_backend=language_backend,
        catalog_manifest_path=path,
        catalog_manifest_sha256=digest,
    )
    app.state.local_stock_mixture_predictor = stock
    app.state.local_profile_sha256 = profile["profile_sha256"]
    return app
