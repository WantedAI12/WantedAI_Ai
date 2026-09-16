"""Explicit hash-bound local V75 app; production entrypoints are unchanged."""

import os
from pathlib import Path


def create_local_app():
    from fragrance_ai.recommender.local_runtime import (
        local_profile,
        local_lotion_provider,
    )
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.aligned_autoregressive import SCHEMA
    from fragrance_ai.recommender.frozen_stock_assay import FrozenStockAssay
    from deploy.web_app import create_web_app

    profile = local_profile()
    core = configured_formulation_core()
    if (
        os.environ.get("PERFUMERY_AI_ENV") != "research"
        or profile is None
        or core is None
        or core.version != SCHEMA
        or core.sha256 != profile["formulation_core"][1]
    ):
        raise ValueError(
            "explicit accepted source-bound V75 research checkpoint required"
        )
    perfume = configured_perception()
    lotion = local_lotion_provider(configured_perception("body_lotion"))
    if perfume.core is not core or lotion.core is not core:
        raise ValueError("both products must use the selected single checkpoint")
    stock = FrozenStockAssay(profile)
    root = Path(profile["profile_path"]).parent
    path, digest = profile["catalog"]
    app = create_web_app(
        str(root / "benchmarks/industrial_ingredient_registry_v1.db"),
        perception_guidance=perfume,
        lotion_perception_guidance=lotion,
        stock_mixture_predictor=stock,
        catalog_manifest_path=path,
        catalog_manifest_sha256=digest,
    )
    app.state.local_stock_mixture_predictor = stock
    app.state.local_profile_sha256 = profile["profile_sha256"]
    return app
