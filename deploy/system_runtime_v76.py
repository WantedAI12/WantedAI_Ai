"""Explicit accepted, source-bound local V76 runtime. No production mutation."""

import os
from pathlib import Path


def create_local_app(*, language_backend=None, registry_path=None, minimum_profile_target=95.):
    from fragrance_ai.recommender.local_runtime import (
        local_profile,
        local_lotion_provider,
    )
    from fragrance_ai.recommender.formulation_core import (
        configured_formulation_core,
        SYSTEM_VERSION,
    )
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.frozen_stock_assay import FrozenStockAssay
    from deploy.web_app import create_web_app

    profile = local_profile()
    core = configured_formulation_core()
    if (
        os.environ.get("PERFUMERY_AI_ENV") != "research"
        or profile is None
        or core is None
        or core.version != SYSTEM_VERSION
        or core.sha256 != profile["formulation_core"][1]
    ):
        raise ValueError(
            "explicit accepted source-bound V76 research checkpoint required"
        )
    perfume = configured_perception()
    lotion = local_lotion_provider(configured_perception("body_lotion"))
    if perfume.core is not core or lotion.core is not core:
        raise ValueError(
            "both product pathways must use the selected single V76 checkpoint"
        )
    if (
        "physical_evidence" not in profile
        or core.manifest.get("process_graph_training") is None
    ):
        raise ValueError(
            "V76 physics evidence and trained process graph must both be connected"
        )
    stock = FrozenStockAssay(profile)
    root = Path(profile["profile_path"]).parent
    path, digest = profile["catalog"]
    app = create_web_app(
        str(Path(registry_path).resolve() if registry_path is not None else root / "benchmarks/industrial_ingredient_registry_v1.db"),
        perception_guidance=perfume,
        lotion_perception_guidance=lotion,
        stock_mixture_predictor=stock,
        catalog_manifest_path=path,
        catalog_manifest_sha256=digest,
        language_backend=language_backend,
        minimum_profile_target=minimum_profile_target,
    )
    app.state.local_stock_mixture_predictor = stock
    app.state.local_profile_sha256 = profile["profile_sha256"]
    return app
