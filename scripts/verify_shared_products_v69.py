"""Actual default-profile API, three product domains, synthetic physical inputs."""

import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from fastapi.testclient import TestClient
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import load_verified_catalog_bundle
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.platform.unified_product_inputs import (
        UnifiedProductContext,
        UnifiedProductRequest,
        context_id,
    )
    from scripts.serve_product_runtime_v42 import create_app

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = local_profile()
    core = configured_formulation_core()
    catalog = load_verified_catalog_bundle(*profile["catalog"])["catalog"]
    items = [
        i
        for i in catalog.ingredients
        if i.formulation_ready
        and not i.blocked
        and i.active_strength_percent == 100
        and i.structure_smiles
    ][:2]
    report = {
        "checkpoint_sha256": core.sha256,
        "profile_sha256": profile["profile_sha256"],
        "physical_input_source": "synthetic_verification_not_measured_product",
        "results": [],
    }
    with TestClient(create_app(enable_language=False)) as client:
        for product in ("perfume", "body_lotion", "body_wash"):
            context = UnifiedProductContext(
                product_type=product,
                application_mass_mg_cm2=2.0,
                fragrance_concentration_percent=0.5,
                temperature_c=25.0,
                relative_humidity_percent=50.0,
                headspace_height_cm=1.0,
                stages=[
                    {
                        "stage_id": "initial",
                        "duration_minutes": 3.0,
                        "formulation_reference": "synthetic verification, not a measured formulation",
                    }
                ],
            )
            stage = {
                "stage_id": "initial",
                "coefficients": [
                    {
                        "ingredient_id": item.ingredient_id,
                        "evaporation_per_min": rate,
                        "uptake_per_min": 0.01,
                        "hydrolysis_per_min": 0.02,
                        "air_return_per_min": 0.1,
                        "ventilation_per_min": 0.2,
                        "capacity_decay_per_min": 0.03,
                        "evaporating_capacity_fraction": 0.5,
                        "nonreactive_capacity_fraction": 0.2,
                        "source_kind": "simulated",
                        "source_reference": "synthetic verification",
                    }
                    for item, rate in zip(items, (0.3, 0.02))
                ],
            }
            if product == "body_wash":
                stage.update(
                    rinse_retained_film_fractions={i.ingredient_id: 0.1 for i in items},
                    rinse_source_reference="synthetic rinse fraction",
                )
            request = UnifiedProductRequest(
                context=context,
                parameter_context_id=context_id(context),
                components=[
                    {"ingredient_id": i.ingredient_id, "concentrate_percent": 50.0}
                    for i in items
                ],
                stages=[stage],
                times_minutes=[0.0, 0.1, 1.0, 2.0, 3.0],
            )
            result = client.post(
                "/v1/applications/unified/predict", json=request.model_dump(mode="json")
            )
            assert result.status_code == 200, result.text
            value = result.json()
            assert value["model"]["shared_odor_backbone_sha256"] == core.sha256
            error = value["diagnostics"]["mass_balance_max_abs_error_mg_cm2"]
            assert error < 1e-12
            assert all(
                r["air_exposure_mg_min_m3"] > 0
                for r in value["integrated_exposure"]["materials"]
            )
            (args.output / (product + ".json")).write_text(
                json.dumps(
                    {"request": request.model_dump(mode="json"), "response": value},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            report["results"].append(
                {
                    "product": product,
                    "http_status": 200,
                    "status": value["status"],
                    "mass_error_mg_cm2": error,
                    "same_checkpoint": True,
                }
            )
    (args.output / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
