"""Verify actual pinned CPU inference and product/API integration, without deployment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def main():
    from fastapi.testclient import TestClient
    from fragrance_ai.recommender.local_runtime import (
        local_profile,
        local_odor_backbone_provider,
        local_atlas_provider,
    )
    from fragrance_ai.research.atlas_profiles import (
        AtlasProfilePredictor,
        predict_atlas,
    )
    from fragrance_ai.recommender.runtime import load_verified_catalog_bundle
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.unified_product import configured_unified_product
    from fragrance_ai.platform.unified_product_inputs import (
        UnifiedProductContext,
        UnifiedProductRequest,
        context_id,
    )
    from scripts.serve_product_runtime_v42 import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

    profile = local_profile()
    model, parent = local_odor_backbone_provider(), local_atlas_provider()
    if (
        model.artifact_version != "scientific-molecular-atlas/v68"
        or not model.development_nonregression_passed
    ):
        raise ValueError("an accepted, explicitly selected V68 checkpoint is required")
    old = AtlasProfilePredictor(
        ROOT / ".benchmarks/main_backbone_v66/train-02/model.json",
        sha256="1b5af57f7902b4efeee2ab991156623adb8f966c9340d7f456ef3c3d790e1cc9",
        experimental=True,
    )
    report = {
        "scope": "actual_local_cpu_integration_not_deployment_or_new_sensory_study",
        "profile_sha256": profile["profile_sha256"],
        "odor_backbone_sha256": model.sha256,
        "frozen_stock_parent_sha256": parent.sha256,
        "parity": {},
        "products": [],
        "api": [],
        "external_language_model_enabled": False,
        "human_similarity_percent": None,
    }
    query = np.tile(model.models["applicability"]["support"], (8, 1))
    for name, predictor in (("v66", old), ("v68", model)):
        expected = {
            head: predict_atlas(head_model, query)
            for head, head_model in predictor.models.items()
        }
        actual, diagnostics = predictor._compiled.predict_with_diagnostics(query)
        difference = max(
            float(np.max(np.abs(actual[head] - expected[head]))) for head in expected
        )
        for head in expected:
            np.testing.assert_allclose(
                actual[head], expected[head], atol=1e-10, rtol=1e-10
            )
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            predictor._compiled.predict_with_diagnostics(query)
            timings.append(time.perf_counter() - start)
        report["parity"][name] = {
            "rows": len(query),
            "outputs_per_head": len(predictor.endpoints),
            "maximum_absolute_difference": difference,
            "median_seconds": float(np.median(timings)),
            "head_kernel_groups": len(predictor._compiled.groups),
            "seconds": timings,
            "diagnostics_per_head": {
                head: len(rows) for head, rows in diagnostics.items()
            },
        }
    before_reference = json.loads(
        (ROOT / ".benchmarks/main_backbone_v66/reference-01/model.json").read_bytes()
    )
    after_reference = json.loads(
        Path(profile["lotion_target_reference"][0]).read_bytes()
    )
    if after_reference.pop("parent_atlas_sha256") != model.sha256:
        raise ValueError("reference backbone binding mismatch")
    before_reference.pop("parent_atlas_sha256")
    if before_reference != after_reference:
        raise ValueError(
            "observed target values must not be changed to improve results"
        )
    report["reference_values_exactly_unchanged"] = True
    bundle = load_verified_catalog_bundle(*profile["catalog"])
    catalog = bundle["catalog"]
    report["catalog_rows"] = len(catalog.ingredients)
    report["catalog_manifest_sha256"] = profile["catalog"][1]
    predictor = configured_unified_product(
        catalog, component_provider=configured_perception()
    )
    items = [
        row
        for row in catalog.ingredients
        if row.formulation_ready
        and not row.blocked
        and row.active_strength_percent == 100
        and row.structure_smiles
    ][:2]
    if len(items) != 2:
        raise ValueError("two actual admissible explicit molecular identities required")
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
                    "formulation_reference": "numerical coefficient fixture, not a measured product",
                }
            ],
        )
        coefficients = [
            {
                "ingredient_id": row.ingredient_id,
                "evaporation_per_min": rate,
                "uptake_per_min": 0.01,
                "hydrolysis_per_min": 0.02,
                "air_return_per_min": 0.1,
                "ventilation_per_min": 0.2,
                "capacity_decay_per_min": 0.03,
                "evaporating_capacity_fraction": 0.5,
                "nonreactive_capacity_fraction": 0.2,
                "source_kind": "simulated",
                "source_reference": "synthetic verification coefficients",
            }
            for row, rate in zip(items, (0.3, 0.02))
        ]
        stage = {"stage_id": "initial", "coefficients": coefficients}
        if product == "body_wash":
            stage.update(
                rinse_retained_film_fractions={row.ingredient_id: 0.1 for row in items},
                rinse_source_reference="synthetic rinse fixture",
            )
        request = UnifiedProductRequest(
            context=context,
            parameter_context_id=context_id(context),
            components=[
                {"ingredient_id": row.ingredient_id, "concentrate_percent": 50.0}
                for row in items
            ],
            stages=[stage],
            times_minutes=[0.0, 0.1, 1.0, 2.0, 3.0],
        )
        result = predictor.predict(request)
        assert result["status"] == "research_prediction"
        assert result["model"]["shared_odor_backbone_sha256"] == model.sha256
        assert all(
            row["diagnostics"] for row in result["molecular_model_applicability"]
        )
        assert result["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
        assert all(
            row["air_exposure_mg_min_m3"] > 0
            for row in result["integrated_exposure"]["materials"]
        )
        save(product + ".json", result)
        report["products"].append(
            {
                "product": product,
                "status": result["status"],
                "backbone": result["model"]["odor_backbone_version"],
                "integrals_connected": True,
                "mass_balance_max_abs_error_mg_cm2": result["diagnostics"][
                    "mass_balance_max_abs_error_mg_cm2"
                ],
                "coefficient_source": "simulated_not_measured",
                "material_transition_evaluations": result["diagnostics"][
                    "transport_material_transitions"
                ],
            }
        )
    app = create_app(enable_language=False)
    with TestClient(app) as client:
        capabilities = client.get("/v1/ai/capabilities")
        assert capabilities.status_code == 200, capabilities.text
        assert (
            capabilities.json()["unified_product_model"]["shared_odor_backbone_sha256"]
            == model.sha256
        )
        save("capabilities.json", capabilities.json())
        previous = json.loads(
            (ROOT / ".benchmarks/main_backbone_v66/api-01/api-report.json").read_bytes()
        )
        for index, row in enumerate(previous["results"]):
            start = time.perf_counter()
            response = client.post(row["path"], json=row["request"])
            seconds = time.perf_counter() - start
            assert response.status_code == 200, response.text
            value = response.json()
            repeated = client.post(row["path"], json=row["request"])
            assert repeated.status_code == 200 and repeated.json() == value
            save(f"recipe-{index}.json", value)
            report["api"].append(
                {
                    "path": row["path"],
                    "request": row["request"],
                    "http_status": 200,
                    "seconds": seconds,
                    "status": value["status"],
                    "repeat_equal": True,
                    "score": value.get(
                        "calculated_profile_similarity", value.get("score")
                    ),
                    "previous_v66_score": row["score"],
                    "previous_v66_status": row["status"],
                    "cache": repeated.headers.get("X-Perfumery-Cache")
                    or repeated.headers.get("X-Perfumery-Lotion-Cache"),
                }
            )
            print(json.dumps(report["api"][-1], ensure_ascii=False), flush=True)
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "fragrance_ai/research/atlas_inference.py",
            ROOT / "fragrance_ai/recommender/exposure_transport.py",
            ROOT / "fragrance_ai/recommender/unified_product.py",
        )
    }
    save("report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
