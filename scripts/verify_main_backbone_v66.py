"""Actual pinned V66 inference/lineage verification, not human validation."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    from fragrance_ai.recommender.local_runtime import (
        local_profile,
        local_atlas_provider,
        local_odor_backbone_provider,
    )
    from fragrance_ai.research.atlas_profiles import predict_atlas

    profile = local_profile()
    parent, model = local_atlas_provider(), local_odor_backbone_provider()
    assert (
        parent.sha256 != model.sha256
        and model.artifact_version == "structured-multioutput-atlas/v66"
    )
    report = {
        "scope": "local_pinned_model_integration_not_new_human_validation",
        "profile_sha256": profile["profile_sha256"],
        "stock_parent_sha256": parent.sha256,
        "odor_backbone_sha256": model.sha256,
        "parity": {},
    }
    for name, predictor in (("legacy", parent), ("v66", model)):
        query = np.tile(predictor.models["applicability"]["support"], (8, 1))

        def reference():
            return {h: predict_atlas(m, query) for h, m in predictor.models.items()}

        expected, actual = reference(), predictor._compiled.predict(query)
        maximum = max(float(np.max(np.abs(expected[h] - actual[h]))) for h in expected)
        for h in expected:
            np.testing.assert_allclose(actual[h], expected[h], rtol=1e-11, atol=1e-11)
        timing = {"reference": [], "compiled": []}
        # Alternate the order to reduce a consistent warm-up advantage.
        for i in range(8):
            operations = [
                ("reference", reference),
                ("compiled", lambda: predictor._compiled.predict(query)),
            ]
            for label, fn in operations[:: 1 if i % 2 else -1]:
                start = time.perf_counter()
                fn()
                timing[label].append(time.perf_counter() - start)
        report["parity"][name] = {
            "rows": len(query),
            "endpoints_per_head": len(predictor.endpoints),
            "maximum_absolute_difference": maximum,
            "compiled_kernel_groups": len(predictor._compiled.groups),
            "seconds": timing,
            "median_seconds": {k: float(np.median(v)) for k, v in timing.items()},
        }
    old_ref = json.loads(
        (ROOT / ".benchmarks/lotion_reference_v59/training-01/model.json").read_text(
            encoding="utf-8"
        )
    )
    new_ref = json.loads(
        Path(profile["lotion_target_reference"][0]).read_text(encoding="utf-8")
    )
    for key in (
        "profiles",
        "background",
        "concept_metadata",
        "endpoints",
        "source",
        "source_stimuli",
    ):
        assert new_ref[key] == old_ref[key], "reference target changed: " + key
    assert new_ref["parent_atlas_sha256"] == model.sha256
    report["reference_targets_exactly_unchanged"] = True
    report["reference_rebuilt_from_same_observations_not_fitted_to_recipe_results"] = (
        True
    )
    from fragrance_ai.recommender.runtime import load_verified_catalog_bundle
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.unified_product import (
        configured_unified_product,
        UnifiedProductPredictor,
    )
    from fragrance_ai.platform.unified_product_inputs import (
        UnifiedProductContext,
        UnifiedProductRequest,
        context_id,
    )

    bundle = load_verified_catalog_bundle(*profile["catalog"])
    catalog = bundle["catalog"]
    provider = configured_perception()
    unified = configured_unified_product(catalog, component_provider=provider)
    baseline = UnifiedProductPredictor(
        unified.transport, parent, provider.structures, catalog
    )
    items = [
        r
        for r in catalog.ingredients
        if r.formulation_ready
        and not r.blocked
        and r.active_strength_percent == 100
        and r.structure_smiles
    ][:2]
    assert len(items) == 2
    report["catalog_rows"], report["products"] = len(catalog.ingredients), []
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
                    "formulation_reference": "synthetic integration coefficients",
                }
            ],
        )
        coefficients = [
            {
                "ingredient_id": row.ingredient_id,
                "evaporation_per_min": e,
                "uptake_per_min": 0.01,
                "hydrolysis_per_min": 0.0,
                "air_return_per_min": 0.1,
                "ventilation_per_min": 0.2,
                "capacity_decay_per_min": 0.03,
                "evaporating_capacity_fraction": 0.5,
                "nonreactive_capacity_fraction": 0.2,
                "source_kind": "simulated",
                "source_reference": "numerical integration verification, not measured product",
            }
            for row, e in zip(items, (0.3, 0.02))
        ]
        stage = {"stage_id": "initial", "coefficients": coefficients}
        if product == "body_wash":
            stage.update(
                rinse_retained_film_fractions={r.ingredient_id: 0.1 for r in items},
                rinse_source_reference="simulated rinse",
            )
        request = UnifiedProductRequest(
            context=context,
            parameter_context_id=context_id(context),
            components=[
                {"ingredient_id": r.ingredient_id, "concentrate_percent": 50.0}
                for r in items
            ],
            stages=[stage],
            times_minutes=[0.0, 0.1, 1.0, 2.0, 3.0],
        )
        old, new = baseline.predict(request), unified.predict(request)
        assert new["status"] == "research_prediction"
        assert new["model"]["shared_odor_backbone_sha256"] == model.sha256
        assert new["model"]["transport_training_atlas_sha256"] == parent.sha256
        for before, after in zip(old["temporal_profile"], new["temporal_profile"]):
            assert before["materials"] == after["materials"]
            assert (
                before["total_air_concentration_mg_m3"]
                == after["total_air_concentration_mg_m3"]
            )
        save(product + ".json", new)
        report["products"].append(
            {
                "product": product,
                "status": new["status"],
                "transport_exactly_unchanged": True,
                "new_odor_backbone_connected": True,
                "mass_balance_max_abs_error_mg_cm2": new["diagnostics"][
                    "mass_balance_max_abs_error_mg_cm2"
                ],
                "coefficient_source": "synthetic_not_measured",
                "human_similarity_percent": None,
            }
        )
    save("report.json", report)
    print(
        json.dumps(
            {
                **report,
                "parity": {
                    k: {a: b for a, b in v.items() if a != "seconds"}
                    for k, v in report["parity"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
