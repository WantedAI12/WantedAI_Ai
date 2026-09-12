"""Evaluate fixed request cases and temporal-response consistency on a wheel."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    installed = output / "installed"
    with zipfile.ZipFile(args.wheel) as wheel:
        if any(
            not (installed / name).resolve().is_relative_to(installed)
            for name in wheel.namelist()
        ):
            raise ValueError("wheel path escapes destination")
        wheel.extractall(installed)
    sys.path.insert(0, str(installed))
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.evaluation import evaluate_benchmark
    from fragrance_ai.recommender.science import ATMOSPHERIC_PRESSURE_PA
    import numpy as np

    path = ROOT / "benchmarks" / "recipe_quality_cases_20260905.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    report = {
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": cases["scope"],
        "parser": [],
        "recipes": [],
    }
    with NaturalLanguagePerfumeryAI() as ai:
        for case in cases["parser_cases"]:
            limits = RecipeConstraints(**case.get("constraints", {}))
            original_limit = limits.max_ingredients
            brief = ai.parser.parse(case["brief"], limits)
            failures = []
            if limits.max_ingredients != original_limit:
                failures.append("input_constraints_mutated")
            for expected_key, values in (
                ("desired", brief.desired_dimensions),
                ("avoided", brief.avoided_dimensions),
                ("excluded", brief.excluded_ingredients),
            ):
                if not set(case.get(expected_key, [])).issubset(values):
                    failures.append(expected_key)
            if set(case.get("not_avoided", [])) & set(brief.avoided_dimensions):
                failures.append("incorrect_exclusion")
            if (
                "max_ingredients" in case
                and brief.constraints.max_ingredients != case["max_ingredients"]
            ):
                failures.append("max_ingredients")
            if "weaker_than" in case:
                weak, strong = case["weaker_than"]
                if not 0 < brief.target_profile[weak] < brief.target_profile[strong]:
                    failures.append("relative_strength")
            report["parser"].append(
                {
                    "id": case["id"],
                    "passed": not failures,
                    "failures": failures,
                    "desired": brief.desired_dimensions,
                    "avoided": brief.avoided_dimensions,
                    "excluded": brief.excluded_ingredients,
                    "max_ingredients": brief.constraints.max_ingredients,
                }
            )
        for case in cases["recipe_cases"]:
            started = time.perf_counter()
            result = ai.create_recipe(
                case["brief"],
                RecipeConstraints(
                    simulation_draws=64,
                    physics_search_population=7,
                    target_similarity=50,
                ),
                as_of=date.fromisoformat(cases["as_of"]),
            )
            failures = []
            if not result.recipe:
                failures.append("empty_recipe")
            if set(case.get("forbidden", [])) & {
                line.name for line in result.closest_candidate
            }:
                failures.append("forbidden_material")
            if len(result.recipe) > case.get(
                "max_lines", result.brief.constraints.max_ingredients
            ):
                failures.append("ingredient_count")
            if result.recipe and not math.isclose(
                sum(line.concentrate_percent for line in result.recipe),
                100,
                abs_tol=0.001,
            ):
                failures.append("formula_sum")
            if result.ingredient_temporal_profile and len(
                result.ingredient_temporal_profile
            ) != len(result.closest_candidate):
                failures.append("temporal_material_count")
            report["recipes"].append(
                {
                    "id": case["id"],
                    "passed": not failures,
                    "failures": failures,
                    "seconds": time.perf_counter() - started,
                    "status": result.status,
                    "lines": len(result.recipe),
                    "similarity": result.similarity_score,
                }
            )
        # Compare the emitted nominal odor contributions with the model's
        # stated Hill response evaluated at the current headspace concentration.
        lines = result.closest_candidate
        ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
        prepared = ai.temporal_simulator._prepare(
            lines, ingredients, ai.scientific_store
        )
        interaction = ai.temporal_simulator._interaction_matrix(prepared)
        gas = np.asarray(
            [
                item.mole_fraction
                * item.activity_coefficient
                * item.vapor_pressure_pa
                / ATMOSPHERIC_PRESSURE_PA
                * 1e6
                for item in prepared
            ]
        )
        thresholds = np.asarray([item.odor_threshold_ppm for item in prepared])
        transport = np.asarray(
            [
                ai.temporal_simulator._air_to_receptor_transport(item.properties)
                for item in prepared
            ]
        )
        half_lives = np.asarray(
            [
                ai.temporal_simulator._half_life_minutes(
                    item.ingredient, item.properties, item.vapor_pressure_pa
                )
                for item in prepared
            ]
        )
        errors = []
        for t_index, minutes in enumerate(result.temporal_timepoints_minutes):
            power = (gas * np.power(0.5, minutes / half_lives) / thresholds) ** 0.55
            activation = power / (1 + power) * transport
            suppressed = activation / (1 + 0.2 * (interaction @ activation))
            expected = 100 * suppressed / suppressed.sum()
            actual = np.asarray(
                [
                    item["points"][t_index]["odor_contribution_percent"]
                    for item in result.ingredient_temporal_profile
                ]
            )
            errors.append(float(np.max(np.abs(expected - actual))))
        report["temporal_response"] = {
            "max_contribution_error_percentage_points": max(errors),
            "passed": max(errors) < 1e-5,
        }
    report["legacy_benchmarks"] = {}
    for name in ("brief_benchmark.json", "holdout_benchmark.json"):
        report["legacy_benchmarks"][name] = evaluate_benchmark(
            ROOT / "benchmarks" / name
        )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "parser_passed": sum(row["passed"] for row in report["parser"]),
                "parser_total": len(report["parser"]),
                "recipe_passed": sum(row["passed"] for row in report["recipes"]),
                "recipe_total": len(report["recipes"]),
                "temporal_response": report["temporal_response"],
                "output": str(output / "report.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
