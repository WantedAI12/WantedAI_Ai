"""Reproducible semantic-gate benchmark evaluation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import mean

from .service import NaturalLanguagePerfumeryAI
from .models import RecipeConstraints


def evaluate_benchmark(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    as_of = date.fromisoformat(payload.get("as_of", "2026-07-11"))
    ai = NaturalLanguagePerfumeryAI()

    passed = 0
    parser_hits = 0
    expected_dimension_count = 0
    avoided_hits = 0
    expected_avoided_count = 0
    ready_scores: list[float] = []
    realism_scores: list[float] = []
    unsafe_leaks: list[str] = []
    invariant_failures: list[str] = []
    full_case_passes = 0
    expected_ready_count = 0
    evidenced_reference_gate_passes = 0
    evidenced_reference_gate_abstentions: list[str] = []
    evidenced_p05_values: list[float] = []
    case_results = []

    blocked_names = {item.name for item in ai.catalog.ingredients if item.blocked}

    for case in payload["cases"]:
        result = ai.create_recipe(
            case["brief"],
            constraints=RecipeConstraints(require_simulation_pass=False),
            as_of=as_of,
        )
        expected_ready = bool(case["expect_recipe"])
        if expected_ready:
            expected_ready_count += 1
        actual_ready = result.status == "prototype_ready"
        classification_passed = actual_ready == expected_ready
        passed += int(classification_passed)

        expected_dimensions = set(case.get("expected_dimensions", []))
        found_dimensions = set(result.brief.desired_dimensions)
        parser_hits += len(expected_dimensions & found_dimensions)
        expected_dimension_count += len(expected_dimensions)
        expected_avoided = set(case.get("expected_avoided", []))
        found_avoided = set(result.brief.avoided_dimensions)
        avoided_hits += len(expected_avoided & found_avoided)
        expected_avoided_count += len(expected_avoided)

        case_failures: list[str] = []
        if not classification_passed:
            case_failures.append("status")
        if not expected_dimensions.issubset(found_dimensions):
            case_failures.append("desired_dimensions")
        if not expected_avoided.issubset(found_avoided):
            case_failures.append("avoided_dimensions")

        if actual_ready:
            ready_scores.append(result.similarity_score)
            leaked = blocked_names.intersection(line.name for line in result.recipe)
            unsafe_leaks.extend(sorted(leaked))
            forbidden = set(case.get("forbidden_ingredients", []))
            used = {line.name for line in result.recipe}
            if forbidden & used:
                case_failures.append("forbidden_ingredient")
            if (
                abs(sum(line.concentrate_percent for line in result.recipe) - 100.0)
                > 0.01
            ):
                case_failures.append("formula_sum")
            if not result.safety.internal_gate_passed:
                case_failures.append("safety_gate")
            if result.similarity_score < float(case.get("minimum_similarity", 90.0)):
                case_failures.append("similarity")
            realism_scores.append(result.realism_score)
            if result.realism_score < float(case.get("minimum_realism", 65.0)):
                case_failures.append("realism")
            if (
                result.simulation_status == "evidenced_nonhuman_pass"
                and result.physsim_comparison_authorized
            ):
                evidenced_reference_gate_passes += 1
                evidenced_p05_values.append(result.simulation_p05)
            else:
                evidenced_reference_gate_abstentions.append(case["brief"])

        for field, expected in case.get("expected_constraints", {}).items():
            actual = getattr(result.brief.constraints, field)
            if actual != expected:
                case_failures.append(f"constraint:{field}")

        if case_failures:
            invariant_failures.append(f"{case['brief']}: {','.join(case_failures)}")
        else:
            full_case_passes += 1

        case_results.append(
            {
                "brief": case["brief"],
                "expected_ready": expected_ready,
                "actual_status": result.status,
                "similarity": result.similarity_score,
                "simulation_status": result.simulation_status,
                "simulation_p05": result.simulation_p05,
                "model_applicability_percent": result.model_applicability_percent,
                "simulation_only_approved": result.simulation_only_approved,
                "passed": not case_failures,
                "failures": case_failures,
            }
        )

    total = len(payload["cases"])
    return {
        "benchmark_version": payload["benchmark_version"],
        "metric": payload["metric"],
        "cases": total,
        "semantic_gate_classification_accuracy": round(passed / total * 100.0, 4),
        "parser_dimension_recall": round(
            parser_hits / max(1, expected_dimension_count) * 100.0, 4
        ),
        "parser_avoidance_recall": (
            round(avoided_hits / expected_avoided_count * 100.0, 4)
            if expected_avoided_count
            else 100.0
        ),
        "full_case_pass_rate": round(full_case_passes / total * 100.0, 4),
        "mean_ready_similarity": round(mean(ready_scores), 4) if ready_scores else 0.0,
        "minimum_ready_similarity": round(min(ready_scores), 4)
        if ready_scores
        else 0.0,
        "mean_ready_realism": round(mean(realism_scores), 4) if realism_scores else 0.0,
        "minimum_ready_realism": round(min(realism_scores), 4)
        if realism_scores
        else 0.0,
        "blocked_material_leaks": sorted(set(unsafe_leaks)),
        "evidenced_reference_gate_pass_rate_on_semantic_ready": round(
            evidenced_reference_gate_passes / max(1, expected_ready_count) * 100.0,
            4,
        ),
        "evidenced_reference_gate_abstentions": (evidenced_reference_gate_abstentions),
        "minimum_passing_evidenced_p05": (
            round(min(evidenced_p05_values), 4) if evidenced_p05_values else None
        ),
        "invariant_failures": invariant_failures,
        "results": case_results,
        "disclaimer": (
            "Text-only cases exercise semantic candidate generation. The "
            "evidenced-reference gate abstains unless an independent quantitative "
            "target is registered; neither endpoint is measured perfume similarity."
        ),
    }
