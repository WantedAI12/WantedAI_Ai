#!/usr/bin/env python
"""Evaluate system components without making sensory-accuracy claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints  # noqa: E402
from fragrance_ai.recommender.brief_parser import BriefParseError  # noqa: E402


CONFIGURATIONS = {
    "full": {},
    "keyword_rules_only": {"enable_semantic_ontology": False},
    "two_candidate_search": {"physics_search_population": 2},
    "no_measured_concentration_response": {"enable_concentration_response": False},
    "no_learned_r2": {"enable_learned_r2": False},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=Path,
        default=PROJECT_ROOT / "benchmarks/system_ablation_benchmark.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "benchmarks/system_ablation_report.json",
    )
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    as_of = date.fromisoformat(benchmark["as_of"])
    ai = NaturalLanguagePerfumeryAI()
    results = {}
    for name, overrides in CONFIGURATIONS.items():
        cases = []
        for case in benchmark["cases"]:
            constraints = RecipeConstraints(
                require_simulation_pass=False,
                simulation_draws=64,
                **overrides,
            )
            try:
                result = ai.create_recipe(case["brief"], constraints, as_of=as_of)
                desired = set(result.brief.desired_dimensions)
                avoided = set(result.brief.avoided_dimensions)
                expected = set(case.get("expected_dimensions", []))
                expected_avoided = set(case.get("expected_avoided", []))
                cases.append({
                    "brief": case["brief"],
                    "parsed": True,
                    "dimension_hits": len(expected & desired),
                    "dimension_total": len(expected),
                    "avoidance_hits": len(expected_avoided & avoided),
                    "avoidance_total": len(expected_avoided),
                    "status": result.status,
                    "semantic_similarity": result.raw_similarity_score,
                    "simulation_p05": result.simulation_p05,
                    "physics_objective": result.physics_search_objective,
                    "variants": result.candidate_variants_evaluated,
                    "physsim_similarity": result.physsim_similarity_score,
                })
            except BriefParseError:
                cases.append({
                    "brief": case["brief"], "parsed": False,
                    "dimension_hits": 0,
                    "dimension_total": len(case.get("expected_dimensions", [])),
                    "avoidance_hits": 0,
                    "avoidance_total": len(case.get("expected_avoided", [])),
                    "status": "parse_failed",
                    "semantic_similarity": 0.0, "simulation_p05": 0.0,
                    "physics_objective": 0.0, "variants": 0,
                    "physsim_similarity": 0.0,
                })
        dimension_hits = sum(row["dimension_hits"] for row in cases)
        dimension_total = sum(row["dimension_total"] for row in cases)
        avoidance_hits = sum(row["avoidance_hits"] for row in cases)
        avoidance_total = sum(row["avoidance_total"] for row in cases)
        parsed = [row for row in cases if row["parsed"]]
        results[name] = {
            "cases": cases,
            "parse_rate_percent": len(parsed) / len(cases) * 100.0,
            "dimension_recall_percent": dimension_hits / max(1, dimension_total) * 100.0,
            "avoidance_recall_percent": avoidance_hits / max(1, avoidance_total) * 100.0,
            "mean_semantic_similarity": mean(row["semantic_similarity"] for row in parsed) if parsed else 0.0,
            "mean_simulation_p05": mean(row["simulation_p05"] for row in parsed) if parsed else 0.0,
            "mean_physics_objective": mean(row["physics_objective"] for row in parsed) if parsed else 0.0,
            "mean_variants_evaluated": mean(row["variants"] for row in parsed) if parsed else 0.0,
            "mean_physsim_similarity": mean(row["physsim_similarity"] for row in parsed) if parsed else 0.0,
        }
    full = results["full"]
    effects = {}
    for name, result in results.items():
        if name == "full":
            continue
        effects[name] = {
            "dimension_recall_delta_vs_full": result["dimension_recall_percent"] - full["dimension_recall_percent"],
            "simulation_p05_delta_vs_full": result["mean_simulation_p05"] - full["mean_simulation_p05"],
            "physics_objective_delta_vs_full": result["mean_physics_objective"] - full["mean_physics_objective"],
            "physsim_similarity_delta_vs_full": result["mean_physsim_similarity"] - full["mean_physsim_similarity"],
        }
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "claim": benchmark["claim"],
        "configurations": CONFIGURATIONS,
        "results": results,
        "effects": effects,
        "pass": (
            full["parse_rate_percent"] == 100.0
            and full["dimension_recall_percent"] == 100.0
            and full["avoidance_recall_percent"] == 100.0
            and results["keyword_rules_only"]["dimension_recall_percent"] < full["dimension_recall_percent"]
        ),
        "claim_boundary": "Ablations quantify internal proxy behavior only and cannot estimate human olfactory accuracy.",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "summary": {
        name: {key: value for key, value in result.items() if key != "cases"}
        for name, result in results.items()
    }, "effects": effects, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
