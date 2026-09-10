"""Exercise actual recipe generation with and without learned research guidance.

This is an integration/objective diagnostic, not independent human validation.
It neither reads public holdout outcomes nor calls a cloud/LLM endpoint.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = (
    "floral fruity woody", "clean fresh citrus woody", "green aromatic woody",
    "gourmand vanilla woody", "smoky earthy woody", "soft clean musk",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=len(CASES))
    parser.add_argument("--model", type=Path, default=ROOT / ".benchmarks/conditional_profiles_v2/final-01/models.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "benchmarks/industrial_ingredient_registry_v1.db")
    parser.add_argument("--solvent", default="pg")
    parser.add_argument("--installed-package", type=Path, help="verify a fresh installed wheel instead of source imports")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_cases <= len(CASES):
        parser.error("use a new output directory and a valid case count")
    args.output.mkdir(parents=True)
    outbound = []

    def guard(event, arguments):
        if event in ("socket.connect", "socket.getaddrinfo"):
            outbound.append(event)
            raise RuntimeError("network disabled for local recipe integration benchmark")
        if event == "open" and isinstance(arguments[0], str) and "actualvalue" in arguments[0].casefold():
            raise RuntimeError("holdout outcomes are forbidden in recipe integration work")
    sys.addaudithook(guard)
    if args.installed_package:
        sys.path.insert(0, str(args.installed_package.resolve()))
    import fragrance_ai
    from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.perception_guidance import PerceptionGuidance
    if args.installed_package and not Path(fragrance_ai.__file__).resolve().is_relative_to(args.installed_package.resolve()):
        raise RuntimeError("requested installed package was not imported")
    provider = PerceptionGuidance(args.model, args.registry, solvent=args.solvent, experimental=True)
    report = {"scope": "actual generator integration; self-scored model objective only, not human accuracy",
              "solvent_scenario": args.solvent, "target_similarity_unchanged": 90.0, "cases": [],
              "imported_package": str(Path(fragrance_ai.__file__).resolve())}
    for index, brief in enumerate(CASES[:args.max_cases]):
        durations, payloads = {}, {}
        for name, guidance in (("baseline", None), ("guided", provider)):
            started = time.perf_counter()
            with NaturalLanguagePerfumeryAI(perception_guidance=guidance) as ai:
                result = ai.create_recipe(brief, RecipeConstraints(), as_of=date(2026, 9, 5))
            durations[name] = time.perf_counter() - started
            payload = result.to_dict()
            payloads[name] = payload
            (args.output / f"{index:02d}_{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            if payload["actual_olfactory_similarity_score"] is not None or payload["human_similarity_90_claim_authorized"]:
                raise AssertionError("research guidance incorrectly manufactured human evidence")
        baseline, guided = payloads["baseline"], payloads["guided"]
        info = guided.get("perception_guidance", {})
        row = {"brief": brief, "baseline_status": baseline["status"], "guided_status": guided["status"],
               "baseline_formula_id": baseline["formula_id"], "guided_formula_id": guided["formula_id"],
               "recipe_changed": baseline["formula_id"] != guided["formula_id"],
               "baseline_semantic_score": baseline["similarity_score"], "guided_semantic_score": guided["similarity_score"],
               "guided_ingredient_count": len(guided["recipe"]), "seconds": durations,
               "guidance_status": info.get("status"), "objective_calls": info.get("grid_objective_calls", 0),
               "exact_evaluations": info.get("exact_formula_evaluations", 0),
               "guidance_variants_added": info.get("guidance_variants_added", 0),
               "baseline_learned_score": (info.get("baseline") or {}).get("score"),
               "guided_learned_score": (info.get("selected") or {}).get("score"),
               "safety_gate_passed": guided["safety"]["internal_gate_passed"],
               "actual_human_accuracy_90_authorized": False}
        report["cases"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report["network_attempts"] = outbound
    report["cases_with_changed_recipe"] = sum(row["recipe_changed"] for row in report["cases"])
    report["code_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "fragrance_ai/recommender/perception_guidance.py", "fragrance_ai/recommender/service.py", "fragrance_ai/cli.py")}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
