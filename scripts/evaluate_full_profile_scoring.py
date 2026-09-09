"""Reproducible real-generator scoring diagnostic; not human validation.

Runs without outbound network or public holdout outcomes. The corpus is
explicitly absent, so source and installed-wheel runs use identical inputs.
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
CASES = (
    "clean fresh citrus woody",
    "floral fruity woody",
    "green aromatic woody",
    "gourmand vanilla woody",
    "soft clean musk",
    "smoky earthy woody",
    "opening citrus, drydown woody musk",
    "opening no sweetness, drydown woody musk",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-package", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory to preserve previous evidence")
    args.output.mkdir(parents=True)
    attempts = []

    def guard(event, arguments):
        if event in ("socket.connect", "socket.getaddrinfo"):
            attempts.append(event)
            raise RuntimeError("network is disabled for local scoring verification")
        if event == "open" and isinstance(arguments[0], str) and "actualvalue" in arguments[0].casefold():
            raise RuntimeError("holdout outcomes are forbidden in generator scoring work")

    sys.addaudithook(guard)
    sys.path.insert(0, str((args.installed_package or ROOT).resolve()))
    import fragrance_ai
    from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus

    package_root = Path(fragrance_ai.__file__).resolve().parent
    if args.installed_package and not package_root.is_relative_to(args.installed_package.resolve()):
        raise RuntimeError("the requested installed wheel was not imported")
    report = {
        "scope": "actual generator integration; model-profile self-score, not actual human similarity",
        "target_similarity_unchanged": 90.0,
        "as_of": "2026-09-05",
        "historical_reference_corpus": "explicitly absent in both source and wheel",
        "actual_human_accuracy_90_authorized": False,
        "imported_package": str(package_root),
        "cases": [],
    }
    for index, brief in enumerate(CASES):
        payloads, durations = {}, {}
        for mode in ("compatibility", "strict"):
            started = time.perf_counter()
            with NaturalLanguagePerfumeryAI(
                corpus=HistoricalReferenceCorpus(args.output / "absent-reference.db"),
                require_full_profile_match=mode == "strict",
            ) as ai:
                payload = ai.create_recipe(brief, RecipeConstraints(), as_of=date(2026, 9, 5)).to_dict()
            durations[mode] = time.perf_counter() - started
            payloads[mode] = payload
            (args.output / f"{index:02d}_{mode}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
            )
            assert payload["brief"]["constraints"]["target_similarity"] == 90
            assert payload["actual_olfactory_similarity_score"] is None
            assert not payload["human_similarity_90_claim_authorized"]
            assert not payload["score_contract"]["actual_human_90_proven_by_this_score"]
            if payload["recipe"]:
                assert payload["safety"]["internal_gate_passed"]
                assert abs(sum(line["concentrate_percent"] for line in payload["recipe"]) - 100) < .001

        compat, strict = payloads["compatibility"], payloads["strict"]
        assessment = compat["full_profile_assessment"]
        search = assessment.get("search", {})
        assert compat["formula_id"] == strict["formula_id"]
        assert compat["calculated_profile_similarity"] == strict["calculated_profile_similarity"]
        assert compat["similarity_score"] == strict["legacy_preference_score"]
        assert strict["similarity_score"] == (strict["calculated_profile_similarity"] or 0.)
        if not strict["full_profile_target_met"]:
            assert strict["recipe"] == [] and strict["status"] == "no_safe_match"
            if strict["manufacturing_plan"] is not None:
                assert not strict["manufacturing_plan"]["ready_for_lab_trial"]
        if strict["recipe"]:
            assert strict["calculated_profile_similarity"] + 1e-8 >= 90
            assert compat["recipe"], "a strict wrapper cannot promote a previously blocked recipe"
        before, after = search.get("baseline_score"), search.get("selected_score")
        if before is not None and after is not None:
            assert after + 1e-8 >= before
            assert abs(assessment["score"] - after) < .001
        row = {
            "brief": brief, "formula_id": compat["formula_id"],
            "compatibility_status": compat["status"], "strict_status": strict["status"],
            "legacy_preference_score": compat["similarity_score"],
            "calculated_profile_similarity": assessment["score"],
            "full_profile_target_met": assessment["target_met"],
            "nominal_full_profile_score": assessment["nominal"]["score"],
            "time_weighted_full_profile_score": assessment["temporal_mean_score"],
            "worst_time_full_profile_score": assessment["temporal_minimum_score"],
            "before_full_profile_search": before, "after_full_profile_search": after,
            "search_gain_points": None if before is None or after is None else after - before,
            "recipe_changed_by_search": search.get("candidate_changed", False),
            "extra_search_variants": search.get("additional_variants", 0),
            "compatibility_ingredient_count": len(compat["recipe"]),
            "strict_ingredient_count": len(strict["recipe"]),
            "closest_candidate_ingredient_count": len(strict["closest_candidate"]),
            "safety_gate_passed": compat["safety"]["internal_gate_passed"],
            "seconds": durations,
        }
        report["cases"].append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    report["case_count"] = len(report["cases"])
    report["computed_score_case_count"] = sum(row["calculated_profile_similarity"] is not None for row in report["cases"])
    report["full_profile_target_met_count"] = sum(row["full_profile_target_met"] for row in report["cases"])
    report["strict_recipe_returned_count"] = sum(row["strict_ingredient_count"] > 0 for row in report["cases"])
    report["cases_improved_by_search"] = sum((row["search_gain_points"] or 0) > 1e-6 for row in report["cases"])
    report["network_attempts"] = attempts
    report["runtime_code_sha256"] = {
        name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        for name in ("recommender/profile_match.py", "recommender/service.py", "recommender/optimizer.py",
                     "recommender/perception_guidance.py", "cli.py", "api_cli.py", "platform/workspace.py", "platform/worker.py")
    }
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["output_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("[0-9][0-9]_*.json"))
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
