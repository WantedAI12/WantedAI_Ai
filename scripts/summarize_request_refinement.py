"""Merge a complete bounded audit with explicit fallback-path reruns.

Successful early-exit paths may be carried forward, never failed requests.
Every row retains its execution provenance; this is not 400 new executions.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-8


def combine(baseline, updated, previous):
    for report in (baseline, updated, previous):
        if report["target_unchanged"] != 90 or report["experimental_disable_safety"] or report["regressions"]:
            raise ValueError("incompatible or regressed audit")
        if len({case["id"] for case in report["cases"]}) != len(report["cases"]):
            raise ValueError("duplicate case IDs")
        if report["case_count"] != len(report["cases"]) or report["catalog_scope"] != baseline["catalog_scope"]:
            raise ValueError("case count or catalog scope changed")
    if baseline["case_count"] != 400 or previous["case_count"] != 400:
        raise ValueError("complete unchanged 400-request baseline required")
    old = {case["id"]: case for case in previous["cases"]}
    latest = {case["id"]: case for case in updated["cases"]}
    if {case["id"] for case in baseline["cases"]} != set(old):
        raise ValueError("complete request denominator changed")
    if set(latest) - set(old):
        raise ValueError("unknown updated request")
    allowed_changes = {"fragrance_ai/recommender/dose_refinement.py", "fragrance_ai/recommender/service.py"}
    for name in set(baseline["source_sha256"]) | set(updated["source_sha256"]):
        if name not in allowed_changes and baseline["source_sha256"].get(name) != updated["source_sha256"].get(name):
            raise ValueError("a scored input or unchanged runtime component changed")
    rows, control_ids, carried_ids = [], [], []
    for base in baseline["cases"]:
        reference = old[base["id"]]
        if base["brief"] != reference["brief"] or not same(base["before_score"], reference["after_score"]):
            raise ValueError("V9 baseline or request changed")
        fresh = latest.get(base["id"])
        if fresh is None:
            if not base["full_profile_target_met"]:
                raise ValueError("a failed or unscorable request was not rerun")
            row = dict(base)
            row["execution_provenance"] = "carried_forward_unchanged_successful_early_exit_path"
            carried_ids.append(base["id"])
        else:
            if fresh["brief"] != base["brief"] or fresh.get("expected_dimensions") != base.get("expected_dimensions"):
                raise ValueError("request intent changed")
            if not same(fresh["before_score"], reference["after_score"]):
                raise ValueError("current V9 prefix differs")
            if base["after_score"] is not None and (fresh["after_score"] is None or fresh["after_score"] + 1e-8 < base["after_score"]):
                raise ValueError("bounded result regressed")
            if base["full_profile_target_met"]:
                if not same(base["after_score"], fresh["after_score"]) or base["after_formula_id"] != fresh["after_formula_id"]:
                    raise ValueError("successful early-exit control changed")
                control_ids.append(base["id"])
            row = dict(fresh)
            row["execution_provenance"] = "fresh_current_source_generator_call"
        score = row["after_score"]
        if score is not None and (isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError("invalid final score")
        if bool(row["full_profile_target_met"]) != (score is not None and score + 1e-8 >= 90):
            raise ValueError("score and pass flag disagree")
        row["bounded_score"] = base["after_score"]
        row["additional_gain_points"] = None if score is None or base["after_score"] is None else score - base["after_score"]
        rows.append(row)
    if len(control_ids) < 10:
        raise ValueError("at least ten unchanged-pass controls required")
    scores = [row["after_score"] for row in rows if row["after_score"] is not None]
    return {
        "scope": "unchanged finite 400-request development cohort; incremental validation, not an independent holdout or 400 fresh current-source calls",
        "target_unchanged": 90, "case_count": len(rows),
        "catalog_scope": baseline["catalog_scope"], "experimental_disable_safety": False,
        "actual_human_accuracy_claim": False,
        "validation": {"complete_bounded_actual_requests": 400, "current_source_actual_requests": len(updated["cases"]),
                       "successful_early_exit_controls": control_ids, "carried_successful_paths": carried_ids,
                       "all_nonpassing_requests_rerun": True, "all_400_freshly_rerun_with_current_source": False},
        "calculable_case_count": len(scores), "full_profile_90_case_count": sum(row["full_profile_target_met"] for row in rows),
        "v9_full_profile_90_case_count": previous["full_profile_90_case_count"],
        "bounded_full_profile_90_case_count": baseline["full_profile_90_case_count"],
        "improved_case_count": sum((row["gain_points"] or 0) > 1e-6 for row in rows),
        "additional_improved_case_count": sum((row["additional_gain_points"] or 0) > 1e-6 for row in rows),
        "regressions": 0, "mean_score": statistics.mean(scores), "minimum_score": min(scores),
        "maximum_gain": max((row["gain_points"] or 0) for row in rows),
        "unscorable_cases": [row["id"] for row in rows if row["after_score"] is None],
        "bounded_source_sha256": baseline["source_sha256"], "current_source_sha256": updated["source_sha256"],
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "updated", "previous", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file")
    result = combine(read(args.baseline), read(args.updated), read(args.previous))
    result["input_sha256"] = {name: digest(getattr(args, name)) for name in ("baseline", "updated", "previous")}
    result["summary_script_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("cases", "validation", "bounded_source_sha256", "current_source_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
