"""Summarize a completed 400-request run without dropping failed requests."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory, previous_path):
    report = read(directory / "report.json")
    protocol = read(directory / "protocol.json")
    previous = read(previous_path)
    rows, old = report["cases"], {row["id"]: row for row in previous["cases"]}
    assert len(rows) == len(old) == len(protocol["cases"]) == 400
    assert len({row["id"] for row in rows}) == 400
    assert {row["id"] for row in rows} == set(old)
    assert report["target_unchanged"] == previous["target_unchanged"] == 95
    assert report["strict"] and not report["experimental_disable_safety"]
    groups = defaultdict(list)
    buckets = Counter()
    comparisons, raw_hashes = [], {}
    for index, row in enumerate(rows):
        assert row["id"] == protocol["cases"][index]["id"]
        assert row["brief"] == old[row["id"]]["brief"]
        path = directory / f"{index:04d}.json"
        payload = read(path)["full_pool"]
        assert payload.get("calculated_profile_similarity") == row["after_score"]
        assert bool(payload.get("recipe")) == row["recipe_returned"]
        assert bool(payload.get("full_profile_target_met")) == row["full_profile_target_met"]
        score = row["after_score"]
        if row["recipe_returned"]:
            assert score is not None and score >= 95 and payload["safety"]["internal_gate_passed"]
        key = "unscorable" if score is None else "95_and_above" if score >= 95 else "90_to_95" if score >= 90 else "80_to_90" if score >= 80 else "below_80"
        buckets[key] += 1
        groups[row["group"]].append(row)
        prior = old[row["id"]]
        delta = None if score is None or prior["after_score"] is None else score - prior["after_score"]
        comparisons.append({"id": row["id"], "brief": row["brief"], "previous_score": prior["after_score"],
            "score": score, "change_points": delta, "previous_pass": prior["full_profile_target_met"],
            "pass": row["full_profile_target_met"], "status": row["after_status"], "seconds": row["seconds"]["full_pool"]})
        raw_hashes[path.name] = sha(path)
    scores = [row["after_score"] for row in rows if row["after_score"] is not None]
    seconds = sorted(row["seconds"]["full_pool"] for row in rows)
    passed = sum(row["full_profile_target_met"] for row in rows)
    return {"scope": "all 400 existing bilingual development requests, freshly executed; not all possible natural language",
        "score_kind": "full_model_profile_agreement_not_human_similarity", "human_accuracy_claim": False,
        "count": 400, "target": 95, "passed": passed, "pass_percent": passed / 4,
        "not_passed": 400 - passed, "score_buckets": dict(buckets), "scorable_count": len(scores),
        "mean_score_scorable_only": statistics.mean(scores), "minimum_score": min(scores),
        "statuses": dict(Counter(row["after_status"] for row in rows)),
        "latency_seconds": {"mean": statistics.mean(seconds), "median": statistics.median(seconds),
            "p95_nearest_rank": seconds[379], "maximum": max(seconds), "scope": "local four-worker offline inference, not Modal HTTP latency"},
        "groups": {name: {"count": len(group), "passed": sum(row["full_profile_target_met"] for row in group)} for name, group in groups.items()},
        "comparison": {"basis": "previous V13 max12 versus V15 automatic max50000; engine and configuration both changed",
            "previous_passed": sum(row["full_profile_target_met"] for row in old.values()),
            "new_passes": [row["id"] for row in comparisons if row["pass"] and not row["previous_pass"]],
            "lost_passes": [row["id"] for row in comparisons if row["previous_pass"] and not row["pass"]],
            "scores_improved": sum(row["change_points"] is not None and row["change_points"] > 1e-8 for row in comparisons),
            "scores_declined": sum(row["change_points"] is not None and row["change_points"] < -1e-8 for row in comparisons)},
        "inputs": {"report_sha256": sha(directory / "report.json"), "protocol_sha256": sha(directory / "protocol.json"),
            "previous_sha256": sha(previous_path), "raw_sha256": raw_hashes}, "cases": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.previous)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs", "cases")}, ensure_ascii=False, indent=2))
