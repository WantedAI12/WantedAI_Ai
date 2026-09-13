"""Preserve measured reference rows exactly while selecting a new odor decoder."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from fragrance_ai.recommender.formulation_core import FormulationCore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--source-reference", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model = FormulationCore(args.model, args.model_sha256)
    raw = args.source_reference.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_sha256:
        raise ValueError("reference hash mismatch")
    old = json.loads(raw)
    if tuple(old["endpoints"]) != model.quantitative_endpoints:
        raise ValueError("reference endpoint identity mismatch")
    value = dict(old)
    value["parent_atlas_sha256"] = model.sha256
    if {k: v for k, v in old.items() if k != "parent_atlas_sha256"} != {
        k: v for k, v in value.items() if k != "parent_atlas_sha256"
    }:
        raise AssertionError("reference observations changed")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "model.json"
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = {
        "source_sha256": args.source_sha256,
        "parent_atlas_sha256": model.sha256,
        "observed_values_exactly_unchanged": True,
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (args.output / "binding.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
