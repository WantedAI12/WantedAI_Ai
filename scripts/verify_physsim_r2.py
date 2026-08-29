#!/usr/bin/env python
"""Verify frozen R2 artifacts, split leakage, and inference invariants."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    MixturePairDataset,
    R2PhysSimCore,
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    load_snitz_pairs,
    sha256_file,
)

# The split helpers live in the executable training protocol so the exact
# published partitioning logic is audited rather than reimplemented loosely.
from train_physsim_r2 import (  # noqa: E402
    molecule_folds,
    scaffold_folds,
    split_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    data = PROJECT_ROOT / "fragrance_ai" / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "benchmarks" / "physsim_r2_validation.json")
    parser.add_argument("--manifest", type=Path, default=data / "physsim_r2_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=data / "physsim_r2_checkpoint.pt")
    parser.add_argument("--component-manifest", type=Path, default=data / "r2_ingredient_components_manifest.json")
    parser.add_argument("--components", type=Path, default=data / "r2_ingredient_components.npz")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "physsim_r2_artifact_audit.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    component_manifest = json.loads(
        args.component_manifest.read_text(encoding="utf-8")
    )
    checks["checkpoint_sha256"] = (
        sha256_file(args.checkpoint) == manifest["checkpoint_sha256"]
        == report["production_checkpoint"]["sha256"]
    )
    checks["component_sha256"] = (
        sha256_file(args.components) == component_manifest["artifact_sha256"]
    )
    checks["descriptor_contract"] = (
        manifest["descriptor_contract_sha256"]
        == component_manifest["descriptor_contract_sha256"]
    )
    source_checks = {}
    for relative, expected in report["data"]["source_files"].items():
        source_checks[relative] = (
            sha256_file(args.data_root / relative) == expected["sha256"]
        )
    checks["source_file_hashes"] = all(source_checks.values())
    details["source_file_checks"] = source_checks

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = R2PhysSimCore()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    checks["parameter_count"] = (
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_PARAMETER_COUNT
        == report["architecture_parameter_count"]
    )

    pairs = load_snitz_pairs(args.data_root)
    expected_neutral = float(np.mean([pair.similarity for pair in pairs]) * 100.0)
    calibration = manifest.get("ensemble_calibration", {})
    checks["ensemble_calibration_contract"] = (
        calibration.get("method") == "centered_residual_on_primary_score"
        and abs(
            float(calibration.get("neutral_similarity_percent", -1.0))
            - expected_neutral
        ) < 1e-9
    )
    details["ensemble_calibration"] = {
        "manifest_neutral_similarity_percent": calibration.get(
            "neutral_similarity_percent"
        ),
        "recomputed_snitz_label_mean_percent": expected_neutral,
    }
    molecules = sorted({molecule for pair in pairs for molecule in pair.molecules})
    seed = int(report["training_configuration"]["seed"])
    fold_count = int(report["training_configuration"]["folds"])
    leakage_records = []
    for split_name, held_out_sets in (
        ("molecule_cold", molecule_folds(molecules, n_splits=fold_count, seed=seed)),
        (
            "bemis_murcko_scaffold_cold",
            scaffold_folds(molecules, n_splits=fold_count, seed=seed),
        ),
    ):
        for fold, held_out in enumerate(held_out_sets, start=1):
            training, validation, strict, used_training = split_pairs(pairs, held_out)
            held_out_scaffolds = {bemis_murcko_scaffold(value) for value in held_out}
            training_scaffolds = {
                bemis_murcko_scaffold(value) for value in used_training
            }
            molecule_leakage = len(used_training & held_out)
            scaffold_leakage = (
                len(training_scaffolds & held_out_scaffolds)
                if split_name == "bemis_murcko_scaffold_cold"
                else None
            )
            reported = report[split_name]["folds"][fold - 1]
            counts_match = (
                len(training) == reported["n_training_pairs"]
                and len(validation) == reported["n_validation_pairs"]
                and len(strict)
                == reported["n_strict_all_components_held_out_pairs"]
            )
            leakage_records.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "molecule_leakage_count": molecule_leakage,
                    "scaffold_leakage_count": scaffold_leakage,
                    "counts_match_training_report": counts_match,
                }
            )
    checks["molecule_leakage_zero"] = all(
        record["molecule_leakage_count"] == 0 for record in leakage_records
    )
    checks["scaffold_leakage_zero"] = all(
        record["scaffold_leakage_count"] in (None, 0)
        for record in leakage_records
    )
    checks["fold_counts_reproduced"] = all(
        record["counts_match_training_report"] for record in leakage_records
    )
    details["split_audit"] = leakage_records

    strict_sections = (
        report.get("strict_molecule_disjoint", {}),
        report.get("strict_scaffold_disjoint", {}),
    )
    checks["strict_disjoint_protocols_present"] = all(
        bool(section.get("repeats")) for section in strict_sections
    )
    checks["strict_disjoint_leakage_zero"] = all(
        int(record.get("molecule_leakage_count", -1)) == 0
        and int(record.get("scaffold_leakage_count", -1)) == 0
        for section in strict_sections
        for record in section.get("repeats", [])
    )
    strict_minimum = float(
        manifest["release_gate"]["minimum_spearman_delta_each_protocol"]
    )
    expected_strict_gate = [
        float(section.get("pooled_spearman_delta", -1.0)) >= strict_minimum
        and float(section.get("fold_mean_spearman_delta", -1.0)) >= strict_minimum
        for section in strict_sections
    ]
    reported_strict_gate = [
        bool(manifest["release_gate"]["checks"].get(
            "strict_molecule_disjoint_improves_baseline", False
        )),
        bool(manifest["release_gate"]["checks"].get(
            "strict_scaffold_disjoint_improves_baseline", False
        )),
    ]
    checks["strict_disjoint_gate_reproduced"] = (
        expected_strict_gate == reported_strict_gate
    )
    details["strict_disjoint_summary"] = {
        "molecule": {
            "repeats": len(strict_sections[0].get("repeats", [])),
            "pooled_spearman_delta": strict_sections[0].get(
                "pooled_spearman_delta"
            ),
        },
        "scaffold": {
            "repeats": len(strict_sections[1].get("repeats", [])),
            "pooled_spearman_delta": strict_sections[1].get(
                "pooled_spearman_delta"
            ),
        },
    }

    raw = build_raw_descriptor_cache(molecules)
    mean = np.asarray(payload["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(payload["normalizer"]["std"], dtype=np.float32)
    cache = {key: ((value - mean) / std).astype(np.float32) for key, value in raw.items()}
    dataset = MixturePairDataset(pairs[:2], cache)
    item = dataset[0]
    with torch.inference_mode():
        forward = model(
            item["mixture_a"].unsqueeze(0), item["mask_a"].unsqueeze(0),
            item["mixture_b"].unsqueeze(0), item["mask_b"].unsqueeze(0),
        )
        reverse = model(
            item["mixture_b"].unsqueeze(0), item["mask_b"].unsqueeze(0),
            item["mixture_a"].unsqueeze(0), item["mask_a"].unsqueeze(0),
        )
    checks["inference_finite"] = bool(torch.isfinite(forward).all())
    checks["pair_symmetry"] = bool(torch.allclose(forward, reverse, atol=1e-7, rtol=0.0))
    details["inference_probe"] = {
        "forward": float(forward.item()),
        "reverse": float(reverse.item()),
    }

    checks["release_gate_reproduced"] = (
        bool(manifest["release_gate"]["passed"])
        == bool(report["release_gate"]["passed"])
        == all(bool(value) for value in report["release_gate"]["checks"].values())
    )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "release_gate": manifest["release_gate"],
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
