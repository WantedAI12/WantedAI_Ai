#!/usr/bin/env python
"""Reproduce the v2 transfer-release artifact and validation gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    DESCRIPTOR_DIM,
    EXPECTED_PARAMETER_COUNT,
    DescriptorNormalizer,
    MixturePairDataset,
    R2PhysSimCore,
    build_raw_descriptor_cache,
    load_ravia_pairs,
    load_snitz_pairs,
    normalized_cache,
    sha256_file,
    sha256_json,
)
from scripts.train_physsim_r2 import (  # noqa: E402
    descriptor_cosine_predictions,
    evaluate_model,
    metric_summary,
)


def build_parser() -> argparse.ArgumentParser:
    data = PROJECT_ROOT / "fragrance_ai" / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-data-root", type=Path, required=True)
    parser.add_argument("--pyrfume-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_validation.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=data / "physsim_r2_manifest.json"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=data / "physsim_r2_checkpoint.pt"
    )
    parser.add_argument(
        "--ensemble-manifest",
        type=Path,
        default=data / "physsim_r2_ensemble_manifest.json",
    )
    parser.add_argument(
        "--concentration-manifest",
        type=Path,
        default=data / "concentration_response_manifest.json",
    )
    parser.add_argument(
        "--component-manifest",
        type=Path,
        default=data / "r2_ingredient_components_manifest.json",
    )
    parser.add_argument(
        "--components", type=Path, default=data / "r2_ingredient_components.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_artifact_audit.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    component_manifest = json.loads(args.component_manifest.read_text(encoding="utf-8"))
    ensemble_manifest = json.loads(args.ensemble_manifest.read_text(encoding="utf-8"))
    concentration_manifest = json.loads(
        args.concentration_manifest.read_text(encoding="utf-8")
    )
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checks["checkpoint_sha256"] = (
        sha256_file(args.checkpoint)
        == manifest["checkpoint_sha256"]
        == report["production_checkpoint"]["sha256"]
    )
    checks["component_sha256"] = (
        sha256_file(args.components) == component_manifest["artifact_sha256"]
    )
    checks["descriptor_contract"] = (
        manifest["descriptor_contract_sha256"]
        == component_manifest["descriptor_contract_sha256"]
    )
    ensemble_release = ensemble_manifest.get("release_gate", {})
    checks["ensemble_release_gate"] = (
        bool(ensemble_release.get("passed"))
        and bool(ensemble_release.get("checks"))
        and all(bool(value) for value in ensemble_release.get("checks", {}).values())
    )
    checks["ensemble_descriptor_contract"] = (
        ensemble_manifest.get("descriptor_contract_sha256")
        == manifest.get("descriptor_contract_sha256")
        == component_manifest.get("descriptor_contract_sha256")
    )
    member_checks = {}
    for member in ensemble_manifest.get("members", []):
        path = args.manifest.parent / member["file"]
        member_checks[member["file"]] = (
            path.is_file() and sha256_file(path) == member["sha256"]
        )
    checks["ensemble_member_hashes"] = len(member_checks) == 2 and all(
        member_checks.values()
    )
    checks["ensemble_weights"] = (
        abs(
            sum(
                float(member["weight"])
                for member in ensemble_manifest.get("members", [])
            )
            - 1.0
        )
        < 1e-12
    )
    details["ensemble_member_checks"] = member_checks
    evidence_checks = {}
    for name, record in ensemble_manifest.get("evidence", {}).items():
        path = args.report.parent / record["file"]
        evidence_checks[name] = path.is_file() and sha256_file(path) == record["sha256"]
    checks["ensemble_evidence_hashes"] = len(evidence_checks) == 2 and all(
        evidence_checks.values()
    )
    details["ensemble_evidence_checks"] = evidence_checks

    concentration_path = (
        args.concentration_manifest.parent / concentration_manifest["runtime_file"]
    )
    concentration_release = concentration_manifest.get("release_gate", {})
    checks["concentration_artifact"] = (
        concentration_path.is_file()
        and sha256_file(concentration_path) == concentration_manifest["runtime_sha256"]
        and bool(concentration_release.get("passed"))
        and bool(concentration_release.get("checks"))
        and all(
            bool(value) for value in concentration_release.get("checks", {}).values()
        )
        and concentration_manifest.get("algorithm") == "concentration_only_ridge"
        and float(concentration_manifest.get("structure_specific_weight", -1.0)) == 0.0
    )
    artifact = report["validation_artifacts"]
    development_path = args.report.parent / artifact["development_report_file"]
    final_path = args.report.parent / artifact["final_report_file"]
    checks["development_report_sha256"] = (
        sha256_file(development_path)
        == artifact["development_report_sha256"]
        == manifest["strict_disjoint_validation"]["development_report_sha256"]
    )
    checks["final_report_sha256"] = (
        sha256_file(final_path)
        == artifact["final_report_sha256"]
        == manifest["strict_disjoint_validation"]["final_report_sha256"]
    )
    development = json.loads(development_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    checks["frozen_recipe_contract"] = (
        development["phase"] == "development"
        and final["phase"] == "final"
        and development["model_seed"]
        == final["model_seed"]
        == report["training_configuration"]["model_seed"]
        and development["selected_configuration"]
        == final["selected_configuration"]
        == "all_low_lr"
    )

    source_results: dict[str, bool] = {}
    for relative, expected in report["data"]["source_files"].items():
        if relative.startswith("dream_mixture/"):
            path = args.mixture_data_root / relative.removeprefix("dream_mixture/")
        elif relative.startswith("pyrfume_all/"):
            path = args.pyrfume_root / relative.removeprefix("pyrfume_all/")
        else:
            source_results[relative] = False
            continue
        source_results[relative] = (
            path.exists()
            and sha256_file(path) == expected["sha256"]
            and path.stat().st_size == int(expected["bytes"])
        )
    checks["source_file_hashes"] = all(source_results.values())
    details["source_file_checks"] = source_results

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = R2PhysSimCore()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    checks["parameter_count"] = (
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_PARAMETER_COUNT
        == report["architecture_parameter_count"]
    )
    mean = np.asarray(payload["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(payload["normalizer"]["std"], dtype=np.float32)
    names = tuple(payload["normalizer"]["descriptor_names"])
    checks["normalizer_contract"] = (
        mean.shape == std.shape == (DESCRIPTOR_DIM,)
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
        and (std > 0).all()
        and sha256_json(names) == manifest["descriptor_contract_sha256"]
    )

    minimum = float(manifest["release_gate"]["minimum_spearman_delta_each_protocol"])
    strict_names = (
        "development_strict_molecule_disjoint",
        "development_strict_scaffold_disjoint",
        "final_strict_molecule_disjoint",
        "final_strict_scaffold_disjoint",
    )
    strict_reproduced: dict[str, bool] = {}
    leakage_zero = True
    training_finite = True
    for name in strict_names:
        section = report[name]
        strict_reproduced[name] = (
            float(section["fold_mean_spearman_delta"]) >= minimum
            and float(section["pooled_spearman_delta"]) >= minimum
        )
        for row in section["repeats"]:
            leakage_zero &= all(
                int(row.get(key, 0)) == 0
                for key in (
                    "molecule_leakage_count",
                    "pretraining_molecule_leakage_count",
                    "scaffold_leakage_count",
                )
            )
            training_finite &= int(row["training"]["nonfinite_batches"]) == 0
    checks["strict_gate_reproduced"] = all(strict_reproduced.values())
    checks["validation_leakage_zero"] = leakage_zero
    checks["validation_training_finite"] = training_finite
    details["strict_gate_reproduction"] = strict_reproduced

    snitz_pairs = load_snitz_pairs(args.mixture_data_root)
    ravia_pairs = load_ravia_pairs(args.mixture_data_root)
    all_molecules = {
        molecule for pair in (*snitz_pairs, *ravia_pairs) for molecule in pair.molecules
    }
    raw_cache = build_raw_descriptor_cache(all_molecules)
    normalizer = DescriptorNormalizer(mean, std, names)
    cache = normalized_cache(raw_cache, normalizer)
    ravia_metrics, _ = evaluate_model(model, ravia_pairs, cache, torch.device("cpu"), 8)
    baseline_predictions = descriptor_cosine_predictions(ravia_pairs, cache)
    baseline_metrics = metric_summary(
        baseline_predictions, [pair.similarity for pair in ravia_pairs]
    )
    reproduced_delta = float(ravia_metrics["spearman"] - baseline_metrics["spearman"])
    checks["ravia_gate_reproduced"] = (
        # CUDA and CPU reductions can reorder a few nearly tied predictions;
        # the release decision must reproduce with a narrow rank tolerance.
        abs(reproduced_delta - float(report["zero_shot_ravia"]["spearman_delta"]))
        < 0.005
        and reproduced_delta >= minimum
    )
    details["ravia_reproduction"] = {
        "model_spearman": ravia_metrics["spearman"],
        "baseline_spearman": baseline_metrics["spearman"],
        "spearman_delta": reproduced_delta,
    }

    dataset = MixturePairDataset(snitz_pairs[:2], cache)
    item = dataset[0]
    with torch.inference_mode():
        forward = model(
            item["mixture_a"].unsqueeze(0),
            item["mask_a"].unsqueeze(0),
            item["mixture_b"].unsqueeze(0),
            item["mask_b"].unsqueeze(0),
        )
        reverse = model(
            item["mixture_b"].unsqueeze(0),
            item["mask_b"].unsqueeze(0),
            item["mixture_a"].unsqueeze(0),
            item["mask_a"].unsqueeze(0),
        )
    checks["inference_finite"] = bool(torch.isfinite(forward).all())
    checks["pair_symmetry"] = bool(
        torch.allclose(forward, reverse, atol=1e-7, rtol=0.0)
    )

    release_checks = manifest["release_gate"]["checks"]
    checks["release_gate_contract"] = (
        bool(manifest["release_gate"]["passed"])
        and bool(report["release_gate"]["passed"])
        and all(bool(value) for value in release_checks.values())
        and float(manifest["release_gate"]["approved_primary_score_weight"]) == 0.10
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
