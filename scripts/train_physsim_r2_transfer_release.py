#!/usr/bin/env python
"""Build the release R2 checkpoint from the frozen transfer recipe."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    DESCRIPTOR_DIM,
    EXPECTED_PARAMETER_COUNT,
    MAX_MOLECULES,
    MODEL_SPEC_VERSION,
    N_STEPS,
    SOFT_CORE_DELTA,
    R2PhysSimCore,
    PU_LABEL_CONTRACT_VERSION,
    STRICT_SPLIT_CONTRACT_VERSION,
    audit_external_source_disjointness,
    descriptor_contract,
    fit_normalizer,
    load_odor_descriptor_records,
    load_ravia_pairs,
    load_snitz_pairs,
    normalized_cache,
    set_deterministic_seed,
    sha256_file,
    sha256_json,
)
from scripts.experiment_physsim_r2_transfer import (  # noqa: E402
    CONFIGURATIONS,
    _cpu_state,
    fine_tune_snitz,
    load_or_build_raw_cache,
    make_ontology_similarity_pairs,
    pretrain_descriptor_encoder,
    pretrain_similarity_model,
)
from scripts.train_physsim_r2 import (  # noqa: E402
    descriptor_cosine_predictions,
    evaluate_model,
    metric_summary,
)


def source_hash(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _strict_summary(report: dict[str, object], section: str) -> dict[str, object]:
    split_contract = report.get("split_contract", {})
    if (
        split_contract.get("version") != STRICT_SPLIT_CONTRACT_VERSION
        or split_contract.get("reported_validation_pairs")
        != "all_components_held_out"
        or split_contract.get(section) != "all_components_held_out"
    ):
        raise RuntimeError(
            "report is not an all-components-held-out validation artifact; "
            "do not label mixed-component scores strict"
        )
    result = report[section]["configurations"]["all_low_lr"]
    return {
        "fold_mean_spearman_delta": result["fold_mean_spearman_delta"],
        "pooled_spearman_delta": result["pooled_spearman_delta"],
        "pooled_model": result["pooled_model"],
        "pooled_baseline": result["pooled_baseline"],
        "repeats": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {"predictions", "baseline_predictions", "targets"}
            }
            for row in result["repeats"]
        ],
    }


def _zero_leakage(report: dict[str, object]) -> bool:
    for section in ("molecule_disjoint", "scaffold_disjoint"):
        result = report[section]["configurations"]["all_low_lr"]
        for row in result["repeats"]:
            if any(
                int(row.get(key, 0)) != 0
                for key in (
                    "molecule_leakage_count",
                    "pretraining_molecule_leakage_count",
                    "scaffold_leakage_count",
                )
            ):
                return False
    return True


def _finite_transfer_training(report: dict[str, object]) -> bool:
    for section in ("molecule_disjoint", "scaffold_disjoint"):
        protocol = report[section]
        if int(protocol["encoder_pretraining"]["nonfinite_batches"]) != 0:
            return False
        if int(protocol["similarity_pretraining"]["nonfinite_batches"]) != 0:
            return False
        result = protocol["configurations"]["all_low_lr"]
        if any(int(row["training"]["nonfinite_batches"]) != 0 for row in result["repeats"]):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    data = PROJECT_ROOT / "fragrance_ai" / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-data-root", type=Path, required=True)
    parser.add_argument("--pyrfume-root", type=Path, required=True)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_development_calibration.json",
    )
    parser.add_argument(
        "--final-report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_final_strict.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_validation.json",
    )
    parser.add_argument("--output-dir", type=Path, default=data)
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_release_descriptor_cache.npz",
    )
    parser.add_argument("--model-seed", type=int, default=20260715)
    parser.add_argument("--encoder-epochs", type=int, default=10)
    parser.add_argument("--similarity-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    final = json.loads(args.final_report.read_text(encoding="utf-8"))
    if development.get("phase") != "development" or final.get("phase") != "final":
        raise RuntimeError("development/final report phase contract mismatch")
    if development.get("model_seed") != args.model_seed or final.get("model_seed") != args.model_seed:
        raise RuntimeError("frozen model seed does not match validation reports")
    if any(
        report.get("selected_configuration") != "all_low_lr"
        for report in (development, final)
    ):
        raise RuntimeError("frozen transfer recipe mismatch")

    vocabulary, descriptor_records, descriptor_source_counts = (
        load_odor_descriptor_records(args.pyrfume_root)
    )
    snitz_pairs = load_snitz_pairs(args.mixture_data_root)
    ravia_pairs = load_ravia_pairs(args.mixture_data_root)
    snitz_molecules = sorted(
        {molecule for pair in snitz_pairs for molecule in pair.molecules}
    )
    ravia_molecules = sorted(
        {molecule for pair in ravia_pairs for molecule in pair.molecules}
    )
    all_molecules = {
        record.smiles for record in descriptor_records
    } | {
        molecule
        for pair in (*snitz_pairs, *ravia_pairs)
        for molecule in pair.molecules
    }
    raw_cache = load_or_build_raw_cache(all_molecules, args.raw_cache)
    descriptor_names = [name for name, _ in descriptor_contract()]
    molecular_weight_index = descriptor_names.index("MolWt")
    heavy_atom_index = descriptor_names.index("HeavyAtomCount")
    before_filter = len(descriptor_records)
    descriptor_records = [
        record
        for record in descriptor_records
        if 20.0 <= float(raw_cache[record.smiles][molecular_weight_index]) <= 750.0
        and float(raw_cache[record.smiles][heavy_atom_index]) <= 70.0
        and float(np.max(np.abs(raw_cache[record.smiles]))) < 1e25
    ]
    # Ravia is an external evaluation source only if it is disjoint from all
    # supervised representations and descriptor-normalizer fitting.  Record
    # the actual overlap before training; a non-zero overlap is a release
    # failure, not a footnote that may be hidden by a zero-shot label policy.
    descriptor_pretraining_molecules = [record.smiles for record in descriptor_records]
    normalizer_fit_molecules = [*descriptor_pretraining_molecules, *snitz_molecules]
    ravia_source_disjointness = audit_external_source_disjointness(
        ravia_molecules,
        {
            "descriptor_pretraining": descriptor_pretraining_molecules,
            "snitz_fine_tuning": snitz_molecules,
            "normalizer_fit": normalizer_fit_molecules,
        },
    )
    normalizer = fit_normalizer(
        raw_cache,
        normalizer_fit_molecules,
    )
    cache = normalized_cache(raw_cache, normalizer)

    set_deterministic_seed(args.model_seed)
    base_model = R2PhysSimCore().to(device)
    encoder_training = pretrain_descriptor_encoder(
        base_model,
        descriptor_records,
        cache,
        epochs=args.encoder_epochs,
        batch_size=max(64, args.batch_size * 16),
        seed=args.model_seed + 1,
        device=device,
    )
    ontology_pairs = make_ontology_similarity_pairs(
        descriptor_records, seed=args.model_seed + 2
    )
    similarity_training = pretrain_similarity_model(
        base_model,
        ontology_pairs,
        cache,
        epochs=args.similarity_epochs,
        batch_size=max(32, args.batch_size * 8),
        seed=args.model_seed + 3,
        device=device,
    )
    config = next(value for value in CONFIGURATIONS if value.name == "all_low_lr")
    production_model, fine_tuning = fine_tune_snitz(
        _cpu_state(base_model),
        snitz_pairs,
        cache,
        config,
        seed=args.model_seed + 4,
        batch_size=args.batch_size,
        device=device,
    )
    del base_model
    snitz_metrics, _ = evaluate_model(
        production_model, snitz_pairs, cache, device, args.batch_size
    )
    # The production checkpoint and recipe are fixed before this first and
    # only read of Ravia similarity labels for release evaluation.
    ravia_metrics, ravia_predictions = evaluate_model(
        production_model, ravia_pairs, cache, device, args.batch_size
    )
    ravia_baseline_predictions = descriptor_cosine_predictions(ravia_pairs, cache)
    ravia_targets = [pair.similarity for pair in ravia_pairs]
    ravia_baseline = metric_summary(ravia_baseline_predictions, ravia_targets)
    ravia_delta = float(ravia_metrics["spearman"] - ravia_baseline["spearman"])

    minimum_delta = 0.01
    development_molecule = _strict_summary(development, "molecule_disjoint")
    development_scaffold = _strict_summary(development, "scaffold_disjoint")
    final_molecule = _strict_summary(final, "molecule_disjoint")
    final_scaffold = _strict_summary(final, "scaffold_disjoint")

    def improves(summary: dict[str, object]) -> bool:
        return (
            float(summary["fold_mean_spearman_delta"]) >= minimum_delta
            and float(summary["pooled_spearman_delta"]) >= minimum_delta
        )

    validation_leakage_zero = _zero_leakage(development) and _zero_leakage(final)
    validation_finite = _finite_transfer_training(development) and _finite_transfer_training(final)
    production_finite = all(
        int(section["nonfinite_batches"]) == 0
        for section in (encoder_training, similarity_training, fine_tuning)
    )
    direct_formulation_capabilities = {
        "relative_ingredient_amounts_directly_encoded": False,
        "finished_product_concentration_directly_encoded": False,
        "time_or_headspace_trajectory_directly_encoded": False,
    }
    direct_formulation_authorized = all(direct_formulation_capabilities.values())
    checks = {
        "development_strict_molecule_disjoint_improves_baseline": improves(development_molecule),
        "development_strict_scaffold_disjoint_improves_baseline": improves(development_scaffold),
        "final_strict_molecule_disjoint_improves_baseline": improves(final_molecule),
        "final_strict_scaffold_disjoint_improves_baseline": improves(final_scaffold),
        "ravia_transfer_improves_baseline": ravia_delta >= minimum_delta,
        "ravia_molecule_and_scaffold_disjoint_from_supervision_and_normalizer": bool(
            ravia_source_disjointness["passed"]
        ),
        "all_pretraining_and_split_leakage_counts_are_zero": validation_leakage_zero,
        "all_training_batches_are_finite": validation_finite and production_finite,
        "pu_safe_positive_only_descriptor_pretraining": (
            encoder_training.get("label_contract_version") == PU_LABEL_CONTRACT_VERSION
            and encoder_training.get("objective") == "non_negative_positive_unlabeled_risk"
        ),
        "strict_scores_use_all_components_held_out_pairs": True,
        "r2_direct_formulation_capability_authorizes_primary_score": (
            direct_formulation_authorized
        ),
    }
    release_passed = all(checks.values())
    approved_weight = 0.10 if release_passed else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "physsim_r2_checkpoint.pt"
    checkpoint = {
        "model_spec_version": MODEL_SPEC_VERSION,
        "model_state_dict": _cpu_state(production_model),
        "normalizer": normalizer.as_dict(),
        "architecture": {
            "descriptor_dim": DESCRIPTOR_DIM,
            "max_molecules": MAX_MOLECULES,
            "n_steps": N_STEPS,
            "soft_core_delta": SOFT_CORE_DELTA,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "active_core_constants": [
                "attraction_G",
                "charge_k_e",
                "lj_epsilon",
                "velocity_limit",
                "mass_decay_kappa",
            ],
        },
        "training": {
            "recipe": asdict(config),
            "model_seed": args.model_seed,
            "descriptor_pretraining": encoder_training,
            "ontology_similarity_pretraining": similarity_training,
            "snitz_fine_tuning": fine_tuning,
            "selection_rule": "frozen by development splits before final and Ravia evaluation",
        },
        "learned_constants": production_model.learned_constants(),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    mixture_sources = (
        "snitz_2013/molecules.csv",
        "snitz_2013/behavior.csv",
        "ravia_2020/molecules.csv",
        "ravia_2020/stimuli.csv",
        "ravia_2020/behavior_2.csv",
    )
    descriptor_sources = (
        "leffingwell/molecules.csv",
        "leffingwell/behavior.csv",
        "goodscents/molecules.csv",
        "goodscents/behavior.csv",
        "goodscents/cas_to_cid.json",
        "flavornet/molecules.csv",
        "flavornet/behavior.csv",
        "aromadb/molecules.csv",
        "aromadb/behavior.csv",
        "ifra_2019/molecules.csv",
        "ifra_2019/behavior.csv",
        "flavordb/molecules.csv",
        "flavordb/behavior.csv",
    )
    sources = {
        **{
            f"dream_mixture/{relative}": source_hash(args.mixture_data_root / relative)
            for relative in mixture_sources
        },
        **{
            f"pyrfume_all/{relative}": source_hash(args.pyrfume_root / relative)
            for relative in descriptor_sources
        },
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    # Compatibility aliases retain the fail-closed runtime contract while the
    # explicit v2 checks above are the authoritative release criteria.
    runtime_checks = {
        "molecule_cold_improves_baseline": checks["final_strict_molecule_disjoint_improves_baseline"],
        "scaffold_cold_improves_baseline": checks["final_strict_scaffold_disjoint_improves_baseline"],
        "strict_molecule_disjoint_improves_baseline": checks["final_strict_molecule_disjoint_improves_baseline"],
        "strict_scaffold_disjoint_improves_baseline": checks["final_strict_scaffold_disjoint_improves_baseline"],
        "ravia_transfer_improves_baseline": checks["ravia_transfer_improves_baseline"],
        "ravia_molecule_and_scaffold_disjoint_from_supervision_and_normalizer": checks[
            "ravia_molecule_and_scaffold_disjoint_from_supervision_and_normalizer"
        ],
        "molecule_split_has_zero_component_leakage": validation_leakage_zero,
        "scaffold_split_has_zero_component_leakage": validation_leakage_zero,
        "strict_molecule_disjoint_has_zero_leakage": validation_leakage_zero,
        "strict_scaffold_disjoint_has_zero_leakage": validation_leakage_zero,
        "training_had_no_nonfinite_batches": validation_finite and production_finite,
        **checks,
    }
    manifest = {
        "schema_version": "2.0",
        "model_spec_version": MODEL_SPEC_VERSION,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "generated_at": generated_at,
        "descriptor_contract_sha256": sha256_json(normalizer.descriptor_names),
        "source_files": sources,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "release_gate": {
            "passed": release_passed,
            "minimum_spearman_delta_each_protocol": minimum_delta,
            "checks": runtime_checks,
            "approved_primary_score_weight": approved_weight,
            "fallback_behavior": (
                "learned R2 contributes at most a 10% centered residual within applicability"
                if release_passed
                else "weight remains zero; deterministic concentration/headspace score remains primary"
            ),
        },
        "capability_contract": {
            "version": "direct-formulation-inputs-1.0",
            "direct_formulation_inputs": direct_formulation_capabilities,
            "authorized_for_primary_score_weight": direct_formulation_authorized,
            "fail_closed_behavior": (
                "primary score weight is zero unless the checkpoint directly encodes "
                "relative ingredient amounts, finished-product concentration, and "
                "time/headspace trajectory"
            ),
        },
        "ensemble_calibration": {
            "method": "centered_residual_on_primary_score",
            "neutral_similarity_percent": float(np.mean([pair.similarity for pair in snitz_pairs]) * 100.0),
            "formula": "primary_score + approved_weight * (r2_similarity_percent - neutral_similarity_percent)",
            "structural_development_blend": {"r2": 1.0, "rdkit_cosine": 0.0},
        },
        "transfer_learning": {
            "ontology_labels": len(vocabulary),
            "descriptor_molecules_before_filter": before_filter,
            "descriptor_molecules_after_filter": len(descriptor_records),
            "source_record_counts": descriptor_source_counts,
            "model_seed": args.model_seed,
            "configuration": asdict(config),
        },
        "strict_disjoint_validation": {
            "split_contract_version": STRICT_SPLIT_CONTRACT_VERSION,
            "validation_pair_contract": "all_components_held_out",
            "development_report_file": args.development_report.name,
            "development_report_sha256": sha256_file(args.development_report),
            "final_report_file": args.final_report.name,
            "final_report_sha256": sha256_file(args.final_report),
            "development_molecule": development_molecule,
            "development_scaffold": development_scaffold,
            "final_molecule": final_molecule,
            "final_scaffold": final_scaffold,
        },
        "external_source_disjointness": {
            "source": "Ravia et al. 2020",
            "audit": ravia_source_disjointness,
            "release_requirement": (
                "molecule and Bemis-Murcko scaffold overlap must both be zero for every "
                "supervised pretraining, fine-tuning, and normalizer-fit population"
            ),
        },
        "label_supervision_contract": {
            "version": PU_LABEL_CONTRACT_VERSION,
            "objective": "non_negative_positive_unlabeled_risk",
            "absent_descriptor_behavior": "unlabeled_not_negative",
        },
        "dataset_provenance": {
            "archive": "Pyrfume Public Data Archive",
            "archive_license": "MIT repository license",
            "pretraining": "Leffingwell ontology with exact-vocabulary unions from five additional monomolecular archives",
            "fine_tuning": "Snitz et al. 2013 mixture similarity",
            "external_transfer": "Ravia et al. 2020 pair similarity; unused for selection",
        },
        "claim_boundary": "Validated historical mixture-similarity model; not a certification of 90% text-to-formula human olfactory equivalence.",
    }
    manifest_path = args.output_dir / "physsim_r2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "schema_version": "2.0",
        "generated_at": generated_at,
        "elapsed_seconds": time.time() - started,
        "model_spec_version": MODEL_SPEC_VERSION,
        "architecture_parameter_count": EXPECTED_PARAMETER_COUNT,
        "data": {
            "snitz_pairs": len(snitz_pairs),
            "snitz_molecules": len(snitz_molecules),
            "ravia_pairs": len(ravia_pairs),
            "ravia_source_disjointness": ravia_source_disjointness,
            "ontology_labels": len(vocabulary),
            "descriptor_molecules": len(descriptor_records),
            "source_files": sources,
        },
        "training_configuration": {
            "model_seed": args.model_seed,
            "recipe": asdict(config),
            "encoder_epochs": args.encoder_epochs,
            "similarity_epochs": args.similarity_epochs,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "validation_artifacts": {
            "development_report_file": args.development_report.name,
            "development_report_sha256": sha256_file(args.development_report),
            "final_report_file": args.final_report.name,
            "final_report_sha256": sha256_file(args.final_report),
        },
        "development_strict_molecule_disjoint": development_molecule,
        "development_strict_scaffold_disjoint": development_scaffold,
        "final_strict_molecule_disjoint": final_molecule,
        "final_strict_scaffold_disjoint": final_scaffold,
        "production_checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "training": checkpoint["training"],
            "snitz_training_metrics": snitz_metrics,
            "learned_constants": production_model.learned_constants(),
        },
        "zero_shot_ravia": {
            "model": ravia_metrics,
            "rdkit_cosine_baseline": ravia_baseline,
            "spearman_delta": ravia_delta,
            "used_for_model_or_hyperparameter_selection": False,
            "predictions_sha256": sha256_json([float(value) for value in ravia_predictions]),
            "source_disjointness_audit": ravia_source_disjointness,
        },
        "release_gate": manifest["release_gate"],
        "capability_contract": manifest["capability_contract"],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "ravia_spearman": ravia_metrics["spearman"],
                "ravia_baseline": ravia_baseline["spearman"],
                "ravia_delta": ravia_delta,
                "release_gate": manifest["release_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if release_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
