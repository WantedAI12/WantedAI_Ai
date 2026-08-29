#!/usr/bin/env python
"""Train, validate, freeze, and gate the JCIM R2 PhysSim-Core checkpoint.

The release gate is intentionally conservative: the learned score receives a
non-zero production ensemble weight only if it improves over the same-fold
RDKit descriptor cosine baseline on both molecule-cold and Bemis-Murcko
scaffold-cold validation and on the untouched Ravia transfer set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

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
    DescriptorNormalizer,
    MixturePair,
    MixturePairDataset,
    R2PhysSimCore,
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    combined_loss,
    fit_normalizer,
    load_ravia_pairs,
    load_snitz_pairs,
    normalized_cache,
    set_deterministic_seed,
    sha256_file,
    sha256_json,
    symmetric_augmentation,
)


def _safe_correlation(function, first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 3 or np.std(first) < 1e-10 or np.std(second) < 1e-10:
        return 0.0
    value = function(first, second)[0]
    return float(value) if np.isfinite(value) else 0.0


def metric_summary(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    prediction = np.asarray(predictions, dtype=float)
    target = np.asarray(targets, dtype=float)
    return {
        "spearman": _safe_correlation(spearmanr, prediction, target),
        "pearson": _safe_correlation(pearsonr, prediction, target),
        "mae": float(np.mean(np.abs(prediction - target))) if len(target) else math.nan,
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))) if len(target) else math.nan,
        "n": int(len(target)),
    }


def evaluate_model(
    model: R2PhysSimCore,
    pairs: Sequence[MixturePair],
    cache: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], list[float]]:
    loader = DataLoader(
        MixturePairDataset(pairs, cache),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[float] = []
    targets: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["mixture_a"].to(device),
                batch["mask_a"].to(device),
                batch["mixture_b"].to(device),
                batch["mask_b"].to(device),
            )
            predictions.extend(float(value) for value in output.cpu())
            targets.extend(float(value) for value in batch["similarity"])
    return metric_summary(predictions, targets), predictions


def descriptor_cosine_predictions(
    pairs: Sequence[MixturePair], cache: dict[str, np.ndarray]
) -> list[float]:
    predictions: list[float] = []
    for pair in pairs:
        first = np.mean([cache[value] for value in pair.mixture_a], axis=0)
        second = np.mean([cache[value] for value in pair.mixture_b], axis=0)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        cosine = float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0
        # Cosine is a rank baseline; mapping to [0, 1] also makes MAE legible.
        predictions.append(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))
    return predictions


def _model_state_on_cpu(model: R2PhysSimCore) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_with_validation(
    train_pairs: Sequence[MixturePair],
    validation_pairs: Sequence[MixturePair],
    cache: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[R2PhysSimCore, dict[str, float | int]]:
    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MixturePairDataset(symmetric_augmentation(train_pairs), cache),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = R2PhysSimCore().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 5)
    )
    best_spearman = -float("inf")
    best_epoch = 0
    best_state = _model_state_on_cpu(model)
    finite_batches = 0
    nonfinite_batches = 0
    for epoch in range(epochs):
        if epoch < 5:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4 * (epoch + 1) / 5.0
        model.train()
        for batch in train_loader:
            prediction = model(
                batch["mixture_a"].to(device),
                batch["mask_a"].to(device),
                batch["mixture_b"].to(device),
                batch["mask_b"].to(device),
            )
            loss = combined_loss(prediction.float(), batch["similarity"].to(device))
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            finite_batches += 1
        if epoch >= 5:
            scheduler.step()
        metrics, _ = evaluate_model(
            model, validation_pairs, cache, device, batch_size
        )
        if metrics["spearman"] > best_spearman:
            best_spearman = metrics["spearman"]
            best_epoch = epoch + 1
            best_state = _model_state_on_cpu(model)
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_spearman": best_spearman,
        "finite_training_batches": finite_batches,
        "nonfinite_training_batches": nonfinite_batches,
    }


def train_fixed_epochs(
    pairs: Sequence[MixturePair],
    cache: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[R2PhysSimCore, dict[str, float | int]]:
    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        MixturePairDataset(symmetric_augmentation(pairs), cache),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = R2PhysSimCore().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 5)
    )
    finite_batches = 0
    nonfinite_batches = 0
    final_loss = math.inf
    for epoch in range(epochs):
        if epoch < 5:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4 * (epoch + 1) / 5.0
        model.train()
        losses: list[float] = []
        for batch in loader:
            prediction = model(
                batch["mixture_a"].to(device),
                batch["mask_a"].to(device),
                batch["mixture_b"].to(device),
                batch["mask_b"].to(device),
            )
            loss = combined_loss(prediction.float(), batch["similarity"].to(device))
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            finite_batches += 1
        if epoch >= 5:
            scheduler.step()
        if losses:
            final_loss = float(np.mean(losses))
    return model, {
        "epochs": epochs,
        "final_training_loss": final_loss,
        "finite_training_batches": finite_batches,
        "nonfinite_training_batches": nonfinite_batches,
    }


def molecule_folds(
    molecules: Sequence[str], *, n_splits: int, seed: int
) -> list[set[str]]:
    ordered = np.asarray(sorted(set(molecules)), dtype=object)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [set(ordered[validation].tolist()) for _, validation in splitter.split(ordered)]


def scaffold_folds(
    molecules: Sequence[str], *, n_splits: int, seed: int
) -> list[set[str]]:
    groups: dict[str, list[str]] = {}
    for molecule in sorted(set(molecules)):
        groups.setdefault(bemis_murcko_scaffold(molecule), []).append(molecule)
    rng = np.random.RandomState(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: -len(item[1]))
    folds = [set() for _ in range(n_splits)]
    sizes = [0 for _ in range(n_splits)]
    for _, members in group_items:
        target = min(range(n_splits), key=lambda index: (sizes[index], index))
        folds[target].update(members)
        sizes[target] += len(members)
    return folds


def split_pairs(
    pairs: Sequence[MixturePair], held_out_molecules: set[str]
) -> tuple[list[MixturePair], list[MixturePair], list[MixturePair], set[str]]:
    all_molecules = {molecule for pair in pairs for molecule in pair.molecules}
    training_molecules = all_molecules - held_out_molecules
    training = [pair for pair in pairs if pair.molecules.issubset(training_molecules)]
    validation = [pair for pair in pairs if pair.molecules & held_out_molecules]
    strict_validation = [
        pair for pair in pairs if pair.molecules.issubset(held_out_molecules)
    ]
    used_training_molecules = {
        molecule for pair in training for molecule in pair.molecules
    }
    leakage = used_training_molecules & held_out_molecules
    if leakage:
        raise RuntimeError(f"component leakage detected: {sorted(leakage)[:3]}")
    return training, validation, strict_validation, used_training_molecules


def run_cross_validation(
    *,
    split_name: str,
    held_out_folds: Sequence[set[str]],
    pairs: Sequence[MixturePair],
    raw_cache: dict[str, np.ndarray],
    epochs: int,
    restarts: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    fold_model_spearman: list[float] = []
    fold_baseline_spearman: list[float] = []
    all_model_predictions: list[float] = []
    all_baseline_predictions: list[float] = []
    all_targets: list[float] = []
    for fold_index, held_out in enumerate(held_out_folds, start=1):
        training, validation, strict_validation, used_training = split_pairs(
            pairs, held_out
        )
        if len(training) < 10 or len(validation) < 5:
            raise RuntimeError(
                f"{split_name} fold {fold_index} is not trainable: "
                f"{len(training)} train / {len(validation)} validation"
            )
        normalizer = fit_normalizer(raw_cache, used_training)
        cache = normalized_cache(raw_cache, normalizer)
        baseline_predictions = descriptor_cosine_predictions(validation, cache)
        baseline_metrics = metric_summary(
            baseline_predictions, [pair.similarity for pair in validation]
        )
        best_model: R2PhysSimCore | None = None
        best_metrics: dict[str, float] | None = None
        best_training: dict[str, float | int] | None = None
        best_restart = 0
        for restart in range(restarts):
            run_seed = seed * 10_000 + fold_index * 100 + restart
            model, training_metrics = train_with_validation(
                training,
                validation,
                cache,
                seed=run_seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
            metrics, _ = evaluate_model(
                model, validation, cache, device, batch_size
            )
            if best_metrics is None or metrics["spearman"] > best_metrics["spearman"]:
                best_model = model
                best_metrics = metrics
                best_training = training_metrics
                best_restart = restart + 1
            else:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        assert best_model is not None and best_metrics is not None and best_training is not None
        _, model_predictions = evaluate_model(
            best_model, validation, cache, device, batch_size
        )
        all_model_predictions.extend(model_predictions)
        all_baseline_predictions.extend(baseline_predictions)
        all_targets.extend(pair.similarity for pair in validation)
        fold_model_spearman.append(float(best_metrics["spearman"]))
        fold_baseline_spearman.append(float(baseline_metrics["spearman"]))
        records.append(
            {
                "fold": fold_index,
                "n_training_pairs": len(training),
                "n_validation_pairs": len(validation),
                "n_strict_all_components_held_out_pairs": len(strict_validation),
                "n_training_molecules": len(used_training),
                "n_held_out_molecules": len(held_out),
                "molecule_leakage_count": len(used_training & held_out),
                "held_out_component_sha256": sha256_json(sorted(held_out)),
                "selected_restart": best_restart,
                "training": best_training,
                "model": best_metrics,
                "rdkit_cosine_baseline": baseline_metrics,
            }
        )
        print(
            f"[{split_name}] fold {fold_index}: "
            f"rho={best_metrics['spearman']:.4f}, "
            f"baseline={baseline_metrics['spearman']:.4f}, "
            f"train={len(training)}, val={len(validation)}",
            flush=True,
        )
        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "protocol": (
            "Training pairs contain only non-held-out components; validation pairs "
            "contain at least one held-out component. Other components in a validation "
            "pair may be seen in training. Strict all-components-held-out counts are "
            "reported separately and are not substituted when too sparse."
        ),
        "folds": records,
        "fold_mean_spearman": float(np.mean(fold_model_spearman)),
        "fold_std_spearman": float(np.std(fold_model_spearman, ddof=1)),
        "baseline_fold_mean_spearman": float(np.mean(fold_baseline_spearman)),
        "baseline_fold_std_spearman": float(
            np.std(fold_baseline_spearman, ddof=1)
        ),
        "fold_mean_delta": float(
            np.mean(np.asarray(fold_model_spearman) - np.asarray(fold_baseline_spearman))
        ),
        "pooled_model": metric_summary(all_model_predictions, all_targets),
        "pooled_rdkit_cosine_baseline": metric_summary(
            all_baseline_predictions, all_targets
        ),
        "total_component_leakage_count": int(
            sum(int(record["molecule_leakage_count"]) for record in records)
        ),
    }


def run_strict_disjoint_validation(
    *,
    split_name: str,
    molecules: Sequence[str],
    pairs: Sequence[MixturePair],
    raw_cache: dict[str, np.ndarray],
    epochs: int,
    repeats: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    scaffold_grouped: bool,
) -> dict[str, object]:
    """Evaluate on pairs whose every component is absent from training.

    A 50/50 component partition is used because strict validation pairs are
    too sparse under the manuscript's five-way component-cold partition.
    Mixed pairs crossing the partition are discarded, not reassigned.
    Fixed-epoch, single-seed training avoids selecting on the strict holdout.
    """
    records: list[dict[str, object]] = []
    all_model_predictions: list[float] = []
    all_baseline_predictions: list[float] = []
    all_targets: list[float] = []
    fold_deltas: list[float] = []
    nonfinite_batches = 0
    for repeat in range(repeats):
        split_seed = seed + repeat
        folds = (
            scaffold_folds(molecules, n_splits=2, seed=split_seed)
            if scaffold_grouped
            else molecule_folds(molecules, n_splits=2, seed=split_seed)
        )
        held_out = folds[0]
        training, _, strict_validation, used_training = split_pairs(pairs, held_out)
        if len(training) < 10 or len(strict_validation) < 10:
            raise RuntimeError(
                f"{split_name} repeat {repeat + 1} is too sparse: "
                f"{len(training)} train / {len(strict_validation)} strict validation"
            )
        validation_molecules = {
            molecule for pair in strict_validation for molecule in pair.molecules
        }
        molecule_leakage = used_training & validation_molecules
        training_scaffolds = {
            bemis_murcko_scaffold(value) for value in used_training
        }
        validation_scaffolds = {
            bemis_murcko_scaffold(value) for value in validation_molecules
        }
        scaffold_leakage = (
            training_scaffolds & validation_scaffolds if scaffold_grouped else set()
        )
        if molecule_leakage or scaffold_leakage:
            raise RuntimeError(f"{split_name} strict leakage detected")
        normalizer = fit_normalizer(raw_cache, used_training)
        cache = normalized_cache(raw_cache, normalizer)
        run_seed = seed * 100_000 + repeat
        model, training_metrics = train_fixed_epochs(
            training,
            cache,
            seed=run_seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        model_metrics, model_predictions = evaluate_model(
            model, strict_validation, cache, device, batch_size
        )
        baseline_predictions = descriptor_cosine_predictions(
            strict_validation, cache
        )
        targets = [pair.similarity for pair in strict_validation]
        baseline_metrics = metric_summary(baseline_predictions, targets)
        delta = float(model_metrics["spearman"] - baseline_metrics["spearman"])
        fold_deltas.append(delta)
        all_model_predictions.extend(model_predictions)
        all_baseline_predictions.extend(baseline_predictions)
        all_targets.extend(targets)
        nonfinite_batches += int(training_metrics["nonfinite_training_batches"])
        records.append(
            {
                "repeat": repeat + 1,
                "split_seed": split_seed,
                "n_training_pairs": len(training),
                "n_validation_pairs": len(strict_validation),
                "n_discarded_cross_partition_pairs": (
                    len(pairs) - len(training) - len(strict_validation)
                ),
                "n_training_molecules": len(used_training),
                "n_validation_molecules": len(validation_molecules),
                "molecule_leakage_count": len(molecule_leakage),
                "scaffold_leakage_count": len(scaffold_leakage),
                "held_out_component_sha256": sha256_json(sorted(held_out)),
                "training": training_metrics,
                "model": model_metrics,
                "rdkit_cosine_baseline": baseline_metrics,
                "spearman_delta": delta,
            }
        )
        print(
            f"[{split_name}] repeat {repeat + 1}: rho={model_metrics['spearman']:.4f}, "
            f"baseline={baseline_metrics['spearman']:.4f}, "
            f"train={len(training)}, val={len(strict_validation)}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pooled_model = metric_summary(all_model_predictions, all_targets)
    pooled_baseline = metric_summary(all_baseline_predictions, all_targets)
    return {
        "protocol": (
            "Repeated 50/50 component partition. Training pairs contain only "
            "training-side components; validation pairs contain only held-out-side "
            "components; all cross-partition pairs are discarded."
        ),
        "repeats": records,
        "fold_mean_spearman_delta": float(np.mean(fold_deltas)),
        "pooled_model": pooled_model,
        "pooled_rdkit_cosine_baseline": pooled_baseline,
        "pooled_spearman_delta": float(
            pooled_model["spearman"] - pooled_baseline["spearman"]
        ),
        "total_molecule_leakage_count": int(
            sum(int(record["molecule_leakage_count"]) for record in records)
        ),
        "total_scaffold_leakage_count": int(
            sum(int(record["scaffold_leakage_count"]) for record in records)
        ),
        "nonfinite_training_batches": nonfinite_batches,
    }


def source_hashes(data_root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for relative in (
        "snitz_2013/molecules.csv",
        "snitz_2013/behavior.csv",
        "ravia_2020/molecules.csv",
        "ravia_2020/stimuli.csv",
        "ravia_2020/behavior_2.csv",
    ):
        path = data_root / relative
        result[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "fragrance_ai" / "data",
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "benchmarks" / "physsim_r2_validation.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--full-restarts", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strict-repeats", type=int, default=5)
    parser.add_argument(
        "--strict-validation-only",
        action="store_true",
        help="append strict disjoint validation to an existing checkpoint/report",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 6 or args.full_epochs < 6:
        raise SystemExit("epochs must be at least 6 to include warmup and cosine phases")
    if args.restarts < 1 or args.full_restarts < 1 or args.strict_repeats < 1:
        raise SystemExit("restart counts must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    started = time.time()
    snitz_pairs = load_snitz_pairs(args.data_root)
    ravia_pairs = load_ravia_pairs(args.data_root)
    snitz_molecules = sorted(
        {molecule for pair in snitz_pairs for molecule in pair.molecules}
    )
    all_molecules = sorted(
        {
            molecule
            for pair in (*snitz_pairs, *ravia_pairs)
            for molecule in pair.molecules
        }
    )
    print(
        f"device={device}; Snitz={len(snitz_pairs)} pairs/{len(snitz_molecules)} molecules; "
        f"Ravia={len(ravia_pairs)} pairs",
        flush=True,
    )
    raw_cache = build_raw_descriptor_cache(all_molecules)
    if args.strict_validation_only:
        strict_molecule = run_strict_disjoint_validation(
            split_name="strict_molecule_disjoint",
            molecules=snitz_molecules,
            pairs=snitz_pairs,
            raw_cache=raw_cache,
            epochs=args.epochs,
            repeats=args.strict_repeats,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            scaffold_grouped=False,
        )
        strict_scaffold = run_strict_disjoint_validation(
            split_name="strict_scaffold_disjoint",
            molecules=snitz_molecules,
            pairs=snitz_pairs,
            raw_cache=raw_cache,
            epochs=args.epochs,
            repeats=args.strict_repeats,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            scaffold_grouped=True,
        )
        report = json.loads(args.report.read_text(encoding="utf-8"))
        manifest_path = args.output_dir / "physsim_r2_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        minimum_delta = float(
            manifest["release_gate"]["minimum_spearman_delta_each_protocol"]
        )
        strict_checks = {
            "strict_molecule_disjoint_improves_baseline": (
                float(strict_molecule["pooled_spearman_delta"]) >= minimum_delta
                and float(strict_molecule["fold_mean_spearman_delta"]) >= minimum_delta
            ),
            "strict_scaffold_disjoint_improves_baseline": (
                float(strict_scaffold["pooled_spearman_delta"]) >= minimum_delta
                and float(strict_scaffold["fold_mean_spearman_delta"]) >= minimum_delta
            ),
            "strict_molecule_disjoint_has_zero_leakage": (
                int(strict_molecule["total_molecule_leakage_count"]) == 0
            ),
            "strict_scaffold_disjoint_has_zero_leakage": (
                int(strict_scaffold["total_molecule_leakage_count"]) == 0
                and int(strict_scaffold["total_scaffold_leakage_count"]) == 0
            ),
            "strict_disjoint_training_had_no_nonfinite_batches": (
                int(strict_molecule["nonfinite_training_batches"]) == 0
                and int(strict_scaffold["nonfinite_training_batches"]) == 0
            ),
        }
        report["strict_molecule_disjoint"] = strict_molecule
        report["strict_scaffold_disjoint"] = strict_scaffold
        report["training_configuration"]["strict_repeats"] = args.strict_repeats
        report["release_gate"]["checks"].update(strict_checks)
        report["release_gate"]["passed"] = all(
            bool(value) for value in report["release_gate"]["checks"].values()
        )
        report["release_gate"]["approved_primary_score_weight"] = (
            0.10 if report["release_gate"]["passed"] else 0.0
        )
        manifest["release_gate"]["checks"].update(strict_checks)
        manifest["release_gate"]["passed"] = report["release_gate"]["passed"]
        manifest["release_gate"]["approved_primary_score_weight"] = report[
            "release_gate"
        ]["approved_primary_score_weight"]
        manifest["release_gate"]["fallback_behavior"] = (
            "learned R2 score contributes only as a centered residual within applicability"
            if manifest["release_gate"]["passed"]
            else "weight remains zero; concentration/headspace deterministic PhysSim remains primary"
        )
        manifest["ensemble_calibration"] = {
            "method": "centered_residual_on_primary_score",
            "neutral_similarity_percent": float(
                np.mean([pair.similarity for pair in snitz_pairs]) * 100.0
            ),
            "formula": (
                "primary_score + approved_weight * "
                "(r2_similarity_percent - neutral_similarity_percent)"
            ),
            "reason": (
                "Historical mixture-similarity labels and the text/headspace proxy "
                "do not share an absolute intercept."
            ),
        }
        manifest["dataset_provenance"] = {
            "archive": "Pyrfume Public Data Archive",
            "archive_license": "MIT",
            "snitz": "Snitz et al. 2013 mixture similarity",
            "ravia": "Ravia et al. 2020 external transfer set",
            "claim_boundary": (
                "Historical mixture-pair labels; not generated-formula validation."
            ),
        }
        manifest["strict_disjoint_validation"] = {
            "report_file": args.report.name,
            "repeats": args.strict_repeats,
            "molecule_pooled_spearman_delta": strict_molecule[
                "pooled_spearman_delta"
            ],
            "molecule_fold_mean_spearman_delta": strict_molecule[
                "fold_mean_spearman_delta"
            ],
            "scaffold_pooled_spearman_delta": strict_scaffold[
                "pooled_spearman_delta"
            ],
            "scaffold_fold_mean_spearman_delta": strict_scaffold[
                "fold_mean_spearman_delta"
            ],
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "strict_validation_only": True,
                    "release_gate": manifest["release_gate"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if manifest["release_gate"]["passed"] else 2
    molecule_cv = run_cross_validation(
        split_name="molecule_cold",
        held_out_folds=molecule_folds(
            snitz_molecules, n_splits=args.folds, seed=args.seed
        ),
        pairs=snitz_pairs,
        raw_cache=raw_cache,
        epochs=args.epochs,
        restarts=args.restarts,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    scaffold_cv = run_cross_validation(
        split_name="bemis_murcko_scaffold_cold",
        held_out_folds=scaffold_folds(
            snitz_molecules, n_splits=args.folds, seed=args.seed
        ),
        pairs=snitz_pairs,
        raw_cache=raw_cache,
        epochs=args.epochs,
        restarts=args.restarts,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        device=device,
    )
    strict_molecule = run_strict_disjoint_validation(
        split_name="strict_molecule_disjoint",
        molecules=snitz_molecules,
        pairs=snitz_pairs,
        raw_cache=raw_cache,
        epochs=args.epochs,
        repeats=args.strict_repeats,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        scaffold_grouped=False,
    )
    strict_scaffold = run_strict_disjoint_validation(
        split_name="strict_scaffold_disjoint",
        molecules=snitz_molecules,
        pairs=snitz_pairs,
        raw_cache=raw_cache,
        epochs=args.epochs,
        repeats=args.strict_repeats,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        scaffold_grouped=True,
    )

    normalizer = fit_normalizer(raw_cache, snitz_molecules)
    cache = normalized_cache(raw_cache, normalizer)
    production_models: list[tuple[R2PhysSimCore, dict[str, float | int], dict[str, float]]] = []
    for restart in range(args.full_restarts):
        run_seed = args.seed * 100_000 + restart
        model, training = train_fixed_epochs(
            snitz_pairs,
            cache,
            seed=run_seed,
            epochs=args.full_epochs,
            batch_size=args.batch_size,
            device=device,
        )
        training_metrics, _ = evaluate_model(
            model, snitz_pairs, cache, device, args.batch_size
        )
        production_models.append((model, training, training_metrics))
        print(
            f"[full] restart {restart + 1}: loss={training['final_training_loss']:.6f}, "
            f"train_rho={training_metrics['spearman']:.4f}",
            flush=True,
        )
    # Selection uses Snitz training loss only. Ravia remains untouched until
    # after the checkpoint is selected.
    selected_index = min(
        range(len(production_models)),
        key=lambda index: float(
            production_models[index][1]["final_training_loss"]
        ),
    )
    selected_model, selected_training, selected_train_metrics = production_models[
        selected_index
    ]
    ravia_model_metrics, _ = evaluate_model(
        selected_model, ravia_pairs, cache, device, args.batch_size
    )
    ravia_baseline_predictions = descriptor_cosine_predictions(ravia_pairs, cache)
    ravia_baseline_metrics = metric_summary(
        ravia_baseline_predictions, [pair.similarity for pair in ravia_pairs]
    )

    minimum_delta = 0.01
    gate_checks = {
        "molecule_cold_improves_baseline": (
            float(molecule_cv["fold_mean_delta"]) >= minimum_delta
        ),
        "scaffold_cold_improves_baseline": (
            float(scaffold_cv["fold_mean_delta"]) >= minimum_delta
        ),
        "strict_molecule_disjoint_improves_baseline": (
            float(strict_molecule["pooled_spearman_delta"]) >= minimum_delta
            and float(strict_molecule["fold_mean_spearman_delta"]) >= minimum_delta
        ),
        "strict_scaffold_disjoint_improves_baseline": (
            float(strict_scaffold["pooled_spearman_delta"]) >= minimum_delta
            and float(strict_scaffold["fold_mean_spearman_delta"]) >= minimum_delta
        ),
        "ravia_transfer_improves_baseline": (
            ravia_model_metrics["spearman"]
            - ravia_baseline_metrics["spearman"]
            >= minimum_delta
        ),
        "molecule_split_has_zero_component_leakage": (
            int(molecule_cv["total_component_leakage_count"]) == 0
        ),
        "scaffold_split_has_zero_component_leakage": (
            int(scaffold_cv["total_component_leakage_count"]) == 0
        ),
        "strict_molecule_disjoint_has_zero_leakage": (
            int(strict_molecule["total_molecule_leakage_count"]) == 0
        ),
        "strict_scaffold_disjoint_has_zero_leakage": (
            int(strict_scaffold["total_molecule_leakage_count"]) == 0
            and int(strict_scaffold["total_scaffold_leakage_count"]) == 0
        ),
        "training_had_no_nonfinite_batches": (
            int(selected_training["nonfinite_training_batches"]) == 0
            and int(strict_molecule["nonfinite_training_batches"]) == 0
            and int(strict_scaffold["nonfinite_training_batches"]) == 0
        ),
    }
    release_gate_passed = all(gate_checks.values())
    approved_weight = 0.10 if release_gate_passed else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "physsim_r2_checkpoint.pt"
    checkpoint_payload = {
        "model_spec_version": MODEL_SPEC_VERSION,
        "model_state_dict": _model_state_on_cpu(selected_model),
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
            "dataset": "Snitz 2013 mixture similarity",
            "selected_restart": selected_index + 1,
            "selection_rule": "lowest final Snitz training loss; external sets not used",
            "seed": args.seed * 100_000 + selected_index,
            **selected_training,
        },
        "learned_constants": selected_model.learned_constants(),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = source_hashes(args.data_root)
    manifest = {
        "schema_version": "1.0",
        "model_spec_version": MODEL_SPEC_VERSION,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "generated_at": generated_at,
        "descriptor_contract_sha256": sha256_json(
            normalizer.descriptor_names
        ),
        "source_files": sources,
        "dataset_provenance": {
            "archive": "Pyrfume Public Data Archive",
            "archive_license": "MIT",
            "snitz": "Snitz et al. 2013 mixture similarity",
            "ravia": "Ravia et al. 2020 external transfer set",
            "claim_boundary": (
                "Historical mixture-pair labels; not generated-formula validation."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "ensemble_calibration": {
            "method": "centered_residual_on_primary_score",
            "neutral_similarity_percent": float(
                np.mean([pair.similarity for pair in snitz_pairs]) * 100.0
            ),
            "formula": (
                "primary_score + approved_weight * "
                "(r2_similarity_percent - neutral_similarity_percent)"
            ),
            "reason": (
                "Historical mixture-similarity labels and the text/headspace proxy "
                "do not share an absolute intercept."
            ),
        },
        "strict_disjoint_validation": {
            "report_file": args.report.name,
            "repeats": args.strict_repeats,
            "molecule_pooled_spearman_delta": strict_molecule[
                "pooled_spearman_delta"
            ],
            "molecule_fold_mean_spearman_delta": strict_molecule[
                "fold_mean_spearman_delta"
            ],
            "scaffold_pooled_spearman_delta": strict_scaffold[
                "pooled_spearman_delta"
            ],
            "scaffold_fold_mean_spearman_delta": strict_scaffold[
                "fold_mean_spearman_delta"
            ],
        },
        "release_gate": {
            "passed": release_gate_passed,
            "minimum_spearman_delta_each_protocol": minimum_delta,
            "checks": gate_checks,
            "approved_primary_score_weight": approved_weight,
            "fallback_behavior": (
                "weight remains zero and the concentration/headspace deterministic "
                "PhysSim continues as the production ranking signal"
                if not release_gate_passed
                else "learned R2 contributes at most a 10% centered residual within applicability"
            ),
        },
        "limitations": [
            "Similarity labels are historical human-panel observations; no new human validation was performed.",
            "Component-cold validation guarantees held-out components are absent from training pairs, but a validation pair may also contain seen components.",
            "Strict disjoint validation discards cross-partition pairs and therefore has fewer validation pairs and wider statistical uncertainty.",
            "R2 predicts mixture-pair similarity and is not validated for monomolecular odor character or text-to-odor accuracy.",
            "A non-zero ensemble weight does not establish 90% real-world olfactory equivalence.",
        ],
    }
    manifest_path = args.output_dir / "physsim_r2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "elapsed_seconds": round(time.time() - started, 3),
        "model_spec_version": MODEL_SPEC_VERSION,
        "architecture_parameter_count": EXPECTED_PARAMETER_COUNT,
        "data": {
            "snitz_pairs": len(snitz_pairs),
            "snitz_molecules": len(snitz_molecules),
            "ravia_pairs": len(ravia_pairs),
            "source_files": sources,
        },
        "training_configuration": {
            "seed": args.seed,
            "folds": args.folds,
            "epochs": args.epochs,
            "restarts": args.restarts,
            "strict_repeats": args.strict_repeats,
            "full_epochs": args.full_epochs,
            "full_restarts": args.full_restarts,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "molecule_cold": molecule_cv,
        "bemis_murcko_scaffold_cold": scaffold_cv,
        "strict_molecule_disjoint": strict_molecule,
        "strict_scaffold_disjoint": strict_scaffold,
        "production_checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "selection_rule": "lowest final Snitz training loss",
            "selected_restart": selected_index + 1,
            "training": selected_training,
            "snitz_training_metrics": selected_train_metrics,
            "learned_constants": selected_model.learned_constants(),
        },
        "zero_shot_ravia": {
            "model": ravia_model_metrics,
            "rdkit_cosine_baseline": ravia_baseline_metrics,
            "spearman_delta": (
                ravia_model_metrics["spearman"]
                - ravia_baseline_metrics["spearman"]
            ),
            "used_for_model_selection": False,
        },
        "release_gate": manifest["release_gate"],
        "scientific_claim_boundary": (
            "A validation-gated non-human mixture-similarity signal; not proof of "
            "90% human olfactory equivalence."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"checkpoint={checkpoint_path} sha256={checkpoint_sha256}\n"
        f"release_gate={release_gate_passed} approved_weight={approved_weight:.2f}\n"
        f"report={args.report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
