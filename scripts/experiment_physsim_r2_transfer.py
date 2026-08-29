#!/usr/bin/env python
"""Develop a leakage-free odor-ontology transfer recipe for R2 PhysSim.

This script is intentionally separate from the release trainer.  It uses a
fixed development split to choose a transfer recipe without touching the
later final seeds or Ravia labels.  Pretraining excludes every Snitz molecule
for molecule-disjoint work, and every Snitz scaffold for scaffold-disjoint
work, including descriptor-normalizer fitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    DESCRIPTOR_DIM,
    MixturePair,
    MixturePairDataset,
    OdorDescriptorRecord,
    R2PhysSimCore,
    PU_LABEL_CONTRACT_VERSION,
    STRICT_SPLIT_CONTRACT_VERSION,
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    combined_loss,
    descriptor_contract,
    fit_normalizer,
    load_odor_descriptor_records,
    load_snitz_pairs,
    normalized_cache,
    positive_unlabeled_descriptor_loss,
    set_deterministic_seed,
    sha256_json,
    symmetric_augmentation,
)
from scripts.train_physsim_r2 import (  # noqa: E402
    descriptor_cosine_predictions,
    evaluate_model,
    metric_summary,
    molecule_folds,
    scaffold_folds,
    split_pairs,
)


@dataclass(frozen=True)
class TransferConfig:
    name: str
    trainable: str
    fine_tune_epochs: int
    learning_rate: float
    baseline_anchor: float = 0.0


CONFIGURATIONS = (
    TransferConfig("head_only", "head", 80, 3e-4),
    TransferConfig("projection_head", "projection_head", 60, 1e-4),
    TransferConfig("all_low_lr", "all", 40, 3e-5),
    TransferConfig("all_anchor_20", "all", 40, 3e-5, 0.20),
)


def load_or_build_raw_cache(
    molecules: set[str], cache_path: Path | None
) -> dict[str, np.ndarray]:
    ordered = sorted(molecules)
    if cache_path is not None and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_smiles = [str(value) for value in payload["smiles"]]
        descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
        if cached_smiles == ordered and descriptors.shape == (
            len(ordered),
            DESCRIPTOR_DIM,
        ):
            print(f"raw descriptor cache hit: {cache_path}", flush=True)
            return {
                smiles: descriptors[index]
                for index, smiles in enumerate(cached_smiles)
            }
    result = build_raw_descriptor_cache(ordered)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            smiles=np.asarray(ordered, dtype=str),
            descriptors=np.asarray([result[value] for value in ordered]),
        )
        print(f"raw descriptor cache written: {cache_path}", flush=True)
    return result


class DescriptorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[OdorDescriptorRecord],
        cache: dict[str, np.ndarray],
    ) -> None:
        self.records = list(records)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        return (
            torch.from_numpy(self.cache[record.smiles]),
            torch.tensor(record.positive_observation_mask, dtype=torch.float32),
        )


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def pretrain_descriptor_encoder(
    model: R2PhysSimCore,
    records: Sequence[OdorDescriptorRecord],
    cache: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        DescriptorDataset(records, cache),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    positive_matrix = np.asarray(
        [record.positive_observation_mask for record in records], dtype=np.float32
    )
    # This is an empirical positive prevalence, not a negative-label rate.
    # The PU loss below keeps absent terms unlabeled and only applies this
    # estimate at the population-risk correction level.
    class_prior = torch.tensor(
        np.clip(positive_matrix.mean(axis=0), 1e-4, 1.0 - 1e-4),
        dtype=torch.float32,
        device=device,
    )
    classifier = nn.Linear(128, label_matrix.shape[1]).to(device)
    optimizer = optim.AdamW(
        [
            {"params": model.chemical_encoder.parameters(), "lr": 5e-4},
            {"params": classifier.parameters(), "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    losses: list[float] = []
    finite_batches = 0
    nonfinite_batches = 0
    for _ in range(epochs):
        model.chemical_encoder.train()
        classifier.train()
        epoch_losses: list[float] = []
        for descriptors, positive_mask in loader:
            logits = classifier(model.chemical_encoder(descriptors.to(device)))
            loss = positive_unlabeled_descriptor_loss(
                logits, positive_mask.to(device), class_prior
            )
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.chemical_encoder.parameters(), *classifier.parameters()], 1.0
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            finite_batches += 1
        scheduler.step()
        if not epoch_losses:
            raise RuntimeError("descriptor pretraining produced no finite batches")
        losses.append(float(np.mean(epoch_losses)))
    return {
        "epochs": epochs,
        "n_records": len(records),
        "objective": "non_negative_positive_unlabeled_risk",
        "label_contract_version": PU_LABEL_CONTRACT_VERSION,
        "final_pu_risk": losses[-1],
        "minimum_pu_risk": min(losses),
        "finite_batches": finite_batches,
        "nonfinite_batches": nonfinite_batches,
    }


def make_ontology_similarity_pairs(
    records: Sequence[OdorDescriptorRecord],
    *,
    seed: int,
    pairs_per_molecule: int = 4,
) -> list[MixturePair]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([record.labels for record in records], dtype=np.float32)
    norms = np.linalg.norm(labels, axis=1)
    by_label = [np.flatnonzero(labels[:, index] > 0.5) for index in range(labels.shape[1])]
    pairs: list[MixturePair] = []
    seen: set[tuple[int, int]] = set()
    for first, record in enumerate(records):
        positive_labels = np.flatnonzero(labels[first] > 0.5)
        candidates: list[int] = []
        if len(positive_labels):
            for _ in range(max(1, pairs_per_molecule // 2)):
                label = int(rng.choice(positive_labels))
                pool = by_label[label]
                if len(pool) > 1:
                    candidates.append(int(rng.choice(pool)))
        for _ in range(pairs_per_molecule - len(candidates)):
            candidates.append(int(rng.integers(0, len(records))))
        for second in candidates:
            if first == second:
                continue
            key = tuple(sorted((first, second)))
            if key in seen:
                continue
            seen.add(key)
            denominator = float(norms[first] * norms[second])
            similarity = (
                float(np.dot(labels[first], labels[second]) / denominator)
                if denominator > 1e-8
                else 0.0
            )
            pairs.append(
                MixturePair(
                    (record.smiles,),
                    (records[second].smiles,),
                    similarity,
                    f"ontology:{first}:{second}",
                )
            )
    return pairs


def pretrain_similarity_model(
    model: R2PhysSimCore,
    pairs: Sequence[MixturePair],
    cache: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        MixturePairDataset(pairs, cache, max_molecules=1),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = optim.AdamW(
        [
            {"params": model.chemical_encoder.parameters(), "lr": 5e-5},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("chemical_encoder.")
                ],
                "lr": 2e-4,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    final_loss = math.inf
    finite_batches = 0
    nonfinite_batches = 0
    for _ in range(epochs):
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
        scheduler.step()
        if not losses:
            raise RuntimeError("ontology similarity pretraining produced no finite batches")
        final_loss = float(np.mean(losses))
    return {
        "epochs": epochs,
        "n_pairs": len(pairs),
        "final_loss": final_loss,
        "finite_batches": finite_batches,
        "nonfinite_batches": nonfinite_batches,
    }


def _trainable_parameters(
    model: R2PhysSimCore, mode: str
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        modules = (model.similarity_head,)
    elif mode == "projection_head":
        modules = (model.fingerprint_projection, model.similarity_head)
    elif mode == "all":
        modules = (model,)
    else:
        raise ValueError(mode)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def fine_tune_snitz(
    base_state: dict[str, torch.Tensor],
    pairs: Sequence[MixturePair],
    cache: dict[str, np.ndarray],
    config: TransferConfig,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[R2PhysSimCore, dict[str, object]]:
    model = R2PhysSimCore().to(device)
    model.load_state_dict(base_state)
    training_pairs = list(pairs)
    if config.baseline_anchor > 0:
        baseline = descriptor_cosine_predictions(training_pairs, cache)
        training_pairs = [
            MixturePair(
                pair.mixture_a,
                pair.mixture_b,
                (1.0 - config.baseline_anchor) * pair.similarity
                + config.baseline_anchor * baseline[index],
                pair.record_id,
            )
            for index, pair in enumerate(training_pairs)
        ]
    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        MixturePairDataset(symmetric_augmentation(training_pairs), cache),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    parameters = _trainable_parameters(model, config.trainable)
    optimizer = optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.fine_tune_epochs)
    )
    final_loss = math.inf
    finite_batches = 0
    nonfinite_batches = 0
    for _ in range(config.fine_tune_epochs):
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
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            finite_batches += 1
        scheduler.step()
        if not losses:
            raise RuntimeError("Snitz fine-tuning produced no finite batches")
        final_loss = float(np.mean(losses))
    return model, {
        "final_loss": final_loss,
        "finite_batches": finite_batches,
        "nonfinite_batches": nonfinite_batches,
        "trainable_parameter_count": sum(value.numel() for value in parameters),
    }


def protocol_pretraining_records(
    records: Sequence[OdorDescriptorRecord],
    snitz_molecules: set[str],
    *,
    scaffold_grouped: bool,
) -> list[OdorDescriptorRecord]:
    if not scaffold_grouped:
        return [record for record in records if record.smiles not in snitz_molecules]
    forbidden_scaffolds = {
        bemis_murcko_scaffold(smiles) for smiles in snitz_molecules
    }
    return [
        record
        for record in records
        if bemis_murcko_scaffold(record.smiles) not in forbidden_scaffolds
    ]


def run_protocol(
    *,
    name: str,
    snitz_pairs: Sequence[MixturePair],
    snitz_molecules: Sequence[str],
    records: Sequence[OdorDescriptorRecord],
    raw_cache: dict[str, np.ndarray],
    configurations: Sequence[TransferConfig],
    split_seed: int,
    model_seed: int,
    repeats: int,
    encoder_epochs: int,
    similarity_epochs: int,
    batch_size: int,
    device: torch.device,
    scaffold_grouped: bool,
) -> dict[str, object]:
    snitz_set = set(snitz_molecules)
    pretraining_records = protocol_pretraining_records(
        records, snitz_set, scaffold_grouped=scaffold_grouped
    )
    normalizer = fit_normalizer(
        raw_cache, [record.smiles for record in pretraining_records]
    )
    model_cache = normalized_cache(raw_cache, normalizer)
    set_deterministic_seed(model_seed)
    base_model = R2PhysSimCore().to(device)
    encoder_training = pretrain_descriptor_encoder(
        base_model,
        pretraining_records,
        model_cache,
        epochs=encoder_epochs,
        batch_size=max(64, batch_size * 16),
        seed=model_seed + 1,
        device=device,
    )
    ontology_pairs = make_ontology_similarity_pairs(
        pretraining_records, seed=model_seed + 2
    )
    similarity_training = pretrain_similarity_model(
        base_model,
        ontology_pairs,
        model_cache,
        epochs=similarity_epochs,
        batch_size=max(32, batch_size * 8),
        seed=model_seed + 3,
        device=device,
    )
    base_state = _cpu_state(base_model)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_configuration: dict[str, dict[str, object]] = {}
    for config in configurations:
        records_out: list[dict[str, object]] = []
        all_predictions: list[float] = []
        all_targets: list[float] = []
        all_baseline: list[float] = []
        deltas: list[float] = []
        for repeat in range(repeats):
            seed = split_seed + repeat
            folds = (
                scaffold_folds(snitz_molecules, n_splits=2, seed=seed)
                if scaffold_grouped
                else molecule_folds(snitz_molecules, n_splits=2, seed=seed)
            )
            held_out = folds[0]
            training, mixed_component_validation, strict_validation, used_training = split_pairs(
                snitz_pairs, held_out
            )
            # ``mixed_component_validation`` contains pairs that include at
            # least one held-out molecule but may retain seen components.  It
            # is diagnostically useful, but is explicitly not the validation
            # set for this protocol.  Every reported score here uses pairs
            # whose *all* components are in the held-out group.
            validation = strict_validation
            if not validation:
                raise RuntimeError(
                    f"{name} has no all-components-held-out validation pairs"
                )
            validation_molecules = {
                molecule for pair in validation for molecule in pair.molecules
            }
            molecule_leakage = validation_molecules & used_training
            pretrain_molecule_leakage = validation_molecules & {
                record.smiles for record in pretraining_records
            }
            validation_scaffolds = {
                bemis_murcko_scaffold(value) for value in validation_molecules
            }
            pretraining_scaffolds = {
                bemis_murcko_scaffold(record.smiles)
                for record in pretraining_records
            }
            scaffold_leakage = (
                validation_scaffolds & pretraining_scaffolds
                if scaffold_grouped
                else set()
            )
            if molecule_leakage or pretrain_molecule_leakage or scaffold_leakage:
                raise RuntimeError(f"{name} leakage detected")
            model, training_stats = fine_tune_snitz(
                base_state,
                training,
                model_cache,
                config,
                seed=seed * 10_000 + 17,
                batch_size=batch_size,
                device=device,
            )
            metrics, predictions = evaluate_model(
                model, validation, model_cache, device, batch_size
            )
            baseline_normalizer = fit_normalizer(raw_cache, used_training)
            baseline_cache = normalized_cache(raw_cache, baseline_normalizer)
            baseline_predictions = descriptor_cosine_predictions(
                validation, baseline_cache
            )
            targets = [pair.similarity for pair in validation]
            baseline_metrics = metric_summary(baseline_predictions, targets)
            delta = float(metrics["spearman"] - baseline_metrics["spearman"])
            deltas.append(delta)
            all_predictions.extend(predictions)
            all_baseline.extend(baseline_predictions)
            all_targets.extend(targets)
            records_out.append(
                {
                    "repeat": repeat + 1,
                    "split_seed": seed,
                    "n_training_pairs": len(training),
                    "n_validation_pairs": len(validation),
                    "n_mixed_component_diagnostic_pairs": len(mixed_component_validation),
                    "validation_pair_contract": "all_components_held_out",
                    "n_pretraining_records": len(pretraining_records),
                    "training": training_stats,
                    "model": metrics,
                    "baseline": baseline_metrics,
                    "spearman_delta": delta,
                    "predictions": [float(value) for value in predictions],
                    "baseline_predictions": [
                        float(value) for value in baseline_predictions
                    ],
                    "targets": [float(value) for value in targets],
                    "molecule_leakage_count": 0,
                    "pretraining_molecule_leakage_count": 0,
                    "scaffold_leakage_count": 0,
                    "held_out_sha256": sha256_json(sorted(held_out)),
                }
            )
            print(
                f"[{name}/{config.name}] repeat {repeat + 1}: "
                f"rho={metrics['spearman']:.4f}, baseline={baseline_metrics['spearman']:.4f}, "
                f"delta={delta:+.4f}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pooled_model = metric_summary(all_predictions, all_targets)
        pooled_baseline = metric_summary(all_baseline, all_targets)
        by_configuration[config.name] = {
            "configuration": asdict(config),
            "repeats": records_out,
            "fold_mean_spearman_delta": float(np.mean(deltas)),
            "pooled_model": pooled_model,
            "pooled_baseline": pooled_baseline,
            "pooled_spearman_delta": float(
                pooled_model["spearman"] - pooled_baseline["spearman"]
            ),
            "pooled_predictions": [float(value) for value in all_predictions],
            "pooled_baseline_predictions": [
                float(value) for value in all_baseline
            ],
            "pooled_targets": [float(value) for value in all_targets],
        }
    return {
        "split_contract_version": STRICT_SPLIT_CONTRACT_VERSION,
        "validation_pair_contract": "all_components_held_out",
        "mixed_component_pairs_are_reported_as_diagnostic_only": True,
        "pretraining_exclusion": (
            "all Snitz Bemis-Murcko scaffolds"
            if scaffold_grouped
            else "all exact Snitz molecules"
        ),
        "n_pretraining_records": len(pretraining_records),
        "encoder_pretraining": encoder_training,
        "similarity_pretraining": similarity_training,
        "configurations": by_configuration,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-data-root", type=Path, required=True)
    parser.add_argument("--pyrfume-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_development.json",
    )
    parser.add_argument("--split-seed", type=int, default=142)
    parser.add_argument("--model-seed", type=int, default=20260715)
    parser.add_argument(
        "--phase", choices=("development", "final"), default="development"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--encoder-epochs", type=int, default=10)
    parser.add_argument("--similarity-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument(
        "--configs",
        default=",".join(config.name for config in CONFIGURATIONS),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--protocol", choices=("both", "molecule", "scaffold"), default="both"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    selected_names = {name.strip() for name in args.configs.split(",") if name.strip()}
    configurations = [
        config for config in CONFIGURATIONS if config.name in selected_names
    ]
    if not configurations:
        raise SystemExit("no recognized transfer configurations selected")
    started = time.time()
    vocabulary, descriptor_records, source_counts = load_odor_descriptor_records(
        args.pyrfume_root
    )
    snitz_pairs = load_snitz_pairs(args.mixture_data_root)
    snitz_molecules = sorted(
        {molecule for pair in snitz_pairs for molecule in pair.molecules}
    )
    all_molecules = {
        record.smiles for record in descriptor_records
    } | set(snitz_molecules)
    print(
        f"device={device}; ontology={len(vocabulary)} labels/"
        f"{len(descriptor_records)} molecules; Snitz={len(snitz_pairs)} pairs",
        flush=True,
    )
    raw_cache = load_or_build_raw_cache(all_molecules, args.raw_cache)
    descriptor_names = [name for name, _ in descriptor_contract()]
    molecular_weight_index = descriptor_names.index("MolWt")
    heavy_atom_index = descriptor_names.index("HeavyAtomCount")
    unfiltered_descriptor_count = len(descriptor_records)
    descriptor_records = [
        record
        for record in descriptor_records
        if 20.0 <= float(raw_cache[record.smiles][molecular_weight_index]) <= 750.0
        and float(raw_cache[record.smiles][heavy_atom_index]) <= 70.0
        and float(np.max(np.abs(raw_cache[record.smiles]))) < 1e25
    ]
    print(
        f"volatile-domain filter: {unfiltered_descriptor_count} -> "
        f"{len(descriptor_records)} pretraining molecules",
        flush=True,
    )
    molecule = run_protocol(
        name=f"{args.phase}_all_components_held_out_molecule_disjoint",
        snitz_pairs=snitz_pairs,
        snitz_molecules=snitz_molecules,
        records=descriptor_records,
        raw_cache=raw_cache,
        configurations=configurations,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        repeats=args.repeats,
        encoder_epochs=args.encoder_epochs,
        similarity_epochs=args.similarity_epochs,
        batch_size=args.batch_size,
        device=device,
        scaffold_grouped=False,
    ) if args.protocol in {"both", "molecule"} else None
    scaffold = run_protocol(
        name=f"{args.phase}_all_components_held_out_scaffold_disjoint",
        snitz_pairs=snitz_pairs,
        snitz_molecules=snitz_molecules,
        records=descriptor_records,
        raw_cache=raw_cache,
        configurations=configurations,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        repeats=args.repeats,
        encoder_epochs=args.encoder_epochs,
        similarity_epochs=args.similarity_epochs,
        batch_size=args.batch_size,
        device=device,
        scaffold_grouped=True,
    ) if args.protocol in {"both", "scaffold"} else None
    scores: dict[str, float] = {}
    for config in configurations:
        candidates: list[float] = []
        for protocol in (molecule, scaffold):
            if protocol is None:
                continue
            result = protocol["configurations"][config.name]
            candidates.extend(
                [
                    float(result["fold_mean_spearman_delta"]),
                    float(result["pooled_spearman_delta"]),
                ]
            )
        scores[config.name] = min(candidates)
    selected = max(scores, key=scores.get)
    report = {
        "purpose": (
            "frozen transfer recipe final strict evaluation"
            if args.phase == "final"
            else "fixed development-only transfer recipe selection"
        ),
        "phase": args.phase,
        "final_evaluation_seeds_touched": args.phase == "final",
        "ravia_labels_touched": False,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "repeats": args.repeats,
        "ontology_labels": list(vocabulary),
        "descriptor_source_record_counts": source_counts,
        "n_unique_descriptor_molecules": len(descriptor_records),
        "n_descriptor_molecules_before_volatile_domain_filter": (
            unfiltered_descriptor_count
        ),
        "volatile_domain_filter": {
            "molecular_weight_range": [20.0, 750.0],
            "maximum_heavy_atoms": 70.0,
            "maximum_absolute_raw_descriptor": 1e25,
        },
        "molecule_disjoint": molecule,
        "scaffold_disjoint": scaffold,
        "split_contract": {
            "version": STRICT_SPLIT_CONTRACT_VERSION,
            "reported_validation_pairs": "all_components_held_out",
            "mixed_component_pairs": "diagnostic_only_not_used_for_scores_or_selection",
            "molecule_disjoint": (
                molecule.get("validation_pair_contract") if molecule is not None else None
            ),
            "scaffold_disjoint": (
                scaffold.get("validation_pair_contract") if scaffold is not None else None
            ),
        },
        "selection_rule": "maximum worst-case of pooled and fold-mean deltas across both protocols",
        "selection_scores": scores,
        "selected_configuration": selected,
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "scores": scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
