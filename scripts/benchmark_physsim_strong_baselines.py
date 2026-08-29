#!/usr/bin/env python
"""Leakage-free strong baselines for frozen R2 mixture-similarity evidence.

Algorithms are selected only on the development seeds.  The selected baseline
is then evaluated on the already frozen final seeds; final labels never choose
the algorithm or its hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    MixturePair,
    fit_normalizer,
    load_snitz_pairs,
    normalized_cache,
    sha256_file,
)
from scripts.train_physsim_r2 import (  # noqa: E402
    metric_summary,
    molecule_folds,
    scaffold_folds,
    split_pairs,
)


ALGORITHMS = (
    "morgan_tanimoto",
    "ridge_pair_descriptors",
    "random_forest_pair_descriptors",
    "extra_trees_pair_descriptors",
    "hist_gradient_boosting_pair_descriptors",
)


def _load_raw_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            str(smiles): np.asarray(row, dtype=np.float32)
            for smiles, row in zip(data["smiles"], data["descriptors"])
        }


def _mixture_descriptor(
    pair_values: frozenset[str], cache: dict[str, np.ndarray]
) -> np.ndarray:
    rows = np.asarray([cache[value] for value in sorted(pair_values)], dtype=np.float32)
    return np.concatenate(
        (
            np.mean(rows, axis=0),
            np.std(rows, axis=0),
            np.asarray([np.log1p(len(rows))], dtype=np.float32),
        )
    )


def _pair_features(pair: MixturePair, cache: dict[str, np.ndarray]) -> np.ndarray:
    first = _mixture_descriptor(pair.mixture_a, cache)
    second = _mixture_descriptor(pair.mixture_b, cache)
    return np.nan_to_num(
        np.concatenate((np.abs(first - second), first * second)),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    ).astype(np.float32)


def _morgan_cache(molecules: set[str]):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = {}
    for smiles in sorted(molecules):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES in frozen dataset: {smiles}")
        result[smiles] = generator.GetFingerprint(molecule)
    return result


def _morgan_predictions(pairs: list[MixturePair], cache) -> list[float]:
    predictions = []
    for pair in pairs:
        first = cache[next(iter(sorted(pair.mixture_a)))].__class__(2048)
        second = cache[next(iter(sorted(pair.mixture_b)))].__class__(2048)
        for molecule in pair.mixture_a:
            first |= cache[molecule]
        for molecule in pair.mixture_b:
            second |= cache[molecule]
        predictions.append(float(DataStructs.TanimotoSimilarity(first, second)))
    return predictions


def _model(name: str, seed: int):
    if name == "ridge_pair_descriptors":
        return Ridge(alpha=20.0)
    if name == "random_forest_pair_descriptors":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=2,
            max_features=0.45,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra_trees_pair_descriptors":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.55,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting_pair_descriptors":
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=4,
            l2_regularization=2.0,
            random_state=seed,
        )
    raise KeyError(name)


def _protocol(
    *,
    pairs: list[MixturePair],
    molecules: list[str],
    raw_cache: dict[str, np.ndarray],
    morgan_cache,
    split_seed: int,
    repeats: int,
    scaffold: bool,
) -> dict:
    pooled = {name: {"predictions": [], "targets": [], "fold_rho": []} for name in ALGORITHMS}
    repeat_rows = []
    for repeat in range(repeats):
        seed = split_seed + repeat
        folds = (
            scaffold_folds(molecules, n_splits=2, seed=seed)
            if scaffold
            else molecule_folds(molecules, n_splits=2, seed=seed)
        )
        held_out = folds[0]
        training, _, validation, used_training = split_pairs(pairs, held_out)
        held_out_used = {value for pair in validation for value in pair.molecules}
        if used_training & held_out_used:
            raise RuntimeError("molecule leakage in strong-baseline partition")
        normalizer = fit_normalizer(raw_cache, used_training)
        cache = normalized_cache(raw_cache, normalizer)
        train_x = np.vstack([_pair_features(pair, cache) for pair in training])
        train_y = np.asarray([pair.similarity for pair in training], dtype=float)
        validation_x = np.vstack([_pair_features(pair, cache) for pair in validation])
        targets = [float(pair.similarity) for pair in validation]
        row = {
            "repeat": repeat + 1,
            "split_seed": seed,
            "n_training_pairs": len(training),
            "n_validation_pairs": len(validation),
            "molecule_leakage_count": 0,
            "held_out_sha256": hashlib.sha256(
                json.dumps(sorted(held_out), separators=(",", ":")).encode()
            ).hexdigest(),
            "algorithms": {},
        }
        for name in ALGORITHMS:
            if name == "morgan_tanimoto":
                predictions = _morgan_predictions(validation, morgan_cache)
            else:
                estimator = _model(name, seed)
                estimator.fit(train_x, train_y)
                predictions = np.clip(estimator.predict(validation_x), 0.0, 1.0).tolist()
            metrics = metric_summary(predictions, targets)
            row["algorithms"][name] = metrics
            pooled[name]["predictions"].extend(float(value) for value in predictions)
            pooled[name]["targets"].extend(targets)
            pooled[name]["fold_rho"].append(float(metrics["spearman"]))
        repeat_rows.append(row)
    summary = {}
    for name, values in pooled.items():
        summary[name] = {
            "pooled": metric_summary(values["predictions"], values["targets"]),
            "fold_mean_spearman": float(np.mean(values["fold_rho"])),
            "fold_std_spearman": float(np.std(values["fold_rho"], ddof=1)) if repeats > 1 else 0.0,
        }
    return {"repeats": repeat_rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-data-root", type=Path, required=True)
    parser.add_argument(
        "--descriptor-cache",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_release_descriptor_cache.npz",
    )
    parser.add_argument(
        "--r2-final-report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_final_strict.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_strong_baselines.json",
    )
    args = parser.parse_args()

    pairs = load_snitz_pairs(args.mixture_data_root)
    molecules = sorted({value for pair in pairs for value in pair.molecules})
    raw_cache = _load_raw_cache(args.descriptor_cache)
    missing = set(molecules) - raw_cache.keys()
    if missing:
        raise RuntimeError(f"descriptor cache missing {len(missing)} Snitz molecules")
    morgan_cache = _morgan_cache(set(molecules))

    phases = {}
    for phase, split_seed, repeats in (("development", 142, 3), ("final", 52908, 5)):
        phases[phase] = {
            "molecule_disjoint": _protocol(
                pairs=pairs,
                molecules=molecules,
                raw_cache=raw_cache,
                morgan_cache=morgan_cache,
                split_seed=split_seed,
                repeats=repeats,
                scaffold=False,
            ),
            "scaffold_disjoint": _protocol(
                pairs=pairs,
                molecules=molecules,
                raw_cache=raw_cache,
                morgan_cache=morgan_cache,
                split_seed=split_seed,
                repeats=repeats,
                scaffold=True,
            ),
        }

    selection_scores = {}
    for name in ALGORITHMS:
        scores = []
        for protocol in ("molecule_disjoint", "scaffold_disjoint"):
            item = phases["development"][protocol]["summary"][name]
            scores.extend((item["pooled"]["spearman"], item["fold_mean_spearman"]))
        selection_scores[name] = min(scores)
    selected = max(selection_scores, key=lambda name: (selection_scores[name], name))

    r2_final = json.loads(args.r2_final_report.read_text(encoding="utf-8"))
    comparisons = {}
    all_improve = True
    for protocol in ("molecule_disjoint", "scaffold_disjoint"):
        baseline = phases["final"][protocol]["summary"][selected]
        r2_config = r2_final[protocol]["configurations"][r2_final["selected_configuration"]]
        comparison = {
            "r2_pooled_spearman": float(r2_config["pooled_model"]["spearman"]),
            "selected_strong_baseline_pooled_spearman": float(baseline["pooled"]["spearman"]),
            "pooled_delta": float(
                r2_config["pooled_model"]["spearman"] - baseline["pooled"]["spearman"]
            ),
            "r2_fold_mean_spearman": float(np.mean([
                row["model"]["spearman"] for row in r2_config["repeats"]
            ])),
            "selected_strong_baseline_fold_mean_spearman": float(baseline["fold_mean_spearman"]),
        }
        comparison["fold_mean_delta"] = (
            comparison["r2_fold_mean_spearman"]
            - comparison["selected_strong_baseline_fold_mean_spearman"]
        )
        comparison["r2_improves_both"] = (
            comparison["pooled_delta"] >= 0.01 and comparison["fold_mean_delta"] >= 0.01
        )
        all_improve = all_improve and comparison["r2_improves_both"]
        comparisons[protocol] = comparison

    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "historical mixture-pair similarity; not text-to-odor or human formula similarity",
        "data": {
            "pairs": len(pairs),
            "molecules": len(molecules),
            "mixture_data_root": str(args.mixture_data_root),
            "descriptor_cache_sha256": sha256_file(args.descriptor_cache),
            "r2_final_report_sha256": sha256_file(args.r2_final_report),
        },
        "algorithms": list(ALGORITHMS),
        "selection": {
            "rule": "maximize worst development-only pooled/fold-mean Spearman across molecule/scaffold protocols",
            "scores": selection_scores,
            "selected": selected,
            "final_labels_used_for_selection": False,
        },
        "phases": phases,
        "final_comparison": comparisons,
        "strong_baseline_release_gate_passed": bool(all_improve),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "selection_scores": selection_scores,
        "final_comparison": comparisons,
        "gate": all_improve,
        "output": str(args.output),
    }, indent=2))
    return 0 if all_improve else 2


if __name__ == "__main__":
    raise SystemExit(main())
