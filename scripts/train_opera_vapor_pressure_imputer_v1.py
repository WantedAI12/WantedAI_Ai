#!/usr/bin/env python
"""Train a portable OPERA vapor-pressure imputer with scaffold validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragrance_ai.research.headspace import (  # noqa: E402
    predict_portable_log_vapor_pressure,
)


SCHEMA = "opera-vp-imputer-retrospective/v1"
RUNTIME_SCHEMA = "opera-vp-portable-ridge/v1"
ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
BOOTSTRAP_SEED = 20_260_828
BOOTSTRAP_DRAWS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def scaffold(molecule: Chem.Mol) -> str:
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    # Treating every acyclic molecule as its own scaffold makes a nominal
    # scaffold split equivalent to a much weaker molecule split. Group all
    # acyclic structures together for the conservative primary audit.
    return value or "acyclic:all"


def load_endpoint(
    connection: sqlite3.Connection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    raw = connection.execute(
        "SELECT inchi_key_skeleton, canonical_smiles, value, split "
        "FROM physchem_observations "
        "WHERE endpoint='log10_vapor_pressure_mmhg' "
        "ORDER BY observation_id"
    ).fetchall()
    grouped: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for skeleton, smiles, value, split in raw:
        grouped[str(skeleton)].append((str(smiles), float(value), str(split)))
    descriptor_names = [str(name) for name, _function in Descriptors.descList]
    descriptor_functions = [function for _name, function in Descriptors.descList]
    features = []
    targets = []
    splits = []
    scaffolds = []
    skeletons = []
    smiles_rows = []
    for skeleton_key, rows in sorted(grouped.items()):
        smiles = next((row[0] for row in rows if row[0]), "")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid OPERA canonical SMILES: {skeleton_key}")
        values = np.asarray(
            [function(molecule) for function in descriptor_functions], dtype=float
        )
        values[~np.isfinite(values)] = np.nan
        source_splits = {row[2] for row in rows}
        if len(source_splits) != 1:
            raise RuntimeError(f"OPERA molecule crosses train/test: {skeleton_key}")
        features.append(values)
        targets.append(float(np.median([row[1] for row in rows])))
        splits.append(next(iter(source_splits)))
        scaffolds.append(scaffold(molecule))
        skeletons.append(skeleton_key)
        smiles_rows.append(Chem.MolToSmiles(molecule, canonical=True))
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    split_values = np.asarray(splits, dtype=object)
    scaffold_values = np.asarray(scaffolds, dtype=object)
    if matrix.shape != (2711, 217) or len(raw) != 2713:
        raise RuntimeError(
            f"unexpected OPERA VP shape: observations={len(raw)}, matrix={matrix.shape}"
        )
    return (
        matrix,
        target,
        split_values,
        scaffold_values,
        descriptor_names,
        smiles_rows,
    )


def prepare(
    training: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(training, axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("OPERA descriptor has no finite training value")
    imputed = np.where(np.isnan(training), median, training)
    mean = np.mean(imputed, axis=0)
    scale = np.std(imputed, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (imputed - mean) / scale
    if not np.isfinite(standardized).all() or not np.isfinite(target).all():
        raise RuntimeError("OPERA standardized training data is non-finite")
    return standardized, median, mean, scale


def apply_preparation(
    features: np.ndarray, median: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    imputed = np.where(np.isnan(features), median, features)
    return (imputed - mean) / scale


def correlation(function, prediction: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 3 or np.std(prediction) < 1e-12 or np.std(target) < 1e-12:
        return 0.0
    result = float(function(prediction, target).statistic)
    return result if math.isfinite(result) else 0.0


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "n": int(len(target)),
        "mae_log10_mmhg": float(np.mean(np.abs(error))),
        "rmse_log10_mmhg": float(np.sqrt(np.mean(error * error))),
        "r2": float(r2_score(target, prediction)),
        "pearson": correlation(pearsonr, prediction, target),
        "spearman": correlation(spearmanr, prediction, target),
        "bias_log10_mmhg": float(np.mean(error)),
        "absolute_error_q95_log10_mmhg": float(np.percentile(np.abs(error), 95)),
    }


def cross_validate(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, Any], float]:
    splitter = GroupKFold(n_splits=5)
    candidates = []
    for alpha in ALPHAS:
        prediction = np.zeros(len(target), dtype=float)
        for training, validation in splitter.split(features, target, groups):
            standardized, median, mean, scale = prepare(
                features[training], target[training]
            )
            model = Ridge(alpha=alpha)
            model.fit(standardized, target[training])
            prediction[validation] = model.predict(
                apply_preparation(features[validation], median, mean, scale)
            )
        candidates.append({"alpha": alpha, "scaffold_oof": metrics(prediction, target)})
    selected = min(
        candidates,
        key=lambda row: (float(row["scaffold_oof"]["rmse_log10_mmhg"]), row["alpha"]),
    )
    return {"candidates": candidates, "selected": selected}, float(selected["alpha"])


def paired_bootstrap(
    target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(target)
    mae_gain = np.empty(BOOTSTRAP_DRAWS)
    rmse_gain = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, count, count)
        candidate_error = candidate[index] - target[index]
        baseline_error = baseline[index] - target[index]
        mae_gain[draw] = np.mean(np.abs(baseline_error)) - np.mean(
            np.abs(candidate_error)
        )
        rmse_gain[draw] = np.sqrt(np.mean(baseline_error**2)) - np.sqrt(
            np.mean(candidate_error**2)
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "baseline_minus_candidate_mae_95_interval": [
            float(np.percentile(mae_gain, 2.5)),
            float(np.percentile(mae_gain, 97.5)),
        ],
        "baseline_minus_candidate_rmse_95_interval": [
            float(np.percentile(rmse_gain, 2.5)),
            float(np.percentile(rmse_gain, 97.5)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.db",
    )
    parser.add_argument(
        "--hub-report",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.json",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "benchmarks" / "opera_vapor_pressure_runtime_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "benchmarks" / "opera_vapor_pressure_imputer_v1.json",
    )
    args = parser.parse_args()
    hub_path = args.hub.expanduser().resolve(strict=True)
    hub_report_path = args.hub_report.expanduser().resolve(strict=True)
    hub_report = json.loads(hub_report_path.read_text(encoding="utf-8"))
    if hub_report.get("database", {}).get("sha256") != sha256(hub_path):
        raise RuntimeError("headspace hub hash does not match report")
    connection = sqlite3.connect(hub_path.as_uri() + "?mode=ro", uri=True)
    try:
        features, target, splits, groups, names, _smiles = load_endpoint(connection)
    finally:
        connection.close()
    training_index = np.flatnonzero(splits == "train")
    test_index = np.flatnonzero(splits == "test")
    if len(training_index) != 2032 or len(test_index) != 679:
        raise RuntimeError("OPERA VP publisher split sizes changed")
    training_scaffolds = set(groups[training_index].tolist())
    strict_test_index = np.asarray(
        [index for index in test_index if groups[index] not in training_scaffolds],
        dtype=int,
    )
    if len(strict_test_index) != 93:
        raise RuntimeError("OPERA VP strict scaffold test size changed")
    selection, alpha = cross_validate(
        features[training_index], target[training_index], groups[training_index]
    )
    standardized, median, mean, scale = prepare(
        features[training_index], target[training_index]
    )
    model = Ridge(alpha=alpha)
    model.fit(standardized, target[training_index])
    prediction_clip = [
        float(np.min(target[training_index])),
        float(np.max(target[training_index])),
    ]
    runtime = {
        "schema": RUNTIME_SCHEMA,
        "status": "research_only_opera_train_fit_not_production_authorized",
        "endpoint": "log10_vapor_pressure_mmhg",
        "descriptor_names": names,
        "descriptor_contract_sha256": canonical_json_sha256(names),
        "feature_median": median.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": float(model.intercept_),
        "alpha": alpha,
        "prediction_clip": prediction_clip,
        "rdkit_version": rdBase.rdkitVersion,
        "allow_pickle": False,
        "runtime_primary_score_weight": 0.0,
        "human_olfactory_90_percent_certified": False,
    }
    runtime_path = args.runtime.expanduser().resolve()
    atomic_json(runtime_path, runtime)
    test_prediction = predict_portable_log_vapor_pressure(
        runtime, features[test_index], names
    )
    strict_prediction = predict_portable_log_vapor_pressure(
        runtime, features[strict_test_index], names
    )
    direct_test_prediction = np.clip(
        model.predict(apply_preparation(features[test_index], median, mean, scale)),
        *prediction_clip,
    )
    portable_error = float(np.max(np.abs(test_prediction - direct_test_prediction)))
    if portable_error > 1e-12:
        raise RuntimeError("portable OPERA VP runtime differs from sklearn")
    baseline_value = float(np.median(target[training_index]))
    strict_baseline = np.full(len(strict_test_index), baseline_value)
    strict_metrics = metrics(strict_prediction, target[strict_test_index])
    baseline_metrics = metrics(strict_baseline, target[strict_test_index])
    bootstrap = paired_bootstrap(
        target[strict_test_index], strict_prediction, strict_baseline
    )
    gate_checks = {
        "strict_scaffold_test_at_least_300": len(strict_test_index) >= 300,
        "strict_scaffold_overlap_zero": not (
            set(groups[strict_test_index].tolist()) & training_scaffolds
        ),
        "strict_r2_at_least_0_75": strict_metrics["r2"] >= 0.75,
        "strict_rmse_below_1_5_log10_mmhg": strict_metrics["rmse_log10_mmhg"] < 1.5,
        "strict_q95_below_2_5_log10_mmhg": strict_metrics[
            "absolute_error_q95_log10_mmhg"
        ]
        < 2.5,
        "bootstrap_mae_gain_lower_above_zero": bootstrap[
            "baseline_minus_candidate_mae_95_interval"
        ][0]
        > 0.0,
        "bootstrap_rmse_gain_lower_above_zero": bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "portable_parity_at_most_1e_12": portable_error <= 1e-12,
    }
    gate = all(gate_checks.values())
    report = {
        "schema": SCHEMA,
        "status": (
            "scaffold_validated_research_imputer_passed"
            if gate
            else "scaffold_validated_research_imputer_failed"
        ),
        "source": {
            "hub_sha256": sha256(hub_path),
            "hub_report_sha256": sha256(hub_report_path),
            "epa_opera_sha256": hub_report["source"]["opera_sha256"],
            "epa_opera_license": hub_report["source"]["opera_license"],
        },
        "timing": {
            "retrospective_public_dataset": True,
            "gate_thresholds_preregistered_before_data_access": False,
            "post_selection_intervals_descriptive_only": True,
            "inferentially_valid_for_production_promotion": False,
        },
        "dataset": {
            "raw_observations": 2713,
            "unique_molecules": 2711,
            "publisher_training_molecules": len(training_index),
            "publisher_test_molecules": len(test_index),
            "strict_scaffold_test_molecules": len(strict_test_index),
            "strict_scaffold_overlap": 0,
            "scaffold_definition": (
                "bemis_murcko_ring_framework_with_all_acyclic_molecules_in_one_group"
            ),
            "descriptor_dimensions": len(names),
        },
        "selection": selection,
        "publisher_test": metrics(test_prediction, target[test_index]),
        "strict_scaffold_test": {
            "candidate": strict_metrics,
            "training_median_baseline": baseline_metrics,
            "paired_bootstrap": bootstrap,
        },
        "gates": {
            "research_imputer": {"checks": gate_checks, "passed": gate},
            "production": {
                "passed": False,
                "runtime_primary_score_weight": 0.0,
            },
        },
        "runtime": {
            "path": str(runtime_path),
            "sha256": sha256(runtime_path),
            "portable_parity_max_abs_error": portable_error,
            "allow_pickle": False,
            "primary_score_weight": 0.0,
        },
        "software": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "claim_boundary": {
            "measured_values_remain_distinct_from_predictions": True,
            "prediction_error_is_log10_mmhg_not_olfactory_error": True,
            "mixture_headspace_validated": False,
            "human_olfactory_90_percent_certified": False,
            "commercial_runtime_weight": 0.0,
        },
    }
    atomic_json(args.report.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
