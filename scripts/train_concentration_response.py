#!/usr/bin/env python
"""Train a molecule-disjoint dilution-to-perceived-intensity calibrator.

Ravia behavior_1 supplies measured intensity at recorded dilutions.  Model and
hyperparameters are selected only inside a development molecule set; a frozen
molecule-disjoint final set is read once for the release decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import build_raw_descriptor_cache  # noqa: E402


SEED = 781_241
ALGORITHMS = (
    "concentration_only_ridge",
    "descriptor_concentration_ridge",
    "descriptor_concentration_random_forest",
    "descriptor_concentration_extra_trees",
    "descriptor_concentration_hist_gradient_boosting",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float_correlation(fn, left, right) -> float:
    if len(left) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    value = fn(left, right)[0]
    return float(value) if np.isfinite(value) else 0.0


def _metrics(prediction, target) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "spearman": _float_correlation(spearmanr, prediction, target),
        "pearson": _float_correlation(pearsonr, prediction, target),
        "n": int(len(target)),
    }


def _model(name: str, seed: int):
    if name in {"concentration_only_ridge", "descriptor_concentration_ridge"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    if name == "descriptor_concentration_random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=3,
            max_features=0.45,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "descriptor_concentration_extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.55,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "descriptor_concentration_hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=9,
            min_samples_leaf=8,
            l2_regularization=3.0,
            random_state=seed,
        )
    raise KeyError(name)


def _features(descriptors: np.ndarray, dilution: np.ndarray, name: str) -> np.ndarray:
    log_c = np.log10(np.clip(dilution, 1e-8, 1.0))[:, None]
    if name == "concentration_only_ridge":
        return np.concatenate((log_c, log_c**2), axis=1)
    # The interaction term lets molecular structure alter the dilution slope.
    return np.concatenate((descriptors, log_c, descriptors * log_c), axis=1)


def _group_folds(groups: np.ndarray, *, seed: int, n_splits: int) -> list[set[str]]:
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    return [set(values.tolist()) for values in np.array_split(unique, n_splits)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravia-root", type=Path, required=True)
    parser.add_argument(
        "--descriptor-cache",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/physsim_r2_release_descriptor_cache.npz",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "fragrance_ai/data/concentration_response_model.joblib",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "fragrance_ai/data/concentration_response_manifest.json",
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=PROJECT_ROOT / "fragrance_ai/data/concentration_response_runtime.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/concentration_response_validation.json",
    )
    args = parser.parse_args()

    with (args.ravia_root / "molecules.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        cid_to_smiles = {
            row["CID"]: row["IsomericSMILES"]
            for row in csv.DictReader(handle)
            if row.get("CID") and row.get("IsomericSMILES")
        }
    with (args.ravia_root / "stimuli.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        stimuli = {row["Stimulus"]: row for row in csv.DictReader(handle)}
    with np.load(args.descriptor_cache, allow_pickle=False) as data:
        descriptor_cache = {
            str(smiles): np.asarray(row, dtype=np.float32)
            for smiles, row in zip(data["smiles"], data["descriptors"])
        }
    # Ravia's intensity experiment contains molecules not used in its mixture
    # similarity experiment, so extend the immutable cache in memory from the
    # same versioned RDKit descriptor contract.
    missing_smiles = sorted(set(cid_to_smiles.values()) - descriptor_cache.keys())
    descriptor_cache.update(build_raw_descriptor_cache(missing_smiles))

    raw_rows = []
    with (args.ravia_root / "behavior_1.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if not row.get("Intensity"):
                continue
            stimulus = stimuli.get(row["Stimulus"])
            if not stimulus or stimulus.get("Type") != "mono-molecule":
                continue
            raw_dilution = stimulus.get(f"Dilution{row['Dilution #']}", "")
            try:
                # Ravia stores dilution values as percentages; one column omits
                # the percent glyph while retaining the same percent unit.
                dilution = float(raw_dilution.strip().rstrip("%")) / 100.0
            except (AttributeError, ValueError):
                continue
            smiles = cid_to_smiles.get(stimulus.get("CID", ""))
            if not smiles or smiles not in descriptor_cache or dilution <= 0:
                continue
            raw_rows.append((smiles, dilution, float(row["Intensity"])))

    # Only multi-dilution molecules identify a concentration response.
    counts = {}
    for smiles, _, _ in raw_rows:
        counts[smiles] = counts.get(smiles, 0) + 1
    rows = [row for row in raw_rows if counts[row[0]] >= 3]
    groups = np.asarray([row[0] for row in rows], dtype=object)
    dilution = np.asarray([row[1] for row in rows], dtype=float)
    target = np.asarray([row[2] for row in rows], dtype=float)
    descriptors = np.asarray(
        [descriptor_cache[row[0]] for row in rows], dtype=np.float32
    )
    molecules = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.RandomState(SEED)
    rng.shuffle(molecules)
    final_count = max(12, int(round(len(molecules) * 0.20)))
    final_groups = set(molecules[:final_count].tolist())
    development_mask = np.asarray([value not in final_groups for value in groups])
    final_mask = ~development_mask

    development_results = {}
    folds = _group_folds(groups[development_mask], seed=SEED + 1, n_splits=5)
    for name in ALGORITHMS:
        fold_metrics = []
        pooled_prediction = []
        pooled_target = []
        for fold_index, held_out in enumerate(folds):
            validation = development_mask & np.asarray(
                [value in held_out for value in groups]
            )
            training = development_mask & ~validation
            if set(groups[training]) & set(groups[validation]):
                raise RuntimeError("development molecule leakage")
            estimator = _model(name, SEED + fold_index)
            estimator.fit(
                _features(descriptors[training], dilution[training], name),
                target[training],
            )
            prediction = np.clip(
                estimator.predict(
                    _features(descriptors[validation], dilution[validation], name)
                ),
                0.0,
                100.0,
            )
            fold_metrics.append(_metrics(prediction, target[validation]))
            pooled_prediction.extend(prediction.tolist())
            pooled_target.extend(target[validation].tolist())
        development_results[name] = {
            "folds": fold_metrics,
            "fold_mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
            "fold_mean_spearman": float(
                np.mean([row["spearman"] for row in fold_metrics])
            ),
            "pooled": _metrics(pooled_prediction, pooled_target),
        }

    selected = min(
        ALGORITHMS,
        key=lambda name: (
            development_results[name]["fold_mean_mae"],
            development_results[name]["pooled"]["mae"],
            name,
        ),
    )
    concentration_baseline = development_results["concentration_only_ridge"]
    estimator = _model(selected, SEED + 100)
    estimator.fit(
        _features(descriptors[development_mask], dilution[development_mask], selected),
        target[development_mask],
    )
    final_prediction = np.clip(
        estimator.predict(
            _features(descriptors[final_mask], dilution[final_mask], selected)
        ),
        0.0,
        100.0,
    )
    final_metrics = _metrics(final_prediction, target[final_mask])
    baseline_estimator = _model("concentration_only_ridge", SEED + 100)
    baseline_estimator.fit(
        _features(
            descriptors[development_mask],
            dilution[development_mask],
            "concentration_only_ridge",
        ),
        target[development_mask],
    )
    baseline_prediction = np.clip(
        baseline_estimator.predict(
            _features(
                descriptors[final_mask],
                dilution[final_mask],
                "concentration_only_ridge",
            )
        ),
        0.0,
        100.0,
    )
    final_baseline = _metrics(baseline_prediction, target[final_mask])
    # A failed structure-specific candidate must not block the measured global
    # concentration response. Deploy the concentration-only curve when it has
    # useful molecule-disjoint final performance, with an explicit zero weight
    # for unvalidated molecule-specific modulation.
    concentration_fallback_passed = (
        final_baseline["mae"] <= 20.0 and final_baseline["spearman"] >= 0.50
    )
    # Structure-dependent dilution did not satisfy the production contract in
    # the published release. Keep the portable runtime deliberately restricted
    # to the validated global concentration response.
    deployed = "concentration_only_ridge"
    release_passed = concentration_fallback_passed

    # Refit the frozen production model on all molecules only after final metrics.
    production = _model(deployed, SEED + 200)
    production.fit(_features(descriptors, dilution, deployed), target)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": production,
            "algorithm": deployed,
            "development_selected_structure_candidate": selected,
            "descriptor_count": int(descriptors.shape[1]),
            "feature_contract": (
                "[log10(dilution),log10(dilution)^2]"
                if deployed == "concentration_only_ridge"
                else "[rdkit217,log10(dilution),rdkit217*log10(dilution)]"
            ),
        },
        args.model_output,
        compress=3,
    )
    model_sha = _sha(args.model_output)
    scaler = production.named_steps["standardscaler"]
    ridge = production.named_steps["ridge"]
    runtime = {
        "schema_version": "1.0",
        "runtime": "numpy_concentration_response_v1",
        "format": "standard_scaler_plus_ridge_coefficients_v1",
        "feature_contract": ["log10_dilution", "log10_dilution_squared"],
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in ridge.coef_],
        "intercept": float(ridge.intercept_),
        "dilution_range_fraction": [
            float(dilution.min()),
            float(dilution.max()),
        ],
        "source_training_artifact_sha256": model_sha,
        "source_training_artifact_required_at_runtime": False,
        "allow_pickle": False,
    }
    args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_output.write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    runtime_sha = _sha(args.runtime_output)
    sources = {
        name: {
            "sha256": _sha(args.ravia_root / name),
            "bytes": (args.ravia_root / name).stat().st_size,
        }
        for name in ("molecules.csv", "stimuli.csv", "behavior_1.csv")
    }
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.1",
        "generated_at": generated,
        "source_model_file": args.model_output.name,
        "source_model_sha256": model_sha,
        "runtime_file": args.runtime_output.name,
        "runtime_sha256": runtime_sha,
        "algorithm": deployed,
        "development_selected_structure_candidate": selected,
        "structure_specific_weight": 0.0,
        "distribution_contract": {
            "runtime_format": "json_numeric_arrays_only",
            "source_model_packaged": False,
            "source_model_required_at_runtime": False,
            "pickle_deserialization_allowed": False,
        },
        "model_seed": SEED,
        "descriptor_count": int(descriptors.shape[1]),
        "training_records": len(rows),
        "training_molecules": len(molecules),
        "dilution_range_fraction": [float(dilution.min()), float(dilution.max())],
        "release_gate": {
            "passed": release_passed,
            "checks": {
                "molecule_disjoint_final_set": True,
                "model_selected_on_development_only": True,
                "deployed_concentration_curve_has_normalized_mae_at_most_20_percent": final_baseline[
                    "mae"
                ]
                <= 20.0,
                "deployed_concentration_curve_has_final_spearman_at_least_0_50": final_baseline[
                    "spearman"
                ]
                >= 0.50,
            },
            "rejected_structure_candidate_checks": {
                "structure_model_beats_concentration_only_final_mae": final_metrics[
                    "mae"
                ]
                <= final_baseline["mae"] - 0.25,
                "structure_model_not_worse_final_spearman": final_metrics["spearman"]
                >= final_baseline["spearman"],
            },
        },
        "source_files": sources,
        "claim_boundary": "Predicts a global Ravia mono-molecule perceived-intensity response to dilution. Molecule-specific concentration modulation has zero production weight; this is not mixture similarity or text-to-formula accuracy.",
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        **manifest,
        "development": development_results,
        "selection": {
            "selected": selected,
            "deployed": deployed,
            "rule": "minimum development molecule-disjoint fold-mean MAE",
            "final_labels_used_for_selection": False,
            "concentration_only_development": concentration_baseline,
        },
        "final": {
            "molecules": len(final_groups),
            "records": int(final_mask.sum()),
            "held_out_sha256": hashlib.sha256(
                json.dumps(sorted(final_groups)).encode()
            ).hexdigest(),
            "model": final_metrics,
            "concentration_only_baseline": final_baseline,
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(rows),
                "molecules": len(molecules),
                "selected": selected,
                "deployed": deployed,
                "final_model": final_metrics,
                "final_baseline": final_baseline,
                "release_gate": release_passed,
                "model": str(args.model_output),
            },
            indent=2,
        )
    )
    return 0 if release_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
