#!/usr/bin/env python
"""Post-outcome residual calibration for the frozen Ma 2021 blind benchmark.

The frozen blind result showed that the strongest measured component is a very
strong binary-mixture intensity baseline.  This builder therefore predicts only
the residual ``IAB - mean(max(IA, IB))``.  Exact pairs are held out in repeated
nested cross-validation; a second all-components-cold protocol evaluates pairs
whose two molecules are absent from training.  The artifact is diagnostic and
keeps runtime weight at zero pending another external reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_ma_2021_binary_mixture_benchmark as parent  # noqa: E402


SCHEMA_VERSION = "2.0"
OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
COMPONENT_COLD_REPEATS = 10
COMPONENT_COLD_FOLDS = 5
BOOTSTRAP_SEED = 20_260_831
BOOTSTRAP_DRAWS = 10_000
FOLD_SALT = "ma-2021-binary-mixture-residual-calibration-v2"
TARGET_MAE = 0.245

CONTINUOUS_FEATURES = (
    "baseline_strongest_component",
    "mean_component_intensity",
    "minimum_component_intensity",
    "component_intensity_gap",
    "component_intensity_balance",
    "component_intensity_sum",
    "component_pleasantness_mean",
    "component_pleasantness_gap",
    "r2_similarity",
    "r2_disagreement",
    "morgan_similarity",
    "log10_concentration_mean",
    "log10_concentration_gap",
    "molecular_weight_mean",
    "molecular_weight_gap",
    "same_solvent",
    "humanpom_intensity_max",
    "humanpom_intensity_gap",
    "humanpom_pleasantness_mean",
    "fechner_minus_max",
)
HINGE_FEATURES = (
    "max_intensity_above_4",
    "max_intensity_above_6",
    "max_intensity_above_8",
    "gap_above_0_5",
    "gap_above_1",
    "gap_above_2",
    "balanced_components",
    "r2_independent_channels",
)


def _candidates() -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = [
        {"name": "training_median_residual", "algorithm": "median", "feature_set": "none"}
    ]
    for feature_set in ("continuous", "hinge", "full"):
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            values.append(
                {
                    "name": f"ridge_{feature_set}_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": feature_set,
                    "alpha": alpha,
                }
            )
    for feature_set in ("continuous", "hinge"):
        for alpha, epsilon in (
            (0.0001, 1.2),
            (0.01, 1.2),
            (0.1, 1.2),
            (0.01, 1.35),
            (0.1, 1.35),
        ):
            values.append(
                {
                    "name": f"huber_{feature_set}_{alpha:g}_{epsilon:g}",
                    "algorithm": "huber",
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "epsilon": epsilon,
                }
            )
    for feature_set in ("continuous", "hinge"):
        for alpha in (0.001, 0.01, 0.1):
            values.append(
                {
                    "name": f"quantile_{feature_set}_{alpha:g}",
                    "algorithm": "quantile",
                    "feature_set": feature_set,
                    "alpha": alpha,
                }
            )
    names = [row["name"] for row in values]
    if len(names) != len(set(names)):
        raise RuntimeError("Ma v2 candidate names are not unique")
    return tuple(values)


CANDIDATES = _candidates()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _hash_fold(value: str, *, folds: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _balanced_folds(values: Sequence[str], *, folds: int, salt: str) -> np.ndarray:
    ordered = sorted(
        range(len(values)),
        key=lambda index: hashlib.sha256(f"{salt}|{values[index]}".encode()).digest(),
    )
    result = np.empty(len(values), dtype=int)
    for position, index in enumerate(ordered):
        result[index] = position % folds
    return result


def _load_rows(
    predictions_path: Path,
    outcome_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    _, individual, parser_audit = parent._load_outcome(outcome_path)
    target_by_cas = {
        str(row["cas"]): row for row in predictions["target_odorants"]
    }
    cas_by_name = {
        str(row["normalized_odorant"]): str(row["cas"])
        for row in predictions["target_odorants"]
    }
    pair_lookup = {row["pair_id"]: row for row in predictions["predictions"]}
    groups: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    participant_rows: list[dict[str, Any]] = []
    for _, source in individual.iterrows():
        name_a = parent.normalize_name(source["odor A"])
        name_b = parent.normalize_name(source["odor B"])
        cas_a = cas_by_name.get(name_a)
        cas_b = cas_by_name.get(name_b)
        if cas_a is None or cas_b is None:
            raise RuntimeError("Ma v2 outcome name did not map to frozen metadata")
        pair_id = parent._pair_id(cas_a, cas_b)
        pair = pair_lookup[pair_id]
        component_a_name = pair["component_a"]["normalized_odorant"]
        if name_a == component_a_name:
            ia, ib = float(source["IA"]), float(source["IB"])
            pa, pb = float(source["PA"]), float(source["PB"])
        elif name_b == component_a_name:
            ia, ib = float(source["IB"]), float(source["IA"])
            pa, pb = float(source["PB"]), float(source["PA"])
        else:
            raise RuntimeError("Ma v2 pair orientation could not be determined")
        baseline = max(ia, ib)
        row = {
            "subject_id": str(source["subject_id"]),
            "pair_id": pair_id,
            "ia": ia,
            "ib": ib,
            "pa": pa,
            "pb": pb,
            "baseline": baseline,
            "target": float(source["IAB"]),
        }
        groups[pair_id].append(row)
        participant_rows.append(row)

    rows = []
    for pair_id in sorted(groups):
        values = groups[pair_id]
        pair = pair_lookup[pair_id]
        first_cas = str(pair["component_a"]["cas"])
        second_cas = str(pair["component_b"]["cas"])
        first_metadata = target_by_cas[first_cas]
        second_metadata = target_by_cas[second_cas]
        ia = float(np.mean([float(row["ia"]) for row in values]))
        ib = float(np.mean([float(row["ib"]) for row in values]))
        pa = float(np.mean([float(row["pa"]) for row in values]))
        pb = float(np.mean([float(row["pb"]) for row in values]))
        baseline = float(np.mean([float(row["baseline"]) for row in values]))
        target = float(np.mean([float(row["target"]) for row in values]))
        maximum = max(ia, ib)
        minimum = min(ia, ib)
        gap = abs(ia - ib)
        structure = pair["structure_similarity"]
        end_to_end = pair["end_to_end"]["primary"]
        concentration_a = float(first_metadata["concentration_mg_ml"])
        concentration_b = float(second_metadata["concentration_mg_ml"])
        weight_a = float(first_metadata["molecular_weight"])
        weight_b = float(second_metadata["molecular_weight"])
        continuous = {
            "baseline_strongest_component": baseline,
            "mean_component_intensity": (ia + ib) / 2.0,
            "minimum_component_intensity": minimum,
            "component_intensity_gap": gap,
            "component_intensity_balance": minimum / max(maximum, 1e-6),
            "component_intensity_sum": ia + ib,
            "component_pleasantness_mean": (pa + pb) / 2.0,
            "component_pleasantness_gap": abs(pa - pb),
            "r2_similarity": float(structure["r2_ensemble"]),
            "r2_disagreement": float(structure["r2_member_disagreement"]),
            "morgan_similarity": float(structure["morgan_tanimoto"]),
            "log10_concentration_mean": (
                math.log10(concentration_a) + math.log10(concentration_b)
            )
            / 2.0,
            "log10_concentration_gap": abs(
                math.log10(concentration_a) - math.log10(concentration_b)
            ),
            "molecular_weight_mean": (weight_a + weight_b) / 2.0,
            "molecular_weight_gap": abs(weight_a - weight_b),
            "same_solvent": float(first_metadata["solvent"] == second_metadata["solvent"]),
            "humanpom_intensity_max": max(
                float(end_to_end["component_a_intensity"]),
                float(end_to_end["component_b_intensity"]),
            ),
            "humanpom_intensity_gap": abs(
                float(end_to_end["component_a_intensity"])
                - float(end_to_end["component_b_intensity"])
            ),
            "humanpom_pleasantness_mean": (
                float(end_to_end["component_a_pleasantness"])
                + float(end_to_end["component_b_pleasantness"])
            )
            / 2.0,
            "fechner_minus_max": float(end_to_end["ravia_weber_fechner_pool"])
            - float(end_to_end["strongest_component"]),
        }
        hinges = {
            "max_intensity_above_4": max(maximum - 4.0, 0.0),
            "max_intensity_above_6": max(maximum - 6.0, 0.0),
            "max_intensity_above_8": max(maximum - 8.0, 0.0),
            "gap_above_0_5": max(gap - 0.5, 0.0),
            "gap_above_1": max(gap - 1.0, 0.0),
            "gap_above_2": max(gap - 2.0, 0.0),
            "balanced_components": math.exp(-gap),
            "r2_independent_channels": 1.0 - float(structure["r2_ensemble"]),
        }
        if not np.all(np.isfinite(list(continuous.values()) + list(hinges.values()))):
            raise RuntimeError("Ma v2 features contain non-finite values")
        rows.append(
            {
                "pair_id": pair_id,
                "component_a_cas": first_cas,
                "component_b_cas": second_cas,
                "baseline": baseline,
                "target": target,
                "residual": target - baseline,
                "continuous": continuous,
                "hinges": hinges,
            }
        )
    if len(rows) != parent.EXPECTED_DISTINCT_MIXTURES:
        raise RuntimeError(f"expected 198 Ma v2 pair rows, found {len(rows)}")
    audit = {
        **parser_audit,
        "distinct_pairs": len(rows),
        "participant_rows": len(participant_rows),
        "continuous_features": list(CONTINUOUS_FEATURES),
        "hinge_features": list(HINGE_FEATURES),
        "forbidden_outcome_features": ["IAmix", "IBmix", "PAB", "Group", "Repeat"],
    }
    return rows, participant_rows, audit


def _component_vocabulary(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row[key])
                for row in rows
                for key in ("component_a_cas", "component_b_cas")
            }
        )
    )


def _design(
    rows: Sequence[Mapping[str, Any]],
    feature_set: str,
    vocabulary: Sequence[str],
) -> np.ndarray:
    if feature_set == "none":
        return np.zeros((len(rows), 0), dtype=float)
    continuous = np.asarray(
        [[float(row["continuous"][name]) for name in CONTINUOUS_FEATURES] for row in rows],
        dtype=float,
    )
    if feature_set == "continuous":
        return continuous
    hinges = np.asarray(
        [[float(row["hinges"][name]) for name in HINGE_FEATURES] for row in rows],
        dtype=float,
    )
    transformed = np.concatenate((continuous, hinges), axis=1)
    if feature_set == "hinge":
        return transformed
    if feature_set == "full":
        index = {value: position for position, value in enumerate(vocabulary)}
        components = np.zeros((len(rows), len(vocabulary)), dtype=float)
        for row_index, row in enumerate(rows):
            components[row_index, index[str(row["component_a_cas"])]] = 1.0
            components[row_index, index[str(row["component_b_cas"])]] = 1.0
        return np.concatenate((transformed, components), axis=1)
    raise KeyError(feature_set)


def _fit_predict(
    candidate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    training_target = np.asarray([float(row["residual"]) for row in training_rows])
    if candidate["algorithm"] == "median":
        median = float(np.median(training_target))
        return np.full(len(target_rows), median), {
            "algorithm": "median",
            "candidate": dict(candidate),
            "intercept": median,
            "feature_names": [],
        }
    training = _design(training_rows, str(candidate["feature_set"]), vocabulary)
    target = _design(target_rows, str(candidate["feature_set"]), vocabulary)
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    standardized_training = (training - mean) / scale
    standardized_target = (target - mean) / scale
    algorithm = str(candidate["algorithm"])
    if algorithm == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=float(candidate["alpha"]))
    elif algorithm == "huber":
        from sklearn.linear_model import HuberRegressor

        model = HuberRegressor(
            alpha=float(candidate["alpha"]),
            epsilon=float(candidate["epsilon"]),
            max_iter=2_000,
            tol=1e-8,
        )
    elif algorithm == "quantile":
        from sklearn.linear_model import QuantileRegressor

        model = QuantileRegressor(
            quantile=0.5,
            alpha=float(candidate["alpha"]),
            solver="highs",
        )
    else:
        raise KeyError(algorithm)
    model.fit(standardized_training, training_target)
    prediction = np.asarray(model.predict(standardized_target), dtype=float)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Ma v2 candidate produced non-finite values: {candidate['name']}")
    continuous_names = list(CONTINUOUS_FEATURES)
    if candidate["feature_set"] in {"hinge", "full"}:
        continuous_names.extend(HINGE_FEATURES)
    if candidate["feature_set"] == "full":
        continuous_names.extend(f"component::{value}" for value in vocabulary)
    parameters = {
        "algorithm": algorithm,
        "candidate": dict(candidate),
        "feature_names": continuous_names,
        "feature_mean": mean.astype(float).tolist(),
        "feature_scale": scale.astype(float).tolist(),
        "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": float(model.intercept_),
    }
    return prediction, parameters


def _portable_predict(
    parameters: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[str],
) -> np.ndarray:
    if parameters["algorithm"] == "median":
        return np.full(len(rows), float(parameters["intercept"]), dtype=float)
    candidate = parameters["candidate"]
    design = _design(rows, str(candidate["feature_set"]), vocabulary)
    mean = np.asarray(parameters["feature_mean"], dtype=float)
    scale = np.asarray(parameters["feature_scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if (
        design.shape[1] != len(mean)
        or mean.shape != scale.shape
        or scale.shape != coefficients.shape
        or np.any(scale <= 0.0)
    ):
        raise RuntimeError("Ma v2 portable parameter shapes are invalid")
    prediction = ((design - mean) / scale) @ coefficients + float(
        parameters["intercept"]
    )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Ma v2 portable prediction is non-finite")
    return np.asarray(prediction, dtype=float)


def _metrics(prediction: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)
    target = np.asarray([float(row["target"]) for row in rows], dtype=float)
    return parent._metrics(predicted, target)


def _candidate_cv(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    salt: str,
    vocabulary: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    assignments = _balanced_folds(
        [str(row["pair_id"]) for row in rows], folds=folds, salt=salt
    )
    predictions = {
        str(candidate["name"]): np.full(len(rows), np.nan, dtype=float)
        for candidate in CANDIDATES
    }
    for fold in range(folds):
        train_indices = np.flatnonzero(assignments != fold)
        test_indices = np.flatnonzero(assignments == fold)
        training = [rows[index] for index in train_indices]
        target = [rows[index] for index in test_indices]
        for candidate in CANDIDATES:
            values, _ = _fit_predict(candidate, training, target, vocabulary)
            predictions[str(candidate["name"])][test_indices] = values
    metrics = {
        name: _metrics(
            np.asarray([float(row["baseline"]) for row in rows]) + values,
            rows,
        )
        for name, values in predictions.items()
    }
    return predictions, metrics


def _complexity(candidate: Mapping[str, Any], vocabulary: Sequence[str]) -> int:
    if candidate["feature_set"] == "none":
        return 1
    count = len(CONTINUOUS_FEATURES)
    if candidate["feature_set"] in {"hinge", "full"}:
        count += len(HINGE_FEATURES)
    if candidate["feature_set"] == "full":
        count += len(vocabulary)
    return count


def _select(
    metrics: Mapping[str, Mapping[str, float]], vocabulary: Sequence[str]
) -> str:
    by_name = {str(candidate["name"]): candidate for candidate in CANDIDATES}
    return min(
        metrics,
        key=lambda name: (
            float(metrics[name]["mae"]) + 0.0001 * _complexity(by_name[name], vocabulary),
            float(metrics[name]["rmse"]),
            -float(metrics[name]["spearman"]),
            _complexity(by_name[name], vocabulary),
            name,
        ),
    )


def _repeated_nested(
    rows: Sequence[Mapping[str, Any]], vocabulary: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    repeated = np.zeros((OUTER_REPEATS, len(rows)), dtype=float)
    audit = []
    by_name = {str(candidate["name"]): candidate for candidate in CANDIDATES}
    for repeat in range(OUTER_REPEATS):
        outer = _balanced_folds(
            [str(row["pair_id"]) for row in rows],
            folds=OUTER_FOLDS,
            salt=f"{FOLD_SALT}|outer|{repeat}",
        )
        for fold in range(OUTER_FOLDS):
            train_indices = np.flatnonzero(outer != fold)
            test_indices = np.flatnonzero(outer == fold)
            training = [rows[index] for index in train_indices]
            target = [rows[index] for index in test_indices]
            _, inner_metrics = _candidate_cv(
                training,
                folds=INNER_FOLDS,
                salt=f"{FOLD_SALT}|inner|{repeat}|{fold}",
                vocabulary=vocabulary,
            )
            selected = _select(inner_metrics, vocabulary)
            residual, _ = _fit_predict(by_name[selected], training, target, vocabulary)
            repeated[repeat, test_indices] = np.asarray(
                [float(row["baseline"]) for row in target]
            ) + residual
            audit.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "training_pairs": len(training),
                    "held_out_pairs": len(target),
                    "selected_candidate": selected,
                    "inner_selected_mae": inner_metrics[selected]["mae"],
                }
            )
    if not np.all(np.isfinite(repeated)):
        raise RuntimeError("Ma v2 nested predictions are incomplete")
    return repeated.mean(axis=0), {
        "folds": audit,
        "selection_counts": {
            name: sum(row["selected_candidate"] == name for row in audit)
            for name in sorted({row["selected_candidate"] for row in audit})
        },
    }


def _component_cold(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[str],
) -> dict[str, Any]:
    molecules = list(vocabulary)
    prediction = []
    baseline = []
    target = []
    folds = []
    evaluated_pair_ids: set[str] = set()
    by_name = {str(candidate["name"]): candidate for candidate in CANDIDATES}
    for repeat in range(COMPONENT_COLD_REPEATS):
        assignments = {
            molecule: _hash_fold(
                molecule,
                folds=COMPONENT_COLD_FOLDS,
                salt=f"{FOLD_SALT}|component-cold|{repeat}",
            )
            for molecule in molecules
        }
        for fold in range(COMPONENT_COLD_FOLDS):
            held = {molecule for molecule, value in assignments.items() if value == fold}
            training = [
                row
                for row in rows
                if str(row["component_a_cas"]) not in held
                and str(row["component_b_cas"]) not in held
            ]
            testing = [
                row
                for row in rows
                if str(row["component_a_cas"]) in held
                and str(row["component_b_cas"]) in held
            ]
            if len(training) < 80 or len(testing) < 2:
                continue
            _, selection_metrics = _candidate_cv(
                training,
                folds=INNER_FOLDS,
                salt=f"{FOLD_SALT}|component-cold-select|{repeat}|{fold}",
                vocabulary=vocabulary,
            )
            selected = _select(selection_metrics, vocabulary)
            residual, _ = _fit_predict(
                by_name[selected], training, testing, vocabulary
            )
            local_baseline = np.asarray([float(row["baseline"]) for row in testing])
            prediction.extend((local_baseline + residual).tolist())
            baseline.extend(local_baseline.tolist())
            target.extend(float(row["target"]) for row in testing)
            evaluated_pair_ids.update(str(row["pair_id"]) for row in testing)
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "training_pairs": len(training),
                    "held_out_pairs": len(testing),
                    "held_out_molecules": len(held),
                    "component_leakage_count": 0,
                    "selected_candidate": selected,
                    "inner_selected_mae": selection_metrics[selected]["mae"],
                    "selection_used_held_out_outcomes": False,
                }
            )
    if len(prediction) < 100:
        raise RuntimeError("too few all-components-cold Ma v2 predictions")
    synthetic_rows = [{"target": value} for value in target]
    return {
        "predictions": len(prediction),
        "evaluated_fold_blocks": len(folds),
        "unique_evaluated_pairs": len(evaluated_pair_ids),
        "selection_nested_within_component_cold_training": True,
        "selection_counts": {
            name: sum(row["selected_candidate"] == name for row in folds)
            for name in sorted({row["selected_candidate"] for row in folds})
        },
        "model": _metrics(prediction, synthetic_rows),
        "strongest_component": _metrics(baseline, synthetic_rows),
        "folds": folds,
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    participant_rows: Sequence[Mapping[str, Any]],
    nested_prediction: np.ndarray,
) -> dict[str, Any]:
    pair_ids = [str(row["pair_id"]) for row in rows]
    pair_index = {value: index for index, value in enumerate(pair_ids)}
    participants = sorted({str(row["subject_id"]) for row in participant_rows})
    participant_index = {value: index for index, value in enumerate(participants)}
    shape = (len(participants), len(pair_ids))
    target = np.full(shape, np.nan, dtype=float)
    baseline = np.full(shape, np.nan, dtype=float)
    buckets: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        buckets[
            (
                participant_index[str(row["subject_id"])],
                pair_index[str(row["pair_id"])],
            )
        ].append(row)
    for (participant, pair), values in buckets.items():
        target[participant, pair] = float(
            np.mean([float(row["target"]) for row in values])
        )
        baseline[participant, pair] = float(
            np.mean([float(row["baseline"]) for row in values])
        )
    residual = nested_prediction - np.asarray([float(row["baseline"]) for row in rows])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    mae_gain = []
    spearman_gain = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_participants = generator.integers(0, len(participants), len(participants))
        sampled_pairs = generator.integers(0, len(pair_ids), len(pair_ids))
        with np.errstate(invalid="ignore"):
            target_mean = np.nanmean(
                target[sampled_participants][:, sampled_pairs], axis=0
            )
            baseline_mean = np.nanmean(
                baseline[sampled_participants][:, sampled_pairs], axis=0
            )
        calibrated = baseline_mean + residual[sampled_pairs]
        valid = (
            np.isfinite(target_mean)
            & np.isfinite(baseline_mean)
            & np.isfinite(calibrated)
        )
        if int(valid.sum()) < 100:
            continue
        observed = target_mean[valid]
        base = baseline_mean[valid]
        model = calibrated[valid]
        mae_gain.append(
            float(np.mean(np.abs(base - observed)) - np.mean(np.abs(model - observed)))
        )
        spearman_gain.append(
            shared.spearman(model, observed) - shared.spearman(base, observed)
        )
    if len(mae_gain) < int(BOOTSTRAP_DRAWS * 0.98):
        raise RuntimeError("too many invalid Ma v2 bootstrap draws")
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": len(mae_gain),
        "strongest_minus_calibrated_mae_95_interval": [
            float(value) for value in np.quantile(mae_gain, [0.025, 0.975])
        ],
        "calibrated_minus_strongest_spearman_95_interval": [
            float(value) for value in np.quantile(spearman_gain, [0.025, 0.975])
        ],
    }


def _fit_final(
    rows: Sequence[Mapping[str, Any]], vocabulary: Sequence[str]
) -> tuple[str, dict[str, Any], dict[str, dict[str, float]]]:
    repeated: dict[str, list[dict[str, float]]] = {
        str(candidate["name"]): [] for candidate in CANDIDATES
    }
    for repeat in range(OUTER_REPEATS):
        _, metrics = _candidate_cv(
            rows,
            folds=OUTER_FOLDS,
            salt=f"{FOLD_SALT}|final|{repeat}",
            vocabulary=vocabulary,
        )
        for name, row in metrics.items():
            repeated[name].append(row)
    averaged = {
        name: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in ("spearman", "mae", "rmse", "bias")
        }
        for name, values in repeated.items()
    }
    selected = _select(averaged, vocabulary)
    candidate = next(row for row in CANDIDATES if row["name"] == selected)
    fitted, parameters = _fit_predict(candidate, rows, rows, vocabulary)
    portable = _portable_predict(parameters, rows, vocabulary)
    maximum_delta = float(np.max(np.abs(fitted - portable)))
    if maximum_delta > 1e-10:
        raise RuntimeError("Ma v2 portable parameters differ from fitted model")
    parameters["component_vocabulary"] = list(vocabulary)
    parameters["portable_parity_maximum_absolute_error"] = maximum_delta
    return selected, parameters, averaged


def _markdown(report: Mapping[str, Any]) -> str:
    model = report["repeated_nested_pair_disjoint"]
    baseline = report["strongest_component"]
    cold = report["all_components_cold"]
    interval = report["bootstrap"]["strongest_minus_calibrated_mae_95_interval"]
    return "\n".join(
        [
            "# Ma 2021 혼합 강도 잔차 보정 v2",
            "",
            "| 평가 | Spearman | MAE | RMSE |",
            "|---|---:|---:|---:|",
            f"| 반복 nested pair-disjoint | {model['spearman']:.4f} | {model['mae']:.4f} | {model['rmse']:.4f} |",
            f"| Strongest component | {baseline['spearman']:.4f} | {baseline['mae']:.4f} | {baseline['rmse']:.4f} |",
            f"| All-components-cold v2 | {cold['model']['spearman']:.4f} | {cold['model']['mae']:.4f} | {cold['model']['rmse']:.4f} |",
            f"| All-components-cold max | {cold['strongest_component']['spearman']:.4f} | {cold['strongest_component']['mae']:.4f} | {cold['strongest_component']['rmse']:.4f} |",
            "",
            f"- MAE 개선 95% 구간: [{interval[0]:+.4f}, {interval[1]:+.4f}]",
            "- 개선 게이트: **"
            + ("PASS" if report["retrospective_improvement_gate"]["passed"] else "FAIL")
            + "**",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite Ma v2 outputs")
    predictions = args.predictions.resolve(strict=True)
    outcome = args.outcome.resolve(strict=True)
    blind_report = args.blind_report.resolve(strict=True)
    receipt = json.loads(args.receipt.resolve(strict=True).read_text(encoding="utf-8"))
    if receipt.get("outcome", {}).get("sha256") != _sha256(outcome):
        raise RuntimeError("Ma v2 outcome/receipt binding mismatch")
    blind = json.loads(blind_report.read_text(encoding="utf-8"))
    if blind.get("source_binding", {}).get("prediction_sha256") != _sha256(predictions):
        raise RuntimeError("Ma v2 blind report/prediction binding mismatch")
    if blind.get("source_binding", {}).get("outcome_sha256") != _sha256(outcome):
        raise RuntimeError("Ma v2 blind report/outcome binding mismatch")
    rows, participant_rows, data_audit = _load_rows(predictions, outcome)
    vocabulary = _component_vocabulary(rows)
    nested_prediction, nested_audit = _repeated_nested(rows, vocabulary)
    nested_metrics = _metrics(nested_prediction, rows)
    baseline_prediction = np.asarray([float(row["baseline"]) for row in rows])
    baseline_metrics = _metrics(baseline_prediction, rows)
    selected, parameters, final_selection = _fit_final(rows, vocabulary)
    selected_candidate = next(row for row in CANDIDATES if row["name"] == selected)
    component_cold = _component_cold(rows, vocabulary)
    bootstrap = _bootstrap(rows, participant_rows, nested_prediction)
    relative_reduction = 1.0 - nested_metrics["mae"] / baseline_metrics["mae"]
    seen_component_checks = {
        "nested_pair_disjoint_mae_at_most_0_245": nested_metrics["mae"] <= TARGET_MAE,
        "nested_pair_disjoint_mae_lower_than_strongest": nested_metrics["mae"]
        < baseline_metrics["mae"],
        "nested_pair_disjoint_rmse_lower_than_strongest": nested_metrics["rmse"]
        < baseline_metrics["rmse"],
        "bootstrap_mae_gain_lower_above_zero": bootstrap[
            "strongest_minus_calibrated_mae_95_interval"
        ][0]
        > 0.0,
        "bootstrap_spearman_gain_lower_at_least_minus_0_02": bootstrap[
            "calibrated_minus_strongest_spearman_95_interval"
        ][0]
        >= -0.02,
    }
    component_cold_checks = {
        "all_components_cold_mae_not_worse": component_cold["model"]["mae"]
        <= component_cold["strongest_component"]["mae"],
        "all_components_cold_zero_leakage": all(
            row["component_leakage_count"] == 0 for row in component_cold["folds"]
        ),
        "all_components_cold_model_selection_target_excluded": all(
            row["selection_used_held_out_outcomes"] is False
            for row in component_cold["folds"]
        ),
        "portable_numeric_parity_at_most_1e_10": parameters[
            "portable_parity_maximum_absolute_error"
        ]
        <= 1e-10,
    }
    checks = {**seen_component_checks, **component_cold_checks}
    seen_component_passed = all(seen_component_checks.values())
    component_cold_passed = all(component_cold_checks.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "ma_2021_v2_pair_disjoint_improvement_confirmed"
            if seen_component_passed and component_cold_passed
            else (
                "ma_2021_v2_pair_disjoint_improvement_but_component_cold_failed"
                if seen_component_passed
                else "ma_2021_v2_pair_disjoint_improvement_not_confirmed"
            )
        ),
        "development_timing": "post_ma_2021_blind_outcome_after_frozen_parent_scoring",
        "source_binding": {
            "blind_predictions_sha256": _sha256(predictions),
            "blind_report_sha256": _sha256(blind_report),
            "outcome_sha256": _sha256(outcome),
            "receipt_sha256": _sha256(args.receipt.resolve(strict=True)),
            "frozen_parent_script_sha256": _sha256(Path(parent.__file__).resolve()),
        },
        "data": data_audit,
        "candidate_contract": list(CANDIDATES),
        "feature_contract": {
            "continuous": list(CONTINUOUS_FEATURES),
            "hinge": list(HINGE_FEATURES),
            "component_vocabulary": list(vocabulary),
            "target": "IAB - participant_mean(max(IA,IB))",
            "forbidden": ["IAmix", "IBmix", "PAB", "Group", "Repeat"],
        },
        "protocol": {
            "outer_repeats": OUTER_REPEATS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "component_cold_repeats": COMPONENT_COLD_REPEATS,
            "component_cold_folds": COMPONENT_COLD_FOLDS,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "nested_audit": nested_audit,
        },
        "repeated_nested_pair_disjoint": nested_metrics,
        "strongest_component": baseline_metrics,
        "relative_mae_reduction": relative_reduction,
        "all_components_cold": component_cold,
        "bootstrap": bootstrap,
        "final_model": {
            "selected_candidate": selected,
            "parameters": parameters,
            "selection_metrics": final_selection[selected],
            "runtime_primary_score_weight": 0.0,
            "portable_numeric_contract": (
                selected_candidate["algorithm"]
                in {"median", "ridge", "huber", "quantile"}
                and parameters["portable_parity_maximum_absolute_error"] <= 1e-10
            ),
        },
        "retrospective_improvement_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "seen_component_new_pair_gate": {
            "passed": seen_component_passed,
            "checks": seen_component_checks,
            "scope": "new exact pairs among components represented in training",
        },
        "strict_all_components_cold_gate": {
            "passed": component_cold_passed,
            "checks": component_cold_checks,
            "scope": "both components absent from fit and model selection",
        },
        "external_reproduction_complete": False,
        "human_olfactory_90_percent_certified": False,
        "claim_boundary": (
            "Post-outcome residual calibration on the Ma 2021 binary-mixture "
            "dataset. Nested exact-pair and all-components-cold estimates reduce "
            "reuse bias, but this remains retrospective until reproduced on a new "
            "external mixture-intensity target. Runtime weight therefore remains zero."
        ),
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = build(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "nested": report["repeated_nested_pair_disjoint"],
                "baseline": report["strongest_component"],
                "selected": report["final_model"]["selected_candidate"],
                "gate": report["retrospective_improvement_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
