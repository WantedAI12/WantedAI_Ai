#!/usr/bin/env python
"""Test concentration-calibrated headspace features on DREAM mixtures.

The DREAM mixture files do not publish quantitative liquid composition for
every component.  This benchmark therefore tests an explicit equal-liquid-
contribution sensitivity analysis.  It never promotes the result to an
absolute headspace or production claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from ogb.utils import smiles2graph
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragrance_ai.research.headspace import (  # noqa: E402
    HeadspaceSensoryHub,
    load_calibrated_response_exponent,
)
from scripts import benchmark_dream_mixture_2025 as base  # noqa: E402
from scripts import benchmark_dream_pair_ensemble_v2 as pair  # noqa: E402


SCHEMA = "dream-headspace-retrospective/v3"
ALPHAS = (1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0, 300_000.0)
BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


def sha256(path: Path) -> str:
    return base._sha256(path)


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()


def required_files(dream_root: Path) -> dict[str, Path]:
    return {
        "training_pairs": dream_root
        / "Training_Dataset"
        / "TrainingData_mixturedist.csv",
        "training_definitions": dream_root
        / "Training_Dataset"
        / "Mixure_Definitions_Training_set.csv",
        "test_pairs": dream_root / "Test_Dataset" / "Test_set_mixturedist.csv",
        "test_definitions": dream_root
        / "Test_Dataset"
        / "Test_set_Mixure_Definitions.csv",
        "validation_raw": dream_root
        / "Validation_Dataset"
        / "Dream_validation_TestRetest.csv",
        "validation_definitions": dream_root
        / "Validation_Dataset"
        / "Mixure_Definitions_Validation_set.csv",
        "public_test_predictions": dream_root
        / "Predictions"
        / "Test_set_SOTA_Ensemble_Post_Challenge_Predictions.csv",
        "public_test_pair_index": dream_root
        / "Predictions"
        / "Test_set_Prediction_top6_Teams.csv",
        "public_validation_predictions": dream_root
        / "Predictions"
        / "Validation_set_Prediction_top6_Teams.csv",
        "pom_profiles": dream_root
        / "PostChallenge_Model"
        / "Dataset"
        / "openpom_ensemble_predictions_results.csv",
        "cid_to_smiles": dream_root
        / "PostChallenge_Model"
        / "Dataset"
        / "cid_to_smiles.csv",
    }


@dataclass(frozen=True)
class WeightedMixture:
    component_ids: tuple[int, ...]
    pom_mean: np.ndarray
    pom_std: np.ndarray
    rdkit_mean: np.ndarray
    rdkit_std: np.ndarray
    morgan_mean: np.ndarray
    pommix_mean: np.ndarray
    pommix_std: np.ndarray
    physical_summary: np.ndarray


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ values
    variance = weights @ ((values - mean) ** 2)
    return mean, np.sqrt(np.maximum(variance, 0.0))


def weighted_mixtures(
    definitions: Mapping[Any, Sequence[int]],
    *,
    pom: Mapping[int, np.ndarray],
    rdkit: Mapping[int, np.ndarray],
    morgan: Mapping[int, np.ndarray],
    pommix: Mapping[int, np.ndarray],
    log_pressure: Mapping[int, float | None],
    missing_log_pressure: float,
    response_exponent: float,
) -> dict[Any, WeightedMixture]:
    result = {}
    for key, component_ids in definitions.items():
        cids = tuple(int(cid) for cid in component_ids)
        logs = np.asarray(
            [
                log_pressure[cid]
                if log_pressure[cid] is not None
                else missing_log_pressure
                for cid in cids
            ],
            dtype=float,
        )
        centered = logs - float(np.mean(logs))
        weights = np.exp(response_exponent * centered)
        weights /= weights.sum()
        pom_mean, pom_std = weighted_mean_std(
            np.asarray([pom[cid] for cid in cids], dtype=float), weights
        )
        rdkit_mean, rdkit_std = weighted_mean_std(
            np.asarray([rdkit[cid] for cid in cids], dtype=float), weights
        )
        morgan_mean, _ = weighted_mean_std(
            np.asarray([morgan[cid] for cid in cids], dtype=float), weights
        )
        pommix_mean, pommix_std = weighted_mean_std(
            np.asarray([pommix[cid] for cid in cids], dtype=float), weights
        )
        coverage = float(np.mean([log_pressure[cid] is not None for cid in cids]))
        summary = np.asarray(
            [
                float(np.mean(logs)),
                float(np.std(logs)),
                float(np.min(logs)),
                float(np.max(logs)),
                coverage,
                float(len(cids)),
            ],
            dtype=float,
        )
        result[key] = WeightedMixture(
            component_ids=cids,
            pom_mean=pom_mean,
            pom_std=pom_std,
            rdkit_mean=rdkit_mean,
            rdkit_std=rdkit_std,
            morgan_mean=morgan_mean,
            pommix_mean=pommix_mean,
            pommix_std=pommix_std,
            physical_summary=summary,
        )
    return result


def vector_pair_features(left: np.ndarray, right: np.ndarray) -> list[float]:
    difference = np.abs(left - right)
    values = [
        base._cosine(left, right),
        base._vector_correlation(left, right),
        float(np.mean(difference)),
        float(np.sqrt(np.mean(difference * difference))),
        float(np.max(difference)),
    ]
    values.extend(difference.tolist())
    values.extend((left * right).tolist())
    return values


def rdkit_pair_features(left: np.ndarray, right: np.ndarray) -> list[float]:
    difference = np.abs(left - right)
    values = [
        base._cosine(left, right),
        base._vector_correlation(left, right),
        float(np.mean(difference)),
        float(np.sqrt(np.mean(difference * difference))),
    ]
    values.extend(np.log1p(difference).tolist())
    return values


def headspace_pair_features(first: WeightedMixture, second: WeightedMixture) -> np.ndarray:
    values = []
    for name in ("pom_mean", "pom_std", "pommix_mean", "pommix_std"):
        values.extend(vector_pair_features(getattr(first, name), getattr(second, name)))
    for name in ("rdkit_mean", "rdkit_std"):
        values.extend(rdkit_pair_features(getattr(first, name), getattr(second, name)))
    morgan_difference = np.abs(first.morgan_mean - second.morgan_mean)
    values.extend(
        [
            base._cosine(first.morgan_mean, second.morgan_mean),
            float(np.mean(morgan_difference)),
            float(np.sqrt(np.mean(morgan_difference * morgan_difference))),
        ]
    )
    lower = np.minimum(first.physical_summary, second.physical_summary)
    upper = np.maximum(first.physical_summary, second.physical_summary)
    values.extend(lower.tolist())
    values.extend(upper.tolist())
    values.extend((lower + upper).tolist())
    values.extend((upper - lower).tolist())
    return np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )


def headspace_feature_names(
    pom_names: Sequence[str], rdkit_names: Sequence[str], pommix_width: int
) -> list[str]:
    names = []
    dimensions = {
        "pom_mean": list(pom_names),
        "pom_std": list(pom_names),
        "pommix_mean": [str(index) for index in range(pommix_width)],
        "pommix_std": [str(index) for index in range(pommix_width)],
    }
    for aggregate, labels in dimensions.items():
        names.extend(
            [
                f"headspace::{aggregate}::cosine",
                f"headspace::{aggregate}::correlation",
                f"headspace::{aggregate}::mean_absolute_difference",
                f"headspace::{aggregate}::root_mean_square_difference",
                f"headspace::{aggregate}::maximum_absolute_difference",
            ]
        )
        names.extend(
            f"headspace::{aggregate}::absolute_difference::{label}" for label in labels
        )
        names.extend(f"headspace::{aggregate}::product::{label}" for label in labels)
    for aggregate in ("rdkit_mean", "rdkit_std"):
        names.extend(
            [
                f"headspace::{aggregate}::cosine",
                f"headspace::{aggregate}::correlation",
                f"headspace::{aggregate}::mean_absolute_difference",
                f"headspace::{aggregate}::root_mean_square_difference",
            ]
        )
        names.extend(
            f"headspace::{aggregate}::log1p_absolute_difference::{label}"
            for label in rdkit_names
        )
    names.extend(
        [
            "headspace::morgan_mean::cosine",
            "headspace::morgan_mean::mean_absolute_difference",
            "headspace::morgan_mean::root_mean_square_difference",
        ]
    )
    summary = ("log_pressure_mean", "log_pressure_std", "log_pressure_min", "log_pressure_max", "coverage", "size")
    for aggregate in ("minimum", "maximum", "sum", "absolute_difference"):
        names.extend(f"headspace::physical::{aggregate}::{name}" for name in summary)
    return names


def headspace_rows(
    pairs: pd.DataFrame,
    representations: Mapping[Any, WeightedMixture],
    *,
    training: bool,
) -> np.ndarray:
    rows = []
    for _, row in pairs.iterrows():
        if training:
            source = str(row["Dataset"]).strip()
            first_key = (source, base._label(row["Mixture 1"]))
            second_key = (source, base._label(row["Mixture 2"]))
        else:
            first_key = base._label(row["Mixture 1"])
            second_key = base._label(row["Mixture 2"])
        rows.append(
            headspace_pair_features(
                representations[first_key], representations[second_key]
            )
        )
    return np.asarray(rows)


def select_ridge(
    features: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> tuple[dict[str, Any], Any]:
    unique_groups = sorted(set(str(value) for value in groups))
    splitter = GroupKFold(n_splits=len(unique_groups))
    rows = []
    for alpha in ALPHAS:
        prediction = np.zeros(len(target), dtype=float)
        for training, validation in splitter.split(features, target, groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[training], target[training])
            prediction[validation] = np.clip(
                model.predict(features[validation]), 0.0, 1.0
            )
        result = base.metrics(prediction, target)
        rows.append({"alpha": alpha, "source_holdout": result})
    selected = min(rows, key=lambda row: (float(row["source_holdout"]["mae"]), row["alpha"]))
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(selected["alpha"])))
    model.fit(features, target)
    return {"candidates": rows, "selected": selected}, model


def group_oof_pair_ensemble(
    features: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    unique_groups = sorted(set(str(value) for value in groups))
    splitter = GroupKFold(n_splits=len(unique_groups))
    prediction = np.zeros(len(target), dtype=float)
    member_weights = np.asarray([weight for _alpha, weight in pair.MEMBER_SPECS])
    for training, validation in splitter.split(features, target, groups):
        models = [
            pair._fit_member(features[training], target[training], alpha)
            for alpha, _weight in pair.MEMBER_SPECS
        ]
        member_prediction = np.asarray(
            [pair._predict(model, features[validation]) for model in models]
        )
        prediction[validation] = np.clip(member_weights @ member_prediction, 0.0, 1.0)
    return prediction


def nested_residual_oof(
    pair_features: np.ndarray,
    headspace_features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Predict each source from residual targets that also exclude that source."""
    unique_groups = sorted(set(str(value) for value in groups))
    outer = GroupKFold(n_splits=len(unique_groups))
    prediction = np.zeros(len(target), dtype=float)
    for outer_training, outer_validation in outer.split(pair_features, target, groups):
        inner_groups = groups[outer_training]
        inner_baseline = group_oof_pair_ensemble(
            pair_features[outer_training], target[outer_training], inner_groups
        )
        inner_residual = target[outer_training] - inner_baseline
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(headspace_features[outer_training], inner_residual)
        prediction[outer_validation] = model.predict(
            headspace_features[outer_validation]
        )
    return prediction


def point_checks(
    candidate_test: Mapping[str, float | int],
    candidate_validation: Mapping[str, float | int],
    current_test: Mapping[str, float | int],
    current_validation: Mapping[str, float | int],
) -> dict[str, bool]:
    return {
        "test_pearson_above_current": candidate_test["pearson"] > current_test["pearson"],
        "test_spearman_above_current": candidate_test["spearman"] > current_test["spearman"],
        "test_rmse_below_current": candidate_test["rmse"] < current_test["rmse"],
        "test_mae_below_current": candidate_test["mae"] < current_test["mae"],
        "validation_pearson_above_current": candidate_validation["pearson"]
        > current_validation["pearson"],
        "validation_spearman_above_current": candidate_validation["spearman"]
        > current_validation["spearman"],
        "validation_rmse_below_current": candidate_validation["rmse"]
        < current_validation["rmse"],
        "validation_mae_below_current": candidate_validation["mae"]
        < current_validation["mae"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-root", type=Path, required=True)
    parser.add_argument("--pommix-root", type=Path, required=True)
    parser.add_argument("--pair-source-root", type=Path, required=True)
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
        "--calibration",
        type=Path,
        default=ROOT / "benchmarks" / "concentration_headspace_calibration_v1.json",
    )
    parser.add_argument(
        "--current-report",
        type=Path,
        default=ROOT / "benchmarks" / "dream_pair_ensemble_retrospective_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "dream_headspace_retrospective_v3.json",
    )
    args = parser.parse_args()
    dream_root = args.dream_root.expanduser().resolve(strict=True)
    pommix_root = args.pommix_root.expanduser().resolve(strict=True)
    pair_root = args.pair_source_root.expanduser().resolve(strict=True)
    hub_path = args.hub.expanduser().resolve(strict=True)
    hub_report_path = args.hub_report.expanduser().resolve(strict=True)
    calibration_path = args.calibration.expanduser().resolve(strict=True)
    current_report_path = args.current_report.expanduser().resolve(strict=True)
    if git_commit(dream_root) != base.DREAM_COMMIT:
        raise RuntimeError("unsupported DREAM source commit")
    required = required_files(dream_root)
    changed = [
        name
        for name, path in required.items()
        if not path.is_file() or sha256(path) != base.DREAM_FILE_SHA256[name]
    ]
    if changed:
        raise RuntimeError("DREAM source files changed: " + ", ".join(changed))

    pom, rdkit, morgan, scaffolds, smiles, pom_names, rdkit_names = base._component_features(
        dream_root
    )
    train_definitions = base._definitions(required["training_definitions"], training=True)
    test_definitions = base._definitions(required["test_definitions"], training=False)
    validation_definitions = base._definitions(
        required["validation_definitions"], training=False
    )
    train_representations = base._representations(train_definitions, pom, rdkit, morgan)
    test_representations = base._representations(test_definitions, pom, rdkit, morgan)
    validation_representations = base._representations(
        validation_definitions, pom, rdkit, morgan
    )
    pommix, pommix_audit = base._pommix_embeddings(pommix_root, smiles)
    train_pommix = base._embedding_aggregates(train_definitions, pommix)
    test_pommix = base._embedding_aggregates(test_definitions, pommix)
    validation_pommix = base._embedding_aggregates(validation_definitions, pommix)
    current_feature_names = [
        *base._mixture_feature_names(pom_names, rdkit_names),
        *base._embedding_feature_names(pommix_audit["embedding_dimensions"]),
    ]
    train_pairs = pd.read_csv(required["training_pairs"])
    current_train, train_y, train_groups = base._training_rows(
        required["training_pairs"], train_representations
    )
    current_train = np.column_stack(
        [current_train, base._embedding_rows(train_pairs, train_pommix, training=True)]
    )
    test_pairs = pd.read_csv(required["test_pairs"])
    current_test, test_y = base._evaluation_rows(
        test_pairs, test_representations, "Experimental values"
    )
    current_test = np.column_stack(
        [current_test, base._embedding_rows(test_pairs, test_pommix, training=False)]
    )
    validation_pairs = pd.read_csv(required["public_validation_predictions"])
    current_validation, validation_y = base._evaluation_rows(
        validation_pairs, validation_representations, "ExpMean_combined"
    )
    current_validation = np.column_stack(
        [
            current_validation,
            base._embedding_rows(
                validation_pairs, validation_pommix, training=False
            ),
        ]
    )
    if current_train.shape[1] != len(current_feature_names):
        raise RuntimeError("current DREAM feature contract changed")

    model_root = dream_root / "SOTA" / "3-Pair_Model" / "finetuned_model"
    pair_model, pair_data, pairdata, _ = pair._load_pair_model(
        pair_root, model_root / "config.json", model_root / "model.pt"
    )
    pair_smiles_path = dream_root / "SOTA" / "data" / "raw" / "cid_to_smiles.json"
    if sha256(pair_smiles_path) != pair.PAIR_SMILES_SHA256:
        raise RuntimeError("odor-pair SMILES changed")
    exact_smiles = {
        int(cid): text
        for cid, text in json.loads(pair_smiles_path.read_text(encoding="utf-8")).items()
    }
    graphs = {
        cid: pairdata.to_torch(smiles2graph(exact_smiles[cid])) for cid in sorted(smiles)
    }
    published_train_embeddings = pair._precomputed_training_embeddings(
        dream_root / "SOTA" / "output" / "pair_model_Embedding.csv", train_definitions
    )
    generated_train_embeddings = pair._generated_embeddings(
        train_definitions, pair_model, pair_data, pairdata, graphs
    )
    reproduction_error = max(
        float(np.max(np.abs(generated_train_embeddings[key] - published_train_embeddings[key])))
        for key in train_definitions
    )
    if reproduction_error > pair.PAIR_REPRODUCTION_TOLERANCE:
        raise RuntimeError("odor-pair embedding reproduction failed")
    test_embeddings = pair._generated_embeddings(
        test_definitions, pair_model, pair_data, pairdata, graphs
    )
    validation_embeddings = pair._generated_embeddings(
        validation_definitions, pair_model, pair_data, pairdata, graphs
    )
    generated_rows = np.vstack(
        [
            *[generated_train_embeddings[key] for key in train_definitions],
            *[test_embeddings[key] for key in test_definitions],
            *[validation_embeddings[key] for key in validation_definitions],
        ]
    ).astype(np.float32)
    generated_hash = hashlib.sha256(generated_rows.tobytes()).hexdigest()
    if generated_hash != pair.PAIR_GENERATED_EMBEDDING_SHA256:
        raise RuntimeError("generated odor-pair embeddings changed")
    pair_train = pair._pair_rows(train_pairs, generated_train_embeddings, training=True)
    pair_test = pair._pair_rows(test_pairs, test_embeddings, training=False)
    pair_validation = pair._pair_rows(
        validation_pairs, validation_embeddings, training=False
    )
    pair_train_x = np.column_stack([current_train, pair_train])
    pair_test_x = np.column_stack([current_test, pair_test])
    pair_validation_x = np.column_stack([current_validation, pair_validation])
    pair_models = [
        pair._fit_member(pair_train_x, train_y, alpha)
        for alpha, _weight in pair.MEMBER_SPECS
    ]
    member_weights = np.asarray([weight for _alpha, weight in pair.MEMBER_SPECS])
    current_test_prediction = np.clip(
        member_weights @ np.asarray([pair._predict(model, pair_test_x) for model in pair_models]),
        0.0,
        1.0,
    )
    current_validation_prediction = np.clip(
        member_weights
        @ np.asarray([pair._predict(model, pair_validation_x) for model in pair_models]),
        0.0,
        1.0,
    )
    current_test_metrics = base.metrics(current_test_prediction, test_y)
    current_validation_metrics = base.metrics(current_validation_prediction, validation_y)
    frozen = json.loads(current_report_path.read_text(encoding="utf-8"))
    if (
        abs(float(frozen["test"]["candidate"]["pearson"]) - float(current_test_metrics["pearson"]))
        > 1e-12
        or abs(
            float(frozen["validation"]["candidate"]["pearson"])
            - float(current_validation_metrics["pearson"])
        )
        > 1e-12
    ):
        raise RuntimeError("reconstructed Pair-GNN v2 differs from frozen report")

    response_exponent = load_calibrated_response_exponent(calibration_path)
    all_needed = sorted(
        {
            cid
            for definitions in (train_definitions, test_definitions, validation_definitions)
            for components in definitions.values()
            for cid in components
        }
    )
    train_cids = {
        cid for components in train_definitions.values() for cid in components
    }
    evidence_counts: dict[str, int] = {}
    log_pressure: dict[int, float | None] = {}
    with HeadspaceSensoryHub(hub_path, report=hub_report_path) as hub:
        for cid in all_needed:
            evidence = hub.vapor_pressure(cid)
            if evidence is None:
                log_pressure[cid] = None
                evidence_class = "missing"
            else:
                log_pressure[cid] = math.log(evidence.pressure_pa)
                evidence_class = evidence.evidence_class
            evidence_counts[evidence_class] = evidence_counts.get(evidence_class, 0) + 1
    training_resolved = [
        float(log_pressure[cid])
        for cid in train_cids
        if log_pressure[cid] is not None
    ]
    if not training_resolved:
        raise RuntimeError("training molecules have no headspace evidence")
    missing_log_pressure = float(np.median(training_resolved))
    weighted_train = weighted_mixtures(
        train_definitions,
        pom=pom,
        rdkit=rdkit,
        morgan=morgan,
        pommix=pommix,
        log_pressure=log_pressure,
        missing_log_pressure=missing_log_pressure,
        response_exponent=response_exponent,
    )
    weighted_test = weighted_mixtures(
        test_definitions,
        pom=pom,
        rdkit=rdkit,
        morgan=morgan,
        pommix=pommix,
        log_pressure=log_pressure,
        missing_log_pressure=missing_log_pressure,
        response_exponent=response_exponent,
    )
    weighted_validation = weighted_mixtures(
        validation_definitions,
        pom=pom,
        rdkit=rdkit,
        morgan=morgan,
        pommix=pommix,
        log_pressure=log_pressure,
        missing_log_pressure=missing_log_pressure,
        response_exponent=response_exponent,
    )
    headspace_names = headspace_feature_names(
        pom_names, rdkit_names, int(pommix_audit["embedding_dimensions"])
    )
    headspace_train = headspace_rows(train_pairs, weighted_train, training=True)
    headspace_test = headspace_rows(test_pairs, weighted_test, training=False)
    headspace_validation = headspace_rows(
        validation_pairs, weighted_validation, training=False
    )
    if headspace_train.shape[1] != len(headspace_names):
        raise RuntimeError("headspace feature contract changed")

    model_inputs = {
        "headspace_only": (headspace_train, headspace_test, headspace_validation),
        "pair_plus_headspace": (
            np.column_stack([pair_train_x, headspace_train]),
            np.column_stack([pair_test_x, headspace_test]),
            np.column_stack([pair_validation_x, headspace_validation]),
        ),
    }
    training_selection = {}
    direct_predictions = {}
    candidate_rows = []
    for name, (training_x, test_x, validation_x) in model_inputs.items():
        selection, model = select_ridge(training_x, train_y, train_groups)
        test_prediction = np.clip(model.predict(test_x), 0.0, 1.0)
        validation_prediction = np.clip(model.predict(validation_x), 0.0, 1.0)
        training_selection[name] = selection
        direct_predictions[name] = (test_prediction, validation_prediction)
        test_metrics = base.metrics(test_prediction, test_y)
        validation_metrics = base.metrics(validation_prediction, validation_y)
        checks = point_checks(
            test_metrics,
            validation_metrics,
            current_test_metrics,
            current_validation_metrics,
        )
        candidate_rows.append(
            {
                "name": name,
                "current_pair_weight": 0.0,
                "headspace_candidate_weight": 1.0,
                "test": test_metrics,
                "validation": validation_metrics,
                "point_checks": checks,
                "point_pareto": all(checks.values()),
            }
        )
        for headspace_weight in BLEND_WEIGHTS:
            if headspace_weight in (0.0, 1.0):
                continue
            blended_test = np.clip(
                (1.0 - headspace_weight) * current_test_prediction
                + headspace_weight * test_prediction,
                0.0,
                1.0,
            )
            blended_validation = np.clip(
                (1.0 - headspace_weight) * current_validation_prediction
                + headspace_weight * validation_prediction,
                0.0,
                1.0,
            )
            test_metrics = base.metrics(blended_test, test_y)
            validation_metrics = base.metrics(blended_validation, validation_y)
            checks = point_checks(
                test_metrics,
                validation_metrics,
                current_test_metrics,
                current_validation_metrics,
            )
            name_with_weight = f"{name}_blend_{headspace_weight:.2f}"
            direct_predictions[name_with_weight] = (blended_test, blended_validation)
            candidate_rows.append(
                {
                    "name": name_with_weight,
                    "current_pair_weight": 1.0 - headspace_weight,
                    "headspace_candidate_weight": headspace_weight,
                    "test": test_metrics,
                    "validation": validation_metrics,
                    "point_checks": checks,
                    "point_pareto": all(checks.values()),
                }
            )

    # Preserve the frozen Pair-GNN prediction and learn only a source-held-out
    # residual. This prevents headspace features from moving the entire scale.
    pair_source_oof = group_oof_pair_ensemble(pair_train_x, train_y, train_groups)
    residual_scale_search = []
    for alpha in ALPHAS:
        residual_oof = nested_residual_oof(
            pair_train_x,
            headspace_train,
            train_y,
            train_groups,
            alpha,
        )
        for scale in BLEND_WEIGHTS:
            training_prediction = np.clip(
                pair_source_oof + scale * residual_oof, 0.0, 1.0
            )
            residual_scale_search.append(
                {
                    "alpha": alpha,
                    "scale": scale,
                    "source_holdout": base.metrics(training_prediction, train_y),
                }
            )
    selected_scale = min(
        residual_scale_search,
        key=lambda row: (
            float(row["source_holdout"]["mae"]),
            row["scale"],
            row["alpha"],
        ),
    )
    residual_alpha = float(selected_scale["alpha"])
    residual_scale = float(selected_scale["scale"])
    residual_target = train_y - pair_source_oof
    residual_model = make_pipeline(StandardScaler(), Ridge(alpha=residual_alpha))
    residual_model.fit(headspace_train, residual_target)
    residual_test_prediction = np.clip(
        current_test_prediction
        + residual_scale * residual_model.predict(headspace_test),
        0.0,
        1.0,
    )
    residual_validation_prediction = np.clip(
        current_validation_prediction
        + residual_scale * residual_model.predict(headspace_validation),
        0.0,
        1.0,
    )
    residual_name = "pair_fixed_plus_headspace_residual"
    training_selection[residual_name] = {
        "residual_scale_search": residual_scale_search,
        "selected_scale": selected_scale,
        "baseline_prediction_is_source_group_oof": True,
        "residual_selection_is_nested_source_group_oof": True,
        "outer_source_is_excluded_from_inner_residual_targets": True,
    }
    direct_predictions[residual_name] = (
        residual_test_prediction,
        residual_validation_prediction,
    )
    residual_test_metrics = base.metrics(residual_test_prediction, test_y)
    residual_validation_metrics = base.metrics(
        residual_validation_prediction, validation_y
    )
    residual_checks = point_checks(
        residual_test_metrics,
        residual_validation_metrics,
        current_test_metrics,
        current_validation_metrics,
    )
    candidate_rows.append(
        {
            "name": residual_name,
            "construction": "frozen_pair_prediction_plus_training_selected_headspace_residual",
            "current_pair_weight": 1.0,
            "headspace_candidate_weight": residual_scale,
            "test": residual_test_metrics,
            "validation": residual_validation_metrics,
            "point_checks": residual_checks,
            "point_pareto": all(residual_checks.values()),
        }
    )
    eligible = [row for row in candidate_rows if row["point_pareto"]]
    selection_pool = eligible or candidate_rows
    selected = max(
        selection_pool,
        key=lambda row: (
            float(row["test"]["pearson"]) + float(row["validation"]["pearson"]),
            -float(row["test"]["rmse"]) - float(row["validation"]["rmse"]),
        ),
    )
    selected_test_prediction, selected_validation_prediction = direct_predictions[
        str(selected["name"])
    ]
    test_bootstrap = base._paired_bootstrap(
        test_y, selected_test_prediction, current_test_prediction
    )
    raw_validation = pd.read_csv(required["validation_raw"])
    validation_bootstrap = base._validation_two_way_bootstrap(
        raw_validation,
        validation_pairs,
        selected_validation_prediction,
        current_validation_prediction,
    )
    first = validation_pairs["ExpMean_test"].to_numpy(float)
    second = validation_pairs["ExpMean_retest"].to_numpy(float)
    test_retest = base._correlation(pearsonr, first, second)
    reliability = 2.0 * test_retest / (1.0 + test_retest)
    noise_ceiling = math.sqrt(max(0.0, reliability))
    normalized = float(selected["validation"]["pearson"]) / noise_ceiling
    normalized_interval = validation_bootstrap[
        "human_ceiling_normalized_candidate_pearson_95_interval"
    ]
    statistical_checks = {
        "test_rmse_gain_lower_above_zero": test_bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "test_pearson_gain_lower_above_zero": test_bootstrap[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
        "validation_rmse_gain_lower_above_zero": validation_bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "validation_pearson_gain_lower_above_zero": validation_bootstrap[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "point_pareto_only_research_candidate"
            if eligible
            else "headspace_candidate_rejected_not_point_pareto"
        ),
        "source": {
            "dream_commit": base.DREAM_COMMIT,
            "dream_files": {
                name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in required.items()
            },
            "pommix": pommix_audit,
            "odor_pair": {
                "commit": pair.PAIR_SOURCE_COMMIT,
                "weights_sha256": pair.PAIR_WEIGHTS_SHA256,
                "generated_embedding_rows_sha256": generated_hash,
                "training_reproduction_max_abs_error": reproduction_error,
            },
            "headspace_hub_sha256": sha256(hub_path),
            "headspace_hub_report_sha256": sha256(hub_report_path),
            "concentration_calibration_sha256": sha256(calibration_path),
            "current_pair_report_sha256": sha256(current_report_path),
        },
        "timing": {
            "candidate_design_after_public_outcomes_existed": True,
            "blend_selection_used_test_and_validation_outcomes": True,
            "post_selection_intervals_descriptive_only": True,
            "prospective_or_outcome_unopened": False,
        },
        "physical_model": {
            "assumption": "equal_liquid_component_contribution_relative_sensitivity_only",
            "response_exponent": response_exponent,
            "response_exponent_selected_on_dream_outcomes": False,
            "missing_pressure_fill": "training_component_log_pressure_median_neutral_weight",
            "training_missing_fill_log_pa": missing_log_pressure,
            "component_evidence": evidence_counts,
            "unique_components": len(all_needed),
            "resolved_components": sum(value is not None for value in log_pressure.values()),
            "absolute_headspace_claimed": False,
        },
        "implementation": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "current_feature_dimensions": len(current_feature_names),
            "odor_pair_feature_dimensions": len(pair._pair_feature_names()),
            "headspace_feature_dimensions": len(headspace_names),
            "headspace_feature_contract_sha256": base._canonical_json_sha256(
                headspace_names
            ),
            "allow_pickle": False,
        },
        "training_selection": training_selection,
        "current_pair_v2": {
            "test": current_test_metrics,
            "validation": current_validation_metrics,
        },
        "candidate_search": candidate_rows,
        "selection": {
            "selected": selected,
            "point_pareto_candidates": len(eligible),
            "outcome_aware": True,
            "eligible_for_production": False,
        },
        "selected_diagnostic": {
            "test": selected["test"],
            "validation": selected["validation"],
            "test_paired_bootstrap": test_bootstrap,
            "validation_subject_pair_two_way_bootstrap": validation_bootstrap,
            "validation_human_ceiling": noise_ceiling,
            "validation_human_ceiling_normalized_pearson": normalized,
            "validation_human_ceiling_normalized_pearson_95_interval": normalized_interval,
        },
        "gates": {
            "point_pareto": {"passed": bool(eligible)},
            "statistical_improvement": {
                "checks": statistical_checks,
                "passed": all(statistical_checks.values()),
            },
            "human_ceiling_90_percent": {
                "threshold": 0.90,
                "passed": normalized_interval[0] >= 0.90,
            },
            "production": {
                "passed": False,
                "runtime_primary_score_weight": 0.0,
            },
        },
        "claim_boundary": {
            "headspace_gc_ms_measurement": False,
            "quantitative_mixture_composition_available": False,
            "natural_language_recipe_accuracy_measured": False,
            "human_olfactory_90_percent_certified": False,
            "commercial_runtime_weight": 0.0,
        },
    }
    base._write_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "current": report["current_pair_v2"],
                "selected": report["selection"],
                "statistical_gate": report["gates"]["statistical_improvement"],
                "human_90_gate": report["gates"]["human_ceiling_90_percent"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
