#!/usr/bin/env python
"""Outcome-aware search for target-design DREAM mixture regressors.

This script is a diagnostic search, not promotion evidence. It keeps the
published test and validation outcomes visible and records that fact. Any
candidate discovered here must be frozen and evaluated on a new outcome-
unopened external source before it can receive non-zero runtime weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from ogb.utils import smiles2graph
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_dream_mixture_2025 as base  # noqa: E402
from scripts import benchmark_dream_pair_ensemble_v2 as pair  # noqa: E402


SCHEMA = "dream-accuracy-outcome-aware-search/v4"
DESIGN_GAMMAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
DESIGN_FLOORS = (0.05, 0.20, 0.50)
RIDGE_ALPHAS = (3_000.0, 10_000.0, 30_000.0, 100_000.0, 300_000.0)
BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
SEED = 20_260_828
EXTRA_FILE_SHA256 = {
    "sota_training_pairs": "8a74c13893144d0a665ea3812002de9af9e0b440b2f5e56ec6ee380174622ee1",
    "sota_all_definitions": "e0aef57ef8c7bc68326fe17f47e041516382c215d9bda50c10060a5b117a8dca",
}
SOURCE_FILE_SHA256 = {**base.DREAM_FILE_SHA256, **EXTRA_FILE_SHA256}
PAIR_V2_REPORT_SHA256 = (
    "e3c8b4862733f2e07ccab1b3811df2732217194aa4a6103eeda6919c4bf5f43c"
)


def sha256(path: Path) -> str:
    return base._sha256(path)


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()


def required_files(root: Path) -> dict[str, Path]:
    return {
        "training_pairs": root / "Training_Dataset" / "TrainingData_mixturedist.csv",
        "training_definitions": root
        / "Training_Dataset"
        / "Mixure_Definitions_Training_set.csv",
        "test_pairs": root / "Test_Dataset" / "Test_set_mixturedist.csv",
        "test_definitions": root
        / "Test_Dataset"
        / "Test_set_Mixure_Definitions.csv",
        "validation_definitions": root
        / "Validation_Dataset"
        / "Mixure_Definitions_Validation_set.csv",
        "validation_raw": root
        / "Validation_Dataset"
        / "Dream_validation_TestRetest.csv",
        "public_validation_predictions": root
        / "Predictions"
        / "Validation_set_Prediction_top6_Teams.csv",
        "pom_profiles": root
        / "PostChallenge_Model"
        / "Dataset"
        / "openpom_ensemble_predictions_results.csv",
        "cid_to_smiles": root
        / "PostChallenge_Model"
        / "Dataset"
        / "cid_to_smiles.csv",
        "sota_training_pairs": root
        / "SOTA"
        / "data"
        / "raw"
        / "TrainingData_mixturedist.csv",
        "sota_all_definitions": root
        / "SOTA"
        / "data"
        / "raw"
        / "Mixure_Definitions_All.csv",
    }


def design_rows(
    pairs: pd.DataFrame,
    definitions: Mapping[Any, Sequence[int]],
    *,
    training: bool,
) -> np.ndarray:
    values = []
    for _, row in pairs.iterrows():
        if training:
            source = str(row["Dataset"]).strip()
            first_key = (source, base._label(row["Mixture 1"]))
            second_key = (source, base._label(row["Mixture 2"]))
        else:
            first_key = base._label(row["Mixture 1"])
            second_key = base._label(row["Mixture 2"])
        first = set(definitions[first_key])
        second = set(definitions[second_key])
        overlap = len(first & second)
        smaller = max(1, min(len(first), len(second)))
        values.append(
            [
                float(len(first)),
                float(len(second)),
                float(overlap),
                float(overlap / smaller),
            ]
        )
    return np.asarray(values, dtype=float)


def design_weight(design: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    size_distance = np.abs(design[:, 0] - 10.0) + np.abs(design[:, 1] - 10.0)
    overlap_penalty = 8.0 * design[:, 3]
    distance = size_distance / 10.0 + overlap_penalty
    return floor + (1.0 - floor) * np.exp(-gamma * distance)


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    sample_weight: np.ndarray,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features, sample_weight=sample_weight)
    model = Ridge(alpha=alpha)
    model.fit(standardized, target, sample_weight=sample_weight)
    return scaler, model


def predict_scaled(model: tuple[StandardScaler, Any], features: np.ndarray) -> np.ndarray:
    scaler, estimator = model
    return np.clip(np.ravel(estimator.predict(scaler.transform(features))), 0.0, 1.0)


def compact_indices(names: Sequence[str], current_width: int) -> np.ndarray:
    indices = []
    scalar_tokens = (
        "::cosine",
        "::correlation",
        "::mean_absolute_difference",
        "::root_mean_square_difference",
        "::maximum_absolute_difference",
    )
    for index, name in enumerate(names):
        if index >= current_width:
            indices.append(index)
        elif name.startswith(("mixture_size::", "component_overlap::")):
            indices.append(index)
        elif name.endswith(scalar_tokens):
            indices.append(index)
    return np.asarray(sorted(set(indices)), dtype=int)


def point_checks(
    candidate_test: Mapping[str, float | int],
    candidate_validation: Mapping[str, float | int],
    current_test: Mapping[str, float | int],
    current_validation: Mapping[str, float | int],
) -> dict[str, bool]:
    return {
        "test_pearson": candidate_test["pearson"] > current_test["pearson"],
        "test_spearman": candidate_test["spearman"] > current_test["spearman"],
        "test_rmse": candidate_test["rmse"] < current_test["rmse"],
        "test_mae": candidate_test["mae"] < current_test["mae"],
        "validation_pearson": candidate_validation["pearson"]
        > current_validation["pearson"],
        "validation_spearman": candidate_validation["spearman"]
        > current_validation["spearman"],
        "validation_rmse": candidate_validation["rmse"]
        < current_validation["rmse"],
        "validation_mae": candidate_validation["mae"]
        < current_validation["mae"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-root", type=Path, required=True)
    parser.add_argument("--pommix-root", type=Path, required=True)
    parser.add_argument("--pair-source-root", type=Path, required=True)
    parser.add_argument(
        "--pair-v2-report",
        type=Path,
        default=ROOT / "benchmarks" / "dream_pair_ensemble_retrospective_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "dream_accuracy_search_v4.json",
    )
    args = parser.parse_args()
    dream_root = args.dream_root.expanduser().resolve(strict=True)
    pommix_root = args.pommix_root.expanduser().resolve(strict=True)
    pair_root = args.pair_source_root.expanduser().resolve(strict=True)
    pair_v2_report_path = args.pair_v2_report.expanduser().resolve(strict=True)
    if sha256(pair_v2_report_path) != PAIR_V2_REPORT_SHA256:
        raise RuntimeError("frozen Pair-GNN v2 report changed")
    if git_commit(dream_root) != base.DREAM_COMMIT:
        raise RuntimeError("unsupported DREAM commit")
    required = required_files(dream_root)
    changed = [
        name
        for name, path in required.items()
        if not path.is_file()
        or (
            name in SOURCE_FILE_SHA256
            and sha256(path) != SOURCE_FILE_SHA256[name]
        )
    ]
    if changed:
        raise RuntimeError("DREAM files changed: " + ", ".join(changed))

    pom, rdkit, morgan, _scaffolds, smiles, pom_names, rdkit_names = (
        base._component_features(dream_root)
    )
    train_definitions = base._definitions(required["training_definitions"], training=True)
    test_definitions = base._definitions(required["test_definitions"], training=False)
    validation_definitions = base._definitions(
        required["validation_definitions"], training=False
    )
    all_sota_definitions = base._definitions(
        required["sota_all_definitions"], training=True
    )
    extra_sources = {"Ravia 5", "Ravia 6"}
    extra_definitions = {
        key: value
        for key, value in all_sota_definitions.items()
        if key[0] in extra_sources
    }
    train_representation = base._representations(train_definitions, pom, rdkit, morgan)
    test_representation = base._representations(test_definitions, pom, rdkit, morgan)
    validation_representation = base._representations(
        validation_definitions, pom, rdkit, morgan
    )
    extra_representation = base._representations(
        extra_definitions, pom, rdkit, morgan
    )
    pommix, pommix_audit = base._pommix_embeddings(pommix_root, smiles)
    train_pommix = base._embedding_aggregates(train_definitions, pommix)
    test_pommix = base._embedding_aggregates(test_definitions, pommix)
    validation_pommix = base._embedding_aggregates(validation_definitions, pommix)
    extra_pommix = base._embedding_aggregates(extra_definitions, pommix)
    current_names = [
        *base._mixture_feature_names(pom_names, rdkit_names),
        *base._embedding_feature_names(pommix_audit["embedding_dimensions"]),
    ]
    train_pairs = pd.read_csv(required["training_pairs"])
    current_train, train_y, train_groups = base._training_rows(
        required["training_pairs"], train_representation
    )
    current_train = np.column_stack(
        [current_train, base._embedding_rows(train_pairs, train_pommix, training=True)]
    )
    test_pairs = pd.read_csv(required["test_pairs"])
    current_test, test_y = base._evaluation_rows(
        test_pairs, test_representation, "Experimental values"
    )
    current_test = np.column_stack(
        [current_test, base._embedding_rows(test_pairs, test_pommix, training=False)]
    )
    validation_pairs = pd.read_csv(required["public_validation_predictions"])
    current_validation, validation_y = base._evaluation_rows(
        validation_pairs, validation_representation, "ExpMean_combined"
    )
    current_validation = np.column_stack(
        [
            current_validation,
            base._embedding_rows(
                validation_pairs, validation_pommix, training=False
            ),
        ]
    )
    sota_pairs = pd.read_csv(required["sota_training_pairs"])
    extra_pairs = sota_pairs[sota_pairs["Dataset"].isin(extra_sources)].copy()
    extra_current = np.asarray(
        [
            base.pair_features(
                extra_representation[
                    (str(row["Dataset"]).strip(), base._label(row["Mixture 1"]))
                ],
                extra_representation[
                    (str(row["Dataset"]).strip(), base._label(row["Mixture 2"]))
                ],
            )
            for _, row in extra_pairs.iterrows()
        ]
    )
    extra_current = np.column_stack(
        [
            extra_current,
            base._embedding_rows(extra_pairs, extra_pommix, training=True),
        ]
    )
    extra_y = extra_pairs["Experimental Values"].to_numpy(float)

    model_root = dream_root / "SOTA" / "3-Pair_Model" / "finetuned_model"
    pair_model, pair_data, pairdata, _config = pair._load_pair_model(
        pair_root, model_root / "config.json", model_root / "model.pt"
    )
    pair_smiles_path = dream_root / "SOTA" / "data" / "raw" / "cid_to_smiles.json"
    if sha256(pair_smiles_path) != pair.PAIR_SMILES_SHA256:
        raise RuntimeError("odor-pair SMILES source changed")
    exact_smiles = {
        int(cid): text
        for cid, text in json.loads(
            pair_smiles_path.read_text(encoding="utf-8")
        ).items()
    }
    graphs = {
        cid: pairdata.to_torch(smiles2graph(exact_smiles[cid])) for cid in sorted(smiles)
    }
    train_embeddings = pair._generated_embeddings(
        train_definitions, pair_model, pair_data, pairdata, graphs
    )
    test_embeddings = pair._generated_embeddings(
        test_definitions, pair_model, pair_data, pairdata, graphs
    )
    validation_embeddings = pair._generated_embeddings(
        validation_definitions, pair_model, pair_data, pairdata, graphs
    )
    extra_embeddings = pair._generated_embeddings(
        extra_definitions, pair_model, pair_data, pairdata, graphs
    )
    generated = np.vstack(
        [
            *[train_embeddings[key] for key in train_definitions],
            *[test_embeddings[key] for key in test_definitions],
            *[validation_embeddings[key] for key in validation_definitions],
        ]
    ).astype(np.float32)
    generated_hash = hashlib.sha256(generated.tobytes()).hexdigest()
    if generated_hash != pair.PAIR_GENERATED_EMBEDDING_SHA256:
        raise RuntimeError("odor-pair embeddings changed")
    pair_train = pair._pair_rows(train_pairs, train_embeddings, training=True)
    pair_test = pair._pair_rows(test_pairs, test_embeddings, training=False)
    pair_validation = pair._pair_rows(
        validation_pairs, validation_embeddings, training=False
    )
    pair_extra = pair._pair_rows(extra_pairs, extra_embeddings, training=True)
    feature_names = [*current_names, *pair._pair_feature_names()]
    train_x = np.column_stack([current_train, pair_train])
    test_x = np.column_stack([current_test, pair_test])
    validation_x = np.column_stack([current_validation, pair_validation])
    extra_x = np.column_stack([extra_current, pair_extra])
    train_design = design_rows(train_pairs, train_definitions, training=True)
    test_design = design_rows(test_pairs, test_definitions, training=False)
    validation_design = design_rows(
        validation_pairs, validation_definitions, training=False
    )
    extra_design = design_rows(extra_pairs, extra_definitions, training=True)
    test_exact10 = (
        (test_design[:, 0] == 10)
        & (test_design[:, 1] == 10)
        & (test_design[:, 2] == 0)
    )
    validation_exact10 = (
        (validation_design[:, 0] == 10)
        & (validation_design[:, 1] == 10)
        & (validation_design[:, 2] == 0)
    )
    extra_exact10 = (
        (extra_design[:, 0] == 10)
        & (extra_design[:, 1] == 10)
        & (extra_design[:, 2] == 0)
    )

    member_models = [
        pair._fit_member(train_x, train_y, alpha) for alpha, _weight in pair.MEMBER_SPECS
    ]
    fixed_weights = np.asarray([weight for _alpha, weight in pair.MEMBER_SPECS])
    baseline_test_prediction = np.clip(
        fixed_weights @ np.asarray([pair._predict(model, test_x) for model in member_models]),
        0.0,
        1.0,
    )
    baseline_validation_prediction = np.clip(
        fixed_weights
        @ np.asarray([pair._predict(model, validation_x) for model in member_models]),
        0.0,
        1.0,
    )
    baseline_extra_prediction = np.clip(
        fixed_weights
        @ np.asarray([pair._predict(model, extra_x) for model in member_models]),
        0.0,
        1.0,
    )
    baseline_test = base.metrics(baseline_test_prediction, test_y)
    baseline_validation = base.metrics(baseline_validation_prediction, validation_y)
    baseline_extra = base.metrics(baseline_extra_prediction, extra_y)
    frozen_pair_v2 = json.loads(pair_v2_report_path.read_text(encoding="utf-8"))
    for split, computed in (
        ("test", baseline_test),
        ("validation", baseline_validation),
    ):
        expected = frozen_pair_v2[split]["candidate"]
        if any(
            abs(float(computed[metric]) - float(expected[metric])) > 1e-12
            for metric in ("pearson", "spearman", "rmse", "mae", "bias")
        ):
            raise RuntimeError(f"Pair-GNN v2 {split} baseline changed")

    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    candidates: list[dict[str, Any]] = []

    def record(
        name: str,
        test_prediction: np.ndarray,
        validation_prediction: np.ndarray,
        extra_prediction: np.ndarray,
        config: Mapping[str, Any],
        *,
        add_exact10_router: bool = True,
    ) -> None:
        test_result = base.metrics(test_prediction, test_y)
        validation_result = base.metrics(validation_prediction, validation_y)
        checks = point_checks(
            test_result, validation_result, baseline_test, baseline_validation
        )
        predictions[name] = (
            test_prediction,
            validation_prediction,
            extra_prediction,
        )
        candidates.append(
            {
                "name": name,
                "config": dict(config),
                "test": test_result,
                "validation": validation_result,
                "extra_ravia_5_6": base.metrics(extra_prediction, extra_y),
                "point_checks": checks,
                "point_pareto": all(checks.values()),
            }
        )
        if add_exact10_router:
            record(
                f"{name}_exact10_router",
                np.where(
                    test_exact10,
                    test_prediction,
                    baseline_test_prediction,
                ),
                np.where(
                    validation_exact10,
                    validation_prediction,
                    baseline_validation_prediction,
                ),
                np.where(
                    extra_exact10,
                    extra_prediction,
                    baseline_extra_prediction,
                ),
                {
                    **dict(config),
                    "protocol_router": (
                        "candidate_for_exact_10x10_zero_overlap_else_pair_v2"
                    ),
                },
                add_exact10_router=False,
            )

    for gamma in DESIGN_GAMMAS:
        for floor in DESIGN_FLOORS:
            weights = design_weight(train_design, gamma, floor)
            for alpha in RIDGE_ALPHAS:
                model = fit_weighted_ridge(
                    train_x,
                    train_y,
                    alpha=alpha,
                    sample_weight=weights,
                )
                test_prediction = predict_scaled(model, test_x)
                validation_prediction = predict_scaled(model, validation_x)
                extra_prediction = predict_scaled(model, extra_x)
                direct_name = f"weighted_ridge_g{gamma:g}_f{floor:g}_a{alpha:g}"
                record(
                    direct_name,
                    test_prediction,
                    validation_prediction,
                    extra_prediction,
                    {"family": "weighted_ridge", "gamma": gamma, "floor": floor, "alpha": alpha},
                )
                for blend in BLEND_WEIGHTS[1:-1]:
                    record(
                        f"{direct_name}_blend{blend:.2f}",
                        np.clip(
                            (1.0 - blend) * baseline_test_prediction
                            + blend * test_prediction,
                            0.0,
                            1.0,
                        ),
                        np.clip(
                            (1.0 - blend) * baseline_validation_prediction
                            + blend * validation_prediction,
                            0.0,
                            1.0,
                        ),
                        np.clip(
                            (1.0 - blend) * baseline_extra_prediction
                            + blend * extra_prediction,
                            0.0,
                            1.0,
                        ),
                        {
                            "family": "weighted_ridge_blend",
                            "gamma": gamma,
                            "floor": floor,
                            "alpha": alpha,
                            "candidate_weight": blend,
                        },
                    )

    compact = compact_indices(feature_names, len(current_names))
    compact_train = train_x[:, compact]
    compact_test = test_x[:, compact]
    compact_validation = validation_x[:, compact]
    compact_extra = extra_x[:, compact]
    nonlinear_models: list[tuple[str, Any, dict[str, Any]]] = []
    for leaf in (3, 7, 15):
        nonlinear_models.append(
            (
                f"extra_trees_leaf{leaf}",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=leaf,
                    max_features=0.7,
                    n_jobs=-1,
                    random_state=SEED,
                ),
                {"family": "extra_trees", "min_samples_leaf": leaf},
            )
        )
    nonlinear_models.append(
        (
            "random_forest_leaf7",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=7,
                max_features=0.7,
                n_jobs=-1,
                random_state=SEED,
            ),
            {"family": "random_forest", "min_samples_leaf": 7},
        )
    )
    for name, model, config in nonlinear_models:
        model.fit(compact_train, train_y)
        candidate_test = np.clip(np.ravel(model.predict(compact_test)), 0.0, 1.0)
        candidate_validation = np.clip(
            np.ravel(model.predict(compact_validation)), 0.0, 1.0
        )
        candidate_extra = np.clip(
            np.ravel(model.predict(compact_extra)), 0.0, 1.0
        )
        record(
            name,
            candidate_test,
            candidate_validation,
            candidate_extra,
            config,
        )
        for blend in BLEND_WEIGHTS[1:-1]:
            record(
                f"{name}_blend{blend:.2f}",
                np.clip(
                    (1.0 - blend) * baseline_test_prediction
                    + blend * candidate_test,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (1.0 - blend) * baseline_validation_prediction
                    + blend * candidate_validation,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (1.0 - blend) * baseline_extra_prediction
                    + blend * candidate_extra,
                    0.0,
                    1.0,
                ),
                {**config, "family": f"{config['family']}_blend", "candidate_weight": blend},
            )

    pareto = [row for row in candidates if row["point_pareto"]]
    selected_pool = pareto or candidates
    selected = max(
        selected_pool,
        key=lambda row: (
            float(row["test"]["pearson"])
            + float(row["validation"]["pearson"]),
            -float(row["test"]["rmse"])
            - float(row["validation"]["rmse"]),
        ),
    )
    source_oof_baseline = np.zeros(len(train_y), dtype=float)
    source_oof_router = np.zeros(len(train_y), dtype=float)
    source_oof_folds = []
    for fold, (training, held_out) in enumerate(
        GroupKFold(n_splits=len(set(train_groups))).split(
            train_x, train_y, train_groups
        )
    ):
        fold_pair_models = [
            pair._fit_member(train_x[training], train_y[training], alpha)
            for alpha, _weight in pair.MEMBER_SPECS
        ]
        fold_baseline = np.clip(
            fixed_weights
            @ np.asarray(
                [
                    pair._predict(model, train_x[held_out])
                    for model in fold_pair_models
                ]
            ),
            0.0,
            1.0,
        )
        fold_tree = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=15,
            max_features=0.7,
            n_jobs=-1,
            random_state=SEED,
        )
        fold_tree.fit(compact_train[training], train_y[training])
        fold_tree_prediction = np.clip(
            fold_tree.predict(compact_train[held_out]), 0.0, 1.0
        )
        fold_blend = 0.65 * fold_baseline + 0.35 * fold_tree_prediction
        fold_exact10 = (
            (train_design[held_out, 0] == 10)
            & (train_design[held_out, 1] == 10)
            & (train_design[held_out, 2] == 0)
        )
        fold_router = np.where(fold_exact10, fold_blend, fold_baseline)
        source_oof_baseline[held_out] = fold_baseline
        source_oof_router[held_out] = fold_router
        source_oof_folds.append(
            {
                "fold": fold,
                "held_out_sources": sorted(set(train_groups[held_out].tolist())),
                "held_out_rows": int(len(held_out)),
                "held_out_exact10_rows": int(np.sum(fold_exact10)),
                "baseline": base.metrics(fold_baseline, train_y[held_out]),
                "router": base.metrics(fold_router, train_y[held_out]),
            }
        )
    exact10_index = np.flatnonzero(
        (train_design[:, 0] == 10)
        & (train_design[:, 1] == 10)
        & (train_design[:, 2] == 0)
    )
    source_oof = {
        "folds": source_oof_folds,
        "pooled_baseline": base.metrics(source_oof_baseline, train_y),
        "pooled_router": base.metrics(source_oof_router, train_y),
        "exact10_baseline": base.metrics(
            source_oof_baseline[exact10_index], train_y[exact10_index]
        ),
        "exact10_router": base.metrics(
            source_oof_router[exact10_index], train_y[exact10_index]
        ),
        "selection_used_test_and_validation_outcomes": True,
    }
    selected_test_prediction, selected_validation_prediction, _selected_extra = (
        predictions[str(selected["name"])]
    )
    test_bootstrap = base._paired_bootstrap(
        test_y, selected_test_prediction, baseline_test_prediction
    )
    validation_bootstrap = base._validation_two_way_bootstrap(
        pd.read_csv(required["validation_raw"]),
        validation_pairs,
        selected_validation_prediction,
        baseline_validation_prediction,
    )
    test_retest = base._correlation(
        pearsonr,
        validation_pairs["ExpMean_test"].to_numpy(float),
        validation_pairs["ExpMean_retest"].to_numpy(float),
    )
    reliability = 2.0 * test_retest / (1.0 + test_retest)
    human_ceiling = math.sqrt(max(0.0, reliability))
    normalized = float(selected["validation"]["pearson"]) / human_ceiling
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
        "status": "outcome_aware_search_only",
        "source": {
            "dream_commit": base.DREAM_COMMIT,
            "pair_commit": pair.PAIR_SOURCE_COMMIT,
            "pair_weights_sha256": pair.PAIR_WEIGHTS_SHA256,
            "generated_embedding_rows_sha256": generated_hash,
            "pair_v2_report_sha256": sha256(pair_v2_report_path),
            "extra_files": {
                name: {
                    "sha256": sha256(required[name]),
                    "bytes": required[name].stat().st_size,
                }
                for name in sorted(EXTRA_FILE_SHA256)
            },
        },
        "timing": {
            "test_outcomes_visible": True,
            "validation_outcomes_visible": True,
            "eligible_for_model_selection_or_promotion": False,
        },
        "implementation": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "full_features": len(feature_names),
            "compact_features": len(compact),
            "candidate_count": len(candidates),
        },
        "target_design": {
            "test": "10_vs_10_components_zero_overlap",
            "validation": "9_to_11_vs_9_to_11_components_zero_overlap",
            "training_exact_10_vs_10_zero_overlap": int(
                np.sum(
                    (train_design[:, 0] == 10)
                    & (train_design[:, 1] == 10)
                    & (train_design[:, 2] == 0)
                )
            ),
            "test_exact_10_vs_10_zero_overlap": int(np.sum(test_exact10)),
            "validation_exact_10_vs_10_zero_overlap": int(
                np.sum(validation_exact10)
            ),
            "extra_ravia_5_6_pairs": int(len(extra_y)),
            "extra_ravia_5_6_exact_10_vs_10_zero_overlap": int(
                np.sum(extra_exact10)
            ),
        },
        "baseline": {
            "test": baseline_test,
            "validation": baseline_validation,
            "extra_ravia_5_6": baseline_extra,
        },
        "selected_diagnostic": selected,
        "selected_post_selection_inference": {
            "test_paired_bootstrap": test_bootstrap,
            "validation_subject_pair_two_way_bootstrap": validation_bootstrap,
            "human_ceiling": human_ceiling,
            "human_ceiling_normalized_pearson": normalized,
            "human_ceiling_normalized_pearson_95_interval": normalized_interval,
            "post_selection_descriptive_only": True,
        },
        "source_group_oof_diagnostic": source_oof,
        "point_pareto_candidates": len(pareto),
        "candidates": candidates,
        "gates": {
            "point_pareto": {"passed": bool(pareto)},
            "statistical_improvement": {
                "checks": statistical_checks,
                "passed": all(statistical_checks.values()),
            },
            "human_ceiling_90_percent": {
                "threshold": 0.90,
                "passed": normalized_interval[0] >= 0.90,
            },
            "production": {"passed": False, "runtime_primary_score_weight": 0.0},
        },
        "claim_boundary": {
            "outcome_aware_hyperparameter_search": True,
            "human_olfactory_90_percent_certified": False,
            "natural_language_recipe_accuracy_measured": False,
        },
    }
    base._write_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "baseline": report["baseline"],
                "selected": selected,
                "point_pareto_candidates": len(pareto),
                "candidate_count": len(candidates),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
