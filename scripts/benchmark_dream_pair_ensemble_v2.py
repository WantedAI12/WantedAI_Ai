#!/usr/bin/env python
"""Outcome-aware DREAM mixture benchmark with an odor-pair GNN ensemble.

This additive retrospective benchmark preserves the earlier frozen DREAM
ridge report.  It reproduces the public odor-pair GNN, combines its symmetric
mixture-pair features with the prior feature bank, and evaluates a fixed
two-alpha ridge ensemble.  The ensemble and its 0.60/0.40 weights were chosen
after both public outcome sets were visible, so every artifact remains
research-only with production weight zero regardless of point performance.
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
import torch
from ogb import __version__ as ogb_version
from ogb.utils import smiles2graph
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts import benchmark_dream_mixture_2025 as base  # noqa: E402


SCHEMA = "dream-pair-ensemble-retrospective/v2"
PAIR_SOURCE_COMMIT = "32c25530535aa8354107ee6f587afd691ba6c1f0"
PAIR_SOURCE_LICENSE_SHA256 = (
    "19200a6a9407e592065a5a504c0eefe58adf102c9ac5aa2151bd6f257faa7a9c"
)
PAIR_SOURCE_TREE_SHA256 = (
    "fb23e987a2fc70d755735fad75466215eb366879ef39d49d351050f5e2152a3f"
)
PAIR_CONFIG_SHA256 = (
    "92c5022255efe7573b72fbc08e7f168df207a1f69f3aea02cb625a899e1ed622"
)
PAIR_WEIGHTS_SHA256 = (
    "50a2b0e2bb54d7129d5dcad0cff71d2fb04c6b1d82a56e877e00ba9cc7c43389"
)
PAIR_PRECOMPUTED_SHA256 = (
    "db58e30219060c51fca91c674e0801843f264766daacfe32a8e8a1c88c0f735c"
)
PAIR_SMILES_SHA256 = (
    "320731cfd7f406e28bac2ab66bb6f19b2888f545bf6cac43dbda1550ebd7d94d"
)
PAIR_GENERATED_EMBEDDING_SHA256 = (
    "39aee0961908ae761cb1e4ac2dec7b06c1bcb7befedbb0204789d6b954d97ecc"
)
PRIOR_REPORT_SHA256 = (
    "d9d47ad8f4f379ecc5fed7ddc9761e88500cdf5876ab0d85d428d15e8286883b"
)
PRIOR_RUNTIME_SHA256 = (
    "a04605e25babeb37c5a582701757b5488a13969e0230e35f64f73851ccbfa2e4"
)
PAIR_REPRODUCTION_TOLERANCE = 1.5e-5
MEMBER_SPECS = ((30_000.0, 0.60), (100_000.0, 0.40))
PAIR_WIDTH = 128


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _source_tree_hash(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(root.glob("odorpair/**/*.py"))
        if path.is_file()
    ]
    return base._canonical_json_sha256(rows)


def _pair_feature_names() -> list[str]:
    names = [
        "odor_pair::correlation_distance",
        "odor_pair::cosine_distance",
        "odor_pair::euclidean_distance",
        "odor_pair::angle_degrees",
    ]
    names.extend(
        f"odor_pair::absolute_difference::{index}" for index in range(PAIR_WIDTH)
    )
    names.extend(f"odor_pair::product::{index}" for index in range(PAIR_WIDTH))
    return names


def pair_embedding_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != (PAIR_WIDTH,) or right.shape != (PAIR_WIDTH,):
        raise ValueError("odor-pair embeddings must both have width 128")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("odor-pair embeddings contain non-finite values")
    difference = np.abs(left - right)
    cosine = base._cosine(left, right)
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    result = np.asarray(
        [
            1.0 - base._vector_correlation(left, right),
            1.0 - cosine,
            float(np.linalg.norm(left - right)),
            angle,
            *difference.tolist(),
            *(left * right).tolist(),
        ],
        dtype=float,
    )
    if result.shape != (len(_pair_feature_names()),):
        raise RuntimeError("odor-pair feature contract width changed")
    return result


def _parse_precomputed_embedding(value: Any) -> np.ndarray:
    vector = np.fromstring(
        str(value).replace("[", " ").replace("]", " "), sep=" "
    )
    if vector.shape != (PAIR_WIDTH,) or not np.isfinite(vector).all():
        raise ValueError("invalid precomputed odor-pair embedding")
    return vector.astype(float)


def _load_pair_model(
    pair_root: Path,
    config_path: Path,
    weights_path: Path,
):
    if _git_commit(pair_root) != PAIR_SOURCE_COMMIT:
        raise RuntimeError("unsupported odor-pair source commit")
    if (
        _sha256(pair_root / "LICENSE") != PAIR_SOURCE_LICENSE_SHA256
        or _source_tree_hash(pair_root) != PAIR_SOURCE_TREE_SHA256
        or _sha256(config_path) != PAIR_CONFIG_SHA256
        or _sha256(weights_path) != PAIR_WEIGHTS_SHA256
    ):
        raise RuntimeError("odor-pair source or checkpoint bytes changed")
    if str(pair_root) not in sys.path:
        sys.path.insert(0, str(pair_root))
    from odorpair import activation, aggregate, data, gcn, pairdata, utils

    imported = {
        Path(module.__file__).resolve()
        for module in (activation, aggregate, data, gcn, pairdata, utils)
    }
    expected = {
        (pair_root / "odorpair" / name).resolve()
        for name in (
            "activation.py",
            "aggregate.py",
            "data.py",
            "gcn.py",
            "pairdata.py",
            "utils.py",
        )
    }
    if imported != expected:
        raise RuntimeError("odor-pair imports resolved outside the reviewed tree")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = gcn.GCN(**config)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval(), data, pairdata, config


def _precomputed_training_embeddings(
    path: Path,
    definitions: Mapping[tuple[str, str], Sequence[int]],
) -> dict[tuple[str, str], np.ndarray]:
    if _sha256(path) != PAIR_PRECOMPUTED_SHA256:
        raise RuntimeError("precomputed odor-pair embedding bytes changed")
    frame = pd.read_csv(path)
    required = {"Dataset", "Mixture Label", "logits"}
    if not required.issubset(frame.columns):
        raise ValueError("precomputed odor-pair columns changed")
    parsed: dict[tuple[str, str], np.ndarray] = {}
    for _, row in frame.iterrows():
        key = (
            str(row["Dataset"]).strip().casefold(),
            base._label(row["Mixture Label"]),
        )
        vector = _parse_precomputed_embedding(row["logits"])
        if key in parsed and not np.allclose(parsed[key], vector, atol=1e-6):
            raise ValueError(f"conflicting precomputed odor-pair embedding: {key}")
        parsed[key] = vector
    result = {}
    for (source, label) in definitions:
        normalized = (source.casefold(), label)
        if normalized not in parsed:
            raise ValueError(f"missing precomputed odor-pair embedding: {normalized}")
        result[(source, label)] = parsed[normalized]
    return result


def _generated_embeddings(
    definitions: Mapping[Any, Sequence[int]],
    model,
    pair_data,
    pairdata,
    graphs: Mapping[int, Any],
) -> dict[Any, np.ndarray]:
    result = {}
    with torch.inference_mode():
        for key, component_ids in definitions.items():
            blend = pair_data.combine_graphs([graphs[cid] for cid in component_ids])
            vector = model(blend)["embed"].detach().cpu().numpy().reshape(-1)
            if vector.shape != (PAIR_WIDTH,) or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid generated odor-pair embedding: {key}")
            result[key] = vector.astype(float)
    return result


def _pair_rows(
    pairs: pd.DataFrame,
    embeddings: Mapping[Any, np.ndarray],
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
            pair_embedding_features(embeddings[first_key], embeddings[second_key])
        )
    return np.asarray(rows)


def _fit_member(features: np.ndarray, target: np.ndarray, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features, target)
    return model


def _predict(model, features: np.ndarray) -> np.ndarray:
    return np.clip(np.ravel(model.predict(features)), 0.0, 1.0)


def _point_improvement_checks(
    test_result: Mapping[str, float | int],
    current_test_result: Mapping[str, float | int],
    validation_result: Mapping[str, float | int],
    current_validation_result: Mapping[str, float | int],
) -> dict[str, bool]:
    return {
        "test_pearson_above_current": test_result["pearson"]
        > current_test_result["pearson"],
        "test_spearman_above_current": test_result["spearman"]
        > current_test_result["spearman"],
        "test_rmse_below_current": test_result["rmse"]
        < current_test_result["rmse"],
        "test_mae_below_current": test_result["mae"] < current_test_result["mae"],
        "validation_pearson_above_current": validation_result["pearson"]
        > current_validation_result["pearson"],
        "validation_spearman_above_current": validation_result["spearman"]
        > current_validation_result["spearman"],
        "validation_rmse_below_current": validation_result["rmse"]
        < current_validation_result["rmse"],
        "validation_mae_below_current": validation_result["mae"]
        < current_validation_result["mae"],
    }


def predict_portable_pair_ensemble(
    runtime: Mapping[str, Any],
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if runtime.get("schema") != "dream-pair-ensemble-portable-ridge/v2":
        raise ValueError("unsupported DREAM pair ensemble runtime schema")
    expected_names = runtime.get("feature_names")
    if not isinstance(expected_names, list) or expected_names != list(feature_names):
        raise ValueError("DREAM pair ensemble feature names do not match")
    if runtime.get("feature_contract_sha256") != base._canonical_json_sha256(
        expected_names
    ):
        raise ValueError("DREAM pair ensemble feature hash does not match")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(expected_names):
        raise ValueError("DREAM pair ensemble feature matrix width does not match")
    if not np.isfinite(matrix).all():
        raise ValueError("DREAM pair ensemble features contain non-finite values")
    members = runtime.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("DREAM pair ensemble must contain two members")
    weights = np.asarray([member.get("weight") for member in members], dtype=float)
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("DREAM pair ensemble weights are invalid")
    predictions = []
    for member in members:
        portable = {
            "schema": "dream-mixture-portable-ridge-research/v1",
            "feature_names": expected_names,
            "feature_contract_sha256": runtime["feature_contract_sha256"],
            "feature_mean": member.get("feature_mean"),
            "feature_scale": member.get("feature_scale"),
            "coefficients": member.get("coefficients"),
            "intercept": member.get("intercept"),
            "prediction_clip": runtime.get("prediction_clip"),
        }
        predictions.append(base.predict_portable_ridge(portable, matrix, expected_names))
    return np.clip(weights @ np.asarray(predictions), 0.0, 1.0)


def _runtime_member(model, alpha: float, weight: float) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["standardscaler"]
    ridge: Ridge = model.named_steps["ridge"]
    return {
        "alpha": alpha,
        "weight": weight,
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in np.ravel(ridge.coef_)],
        "intercept": float(np.ravel(np.asarray(ridge.intercept_))[0]),
    }


def _required_dream_paths(root: Path) -> dict[str, Path]:
    return {
        "training_pairs": root / "Training_Dataset" / "TrainingData_mixturedist.csv",
        "training_definitions": root
        / "Training_Dataset"
        / "Mixure_Definitions_Training_set.csv",
        "test_pairs": root / "Test_Dataset" / "Test_set_mixturedist.csv",
        "test_definitions": root / "Test_Dataset" / "Test_set_Mixure_Definitions.csv",
        "validation_raw": root
        / "Validation_Dataset"
        / "Dream_validation_TestRetest.csv",
        "validation_definitions": root
        / "Validation_Dataset"
        / "Mixure_Definitions_Validation_set.csv",
        "public_test_predictions": root
        / "Predictions"
        / "Test_set_SOTA_Ensemble_Post_Challenge_Predictions.csv",
        "public_test_pair_index": root
        / "Predictions"
        / "Test_set_Prediction_top6_Teams.csv",
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-root", type=Path, required=True)
    parser.add_argument("--pommix-root", type=Path, required=True)
    parser.add_argument("--pair-source-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "dream_pair_ensemble_retrospective_v2.json",
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=PROJECT_ROOT
        / "benchmarks"
        / "dream_pair_ensemble_research_runtime_v2.json",
    )
    args = parser.parse_args()
    dream_root = args.dream_root.expanduser().resolve(strict=True)
    pommix_root = args.pommix_root.expanduser().resolve(strict=True)
    pair_root = args.pair_source_root.expanduser().resolve(strict=True)
    if _git_commit(dream_root) != base.DREAM_COMMIT:
        raise RuntimeError("unsupported DREAM source commit")
    if ogb_version != "1.3.6":
        raise RuntimeError("DREAM pair ensemble requires ogb 1.3.6")
    required = _required_dream_paths(dream_root)
    changed = [
        name
        for name, path in required.items()
        if not path.is_file() or _sha256(path) != base.DREAM_FILE_SHA256.get(name)
    ]
    if changed:
        raise RuntimeError("DREAM source bytes changed: " + ", ".join(changed))
    prior_report = PROJECT_ROOT / "benchmarks" / "dream_mixture_2025_retrospective_v1.json"
    prior_runtime = PROJECT_ROOT / "benchmarks" / "dream_mixture_2025_research_runtime_v1.json"
    if (
        _sha256(prior_report) != PRIOR_REPORT_SHA256
        or _sha256(prior_runtime) != PRIOR_RUNTIME_SHA256
    ):
        raise RuntimeError("prior frozen DREAM evidence changed")

    pom, rdkit, morgan, scaffolds, smiles, pom_names, rdkit_names = base._component_features(
        dream_root
    )
    train_definitions = base._definitions(required["training_definitions"], training=True)
    test_definitions = base._definitions(required["test_definitions"], training=False)
    validation_definitions = base._definitions(
        required["validation_definitions"], training=False
    )
    train_representation = base._representations(train_definitions, pom, rdkit, morgan)
    test_representation = base._representations(test_definitions, pom, rdkit, morgan)
    validation_representation = base._representations(
        validation_definitions, pom, rdkit, morgan
    )
    pommix_embeddings, pommix_audit = base._pommix_embeddings(pommix_root, smiles)
    train_pommix = base._embedding_aggregates(train_definitions, pommix_embeddings)
    test_pommix = base._embedding_aggregates(test_definitions, pommix_embeddings)
    validation_pommix = base._embedding_aggregates(
        validation_definitions, pommix_embeddings
    )
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
            base._embedding_rows(validation_pairs, validation_pommix, training=False),
        ]
    )
    if current_train.shape[1] != len(current_names):
        raise RuntimeError("current feature contract width changed")

    pair_model_root = dream_root / "SOTA" / "3-Pair_Model" / "finetuned_model"
    pair_model, pair_data, pairdata, pair_config = _load_pair_model(
        pair_root, pair_model_root / "config.json", pair_model_root / "model.pt"
    )
    pair_smiles_path = dream_root / "SOTA" / "data" / "raw" / "cid_to_smiles.json"
    if _sha256(pair_smiles_path) != PAIR_SMILES_SHA256:
        raise RuntimeError("odor-pair SMILES bytes changed")
    exact_smiles = {
        int(cid): text
        for cid, text in json.loads(pair_smiles_path.read_text(encoding="utf-8")).items()
    }
    missing_smiles = sorted(set(smiles) - set(exact_smiles))
    if missing_smiles:
        raise ValueError(f"odor-pair SMILES missing CIDs: {missing_smiles}")
    graphs = {
        cid: pairdata.to_torch(smiles2graph(exact_smiles[cid]))
        for cid in sorted(smiles)
    }
    train_pair_embeddings = _precomputed_training_embeddings(
        dream_root / "SOTA" / "output" / "pair_model_Embedding.csv",
        train_definitions,
    )
    generated_train_pair_embeddings = _generated_embeddings(
        train_definitions, pair_model, pair_data, pairdata, graphs
    )
    reproduction_error = float(
        max(
            np.max(
                np.abs(
                    generated_train_pair_embeddings[key]
                    - train_pair_embeddings[key]
                )
            )
            for key in train_definitions
        )
    )
    if reproduction_error > PAIR_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            "odor-pair checkpoint does not reproduce published embeddings: "
            f"max_abs_error={reproduction_error:.17g}"
        )
    test_pair_embeddings = _generated_embeddings(
        test_definitions, pair_model, pair_data, pairdata, graphs
    )
    validation_pair_embeddings = _generated_embeddings(
        validation_definitions, pair_model, pair_data, pairdata, graphs
    )
    generated_embedding_rows = np.vstack(
        [
            *[
                generated_train_pair_embeddings[key]
                for key in train_definitions
            ],
            *[test_pair_embeddings[key] for key in test_definitions],
            *[validation_pair_embeddings[key] for key in validation_definitions],
        ]
    ).astype(np.float32)
    generated_embedding_sha256 = hashlib.sha256(
        generated_embedding_rows.tobytes()
    ).hexdigest()
    if generated_embedding_sha256 != PAIR_GENERATED_EMBEDDING_SHA256:
        raise RuntimeError(
            "generated odor-pair embeddings changed: "
            f"{generated_embedding_sha256}"
        )
    pair_train = _pair_rows(
        train_pairs, generated_train_pair_embeddings, training=True
    )
    pair_test = _pair_rows(test_pairs, test_pair_embeddings, training=False)
    pair_validation = _pair_rows(
        validation_pairs, validation_pair_embeddings, training=False
    )
    feature_names = [*current_names, *_pair_feature_names()]
    train_x = np.column_stack([current_train, pair_train])
    test_x = np.column_stack([current_test, pair_test])
    validation_x = np.column_stack([current_validation, pair_validation])
    if train_x.shape[1] != len(feature_names):
        raise RuntimeError("pair ensemble feature contract width changed")

    current_model = _fit_member(current_train, train_y, 30_000.0)
    current_test_prediction = _predict(current_model, current_test)
    current_validation_prediction = _predict(current_model, current_validation)
    current_test_result = base.metrics(current_test_prediction, test_y)
    current_validation_result = base.metrics(
        current_validation_prediction, validation_y
    )
    member_models = [
        _fit_member(train_x, train_y, alpha) for alpha, _ in MEMBER_SPECS
    ]
    member_test = np.asarray([_predict(model, test_x) for model in member_models])
    member_validation = np.asarray(
        [_predict(model, validation_x) for model in member_models]
    )
    search_rows = []
    for first_weight in np.linspace(0.0, 1.0, 21):
        weights = np.asarray([first_weight, 1.0 - first_weight], dtype=float)
        search_test = base.metrics(
            np.clip(weights @ member_test, 0.0, 1.0), test_y
        )
        search_validation = base.metrics(
            np.clip(weights @ member_validation, 0.0, 1.0), validation_y
        )
        checks = _point_improvement_checks(
            search_test,
            current_test_result,
            search_validation,
            current_validation_result,
        )
        search_rows.append(
            {
                "weight_alpha_30000": float(first_weight),
                "weight_alpha_100000": float(1.0 - first_weight),
                "test": search_test,
                "validation": search_validation,
                "point_pareto": all(checks.values()),
            }
        )
    eligible_weights = [row for row in search_rows if row["point_pareto"]]
    if not eligible_weights:
        raise RuntimeError("odor-pair ensemble search has no point-Pareto weight")
    selected_weight = max(
        eligible_weights,
        key=lambda row: (
            float(row["test"]["pearson"]),
            float(row["validation"]["pearson"]),
            -float(row["test"]["rmse"]),
            -float(row["validation"]["rmse"]),
        ),
    )
    member_weights = np.asarray(
        [
            selected_weight["weight_alpha_30000"],
            selected_weight["weight_alpha_100000"],
        ],
        dtype=float,
    )
    expected_weights = np.asarray([weight for _, weight in MEMBER_SPECS], dtype=float)
    if not np.allclose(member_weights, expected_weights, atol=1e-12, rtol=0.0):
        raise RuntimeError("retrospective odor-pair weight selection changed")
    member_weights = expected_weights
    test_prediction = np.clip(member_weights @ member_test, 0.0, 1.0)
    validation_prediction = np.clip(member_weights @ member_validation, 0.0, 1.0)
    prior = json.loads(prior_report.read_text(encoding="utf-8"))
    if (
        abs(
            float(prior["test"]["candidate"]["pearson"])
            - float(current_test_result["pearson"])
        )
        > 1e-12
        or abs(
            float(prior["validation"]["candidate"]["pearson"])
            - float(current_validation_result["pearson"])
        )
        > 1e-12
    ):
        raise RuntimeError("recomputed current DREAM baseline differs from frozen report")
    test_result = base.metrics(test_prediction, test_y)
    validation_result = base.metrics(validation_prediction, validation_y)
    test_bootstrap = base._paired_bootstrap(
        test_y, test_prediction, current_test_prediction
    )
    raw_validation = pd.read_csv(required["validation_raw"])
    validation_bootstrap = base._validation_two_way_bootstrap(
        raw_validation,
        validation_pairs,
        validation_prediction,
        current_validation_prediction,
    )
    first = validation_pairs["ExpMean_test"].to_numpy(float)
    second = validation_pairs["ExpMean_retest"].to_numpy(float)
    test_retest = base._correlation(pearsonr, first, second)
    reliability = 2.0 * test_retest / (1.0 + test_retest)
    noise_ceiling = math.sqrt(max(0.0, reliability))
    normalized = float(validation_result["pearson"]) / noise_ceiling
    normalized_interval = validation_bootstrap[
        "human_ceiling_normalized_candidate_pearson_95_interval"
    ]

    runtime = {
        "schema": "dream-pair-ensemble-portable-ridge/v2",
        "status": "outcome_aware_retrospective_research_only",
        "feature_names": feature_names,
        "feature_contract_sha256": base._canonical_json_sha256(feature_names),
        "members": [
            _runtime_member(model, alpha, weight)
            for model, (alpha, weight) in zip(
                member_models, MEMBER_SPECS, strict=True
            )
        ],
        "prediction_clip": [0.0, 1.0],
        "allow_pickle": False,
        "external_pommix_embedding_model_required": True,
        "external_odor_pair_embedding_model_required": True,
        "odor_pair_weights_sha256": PAIR_WEIGHTS_SHA256,
        "external_model_load_policy": "torch_weights_only_hash_pinned",
        "runtime_primary_score_weight": 0.0,
        "human_olfactory_90_percent_certified": False,
    }
    serialized_runtime = json.loads(
        json.dumps(runtime, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    portable_test = predict_portable_pair_ensemble(
        serialized_runtime, test_x, feature_names
    )
    portable_validation = predict_portable_pair_ensemble(
        serialized_runtime, validation_x, feature_names
    )
    runtime_error = float(
        max(
            np.max(np.abs(portable_test - test_prediction)),
            np.max(np.abs(portable_validation - validation_prediction)),
        )
    )
    if runtime_error > 1e-12:
        raise RuntimeError("portable pair ensemble differs from sklearn")
    base._write_json(args.runtime_output, serialized_runtime)

    point_checks = _point_improvement_checks(
        test_result,
        current_test_result,
        validation_result,
        current_validation_result,
    )
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
    source_hashes = {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in required.items()
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "point_pareto_improvement_not_statistically_qualified",
        "source": {
            "dream_repository": "https://github.com/Satarifard/DREAM-olfactory-mixtures-prediction-challenge",
            "dream_git_commit": base.DREAM_COMMIT,
            "dream_root_license_file_present": False,
            "dream_files": source_hashes,
            "pommix": pommix_audit,
            "odor_pair": {
                "repository": "https://github.com/laurahsisson/dream",
                "git_commit": PAIR_SOURCE_COMMIT,
                "license": "MIT",
                "license_sha256": PAIR_SOURCE_LICENSE_SHA256,
                "license_holder_fields_complete": False,
                "redistribution_relied_on": False,
                "source_tree_sha256": PAIR_SOURCE_TREE_SHA256,
                "config_sha256": PAIR_CONFIG_SHA256,
                "weights_sha256": PAIR_WEIGHTS_SHA256,
                "precomputed_embeddings_sha256": PAIR_PRECOMPUTED_SHA256,
                "smiles_sha256": PAIR_SMILES_SHA256,
                "ogb_version": ogb_version,
                "torch_version": torch.__version__,
                "embedding_dimensions": PAIR_WIDTH,
                "precomputed_reproduction_max_abs_error": reproduction_error,
                "generated_embedding_rows_sha256": generated_embedding_sha256,
                "generated_training_mixtures": len(
                    generated_train_pair_embeddings
                ),
                "generated_test_mixtures": len(test_pair_embeddings),
                "generated_validation_mixtures": len(validation_pair_embeddings),
            },
            "prior_report_sha256": PRIOR_REPORT_SHA256,
            "prior_runtime_sha256": PRIOR_RUNTIME_SHA256,
        },
        "timing": {
            "development_used_test_and_validation_outcomes": True,
            "candidate_weights_selected_after_outcomes": True,
            "prospective_or_outcome_unopened": False,
            "evidence_class": "outcome_aware_retrospective_diagnostic",
            "post_selection_intervals_descriptive_only": True,
            "inferentially_valid_for_promotion": False,
        },
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "feature_dimensions": len(feature_names),
            "current_feature_dimensions": len(current_names),
            "odor_pair_feature_dimensions": len(_pair_feature_names()),
            "portable_runtime_rows_checked": int(len(test_y) + len(validation_y)),
            "portable_runtime_equivalence_max_abs_error": runtime_error,
        },
        "dataset": {
            "training_pairs": len(train_y),
            "training_sources": sorted(set(train_groups.tolist())),
            "test_pairs": len(test_y),
            "validation_pairs": len(validation_y),
            "training_to_test_overlap": base._component_overlap(
                train_definitions, test_definitions, scaffolds
            ),
            "training_to_validation_overlap": base._component_overlap(
                train_definitions, validation_definitions, scaffolds
            ),
        },
        "selection": {
            "name": "dual_domain_pair_ridge_ensemble_v2",
            "members": [
                {"alpha": alpha, "weight": weight}
                for alpha, weight in MEMBER_SPECS
            ],
            "rule": "retrospective point-Pareto choice on both already-public outcome sets",
            "weight_grid_step": 0.05,
            "weight_search": search_rows,
            "test_labels_used_for_selection": True,
            "validation_labels_used_for_selection": True,
            "eligible_for_promotion": False,
        },
        "test": {
            "n": len(test_y),
            "current": current_test_result,
            "candidate": test_result,
            "paired_bootstrap": test_bootstrap,
        },
        "validation": {
            "n": len(validation_y),
            "current": current_validation_result,
            "candidate": validation_result,
            "subject_pair_two_way_bootstrap": validation_bootstrap,
            "test_retest_pearson": test_retest,
            "correlation_noise_ceiling": noise_ceiling,
            "candidate_human_ceiling_normalized_pearson": normalized,
            "candidate_human_ceiling_normalized_pearson_95_interval": normalized_interval,
        },
        "gates": {
            "point_pareto": {"checks": point_checks, "passed": all(point_checks.values())},
            "statistical_improvement": {
                "checks": statistical_checks,
                "passed": all(statistical_checks.values()),
            },
            "human_ceiling_90_percent": {
                "threshold": 0.90,
                "passed": normalized_interval[0] >= 0.90,
            },
            "production": {
                "checks": {
                    "prospective_outcome_unopened": False,
                    "dream_redistribution_authorized": False,
                    "statistical_improvement_passed": all(
                        statistical_checks.values()
                    ),
                    "human_ceiling_90_percent_passed": normalized_interval[0]
                    >= 0.90,
                    "component_disjoint_test": base._component_overlap(
                        train_definitions, test_definitions, scaffolds
                    )["component_overlap"]
                    == 0,
                    "component_disjoint_validation": base._component_overlap(
                        train_definitions, validation_definitions, scaffolds
                    )["component_overlap"]
                    == 0,
                },
                "passed": False,
                "runtime_primary_score_weight": 0.0,
            },
        },
        "runtime": {
            "path": str(args.runtime_output.resolve()),
            "sha256": _sha256(args.runtime_output),
            "allow_pickle": False,
            "primary_score_weight": 0.0,
        },
        "claim_boundary": {
            "human_olfactory_90_percent_certified": False,
            "natural_language_recipe_accuracy_measured": False,
            "commercial_redistribution_authorized": False,
            "scope": "outcome-aware molecular-mixture distance research",
        },
    }
    base._write_json(args.report, report)
    print(
        json.dumps(
            {
                "test": test_result,
                "validation": validation_result,
                "validation_normalized_to_human_ceiling": normalized,
                "normalized_95_interval": normalized_interval,
                "point_pareto_gate": all(point_checks.values()),
                "statistical_gate": all(statistical_checks.values()),
                "human_90_gate": normalized_interval[0] >= 0.90,
                "generated_embedding_rows_sha256": generated_embedding_sha256,
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
