#!/usr/bin/env python
"""Retrospective DREAM-2025 human mixture-similarity benchmark.

The upstream repository publishes 507 labelled training pairs, a formerly
hidden 46-pair test set, a separately collected 50-pair validation set, and
predictions from challenge systems.  This script selects a portable ridge
model using training-source-held-out folds only, then opens test/validation
labels once for reporting.  The upstream repository currently has no root
license file, so neither source data nor fitted parameters are distributable
runtime assets.  Every report therefore fixes production weight to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dream-mixture-retrospective-v1"
SEED = 20_260_828
BOOTSTRAP_DRAWS = 20_000
ALPHAS = (
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    300_000.0,
    1_000_000.0,
)
POM_METADATA_COLUMNS = (
    "CID",
    "Molecular Formula",
    "Canonical SMILES",
    "IUPAC Name",
)
POMMIX_COMMIT = "2558557ac4793ce982d0bd26cfa79cdf4dfbbb6f"
DREAM_COMMIT = "d4294949fdc55d6bab145e8d100d58c87daf1bc6"
POMMIX_LICENSE_SHA256 = (
    "e7e0f16526da0b53905dd277a585d3e407d7ce93125e38429785da04593fbebe"
)
POMMIX_WEIGHTS_SHA256 = (
    "31b8d75daae6a4a36876a14373f43803356af05a8b111751d95ac664026d83fc"
)
POMMIX_HPARAMS_SHA256 = (
    "3f93365d45813ccc27f0999849720d75f5019cc36c0f82fe04afcdbb72467549"
)
POMMIX_IMPORT_TREE_SHA256 = (
    "5e4478a196b7e8ffac4c3f5c08d827c7f4241bf5c62fcf291c1506786da2bed8"
)
POMMIX_EMBEDDING_ROWS_SHA256 = (
    "2f74ea513256da7772cb6cde51eb6af407ff532e83350edc6a78a7435f650e7b"
)
DESCRIPTASTORUS_COMMIT = "9b133e2c91bb6a67df53db4cba992776db219ab7"
DREAM_FILE_SHA256 = {
    "cid_to_smiles": "2292f17b0f575815bb72e935b39e43e4cf81b31b1eef35b39ecf5d4035ee86b6",
    "pom_profiles": "c735dc812307be70dbebfb23d73feab1ba591f5c79b4ae749f5b45def5f98f80",
    "public_test_pair_index": "f04876bda62b05e7dd933f61dd7333a5489e45b549f72d6027ff989b1bcb327f",
    "public_test_predictions": "e51906604fee660f2a7ddc516b60bad83f33c869877b7f8d2f756c5402fe5a7e",
    "public_validation_predictions": "2c59fc1e0c37326a59e523224a21a08fd9ae946d3111e24200edeb9183aac1f0",
    "test_definitions": "be8bbadd23f46f0725c8b9050c184dd908950604342a05fbfe06bd01cee97ba8",
    "test_pairs": "b8953a3895d8168ab318805ecb63a0b5fc3e97ac601740301db3f7c9814563e3",
    "training_definitions": "a1bb25e5fcc712143c5d9060b71f90235044a543f2dbd2f72c2a8c158f5d76cf",
    "training_pairs": "28ef260ce719be120dc3bd73062126511e229c278dde4df04186e8ddc4fe2fb2",
    "validation_definitions": "f6d6e44314ff3d2345303073714e5568e2fcf7cec064e7cff7f5b5a4d65c8de3",
    "validation_raw": "ca8691bacd8ba8beb14027b377146e9a334d5530e328333b81ab0374fa435a22",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pommix_import_tree_hash(root: Path) -> str:
    files = sorted(
        [
            *root.glob("src/dataloader/**/*.py"),
            *root.glob("src/pom/gnn/**/*.py"),
            root / "src" / "pom" / "__init__.py",
        ]
    )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in files
        if path.is_file()
    ]
    return _canonical_json_sha256(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
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


def _label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    result = str(value).strip()
    if not result:
        raise ValueError("mixture label is empty")
    return result


def _correlation(function, left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    value = float(function(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float | int]:
    prediction_array = np.asarray(prediction, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 1:
        raise ValueError("metric arrays must be equally shaped vectors")
    if not np.all(np.isfinite(prediction_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("metric arrays contain non-finite values")
    error = prediction_array - target_array
    return {
        "n": int(len(target_array)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson": _correlation(pearsonr, prediction_array, target_array),
        "spearman": _correlation(spearmanr, prediction_array, target_array),
        "bias": float(np.mean(error)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def _vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


@dataclass(frozen=True)
class MixtureRepresentation:
    component_ids: frozenset[int]
    pom_mean: np.ndarray
    pom_max: np.ndarray
    pom_noisy_or: np.ndarray
    rdkit_mean: np.ndarray
    rdkit_std: np.ndarray
    morgan_mean: np.ndarray

    @property
    def size(self) -> int:
        return len(self.component_ids)


def _mixture_feature_names(
    pom_names: Sequence[str], rdkit_names: Sequence[str]
) -> list[str]:
    names: list[str] = []
    for aggregate in ("pom_mean", "pom_max", "pom_noisy_or"):
        names.extend(
            [
                f"{aggregate}::cosine",
                f"{aggregate}::correlation",
                f"{aggregate}::mean_absolute_difference",
                f"{aggregate}::root_mean_square_difference",
                f"{aggregate}::maximum_absolute_difference",
            ]
        )
        names.extend(f"{aggregate}::absolute_difference::{name}" for name in pom_names)
        names.extend(f"{aggregate}::product::{name}" for name in pom_names)
    names.extend(
        [
            "rdkit_mean::cosine",
            "rdkit_mean::correlation",
            "rdkit_mean::mean_absolute_difference",
            "rdkit_mean::root_mean_square_difference",
        ]
    )
    names.extend(f"rdkit_mean::log1p_absolute_difference::{name}" for name in rdkit_names)
    names.extend(
        [
            "rdkit_std::mean_absolute_difference",
            "rdkit_std::root_mean_square_difference",
            "morgan_mean::cosine",
            "morgan_mean::mean_absolute_difference",
            "morgan_mean::root_mean_square_difference",
            "mixture_size::minimum",
            "mixture_size::maximum",
            "mixture_size::sum",
            "mixture_size::absolute_difference",
            "component_overlap::count",
            "component_overlap::jaccard",
            "component_overlap::fraction_of_smaller",
        ]
    )
    return names


def pair_features(first: MixtureRepresentation, second: MixtureRepresentation) -> np.ndarray:
    values: list[float] = []
    for name in ("pom_mean", "pom_max", "pom_noisy_or"):
        left = getattr(first, name)
        right = getattr(second, name)
        difference = np.abs(left - right)
        values.extend(
            (
                _cosine(left, right),
                _vector_correlation(left, right),
                float(np.mean(difference)),
                float(np.sqrt(np.mean(difference * difference))),
                float(np.max(difference)),
            )
        )
        values.extend(difference.tolist())
        values.extend((left * right).tolist())
    rdkit_difference = np.abs(first.rdkit_mean - second.rdkit_mean)
    values.extend(
        (
            _cosine(first.rdkit_mean, second.rdkit_mean),
            _vector_correlation(first.rdkit_mean, second.rdkit_mean),
            float(np.mean(rdkit_difference)),
            float(np.sqrt(np.mean(rdkit_difference * rdkit_difference))),
        )
    )
    values.extend(np.log1p(rdkit_difference).tolist())
    rdkit_std_difference = np.abs(first.rdkit_std - second.rdkit_std)
    values.extend(
        (
            float(np.mean(rdkit_std_difference)),
            float(np.sqrt(np.mean(rdkit_std_difference * rdkit_std_difference))),
        )
    )
    morgan_difference = np.abs(first.morgan_mean - second.morgan_mean)
    values.extend(
        (
            _cosine(first.morgan_mean, second.morgan_mean),
            float(np.mean(morgan_difference)),
            float(np.sqrt(np.mean(morgan_difference * morgan_difference))),
        )
    )
    overlap = len(first.component_ids & second.component_ids)
    union = len(first.component_ids | second.component_ids)
    smaller = min(first.size, second.size)
    values.extend(
        (
            float(smaller),
            float(max(first.size, second.size)),
            float(first.size + second.size),
            float(abs(first.size - second.size)),
            float(overlap),
            float(overlap / max(1, union)),
            float(overlap / max(1, smaller)),
        )
    )
    result = np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    return result


def _embedding_feature_names(width: int) -> list[str]:
    names: list[str] = []
    for aggregate in ("mean", "std", "maximum", "minimum"):
        names.extend(
            [
                f"pommix_{aggregate}::cosine",
                f"pommix_{aggregate}::correlation",
                f"pommix_{aggregate}::mean_absolute_difference",
                f"pommix_{aggregate}::root_mean_square_difference",
                f"pommix_{aggregate}::maximum_absolute_difference",
            ]
        )
        names.extend(
            f"pommix_{aggregate}::absolute_difference::{index}"
            for index in range(width)
        )
        names.extend(f"pommix_{aggregate}::product::{index}" for index in range(width))
    return names


def _embedding_aggregates(
    definitions: Mapping[Any, Sequence[int]],
    embeddings: Mapping[int, np.ndarray],
) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for key, component_ids in definitions.items():
        missing = [cid for cid in component_ids if cid not in embeddings]
        if missing:
            raise ValueError(f"POMMix embeddings missing for mixture {key}: {missing}")
        values = np.asarray([embeddings[cid] for cid in component_ids], dtype=float)
        result[key] = (
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.max(values, axis=0),
            np.min(values, axis=0),
        )
    return result


def _embedding_pair_features(
    first: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    features: list[float] = []
    for left, right in zip(first, second, strict=True):
        difference = np.abs(left - right)
        features.extend(
            (
                _cosine(left, right),
                _vector_correlation(left, right),
                float(np.mean(difference)),
                float(np.sqrt(np.mean(difference * difference))),
                float(np.max(difference)),
            )
        )
        features.extend(difference.tolist())
        features.extend((left * right).tolist())
    return np.nan_to_num(
        np.asarray(features, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )


def _definitions(path: Path, *, training: bool) -> dict[Any, list[int]]:
    frame = pd.read_csv(path)
    cid_columns = [column for column in frame.columns if str(column).startswith("CID")]
    if "Mixture Label" not in frame or not cid_columns:
        raise ValueError(f"invalid mixture definition file: {path.name}")
    result: dict[Any, list[int]] = {}
    for _, row in frame.iterrows():
        key = (
            (str(row["Dataset"]).strip(), _label(row["Mixture Label"]))
            if training
            else _label(row["Mixture Label"])
        )
        component_ids = [
            int(row[column])
            for column in cid_columns
            if pd.notna(row[column]) and int(row[column]) != 0
        ]
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError(f"invalid component list for mixture {key}")
        if key in result:
            if result[key] == component_ids:
                continue
            raise ValueError(f"conflicting duplicate mixture definition: {key}")
        result[key] = component_ids
    return result


def _component_features(
    dream_root: Path,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, str],
    dict[int, str],
    list[str],
    list[str],
]:
    source = dream_root / "PostChallenge_Model" / "Dataset"
    pom_frame = pd.read_csv(source / "openpom_ensemble_predictions_results.csv")
    pom_names = [
        column for column in pom_frame.columns if column not in POM_METADATA_COLUMNS
    ]
    if not pom_names:
        raise ValueError("OpenPOM profile columns are missing")
    pom = {
        int(row["CID"]): row[pom_names].to_numpy(dtype=float)
        for _, row in pom_frame.iterrows()
    }
    smiles_frame = pd.read_csv(source / "cid_to_smiles.csv")
    smiles = {
        int(row["CID"]): str(row["smiles"])
        for _, row in smiles_frame.iterrows()
    }
    rdkit_names = [name for name, _ in Descriptors._descList[:217]]
    rdkit_functions = [function for _, function in Descriptors._descList[:217]]
    rdkit: dict[int, np.ndarray] = {}
    morgan: dict[int, np.ndarray] = {}
    scaffolds: dict[int, str] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    for cid, text in smiles.items():
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"invalid SMILES for CID {cid}")
        values = np.asarray([function(molecule) for function in rdkit_functions], dtype=float)
        rdkit[cid] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        morgan[cid] = np.asarray(generator.GetFingerprintAsNumPy(molecule), dtype=float)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        scaffolds[cid] = scaffold or f"acyclic:{Chem.MolToSmiles(molecule, canonical=True)}"
    return pom, rdkit, morgan, scaffolds, smiles, pom_names, rdkit_names


def _representations(
    definitions: Mapping[Any, Sequence[int]],
    pom: Mapping[int, np.ndarray],
    rdkit: Mapping[int, np.ndarray],
    morgan: Mapping[int, np.ndarray],
) -> dict[Any, MixtureRepresentation]:
    result: dict[Any, MixtureRepresentation] = {}
    for key, component_ids in definitions.items():
        missing = [
            cid
            for cid in component_ids
            if cid not in pom or cid not in rdkit or cid not in morgan
        ]
        if missing:
            raise ValueError(f"mixture {key} has missing component features: {missing}")
        pom_values = np.asarray([pom[cid] for cid in component_ids], dtype=float)
        rdkit_values = np.asarray([rdkit[cid] for cid in component_ids], dtype=float)
        morgan_values = np.asarray([morgan[cid] for cid in component_ids], dtype=float)
        result[key] = MixtureRepresentation(
            component_ids=frozenset(component_ids),
            pom_mean=np.mean(pom_values, axis=0),
            pom_max=np.max(pom_values, axis=0),
            pom_noisy_or=1.0 - np.prod(1.0 - np.clip(pom_values, 0.0, 1.0), axis=0),
            rdkit_mean=np.mean(rdkit_values, axis=0),
            rdkit_std=np.std(rdkit_values, axis=0),
            morgan_mean=np.mean(morgan_values, axis=0),
        )
    return result


def _training_rows(
    path: Path, representations: Mapping[Any, MixtureRepresentation]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    features: list[np.ndarray] = []
    targets: list[float] = []
    groups: list[str] = []
    for _, row in frame.iterrows():
        source = str(row["Dataset"]).strip()
        first = representations[(source, _label(row["Mixture 1"]))]
        second = representations[(source, _label(row["Mixture 2"]))]
        features.append(pair_features(first, second))
        targets.append(float(row["Experimental Values"]))
        groups.append(source)
    return np.asarray(features), np.asarray(targets), np.asarray(groups, dtype=object)


def _evaluation_rows(
    pairs: pd.DataFrame,
    representations: Mapping[Any, MixtureRepresentation],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            pair_features(
                representations[_label(row["Mixture 1"])],
                representations[_label(row["Mixture 2"])],
            )
            for _, row in pairs.iterrows()
        ]
    )
    return features, pairs[target_column].to_numpy(dtype=float)


def _embedding_rows(
    pairs: pd.DataFrame,
    representations: Mapping[
        Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    *,
    training: bool,
) -> np.ndarray:
    rows = []
    for _, row in pairs.iterrows():
        if training:
            source = str(row["Dataset"]).strip()
            first_key = (source, _label(row["Mixture 1"]))
            second_key = (source, _label(row["Mixture 2"]))
        else:
            first_key = _label(row["Mixture 1"])
            second_key = _label(row["Mixture 2"])
        rows.append(
            _embedding_pair_features(
                representations[first_key], representations[second_key]
            )
        )
    return np.asarray(rows)


def _candidate_selection(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[dict[str, Any], Any, np.ndarray]:
    rdkit_indices = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if name.startswith(("rdkit_", "morgan_", "mixture_size", "component_overlap"))
        ],
        dtype=int,
    )
    base_indices = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if not name.startswith("pommix_")
        ],
        dtype=int,
    )
    feature_sets = [
        ("pom_rdkit", base_indices),
        ("rdkit_only", rdkit_indices),
    ]
    if len(base_indices) != len(feature_names):
        feature_sets.insert(
            0, ("pommix_pom_rdkit", np.arange(features.shape[1]))
        )
    candidates: list[dict[str, Any]] = []
    folds = GroupKFold(n_splits=len(set(groups.tolist())))
    for feature_set, indices in feature_sets:
        for alpha in ALPHAS:
            prediction = np.zeros(len(target), dtype=float)
            fold_rows: list[dict[str, Any]] = []
            for fold_index, (training, validation) in enumerate(
                folds.split(features, target, groups)
            ):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(features[training][:, indices], target[training])
                fold_prediction = np.clip(
                    np.ravel(model.predict(features[validation][:, indices])), 0.0, 1.0
                )
                prediction[validation] = fold_prediction
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "held_out_sources": sorted(set(groups[validation].tolist())),
                        "metrics": metrics(fold_prediction, target[validation]),
                    }
                )
            candidates.append(
                {
                    "name": f"ridge_{feature_set}_{int(alpha)}",
                    "feature_set": feature_set,
                    "feature_indices": indices.tolist(),
                    "alpha": alpha,
                    "pooled_metrics": metrics(prediction, target),
                    "folds": fold_rows,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["pooled_metrics"]["rmse"]),
            float(row["pooled_metrics"]["mae"]),
            str(row["name"]),
        ),
    )
    selected_indices = np.asarray(selected["feature_indices"], dtype=int)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(selected["alpha"])))
    model.fit(features[:, selected_indices], target)
    public_rows = [
        {
            key: value
            for key, value in candidate.items()
            if key != "feature_indices"
        }
        for candidate in candidates
    ]
    return {"selected": selected["name"], "candidates": public_rows}, model, selected_indices


def predict_portable_ridge(
    runtime: Mapping[str, Any],
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Execute a fail-closed portable DREAM research runtime."""
    if runtime.get("schema") != "dream-mixture-portable-ridge-research/v1":
        raise ValueError("unsupported DREAM portable runtime schema")
    expected_names = runtime.get("feature_names")
    if not isinstance(expected_names, list) or expected_names != list(feature_names):
        raise ValueError("DREAM portable feature names do not match")
    if runtime.get("feature_contract_sha256") != _canonical_json_sha256(
        expected_names
    ):
        raise ValueError("DREAM portable feature contract hash does not match")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(expected_names):
        raise ValueError("DREAM portable feature matrix width does not match")
    mean = np.asarray(runtime.get("feature_mean"), dtype=float)
    scale = np.asarray(runtime.get("feature_scale"), dtype=float)
    coefficients = np.asarray(runtime.get("coefficients"), dtype=float)
    if any(value.shape != (matrix.shape[1],) for value in (mean, scale, coefficients)):
        raise ValueError("DREAM portable parameter width does not match")
    clip = np.asarray(runtime.get("prediction_clip"), dtype=float)
    intercept = float(runtime.get("intercept", math.nan))
    if (
        clip.shape != (2,)
        or clip[0] > clip[1]
        or np.any(scale <= 0.0)
        or not math.isfinite(intercept)
        or not all(
            np.all(np.isfinite(value))
            for value in (matrix, mean, scale, coefficients, clip)
        )
    ):
        raise ValueError("DREAM portable runtime contains invalid numeric values")
    prediction = ((matrix - mean) / scale) @ coefficients + intercept
    return np.clip(np.ravel(prediction), clip[0], clip[1])


def _paired_bootstrap(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    rng = np.random.RandomState(SEED)
    rmse_gain = np.empty(draws, dtype=float)
    pearson_gain = np.empty(draws, dtype=float)
    normalized_correlation = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.randint(0, len(target), size=len(target))
        truth = target[indices]
        primary = candidate[indices]
        control = baseline[indices]
        rmse_gain[draw] = float(
            np.sqrt(np.mean((control - truth) ** 2))
            - np.sqrt(np.mean((primary - truth) ** 2))
        )
        pearson_gain[draw] = _correlation(pearsonr, primary, truth) - _correlation(
            pearsonr, control, truth
        )
        normalized_correlation[draw] = _correlation(pearsonr, primary, truth)
    return {
        "draws": draws,
        "seed": SEED,
        "baseline_minus_candidate_rmse_95_interval": [
            float(value) for value in np.quantile(rmse_gain, [0.025, 0.975])
        ],
        "candidate_minus_baseline_pearson_95_interval": [
            float(value) for value in np.quantile(pearson_gain, [0.025, 0.975])
        ],
        "candidate_pearson_95_interval": [
            float(value)
            for value in np.quantile(normalized_correlation, [0.025, 0.975])
        ],
    }


def _validation_two_way_bootstrap(
    raw: pd.DataFrame,
    pairs: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    required = {"Subject Code", "Mixture 1", "Mixture 2", "Rep", "Distance"}
    if not required.issubset(raw.columns):
        raise ValueError("validation raw columns changed")
    working = raw.copy()
    working["pair_key"] = working.apply(
        lambda row: "||".join(
            sorted(
                (str(row["Mixture 1"]).strip(), str(row["Mixture 2"]).strip())
            )
        ),
        axis=1,
    )
    pair_keys = [
        "||".join(
            sorted((str(row["Mixture 1"]).strip(), str(row["Mixture 2"]).strip()))
        )
        for _, row in pairs.iterrows()
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("validation prediction pairs are duplicated")
    combined = (
        working.groupby(["Subject Code", "pair_key"])["Distance"]
        .mean()
        .unstack()[pair_keys]
        .to_numpy(dtype=float)
        / 100.0
    )
    first = (
        working.loc[working["Rep"] == 1]
        .groupby(["Subject Code", "pair_key"])["Distance"]
        .mean()
        .unstack()[pair_keys]
        .to_numpy(dtype=float)
        / 100.0
    )
    second = (
        working.loc[working["Rep"] == 2]
        .groupby(["Subject Code", "pair_key"])["Distance"]
        .mean()
        .unstack()[pair_keys]
        .to_numpy(dtype=float)
        / 100.0
    )
    if (
        combined.shape != (16, len(pair_keys))
        or first.shape != combined.shape
        or second.shape != combined.shape
        or not np.isfinite(combined).all()
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("validation subject-by-pair matrix changed")
    if "ExpMean_combined" not in pairs or not np.allclose(
        np.mean(combined, axis=0),
        pairs["ExpMean_combined"].to_numpy(dtype=float),
        atol=1e-12,
    ):
        raise ValueError("validation raw means differ from published pair outcomes")
    rng = np.random.RandomState(SEED + 1)
    rmse_gain = np.empty(draws, dtype=float)
    pearson_gain = np.empty(draws, dtype=float)
    candidate_pearson = np.empty(draws, dtype=float)
    normalized = []
    for draw in range(draws):
        subjects = rng.randint(0, combined.shape[0], size=combined.shape[0])
        pair_indices = rng.randint(0, combined.shape[1], size=combined.shape[1])
        truth = np.mean(combined[subjects], axis=0)[pair_indices]
        primary = candidate[pair_indices]
        control = baseline[pair_indices]
        rmse_gain[draw] = float(
            np.sqrt(np.mean((control - truth) ** 2))
            - np.sqrt(np.mean((primary - truth) ** 2))
        )
        primary_correlation = _correlation(pearsonr, primary, truth)
        candidate_pearson[draw] = primary_correlation
        pearson_gain[draw] = primary_correlation - _correlation(
            pearsonr, control, truth
        )
        first_mean = np.mean(first[subjects], axis=0)[pair_indices]
        second_mean = np.mean(second[subjects], axis=0)[pair_indices]
        reliability = _correlation(pearsonr, first_mean, second_mean)
        corrected = (
            2.0 * reliability / (1.0 + reliability)
            if reliability > 0.0
            else 0.0
        )
        ceiling = math.sqrt(max(0.0, corrected))
        if ceiling > 1e-12:
            normalized.append(primary_correlation / ceiling)
    if len(normalized) < draws * 0.95:
        raise RuntimeError("too few valid two-way human-ceiling bootstrap draws")
    return {
        "draws": draws,
        "valid_human_ceiling_draws": len(normalized),
        "seed": SEED + 1,
        "resampling_unit": "subjects and mixture pairs",
        "baseline_minus_candidate_rmse_95_interval": [
            float(value) for value in np.quantile(rmse_gain, [0.025, 0.975])
        ],
        "candidate_minus_baseline_pearson_95_interval": [
            float(value) for value in np.quantile(pearson_gain, [0.025, 0.975])
        ],
        "candidate_pearson_95_interval": [
            float(value) for value in np.quantile(candidate_pearson, [0.025, 0.975])
        ],
        "human_ceiling_normalized_candidate_pearson_95_interval": [
            float(value) for value in np.quantile(normalized, [0.025, 0.975])
        ],
    }


def _component_overlap(
    first: Mapping[Any, Sequence[int]],
    second: Mapping[Any, Sequence[int]],
    scaffolds: Mapping[int, str],
) -> dict[str, Any]:
    left = {cid for values in first.values() for cid in values}
    right = {cid for values in second.values() for cid in values}
    left_scaffolds = {scaffolds[cid] for cid in left}
    right_scaffolds = {scaffolds[cid] for cid in right}
    return {
        "first_components": len(left),
        "second_components": len(right),
        "component_overlap": len(left & right),
        "component_overlap_fraction_of_second": len(left & right) / max(1, len(right)),
        "first_scaffolds": len(left_scaffolds),
        "second_scaffolds": len(right_scaffolds),
        "scaffold_overlap": len(left_scaffolds & right_scaffolds),
        "scaffold_overlap_fraction_of_second": len(left_scaffolds & right_scaffolds)
        / max(1, len(right_scaffolds)),
    }


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


def _pommix_embeddings(
    pommix_root: Path,
    smiles: Mapping[int, str],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    commit = _git_commit(pommix_root)
    if commit != POMMIX_COMMIT:
        raise RuntimeError(f"unsupported POMMix commit: {commit}")
    import torch
    import torch_geometric
    from torch_geometric.data import Batch

    model_root = pommix_root / "scripts" / "pom" / "gs-lf_models" / "pretrained_pom"
    hparams_path = model_root / "hparams.json"
    weights_path = model_root / "gnn_embedder.pt"
    graph_utils_path = (
        pommix_root / "src" / "dataloader" / "representations" / "graph_utils.py"
    )
    graphnets_path = pommix_root / "src" / "pom" / "gnn" / "graphnets.py"
    for path in (hparams_path, weights_path, graph_utils_path, graphnets_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing POMMix artifact: {path}")
    import_tree_hash = _pommix_import_tree_hash(pommix_root)
    if (
        _sha256(pommix_root / "LICENSE") != POMMIX_LICENSE_SHA256
        or _sha256(hparams_path) != POMMIX_HPARAMS_SHA256
        or _sha256(weights_path) != POMMIX_WEIGHTS_SHA256
        or import_tree_hash != POMMIX_IMPORT_TREE_SHA256
    ):
        raise RuntimeError("POMMix source or model bytes differ from the reviewed release")
    source_root = pommix_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from dataloader.representations import graph_utils as graph_utils_module
    from pom.gnn import graphnets as graphnets_module

    imported_paths = {
        Path(graph_utils_module.__file__).resolve(),
        Path(graphnets_module.__file__).resolve(),
    }
    expected_paths = {graph_utils_path.resolve(), graphnets_path.resolve()}
    if imported_paths != expected_paths:
        raise RuntimeError("POMMix imports resolved outside the reviewed source tree")
    EDGE_DIM = graph_utils_module.EDGE_DIM
    NODE_DIM = graph_utils_module.NODE_DIM
    from_smiles = graph_utils_module.from_smiles
    GraphNets = graphnets_module.GraphNets

    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    model = GraphNets(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=int(hparams["global_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        depth=int(hparams["depth"]),
        dropout=float(hparams["dropout"]),
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    # CPU inference avoids scatter/reduction nondeterminism across CUDA builds;
    # this benchmark is small enough that deterministic evidence is preferable.
    device = torch.device("cpu")
    model = model.to(device).eval()
    ordered = sorted(smiles)
    graphs = [from_smiles(smiles[cid]) for cid in ordered]
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(graphs), 64):
            batch = Batch.from_data_list(graphs[start : start + 64]).to(device)
            rows.append(model(batch).detach().cpu().numpy())
    values = np.vstack(rows).astype(np.float32)
    if values.shape != (len(ordered), int(hparams["global_dim"])):
        raise RuntimeError("POMMix embedding shape changed")
    if not np.isfinite(values).all():
        raise RuntimeError("POMMix embeddings contain non-finite values")
    embedding_rows_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    if embedding_rows_sha256 != POMMIX_EMBEDDING_ROWS_SHA256:
        raise RuntimeError("POMMix embeddings differ from the reviewed CPU output")
    try:
        import descriptastorus

        descriptastorus_version = getattr(descriptastorus, "__version__", "unknown")
    except ImportError:
        descriptastorus_version = "unavailable"
    if descriptastorus_version != "2.5.0.25":
        raise RuntimeError(
            "POMMix benchmark requires descriptastorus 2.5.0.25 from the pinned commit"
        )
    return (
        {cid: values[index] for index, cid in enumerate(ordered)},
        {
            "repository": "https://github.com/chemcognition-lab/pom-mix",
            "git_commit": commit,
            "license_file_sha256": _sha256(pommix_root / "LICENSE"),
            "license": "MIT",
            "hparams_sha256": _sha256(hparams_path),
            "weights_sha256": _sha256(weights_path),
            "graph_utils_sha256": _sha256(graph_utils_path),
            "graphnets_sha256": _sha256(graphnets_path),
            "import_tree_sha256": import_tree_hash,
            "embedding_dimensions": int(values.shape[1]),
            "molecules": int(values.shape[0]),
            "device": str(device),
            "torch_version": torch.__version__,
            "torch_geometric_version": torch_geometric.__version__,
            "descriptastorus_version": descriptastorus_version,
            "descriptastorus_commit": DESCRIPTASTORUS_COMMIT,
            "embedding_rows_sha256": embedding_rows_sha256,
        },
    )


def _frozen_r2_predictions(
    definitions: Mapping[Any, Sequence[int]],
    pairs: pd.DataFrame,
    smiles: Mapping[int, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from fragrance_ai.recommender.numpy_r2 import (
        EXPECTED_STATE_SHAPES,
        NumpyR2Model,
    )
    from fragrance_ai.research.r2_physsim import build_raw_descriptor_cache

    data_root = PROJECT_ROOT / "fragrance_ai" / "data"
    runtime_path = data_root / "physsim_r2_runtime_weights.npz"
    runtime_manifest_path = data_root / "physsim_r2_runtime_manifest.json"
    ensemble_manifest_path = data_root / "physsim_r2_ensemble_manifest.json"
    runtime_bytes = runtime_path.read_bytes()
    runtime_manifest_bytes = runtime_manifest_path.read_bytes()
    ensemble_manifest_bytes = ensemble_manifest_path.read_bytes()
    runtime_manifest = json.loads(runtime_manifest_bytes)
    if _sha256(runtime_path) != runtime_manifest.get("artifact_sha256"):
        raise RuntimeError("frozen R2 portable runtime hash mismatch")
    if hashlib.sha256(ensemble_manifest_bytes).hexdigest() != runtime_manifest.get(
        "ensemble_manifest_sha256"
    ):
        raise RuntimeError("frozen R2 ensemble binding mismatch")
    state_keys = list(runtime_manifest.get("state_keys", []))
    if set(state_keys) != set(EXPECTED_STATE_SHAPES):
        raise RuntimeError("frozen R2 state contract changed")
    members = list(runtime_manifest.get("members", []))
    weights = np.asarray([float(member["weight"]) for member in members], dtype=float)
    if len(weights) != 2 or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("frozen R2 ensemble weights are invalid")
    needed = sorted({cid for values in definitions.values() for cid in values})
    missing = [cid for cid in needed if cid not in smiles]
    if missing:
        raise ValueError(f"frozen R2 lacks SMILES for CIDs: {missing}")
    raw = build_raw_descriptor_cache(smiles[cid] for cid in needed)
    models: list[NumpyR2Model] = []
    with np.load(io.BytesIO(runtime_bytes), allow_pickle=False) as data:
        mean = np.asarray(data["normalizer_mean"], dtype=float)
        std = np.asarray(data["normalizer_std"], dtype=float)
        if mean.shape != (217,) or std.shape != (217,) or np.any(std <= 0):
            raise RuntimeError("frozen R2 normalizer is invalid")
        standardized = {
            cid: np.clip(
                np.nan_to_num(
                    (np.asarray(raw[smiles[cid]], dtype=float) - mean) / std,
                    nan=0.0,
                    posinf=100.0,
                    neginf=-100.0,
                ),
                -100.0,
                100.0,
            ).astype(np.float32)
            for cid in needed
        }
        for index in range(len(members)):
            state = {
                key: np.asarray(data[f"member_{index}::{key}"], dtype=np.float32)
                for key in state_keys
            }
            models.append(NumpyR2Model(state))
    prediction: list[float] = []
    member_rows: list[list[float]] = []
    for _, row in pairs.iterrows():
        first_ids = definitions[_label(row["Mixture 1"])]
        second_ids = definitions[_label(row["Mixture 2"])]
        first = np.asarray([standardized[cid] for cid in first_ids], dtype=np.float32)
        second = np.asarray([standardized[cid] for cid in second_ids], dtype=np.float32)
        member_values = [float(model.predict(first, second)) for model in models]
        # Frozen R2 emits similarity; DREAM labels perceptual distance.
        prediction.append(float(np.clip(1.0 - weights @ member_values, 0.0, 1.0)))
        member_rows.append(member_values)
    return np.asarray(prediction), {
        "runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "runtime_manifest_sha256": hashlib.sha256(runtime_manifest_bytes).hexdigest(),
        "ensemble_manifest_sha256": hashlib.sha256(ensemble_manifest_bytes).hexdigest(),
        "member_weights": weights.tolist(),
        "maximum_member_disagreement": float(
            np.max(np.ptp(np.asarray(member_rows, dtype=float), axis=1))
        ),
        "output_transform": "dream_distance=1-frozen_r2_similarity",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-root", type=Path, required=True)
    parser.add_argument("--pommix-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/dream_mixture_2025_retrospective_v1.json",
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/dream_mixture_2025_research_runtime_v1.json",
    )
    args = parser.parse_args()
    dream_root = args.dream_root.expanduser().resolve(strict=True)
    pommix_root = args.pommix_root.expanduser().resolve(strict=True)
    dream_commit = _git_commit(dream_root)
    if dream_commit != DREAM_COMMIT:
        raise RuntimeError(f"unsupported DREAM source commit: {dream_commit}")
    required = {
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
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing DREAM files: " + ", ".join(missing))
    changed = [
        name
        for name, path in required.items()
        if _sha256(path) != DREAM_FILE_SHA256.get(name)
    ]
    if changed:
        raise RuntimeError("DREAM benchmark bytes changed: " + ", ".join(changed))

    pom, rdkit, morgan, scaffolds, smiles, pom_names, rdkit_names = _component_features(
        dream_root
    )
    training_definitions = _definitions(required["training_definitions"], training=True)
    test_definitions = _definitions(required["test_definitions"], training=False)
    validation_definitions = _definitions(
        required["validation_definitions"], training=False
    )
    training_representations = _representations(
        training_definitions, pom, rdkit, morgan
    )
    test_representations = _representations(test_definitions, pom, rdkit, morgan)
    validation_representations = _representations(
        validation_definitions, pom, rdkit, morgan
    )
    pommix_embeddings, pommix_audit = _pommix_embeddings(pommix_root, smiles)
    training_pommix = _embedding_aggregates(
        training_definitions, pommix_embeddings
    )
    test_pommix = _embedding_aggregates(test_definitions, pommix_embeddings)
    validation_pommix = _embedding_aggregates(
        validation_definitions, pommix_embeddings
    )
    feature_names = [
        *_mixture_feature_names(pom_names, rdkit_names),
        *_embedding_feature_names(pommix_audit["embedding_dimensions"]),
    ]
    training_pairs_frame = pd.read_csv(required["training_pairs"])
    training_x, training_y, training_groups = _training_rows(
        required["training_pairs"], training_representations
    )
    training_x = np.column_stack(
        (
            training_x,
            _embedding_rows(
                training_pairs_frame, training_pommix, training=True
            ),
        )
    )
    if training_x.shape[1] != len(feature_names):
        raise RuntimeError("feature names do not match computed feature width")
    test_pairs = pd.read_csv(required["test_pairs"])
    test_x, test_y = _evaluation_rows(
        test_pairs, test_representations, "Experimental values"
    )
    test_x = np.column_stack(
        (test_x, _embedding_rows(test_pairs, test_pommix, training=False))
    )
    validation_predictions = pd.read_csv(required["public_validation_predictions"])
    validation_x, validation_y = _evaluation_rows(
        validation_predictions,
        validation_representations,
        "ExpMean_combined",
    )
    validation_x = np.column_stack(
        (
            validation_x,
            _embedding_rows(
                validation_predictions, validation_pommix, training=False
            ),
        )
    )

    selection, model, selected_indices = _candidate_selection(
        training_x, training_y, training_groups, feature_names
    )
    test_prediction = np.clip(
        np.ravel(model.predict(test_x[:, selected_indices])), 0.0, 1.0
    )
    validation_prediction = np.clip(
        np.ravel(model.predict(validation_x[:, selected_indices])), 0.0, 1.0
    )
    public_test = pd.read_csv(required["public_test_predictions"])
    test_baseline = public_test["Predicted_Values_Ensemble"].to_numpy(float)
    validation_baseline = validation_predictions[
        "Predicted_Values_Ensemble"
    ].to_numpy(float)

    public_test_index = pd.read_csv(required["public_test_pair_index"])
    official_pair_indices = {
        (_label(row["Mixture 1"]), _label(row["Mixture 2"])): index
        for index, (_, row) in enumerate(test_pairs.iterrows())
    }
    reorder = np.asarray(
        [
            official_pair_indices[
                (_label(row["Mixture 1"]), _label(row["Mixture 2"]))
            ]
            for _, row in public_test_index.iterrows()
        ],
        dtype=int,
    )
    test_prediction = test_prediction[reorder]
    test_y = test_y[reorder]
    frozen_r2_test, frozen_r2_audit = _frozen_r2_predictions(
        test_definitions, test_pairs, {cid: text for cid, text in smiles.items()}
    )
    frozen_r2_test = frozen_r2_test[reorder]
    frozen_r2_validation, frozen_r2_validation_audit = _frozen_r2_predictions(
        validation_definitions,
        validation_predictions,
        {cid: text for cid, text in smiles.items()},
    )
    if not np.allclose(
        public_test_index["Experimental_Values"].to_numpy(float), test_y, atol=1e-12
    ):
        raise RuntimeError("public test pair index differs from official test outcomes")
    if len(public_test) != len(test_y) or not np.allclose(
        public_test["Experimental_Values"].to_numpy(float), test_y, atol=1e-12
    ):
        raise RuntimeError("public SOTA predictions differ from test pair index")

    test_result = metrics(test_prediction, test_y)
    validation_result = metrics(validation_prediction, validation_y)
    test_baseline_result = metrics(test_baseline, test_y)
    validation_baseline_result = metrics(validation_baseline, validation_y)
    frozen_r2_test_result = metrics(frozen_r2_test, test_y)
    frozen_r2_validation_result = metrics(frozen_r2_validation, validation_y)
    test_bootstrap = _paired_bootstrap(test_y, test_prediction, test_baseline)
    validation_bootstrap = _paired_bootstrap(
        validation_y, validation_prediction, validation_baseline
    )
    frozen_r2_test_bootstrap = _paired_bootstrap(
        test_y, test_prediction, frozen_r2_test
    )
    frozen_r2_validation_bootstrap = _paired_bootstrap(
        validation_y, validation_prediction, frozen_r2_validation
    )

    raw_validation = pd.read_csv(required["validation_raw"])
    validation_two_way = _validation_two_way_bootstrap(
        raw_validation,
        validation_predictions,
        validation_prediction,
        validation_baseline,
    )
    first = validation_predictions["ExpMean_test"].to_numpy(float)
    second = validation_predictions["ExpMean_retest"].to_numpy(float)
    test_retest = _correlation(pearsonr, first, second)
    spearman_brown = 2.0 * test_retest / (1.0 + test_retest)
    noise_ceiling = math.sqrt(max(0.0, spearman_brown))
    normalized_correlation = (
        float(validation_result["pearson"]) / noise_ceiling
        if noise_ceiling > 1e-12
        else 0.0
    )
    pair_only_normalized_interval = [
        value / noise_ceiling
        for value in validation_bootstrap["candidate_pearson_95_interval"]
    ]
    normalized_interval = validation_two_way[
        "human_ceiling_normalized_candidate_pearson_95_interval"
    ]

    pipeline = model
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    ridge: Ridge = pipeline.named_steps["ridge"]
    selected_names = [feature_names[index] for index in selected_indices]
    runtime = {
        "schema": "dream-mixture-portable-ridge-research/v1",
        "status": "retrospective_research_only_weight_zero",
        "feature_contract_sha256": _canonical_json_sha256(selected_names),
        "feature_names": selected_names,
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in np.ravel(ridge.coef_)],
        "intercept": float(np.ravel(np.asarray(ridge.intercept_))[0]),
        "prediction_clip": [0.0, 1.0],
        "allow_pickle": False,
        "external_openpom_profile_registry_required": True,
        "external_pommix_embedding_model_required": True,
        "pommix_weights_sha256": pommix_audit["weights_sha256"],
        "runtime_primary_score_weight": 0.0,
        "human_olfactory_90_percent_certified": False,
    }
    serialized_runtime = json.loads(
        json.dumps(runtime, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    portable_test_prediction = predict_portable_ridge(
        serialized_runtime, test_x[reorder][:, selected_indices], selected_names
    )
    portable_validation_prediction = predict_portable_ridge(
        serialized_runtime, validation_x[:, selected_indices], selected_names
    )
    portable_equivalence_error = float(
        max(
            np.max(np.abs(portable_test_prediction - test_prediction)),
            np.max(
                np.abs(portable_validation_prediction - validation_prediction)
            ),
        )
    )
    if portable_equivalence_error > 1e-12:
        raise RuntimeError(
            "portable DREAM runtime is not equivalent to sklearn: "
            f"max_abs_error={portable_equivalence_error:.17g}"
        )
    _write_json(args.runtime_output, serialized_runtime)

    test_gate = {
        "candidate_pearson_above_public_ensemble": test_result["pearson"]
        > test_baseline_result["pearson"],
        "candidate_rmse_below_public_ensemble": test_result["rmse"]
        < test_baseline_result["rmse"],
        "rmse_gain_bootstrap_lower_above_zero": test_bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "pearson_gain_bootstrap_lower_above_zero": test_bootstrap[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
    }
    validation_gate = {
        "candidate_pearson_above_public_ensemble": validation_result["pearson"]
        > validation_baseline_result["pearson"],
        "candidate_rmse_below_public_ensemble": validation_result["rmse"]
        < validation_baseline_result["rmse"],
        "rmse_gain_two_way_bootstrap_lower_above_zero": validation_two_way[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "pearson_gain_two_way_bootstrap_lower_above_zero": validation_two_way[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
    }
    internal_improvement_checks = {
        "test_candidate_pearson_above_current_r2": test_result["pearson"]
        > frozen_r2_test_result["pearson"],
        "test_candidate_rmse_below_current_r2": test_result["rmse"]
        < frozen_r2_test_result["rmse"],
        "test_rmse_gain_bootstrap_lower_above_zero": frozen_r2_test_bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "test_pearson_gain_bootstrap_lower_above_zero": frozen_r2_test_bootstrap[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
        "validation_candidate_pearson_above_current_r2": validation_result["pearson"]
        > frozen_r2_validation_result["pearson"],
        "validation_candidate_rmse_below_current_r2": validation_result["rmse"]
        < frozen_r2_validation_result["rmse"],
        "validation_rmse_gain_bootstrap_lower_above_zero": frozen_r2_validation_bootstrap[
            "baseline_minus_candidate_rmse_95_interval"
        ][0]
        > 0.0,
        "validation_pearson_gain_bootstrap_lower_above_zero": frozen_r2_validation_bootstrap[
            "candidate_minus_baseline_pearson_95_interval"
        ][0]
        > 0.0,
    }
    license_present = any(
        (dream_root / name).is_file()
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
    )
    test_overlap = _component_overlap(
        training_definitions, test_definitions, scaffolds
    )
    validation_overlap = _component_overlap(
        training_definitions, validation_definitions, scaffolds
    )
    source_hashes = {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in required.items()
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_external_improvement_not_production_eligible",
        "source": {
            "repository": "https://github.com/Satarifard/DREAM-olfactory-mixtures-prediction-challenge",
            "git_commit": dream_commit,
            "root_license_file_present": license_present,
            "license_status": (
                "declared_in_repository_root"
                if license_present
                else "not_declared_in_repository_root"
            ),
            "files": source_hashes,
            "pommix": pommix_audit,
        },
        "timing": {
            "upstream_test_was_formerly_hidden_but_public_before_this_run": True,
            "upstream_validation_outcomes_were_public_before_this_run": True,
            "internal_outcome_unopened_prediction_seal": False,
            "development_used_test_or_validation_labels": True,
            "formal_candidate_ranking_used_test_or_validation_labels": False,
            "promotion_evidence_class": "retrospective_external_publication",
        },
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "rdkit_version": rdBase.rdkitVersion,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__,
            "sklearn_model": "StandardScaler plus Ridge",
            "portable_runtime_equivalence_max_abs_error": portable_equivalence_error,
            "portable_runtime_rows_checked": int(len(test_y) + len(validation_y)),
        },
        "dataset": {
            "training_pairs": len(training_y),
            "upstream_readme_claimed_training_pairs": 507,
            "repository_actual_training_pair_rows": len(training_y),
            "readme_training_pair_count_matches_repository": len(training_y) == 507,
            "training_sources": sorted(set(training_groups.tolist())),
            "test_pairs": len(test_y),
            "validation_pairs": len(validation_y),
            "validation_raw_rows": len(raw_validation),
            "pom_profile_dimensions": len(pom_names),
            "rdkit_descriptor_dimensions": len(rdkit_names),
            "pair_feature_dimensions": len(feature_names),
            "pommix_embedding_dimensions": pommix_audit["embedding_dimensions"],
            "training_to_test_overlap": test_overlap,
            "training_to_validation_overlap": validation_overlap,
        },
        "selection": {
            **selection,
            "rule": "minimum pooled leave-entire-source-out training RMSE, then MAE",
            "test_labels_used_for_selection": False,
            "validation_labels_used_for_selection": False,
            "development_outcome_aware": True,
        },
        "test": {
            "candidate": test_result,
            "current_frozen_r2": frozen_r2_test_result,
            "current_frozen_r2_audit": frozen_r2_audit,
            "candidate_vs_current_frozen_r2_bootstrap": frozen_r2_test_bootstrap,
            "fixed_public_sota_ensemble": test_baseline_result,
            "paired_bootstrap": test_bootstrap,
            "gate_checks": test_gate,
            "gate_passed": all(test_gate.values()),
        },
        "validation": {
            "candidate": validation_result,
            "current_frozen_r2": frozen_r2_validation_result,
            "current_frozen_r2_audit": frozen_r2_validation_audit,
            "candidate_vs_current_frozen_r2_bootstrap": (
                frozen_r2_validation_bootstrap
            ),
            "fixed_public_top6_ensemble": validation_baseline_result,
            "paired_bootstrap": validation_bootstrap,
            "two_way_subject_pair_bootstrap": validation_two_way,
            "gate_checks": validation_gate,
            "gate_passed": all(validation_gate.values()),
            "test_retest_pearson": test_retest,
            "spearman_brown_reliability": spearman_brown,
            "correlation_noise_ceiling": noise_ceiling,
            "candidate_human_ceiling_normalized_pearson": normalized_correlation,
            "candidate_human_ceiling_normalized_pearson_95_interval": normalized_interval,
            "pair_only_fixed_ceiling_normalized_pearson_95_interval": (
                pair_only_normalized_interval
            ),
            "human_ceiling_90_percent_gate": {
                "threshold": 0.90,
                "passed": normalized_interval[0] >= 0.90,
            },
        },
        "release_gate": {
            "checks": {
                "root_license_declared": license_present,
                "internally_outcome_unopened": False,
                "training_to_test_component_disjoint": test_overlap[
                    "component_overlap"
                ]
                == 0,
                "training_to_validation_component_disjoint": validation_overlap[
                    "component_overlap"
                ]
                == 0,
                "test_external_gate_passed": all(test_gate.values()),
                "validation_external_gate_passed": all(validation_gate.values()),
                "human_ceiling_90_percent_lower_bound_passed": normalized_interval[0]
                >= 0.90,
            },
            "passed": False,
            "runtime_primary_score_weight": 0.0,
        },
        "internal_large_improvement_gate": {
            "checks": internal_improvement_checks,
            "passed": all(internal_improvement_checks.values()),
            "scope": "retrospective improvement over the current frozen R2 baseline",
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
            "scope": "retrospective molecular-mixture perceptual-distance research",
        },
    }
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "selected": selection["selected"],
                "test": test_result,
                "validation": validation_result,
                "validation_normalized_to_human_ceiling": normalized_correlation,
                "release_gate": False,
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
