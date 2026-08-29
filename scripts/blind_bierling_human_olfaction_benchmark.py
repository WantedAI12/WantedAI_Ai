#!/usr/bin/env python
"""Outcome-unopened public-human olfaction benchmark on Bierling et al. 2025.

The workflow has four irreversible stages:

1. ``prepare`` reads only the public target odor table and Keller 2016 training
   outcomes.  It excludes every target molecule from Keller supervision,
   selects the molecular predictor only by deterministic Keller
   molecule-disjoint cross-validation, and writes target predictions.
2. ``seal`` binds the predictions, this script, the predeclared outcome file
   identity, and the scoring contract while requiring the target outcome path
   to be absent.
3. The seal is externally timestamped. ``acquire`` verifies that RFC 3161
   timestamp before downloading the target outcome from the fixed Zenodo URL.
4. ``score`` re-verifies every binding and performs the single target scoring.

The endpoint is cross-odor prediction of population-mean monomolecular ratings.
It is not a generated-perfume test and never authorizes a 90% olfactory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "1.0"
DATASET_RECORD_ID = 15_657_278
DATASET_DOI = "10.5281/zenodo.15657278"
DATASET_ARTICLE_DOI = "10.1038/s41597-025-04644-2"
OUTCOME_URL = (
    "https://zenodo.org/api/records/15657278/files/data.csv/content"
)
OUTCOME_FILE = "data.csv"
OUTCOME_BYTES = 4_898_146
OUTCOME_MD5 = "2b0591617a6e806ee619c2c2466a03ed"
TARGET_METADATA_SHA256 = "04bda89e4112521f4a87a7f1ba726d577ba357ebe590bc5b4fa4e7a386f34a80"
TARGET_METADATA_BYTES = 9_917
VARIABLE_DICTIONARY_SHA256 = (
    "c921d8cfad6874bfc139f245ee2995d36dca3cd6322506717950217d63afc20b"
)
VARIABLE_DICTIONARY_BYTES = 24_605
KELLER_SOURCE_CONTRACT = {
    "molecules.csv": {
        "sha256": "5f862b6ea62dd3ee0a15138605ea2567d06a527263284348a5982844f9bec81f",
        "bytes": 35_670,
    },
    "stimuli.csv": {
        "sha256": "49c1a6bfb8c1665a31e13f515add95a95291f29416be1bb0b54d3317efb6665b",
        "bytes": 59_481,
    },
    "behavior.csv": {
        "sha256": "ab31c17c475c8b471706822f89aaf9e148bebe5f50f33ff03ebc42b32ac94df8",
        "bytes": 68_163_172,
    },
}
TARGET_ODOR_COUNT = 74
CV_FOLDS = 5
CV_SALT = "bierling-2025-target-excluded-keller-cv-v1"
BOOTSTRAP_SEED = 20_260_826
BOOTSTRAP_DRAWS = 1_000
RELIABILITY_REPEATS = 250
QUALITATIVE_ENDPOINTS = (
    "sweet",
    "sour",
    "fruity",
    "spices",
    "bakery",
    "garlic",
    "fish",
    "burnt",
    "decayed",
    "grass",
    "wood",
    "chemical",
    "flower",
    "musky",
    "sweaty",
    "ammonia/urinuos",
)
ENDPOINTS: OrderedDict[str, dict[str, str]] = OrderedDict(
    (
        (
            "intensive",
            {"keller": "HOW STRONG IS THE SMELL?", "kind": "vas"},
        ),
        (
            "pleasant",
            {"keller": "HOW PLEASANT IS THE SMELL?", "kind": "vas"},
        ),
        (
            "familiar",
            {"keller": "HOW FAMILIAR IS THE SMELL?", "kind": "vas"},
        ),
        ("warm", {"keller": "WARM", "kind": "semantic"}),
        ("cold", {"keller": "COLD", "kind": "semantic"}),
        ("edible", {"keller": "EDIBLE", "kind": "semantic"}),
        ("sweet", {"keller": "SWEET", "kind": "binary"}),
        ("sour", {"keller": "SOUR", "kind": "binary"}),
        ("fruity", {"keller": "FRUIT", "kind": "binary"}),
        ("spices", {"keller": "SPICES", "kind": "binary"}),
        ("bakery", {"keller": "BAKERY", "kind": "binary"}),
        ("garlic", {"keller": "GARLIC", "kind": "binary"}),
        ("fish", {"keller": "FISH", "kind": "binary"}),
        ("burnt", {"keller": "BURNT", "kind": "binary"}),
        ("decayed", {"keller": "DECAYED", "kind": "binary"}),
        ("grass", {"keller": "GRASS", "kind": "binary"}),
        ("wood", {"keller": "WOOD", "kind": "binary"}),
        ("chemical", {"keller": "CHEMICAL", "kind": "binary"}),
        ("flower", {"keller": "FLOWER", "kind": "binary"}),
        ("musky", {"keller": "MUSKY", "kind": "binary"}),
        ("sweaty", {"keller": "SWEATY", "kind": "binary"}),
        (
            "ammonia/urinuos",
            {"keller": "AMMONIA/URINOUS", "kind": "binary"},
        ),
    )
)
FIXED_BASELINE = "rdkit_ridge_alpha_100"
CANDIDATES: tuple[dict[str, Any], ...] = (
    *tuple(
        {
            "name": f"rdkit_ridge_alpha_{alpha}",
            "algorithm": "ridge",
            "feature": "rdkit",
            "alpha": float(alpha),
        }
        for alpha in (1, 10, 100, 1000)
    ),
    *tuple(
        {
            "name": f"morgan_ridge_alpha_{alpha}",
            "algorithm": "ridge",
            "feature": "morgan",
            "alpha": float(alpha),
        }
        for alpha in (10, 100, 1000)
    ),
    *tuple(
        {
            "name": f"molformer_ridge_alpha_{alpha}",
            "algorithm": "ridge",
            "feature": "molformer",
            "alpha": float(alpha),
        }
        for alpha in (1, 10, 100, 1000)
    ),
    *tuple(
        {
            "name": f"fusion_ridge_alpha_{alpha}",
            "algorithm": "ridge",
            "feature": "fusion",
            "alpha": float(alpha),
        }
        for alpha in (10, 100, 1000)
    ),
    {
        "name": "rdkit_extra_trees_leaf_2",
        "algorithm": "extra_trees",
        "feature": "rdkit",
        "min_samples_leaf": 2,
    },
    {
        "name": "rdkit_extra_trees_leaf_5",
        "algorithm": "extra_trees",
        "feature": "rdkit",
        "min_samples_leaf": 5,
    },
    {
        "name": "molformer_extra_trees_leaf_2",
        "algorithm": "extra_trees",
        "feature": "molformer",
        "min_samples_leaf": 2,
    },
    {
        "name": "molformer_extra_trees_leaf_5",
        "algorithm": "extra_trees",
        "feature": "molformer",
        "min_samples_leaf": 5,
    },
    *tuple(
        {
            "name": f"morgan_similarity_knn_{neighbors}",
            "algorithm": "similarity_knn",
            "feature": "morgan",
            "neighbors": neighbors,
        }
        for neighbors in (5, 15, 30)
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _identifier(value: object) -> str:
    text = str(value).strip().replace("\u00a0", "")
    return text[:-2] if text.endswith(".0") else text


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        left_array.shape != right_array.shape
        or left_array.ndim != 1
        or left_array.size < 3
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        raise ValueError("Spearman inputs must be equal finite vectors of length >= 3")
    left_rank = _average_ranks(left_array)
    right_rank = _average_ranks(right_array)
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _canonical_smiles(value: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        raise ValueError(f"invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _murcko_scaffold(value: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(value)


def load_target_odors(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path, sep=";")
    required = {
        "odor_group",
        "molcode",
        "name",
        "SMILES",
        "cid",
        "cas",
        "concentration_final",
        "volume_final",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Bierling odor metadata columns changed")
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for _, source in frame.iterrows():
        molcode = str(source["molcode"]).strip()
        canonical = _canonical_smiles(str(source["SMILES"]))
        if molcode in seen:
            if seen[molcode] != canonical:
                raise RuntimeError(f"target molcode {molcode} maps to two structures")
            continue
        seen[molcode] = canonical
        rows.append(
            {
                "molcode": molcode,
                "name": str(source["name"]).strip(),
                "cid": _identifier(source["cid"]),
                "cas": str(source["cas"]).strip(),
                "canonical_smiles": canonical,
                "murcko_scaffold": _murcko_scaffold(canonical),
                "concentration_final": str(source["concentration_final"]).strip(),
                "volume_final": str(source["volume_final"]).strip(),
            }
        )
    rows.sort(key=lambda item: item["molcode"])
    if len(rows) != TARGET_ODOR_COUNT:
        raise RuntimeError(
            f"expected {TARGET_ODOR_COUNT} unique target odors, found {len(rows)}"
        )
    return rows


def _keller_means(
    molecules_path: Path,
    stimuli_path: Path,
    behavior_path: Path,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, float]], dict[str, Any]]:
    """Aggregate the highest-concentration Keller stimulus per molecule."""

    import pandas as pd

    molecules = pd.read_csv(molecules_path)
    stimuli = pd.read_csv(stimuli_path)
    behavior = pd.read_csv(
        behavior_path,
        usecols=["Stimulus", "Subject", "Experiment", "MeasurementValue", "Value"],
        low_memory=False,
    )
    raw_behavior_rows = len(behavior)
    required_molecule_columns = {"CID", "CanonicalSMILES"}
    required_stimulus_columns = {"Stimulus", "CIDs", "Concentration"}
    if not required_molecule_columns.issubset(molecules.columns):
        raise RuntimeError("Keller molecule columns changed")
    if not required_stimulus_columns.issubset(stimuli.columns):
        raise RuntimeError("Keller stimulus columns changed")

    cid_to_smiles: dict[str, str] = {}
    for _, row in molecules.iterrows():
        cid = _identifier(row["CID"])
        raw = str(row["CanonicalSMILES"]).strip()
        if not cid or not raw or raw.lower() == "nan":
            continue
        canonical = _canonical_smiles(raw)
        previous = cid_to_smiles.get(cid)
        if previous is not None and previous != canonical:
            raise RuntimeError(f"Keller CID {cid} maps to two structures")
        cid_to_smiles[cid] = canonical

    stimuli = stimuli.copy()
    stimuli["cid"] = stimuli["CIDs"].map(_identifier)
    stimuli["Stimulus"] = stimuli["Stimulus"].astype(int)
    stimuli["Concentration"] = pd.to_numeric(stimuli["Concentration"], errors="raise")
    highest = (
        stimuli.sort_values(
            ["cid", "Concentration", "Stimulus"],
            ascending=[True, False, True],
        )
        .drop_duplicates("cid", keep="first")
        .set_index("Stimulus")
    )
    selected_stimuli = set(int(value) for value in highest.index)

    measurement_to_endpoint = {
        contract["keller"]: endpoint for endpoint, contract in ENDPOINTS.items()
    }
    behavior = behavior[
        behavior["MeasurementValue"].isin(measurement_to_endpoint)
        & behavior["Stimulus"].astype(int).isin(selected_stimuli)
    ].copy()
    behavior["Stimulus"] = behavior["Stimulus"].astype(int)
    behavior["Subject"] = behavior["Subject"].map(_identifier)
    behavior["numeric"] = pd.to_numeric(behavior["Value"], errors="coerce")
    pivot = behavior.pivot_table(
        index=["Stimulus", "Subject"],
        columns="MeasurementValue",
        values="numeric",
        aggfunc="mean",
        dropna=False,
    )
    intensity_name = ENDPOINTS["intensive"]["keller"]
    pivot = pivot[pivot[intensity_name].notna()].copy()
    for endpoint, contract in ENDPOINTS.items():
        measurement = contract["keller"]
        if measurement not in pivot:
            raise RuntimeError(f"Keller endpoint is missing: {measurement}")
        if endpoint not in {"intensive", "pleasant", "familiar"}:
            # Keller's descriptor sliders defaulted to zero; empty spreadsheet
            # cells therefore mean that the descriptor was not applied.
            pivot[measurement] = pivot[measurement].fillna(0.0)
    pivot = pivot.dropna(
        subset=[
            ENDPOINTS["intensive"]["keller"],
            ENDPOINTS["pleasant"]["keller"],
            ENDPOINTS["familiar"]["keller"],
        ]
    )
    renamed = pivot.rename(columns=measurement_to_endpoint)
    odor_means = renamed[list(ENDPOINTS)].groupby(level="Stimulus").mean()

    rows: list[dict[str, Any]] = []
    for stimulus, values in odor_means.iterrows():
        cid = str(highest.loc[int(stimulus), "cid"])
        canonical = cid_to_smiles.get(cid)
        if canonical is None:
            continue
        row = {
            "canonical_smiles": canonical,
            "cid": cid,
            "stimulus": int(stimulus),
            **{endpoint: float(values[endpoint]) for endpoint in ENDPOINTS},
        }
        rows.append(row)
    if len(rows) < 450:
        raise RuntimeError("too few usable Keller molecules")

    # A few aliases can map to the same canonical structure. Pool those before
    # splitting so a chemical structure cannot cross a fold boundary.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["canonical_smiles"], []).append(row)
    ordered = sorted(grouped)
    matrix = np.asarray(
        [
            [
                float(np.mean([entry[endpoint] for entry in grouped[smiles]]))
                for endpoint in ENDPOINTS
            ]
            for smiles in ordered
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 100.0)):
        raise RuntimeError("Keller aggregate values are invalid")
    reference = {
        smiles: {
            endpoint: float(matrix[index, endpoint_index])
            for endpoint_index, endpoint in enumerate(ENDPOINTS)
        }
        for index, smiles in enumerate(ordered)
    }
    audit = {
        "raw_behavior_rows": int(raw_behavior_rows),
        "highest_concentration_stimuli": len(selected_stimuli),
        "usable_canonical_molecules": len(ordered),
        "participant_stimulus_rows": int(len(pivot)),
        "subjects": int(behavior["Subject"].nunique()),
        "aggregation": (
            "highest concentration per CID; participant means; blank Keller "
            "semantic slider values treated as the documented zero default"
        ),
    }
    return ordered, matrix, reference, audit


def _morgan_matrix(smiles: Sequence[str]) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid canonical SMILES: {value}")
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        rows.append(array)
    return np.asarray(rows, dtype=np.float32)


def _feature_matrices(
    smiles: Sequence[str],
    *,
    molformer_root: Path,
    hf_home: Path,
    batch_size: int,
    torch_threads: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from fragrance_ai.research.r2_physsim import build_raw_descriptor_cache
    from scripts.benchmark_huggingface_olfaction import encode_molformer

    ordered = list(smiles)
    raw = build_raw_descriptor_cache(ordered)
    rdkit_matrix = np.asarray([raw[value] for value in ordered], dtype=np.float64)
    morgan_matrix = _morgan_matrix(ordered).astype(np.float64)
    embeddings, molformer_audit = encode_molformer(
        molformer_root,
        ordered,
        batch_size=batch_size,
        torch_threads=torch_threads,
        hf_home=hf_home,
    )
    molformer_matrix = np.asarray(
        [embeddings[value] for value in ordered], dtype=np.float64
    )
    matrices = {
        "rdkit": rdkit_matrix,
        "morgan": morgan_matrix,
        "molformer": molformer_matrix,
        "fusion": np.concatenate((rdkit_matrix, molformer_matrix), axis=1),
    }
    if any(not np.all(np.isfinite(value)) for value in matrices.values()):
        raise RuntimeError("molecular features contain non-finite values")
    return matrices, {
        "rdkit_descriptor_count": int(rdkit_matrix.shape[1]),
        "morgan_bits": int(morgan_matrix.shape[1]),
        "molformer": molformer_audit,
    }


def _fit_predict_candidate(
    specification: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    training_indices: np.ndarray,
    target_indices: np.ndarray,
    outcomes: np.ndarray,
    *,
    seed: int,
    threads: int,
) -> np.ndarray:
    feature_name = str(specification["feature"])
    training = features[feature_name][training_indices]
    target = features[feature_name][target_indices]
    training_outcomes = outcomes[training_indices]
    algorithm = specification["algorithm"]
    if algorithm == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(), Ridge(alpha=float(specification["alpha"]))
        )
        estimator.fit(training, training_outcomes)
        return np.asarray(estimator.predict(target), dtype=float)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.7,
            min_samples_leaf=int(specification["min_samples_leaf"]),
            random_state=seed,
            n_jobs=threads,
        )
        estimator.fit(training, training_outcomes)
        return np.asarray(estimator.predict(target), dtype=float)
    if algorithm == "similarity_knn":
        train_bits = training > 0.5
        target_bits = target > 0.5
        intersection = target_bits.astype(float) @ train_bits.astype(float).T
        denominator = (
            target_bits.sum(axis=1, keepdims=True)
            + train_bits.sum(axis=1)[None, :]
            - intersection
        )
        similarities = np.divide(
            intersection,
            denominator,
            out=np.zeros_like(intersection),
            where=denominator > 0,
        )
        neighbors = min(int(specification["neighbors"]), len(training_indices))
        selected = np.argpartition(-similarities, neighbors - 1, axis=1)[:, :neighbors]
        predictions = np.empty((len(target_indices), outcomes.shape[1]), dtype=float)
        for index, nearest in enumerate(selected):
            weights = np.maximum(similarities[index, nearest], 1e-6) ** 3
            predictions[index] = np.average(
                training_outcomes[nearest], axis=0, weights=weights
            )
        return predictions
    raise KeyError(f"unknown candidate algorithm: {algorithm}")


def _fold_assignments(smiles: Sequence[str]) -> np.ndarray:
    ordered = sorted(
        range(len(smiles)),
        key=lambda index: hashlib.sha256(
            f"{CV_SALT}|{smiles[index]}".encode("utf-8")
        ).hexdigest(),
    )
    result = np.empty(len(smiles), dtype=int)
    for position, index in enumerate(ordered):
        result[index] = position % CV_FOLDS
    return result


def _endpoint_scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        endpoint: spearman(prediction[:, index], target[:, index])
        for index, endpoint in enumerate(ENDPOINTS)
    }


def _macro(values: Mapping[str, float]) -> float:
    return float(np.mean([float(values[endpoint]) for endpoint in ENDPOINTS]))


def _candidate_development(
    features: Mapping[str, np.ndarray],
    training_indices: np.ndarray,
    outcomes: np.ndarray,
    training_smiles: Sequence[str],
    *,
    threads: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    local_features = {
        name: values[training_indices] for name, values in features.items()
    }
    local_outcomes = outcomes[training_indices]
    folds = _fold_assignments(training_smiles)
    oof = {
        str(specification["name"]): np.full_like(local_outcomes, np.nan, dtype=float)
        for specification in CANDIDATES
    }
    for fold in range(CV_FOLDS):
        fit_indices = np.flatnonzero(folds != fold)
        validation_indices = np.flatnonzero(folds == fold)
        if len(validation_indices) == 0:
            raise RuntimeError("empty Keller cross-validation fold")
        for candidate_index, specification in enumerate(CANDIDATES):
            prediction = _fit_predict_candidate(
                specification,
                local_features,
                fit_indices,
                validation_indices,
                local_outcomes,
                seed=BOOTSTRAP_SEED + fold * 100 + candidate_index,
                threads=threads,
            )
            oof[str(specification["name"])][validation_indices] = prediction
    if any(not np.all(np.isfinite(value)) for value in oof.values()):
        raise RuntimeError("incomplete Keller out-of-fold predictions")
    candidate_scores = {
        name: {
            "endpoint_spearman": _endpoint_scores(prediction, local_outcomes),
        }
        for name, prediction in oof.items()
    }
    for row in candidate_scores.values():
        row["macro_endpoint_spearman"] = _macro(row["endpoint_spearman"])
    selected_by_endpoint: dict[str, str] = {}
    for endpoint in ENDPOINTS:
        selected_by_endpoint[endpoint] = max(
            candidate_scores,
            key=lambda name: (
                candidate_scores[name]["endpoint_spearman"][endpoint],
                -next(
                    index
                    for index, item in enumerate(CANDIDATES)
                    if item["name"] == name
                ),
            ),
        )
    primary_oof = np.column_stack(
        [
            oof[selected_by_endpoint[endpoint]][:, index]
            for index, endpoint in enumerate(ENDPOINTS)
        ]
    )
    primary_scores = _endpoint_scores(primary_oof, local_outcomes)
    global_best = max(
        candidate_scores,
        key=lambda name: (
            candidate_scores[name]["macro_endpoint_spearman"],
            -next(
                index
                for index, item in enumerate(CANDIDATES)
                if item["name"] == name
            ),
        ),
    )
    development = {
        "protocol": "five_fold_deterministic_molecule_disjoint_keller_only",
        "folds": CV_FOLDS,
        "fold_salt_sha256": hashlib.sha256(CV_SALT.encode("utf-8")).hexdigest(),
        "training_molecules": len(training_smiles),
        "candidate_scores": candidate_scores,
        "selected_by_endpoint": selected_by_endpoint,
        "selected_candidate_counts": {
            name: sum(value == name for value in selected_by_endpoint.values())
            for name in sorted(set(selected_by_endpoint.values()))
        },
        "primary_endpoint_spearman": primary_scores,
        "primary_macro_endpoint_spearman": _macro(primary_scores),
        "fixed_baseline": FIXED_BASELINE,
        "fixed_baseline_macro_endpoint_spearman": candidate_scores[FIXED_BASELINE][
            "macro_endpoint_spearman"
        ],
        "primary_minus_fixed_baseline": (
            _macro(primary_scores)
            - candidate_scores[FIXED_BASELINE]["macro_endpoint_spearman"]
        ),
        "global_best_candidate": global_best,
    }
    return development, oof, selected_by_endpoint


def _full_candidate_predictions(
    features: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    training_indices: np.ndarray,
    target_indices: np.ndarray,
    candidate_names: Sequence[str],
    *,
    seed_offset: int,
    threads: int,
) -> dict[str, np.ndarray]:
    specifications = {str(item["name"]): item for item in CANDIDATES}
    result = {}
    for index, name in enumerate(sorted(set(candidate_names))):
        result[name] = _fit_predict_candidate(
            specifications[name],
            features,
            training_indices,
            target_indices,
            outcomes,
            seed=BOOTSTRAP_SEED + seed_offset + index,
            threads=threads,
        )
    return result


def _selected_matrix(
    predictions: Mapping[str, np.ndarray], selected_by_endpoint: Mapping[str, str]
) -> np.ndarray:
    return np.column_stack(
        [
            predictions[selected_by_endpoint[endpoint]][:, index]
            for index, endpoint in enumerate(ENDPOINTS)
        ]
    )


def _asset_hashes(root: Path) -> dict[str, str]:
    names = (
        "config.json",
        "model.safetensors",
        "modeling_molformer.py",
        "configuration_molformer.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen MolFormer asset: {path}")
        result[name] = sha256_file(path)
    return result


def assert_target_outcomes_absent(path: Path) -> None:
    parent = path.resolve().parent
    present = [
        name for name in ("data.csv", "data.xlsx") if (parent / name).exists()
    ]
    if present:
        raise RuntimeError(
            "target human outcome exists before the permitted acquisition: "
            + ",".join(present)
        )


def prepare_predictions(args: argparse.Namespace) -> dict[str, Any]:
    forbidden = args.forbidden_outcome.resolve()
    assert_target_outcomes_absent(forbidden)
    if args.predictions.resolve().exists():
        raise RuntimeError("refusing to overwrite existing blind predictions")
    target_odors_path = args.target_odors.resolve(strict=True)
    variable_dictionary = args.variable_dictionary.resolve(strict=True)
    if (
        target_odors_path.stat().st_size != TARGET_METADATA_BYTES
        or sha256_file(target_odors_path) != TARGET_METADATA_SHA256
    ):
        raise RuntimeError("Bierling target odor metadata differs from Zenodo record")
    if (
        variable_dictionary.stat().st_size != VARIABLE_DICTIONARY_BYTES
        or sha256_file(variable_dictionary) != VARIABLE_DICTIONARY_SHA256
    ):
        raise RuntimeError("Bierling variable dictionary differs from Zenodo record")
    target_rows = load_target_odors(target_odors_path)
    target_smiles = [row["canonical_smiles"] for row in target_rows]
    target_set = set(target_smiles)
    target_ring_scaffolds = {
        row["murcko_scaffold"] for row in target_rows if row["murcko_scaffold"]
    }

    keller_smiles, keller_outcomes, keller_reference, keller_audit = _keller_means(
        args.keller_molecules.resolve(strict=True),
        args.keller_stimuli.resolve(strict=True),
        args.keller_behavior.resolve(strict=True),
    )
    exact_training_smiles = [value for value in keller_smiles if value not in target_set]
    strict_training_smiles = [
        value
        for value in exact_training_smiles
        if not _murcko_scaffold(value)
        or _murcko_scaffold(value) not in target_ring_scaffolds
    ]
    if len(exact_training_smiles) < 300 or len(strict_training_smiles) < 150:
        raise RuntimeError("target-excluded Keller training set is too small")

    all_smiles = sorted(set(keller_smiles) | target_set)
    global_index = {value: index for index, value in enumerate(all_smiles)}
    keller_index = {value: index for index, value in enumerate(keller_smiles)}
    reordered_outcomes = np.asarray(
        [keller_outcomes[keller_index[value]] for value in keller_smiles], dtype=float
    )
    # Feature arrays and outcome arrays share all_smiles indexing. Target rows
    # carry NaN outcomes and can never enter a fit index.
    outcomes = np.full((len(all_smiles), len(ENDPOINTS)), np.nan, dtype=float)
    for smiles, row in zip(keller_smiles, reordered_outcomes, strict=True):
        outcomes[global_index[smiles]] = row

    features, feature_audit = _feature_matrices(
        all_smiles,
        molformer_root=args.molformer_root.resolve(strict=True),
        hf_home=args.hf_home.resolve(),
        batch_size=args.batch_size,
        torch_threads=args.threads,
    )
    exact_indices = np.asarray(
        [global_index[value] for value in exact_training_smiles], dtype=int
    )
    strict_indices = np.asarray(
        [global_index[value] for value in strict_training_smiles], dtype=int
    )
    target_indices = np.asarray(
        [global_index[value] for value in target_smiles], dtype=int
    )
    development, _, selected_by_endpoint = _candidate_development(
        features,
        exact_indices,
        outcomes,
        exact_training_smiles,
        threads=args.threads,
    )
    candidate_names = set(selected_by_endpoint.values()) | {
        FIXED_BASELINE,
        development["global_best_candidate"],
    }
    exact_predictions = _full_candidate_predictions(
        features,
        outcomes,
        exact_indices,
        target_indices,
        sorted(candidate_names),
        seed_offset=10_000,
        threads=args.threads,
    )
    strict_predictions = _full_candidate_predictions(
        features,
        outcomes,
        strict_indices,
        target_indices,
        sorted(set(selected_by_endpoint.values())),
        seed_offset=20_000,
        threads=args.threads,
    )
    primary = np.clip(
        _selected_matrix(exact_predictions, selected_by_endpoint), 0.0, 100.0
    )
    strict_primary = np.clip(
        _selected_matrix(strict_predictions, selected_by_endpoint), 0.0, 100.0
    )
    fixed_baseline = np.clip(exact_predictions[FIXED_BASELINE], 0.0, 100.0)
    global_best = np.clip(
        exact_predictions[development["global_best_candidate"]], 0.0, 100.0
    )
    if any(
        not np.all(np.isfinite(value))
        for value in (primary, strict_primary, fixed_baseline, global_best)
    ):
        raise RuntimeError("target predictions are non-finite")

    prediction_rows = []
    for row_index, target in enumerate(target_rows):
        reference = keller_reference.get(target["canonical_smiles"])
        prediction_rows.append(
            {
                **target,
                "primary_target_exact_label_excluded": {
                    endpoint: round(float(primary[row_index, endpoint_index]), 8)
                    for endpoint_index, endpoint in enumerate(ENDPOINTS)
                },
                "strict_target_ring_scaffold_label_excluded": {
                    endpoint: round(
                        float(strict_primary[row_index, endpoint_index]), 8
                    )
                    for endpoint_index, endpoint in enumerate(ENDPOINTS)
                },
                "fixed_rdkit_baseline": {
                    endpoint: round(
                        float(fixed_baseline[row_index, endpoint_index]), 8
                    )
                    for endpoint_index, endpoint in enumerate(ENDPOINTS)
                },
                "global_best_candidate": {
                    endpoint: round(float(global_best[row_index, endpoint_index]), 8)
                    for endpoint_index, endpoint in enumerate(ENDPOINTS)
                },
                "keller_cross_population_reference": (
                    {
                        endpoint: round(float(reference[endpoint]), 8)
                        for endpoint in ENDPOINTS
                    }
                    if reference is not None
                    else None
                ),
            }
        )
    prediction_rows_hash = canonical_json_sha256(prediction_rows)
    script_path = Path(__file__).resolve()
    training_sources = {
        "molecules.csv": {
            "path": str(args.keller_molecules.resolve()),
            "sha256": sha256_file(args.keller_molecules.resolve(strict=True)),
            "bytes": args.keller_molecules.resolve().stat().st_size,
        },
        "stimuli.csv": {
            "path": str(args.keller_stimuli.resolve()),
            "sha256": sha256_file(args.keller_stimuli.resolve(strict=True)),
            "bytes": args.keller_stimuli.resolve().stat().st_size,
        },
        "behavior.csv": {
            "path": str(args.keller_behavior.resolve()),
            "sha256": sha256_file(args.keller_behavior.resolve(strict=True)),
            "bytes": args.keller_behavior.resolve().stat().st_size,
        },
    }
    for name, expected in KELLER_SOURCE_CONTRACT.items():
        actual = training_sources[name]
        if (
            actual["sha256"] != expected["sha256"]
            or actual["bytes"] != expected["bytes"]
        ):
            raise RuntimeError(f"Keller source differs from frozen contract: {name}")
    release_checks = {
        "target_outcome_absent": not forbidden.exists(),
        "target_count_exact": len(target_rows) == TARGET_ODOR_COUNT,
        "target_exact_human_label_leakage_zero": not bool(
            target_set & set(exact_training_smiles)
        ),
        "development_molecule_disjoint_folds": True,
        "development_primary_beats_fixed_rdkit_by_0_01": (
            development["primary_minus_fixed_baseline"] >= 0.01
        ),
        "strict_ring_scaffold_training_minimum_150": len(strict_training_smiles)
        >= 150,
        "target_predictions_finite": True,
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "blind_predictions_ready_before_target_outcomes",
        "blind_contract": {
            "target_outcome_file_read": False,
            "prepare_command_can_read_target_outcome": False,
            "target_outcomes_used_for_training": False,
            "target_outcomes_used_for_model_selection": False,
            "target_outcomes_used_for_endpoint_selection": False,
            "model_selection_source": "Keller 2016 human outcomes only",
            "target_exact_molecules_excluded_from_human_supervision": True,
            "predeclared_primary_metric": (
                "macro mean of 22 endpoint-wise cross-odor Spearman correlations"
            ),
            "predeclared_population": (
                "Bierling main-study included healthy home/lab participants; "
                "patients and retest excluded"
            ),
            "predeclared_uncertainty": (
                "1000-draw participant-cluster plus odor bootstrap"
            ),
            "prediction_rows_sha256": prediction_rows_hash,
        },
        "target_dataset": {
            "name": "Bierling et al. 2025 laypeople olfactory perception",
            "zenodo_record_id": DATASET_RECORD_ID,
            "doi": DATASET_DOI,
            "article_doi": DATASET_ARTICLE_DOI,
            "license": "CC-BY-4.0",
            "odor_metadata": {
                "path": str(target_odors_path),
                "sha256": sha256_file(target_odors_path),
                "bytes": target_odors_path.stat().st_size,
            },
            "variable_dictionary": {
                "path": str(variable_dictionary),
                "sha256": sha256_file(variable_dictionary),
                "bytes": variable_dictionary.stat().st_size,
            },
            "expected_outcome": {
                "filename": OUTCOME_FILE,
                "url": OUTCOME_URL,
                "bytes": OUTCOME_BYTES,
                "md5": OUTCOME_MD5,
                "downloaded_or_opened": False,
            },
            "odors": len(target_rows),
            "endpoints": list(ENDPOINTS),
        },
        "training_dataset": {
            "name": "Keller and Vosshall 2016",
            "doi": "10.1186/s12868-016-0287-2",
            "sources": training_sources,
            "aggregation_audit": keller_audit,
            "all_usable_molecules": len(keller_smiles),
            "exact_target_overlap_excluded": len(target_set & set(keller_smiles)),
            "exact_label_disjoint_training_molecules": len(exact_training_smiles),
            "ring_scaffold_and_exact_label_disjoint_training_molecules": len(
                strict_training_smiles
            ),
            "target_ring_scaffolds": len(target_ring_scaffolds),
        },
        "model": {
            "name": "HumanPOM target-excluded public-human ensemble v1",
            "feature_contract": feature_audit,
            "candidate_contract": list(CANDIDATES),
            "development": development,
            "molformer_asset_sha256": _asset_hashes(
                args.molformer_root.resolve(strict=True)
            ),
            "primary_prediction": "primary_target_exact_label_excluded",
            "fixed_comparator": "fixed_rdkit_baseline",
            "strict_sensitivity": "strict_target_ring_scaffold_label_excluded",
            "cross_population_human_reference": (
                "Keller highest-concentration means for exact overlapping molecules"
            ),
        },
        "implementation": {
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "huggingface_adapter_script_sha256": sha256_file(
                PROJECT_ROOT / "scripts" / "benchmark_huggingface_olfaction.py"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "release_gate": {
            "passed": all(release_checks.values()),
            "checks": release_checks,
            "scope": "permission to seal target predictions only",
        },
        "predictions": prediction_rows,
        "claim_boundary": (
            "Outcome-unopened monomolecular population-mean descriptor prediction. "
            "Not mixture, recipe, product-matrix, or 90% human olfactory validation."
        ),
    }
    if not document["release_gate"]["passed"]:
        raise RuntimeError(f"prediction release gate failed: {release_checks}")
    write_json(args.predictions.resolve(), document)
    return document


def create_seal(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve(strict=True)
    outcome_path = args.outcome.resolve()
    assert_target_outcomes_absent(outcome_path)
    if args.seal.resolve().exists():
        raise RuntimeError("refusing to overwrite an existing prediction seal")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if predictions.get("status") != "blind_predictions_ready_before_target_outcomes":
        raise RuntimeError("prediction artifact is not sealable")
    if predictions.get("release_gate", {}).get("passed") is not True:
        raise RuntimeError("prediction artifact release gate is closed")
    rows = predictions.get("predictions", [])
    rows_hash = canonical_json_sha256(rows)
    if rows_hash != predictions.get("blind_contract", {}).get(
        "prediction_rows_sha256"
    ):
        raise RuntimeError("prediction row hash changed")
    script_hash = sha256_file(Path(__file__).resolve())
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("benchmark script changed after prediction")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": utc_now(),
        "prediction_file": predictions_path.name,
        "prediction_file_sha256": sha256_file(predictions_path),
        "prediction_file_bytes": predictions_path.stat().st_size,
        "prediction_rows_sha256": rows_hash,
        "benchmark_script_sha256": script_hash,
        "target_outcome": {
            "filename": OUTCOME_FILE,
            "url": OUTCOME_URL,
            "expected_bytes": OUTCOME_BYTES,
            "expected_md5": OUTCOME_MD5,
            "path": str(outcome_path),
            "present_before_seal": False,
        },
        "scoring_contract": {
            "population": (
                "study=main, inclusion=1, sampling_group in {home,lab}"
            ),
            "odors": TARGET_ODOR_COUNT,
            "endpoints": list(ENDPOINTS),
            "primary_metric": (
                "macro mean of endpoint-wise cross-odor Spearman"
            ),
            "primary_model": "primary_target_exact_label_excluded",
            "fixed_comparator": "fixed_rdkit_baseline",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "reliability_repeats": RELIABILITY_REPEATS,
        },
    }
    write_json(args.seal.resolve(), seal)
    return seal


def verify_prediction_seal(predictions_path: Path, seal_path: Path) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported seal schema")
    if seal.get("prediction_file_sha256") != sha256_file(predictions_path):
        raise RuntimeError("sealed prediction file hash mismatch")
    if seal.get("prediction_file_bytes") != predictions_path.stat().st_size:
        raise RuntimeError("sealed prediction file size mismatch")
    rows_hash = canonical_json_sha256(predictions.get("predictions", []))
    if rows_hash != seal.get("prediction_rows_sha256"):
        raise RuntimeError("sealed prediction rows mismatch")
    script_hash = sha256_file(Path(__file__).resolve())
    if script_hash != seal.get("benchmark_script_sha256"):
        raise RuntimeError("benchmark script changed after seal")
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("prediction implementation hash mismatch")
    expected = seal.get("target_outcome", {})
    if expected != {
        "filename": OUTCOME_FILE,
        "url": OUTCOME_URL,
        "expected_bytes": OUTCOME_BYTES,
        "expected_md5": OUTCOME_MD5,
        "path": expected.get("path"),
        "present_before_seal": False,
    }:
        raise RuntimeError("sealed target outcome contract changed")
    return {"predictions": predictions, "seal": seal}


def verify_rfc3161_timestamp(
    *,
    openssl: Path,
    seal_path: Path,
    response_path: Path,
    ca_path: Path,
    tsa_path: Path,
) -> dict[str, Any]:
    command = [
        str(openssl.resolve(strict=True)),
        "ts",
        "-verify",
        "-data",
        str(seal_path.resolve(strict=True)),
        "-in",
        str(response_path.resolve(strict=True)),
        "-CAfile",
        str(ca_path.resolve(strict=True)),
        "-untrusted",
        str(tsa_path.resolve(strict=True)),
    ]
    verification = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if verification.returncode != 0 or "Verification: OK" not in (
        verification.stdout + verification.stderr
    ):
        raise RuntimeError(
            "RFC3161 timestamp verification failed: "
            + (verification.stdout + verification.stderr)[-1000:]
        )
    reply = subprocess.run(
        [
            str(openssl.resolve(strict=True)),
            "ts",
            "-reply",
            "-in",
            str(response_path.resolve(strict=True)),
            "-text",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    match = re.search(r"^Time stamp:\s*(.+)$", reply.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("RFC3161 reply has no timestamp")
    return {
        "verified": True,
        "time_stamp": match.group(1).strip(),
        "response_sha256": sha256_file(response_path),
        "ca_sha256": sha256_file(ca_path),
        "tsa_sha256": sha256_file(tsa_path),
        "reply_text_sha256": hashlib.sha256(reply.stdout.encode("utf-8")).hexdigest(),
    }


def acquire_outcome(args: argparse.Namespace) -> dict[str, Any]:
    verified = verify_prediction_seal(args.predictions, args.seal)
    outcome_path = args.outcome.resolve()
    sealed_outcome_path = Path(
        verified["seal"]["target_outcome"]["path"]
    ).resolve()
    if outcome_path != sealed_outcome_path:
        raise RuntimeError("acquisition outcome path differs from sealed path")
    assert_target_outcomes_absent(outcome_path)
    if args.receipt.resolve().exists():
        raise RuntimeError("refusing to overwrite an outcome acquisition receipt")
    timestamp = verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    started = utc_now()
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="bierling-outcome-", suffix=".part", dir=outcome_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            OUTCOME_URL,
            headers={"User-Agent": "Perfumery-AI blind benchmark/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if temporary.stat().st_size != OUTCOME_BYTES:
            raise RuntimeError("downloaded outcome byte count changed")
        if md5_file(temporary) != OUTCOME_MD5:
            raise RuntimeError("downloaded outcome MD5 differs from Zenodo metadata")
        if outcome_path.exists():
            raise RuntimeError("target outcome appeared during acquisition")
        os.rename(temporary, outcome_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "download_started_at": started,
        "download_completed_at": utc_now(),
        "url": OUTCOME_URL,
        "zenodo_record_id": DATASET_RECORD_ID,
        "outcome_path": str(outcome_path),
        "outcome_bytes": outcome_path.stat().st_size,
        "outcome_md5": md5_file(outcome_path),
        "outcome_sha256": sha256_file(outcome_path),
        "prediction_sha256": sha256_file(args.predictions.resolve(strict=True)),
        "seal_sha256": sha256_file(args.seal.resolve(strict=True)),
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "timestamp": timestamp,
        "prediction_status": verified["predictions"]["status"],
    }
    write_json(args.receipt.resolve(), receipt)
    return receipt


def _parse_binary_series(series: Any) -> np.ndarray:
    mapping = {
        "1": 1.0,
        "1.0": 1.0,
        "true": 1.0,
        "yes": 1.0,
        "ja": 1.0,
        "0": 0.0,
        "0.0": 0.0,
        "false": 0.0,
        "no": 0.0,
        "nein": 0.0,
        "": np.nan,
        "nan": np.nan,
        "none": np.nan,
    }
    values = []
    for value in series:
        key = str(value).strip().lower()
        if key not in mapping:
            raise RuntimeError(f"unexpected binary odor rating: {value!r}")
        values.append(mapping[key])
    return np.asarray(values, dtype=float)


def _load_target_outcomes(
    path: Path, target_rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path, sep=None, engine="python")
    required = {
        "inclusion",
        "study",
        "sampling_group",
        "code",
        "molcode",
        *ENDPOINTS.keys(),
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise RuntimeError(f"target outcome columns changed: {missing}")
    inclusion = pd.to_numeric(frame["inclusion"], errors="coerce") == 1
    selected = frame[
        inclusion
        & frame["study"].astype(str).str.strip().str.lower().eq("main")
        & frame["sampling_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"home", "lab"})
    ].copy()
    selected["code"] = selected["code"].map(_identifier)
    selected["molcode"] = selected["molcode"].astype(str).str.strip()
    target_codes = [str(row["molcode"]) for row in target_rows]
    selected = selected[selected["molcode"].isin(target_codes)].copy()
    if selected.duplicated(["code", "molcode"]).any():
        raise RuntimeError("target main population has duplicate participant-odor rows")
    for endpoint, contract in ENDPOINTS.items():
        if contract["kind"] == "binary":
            selected[endpoint] = _parse_binary_series(selected[endpoint]) * 100.0
        else:
            selected[endpoint] = pd.to_numeric(selected[endpoint], errors="coerce")
            nonmissing = selected[endpoint].dropna()
            if ((nonmissing < 1.0) | (nonmissing > 100.0)).any():
                raise RuntimeError(f"target VAS endpoint outside 1..100: {endpoint}")
    counts = selected.groupby("molcode").size()
    if set(counts.index) != set(target_codes) or int(counts.min()) < 80:
        raise RuntimeError("target odor population coverage is incomplete")
    means = selected.groupby("molcode")[list(ENDPOINTS)].mean()
    matrix = np.asarray(
        [means.loc[code, list(ENDPOINTS)].to_numpy(dtype=float) for code in target_codes],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("target population means are non-finite")
    return matrix, selected, {
        "raw_rows": int(len(frame)),
        "included_main_healthy_rows": int(len(selected)),
        "participants": int(selected["code"].nunique()),
        "odors": int(selected["molcode"].nunique()),
        "minimum_ratings_per_odor": int(counts.min()),
        "maximum_ratings_per_odor": int(counts.max()),
        "sampling_groups": sorted(selected["sampling_group"].unique().tolist()),
    }


def _matrix_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> np.ndarray:
    return np.asarray(
        [
            [float(row[field][endpoint]) for endpoint in ENDPOINTS]
            for row in rows
        ],
        dtype=float,
    )


def _method_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    endpoint_rows = {}
    for index, endpoint in enumerate(ENDPOINTS):
        left = prediction[:, index]
        right = target[:, index]
        endpoint_rows[endpoint] = {
            "spearman": spearman(left, right),
            "pearson": _pearson(left, right),
            "mae_percentage_points": float(np.mean(np.abs(left - right))),
        }
    qualitative_indices = [list(ENDPOINTS).index(name) for name in QUALITATIVE_ENDPOINTS]
    profile_spearman = []
    top_three_recall = []
    for row in range(len(prediction)):
        predicted_profile = prediction[row, qualitative_indices]
        target_profile = target[row, qualitative_indices]
        profile_spearman.append(spearman(predicted_profile, target_profile))
        predicted_top = set(np.argsort(-predicted_profile, kind="stable")[:3])
        target_top = set(np.argsort(-target_profile, kind="stable")[:3])
        top_three_recall.append(len(predicted_top & target_top) / 3.0)
    return {
        "endpoint_metrics": endpoint_rows,
        "macro_endpoint_spearman": float(
            np.mean([row["spearman"] for row in endpoint_rows.values()])
        ),
        "median_endpoint_spearman": float(
            np.median([row["spearman"] for row in endpoint_rows.values()])
        ),
        "positive_endpoint_count": sum(
            row["spearman"] > 0.0 for row in endpoint_rows.values()
        ),
        "macro_mae_percentage_points": float(
            np.mean([row["mae_percentage_points"] for row in endpoint_rows.values()])
        ),
        "mean_qualitative_profile_spearman": float(np.mean(profile_spearman)),
        "mean_top_three_qualitative_recall": float(np.mean(top_three_recall)),
    }


def _participant_cube(selected: Any, target_codes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    participants = sorted(selected["code"].unique().tolist())
    participant_index = {value: index for index, value in enumerate(participants)}
    odor_index = {value: index for index, value in enumerate(target_codes)}
    cube = np.full(
        (len(participants), len(target_codes), len(ENDPOINTS)), np.nan, dtype=float
    )
    for _, row in selected.iterrows():
        cube[
            participant_index[row["code"]], odor_index[row["molcode"]]
        ] = np.asarray([row[endpoint] for endpoint in ENDPOINTS], dtype=float)
    return np.nan_to_num(cube, nan=0.0), np.isfinite(cube).astype(float)


def _two_way_bootstrap(
    selected: Any,
    target_codes: Sequence[str],
    primary: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    values, observed = _participant_cube(selected, target_codes)
    participant_count = values.shape[0]
    value_flat = values.reshape(participant_count, -1)
    observed_flat = observed.reshape(participant_count, -1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_scores = []
    baseline_scores = []
    endpoint_count = len(ENDPOINTS)
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(
            participant_count, np.full(participant_count, 1.0 / participant_count)
        ).astype(float)
        counts = weights @ observed_flat
        if np.any(counts <= 0):
            continue
        means = ((weights @ value_flat) / counts).reshape(
            len(target_codes), endpoint_count
        )
        sampled_odors = rng.integers(0, len(target_codes), size=len(target_codes))
        target_sample = means[sampled_odors]
        primary_endpoint = _endpoint_scores(primary[sampled_odors], target_sample)
        baseline_endpoint = _endpoint_scores(baseline[sampled_odors], target_sample)
        primary_scores.append(_macro(primary_endpoint))
        baseline_scores.append(_macro(baseline_endpoint))
    if len(primary_scores) < int(BOOTSTRAP_DRAWS * 0.99):
        raise RuntimeError("too many invalid two-way bootstrap draws")
    primary_array = np.asarray(primary_scores, dtype=float)
    baseline_array = np.asarray(baseline_scores, dtype=float)
    delta = primary_array - baseline_array
    return {
        "seed": BOOTSTRAP_SEED,
        "requested_draws": BOOTSTRAP_DRAWS,
        "valid_draws": len(primary_scores),
        "unit": "participant-cluster weights plus odor resampling",
        "primary_95_interval": [
            float(value) for value in np.quantile(primary_array, [0.025, 0.975])
        ],
        "fixed_baseline_95_interval": [
            float(value) for value in np.quantile(baseline_array, [0.025, 0.975])
        ],
        "primary_minus_baseline_95_interval": [
            float(value) for value in np.quantile(delta, [0.025, 0.975])
        ],
        "primary_win_probability": float(np.mean(delta > 0.0)),
    }


def _human_reliability(
    selected: Any, target_codes: Sequence[str]
) -> dict[str, Any]:
    values, observed = _participant_cube(selected, target_codes)
    participant_count = values.shape[0]
    value_flat = values.reshape(participant_count, -1)
    observed_flat = observed.reshape(participant_count, -1)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    for _ in range(RELIABILITY_REPEATS):
        assignment = rng.permutation(participant_count) < (participant_count // 2)
        first_weights = assignment.astype(float)
        second_weights = (~assignment).astype(float)
        first_counts = first_weights @ observed_flat
        second_counts = second_weights @ observed_flat
        if np.any(first_counts <= 0) or np.any(second_counts <= 0):
            continue
        first = ((first_weights @ value_flat) / first_counts).reshape(
            len(target_codes), len(ENDPOINTS)
        )
        second = ((second_weights @ value_flat) / second_counts).reshape(
            len(target_codes), len(ENDPOINTS)
        )
        correlations = np.asarray(
            [spearman(first[:, index], second[:, index]) for index in range(len(ENDPOINTS))]
        )
        corrected = np.clip(
            np.divide(
                2.0 * correlations,
                1.0 + correlations,
                out=np.zeros_like(correlations),
                where=np.abs(1.0 + correlations) > 1e-12,
            ),
            0.0,
            1.0,
        )
        rows.append(np.sqrt(corrected))
    matrix = np.asarray(rows, dtype=float)
    if len(matrix) < int(RELIABILITY_REPEATS * 0.95):
        raise RuntimeError("too many invalid human reliability splits")
    endpoint_ceiling = {
        endpoint: float(np.median(matrix[:, index]))
        for index, endpoint in enumerate(ENDPOINTS)
    }
    return {
        "repeats": len(matrix),
        "method": "participant split-half Spearman-Brown then square-root ceiling",
        "endpoint_ceiling": endpoint_ceiling,
        "macro_endpoint_ceiling": float(np.mean(list(endpoint_ceiling.values()))),
    }


def _score_cross_population_reference(
    rows: Sequence[Mapping[str, Any]], target: np.ndarray
) -> dict[str, Any]:
    indices = [
        index
        for index, row in enumerate(rows)
        if row["keller_cross_population_reference"] is not None
    ]
    reference = np.asarray(
        [
            [
                float(rows[index]["keller_cross_population_reference"][endpoint])
                for endpoint in ENDPOINTS
            ]
            for index in indices
        ],
        dtype=float,
    )
    result = _method_metrics(reference, target[indices])
    result["overlapping_odors"] = len(indices)
    result["scope"] = (
        "cross-population human-reference transfer on exact overlapping molecules; "
        "not a structure-model evaluation"
    )
    return result


def score_outcome(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.resolve().exists() or args.markdown.resolve().exists():
        raise RuntimeError("refusing to overwrite an existing blind score report")
    verified = verify_prediction_seal(args.predictions, args.seal)
    predictions = verified["predictions"]
    receipt = json.loads(args.receipt.resolve(strict=True).read_text(encoding="utf-8"))
    if receipt.get("seal_sha256") != sha256_file(args.seal.resolve(strict=True)):
        raise RuntimeError("outcome acquisition receipt is bound to another seal")
    if receipt.get("prediction_sha256") != sha256_file(
        args.predictions.resolve(strict=True)
    ):
        raise RuntimeError("outcome acquisition receipt prediction mismatch")
    outcome_path = args.outcome.resolve(strict=True)
    sealed_outcome_path = Path(
        verified["seal"]["target_outcome"]["path"]
    ).resolve()
    if outcome_path != sealed_outcome_path:
        raise RuntimeError("scoring outcome path differs from sealed path")
    if receipt.get("outcome_path") != str(outcome_path):
        raise RuntimeError("outcome acquisition receipt path mismatch")
    if (
        receipt.get("url") != OUTCOME_URL
        or receipt.get("zenodo_record_id") != DATASET_RECORD_ID
        or receipt.get("outcome_bytes") != OUTCOME_BYTES
        or receipt.get("outcome_md5") != OUTCOME_MD5
        or receipt.get("benchmark_script_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("outcome acquisition receipt contract changed")
    timestamp = verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if timestamp["response_sha256"] != receipt.get("timestamp", {}).get(
        "response_sha256"
    ):
        raise RuntimeError("timestamp response changed after acquisition")
    if (
        outcome_path.stat().st_size != OUTCOME_BYTES
        or md5_file(outcome_path) != OUTCOME_MD5
        or sha256_file(outcome_path) != receipt.get("outcome_sha256")
    ):
        raise RuntimeError("target outcome file differs from acquisition receipt")

    rows = predictions["predictions"]
    target_rows = [
        {
            "molcode": row["molcode"],
            "canonical_smiles": row["canonical_smiles"],
        }
        for row in rows
    ]
    target, selected, population_audit = _load_target_outcomes(
        outcome_path, target_rows
    )
    methods = {
        "primary_target_exact_label_excluded": _matrix_from_prediction_rows(
            rows, "primary_target_exact_label_excluded"
        ),
        "strict_target_ring_scaffold_label_excluded": _matrix_from_prediction_rows(
            rows, "strict_target_ring_scaffold_label_excluded"
        ),
        "fixed_rdkit_baseline": _matrix_from_prediction_rows(
            rows, "fixed_rdkit_baseline"
        ),
        "global_best_candidate": _matrix_from_prediction_rows(
            rows, "global_best_candidate"
        ),
    }
    method_results = {
        name: _method_metrics(prediction, target)
        for name, prediction in methods.items()
    }
    primary = methods["primary_target_exact_label_excluded"]
    baseline = methods["fixed_rdkit_baseline"]
    bootstrap = _two_way_bootstrap(
        selected,
        [row["molcode"] for row in rows],
        primary,
        baseline,
    )
    reliability = _human_reliability(selected, [row["molcode"] for row in rows])
    cross_population = _score_cross_population_reference(rows, target)
    primary_macro = method_results["primary_target_exact_label_excluded"][
        "macro_endpoint_spearman"
    ]
    baseline_macro = method_results["fixed_rdkit_baseline"][
        "macro_endpoint_spearman"
    ]
    improvement_checks = {
        "blind_integrity_verified": True,
        "all_74_target_odors_scored": population_audit["odors"]
        == TARGET_ODOR_COUNT,
        "primary_point_estimate_beats_fixed_rdkit": primary_macro
        > baseline_macro,
        "paired_two_way_bootstrap_delta_lower_above_zero": bootstrap[
            "primary_minus_baseline_95_interval"
        ][0]
        > 0.0,
        "at_least_15_of_22_endpoints_positive": method_results[
            "primary_target_exact_label_excluded"
        ]["positive_endpoint_count"]
        >= 15,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scored_at": utc_now(),
        "status": (
            "blind_external_improvement_confirmed"
            if all(improvement_checks.values())
            else "blind_external_evaluation_completed_without_full_improvement_gate"
        ),
        "blind_integrity": {
            "prediction_sha256": sha256_file(args.predictions.resolve(strict=True)),
            "seal_sha256": sha256_file(args.seal.resolve(strict=True)),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
            "target_outcome_sha256": sha256_file(outcome_path),
            "target_outcome_opened_only_after_verified_timestamp": True,
            "timestamp": timestamp,
            "acquisition_receipt_sha256": sha256_file(
                args.receipt.resolve(strict=True)
            ),
        },
        "dataset": {
            "name": predictions["target_dataset"]["name"],
            "doi": DATASET_DOI,
            "article_doi": DATASET_ARTICLE_DOI,
            "population": population_audit,
            "endpoints": list(ENDPOINTS),
        },
        "training_label_disjointness": predictions["training_dataset"],
        "development_only_model_selection": predictions["model"]["development"],
        "results": method_results,
        "cross_population_human_reference": cross_population,
        "human_reliability": reliability,
        "two_way_bootstrap": bootstrap,
        "primary_minus_fixed_rdkit_point_estimate": primary_macro
        - baseline_macro,
        "improvement_gate": {
            "passed": all(improvement_checks.values()),
            "checks": improvement_checks,
        },
        "human_olfactory_90_percent_certified": False,
        "generated_recipe_similarity_validated": False,
        "mixture_similarity_validated": False,
        "claim_boundary": (
            "Behavior-label-unopened external validation of 22 population-mean "
            "monomolecular endpoints. It is not a perfume recipe, mixture, product "
            "matrix, individual-person, or 90% olfactory-similarity result."
        ),
    }
    write_json(args.report.resolve(), report)
    _write_markdown(args.markdown.resolve(), report)
    return report


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    results = report["results"]
    primary = results["primary_target_exact_label_excluded"]
    baseline = results["fixed_rdkit_baseline"]
    strict = results["strict_target_ring_scaffold_label_excluded"]
    human = report["cross_population_human_reference"]
    bootstrap = report["two_way_bootstrap"]
    endpoint_lines = []
    for endpoint in ENDPOINTS:
        endpoint_lines.append(
            "| "
            + endpoint
            + " | "
            + f"{primary['endpoint_metrics'][endpoint]['spearman']:.4f}"
            + " | "
            + f"{baseline['endpoint_metrics'][endpoint]['spearman']:.4f}"
            + " |"
        )
    text = "\n".join(
        [
            "# Bierling 2025 공개 인간 후각 outcome-unopened 블라인드 검증",
            "",
            "목표 인간 결과 파일을 내려받기 전에 모델·예측·지표·대상 모집단을 고정하고 RFC 3161 외부 시각으로 봉인했습니다.",
            "",
            "| 평가 | Macro endpoint Spearman | 양의 endpoint | 정성 profile Spearman | Top-3 recall |",
            "|---|---:|---:|---:|---:|",
            (
                "| HumanPOM primary | "
                f"{primary['macro_endpoint_spearman']:.4f} | "
                f"{primary['positive_endpoint_count']}/22 | "
                f"{primary['mean_qualitative_profile_spearman']:.4f} | "
                f"{primary['mean_top_three_qualitative_recall']:.4f} |"
            ),
            (
                "| 고정 RDKit ridge | "
                f"{baseline['macro_endpoint_spearman']:.4f} | "
                f"{baseline['positive_endpoint_count']}/22 | "
                f"{baseline['mean_qualitative_profile_spearman']:.4f} | "
                f"{baseline['mean_top_three_qualitative_recall']:.4f} |"
            ),
            (
                "| Ring-scaffold sensitivity | "
                f"{strict['macro_endpoint_spearman']:.4f} | "
                f"{strict['positive_endpoint_count']}/22 | "
                f"{strict['mean_qualitative_profile_spearman']:.4f} | "
                f"{strict['mean_top_three_qualitative_recall']:.4f} |"
            ),
            (
                "| Keller 인간-인간 교차집단 기준 | "
                f"{human['macro_endpoint_spearman']:.4f} | "
                f"{human['positive_endpoint_count']}/22 | "
                f"{human['mean_qualitative_profile_spearman']:.4f} | "
                f"{human['mean_top_three_qualitative_recall']:.4f} |"
            ),
            "",
            (
                "Primary-RDKit 차이는 "
                f"{report['primary_minus_fixed_rdkit_point_estimate']:+.4f}, "
                "participant×odor bootstrap 95% 구간은 "
                f"[{bootstrap['primary_minus_baseline_95_interval'][0]:+.4f}, "
                f"{bootstrap['primary_minus_baseline_95_interval'][1]:+.4f}]입니다."
            ),
            "",
            "| Endpoint | HumanPOM | RDKit baseline |",
            "|---|---:|---:|",
            *endpoint_lines,
            "",
            (
                "개선 게이트: **"
                + ("PASS" if report["improvement_gate"]["passed"] else "FAIL")
                + "**"
            ),
            "",
            "이 결과는 74개 단일 분자의 집단 평균 22개 지각 endpoint에 한정됩니다. 혼합물·배합비·완제품·자연어 레시피 또는 실제 후각 90% 검증이 아닙니다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--target-odors", type=Path, required=True)
    prepare.add_argument("--variable-dictionary", type=Path, required=True)
    prepare.add_argument("--forbidden-outcome", type=Path, required=True)
    prepare.add_argument("--keller-molecules", type=Path, required=True)
    prepare.add_argument("--keller-stimuli", type=Path, required=True)
    prepare.add_argument("--keller-behavior", type=Path, required=True)
    prepare.add_argument("--molformer-root", type=Path, required=True)
    prepare.add_argument("--hf-home", type=Path, required=True)
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--threads", type=int, default=4)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--predictions", type=Path, required=True)
    seal.add_argument("--outcome", type=Path, required=True)
    seal.add_argument("--seal", type=Path, required=True)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--predictions", type=Path, required=True)
    acquire.add_argument("--seal", type=Path, required=True)
    acquire.add_argument("--outcome", type=Path, required=True)
    acquire.add_argument("--receipt", type=Path, required=True)
    acquire.add_argument("--openssl", type=Path, required=True)
    acquire.add_argument("--timestamp-response", type=Path, required=True)
    acquire.add_argument("--timestamp-ca", type=Path, required=True)
    acquire.add_argument("--timestamp-tsa", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--seal", type=Path, required=True)
    score.add_argument("--outcome", type=Path, required=True)
    score.add_argument("--receipt", type=Path, required=True)
    score.add_argument("--openssl", type=Path, required=True)
    score.add_argument("--timestamp-response", type=Path, required=True)
    score.add_argument("--timestamp-ca", type=Path, required=True)
    score.add_argument("--timestamp-tsa", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    if args.command == "prepare":
        result = prepare_predictions(args)
        summary = {
            "status": result["status"],
            "release_gate": result["release_gate"],
            "development": result["model"]["development"],
        }
    elif args.command == "seal":
        summary = create_seal(args)
    elif args.command == "acquire":
        summary = acquire_outcome(args)
    else:
        result = score_outcome(args)
        summary = {
            "status": result["status"],
            "improvement_gate": result["improvement_gate"],
            "results": result["results"],
        }
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
