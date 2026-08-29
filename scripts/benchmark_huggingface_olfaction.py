#!/usr/bin/env python
"""Reproducible Hugging Face olfaction benchmark with strict task boundaries.

The only direct leaderboard compares historical mixture-pair similarity on the
same frozen Snitz splits.  Molecular odor classification and Odor2MS language
coverage are reported as separate diagnostics because neither task is
text-to-formula generation or human sensory validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender.brief_parser import (  # noqa: E402
    BriefParseError,
    NaturalLanguageBriefParser,
    UnsupportedOdorDescriptorError,
)
from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.research.r2_physsim import (  # noqa: E402
    MixturePair,
    load_snitz_pairs,
)
from scripts.train_physsim_r2 import (  # noqa: E402
    metric_summary,
    molecule_folds,
    scaffold_folds,
    split_pairs,
)


ALGORITHMS = (
    "zero_shot_cosine",
    "cosine_ridge_alpha_1",
    "embedding_ridge_alpha_10",
    "embedding_ridge_alpha_100",
)
DEVELOPMENT_SPLIT_SEED = 142
DEVELOPMENT_REPEATS = 3
FINAL_SPLIT_SEED = 52908
FINAL_REPEATS = 5
PROTOCOLS = ("molecule_disjoint", "scaffold_disjoint")
UNIQUE_RECORD_BOOTSTRAP_DRAWS = 100_000
PAIRED_RANDOMIZATION_DRAWS = 100_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _metadata_path(root: Path, relative: str) -> Path:
    return root / ".cache" / "huggingface" / "download" / f"{relative}.metadata"


def verify_huggingface_assets(
    manifest_path: Path,
    roots: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_assets = manifest.get("assets")
    if not isinstance(expected_assets, dict) or set(expected_assets) != set(roots):
        raise ValueError("asset manifest keys do not match supplied asset roots")

    verified: dict[str, Any] = {}
    for name, root in roots.items():
        spec = expected_assets[name]
        revision = str(spec["revision"])
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"missing Hugging Face snapshot: {root}")
        file_results: dict[str, Any] = {}
        expected_python = {
            relative for relative in spec["files"] if relative.endswith(".py")
        }
        actual_python = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if ".cache" not in path.parts
        }
        if actual_python != expected_python:
            raise RuntimeError(
                f"unexpected executable Python in {name}: "
                f"{sorted(actual_python ^ expected_python)}"
            )
        for relative, expected in spec["files"].items():
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"missing or symbolic asset: {path}")
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise RuntimeError(f"asset escapes snapshot root: {path}")
            size = path.stat().st_size
            digest = sha256_file(path)
            if size != int(expected["bytes"]) or digest != expected["sha256"]:
                raise RuntimeError(f"asset integrity mismatch: {name}/{relative}")
            metadata = _metadata_path(root, relative)
            if not metadata.is_file():
                raise RuntimeError(f"snapshot revision metadata missing: {metadata}")
            metadata_lines = metadata.read_text(encoding="utf-8").splitlines()
            if not metadata_lines or metadata_lines[0] != revision:
                raise RuntimeError(f"snapshot revision mismatch: {name}/{relative}")
            file_results[relative] = {
                "bytes": size,
                "sha256": digest,
                "hub_etag": metadata_lines[1] if len(metadata_lines) > 1 else None,
            }
        verified[name] = {
            "repo_id": spec["repo_id"],
            "repo_type": spec["repo_type"],
            "revision": revision,
            "license": spec["license"],
            "revision_metadata_verified": True,
            "files": file_results,
        }
    return manifest, verified


def verify_r2_lineage(
    mixture_root: Path,
    r2_report_path: Path,
    r2_manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(r2_manifest_path.read_text(encoding="utf-8"))
    report = json.loads(r2_report_path.read_text(encoding="utf-8"))
    expected_sources = manifest["source_files"]
    source_files = {
        "molecules.csv": mixture_root / "snitz_2013" / "molecules.csv",
        "behavior.csv": mixture_root / "snitz_2013" / "behavior.csv",
    }
    source_hashes: dict[str, str] = {}
    for filename, path in source_files.items():
        digest = sha256_file(path)
        expected = expected_sources[f"dream_mixture/snitz_2013/{filename}"]["sha256"]
        if digest != expected:
            raise RuntimeError(f"Snitz source hash mismatch: {path}")
        source_hashes[filename] = digest
    report_hash = sha256_file(r2_report_path)
    expected_report_hash = manifest["strict_disjoint_validation"]["final_report_sha256"]
    if report_hash != expected_report_hash:
        raise RuntimeError("R2 final report is not the manifest-bound artifact")
    if (
        report.get("phase") != "final"
        or int(report.get("split_seed", -1)) != FINAL_SPLIT_SEED
        or int(report.get("repeats", -1)) != FINAL_REPEATS
    ):
        raise RuntimeError("R2 final report split contract changed")
    return {
        "r2_manifest_sha256": sha256_file(r2_manifest_path),
        "r2_report_sha256": report_hash,
        "snitz_source_sha256": source_hashes,
        "selected_configuration": report["selected_configuration"],
        "report": report,
    }


def verify_r2_ensemble_lineage(
    ensemble_report_path: Path,
    ensemble_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the development-selected two-seed evidence and checkpoints."""

    manifest = json.loads(ensemble_manifest_path.read_text(encoding="utf-8"))
    report = json.loads(ensemble_report_path.read_text(encoding="utf-8"))
    expected = manifest["evidence"]["ensemble_validation"]
    if Path(expected["file"]).name != ensemble_report_path.name:
        raise RuntimeError("R2 ensemble evidence filename mismatch")
    report_hash = sha256_file(ensemble_report_path)
    if report_hash != str(expected["sha256"]):
        raise RuntimeError("R2 ensemble report is not manifest-bound")
    builder = PROJECT_ROOT / "scripts" / "build_physsim_ensemble_evidence.py"
    if report.get("implementation", {}).get("sha256") != sha256_file(builder):
        raise RuntimeError("R2 ensemble report builder lineage mismatch")

    member_weights = {
        str(int(member["model_seed"])): float(member["weight"])
        for member in manifest["members"]
    }
    if set(member_weights) != {"20260715", "20260713"} or not math.isclose(
        sum(member_weights.values()), 1.0, abs_tol=1e-12
    ):
        raise RuntimeError("R2 ensemble member contract changed")
    report_selection = report.get("selection_contract", {})
    manifest_selection = manifest.get("selection", {})
    if (
        bool(report_selection.get("final_labels_used_for_selection", True))
        or report_selection.get("member_weights") != member_weights
        or manifest_selection.get("final_labels_used_for_selection") is not False
        or not math.isclose(
            float(manifest_selection.get("selected_seed_20260715_weight", -1.0)),
            member_weights["20260715"],
            abs_tol=1e-12,
        )
        or manifest_selection.get("seed_20260715_weight_grid")
        != report_selection.get("weight_grid")
        or manifest_selection.get("weights_selected_on")
        != report_selection.get("weights_selected_on")
    ):
        raise RuntimeError("R2 ensemble was not selected development-only")

    checkpoint_hashes: dict[str, str] = {}
    for member in manifest["members"]:
        checkpoint = ensemble_manifest_path.parent / str(member["file"])
        digest = sha256_file(checkpoint)
        if digest != str(member["sha256"]):
            raise RuntimeError("R2 ensemble checkpoint integrity mismatch")
        checkpoint_hashes[str(member["model_seed"])] = digest

    verified_sources: dict[str, str] = {}
    for name, spec in report.get("source_files", {}).items():
        source = PROJECT_ROOT / "benchmarks" / Path(str(spec["path"])).name
        digest = sha256_file(source)
        if digest != str(spec["sha256"]):
            raise RuntimeError(f"R2 ensemble source report mismatch: {name}")
        verified_sources[name] = digest

    for phase, seed, repeats in (
        ("development", DEVELOPMENT_SPLIT_SEED, DEVELOPMENT_REPEATS),
        ("final", FINAL_SPLIT_SEED, FINAL_REPEATS),
    ):
        for protocol in PROTOCOLS:
            rows = report["phases"][phase][protocol]["repeats"]
            if len(rows) != repeats or any(
                int(row["split_seed"]) != seed + index
                for index, row in enumerate(rows)
            ):
                raise RuntimeError("R2 ensemble split contract changed")

    return {
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest_path),
        "ensemble_report_sha256": report_hash,
        "member_checkpoint_sha256": checkpoint_hashes,
        "source_report_sha256": verified_sources,
        "selection_contract": report_selection,
        "report": report,
    }


def encode_molformer(
    model_root: Path,
    smiles: Sequence[str],
    *,
    batch_size: int,
    torch_threads: int,
    hf_home: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    ordered = sorted(set(smiles))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_root,
        trust_remote_code=True,
        local_files_only=True,
        deterministic_eval=True,
        use_safetensors=True,
    )
    model.eval()
    raw_tokens = tokenizer(ordered, add_special_tokens=True, truncation=False)[
        "input_ids"
    ]
    lengths = [len(row) for row in raw_tokens]
    maximum = int(model.config.max_position_embeddings)
    if max(lengths, default=0) > maximum:
        raise RuntimeError("a benchmark SMILES exceeds MolFormer context length")
    unknown_id = tokenizer.unk_token_id
    unknown_tokens = sum(token == unknown_id for row in raw_tokens for token in row)
    embeddings: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for offset in range(0, len(ordered), batch_size):
            values = ordered[offset : offset + batch_size]
            inputs = tokenizer(values, padding=True, return_tensors="pt")
            output = model(**inputs).pooler_output.detach().cpu().numpy()
            if output.shape != (len(values), int(model.config.hidden_size)):
                raise RuntimeError("unexpected MolFormer embedding shape")
            if not np.isfinite(output).all():
                raise RuntimeError("non-finite MolFormer embedding")
            embeddings.update(
                (value, np.asarray(row, dtype=np.float32))
                for value, row in zip(values, output, strict=True)
            )
    matrix = np.vstack([embeddings[value] for value in ordered])
    return embeddings, {
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "embedding_dimension": int(matrix.shape[1]),
        "molecule_count": len(ordered),
        "maximum_token_length": max(lengths, default=0),
        "unknown_token_count": int(unknown_tokens),
        "embedding_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "encoding_seconds": round(time.perf_counter() - started, 6),
        "device": "cpu",
        "local_files_only": True,
        "offline_environment": True,
        "deterministic_eval": bool(model.config.deterministic_eval),
        "checkpoint_format": "safetensors",
    }


def _mixture_embedding(
    mixture: Sequence[str], embeddings: dict[str, np.ndarray]
) -> np.ndarray:
    return np.mean([embeddings[value] for value in mixture], axis=0)


def _pair_features(
    pairs: Sequence[MixturePair], embeddings: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    cosines: list[float] = []
    for pair in pairs:
        first = _mixture_embedding(pair.mixture_a, embeddings)
        second = _mixture_embedding(pair.mixture_b, embeddings)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        cosine = float(np.dot(first, second) / denominator) if denominator > 0 else 0.0
        cosines.append(max(-1.0, min(1.0, cosine)))
        features.append(np.concatenate((np.abs(first - second), first * second)))
    width = len(next(iter(embeddings.values()))) * 2
    matrix = np.vstack(features) if features else np.empty((0, width), dtype=float)
    return matrix, np.asarray(cosines, dtype=float)


def _predict_algorithm(
    name: str,
    training: Sequence[MixturePair],
    validation: Sequence[MixturePair],
    embeddings: dict[str, np.ndarray],
) -> list[float]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, train_cosine = _pair_features(training, embeddings)
    validation_x, validation_cosine = _pair_features(validation, embeddings)
    train_y = np.asarray([pair.similarity for pair in training], dtype=float)
    if name == "zero_shot_cosine":
        predictions = (validation_cosine + 1.0) / 2.0
    elif name == "cosine_ridge_alpha_1":
        estimator = Ridge(alpha=1.0, solver="lsqr", tol=1e-10)
        estimator.fit(train_cosine.reshape(-1, 1), train_y)
        predictions = estimator.predict(validation_cosine.reshape(-1, 1))
    elif name.startswith("embedding_ridge_alpha_"):
        alpha = float(name.rsplit("_", 1)[1])
        estimator = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, solver="lsqr", tol=1e-10),
        )
        estimator.fit(train_x, train_y)
        predictions = estimator.predict(validation_x)
    else:
        raise KeyError(name)
    return np.clip(predictions, 0.0, 1.0).astype(float).tolist()


def _mixture_protocol(
    *,
    pairs: Sequence[MixturePair],
    molecules: Sequence[str],
    embeddings: dict[str, np.ndarray],
    split_seed: int,
    repeats: int,
    scaffold: bool,
) -> dict[str, Any]:
    pooled = {
        name: {"predictions": [], "targets": [], "repeat_spearman": []}
        for name in ALGORITHMS
    }
    repeat_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        seed = split_seed + repeat
        folds = (
            scaffold_folds(molecules, n_splits=2, seed=seed)
            if scaffold
            else molecule_folds(molecules, n_splits=2, seed=seed)
        )
        held_out = folds[0]
        training, validation, strict_validation, used_training = split_pairs(
            pairs, held_out
        )
        if len(training) < 2 or len(strict_validation) < 3:
            raise RuntimeError("mixture benchmark split is not trainable")
        if any(
            not set(pair.molecules).issubset(held_out) for pair in strict_validation
        ):
            raise RuntimeError("strict evaluation contains a training-side component")
        held_out_used = {
            value for pair in strict_validation for value in pair.molecules
        }
        leakage = used_training & held_out
        if leakage:
            raise RuntimeError("held-out molecule leakage in mixture benchmark")
        targets = [float(pair.similarity) for pair in strict_validation]
        algorithms: dict[str, Any] = {}
        algorithm_predictions: dict[str, list[float]] = {}
        for name in ALGORITHMS:
            predictions = _predict_algorithm(
                name, training, strict_validation, embeddings
            )
            metrics = metric_summary(predictions, targets)
            algorithms[name] = metrics
            algorithm_predictions[name] = predictions
            pooled[name]["predictions"].extend(predictions)
            pooled[name]["targets"].extend(targets)
            pooled[name]["repeat_spearman"].append(float(metrics["spearman"]))
        repeat_rows.append(
            {
                "repeat": repeat + 1,
                "split_seed": seed,
                "training_pairs": len(training),
                "evaluation_pairs": len(strict_validation),
                "intersection_validation_pairs": len(validation),
                "all_evaluation_components_held_out": True,
                "held_out_molecules": len(held_out),
                "held_out_molecules_observed_in_validation": len(
                    held_out & held_out_used
                ),
                "held_out_component_leakage_count": len(leakage),
                "record_ids": [pair.record_id for pair in strict_validation],
                "targets": targets,
                "held_out_sha256": hashlib.sha256(
                    json.dumps(sorted(held_out), separators=(",", ":")).encode()
                ).hexdigest(),
                "algorithms": algorithms,
                "algorithm_predictions": algorithm_predictions,
            }
        )
    summary: dict[str, Any] = {}
    for name, values in pooled.items():
        repeat_spearman = values["repeat_spearman"]
        summary[name] = {
            "pooled": metric_summary(values["predictions"], values["targets"]),
            "repeat_mean_spearman": float(np.mean(repeat_spearman)),
            "repeat_std_spearman": (
                float(np.std(repeat_spearman, ddof=1)) if repeats > 1 else 0.0
            ),
        }
    return {"repeats": repeat_rows, "summary": summary}


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 20260818, draws: int = 20000
) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    left_rank = rankdata(left, axis=1, method="average")
    right_rank = rankdata(right, axis=1, method="average")
    left_centered = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    numerator = np.sum(left_centered * right_centered, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _paired_unique_record_inference(
    r2_rows: Sequence[dict[str, Any]],
    hf_rows: Sequence[dict[str, Any]],
    *,
    algorithm: str,
    seed: int,
    bootstrap_draws: int = UNIQUE_RECORD_BOOTSTRAP_DRAWS,
    randomization_draws: int = PAIRED_RANDOMIZATION_DRAWS,
) -> dict[str, Any]:
    """Remove repeated-split duplication before paired model inference."""

    if len(r2_rows) != len(hf_rows):
        raise RuntimeError("R2/Hugging Face repeat count mismatch")
    records: dict[str, dict[str, Any]] = {}
    appearances = 0
    for r2_row, hf_row in zip(r2_rows, hf_rows, strict=True):
        r2_predictions = np.asarray(r2_row["predictions"], dtype=float)
        hf_predictions = np.asarray(
            hf_row["algorithm_predictions"][algorithm], dtype=float
        )
        r2_targets = np.asarray(r2_row["targets"], dtype=float)
        hf_targets = np.asarray(hf_row["targets"], dtype=float)
        record_ids = [str(value) for value in hf_row["record_ids"]]
        if (
            r2_predictions.shape != hf_predictions.shape
            or r2_targets.shape != hf_targets.shape
            or r2_predictions.shape != r2_targets.shape
            or len(record_ids) != len(r2_targets)
            or not np.allclose(r2_targets, hf_targets, atol=1e-12, rtol=0.0)
        ):
            raise RuntimeError("R2/Hugging Face prediction alignment mismatch")
        for record_id, target, r2_prediction, hf_prediction in zip(
            record_ids,
            r2_targets,
            r2_predictions,
            hf_predictions,
            strict=True,
        ):
            row = records.setdefault(
                record_id,
                {"target": float(target), "r2": [], "hf": []},
            )
            if not math.isclose(
                float(row["target"]), float(target), abs_tol=1e-12
            ):
                raise RuntimeError("one record id has inconsistent targets")
            row["r2"].append(float(r2_prediction))
            row["hf"].append(float(hf_prediction))
            appearances += 1

    ordered = sorted(records)
    targets = np.asarray([records[key]["target"] for key in ordered], dtype=float)
    r2_predictions = np.asarray(
        [np.mean(records[key]["r2"]) for key in ordered], dtype=float
    )
    hf_predictions = np.asarray(
        [np.mean(records[key]["hf"]) for key in ordered], dtype=float
    )
    if len(ordered) < 3:
        raise RuntimeError("unique-record inference requires at least three records")
    r2_metrics = metric_summary(r2_predictions, targets)
    hf_metrics = metric_summary(hf_predictions, targets)
    observed_delta = float(r2_metrics["spearman"] - hf_metrics["spearman"])

    rng = np.random.default_rng(seed)
    bootstrap_values: list[np.ndarray] = []
    chunk_size = 2_000
    for offset in range(0, bootstrap_draws, chunk_size):
        count = min(chunk_size, bootstrap_draws - offset)
        indices = rng.integers(0, len(ordered), size=(count, len(ordered)))
        sampled_targets = targets[indices]
        r2_correlations = _rowwise_spearman(
            r2_predictions[indices], sampled_targets
        )
        hf_correlations = _rowwise_spearman(
            hf_predictions[indices], sampled_targets
        )
        values = r2_correlations - hf_correlations
        bootstrap_values.append(values[np.isfinite(values)])
    bootstrap = np.concatenate(bootstrap_values)
    if len(bootstrap) < bootstrap_draws * 0.999:
        raise RuntimeError("too many non-finite unique-record bootstrap draws")
    interval = [
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]

    target_matrix = np.broadcast_to(
        targets, (min(chunk_size, randomization_draws), len(targets))
    )
    extreme = 0
    finite_randomizations = 0
    for offset in range(0, randomization_draws, chunk_size):
        count = min(chunk_size, randomization_draws - offset)
        swap = rng.random((count, len(ordered))) < 0.5
        left = np.where(swap, hf_predictions, r2_predictions)
        right = np.where(swap, r2_predictions, hf_predictions)
        current_targets = target_matrix[:count]
        null_delta = _rowwise_spearman(
            left, current_targets
        ) - _rowwise_spearman(right, current_targets)
        finite = null_delta[np.isfinite(null_delta)]
        finite_randomizations += len(finite)
        extreme += int(np.sum(np.abs(finite) >= abs(observed_delta)))
    if finite_randomizations < randomization_draws * 0.999:
        raise RuntimeError("too many non-finite paired randomization draws")
    randomization_p = float((extreme + 1) / (finite_randomizations + 1))

    return {
        "unit": "unique Snitz experimental record_id",
        "unique_records": len(ordered),
        "repeated_split_appearances": appearances,
        "aggregation": "mean prediction per record before correlation",
        "perfumery_ai_r2_spearman": float(r2_metrics["spearman"]),
        "huggingface_molformer_spearman": float(hf_metrics["spearman"]),
        "spearman_delta": observed_delta,
        "paired_record_bootstrap_95_interval": interval,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "bootstrap_interval_excludes_zero": interval[0] > 0 or interval[1] < 0,
        "bootstrap_fraction_nonpositive": float(np.mean(bootstrap <= 0.0)),
        "paired_randomization_two_sided_p": randomization_p,
        "paired_randomization_draws": randomization_draws,
        "inference_boundary": (
            "Conditional descriptive inference over unique experiment records; "
            "shared molecules and label-estimation uncertainty are not independent clusters."
        ),
    }


def _r2_final_section(report: dict[str, Any], protocol: str) -> dict[str, Any]:
    if "phases" in report:
        return report["phases"]["final"][protocol]
    configuration = report["selected_configuration"]
    source = report[protocol]["configurations"][configuration]
    return {
        "repeats": [
            {
                "repeat": row["repeat"],
                "split_seed": row["split_seed"],
                "held_out_sha256": row["held_out_sha256"],
                "metrics": row["model"],
                "predictions": row["predictions"],
                "targets": row["targets"],
            }
            for row in source["repeats"]
        ],
        "pooled": source["pooled_model"],
        "fold_mean_spearman": float(
            np.mean([row["model"]["spearman"] for row in source["repeats"]])
        ),
    }


def _basic_model_comparison(
    r2_report: dict[str, Any],
    phases: dict[str, Any],
    selected_by_protocol: dict[str, str],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for index, protocol in enumerate(PROTOCOLS):
        selected = selected_by_protocol[protocol]
        hf = phases["final"][protocol]["summary"][selected]
        hf_rows = phases["final"][protocol]["repeats"]
        hf_repeats = [
            float(row["algorithms"][selected]["spearman"]) for row in hf_rows
        ]
        r2 = _r2_final_section(r2_report, protocol)
        r2_rows = r2["repeats"]
        r2_repeats = [float(row["metrics"]["spearman"]) for row in r2_rows]
        if len(r2_repeats) != len(hf_repeats):
            raise RuntimeError("R2/Hugging Face repeat count mismatch")
        for r2_row, hf_row in zip(r2_rows, hf_rows, strict=True):
            if (
                int(r2_row["split_seed"]) != int(hf_row["split_seed"])
                or int(r2_row["metrics"]["n"])
                != int(hf_row["evaluation_pairs"])
                or str(r2_row["held_out_sha256"])
                != str(hf_row["held_out_sha256"])
                or not np.allclose(
                    r2_row["targets"], hf_row["targets"], atol=1e-12, rtol=0.0
                )
            ):
                raise RuntimeError("R2/Hugging Face strict split identity mismatch")
        if int(hf["pooled"]["n"]) != int(r2["pooled"]["n"]):
            raise RuntimeError("R2/Hugging Face evaluation-pair count mismatch")
        paired = [
            left - right
            for left, right in zip(r2_repeats, hf_repeats, strict=True)
        ]
        pooled_delta = float(r2["pooled"]["spearman"] - hf["pooled"]["spearman"])
        repeat_delta = float(np.mean(paired))
        interval = _bootstrap_mean_interval(paired, seed=20260818 + index)
        if pooled_delta > 0 and repeat_delta > 0:
            point_estimate = "perfumery_ai_r2_higher"
        elif pooled_delta < 0 and repeat_delta < 0:
            point_estimate = "huggingface_molformer_higher"
        else:
            point_estimate = "mixed"
        comparisons[protocol] = {
            "evaluation_pair_appearances": int(hf["pooled"]["n"]),
            "evaluation_pairs": int(hf["pooled"]["n"]),
            "perfumery_ai_r2": {
                "pooled_spearman": float(r2["pooled"]["spearman"]),
                "repeat_mean_spearman": float(np.mean(r2_repeats)),
                "repeat_spearman": r2_repeats,
            },
            "huggingface_molformer": {
                "algorithm": selected,
                "pooled_spearman": float(hf["pooled"]["spearman"]),
                "repeat_mean_spearman": float(hf["repeat_mean_spearman"]),
                "repeat_spearman": hf_repeats,
            },
            "r2_minus_molformer": {
                "pooled_spearman_delta": pooled_delta,
                "paired_repeat_mean_delta": repeat_delta,
                "paired_repeat_bootstrap_95_interval": interval,
                "r2_repeat_wins": sum(value > 0 for value in paired),
                "repeats": len(paired),
            },
            "point_estimate": point_estimate,
            "repeat_bootstrap_interval_excludes_zero": (
                interval[0] > 0 or interval[1] < 0
            ),
            "bootstrap_interval_excludes_zero": (
                interval[0] > 0 or interval[1] < 0
            ),
            "repeat_bootstrap_interpretation": (
                "Descriptive repeat-resampling interval; repeats overlap and "
                "are not independent experimental replicates."
            ),
            "bootstrap_interpretation": (
                "Descriptive repeat-resampling interval; repeats overlap and "
                "are not independent experimental replicates."
            ),
        }
    return comparisons


def compare_mixture_models(
    pairs: Sequence[MixturePair],
    embeddings: dict[str, np.ndarray],
    r2_ensemble_report: dict[str, Any],
    legacy_r2_report: dict[str, Any],
) -> dict[str, Any]:
    molecules = sorted({value for pair in pairs for value in pair.molecules})
    phases: dict[str, Any] = {}
    for phase, seed, repeats in (
        ("development", DEVELOPMENT_SPLIT_SEED, DEVELOPMENT_REPEATS),
        ("final", FINAL_SPLIT_SEED, FINAL_REPEATS),
    ):
        phases[phase] = {
            "molecule_disjoint": _mixture_protocol(
                pairs=pairs,
                molecules=molecules,
                embeddings=embeddings,
                split_seed=seed,
                repeats=repeats,
                scaffold=False,
            ),
            "scaffold_disjoint": _mixture_protocol(
                pairs=pairs,
                molecules=molecules,
                embeddings=embeddings,
                split_seed=seed,
                repeats=repeats,
                scaffold=True,
            ),
        }
    selection_scores: dict[str, dict[str, float]] = {}
    selected_by_protocol: dict[str, str] = {}
    for protocol in PROTOCOLS:
        protocol_scores: dict[str, float] = {}
        for name in ALGORITHMS:
            item = phases["development"][protocol]["summary"][name]
            protocol_scores[name] = min(
                float(item["pooled"]["spearman"]),
                float(item["repeat_mean_spearman"]),
            )
        selection_scores[protocol] = protocol_scores
        selected_by_protocol[protocol] = max(
            protocol_scores,
            key=lambda name: (protocol_scores[name], name),
        )

    comparisons = _basic_model_comparison(
        r2_ensemble_report, phases, selected_by_protocol
    )
    legacy_comparisons = _basic_model_comparison(
        legacy_r2_report, phases, selected_by_protocol
    )
    for index, protocol in enumerate(PROTOCOLS):
        r2_rows = _r2_final_section(r2_ensemble_report, protocol)["repeats"]
        comparisons[protocol]["unique_record_inference"] = (
            _paired_unique_record_inference(
                r2_rows,
                phases["final"][protocol]["repeats"],
                algorithm=selected_by_protocol[protocol],
                seed=20260820 + index,
            )
        )

    molecule = comparisons["molecule_disjoint"]
    molecule_unique = molecule["unique_record_inference"]
    molecule_requirements = {
        "positive_pooled_delta": (
            molecule["r2_minus_molformer"]["pooled_spearman_delta"] > 0
        ),
        "repeat_resampling_interval_lower_above_zero": (
            molecule["repeat_bootstrap_interval_excludes_zero"]
            and molecule["r2_minus_molformer"][
                "paired_repeat_bootstrap_95_interval"
            ][0]
            > 0
        ),
        "unique_record_bootstrap_interval_lower_above_zero": (
            molecule_unique["bootstrap_interval_excludes_zero"]
            and molecule_unique["paired_record_bootstrap_95_interval"][0] > 0
        ),
        "paired_randomization_two_sided_p_below_0_05": (
            molecule_unique["paired_randomization_two_sided_p"] < 0.05
        ),
    }
    molecule_gate = all(molecule_requirements.values())
    return {
        "task": "historical_mixture_pair_similarity",
        "task_is_directly_comparable": True,
        "dataset": {
            "name": "Snitz 2013",
            "pairs": len(pairs),
            "molecules": len(molecules),
        },
        "split_contract": {
            "description": "same all-components-held-out strict protocol used by the frozen R2 final report",
            "development_seed": DEVELOPMENT_SPLIT_SEED,
            "development_repeats": DEVELOPMENT_REPEATS,
            "final_seed": FINAL_SPLIT_SEED,
            "final_repeats": FINAL_REPEATS,
            "final_labels_used_for_algorithm_selection": False,
        },
        "algorithms": list(ALGORITHMS),
        "selection": {
            "molformer_rule": (
                "for each declared protocol, maximize the worse of "
                "development-only pooled and repeat-mean Spearman"
            ),
            "molformer_scores_by_protocol": selection_scores,
            "molformer_selected_by_protocol": selected_by_protocol,
            "perfumery_ai_r2": r2_ensemble_report["selection_contract"],
        },
        "training_budget_note": (
            "R2 uses two frozen split-specific Snitz training seeds combined "
            "by one coarse-grid weight selected before reading final labels. "
            "MoLFormer candidates use the same training pairs and are selected "
            "per declared protocol on development data. Final metrics use "
            "identical all-components-held-out pairs."
        ),
        "phases": phases,
        "final_comparison": comparisons,
        "superseded_single_seed_reference": legacy_comparisons,
        "molecule_disjoint_statistical_gate": {
            "passed": molecule_gate,
            "requirements": molecule_requirements,
            "retrospective_status": (
                "Post-outcome remediation of a previously reported single-seed "
                "comparison; the weight-selection code path is development-only, "
                "but this is not a newly unopened external dataset."
            ),
        },
        "claim_boundary": "Historical reported mixture similarity, not text-to-formula accuracy and not a percentage of human olfactory equivalence.",
    }


def _classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        hamming_loss,
        roc_auc_score,
    )

    predictions = (probabilities >= threshold).astype(np.int8)
    aucs: list[float] = []
    average_precisions: list[float] = []
    for index in range(targets.shape[1]):
        column = targets[:, index]
        if len(np.unique(column)) == 2:
            aucs.append(float(roc_auc_score(column, probabilities[:, index])))
            average_precisions.append(
                float(average_precision_score(column, probabilities[:, index]))
            )
    return {
        "threshold": float(threshold),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(targets, predictions, average="micro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "exact_match_accuracy": float(accuracy_score(targets, predictions)),
        "macro_auroc": float(np.mean(aucs)),
        "macro_auprc": float(np.mean(average_precisions)),
        "auroc_eligible_labels": len(aucs),
    }


def evaluate_hari_model(
    model_root: Path, dataset_root: Path, *, batch_size: int
) -> dict[str, Any]:
    import pandas as pd
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from torch import nn

    labels = json.loads((model_root / "labels.json").read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) != 50 or len(set(labels)) != 50:
        raise RuntimeError("unexpected Hari label vocabulary")
    frames = {
        split: pd.read_csv(dataset_root / "data" / filename)
        for split, filename in (("validation", "val.csv"), ("test", "test.csv"))
    }
    for split, frame in frames.items():
        if list(frame.columns) != ["smiles", *labels]:
            raise RuntimeError(f"{split} columns do not match checkpoint labels")
        values = frame[labels].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise RuntimeError(f"{split} labels are not binary")

    class OdorMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, len(labels)),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.net(values)

    model = OdorMLP()
    state = torch.load(
        model_root / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )

    def predict(frame) -> tuple[np.ndarray, int, float]:
        started = time.perf_counter()
        arrays: list[np.ndarray] = []
        invalid = 0
        for value in frame["smiles"].astype(str):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                invalid += 1
                arrays.append(np.zeros(2048, dtype=np.float32))
                continue
            array = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(
                fingerprint_generator.GetFingerprint(molecule), array
            )
            arrays.append(array)
        if invalid:
            raise RuntimeError(f"Hari dataset contains {invalid} invalid SMILES")
        matrix = np.vstack(arrays)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(matrix), batch_size):
                logits = model(torch.from_numpy(matrix[offset : offset + batch_size]))
                batches.append(torch.sigmoid(logits).numpy())
        return np.vstack(batches), invalid, time.perf_counter() - started

    validation_probabilities, validation_invalid, validation_seconds = predict(
        frames["validation"]
    )
    test_probabilities, test_invalid, test_seconds = predict(frames["test"])
    validation_targets = frames["validation"][labels].to_numpy(dtype=np.int8)
    test_targets = frames["test"][labels].to_numpy(dtype=np.int8)
    candidates = [round(value, 2) for value in np.arange(0.05, 0.81, 0.01)]
    scored = [
        _classification_metrics(validation_targets, validation_probabilities, value)
        for value in candidates
    ]
    selected = max(
        scored,
        key=lambda row: (
            row["macro_f1"],
            row["micro_f1"],
            -abs(row["threshold"] - 0.3),
            -row["threshold"],
        ),
    )["threshold"]
    default_test_metrics = _classification_metrics(
        test_targets, test_probabilities, 0.3
    )
    selected_test_metrics = _classification_metrics(
        test_targets, test_probabilities, selected
    )
    claimed = {
        "macro_f1": 0.421,
        "micro_f1": 0.498,
        "hamming_loss": 0.080,
        "test_rows": 545,
    }
    reproduction_deltas = {
        metric: float(default_test_metrics[metric] - claimed[metric])
        for metric in ("macro_f1", "micro_f1", "hamming_loss")
    }
    return {
        "task": "single_molecule_multilabel_odor_descriptor_prediction",
        "task_is_directly_comparable_to_text_to_formula": False,
        "checkpoint_loading": "torch.load(weights_only=True), strict state_dict",
        "fingerprint": "Morgan ECFP4 radius=2, 2048 bits",
        "labels": len(labels),
        "validation_rows": len(frames["validation"]),
        "test_rows": len(frames["test"]),
        "invalid_smiles": validation_invalid + test_invalid,
        "threshold_selection": {
            "source": "validation split only",
            "objective": "macro F1, then micro F1",
            "selected": selected,
            "test_labels_used": False,
        },
        "reproduced_test_metrics": {
            "model_card_usage_threshold_0_3": default_test_metrics,
            "validation_selected_threshold": selected_test_metrics,
        },
        "model_card_claimed_test_metrics": claimed,
        "model_card_reproduction": {
            "absolute_tolerance": 0.001,
            "deltas_reproduced_minus_claimed": reproduction_deltas,
            "passed": all(
                abs(value) <= 0.001 for value in reproduction_deltas.values()
            ),
        },
        "comparison_exclusion": "The dataset is derived from GoodScents and Leffingwell, which overlap the R2 encoder's monomolecular pretraining sources; this track is therefore excluded from the Perfumery-AI-vs-HF leaderboard.",
        "runtime": {
            "validation_seconds": round(validation_seconds, 6),
            "test_seconds": round(test_seconds, 6),
            "device": "cpu",
        },
        "claim_boundary": "This checkpoint predicts labels for individual molecules. It does not generate mixtures or fragrance formulas.",
    }


def evaluate_language_coverage(
    odor2ms_test: Path,
    hari_labels: Sequence[str],
    *,
    split_manifest: Path,
) -> dict[str, Any]:
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    rows: list[dict[str, Any]] = []
    with odor2ms_test.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            text = row.get("text")
            spectrum = row.get("ms")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Odor2MS row {line_number} has invalid text")
            if (
                not isinstance(spectrum, list)
                or len(spectrum) != 401
                or not all(math.isfinite(float(value)) for value in spectrum)
            ):
                raise RuntimeError(f"Odor2MS row {line_number} has invalid spectrum")
            rows.append({"text": text})

    def coverage(texts: Sequence[str]) -> dict[str, Any]:
        dimensions: Counter[str] = Counter()
        backends: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        unsupported: Counter[str] = Counter()
        confidences: list[float] = []
        parsed = 0
        recognized = 0
        for text in texts:
            try:
                brief = parser.parse(text)
            except UnsupportedOdorDescriptorError as error:
                recognized += 1
                unsupported.update(error.descriptors)
                continue
            except BriefParseError as error:
                errors[str(error)] += 1
                continue
            parsed += 1
            recognized += 1
            dimensions.update(brief.desired_dimensions)
            backends[brief.semantic_backend] += 1
            confidences.append(float(brief.semantic_confidence))
        return {
            "inputs": len(texts),
            "parsed": parsed,
            "formula_ready_parsed": parsed,
            "parse_coverage_percent": round(100.0 * parsed / len(texts), 4),
            "formula_ready_parse_coverage_percent": round(
                100.0 * parsed / len(texts), 4
            ),
            "recognized": recognized,
            "recognition_coverage_percent": round(100.0 * recognized / len(texts), 4),
            "mean_semantic_confidence": (
                round(float(np.mean(confidences)), 6) if confidences else 0.0
            ),
            "semantic_confidence_interpretation": (
                "Uncalibrated parser-evidence score on [0, 1]; not a probability, "
                "formula accuracy, or olfactory similarity."
            ),
            "high_confidence_parse_percent": (
                round(
                    100.0
                    * sum(value >= 0.70 for value in confidences)
                    / len(confidences),
                    4,
                )
                if confidences
                else 0.0
            ),
            "semantic_backends": dict(sorted(backends.items())),
            "recognized_dimension_counts": dict(sorted(dimensions.items())),
            "recognized_but_formula_unsupported_counts": dict(
                sorted(unsupported.items())
            ),
            "parse_error_counts": dict(sorted(errors.items())),
        }

    texts = [row["text"] for row in rows]
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if sha256_file(odor2ms_test) != split.get("dataset_sha256"):
        raise RuntimeError("Odor2MS parser split dataset hash mismatch")
    salt = str(split["salt"])
    separator = split.get("separator")
    modulus = int(split["modulus"])
    holdout_bucket = int(split["holdout_bucket"])
    if separator != "\\0" or modulus <= 1 or not 0 <= holdout_bucket < modulus:
        raise RuntimeError("Odor2MS parser split parameters are invalid")
    development_texts: list[str] = []
    holdout_texts: list[str] = []
    development_hashes: list[str] = []
    holdout_hashes: list[str] = []
    for text in texts:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        split_hash = hashlib.sha256(
            f"{salt}{separator}{text}".encode("utf-8")
        ).hexdigest()
        if int(split_hash[:8], 16) % modulus == holdout_bucket:
            holdout_texts.append(text)
            holdout_hashes.append(text_hash)
        else:
            development_texts.append(text)
            development_hashes.append(text_hash)

    def hash_inventory(values: Sequence[str]) -> str:
        payload = "\n".join(sorted(values)) + "\n"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if (
        len(texts) != int(split["total"])
        or len(development_texts) != int(split["development_count"])
        or len(holdout_texts) != int(split["holdout_count"])
        or hash_inventory(development_hashes) != split["development_text_hashes_sha256"]
        or hash_inventory(holdout_hashes) != split["holdout_text_hashes_sha256"]
    ):
        raise RuntimeError("Odor2MS parser split inventory mismatch")

    odor2ms = coverage(texts)
    label_prompts = coverage([f"a {label} fragrance" for label in hari_labels])
    return {
        "task": "natural_language_parser_coverage",
        "odor2ms_test": odor2ms,
        "odor2ms_development": coverage(development_texts),
        "odor2ms_uninspected_holdout": coverage(holdout_texts),
        "odor2ms_split": {
            "manifest": str(split_manifest.resolve()),
            "manifest_sha256": sha256_file(split_manifest),
            "salt": salt,
            "separator_utf8_hex": separator.encode("utf-8").hex(),
            "development_count": len(development_texts),
            "holdout_count": len(holdout_texts),
            "holdout_text_inspected_during_development": False,
        },
        "huggingface_50_label_prompts": label_prompts,
        "gold_formula_available": False,
        "text_to_mass_spectrum_generation_supported": False,
        "accuracy_claimed": False,
        "claim_boundary": "Recognition and formula-ready parser coverage only. Recognized unsupported descriptors fail closed instead of receiving a misleading coarse formula. Odor2MS has mass-spectrum targets, not gold perfume formulas, so no formula accuracy or olfactory accuracy is computed.",
    }


def compare_language_coverage(
    current: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tracks = baseline.get("tracks")
    if not isinstance(tracks, dict):
        raise RuntimeError("parser coverage baseline tracks are missing")
    comparisons: dict[str, Any] = {}
    for name in ("odor2ms_test", "huggingface_50_label_prompts"):
        before = tracks.get(name)
        after = current[name]
        if not isinstance(before, dict) or int(before.get("inputs", -1)) != int(
            after["inputs"]
        ):
            raise RuntimeError(f"parser coverage baseline input mismatch: {name}")
        before_coverage = float(before["parse_coverage_percent"])
        after_coverage = float(after["formula_ready_parse_coverage_percent"])
        before_confidence = float(before["mean_semantic_confidence"])
        after_confidence = float(after["mean_semantic_confidence"])
        comparisons[name] = {
            "baseline_legacy_parse_coverage_percent": before_coverage,
            "current_formula_ready_parse_coverage_percent": after_coverage,
            "current_formula_ready_minus_legacy_parse_delta_points": round(
                after_coverage - before_coverage, 4
            ),
            "current_recognition_coverage_percent": float(
                after["recognition_coverage_percent"]
            ),
            "baseline_mean_semantic_confidence": before_confidence,
            "current_mean_semantic_confidence": after_confidence,
            "mean_semantic_confidence_comparable_across_versions": False,
            "confidence_comparison_exclusion": (
                "The legacy hash-ngram score and current lexical/projection "
                "evidence score do not share a calibrated scale."
            ),
        }
    return {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "tracks": comparisons,
        "claim_boundary": baseline["claim_boundary"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    mixture = report["mixture_similarity"]["final_comparison"]
    hari = report["molecular_odor_classification"]["reproduced_test_metrics"]
    language_change = report["language_coverage_change"]["tracks"]
    selected = report["mixture_similarity"]["selection"][
        "molformer_selected_by_protocol"
    ]
    r2_weights = report["mixture_similarity"]["selection"]["perfumery_ai_r2"][
        "member_weights"
    ]
    molecule_gate = report["mixture_similarity"][
        "molecule_disjoint_statistical_gate"
    ]
    lines = [
        "# Hugging Face 관련 후각 모델 벤치마크",
        "",
        f"- 생성 시각: `{report['generated_at']}`",
        f"- 상태: `{report['status']}`",
        "- 선택된 MoLFormer 적응 방식(개발 분할 전용 선택): "
        + ", ".join(f"`{key}={value}`" for key, value in selected.items()),
        "- Perfumery AI R2 두-시드 가중치(개발 분할 전용 선택): "
        + f"`20260715={r2_weights['20260715']:.1f}`, "
        + f"`20260713={r2_weights['20260713']:.1f}`",
        "",
        "## 직접 비교: Snitz 혼합물 유사도",
        "",
        "| 엄격 분할 | 반복 포함 평가(n) | 고유 실험쌍(n) | Perfumery AI R2 pooled Spearman | HF MoLFormer pooled Spearman | R2-HF | R2 반복 승수 | 반복 차이 95% 구간 | 고유쌍 차이 95% 구간 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("molecule_disjoint", "scaffold_disjoint"):
        row = mixture[protocol]
        unique = row["unique_record_inference"]
        lines.append(
            "| "
            + protocol
            + f" | {row['evaluation_pair_appearances']}"
            + f" | {unique['unique_records']}"
            + f" | {row['perfumery_ai_r2']['pooled_spearman']:.4f}"
            + f" | {row['huggingface_molformer']['pooled_spearman']:.4f}"
            + f" | {row['r2_minus_molformer']['pooled_spearman_delta']:+.4f}"
            + f" | {row['r2_minus_molformer']['r2_repeat_wins']}/{row['r2_minus_molformer']['repeats']}"
            + " | ["
            + f"{row['r2_minus_molformer']['paired_repeat_bootstrap_95_interval'][0]:+.4f}, "
            + f"{row['r2_minus_molformer']['paired_repeat_bootstrap_95_interval'][1]:+.4f}]"
            + " | ["
            + f"{unique['paired_record_bootstrap_95_interval'][0]:+.4f}, "
            + f"{unique['paired_record_bootstrap_95_interval'][1]:+.4f}] |"
        )
    default_metrics = hari["model_card_usage_threshold_0_3"]
    selected_metrics = hari["validation_selected_threshold"]
    lines.extend(
        [
            "",
            "이 표만 동일 과제·동일 엄격 all-components-held-out 평가 쌍의 직접 비교입니다. 반복 구간은 서로 겹치는 5회 분할의 기술적 재표집이고, 고유쌍 구간은 중복 record_id를 먼저 평균한 대응 bootstrap입니다.",
            f"분자 분리 통계 게이트: `{'PASS' if molecule_gate['passed'] else 'FAIL'}`. 고유 실험쌍 대응 무작위화 양측 p값은 `{mixture['molecule_disjoint']['unique_record_inference']['paired_randomization_two_sided_p']:.6f}`입니다.",
            "기존 단일 시드 결과는 보고서의 `superseded_single_seed_reference`에 보존했습니다. 이번 가중치 선택은 개발 분할만 읽지만, 같은 Snitz 최종 데이터가 과거 분석에서 이미 공개된 뒤 수행한 회고적 개선이므로 새 외부 전향 검증으로 표현하지 않습니다.",
            "Spearman은 인간 후각 정확도 백분율이 아닙니다.",
            "",
            "## HF 단일분자 후각 분류 재현",
            "",
            "| 설정 | Macro F1 | Micro F1 | Hamming loss |",
            "|---|---:|---:|---:|",
            f"| 모델카드 사용 임계값 0.30 | {default_metrics['macro_f1']:.4f} | {default_metrics['micro_f1']:.4f} | {default_metrics['hamming_loss']:.4f} |",
            f"| validation 선택 임계값 | {selected_metrics['macro_f1']:.4f} | {selected_metrics['micro_f1']:.4f} | {selected_metrics['hamming_loss']:.4f} |",
            "",
            "이 모델은 한 분자의 50개 냄새 라벨을 예측하며 처방 생성 모델이 아닙니다.",
            "",
            "## 자연어 커버리지 진단",
            "",
            f"- Odor2MS test 303건: 기존 느슨한 파싱 `{language_change['odor2ms_test']['baseline_legacy_parse_coverage_percent']:.4f}%` → 현재 fail-closed 조향 가능 파싱 `{language_change['odor2ms_test']['current_formula_ready_parse_coverage_percent']:.4f}%`; 표현 인식 `{language_change['odor2ms_test']['current_recognition_coverage_percent']:.4f}%`",
            f"- 개발 중 문장을 열지 않은 고정 Odor2MS 63건 holdout: 조향 가능 `{report['language_coverage']['odor2ms_uninspected_holdout']['formula_ready_parse_coverage_percent']:.4f}%`, 표현 인식 `{report['language_coverage']['odor2ms_uninspected_holdout']['recognition_coverage_percent']:.4f}%`",
            f"- HF 50개 후각 라벨: 기존 느슨한 파싱 `{language_change['huggingface_50_label_prompts']['baseline_legacy_parse_coverage_percent']:.4f}%` → 현재 fail-closed 조향 가능 파싱 `{language_change['huggingface_50_label_prompts']['current_formula_ready_parse_coverage_percent']:.4f}%`; 표현 인식 `{language_change['huggingface_50_label_prompts']['current_recognition_coverage_percent']:.4f}%`",
            f"- 현재 평균 비보정 파서 증거 점수: Odor2MS `{language_change['odor2ms_test']['current_mean_semantic_confidence']:.6f}`, HF 라벨 `{language_change['huggingface_50_label_prompts']['current_mean_semantic_confidence']:.6f}`. 기존 해시 점수와 척도가 달라 버전 간 차이는 계산하지 않았습니다.",
            "- 현재 조향 축으로 정직하게 투영할 수 없는 세부 표현은 인식하되 레시피 생성을 fail-closed 처리합니다.",
            "- Odor2MS의 정답은 401-bin 질량스펙트럼이며 정답 향수 처방이 아니므로 처방 정확도는 계산하지 않았습니다.",
            "",
            "## 결론 경계",
            "",
            "이 결과는 인간 후각 90% 유사성을 입증하지 않습니다. Snitz 혼합물 유사도 트랙만 동일 과제 직접 비교이며 나머지는 기능 진단입니다.",
            "",
            "## 고정된 공개 출처",
            "",
            "- [IBM Research MoLFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct)",
            "- [Hari5115 Molecular Odor Predictor](https://huggingface.co/Hari5115/molecular-odor-predictor)",
            "- [Hari5115 Molecular Odor Dataset](https://huggingface.co/datasets/Hari5115/molecular-odor-dataset)",
            "- [Odor2MS Dataset](https://huggingface.co/datasets/zjuermath/odor2ms_dataset)",
            "",
        ]
    )
    return "\n".join(lines)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture-data-root", type=Path, required=True)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "huggingface_benchmark_assets.json",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "hf_benchmark_assets",
    )
    parser.add_argument(
        "--r2-final-report",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_final_strict.json",
    )
    parser.add_argument(
        "--r2-manifest",
        type=Path,
        default=PROJECT_ROOT / "fragrance_ai" / "data" / "physsim_r2_manifest.json",
    )
    parser.add_argument(
        "--r2-ensemble-report",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "physsim_r2_ensemble_validation.json"
        ),
    )
    parser.add_argument(
        "--r2-ensemble-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "fragrance_ai"
            / "data"
            / "physsim_r2_ensemble_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "huggingface_olfaction_benchmark.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "huggingface_olfaction_benchmark.md",
    )
    parser.add_argument(
        "--parser-baseline",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "huggingface_parser_baseline_20260818.json"
        ),
    )
    parser.add_argument(
        "--parser-split",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "odor2ms_parser_split_v1.json",
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--torch-threads", type=positive_int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()

    roots = {
        "molformer": args.asset_root / "molformer_compat_v4",
        "hari_model": args.asset_root / "hari_model",
        "hari_dataset": args.asset_root / "hari_dataset",
        "odor2ms": args.asset_root / "odor2ms",
    }
    asset_manifest, verified_assets = verify_huggingface_assets(
        args.asset_manifest, roots
    )
    lineage = verify_r2_lineage(
        args.mixture_data_root.resolve(),
        args.r2_final_report.resolve(),
        args.r2_manifest.resolve(),
    )
    ensemble_lineage = verify_r2_ensemble_lineage(
        args.r2_ensemble_report.resolve(),
        args.r2_ensemble_manifest.resolve(),
    )
    legacy_r2_report = lineage.pop("report")
    r2_ensemble_report = ensemble_lineage.pop("report")
    pairs = load_snitz_pairs(args.mixture_data_root)
    molecules = sorted({value for pair in pairs for value in pair.molecules})
    embeddings, embedding_runtime = encode_molformer(
        roots["molformer"],
        molecules,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        hf_home=args.asset_root / ".hf_home",
    )
    mixture = compare_mixture_models(
        pairs,
        embeddings,
        r2_ensemble_report,
        legacy_r2_report,
    )
    hari = evaluate_hari_model(
        roots["hari_model"], roots["hari_dataset"], batch_size=args.batch_size
    )
    hari_labels = json.loads(
        (roots["hari_model"] / "labels.json").read_text(encoding="utf-8")
    )
    language = evaluate_language_coverage(
        roots["odor2ms"] / "data" / "test.jsonl",
        hari_labels,
        split_manifest=args.parser_split,
    )
    language_change = compare_language_coverage(language, args.parser_baseline)
    report = {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "claim_boundary": "This benchmark does not establish 90% human olfactory similarity. Only the Snitz mixture-similarity track is a direct same-task model comparison; all other tracks are capability diagnostics.",
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "ensemble_builder_sha256": sha256_file(
                PROJECT_ROOT / "scripts" / "build_physsim_ensemble_evidence.py"
            ),
            "unique_record_bootstrap_draws": UNIQUE_RECORD_BOOTSTRAP_DRAWS,
            "paired_randomization_draws": PAIRED_RANDOMIZATION_DRAWS,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not_reported",
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": args.torch_threads,
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "asset_manifest": {
            "path": str(args.asset_manifest.resolve()),
            "sha256": sha256_file(args.asset_manifest),
            "frozen_on": asset_manifest["frozen_on"],
        },
        "verified_huggingface_assets": verified_assets,
        "remote_code_review": {
            "scope": [
                "configuration_molformer.py",
                "modeling_molformer.py",
                "tokenization_molformer.py",
                "tokenization_molformer_fast.py",
            ],
            "pinned_hashes_verified_before_import": True,
            "network_process_and_dynamic_execution_sinks_found": 0,
        },
        "r2_lineage": {
            "primary_single_seed": lineage,
            "two_seed_ensemble": ensemble_lineage,
        },
        "molformer_runtime": embedding_runtime,
        "mixture_similarity": mixture,
        "molecular_odor_classification": hari,
        "language_coverage": language,
        "language_coverage_change": language_change,
        "capability_matrix": {
            "perfumery_ai_core": {
                "natural_language_to_safe_formula": {
                    "supported": True,
                    "measured_here": False,
                },
                "historical_mixture_similarity": {
                    "supported": True,
                    "measured_here": True,
                },
                "arbitrary_smiles_to_50_odor_labels": False,
                "text_to_mass_spectrum": False,
            },
            "huggingface_molformer": {
                "natural_language_to_safe_formula": False,
                "historical_mixture_similarity_via_frozen_embedding_adapter": {
                    "supported": True,
                    "measured_here": True,
                },
                "molecular_feature_extraction": True,
                "text_to_mass_spectrum": False,
            },
            "huggingface_hari_odor_mlp": {
                "natural_language_to_safe_formula": False,
                "single_molecule_odor_labels": True,
                "mixture_support": False,
            },
        },
    }
    json_text = (
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    _atomic_write(args.output, json_text)
    _atomic_write(args.markdown_output, render_markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_molformer_adapter": mixture["selection"][
                    "molformer_selected_by_protocol"
                ],
                "molecule_disjoint_statistical_gate": mixture[
                    "molecule_disjoint_statistical_gate"
                ],
                "final_comparison": mixture["final_comparison"],
                "hari_reproduced": hari["reproduced_test_metrics"],
                "language_coverage": language,
                "language_coverage_change": language_change,
                "output": str(args.output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
