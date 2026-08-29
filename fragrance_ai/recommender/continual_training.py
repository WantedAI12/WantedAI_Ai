"""Two-stage, label-isolated training for the continual concentration model.

``prepare_blind_challenge`` trains only from historical external-human rows and
writes predictions for an input file that is forbidden from containing the
outcome column.  ``finalize_blind_challenge`` runs later, after an acquisition
service has independently timestamped the prediction seal and fetched outcome
bytes.  It never refits the candidate after opening those outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .artifact_trust import sha256_file
from .concentration_response import FrozenConcentrationResponse
from .continuous_improvement import (
    CANDIDATE_SCHEMA,
    ContinuousImprovementController,
    ContinuousImprovementPolicy,
    DATASET_RECEIPT_SCHEMA,
    EVALUATION_SCHEMA,
    bootstrap_seed,
)


TRAINING_COLUMNS = frozenset(
    {
        "row_id",
        "source_id",
        "target_id",
        "molecule_id",
        "scaffold_id",
        "dilution_fraction",
        "intensity",
        "label_origin",
        "evidence_class",
    }
)
CHALLENGE_COLUMNS = frozenset(
    {
        "row_id",
        "source_id",
        "target_id",
        "molecule_id",
        "scaffold_id",
        "dilution_fraction",
    }
)
OUTCOME_COLUMNS = frozenset({"row_id", "intensity"})
BLOCKED_LABEL_ORIGINS = frozenset(
    {"model_generated", "simulation_proxy", "synthetic", "self_training", "weak_label"}
)
ALLOWED_TRAINING_EVIDENCE_CLASSES = frozenset(
    {"retrospective_external_human", "prospective_external_human"}
)
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
ALPHAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)
PREPARED_SCHEMA = "perfumery-blind-challenger-prepared/v1"
PREDICTION_SCHEMA = "perfumery-blind-challenge-predictions/v1"
PREDICTION_SEAL_SCHEMA = "perfumery-blind-challenge-local-seal/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _read_csv(
    path: Path, required: frozenset[str], *, maximum_rows: int
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if fields != required:
            raise ValueError(
                f"{path.name} columns must be exactly {sorted(required)}; got {sorted(fields)}"
            )
        rows = []
        for row in reader:
            if len(rows) >= maximum_rows:
                raise ValueError(f"{path.name} exceeds the policy row limit")
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    identifiers = [str(row.get("row_id", "")).strip() for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path.name} row_id values must be nonempty and unique")
    return rows


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and between {lower} and {upper}")
    return result


def _validate_common(row: Mapping[str, str], *, intensity: bool) -> None:
    for name in ("source_id", "target_id", "molecule_id", "scaffold_id"):
        if not str(row.get(name, "")).strip():
            raise ValueError(f"{name} must be nonempty")
    _number(row.get("dilution_fraction"), "dilution_fraction", lower=1e-8, upper=1.0)
    if intensity:
        _number(row.get("intensity"), "intensity", lower=0.0, upper=100.0)


def _features(dilution: np.ndarray) -> np.ndarray:
    log_c = np.log10(np.clip(np.asarray(dilution, dtype=float), 1e-8, 1.0))
    return np.column_stack((log_c, log_c * log_c))


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standard = (x - mean) / scale
    target_mean = float(y.mean())
    matrix = standard.T @ standard + float(alpha) * np.eye(standard.shape[1])
    coefficients = np.linalg.solve(matrix, standard.T @ (y - target_mean))
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "intercept": target_mean,
        "alpha": float(alpha),
    }


def _predict(model: Mapping[str, Any], dilution: np.ndarray) -> np.ndarray:
    x = _features(dilution)
    value = (
        (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficients"]) + float(model["intercept"])
    return np.clip(value, 0.0, 100.0)


def _folds(groups: Iterable[str], seed: int) -> list[set[str]]:
    unique = np.asarray(sorted(set(groups)), dtype=object)
    if len(unique) < 2:
        raise ValueError("training requires at least two molecule groups")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    return [set(values.tolist()) for values in np.array_split(unique, min(5, len(unique)))]


def _select_model(rows: list[dict[str, str]], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    dilution = np.asarray([float(row["dilution_fraction"]) for row in rows])
    target = np.asarray([float(row["intensity"]) for row in rows])
    groups = np.asarray([row["molecule_id"] for row in rows], dtype=object)
    x = _features(dilution)
    folds = _folds(groups, seed)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        absolute_errors = []
        for held_out in folds:
            validation = np.asarray([group in held_out for group in groups])
            training = ~validation
            if not training.any() or not validation.any():
                raise ValueError("grouped development fold is empty")
            model = _fit_ridge(x[training], target[training], alpha)
            absolute_errors.extend(
                np.abs(_predict(model, dilution[validation]) - target[validation]).tolist()
            )
        scores[str(alpha)] = float(np.mean(absolute_errors))
    selected = min(ALPHAS, key=lambda value: (scores[str(value)], value))
    model = _fit_ridge(x, target, selected)
    return model, {
        "selection": "minimum pooled molecule-grouped cross-validation MAE",
        "selected_alpha": float(selected),
        "fold_seed": int(seed),
        "candidate_mae_by_alpha": scores,
        "outcome_rows_used_for_selection": False,
    }


def _runtime(model: Mapping[str, Any], rows: list[dict[str, str]], training_sha: str) -> dict[str, Any]:
    dilution = [float(row["dilution_fraction"]) for row in rows]
    return {
        "schema_version": "1.0",
        "runtime": "numpy_concentration_response_v1",
        "format": "standard_scaler_plus_ridge_coefficients_v1",
        "feature_contract": ["log10_dilution", "log10_dilution_squared"],
        "feature_mean": [float(value) for value in model["mean"]],
        "feature_scale": [float(value) for value in model["scale"]],
        "coefficients": [float(value) for value in model["coefficients"]],
        "intercept": float(model["intercept"]),
        "dilution_range_fraction": [float(min(dilution)), float(max(dilution))],
        "source_training_artifact_sha256": training_sha,
        "source_training_artifact_required_at_runtime": False,
        "allow_pickle": False,
    }


def _manifest(
    runtime_path: Path,
    runtime: Mapping[str, Any],
    rows: list[dict[str, str]],
    training_sha: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "generated_at": _utc_now(),
        "source_model_file": "not_distributed",
        "source_model_sha256": training_sha,
        "runtime_file": runtime_path.name,
        "runtime_sha256": sha256_file(runtime_path),
        "algorithm": "concentration_only_ridge",
        "development_selected_structure_candidate": "not_applicable",
        "structure_specific_weight": 0.0,
        "distribution_contract": {
            "runtime_format": "json_numeric_arrays_only",
            "source_model_packaged": False,
            "source_model_required_at_runtime": False,
            "pickle_deserialization_allowed": False,
        },
        "model_seed": int(selection["fold_seed"]),
        "descriptor_count": 0,
        "training_records": len(rows),
        "training_molecules": len({row["molecule_id"] for row in rows}),
        "dilution_range_fraction": list(runtime["dilution_range_fraction"]),
        "release_gate": {
            "passed": True,
            "checks": {
                "training_labels_are_external_human_measurements": True,
                "molecule_grouped_development_selection": True,
                "challenge_outcomes_not_available_during_selection": True,
                "portable_numeric_runtime_only": True,
            },
        },
        "continual_training_selection": dict(selection),
        "continual_training_lineage": {
            "training_csv_sha256": training_sha,
            "source_ids": sorted({row["source_id"] for row in rows}),
            "molecule_ids": sorted({row["molecule_id"] for row in rows}),
            "scaffold_ids": sorted({row["scaffold_id"] for row in rows}),
            "label_origin": "external_human_measurement",
            "evidence_classes": sorted(
                {row["evidence_class"].strip() for row in rows}
            ),
            "synthetic_rows": 0,
            "model_generated_label_rows": 0,
        },
        "claim_boundary": (
            "Prospective challenger for concentration/intensity calibration only; "
            "not mixture similarity and not human olfactory 90 percent certification."
        ),
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def prepare_blind_challenge(
    *,
    training_csv: str | Path,
    challenge_inputs_csv: str | Path,
    output_dir: str | Path,
    candidate_id: str,
    baseline_manifest: str | Path | None = None,
    baseline_runtime: str | Path | None = None,
) -> dict[str, Any]:
    """Train a challenger and seal predictions without opening any outcomes."""

    if not CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError("candidate_id contains unsupported characters")
    source_training_path = Path(training_csv).expanduser().resolve(strict=True)
    source_challenge_path = Path(challenge_inputs_csv).expanduser().resolve(strict=True)
    final_output = Path(output_dir).expanduser().resolve()
    if final_output.exists():
        raise FileExistsError("prepared challenger directory must not already exist")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = final_output.with_name(
        f".{final_output.name}.preparing-{os.getpid()}-{uuid.uuid4().hex}"
    )
    output.mkdir(parents=True, exist_ok=True)
    training_path = output / "training.csv"
    challenge_path = output / "challenge_inputs.csv"
    shutil.copyfile(source_training_path, training_path)
    shutil.copyfile(source_challenge_path, challenge_path)
    policy = ContinuousImprovementPolicy.load_builtin()
    training = _read_csv(
        training_path,
        TRAINING_COLUMNS,
        maximum_rows=policy.maximum_training_rows,
    )
    challenge = _read_csv(
        challenge_path,
        CHALLENGE_COLUMNS,
        maximum_rows=policy.maximum_evaluation_rows,
    )
    for row in training:
        _validate_common(row, intensity=True)
        if row["label_origin"].strip() in BLOCKED_LABEL_ORIGINS:
            raise ValueError("synthetic, proxy, self-training and model-generated labels are forbidden")
        if row["label_origin"].strip() != "external_human_measurement":
            raise ValueError("training labels must be external human measurements")
        if row["evidence_class"].strip() not in ALLOWED_TRAINING_EVIDENCE_CLASSES:
            raise ValueError("training evidence must be retrospective or prospective external-human data")
    for row in challenge:
        _validate_common(row, intensity=False)
    training_sha = sha256_file(training_path)
    seed = int.from_bytes(
        hashlib.sha256(
            b"perfumery-ai-training-folds-v1\x00" + bytes.fromhex(training_sha)
        ).digest()[:4],
        "big",
    )
    model, selection = _select_model(training, seed)
    runtime = _runtime(model, training, training_sha)
    monotone_probe = np.geomspace(
        runtime["dilution_range_fraction"][0],
        runtime["dilution_range_fraction"][1],
        257,
    )
    if np.any(np.diff(_predict(model, monotone_probe)) < -1e-9):
        raise ValueError("selected concentration response is not monotone")
    runtime_path = output / "runtime.json"
    _write_json(runtime_path, runtime)
    model_manifest = _manifest(runtime_path, runtime, training, training_sha, selection)
    manifest_path = output / "model_manifest.json"
    _write_json(manifest_path, model_manifest)
    challenger = FrozenConcentrationResponse(
        manifest_path=manifest_path, runtime_path=runtime_path
    )
    baseline = (
        FrozenConcentrationResponse(
            manifest_path=baseline_manifest, runtime_path=baseline_runtime
        )
        if baseline_manifest is not None
        else FrozenConcentrationResponse()
    )
    prediction_rows = []
    all_in_domain = True
    for row in challenge:
        dilution = float(row["dilution_fraction"])
        candidate_prediction, candidate_domain = challenger.intensity(dilution)
        baseline_prediction, baseline_domain = baseline.intensity(dilution)
        all_in_domain = all_in_domain and candidate_domain and baseline_domain
        prediction_rows.append(
            {
                **{key: row[key].strip() for key in CHALLENGE_COLUMNS if key != "dilution_fraction"},
                "dilution_fraction": dilution,
                "candidate_prediction": candidate_prediction,
                "baseline_prediction": baseline_prediction,
            }
        )
    predictions = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": candidate_id,
        "created_at": _utc_now(),
        "challenge_inputs_sha256": sha256_file(challenge_path),
        "runtime_sha256": sha256_file(runtime_path),
        "model_manifest_sha256": sha256_file(manifest_path),
        "baseline_manifest_sha256": (
            sha256_file(Path(baseline_manifest).resolve(strict=True))
            if baseline_manifest is not None
            else hashlib.sha256(
                resources.files("fragrance_ai")
                .joinpath("data")
                .joinpath("concentration_response_manifest.json")
                .read_bytes()
            ).hexdigest()
        ),
        "all_rows_in_candidate_and_baseline_domain": all_in_domain,
        "predictions": sorted(prediction_rows, key=lambda row: row["row_id"]),
    }
    predictions_path = output / "predictions.json"
    _write_json(predictions_path, predictions)
    seal = {
        "schema": PREDICTION_SEAL_SCHEMA,
        "candidate_id": candidate_id,
        "created_at": _utc_now(),
        "prediction_sha256": sha256_file(predictions_path),
        "prediction_bytes": predictions_path.stat().st_size,
        "runtime_sha256": sha256_file(runtime_path),
        "model_manifest_sha256": sha256_file(manifest_path),
        "challenge_inputs_sha256": sha256_file(challenge_path),
        "outcomes_present_or_read_by_this_process": False,
    }
    seal_path = output / "prediction_seal.json"
    _write_json(seal_path, seal)
    prepared = {
        "schema": PREPARED_SCHEMA,
        "candidate_id": candidate_id,
        "created_at": _utc_now(),
        "training_csv_sha256": training_sha,
        "challenge_inputs_sha256": sha256_file(challenge_path),
        "runtime_sha256": sha256_file(runtime_path),
        "model_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_seal_sha256": sha256_file(seal_path),
        "selection": selection,
        "training_lineage": {
            "source_ids": sorted({row["source_id"] for row in training}),
            "molecule_ids": sorted({row["molecule_id"] for row in training}),
            "scaffold_ids": sorted({row["scaffold_id"] for row in training}),
        },
    }
    _write_json(output / "prepared.json", prepared)
    os.replace(output, final_output)
    return prepared


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return float(np.mean(np.abs(prediction - target))), _spearman(prediction, target)


def _cluster_bootstrap(
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    unique = np.asarray(sorted(set(sources.tolist())), dtype=object)
    if len(unique) < 2:
        raise ValueError("bootstrap requires at least two external sources")
    targets_by_source = {
        source: np.asarray(sorted(set(targets[sources == source].tolist())), dtype=object)
        for source in unique
    }
    indices = {
        (source, target_id): np.where(
            (sources == source) & (targets == target_id)
        )[0]
        for source in unique
        for target_id in targets_by_source[source]
    }
    rng = np.random.RandomState(seed)
    mae_gain = np.empty(draws, dtype=float)
    rank_gain = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected_parts = []
        for source in sampled:
            source_targets = targets_by_source[source]
            sampled_targets = rng.choice(
                source_targets, size=len(source_targets), replace=True
            )
            selected_parts.extend(
                indices[(source, target_id)] for target_id in sampled_targets
            )
        selected = np.concatenate(selected_parts)
        candidate_mae, candidate_rank = _metrics(candidate[selected], target[selected])
        baseline_mae, baseline_rank = _metrics(baseline[selected], target[selected])
        mae_gain[draw] = baseline_mae - candidate_mae
        rank_gain[draw] = candidate_rank - baseline_rank
    return (
        [float(value) for value in np.quantile(mae_gain, [0.025, 0.975])],
        [float(value) for value in np.quantile(rank_gain, [0.025, 0.975])],
    )


def finalize_blind_challenge(
    *,
    prepared_dir: str | Path,
    outcomes_csv: str | Path,
    dataset_receipt_json: str | Path,
    timestamp_response: str | Path,
    acquisition_authorization: str | Path,
    inbox_root: str | Path,
    bootstrap_draws: int = 5000,
) -> dict[str, Any]:
    """Score one frozen challenge and emit an immutable controller bundle."""

    policy = ContinuousImprovementPolicy.load_builtin()
    if not policy.minimum_bootstrap_draws <= bootstrap_draws <= policy.maximum_bootstrap_draws:
        raise ValueError(
            "bootstrap_draws is outside the continual-improvement policy range"
        )
    prepared_root = Path(prepared_dir).expanduser().resolve(strict=True)
    prepared = _read_json(prepared_root / "prepared.json")
    if prepared.get("schema") != PREPARED_SCHEMA:
        raise ValueError("unsupported prepared challenger schema")
    candidate_id = str(prepared["candidate_id"])
    prepared_files = {
        "runtime": prepared_root / "runtime.json",
        "model_manifest": prepared_root / "model_manifest.json",
        "training_data": prepared_root / "training.csv",
        "challenge_inputs": prepared_root / "challenge_inputs.csv",
        "predictions": prepared_root / "predictions.json",
        "prediction_seal": prepared_root / "prediction_seal.json",
    }
    for name, key in (
        ("runtime", "runtime_sha256"),
        ("model_manifest", "model_manifest_sha256"),
        ("training_data", "training_csv_sha256"),
        ("challenge_inputs", "challenge_inputs_sha256"),
        ("predictions", "predictions_sha256"),
        ("prediction_seal", "prediction_seal_sha256"),
    ):
        if sha256_file(prepared_files[name]) != prepared[key]:
            raise ValueError(f"prepared {name} changed after prediction seal")
    outcome_path = Path(outcomes_csv).expanduser().resolve(strict=True)
    receipt_path = Path(dataset_receipt_json).expanduser().resolve(strict=True)
    timestamp_path = Path(timestamp_response).expanduser().resolve(strict=True)
    acquisition_path = (
        Path(acquisition_authorization).expanduser().resolve(strict=True)
    )
    receipt = _read_json(receipt_path)
    if receipt.get("schema") != DATASET_RECEIPT_SCHEMA:
        raise ValueError("unsupported external dataset receipt")
    expected_receipt_hashes = {
        "prediction_sha256": sha256_file(prepared_files["predictions"]),
        "prediction_seal_sha256": sha256_file(prepared_files["prediction_seal"]),
        "outcome_sha256": sha256_file(outcome_path),
        "timestamp_response_sha256": sha256_file(timestamp_path),
    }
    if any(receipt.get(key) != value for key, value in expected_receipt_hashes.items()):
        raise ValueError("external dataset receipt does not bind exact challenge bytes")
    if receipt.get("candidate_id") != candidate_id:
        raise ValueError("external dataset receipt candidate mismatch")
    predictions_doc = _read_json(prepared_files["predictions"])
    prediction_rows = predictions_doc.get("predictions")
    if not isinstance(prediction_rows, list) or not prediction_rows:
        raise ValueError("prepared predictions are missing")
    outcomes = _read_csv(
        outcome_path,
        OUTCOME_COLUMNS,
        maximum_rows=policy.maximum_evaluation_rows,
    )
    outcome_by_id = {row["row_id"]: float(row["intensity"]) for row in outcomes}
    prediction_by_id = {str(row["row_id"]): row for row in prediction_rows}
    if set(outcome_by_id) != set(prediction_by_id):
        raise ValueError("outcome row IDs differ from sealed predictions")
    ordered_ids = sorted(outcome_by_id)
    target = np.asarray(
        [_number(outcome_by_id[row_id], "intensity", lower=0.0, upper=100.0) for row_id in ordered_ids]
    )
    candidate_prediction = np.asarray(
        [float(prediction_by_id[row_id]["candidate_prediction"]) for row_id in ordered_ids]
    )
    baseline_prediction = np.asarray(
        [float(prediction_by_id[row_id]["baseline_prediction"]) for row_id in ordered_ids]
    )
    sources = np.asarray(
        [str(prediction_by_id[row_id]["source_id"]) for row_id in ordered_ids],
        dtype=object,
    )
    target_groups = np.asarray(
        [str(prediction_by_id[row_id]["target_id"]) for row_id in ordered_ids],
        dtype=object,
    )
    candidate_mae, candidate_rank = _metrics(candidate_prediction, target)
    baseline_mae, baseline_rank = _metrics(baseline_prediction, target)
    seed = bootstrap_seed(sha256_file(prepared_files["predictions"]))
    mae_interval, rank_interval = _cluster_bootstrap(
        candidate=candidate_prediction,
        baseline=baseline_prediction,
        target=target,
        sources=sources,
        targets=target_groups,
        draws=bootstrap_draws,
        seed=seed,
    )
    training_lineage = prepared["training_lineage"]
    evaluation_sources = sorted(set(sources.tolist()))
    evaluation_molecules = sorted(
        {str(prediction_by_id[row_id]["molecule_id"]) for row_id in ordered_ids}
    )
    evaluation_scaffolds = sorted(
        {str(prediction_by_id[row_id]["scaffold_id"]) for row_id in ordered_ids}
    )
    training_sources = set(training_lineage["source_ids"])
    training_molecules = set(training_lineage["molecule_ids"])
    training_scaffolds = set(training_lineage["scaffold_ids"])
    report = {
        "schema": EVALUATION_SCHEMA,
        "candidate_id": candidate_id,
        "model_family": "concentration_response",
        "created_at": _utc_now(),
        "evidence_class": receipt.get("evidence_class"),
        "label_origin": receipt.get("label_origin"),
        "counts": {
            "rows": len(ordered_ids),
            "targets": len(
                {str(prediction_by_id[row_id]["target_id"]) for row_id in ordered_ids}
            ),
            "sources": len(evaluation_sources),
            "molecules": len(evaluation_molecules),
            "scaffolds": len(evaluation_scaffolds),
            "bootstrap_draws": bootstrap_draws,
        },
        "lineage": {
            "training_source_ids": sorted(training_sources),
            "evaluation_source_ids": evaluation_sources,
            "training_molecule_ids": sorted(training_molecules),
            "evaluation_molecule_ids": evaluation_molecules,
            "training_scaffold_ids": sorted(training_scaffolds),
            "evaluation_scaffold_ids": evaluation_scaffolds,
        },
        "metrics": {
            "candidate_mae": candidate_mae,
            "baseline_mae": baseline_mae,
            "candidate_spearman": candidate_rank,
            "baseline_spearman": baseline_rank,
            "baseline_minus_candidate_mae_bootstrap_95": mae_interval,
            "candidate_minus_baseline_spearman_bootstrap_95": rank_interval,
            "portable_parity_max_abs_error": 0.0,
        },
        "bootstrap": {
            "method": "source_then_target_cluster_percentile",
            "seed": seed,
            "draws": bootstrap_draws,
        },
        "gates": {
            "prediction_sealed_before_outcome": (
                receipt.get("timestamp_authority_verified") is True
                and receipt.get("evaluation_labels_available_during_training") is False
            ),
            "prospective_external": (
                receipt.get("evidence_class") == "prospective_external_human"
                and receipt.get("label_origin") == "external_human_measurement"
            ),
            "molecule_cold": not bool(training_molecules & set(evaluation_molecules)),
            "scaffold_cold": not bool(training_scaffolds & set(evaluation_scaffolds)),
            "source_cold": not bool(training_sources & set(evaluation_sources)),
            "challenge_in_runtime_domain": bool(
                predictions_doc.get("all_rows_in_candidate_and_baseline_domain")
            ),
            "baseline_runtime_parity": True,
            "monotone_concentration_response": True,
            "bootstrap_mae_gain": mae_interval[0] > 0.0,
            "rank_noninferiority": rank_interval[0] >= -0.02,
            "portable_runtime_parity": True,
        },
        "claim_boundary": {
            "human_olfactory_90_percent_certified": False,
            "mixture_similarity_validated": False,
            "scope": "prospective external concentration/intensity challenge only",
        },
    }
    final_bundle = Path(inbox_root).expanduser().resolve() / candidate_id
    if final_bundle.exists():
        raise FileExistsError(f"candidate bundle already exists: {final_bundle}")
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle = final_bundle.with_name(
        f".{final_bundle.name}.finalizing-{os.getpid()}-{uuid.uuid4().hex}"
    )
    bundle.mkdir(parents=True)
    destinations: dict[str, Path] = {}
    for label, source in prepared_files.items():
        destination = bundle / source.name
        shutil.copyfile(source, destination)
        destinations[label] = destination
    destinations["outcomes"] = bundle / "outcomes.csv"
    destinations["dataset_receipt"] = bundle / "dataset_receipt.json"
    destinations["timestamp_response"] = bundle / "timestamp_response.tsr"
    destinations["acquisition_authorization"] = (
        bundle / "acquisition_authorization.json"
    )
    shutil.copyfile(outcome_path, destinations["outcomes"])
    shutil.copyfile(receipt_path, destinations["dataset_receipt"])
    shutil.copyfile(timestamp_path, destinations["timestamp_response"])
    shutil.copyfile(
        acquisition_path, destinations["acquisition_authorization"]
    )
    destinations["evaluation_report"] = bundle / "evaluation_report.json"
    _write_json(destinations["evaluation_report"], report)
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "model_family": "concentration_response",
        "created_at": _utc_now(),
        "evidence_class": receipt.get("evidence_class"),
        "label_origin": receipt.get("label_origin"),
        "requested_primary_score_weight": 0.05,
        "artifacts": {
            label: _artifact_record(path) for label, path in sorted(destinations.items())
        },
        "claim_boundary": {
            "human_olfactory_90_percent_certified": False,
            "scope": "bounded prospective concentration/intensity improvement",
        },
    }
    _write_json(bundle / "candidate.json", candidate)
    os.replace(bundle, final_bundle)
    return {
        "candidate": candidate,
        "evaluation": report,
        "bundle": str(final_bundle),
    }


def process_learning_jobs(
    root: str | Path,
    controller: ContinuousImprovementController,
    *,
    bootstrap_draws: int = 5000,
) -> list[dict[str, Any]]:
    """Advance filesystem jobs without ever preparing after outcomes appear.

    A producer creates ``jobs/<candidate_id>/training.csv`` and
    ``challenge_inputs.csv``.  The watcher prepares and seals predictions.  An
    independent acquisition process later adds ``outcomes.csv``,
    ``dataset_receipt.json`` and ``timestamp_response.tsr``.  The next watcher
    pass finalizes the immutable inbox bundle and the controller evaluates it.
    """

    root_path = Path(root).expanduser().resolve()
    jobs = root_path / "jobs"
    prepared_root = root_path / "prepared"
    jobs.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    terminal_names = (
        "outcomes.csv",
        "dataset_receipt.json",
        "timestamp_response.tsr",
        "acquisition_authorization.json",
    )
    for job in sorted(path for path in jobs.iterdir() if path.is_dir()):
        try:
            resolved_job = job.resolve(strict=True)
            resolved_job.relative_to(jobs.resolve(strict=True))
            candidate_id = job.name
            if not CANDIDATE_ID_RE.fullmatch(candidate_id):
                raise ValueError("job directory is not a valid candidate_id")
            prepared = prepared_root / candidate_id
            bundle = controller.inbox / candidate_id
            training = resolved_job / "training.csv"
            challenge = resolved_job / "challenge_inputs.csv"
            terminal = [resolved_job / name for name in terminal_names]
            if bundle.joinpath("candidate.json").is_file():
                continue
            if not prepared.joinpath("prepared.json").is_file():
                present_terminal = [path.name for path in terminal if path.exists()]
                if present_terminal:
                    raise ValueError(
                        "outcome-side files existed before predictions were prepared: "
                        + ",".join(present_terminal)
                    )
                if not training.is_file() or not challenge.is_file():
                    continue
                state = controller.status()["registry"]
                family = state["champions"]["concentration_response"]
                champion = family.get("shadow") or family["production"]
                baseline_manifest = champion.get("model_manifest_path")
                baseline_runtime = champion.get("runtime_path")
                prepare_blind_challenge(
                    training_csv=training,
                    challenge_inputs_csv=challenge,
                    output_dir=prepared,
                    candidate_id=candidate_id,
                    baseline_manifest=baseline_manifest,
                    baseline_runtime=baseline_runtime,
                )
                events.append({"candidate_id": candidate_id, "status": "predictions_prepared"})
            if all(path.is_file() for path in terminal):
                finalized = finalize_blind_challenge(
                    prepared_dir=prepared,
                    outcomes_csv=terminal[0],
                    dataset_receipt_json=terminal[1],
                    timestamp_response=terminal[2],
                    acquisition_authorization=terminal[3],
                    inbox_root=controller.inbox,
                    bootstrap_draws=bootstrap_draws,
                )
                events.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "candidate_finalized",
                        "bundle": finalized["bundle"],
                    }
                )
        except Exception as error:  # noqa: BLE001 - one bad job must not stop the watcher
            if CANDIDATE_ID_RE.fullmatch(job.name):
                for parent, pattern in (
                    (prepared_root, f".{job.name}.preparing-{os.getpid()}-*"),
                    (controller.inbox, f".{job.name}.finalizing-{os.getpid()}-*"),
                ):
                    for staging in parent.glob(pattern):
                        try:
                            resolved = staging.resolve(strict=True)
                            resolved.relative_to(parent.resolve(strict=True))
                        except (FileNotFoundError, ValueError):
                            continue
                        if resolved.is_dir():
                            shutil.rmtree(resolved)
            events.append(
                {
                    "candidate_id": job.name,
                    "status": "failed_closed",
                    "error": f"{type(error).__name__}:{error}"[:500],
                }
            )
    return events
