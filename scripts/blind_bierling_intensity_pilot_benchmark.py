#!/usr/bin/env python
"""Blind concentration/intensity transfer on the unopened Bierling pilot.

``prepare`` learns a portable concentration-aware intensity curve from Keller
2016 while excluding every Bierling target molecule from human supervision.
It also creates a separately labelled condition-transfer curve anchored to the
already-opened Bierling main-study intensity means.  Both curves are evaluated
on a fixed concentration grid before the independent ``intensity_piloting.csv``
file is downloaded.  ``seal`` and ``acquire`` enforce an RFC 3161 timestamped
ordering; ``score`` evaluates the frozen curves once.

This is monomolecular perceived intensity, not mixture or perfume similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender.concentration_response import (  # noqa: E402
    FrozenConcentrationResponse,
)
from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402


SCHEMA_VERSION = "1.0"
PILOT_URL = (
    "https://zenodo.org/api/records/15657278/files/"
    "intensity_piloting.csv/content"
)
PILOT_FILE = "intensity_piloting.csv"
PILOT_BYTES = 37_054
PILOT_MD5 = "3f7e0317b8e05c84545a1d9adb00e93c"
GRID_LOG10 = np.linspace(-6.0, 0.0, 241, dtype=float)
GRID_FRACTIONS = np.power(10.0, GRID_LOG10)
REFERENCE_FRACTION = 0.001
CV_FOLDS = 5
CV_SALT = "bierling-intensity-target-excluded-keller-v1"
BOOTSTRAP_SEED = 20_260_827
BOOTSTRAP_DRAWS = 1_000
FIXED_BASELINE = "concentration_ridge_alpha_100"
CANDIDATES: tuple[dict[str, Any], ...] = tuple(
    {
        "name": f"{feature}_ridge_alpha_{alpha}",
        "feature": feature,
        "alpha": float(alpha),
    }
    for feature in (
        "concentration",
        "rdkit",
        "rdkit_interaction",
        "morgan",
        "molformer",
        "fusion",
    )
    for alpha in (10, 100, 1000, 10000)
)
MAIN_OUTCOME_SHA256 = (
    "4e7ec47089cfc43df3e008ed558ffd1ee05d23f51c364e5e2538ce247ef163a4"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _md5(path: Path) -> str:
    return shared.md5_file(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    shared.write_json(path, value)


def _identifier(value: object) -> str:
    text = str(value).strip().replace("\u00a0", "")
    return text[:-2] if text.endswith(".0") else text


def parse_concentration(value: object) -> float:
    text = str(value).strip().lower().replace(" ", "").replace(",", ".")
    if text == "undiluted":
        return 1.0
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)/([0-9]*\.?[0-9]+)", text)
    if match is None:
        raise ValueError(f"unsupported dimensionless concentration: {value!r}")
    numerator, denominator = (float(item) for item in match.groups())
    result = numerator / denominator
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"concentration outside (0, 1]: {value!r}")
    return result


def assert_pilot_absent(path: Path) -> None:
    parent = path.resolve().parent
    present = [
        name
        for name in ("intensity_piloting.csv", "intensity_piloting.xlsx")
        if (parent / name).exists()
    ]
    if present:
        raise RuntimeError("pilot outcome exists before acquisition: " + ",".join(present))


def _ravia_curve(fractions: np.ndarray) -> np.ndarray:
    adapter = FrozenConcentrationResponse()
    values = np.asarray([adapter.intensity(float(value))[0] for value in fractions])
    if not np.all(np.isfinite(values)) or not np.all((0.0 <= values) & (values <= 100.0)):
        raise RuntimeError("frozen Ravia curve is invalid")
    return values


def _load_keller_intensity(
    molecules_path: Path,
    stimuli_path: Path,
    behavior_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    molecules = pd.read_csv(molecules_path)
    stimuli = pd.read_csv(stimuli_path)
    behavior = pd.read_csv(
        behavior_path,
        usecols=["Stimulus", "Subject", "MeasurementValue", "Value"],
        low_memory=False,
    )
    cid_to_smiles = {}
    for _, row in molecules.iterrows():
        cid = _identifier(row["CID"])
        raw = str(row["CanonicalSMILES"]).strip()
        if cid and raw and raw.lower() != "nan":
            cid_to_smiles[cid] = shared._canonical_smiles(raw)
    stimuli = stimuli.copy()
    stimuli["Stimulus"] = stimuli["Stimulus"].astype(int)
    stimuli["cid"] = stimuli["CIDs"].map(_identifier)
    stimuli["Concentration"] = pd.to_numeric(stimuli["Concentration"], errors="raise")
    intensity = behavior[
        behavior["MeasurementValue"].eq("HOW STRONG IS THE SMELL?")
    ].copy()
    intensity["Stimulus"] = intensity["Stimulus"].astype(int)
    intensity["numeric"] = pd.to_numeric(intensity["Value"], errors="coerce")
    means = intensity.groupby("Stimulus")["numeric"].mean()
    rows = []
    for _, stimulus in stimuli.iterrows():
        identifier = int(stimulus["Stimulus"])
        smiles = cid_to_smiles.get(stimulus["cid"])
        if smiles is None or identifier not in means or not np.isfinite(means[identifier]):
            continue
        rows.append(
            {
                "canonical_smiles": smiles,
                "cid": stimulus["cid"],
                "concentration_fraction": float(stimulus["Concentration"]),
                "intensity": float(means[identifier]),
                "stimulus": identifier,
            }
        )
    if len(rows) < 900:
        raise RuntimeError("too few Keller concentration/intensity rows")
    return rows, {
        "rows": len(rows),
        "molecules": len({row["canonical_smiles"] for row in rows}),
        "subjects": int(intensity["Subject"].nunique()),
        "concentration_minimum": min(row["concentration_fraction"] for row in rows),
        "concentration_maximum": max(row["concentration_fraction"] for row in rows),
    }


def _load_main_intensity_anchors(
    path: Path, target_rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    import pandas as pd

    if _sha256(path) != MAIN_OUTCOME_SHA256:
        raise RuntimeError("Bierling main outcome differs from frozen acquisition")
    frame = pd.read_csv(path, sep=";", low_memory=False)
    selected = frame[
        pd.to_numeric(frame["inclusion"], errors="coerce").eq(1)
        & frame["study"].astype(str).str.strip().str.lower().eq("main")
        & frame["sampling_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"home", "lab"})
    ].copy()
    selected["molcode"] = selected["molcode"].astype(str).str.strip()
    selected["intensive"] = pd.to_numeric(selected["intensive"], errors="coerce")
    means = selected.groupby("molcode")["intensive"].mean().to_dict()
    allowed = {str(row["molcode"]) for row in target_rows}
    return {
        str(key): float(value)
        for key, value in means.items()
        if key in allowed and np.isfinite(value)
    }


def _folds(smiles: Sequence[str]) -> np.ndarray:
    unique = sorted(
        set(smiles),
        key=lambda value: hashlib.sha256(f"{CV_SALT}|{value}".encode()).hexdigest(),
    )
    mapping = {value: index % CV_FOLDS for index, value in enumerate(unique)}
    return np.asarray([mapping[value] for value in smiles], dtype=int)


def _row_features(
    kind: str,
    molecule_features: Mapping[str, np.ndarray],
    molecule_indices: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    logc = np.log10(np.clip(fractions, 1e-6, 1.0))[:, None]
    logc2 = logc * logc
    ravia = _ravia_curve(fractions)[:, None]
    if kind == "concentration":
        return np.concatenate((logc, logc2, ravia), axis=1)
    base_name = "fusion" if kind == "fusion" else kind.replace("_interaction", "")
    base = molecule_features[base_name][molecule_indices]
    pieces = [base, logc, logc2, ravia]
    if kind == "rdkit_interaction":
        pieces.append(base * logc)
    return np.concatenate(pieces, axis=1)


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, *, alpha: float
) -> tuple[dict[str, Any], np.ndarray]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    normalized = scaler.fit_transform(x)
    model = Ridge(alpha=alpha).fit(normalized, y)
    portable = {
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "alpha": float(alpha),
    }
    return portable, np.asarray(model.predict(normalized), dtype=float)


def _portable_predict(parameters: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(parameters["feature_mean"], dtype=float)
    scale = np.asarray(parameters["feature_scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    intercept = float(parameters["intercept"])
    result = ((x - mean) / scale) @ coefficients + intercept
    return np.clip(np.asarray(result, dtype=float), 0.0, 100.0)


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    smiles: Sequence[str],
    fractions: np.ndarray,
) -> dict[str, float]:
    row_spearman = shared.spearman(prediction, target)
    deltas_predicted = []
    deltas_target = []
    direction = []
    molecule_correlations = []
    for molecule in sorted(set(smiles)):
        indices = np.flatnonzero(np.asarray(smiles) == molecule)
        order = indices[np.argsort(fractions[indices])]
        if len(order) < 2:
            continue
        molecule_correlations.append(
            shared.spearman(prediction[order], target[order])
            if len(order) >= 3
            else float(np.sign(np.diff(prediction[order])[0]) == np.sign(np.diff(target[order])[0]))
        )
        for low_position in range(len(order) - 1):
            for high_position in range(low_position + 1, len(order)):
                low = order[low_position]
                high = order[high_position]
                predicted_delta = float(prediction[high] - prediction[low])
                target_delta = float(target[high] - target[low])
                deltas_predicted.append(predicted_delta)
                deltas_target.append(target_delta)
                direction.append(predicted_delta * target_delta >= 0.0)
    delta_spearman = (
        shared.spearman(deltas_predicted, deltas_target)
        if len(deltas_predicted) >= 3
        else 0.0
    )
    return {
        "row_spearman": row_spearman,
        "delta_spearman": delta_spearman,
        "direction_accuracy": float(np.mean(direction)) if direction else 0.0,
        "mean_molecule_concentration_spearman": (
            float(np.mean(molecule_correlations)) if molecule_correlations else 0.0
        ),
        "mae": float(np.mean(np.abs(prediction - target))),
        "selection_score": 0.7 * row_spearman + 0.3 * delta_spearman,
    }


def _monotonic_by_molecule(
    prediction: np.ndarray,
    smiles: Sequence[str],
    fractions: np.ndarray,
) -> np.ndarray:
    result = np.asarray(prediction, dtype=float).copy()
    smiles_array = np.asarray(smiles)
    for molecule in sorted(set(smiles)):
        indices = np.flatnonzero(smiles_array == molecule)
        order = indices[np.argsort(fractions[indices])]
        result[order] = np.maximum.accumulate(result[order])
    return result


def _development(
    rows: Sequence[Mapping[str, Any]],
    all_smiles: Sequence[str],
    molecule_features: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    index = {value: position for position, value in enumerate(all_smiles)}
    row_smiles = [str(row["canonical_smiles"]) for row in rows]
    molecule_indices = np.asarray([index[value] for value in row_smiles], dtype=int)
    fractions = np.asarray([row["concentration_fraction"] for row in rows], dtype=float)
    target = np.asarray([row["intensity"] for row in rows], dtype=float)
    folds = _folds(row_smiles)
    oof = {
        specification["name"]: np.full(len(rows), np.nan, dtype=float)
        for specification in CANDIDATES
    }
    for fold in range(CV_FOLDS):
        train = np.flatnonzero(folds != fold)
        validate = np.flatnonzero(folds == fold)
        for specification in CANDIDATES:
            x = _row_features(
                specification["feature"], molecule_features, molecule_indices, fractions
            )
            parameters, _ = _fit_ridge(x[train], target[train], alpha=specification["alpha"])
            oof[specification["name"]][validate] = _portable_predict(parameters, x[validate])
    oof = {
        name: _monotonic_by_molecule(prediction, row_smiles, fractions)
        for name, prediction in oof.items()
    }
    results = {
        name: _metrics(prediction, target, row_smiles, fractions)
        for name, prediction in oof.items()
    }
    selected_name = max(
        results,
        key=lambda name: (
            results[name]["selection_score"],
            -results[name]["mae"],
            -next(index for index, row in enumerate(CANDIDATES) if row["name"] == name),
        ),
    )
    specifications = {row["name"]: row for row in CANDIDATES}
    selected_spec = specifications[selected_name]
    fixed_spec = specifications[FIXED_BASELINE]
    selected_x = _row_features(
        selected_spec["feature"], molecule_features, molecule_indices, fractions
    )
    fixed_x = _row_features(
        fixed_spec["feature"], molecule_features, molecule_indices, fractions
    )
    selected_parameters, _ = _fit_ridge(
        selected_x, target, alpha=selected_spec["alpha"]
    )
    fixed_parameters, _ = _fit_ridge(fixed_x, target, alpha=fixed_spec["alpha"])
    return (
        {
            "folds": CV_FOLDS,
            "fold_salt_sha256": hashlib.sha256(CV_SALT.encode()).hexdigest(),
            "rows": len(rows),
            "molecules": len(set(row_smiles)),
            "candidate_results": results,
            "selected_candidate": selected_name,
            "fixed_baseline": FIXED_BASELINE,
            "selected_minus_fixed_selection_score": (
                results[selected_name]["selection_score"]
                - results[FIXED_BASELINE]["selection_score"]
            ),
        },
        {"specification": selected_spec, "parameters": selected_parameters},
        {"specification": fixed_spec, "parameters": fixed_parameters},
    )


def _predict_grid(
    portable: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    molecule_index: int,
) -> np.ndarray:
    indices = np.full(len(GRID_FRACTIONS), molecule_index, dtype=int)
    x = _row_features(
        portable["specification"]["feature"], features, indices, GRID_FRACTIONS
    )
    return _portable_predict(portable["parameters"], x)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    pilot_path = args.forbidden_pilot.resolve()
    assert_pilot_absent(pilot_path)
    if args.predictions.resolve().exists():
        raise RuntimeError("refusing to overwrite intensity blind predictions")
    target_odors = args.target_odors.resolve(strict=True)
    if (
        target_odors.stat().st_size != shared.TARGET_METADATA_BYTES
        or _sha256(target_odors) != shared.TARGET_METADATA_SHA256
    ):
        raise RuntimeError("target odor metadata hash mismatch")
    target_rows = shared.load_target_odors(target_odors)
    target_set = {row["canonical_smiles"] for row in target_rows}
    keller_paths = {
        "molecules.csv": args.keller_molecules.resolve(strict=True),
        "stimuli.csv": args.keller_stimuli.resolve(strict=True),
        "behavior.csv": args.keller_behavior.resolve(strict=True),
    }
    for name, path in keller_paths.items():
        expected = shared.KELLER_SOURCE_CONTRACT[name]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Keller intensity source hash mismatch: {name}")
    keller_rows, keller_audit = _load_keller_intensity(
        keller_paths["molecules.csv"],
        keller_paths["stimuli.csv"],
        keller_paths["behavior.csv"],
    )
    all_keller_molecules = {row["canonical_smiles"] for row in keller_rows}
    keller_rows = [
        row for row in keller_rows if row["canonical_smiles"] not in target_set
    ]
    if len({row["canonical_smiles"] for row in keller_rows}) < 300:
        raise RuntimeError("target-excluded Keller intensity training is too small")
    all_smiles = sorted(
        {row["canonical_smiles"] for row in keller_rows} | target_set
    )
    features, feature_audit = shared._feature_matrices(
        all_smiles,
        molformer_root=args.molformer_root.resolve(strict=True),
        hf_home=args.hf_home.resolve(),
        batch_size=args.batch_size,
        torch_threads=args.threads,
    )
    development, primary_portable, fixed_portable = _development(
        keller_rows, all_smiles, features
    )
    anchors = _load_main_intensity_anchors(
        args.main_outcome.resolve(strict=True), target_rows
    )
    index = {value: position for position, value in enumerate(all_smiles)}
    ravia_grid = np.maximum.accumulate(_ravia_curve(GRID_FRACTIONS))
    prediction_rows = []
    anchor_fallbacks = []
    for row in target_rows:
        molecule_index = index[row["canonical_smiles"]]
        strict_curve = np.maximum.accumulate(
            _predict_grid(primary_portable, features, molecule_index)
        )
        fixed_curve = np.maximum.accumulate(
            _predict_grid(fixed_portable, features, molecule_index)
        )
        structure_only = float(
            np.interp(
                math_log10(REFERENCE_FRACTION), GRID_LOG10, strict_curve
            )
        )
        anchor = anchors.get(row["molcode"])
        try:
            final_fraction = parse_concentration(row["concentration_final"])
        except ValueError:
            final_fraction = None
        if anchor is not None and final_fraction is not None:
            final_ravia = float(
                np.interp(math_log10(final_fraction), GRID_LOG10, ravia_grid)
            )
            anchored_curve = np.clip(anchor + ravia_grid - final_ravia, 0.0, 100.0)
            anchor_status = "main_human_intensity_plus_frozen_ravia_delta"
        else:
            anchored_curve = strict_curve.copy()
            anchor_status = "strict_structure_curve_fallback"
            anchor_fallbacks.append(row["molcode"])
        prediction_rows.append(
            {
                **row,
                "grid_log10_fraction": [round(float(value), 6) for value in GRID_LOG10],
                "strict_structure_concentration_curve": [
                    round(float(value), 8) for value in strict_curve
                ],
                "condition_transfer_anchored_curve": [
                    round(float(value), 8) for value in anchored_curve
                ],
                "fixed_keller_concentration_baseline": [
                    round(float(value), 8) for value in fixed_curve
                ],
                "frozen_ravia_global_curve": [
                    round(float(value), 8) for value in ravia_grid
                ],
                "structure_only_intensity": round(structure_only, 8),
                "main_intensity_anchor": anchor,
                "main_anchor_final_fraction": final_fraction,
                "anchor_status": anchor_status,
            }
        )
    script = Path(__file__).resolve()
    checks = {
        "pilot_outcome_absent": not pilot_path.exists(),
        "target_predictions_74": len(prediction_rows) == 74,
        "target_exact_keller_label_leakage_zero": not bool(
            target_set & {row["canonical_smiles"] for row in keller_rows}
        ),
        "development_selected_beats_fixed": development[
            "selected_minus_fixed_selection_score"
        ]
        > 0.01,
        "curves_finite": all(
            np.all(np.isfinite(row["strict_structure_concentration_curve"]))
            and np.all(np.isfinite(row["condition_transfer_anchored_curve"]))
            for row in prediction_rows
        ),
        "all_concentration_curves_monotonic": all(
            np.all(np.diff(row["strict_structure_concentration_curve"]) >= -1e-12)
            and np.all(np.diff(row["condition_transfer_anchored_curve"]) >= -1e-12)
            and np.all(np.diff(row["fixed_keller_concentration_baseline"]) >= -1e-12)
            and np.all(np.diff(row["frozen_ravia_global_curve"]) >= -1e-12)
            for row in prediction_rows
        ),
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "intensity_pilot_curves_ready_before_outcomes",
        "blind_contract": {
            "pilot_outcome_read": False,
            "pilot_outcome_used_for_model_selection": False,
            "parametric_curves_fixed_before_pilot_conditions_or_ratings": True,
            "strict_track_excludes_all_exact_target_human_labels": True,
            "condition_transfer_track_uses_main_study_anchors": True,
            "primary_metrics": [
                "condition_mean_spearman",
                "delta_spearman",
                "direction_accuracy",
            ],
            "prediction_rows_sha256": shared.canonical_json_sha256(prediction_rows),
        },
        "target": {
            "dataset_doi": shared.DATASET_DOI,
            "outcome_url": PILOT_URL,
            "expected_bytes": PILOT_BYTES,
            "expected_md5": PILOT_MD5,
            "downloaded": False,
            "target_odors_sha256": _sha256(target_odors),
        },
        "training": {
            "keller": keller_audit,
            "keller_source_sha256": {
                name: _sha256(path) for name, path in keller_paths.items()
            },
            "exact_target_molecules_excluded": len(all_keller_molecules & target_set),
            "target_excluded_rows": len(keller_rows),
            "target_excluded_molecules": len(
                {row["canonical_smiles"] for row in keller_rows}
            ),
            "main_anchor_outcome_sha256": MAIN_OUTCOME_SHA256,
            "main_anchor_molecules": len(anchors),
            "anchor_fallback_molcodes": anchor_fallbacks,
        },
        "model": {
            "development": development,
            "primary_portable": primary_portable,
            "fixed_portable": fixed_portable,
            "features": feature_audit,
            "grid_log10_minimum": float(GRID_LOG10[0]),
            "grid_log10_maximum": float(GRID_LOG10[-1]),
            "grid_points": len(GRID_LOG10),
            "ravia_manifest_sha256": _sha256(
                PROJECT_ROOT / "fragrance_ai" / "data" / "concentration_response_manifest.json"
            ),
            "ravia_runtime_sha256": _sha256(
                PROJECT_ROOT / "fragrance_ai" / "data" / "concentration_response_runtime.json"
            ),
        },
        "implementation": {
            "script_sha256": _sha256(script),
            "shared_script_sha256": _sha256(Path(shared.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "release_gate": {"passed": all(checks.values()), "checks": checks},
        "predictions": prediction_rows,
        "claim_boundary": (
            "Unopened-pilot monomolecular intensity/concentration curves; not "
            "mixture, recipe, product-matrix, or 90% olfactory validation."
        ),
    }
    if not document["release_gate"]["passed"]:
        raise RuntimeError(f"intensity curve release gate failed: {checks}")
    _write_json(args.predictions.resolve(), document)
    return document


def math_log10(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("log10 input must be positive and finite")
    return float(np.log10(value))


def create_seal(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve(strict=True)
    pilot_path = args.pilot.resolve()
    assert_pilot_absent(pilot_path)
    if args.seal.resolve().exists():
        raise RuntimeError("refusing to overwrite intensity pilot seal")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if predictions.get("status") != "intensity_pilot_curves_ready_before_outcomes":
        raise RuntimeError("intensity predictions are not sealable")
    if predictions.get("release_gate", {}).get("passed") is not True:
        raise RuntimeError("intensity prediction release gate is closed")
    rows_hash = shared.canonical_json_sha256(predictions.get("predictions", []))
    if rows_hash != predictions.get("blind_contract", {}).get("prediction_rows_sha256"):
        raise RuntimeError("intensity prediction row hash mismatch")
    script_hash = _sha256(Path(__file__).resolve())
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("intensity benchmark script changed after prepare")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": utc_now(),
        "prediction_sha256": _sha256(predictions_path),
        "prediction_bytes": predictions_path.stat().st_size,
        "prediction_rows_sha256": rows_hash,
        "script_sha256": script_hash,
        "pilot": {
            "path": str(pilot_path),
            "url": PILOT_URL,
            "expected_bytes": PILOT_BYTES,
            "expected_md5": PILOT_MD5,
            "present_before_seal": False,
        },
        "score_contract": {
            "grid_interpolation": "linear_in_log10_concentration",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
    }
    _write_json(args.seal.resolve(), seal)
    return seal


def _verify_seal(predictions_path: Path, seal_path: Path) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("prediction_sha256") != _sha256(predictions_path):
        raise RuntimeError("sealed intensity prediction hash mismatch")
    if seal.get("prediction_bytes") != predictions_path.stat().st_size:
        raise RuntimeError("sealed intensity prediction size mismatch")
    if seal.get("script_sha256") != _sha256(Path(__file__).resolve()):
        raise RuntimeError("intensity script changed after seal")
    rows_hash = shared.canonical_json_sha256(predictions.get("predictions", []))
    if rows_hash != seal.get("prediction_rows_sha256"):
        raise RuntimeError("sealed intensity prediction rows mismatch")
    return {"predictions": predictions, "seal": seal}


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    verified = _verify_seal(args.predictions, args.seal)
    pilot_path = args.pilot.resolve()
    if pilot_path != Path(verified["seal"]["pilot"]["path"]).resolve():
        raise RuntimeError("pilot path differs from seal")
    assert_pilot_absent(pilot_path)
    if args.receipt.resolve().exists():
        raise RuntimeError("refusing to overwrite intensity acquisition receipt")
    timestamp = shared.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    started = utc_now()
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="bierling-intensity-", suffix=".part", dir=pilot_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            PILOT_URL, headers={"User-Agent": "Perfumery-AI intensity blind/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if temporary.stat().st_size != PILOT_BYTES or _md5(temporary) != PILOT_MD5:
            raise RuntimeError("downloaded intensity pilot differs from Zenodo metadata")
        if pilot_path.exists():
            raise RuntimeError("intensity pilot appeared during acquisition")
        os.rename(temporary, pilot_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "download_started_at": started,
        "download_completed_at": utc_now(),
        "url": PILOT_URL,
        "pilot_path": str(pilot_path),
        "pilot_bytes": pilot_path.stat().st_size,
        "pilot_md5": _md5(pilot_path),
        "pilot_sha256": _sha256(pilot_path),
        "prediction_sha256": _sha256(args.predictions.resolve(strict=True)),
        "seal_sha256": _sha256(args.seal.resolve(strict=True)),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "timestamp": timestamp,
    }
    _write_json(args.receipt.resolve(), receipt)
    return receipt


def _curve_prediction(row: Mapping[str, Any], field: str, fraction: float) -> float:
    grid = np.asarray(row["grid_log10_fraction"], dtype=float)
    curve = np.asarray(row[field], dtype=float)
    logc = math_log10(fraction)
    if logc < grid[0] or logc > grid[-1]:
        raise RuntimeError("pilot concentration lies outside frozen prediction grid")
    return float(np.interp(logc, grid, curve))


def _score_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    molcodes: Sequence[str],
    fractions: np.ndarray,
) -> dict[str, float]:
    return _metrics(prediction, target, molcodes, fractions)


def _bootstrap(
    raw: Any,
    condition_keys: Sequence[tuple[str, float]],
    primary: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    participants = sorted(raw["code"].astype(str).unique().tolist())
    p_index = {value: index for index, value in enumerate(participants)}
    c_index = {value: index for index, value in enumerate(condition_keys)}
    values = np.zeros((len(participants), len(condition_keys)), dtype=float)
    observed = np.zeros_like(values)
    for _, row in raw.iterrows():
        key = (row["molcode"], float(row["fraction"]))
        values[p_index[str(row["code"])], c_index[key]] = float(row["intensive"])
        observed[p_index[str(row["code"])], c_index[key]] = 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    primary_scores = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(
            len(participants), np.full(len(participants), 1.0 / len(participants))
        )
        counts = weights @ observed
        if np.any(counts <= 0):
            continue
        means = (weights @ values) / counts
        sampled = rng.integers(0, len(condition_keys), size=len(condition_keys))
        primary_score = shared.spearman(primary[sampled], means[sampled])
        baseline_score = shared.spearman(baseline[sampled], means[sampled])
        primary_scores.append(primary_score)
        deltas.append(primary_score - baseline_score)
    if len(deltas) < int(BOOTSTRAP_DRAWS * 0.95):
        raise RuntimeError("too many invalid intensity bootstrap draws")
    return {
        "seed": BOOTSTRAP_SEED,
        "valid_draws": len(deltas),
        "primary_95_interval": [
            float(value) for value in np.quantile(primary_scores, [0.025, 0.975])
        ],
        "primary_minus_ravia_95_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "primary_win_probability": float(np.mean(np.asarray(deltas) > 0.0)),
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.resolve().exists() or args.markdown.resolve().exists():
        raise RuntimeError("refusing to overwrite intensity blind report")
    verified = _verify_seal(args.predictions, args.seal)
    predictions = verified["predictions"]
    seal = verified["seal"]
    pilot_path = args.pilot.resolve(strict=True)
    receipt = json.loads(args.receipt.resolve(strict=True).read_text(encoding="utf-8"))
    if pilot_path != Path(seal["pilot"]["path"]).resolve():
        raise RuntimeError("scoring pilot path differs from seal")
    if (
        receipt.get("pilot_sha256") != _sha256(pilot_path)
        or receipt.get("pilot_md5") != PILOT_MD5
        or receipt.get("pilot_bytes") != PILOT_BYTES
        or receipt.get("pilot_path") != str(pilot_path)
        or receipt.get("url") != PILOT_URL
        or receipt.get("prediction_sha256") != _sha256(args.predictions.resolve(strict=True))
        or receipt.get("seal_sha256") != _sha256(args.seal.resolve(strict=True))
        or receipt.get("script_sha256") != _sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("intensity pilot receipt binding failed")
    timestamp = shared.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if timestamp["response_sha256"] != receipt.get("timestamp", {}).get(
        "response_sha256"
    ):
        raise RuntimeError("intensity timestamp changed after acquisition")
    import pandas as pd

    raw = pd.read_csv(pilot_path, sep=";", low_memory=False)
    required = {"code", "concentration", "volume", "molcode", "cas", "intensive"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"intensity pilot columns changed: {sorted(required-set(raw.columns))}")
    raw = raw.copy()
    raw["molcode"] = raw["molcode"].astype(str).str.strip()
    raw["code"] = raw["code"].map(_identifier)
    raw["fraction"] = raw["concentration"].map(parse_concentration)
    raw["intensive"] = pd.to_numeric(raw["intensive"], errors="coerce")
    if raw["intensive"].isna().any() or ((raw["intensive"] < 1) | (raw["intensive"] > 100)).any():
        raise RuntimeError("intensity pilot ratings are outside 1..100")
    if raw.duplicated(["code", "molcode", "fraction"]).any():
        raise RuntimeError("intensity pilot has duplicate participant conditions")
    rows_by_molcode = {row["molcode"]: row for row in predictions["predictions"]}
    if not set(raw["molcode"]).issubset(rows_by_molcode):
        raise RuntimeError("intensity pilot contains an unpredicted molecule")
    grouped = raw.groupby(["molcode", "fraction"], sort=True)["intensive"].mean()
    condition_keys = [(str(key[0]), float(key[1])) for key in grouped.index]
    target = grouped.to_numpy(dtype=float)
    methods = {
        "strict_structure_concentration_curve": [],
        "condition_transfer_anchored_curve": [],
        "fixed_keller_concentration_baseline": [],
        "frozen_ravia_global_curve": [],
        "structure_only_intensity": [],
    }
    for molcode, fraction in condition_keys:
        row = rows_by_molcode[molcode]
        for field in methods:
            if field == "structure_only_intensity":
                methods[field].append(float(row[field]))
            else:
                methods[field].append(_curve_prediction(row, field, fraction))
    arrays = {name: np.asarray(values, dtype=float) for name, values in methods.items()}
    molcodes = [key[0] for key in condition_keys]
    fractions = np.asarray([key[1] for key in condition_keys], dtype=float)
    results = {
        name: _score_metrics(values, target, molcodes, fractions)
        for name, values in arrays.items()
    }
    primary_name = "condition_transfer_anchored_curve"
    strict_name = "strict_structure_concentration_curve"
    baseline_name = "frozen_ravia_global_curve"
    bootstrap = _bootstrap(
        raw,
        condition_keys,
        arrays[primary_name],
        arrays[baseline_name],
    )
    checks = {
        "timestamp_and_receipt_verified": True,
        "primary_condition_spearman_beats_ravia": results[primary_name]["row_spearman"]
        > results[baseline_name]["row_spearman"],
        "primary_delta_spearman_beats_ravia": results[primary_name]["delta_spearman"]
        > results[baseline_name]["delta_spearman"],
        "bootstrap_primary_minus_ravia_lower_above_zero": bootstrap[
            "primary_minus_ravia_95_interval"
        ][0]
        > 0.0,
    }
    strict_checks = {
        "strict_condition_spearman_beats_ravia": results[strict_name]["row_spearman"]
        > results[baseline_name]["row_spearman"],
        "strict_delta_spearman_beats_ravia": results[strict_name]["delta_spearman"]
        > results[baseline_name]["delta_spearman"],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scored_at": utc_now(),
        "status": (
            "blind_intensity_condition_transfer_improvement_confirmed"
            if all(checks.values())
            else "blind_intensity_pilot_completed_without_full_improvement_gate"
        ),
        "blind_integrity": {
            "prediction_sha256": _sha256(args.predictions.resolve(strict=True)),
            "seal_sha256": _sha256(args.seal.resolve(strict=True)),
            "pilot_sha256": _sha256(pilot_path),
            "receipt_sha256": _sha256(args.receipt.resolve(strict=True)),
            "timestamp": timestamp,
        },
        "population": {
            "participants": int(raw["code"].nunique()),
            "ratings": int(len(raw)),
            "molecules": int(raw["molcode"].nunique()),
            "conditions": len(condition_keys),
            "concentration_minimum": float(fractions.min()),
            "concentration_maximum": float(fractions.max()),
        },
        "results": results,
        "bootstrap": bootstrap,
        "condition_transfer_improvement_gate": {"passed": all(checks.values()), "checks": checks},
        "strict_external_gate": {"passed": all(strict_checks.values()), "checks": strict_checks},
        "human_olfactory_90_percent_certified": False,
        "mixture_or_recipe_validated": False,
        "claim_boundary": (
            "Independent-participant monomolecular concentration/intensity pilot. "
            "The anchored track uses same-molecule main-study intensity; the strict "
            "track excludes exact target human labels. Neither validates mixtures or recipes."
        ),
    }
    _write_json(args.report.resolve(), report)
    markdown_path = args.markdown.resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bierling 2025 intensity pilot 블라인드 검증",
        "",
        "| 모델 | 조건 Spearman | Delta Spearman | 방향 정확도 | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["results"].items():
        lines.append(
            f"| {name} | {row['row_spearman']:.4f} | {row['delta_spearman']:.4f} | "
            f"{row['direction_accuracy']:.4f} | {row['mae']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Condition-transfer gate: **"
            + ("PASS" if report["condition_transfer_improvement_gate"]["passed"] else "FAIL")
            + "**",
            "",
            "이 결과는 단분자 농도–강도 조건전이이며 혼합물·레시피·실제 후각 90% 검증이 아닙니다.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--target-odors", type=Path, required=True)
    prepare_parser.add_argument("--forbidden-pilot", type=Path, required=True)
    prepare_parser.add_argument("--keller-molecules", type=Path, required=True)
    prepare_parser.add_argument("--keller-stimuli", type=Path, required=True)
    prepare_parser.add_argument("--keller-behavior", type=Path, required=True)
    prepare_parser.add_argument("--main-outcome", type=Path, required=True)
    prepare_parser.add_argument("--molformer-root", type=Path, required=True)
    prepare_parser.add_argument("--hf-home", type=Path, required=True)
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--batch-size", type=int, default=32)
    prepare_parser.add_argument("--threads", type=int, default=4)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--predictions", type=Path, required=True)
    seal_parser.add_argument("--pilot", type=Path, required=True)
    seal_parser.add_argument("--seal", type=Path, required=True)
    for name in ("acquire", "score"):
        item = subparsers.add_parser(name)
        item.add_argument("--predictions", type=Path, required=True)
        item.add_argument("--seal", type=Path, required=True)
        item.add_argument("--pilot", type=Path, required=True)
        item.add_argument("--receipt", type=Path, required=True)
        item.add_argument("--openssl", type=Path, required=True)
        item.add_argument("--timestamp-response", type=Path, required=True)
        item.add_argument("--timestamp-ca", type=Path, required=True)
        item.add_argument("--timestamp-tsa", type=Path, required=True)
        if name == "score":
            item.add_argument("--report", type=Path, required=True)
            item.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    if args.command == "prepare":
        result = prepare(args)
        summary = {
            "status": result["status"],
            "development": result["model"]["development"],
            "release_gate": result["release_gate"],
        }
    elif args.command == "seal":
        summary = create_seal(args)
    elif args.command == "acquire":
        summary = acquire(args)
    else:
        result = score(args)
        summary = {
            "status": result["status"],
            "population": result["population"],
            "results": result["results"],
            "condition_transfer_gate": result["condition_transfer_improvement_gate"],
            "strict_external_gate": result["strict_external_gate"],
        }
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
