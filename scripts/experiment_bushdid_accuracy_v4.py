#!/usr/bin/env python
"""Outcome-aware Bushdid accuracy development with odor-pair embeddings.

This experiment replaces the weak frozen-R2 ranking with a compact feature
bank that combines the registered Bushdid protocol and the frozen DREAM
odor-pair embedding.  Every target outcome in this dataset is already public,
so the resulting model is a retrospective development artifact.  It is never
promoted into the production score by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import catboost
import numpy as np
import sklearn
import torch
from catboost import CatBoostRegressor
from ogb.utils import smiles2graph
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_dream_pair_ensemble_v2 as dream_pair  # noqa: E402
import build_human_mixture_calibration as human_calibration  # noqa: E402


SCHEMA = "bushdid-outcome-aware-accuracy-v4"
SEED = 20_260_829
BOOTSTRAP_DRAWS = 20_000
MODEL_PARAMETERS = {
    "iterations": 600,
    "depth": 3,
    "learning_rate": 0.02,
    "l2_leaf_reg": 300.0,
    "loss_function": "MAE",
    "random_seed": 1,
    "thread_count": 1,
    "verbose": False,
}
FEATURE_NAMES = (
    "protocol::component_overlap_dissimilarity",
    "protocol::component_overlap_dissimilarity_squared",
    "protocol::component_overlap_dissimilarity_cubed",
    "protocol::component_overlap_dissimilarity_sqrt",
    "protocol::components_per_mixture",
    "protocol::components_is_10",
    "protocol::components_is_20",
    "protocol::components_is_30",
    "protocol::right_dilution",
    "protocol::right_dilution_is_0_25",
    "protocol::right_dilution_is_0_50",
    "protocol::right_dilution_is_1_00",
    "protocol::log2_right_dilution",
    "protocol::wrong_log10_dilution_spread",
    "legacy::r2_dissimilarity",
    "legacy::r2_member_disagreement",
    "interaction::overlap_x_log2_right_dilution",
    "interaction::overlap_x_normalized_mixture_size",
    "odor_pair::correlation_distance",
    "odor_pair::cosine_distance",
    "odor_pair::euclidean_distance",
    "odor_pair::angle_degrees",
    "odor_pair::absolute_difference_mean",
    "odor_pair::absolute_difference_maximum",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or any(None in row for row in rows):
        raise ValueError(f"invalid CSV rows: {path}")
    return rows


def _stimulus_definitions(stimuli_path: Path) -> dict[tuple[str, str], list[int]]:
    definitions: dict[tuple[str, str], list[int]] = {}
    for row in _read_csv(stimuli_path):
        stimulus_id = str(int(float(row["Stimulus"])))
        answer = row["Answer"].strip().casefold()
        if answer not in {"right", "wrong"}:
            raise ValueError(f"invalid answer class for stimulus {stimulus_id}")
        key = (stimulus_id, "right" if answer == "right" else "wrong")
        components = [
            int(row[f"Molecule {index}"])
            for index in range(1, 31)
            if row.get(f"Molecule {index}", "").strip()
            and int(row[f"Molecule {index}"]) != 0
        ]
        if not components or len(components) != len(set(components)):
            raise ValueError(f"invalid component list for stimulus {stimulus_id}")
        if key in definitions and definitions[key] != components:
            raise ValueError(f"duplicate stimulus rows disagree: {key}")
        definitions[key] = components
    if len(definitions) != 528:
        raise ValueError("Bushdid stimulus definition count changed")
    return definitions


def _cid_smiles(molecules_path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in _read_csv(molecules_path):
        cid = int(row["CID"])
        text = row["IsomericSMILES"].strip()
        if not text or cid in result:
            raise ValueError("Bushdid molecule table is invalid")
        result[cid] = text
    if len(result) != 128:
        raise ValueError("Bushdid molecule count changed")
    return result


def _pair_embeddings(
    definitions: Mapping[tuple[str, str], Sequence[int]],
    smiles: Mapping[int, str],
    *,
    pair_root: Path,
    dream_root: Path,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    model_root = dream_root / "SOTA" / "3-Pair_Model" / "finetuned_model"
    model, pair_data, pairdata, _config = dream_pair._load_pair_model(
        pair_root,
        model_root / "config.json",
        model_root / "model.pt",
    )
    graphs = {
        cid: pairdata.to_torch(smiles2graph(text))
        for cid, text in sorted(smiles.items())
    }
    embeddings = dream_pair._generated_embeddings(
        definitions,
        model,
        pair_data,
        pairdata,
        graphs,
    )
    values = np.vstack([embeddings[key] for key in sorted(embeddings)]).astype(
        np.float32
    )
    return embeddings, {
        "pair_source_commit": dream_pair._git_commit(pair_root),
        "pair_source_tree_sha256": dream_pair._source_tree_hash(pair_root),
        "pair_config_sha256": sha256_file(model_root / "config.json"),
        "pair_weights_sha256": sha256_file(model_root / "model.pt"),
        "embedding_rows": len(values),
        "embedding_width": int(values.shape[1]),
        "embedding_rows_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _feature_row(
    row: Mapping[str, Any],
    embeddings: Mapping[tuple[str, str], np.ndarray],
) -> np.ndarray:
    stimulus_id = str(row["stimulus_id"])
    pair = dream_pair.pair_embedding_features(
        embeddings[(stimulus_id, "right")],
        embeddings[(stimulus_id, "wrong")],
    )
    overlap = float(row["component_overlap_dissimilarity"])
    components = int(row["components_per_mixture"])
    right_dilution = float(row["right_dilution"])
    log_right = math.log2(right_dilution)
    result = np.asarray(
        [
            overlap,
            overlap**2,
            overlap**3,
            math.sqrt(overlap),
            components,
            float(components == 10),
            float(components == 20),
            float(components == 30),
            right_dilution,
            float(math.isclose(right_dilution, 0.25, abs_tol=1e-12)),
            float(math.isclose(right_dilution, 0.50, abs_tol=1e-12)),
            float(math.isclose(right_dilution, 1.00, abs_tol=1e-12)),
            log_right,
            float(row["wrong_log10_dilution_spread"]),
            float(row["r2_dissimilarity"]),
            float(row.get("member_disagreement", 0.0)),
            overlap * log_right,
            overlap * components / 30.0,
            *pair[:4].tolist(),
            float(np.mean(pair[4:132])),
            float(np.max(pair[4:132])),
        ],
        dtype=float,
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("Bushdid accuracy feature contract changed")
    return result


def _accuracy_metrics(prediction: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    outcome = np.asarray(outcome, dtype=float)
    mae = float(np.mean(np.abs(prediction - outcome)))
    return {
        "absolute_accuracy_percent": 100.0 * (1.0 - mae),
        "mae_percentage_points": 100.0 * mae,
        "rmse_percentage_points": 100.0
        * float(np.sqrt(np.mean((prediction - outcome) ** 2))),
        "spearman": human_calibration._spearman(prediction, outcome),
    }


def _crossfit_isotonic(
    score: np.ndarray,
    outcome: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.zeros(len(outcome), dtype=float)
    for train, held_out in splits:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(score[train], outcome[train])
        prediction[held_out] = model.predict(score[held_out])
    return prediction


def _bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    outcome: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    candidate_accuracy = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    accuracy_gain = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(outcome), len(outcome))
        candidate_mae = float(np.mean(np.abs(candidate[indices] - outcome[indices])))
        baseline_mae = float(np.mean(np.abs(baseline[indices] - outcome[indices])))
        candidate_accuracy[draw] = 100.0 * (1.0 - candidate_mae)
        accuracy_gain[draw] = 100.0 * (baseline_mae - candidate_mae)
    return {
        "method": "fixed_out_of_fold_prediction_stimulus_bootstrap",
        "post_selection_model_uncertainty_included": False,
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "candidate_accuracy_95_interval": [
            float(value) for value in np.quantile(candidate_accuracy, [0.025, 0.975])
        ],
        "candidate_minus_overlap_accuracy_95_interval": [
            float(value) for value in np.quantile(accuracy_gain, [0.025, 0.975])
        ],
    }


def build(
    *,
    report_path: Path,
    prediction_path: Path,
    molecules_path: Path,
    stimuli_path: Path,
    pair_root: Path,
    dream_root: Path,
    model_output: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sealed_prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    expected_prediction_hash = report.get("blind_integrity", {}).get(
        "prediction_file_sha256"
    )
    if expected_prediction_hash != sha256_file(prediction_path):
        raise RuntimeError("Bushdid outcome report is not bound to sealed predictions")
    registered = sealed_prediction["dataset"]["input_files_without_human_outcomes"]
    for name, path in (("molecules.csv", molecules_path), ("stimuli.csv", stimuli_path)):
        evidence = registered[name]
        if (
            sha256_file(path) != evidence["sha256"]
            or path.stat().st_size != evidence["bytes"]
        ):
            raise RuntimeError(f"Bushdid source bytes changed: {name}")

    definitions = _stimulus_definitions(stimuli_path)
    smiles = _cid_smiles(molecules_path)
    embeddings, embedding_audit = _pair_embeddings(
        definitions,
        smiles,
        pair_root=pair_root,
        dream_root=dream_root,
    )
    protocol = human_calibration._read_protocol_features(stimuli_path)
    rows = human_calibration._attach_protocol(report["stimulus_results"], protocol)
    rows = [
        row
        for row in rows
        if row["evaluation_partition"] in {"calibration", "final_test"}
    ]
    if len(rows) != 260:
        raise RuntimeError("Bushdid development population changed")
    matrix = np.asarray([_feature_row(row, embeddings) for row in rows], dtype=float)
    outcome = np.asarray([row["human_correct_rate"] for row in rows], dtype=float)
    strata = np.asarray(
        [
            f"{row['components_per_mixture']}|{row['declared_overlap_percent']}"
            for row in rows
        ]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(matrix, strata))
    prediction = np.zeros(len(rows), dtype=float)
    fold_rows = []
    for fold, (train, held_out) in enumerate(splits):
        model = CatBoostRegressor(**MODEL_PARAMETERS)
        model.fit(matrix[train], outcome[train])
        prediction[held_out] = np.asarray(model.predict(matrix[held_out])).reshape(-1)
        fold_rows.append(
            {
                "fold": fold,
                "train_stimuli": len(train),
                "held_out_stimuli": len(held_out),
                "held_out_stimulus_ids_sha256": canonical_json_sha256(
                    sorted(rows[index]["stimulus_id"] for index in held_out)
                ),
            }
        )

    overlap_score = np.asarray(
        [row["component_overlap_dissimilarity"] for row in rows], dtype=float
    )
    pair_score = np.asarray(
        [
            dream_pair.pair_embedding_features(
                embeddings[(str(row["stimulus_id"]), "right")],
                embeddings[(str(row["stimulus_id"]), "wrong")],
            )[1]
            for row in rows
        ],
        dtype=float,
    )
    overlap_prediction = _crossfit_isotonic(overlap_score, outcome, splits)
    pair_prediction = _crossfit_isotonic(pair_score, outcome, splits)
    candidate_metrics = _accuracy_metrics(prediction, outcome)
    overlap_metrics = _accuracy_metrics(overlap_prediction, outcome)
    pair_metrics = _accuracy_metrics(pair_prediction, outcome)
    noise_ceiling = float(
        report["final_test_results"]["human_noise_ceiling"][
            "correlation_noise_ceiling"
        ]
    )
    candidate_metrics["human_ceiling_normalized_spearman"] = (
        candidate_metrics["spearman"] / noise_ceiling
    )
    old_gate = report["final_test_results"]["human_ceiling_90_percent_gate"]

    final_model = CatBoostRegressor(**MODEL_PARAMETERS)
    final_model.fit(matrix, outcome)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".json", dir=model_output.parent, delete=False
    ) as handle:
        temporary_model = Path(handle.name)
    final_model.save_model(temporary_model, format="json")
    model_payload = json.loads(temporary_model.read_text(encoding="utf-8"))
    temporary_model.unlink()
    model_info = model_payload.get("model_info")
    if not isinstance(model_info, dict):
        raise RuntimeError("CatBoost JSON has no model_info object")
    model_info["model_guid"] = "bushdid-accuracy-v4-deterministic"
    model_info["train_finish_time"] = "1970-01-01T00:00:00Z"
    atomic_json(model_output, model_payload)

    result = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "outcome_aware_retrospective_development",
        "target": {
            "absolute_accuracy_percent": 90.0,
            "human_ceiling_normalized_spearman": 0.90,
        },
        "source_binding": {
            "human_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            },
            "sealed_prediction": {
                "path": str(prediction_path.resolve()),
                "sha256": sha256_file(prediction_path),
            },
            "molecules": {
                "path": str(molecules_path.resolve()),
                "sha256": sha256_file(molecules_path),
            },
            "stimuli": {
                "path": str(stimuli_path.resolve()),
                "sha256": sha256_file(stimuli_path),
            },
            "embedding_model": embedding_audit,
        },
        "development_contract": {
            "outcomes_visible_before_model_design": True,
            "stimulus_level_crossfit": True,
            "folds": 5,
            "stratification": "components_per_mixture_x_declared_overlap",
            "controls_excluded": True,
            "stimuli": len(rows),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_contract_sha256": canonical_json_sha256(FEATURE_NAMES),
            "model": "CatBoostRegressor",
            "model_parameters": MODEL_PARAMETERS,
            "selection_disclosure": (
                "Bushdid outcomes were visible while protocol, odor-pair, POMMix, "
                "OpenPOM, RDKit, measured-intensity feature banks and more than "
                "100 CatBoost configurations were compared. The selected-fold "
                "score is post-selection descriptive evidence."
            ),
            "post_selection_intervals_descriptive_only": True,
            "fold_rows": fold_rows,
        },
        "results": {
            "frozen_r2_historical_final": {
                "human_ceiling_normalized_spearman": float(
                    old_gate["point_estimate"]
                ),
                "spearman": float(
                    report["final_test_results"]["continuous_human_correct_rate"][
                        "r2_spearman"
                    ]
                ),
            },
            "crossfit_component_overlap": overlap_metrics,
            "crossfit_odor_pair_isotonic": pair_metrics,
            "crossfit_compact_candidate": candidate_metrics,
            "bootstrap": _bootstrap(
                np.clip(prediction, 0.0, 1.0),
                np.clip(overlap_prediction, 0.0, 1.0),
                outcome,
            ),
        },
        "target_gate": {
            "absolute_accuracy_point_at_least_90": (
                candidate_metrics["absolute_accuracy_percent"] >= 90.0
            ),
            "normalized_rank_point_at_least_0_90": (
                candidate_metrics["human_ceiling_normalized_spearman"] >= 0.90
            ),
            "prospective_external_validation": False,
            "passed": False,
        },
        "artifact": {
            "model_path": str(model_output.resolve()),
            "model_sha256": sha256_file(model_output),
            "model_bytes": model_output.stat().st_size,
            "format": "catboost_json_no_pickle",
            "runtime_primary_score_weight": 0.0,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "The compact candidate is selected after Bushdid outcomes were public. "
            "Its cross-fitted score is a retrospective development diagnostic and "
            "does not authorize a human-olfactory or generated-formula claim."
        ),
    }
    result["target_gate"]["passed"] = all(
        bool(value)
        for key, value in result["target_gate"].items()
        if key != "prospective_external_validation"
    ) and bool(result["target_gate"]["prospective_external_validation"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            r"C:\Users\user\Desktop\Game\server\data\pom_data\dream_mixture\bushdid_2014"
        ),
    )
    parser.add_argument(
        "--human-report",
        type=Path,
        default=ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.json",
    )
    parser.add_argument(
        "--sealed-predictions",
        type=Path,
        default=ROOT / "benchmarks" / "bushdid_blind_predictions_v1.json",
    )
    parser.add_argument(
        "--pair-root",
        type=Path,
        default=ROOT / "tmp" / "laura_dream_source_20260828",
    )
    parser.add_argument(
        "--dream-root",
        type=Path,
        default=ROOT / "tmp" / "dream_olfactory_mixtures_2025_source",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "benchmarks" / "bushdid_accuracy_v4_catboost.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "bushdid_accuracy_v4.json",
    )
    args = parser.parse_args()
    result = build(
        report_path=args.human_report.resolve(strict=True),
        prediction_path=args.sealed_predictions.resolve(strict=True),
        molecules_path=(args.dataset_root / "molecules.csv").resolve(strict=True),
        stimuli_path=(args.dataset_root / "stimuli.csv").resolve(strict=True),
        pair_root=args.pair_root.resolve(strict=True),
        dream_root=args.dream_root.resolve(strict=True),
        model_output=args.model_output.resolve(),
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
