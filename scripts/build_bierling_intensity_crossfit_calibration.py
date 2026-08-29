#!/usr/bin/env python
"""Retrospective nested-crossfit calibration of frozen intensity curves.

This analysis is intentionally post-outcome. It combines only predictions that
were frozen before the Bierling intensity pilot was opened, selects a linear
calibrator inside each outer molecule-disjoint fold, evaluates on the outer
fold, and exports a diagnostic-only portable affine model. It cannot upgrade
the blind pilot's failed concentration-delta gate or authorize runtime weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_bierling_intensity_pilot_benchmark as blind  # noqa: E402


SCHEMA_VERSION = "1.0"
FOLD_SALT = "bierling-intensity-retrospective-nested-crossfit-v1"
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_SEED = 20_260_828
BOOTSTRAP_DRAWS = 5_000
FEATURE_SETS = (
    ("anchored", ("condition_transfer_anchored_curve",)),
    (
        "anchored_plus_ravia",
        ("condition_transfer_anchored_curve", "frozen_ravia_global_curve"),
    ),
    (
        "anchored_strict_ravia",
        (
            "condition_transfer_anchored_curve",
            "strict_structure_concentration_curve",
            "frozen_ravia_global_curve",
        ),
    ),
    (
        "all_frozen_branches",
        (
            "condition_transfer_anchored_curve",
            "strict_structure_concentration_curve",
            "fixed_keller_concentration_baseline",
            "frozen_ravia_global_curve",
            "structure_only_intensity",
        ),
    ),
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CANDIDATES = tuple(
    {"name": f"{name}_ridge_{alpha:g}", "features": features, "alpha": alpha}
    for name, features in FEATURE_SETS
    for alpha in ALPHAS
)


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _folds(molcodes: Sequence[str], count: int, salt: str) -> np.ndarray:
    unique = sorted(
        set(molcodes),
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(),
    )
    mapping = {value: index % count for index, value in enumerate(unique)}
    return np.asarray([mapping[value] for value in molcodes], dtype=int)


def _load_conditions(
    prediction_path: Path, pilot_path: Path
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    import pandas as pd

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows_by_molcode = {row["molcode"]: row for row in predictions["predictions"]}
    frame = pd.read_csv(pilot_path, sep=";", low_memory=False)
    frame = frame.rename(columns={"intensity": "intensive"})
    frame["intensive"] = pd.to_numeric(frame["intensive"], errors="raise")
    if ((frame["intensive"] < 0) | (frame["intensive"] > 100)).any():
        raise RuntimeError("pilot intensity outside adjudicated 0..100 range")
    frame["molcode"] = frame["molcode"].astype(str).str.strip()
    frame["fraction"] = frame["concentration"].map(blind.parse_concentration)
    duplicate_key = ["code", "molcode", "concentration", "volume", "cas"]
    frame = (
        frame.groupby(duplicate_key, as_index=False, sort=False)
        .agg({"intensive": "mean", "odor_group": "first", "fraction": "first"})
    )
    grouped = frame.groupby(["molcode", "fraction"], sort=True)["intensive"].mean()
    conditions = []
    for (molcode, fraction), target in grouped.items():
        source = rows_by_molcode[str(molcode)]
        feature_values = {}
        for _, features in FEATURE_SETS:
            for feature in features:
                if feature == "structure_only_intensity":
                    feature_values[feature] = float(source[feature])
                else:
                    feature_values[feature] = blind._curve_prediction(
                        source, feature, float(fraction)
                    )
        conditions.append(
            {
                "molcode": str(molcode),
                "fraction": float(fraction),
                "target": float(target),
                **feature_values,
            }
        )
    if len(conditions) < 70 or len({row["molcode"] for row in conditions}) < 70:
        raise RuntimeError("too few pilot conditions for calibration")
    return conditions, frame, {
        "ratings_after_within_participant_anchor_collapse": len(frame),
        "conditions": len(conditions),
        "molecules": len({row["molcode"] for row in conditions}),
    }


def _design(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=float
    )


def _fit(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[dict[str, Any], np.ndarray]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    normalized = scaler.fit_transform(x)
    estimator = Ridge(alpha=alpha).fit(normalized, y)
    parameters = {
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in estimator.coef_],
        "intercept": float(estimator.intercept_),
        "alpha": float(alpha),
    }
    return parameters, np.clip(estimator.predict(normalized), 0.0, 100.0)


def _predict(parameters: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(parameters["mean"], dtype=float)
    scale = np.asarray(parameters["scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    result = ((x - mean) / scale) @ coefficients + float(parameters["intercept"])
    return np.clip(result, 0.0, 100.0)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "spearman": shared.spearman(prediction, target),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def _candidate_oof(
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    *,
    folds: int,
    salt: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    local_rows = [rows[index] for index in indices]
    molcodes = [str(row["molcode"]) for row in local_rows]
    target = np.asarray([row["target"] for row in local_rows], dtype=float)
    assignments = _folds(molcodes, folds, salt)
    predictions = {
        candidate["name"]: np.full(len(local_rows), np.nan) for candidate in CANDIDATES
    }
    for fold in range(folds):
        training = np.flatnonzero(assignments != fold)
        validation = np.flatnonzero(assignments == fold)
        for candidate in CANDIDATES:
            x = _design(local_rows, candidate["features"])
            parameters, _ = _fit(x[training], target[training], candidate["alpha"])
            predictions[candidate["name"]][validation] = _predict(
                parameters, x[validation]
            )
    if any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("incomplete calibration OOF prediction")
    metrics = {
        name: _metrics(prediction, target) for name, prediction in predictions.items()
    }
    return predictions, metrics


def _select(metrics: Mapping[str, Mapping[str, float]]) -> str:
    return min(
        metrics,
        key=lambda name: (
            metrics[name]["mae"],
            -metrics[name]["spearman"],
            next(index for index, row in enumerate(CANDIDATES) if row["name"] == name),
        ),
    )


def _nested_crossfit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    molcodes = [str(row["molcode"]) for row in rows]
    target = np.asarray([row["target"] for row in rows], dtype=float)
    outer = _folds(molcodes, OUTER_FOLDS, FOLD_SALT + "|outer")
    prediction = np.full(len(rows), np.nan)
    audit = []
    specifications = {row["name"]: row for row in CANDIDATES}
    for fold in range(OUTER_FOLDS):
        training = np.flatnonzero(outer != fold)
        validation = np.flatnonzero(outer == fold)
        _, inner_metrics = _candidate_oof(
            rows,
            training,
            folds=INNER_FOLDS,
            salt=f"{FOLD_SALT}|inner|{fold}",
        )
        selected = _select(inner_metrics)
        candidate = specifications[selected]
        x = _design(rows, candidate["features"])
        parameters, _ = _fit(x[training], target[training], candidate["alpha"])
        prediction[validation] = _predict(parameters, x[validation])
        audit.append(
            {
                "fold": fold,
                "training_molecules": len({molcodes[index] for index in training}),
                "validation_molecules": len({molcodes[index] for index in validation}),
                "selected_candidate": selected,
                "inner_selected_metrics": inner_metrics[selected],
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("nested crossfit prediction incomplete")
    return prediction, audit


def _bootstrap(
    calibrated: np.ndarray,
    ravia: np.ndarray,
    raw: Any,
    condition_keys: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    participants = sorted(raw["code"].astype(str).unique().tolist())
    participant_index = {value: index for index, value in enumerate(participants)}
    condition_index = {value: index for index, value in enumerate(condition_keys)}
    values = np.zeros((len(participants), len(condition_keys)), dtype=float)
    observed = np.zeros_like(values)
    for _, row in raw.iterrows():
        key = (str(row["molcode"]), float(row["fraction"]))
        values[participant_index[str(row["code"])], condition_index[key]] = float(
            row["intensive"]
        )
        observed[participant_index[str(row["code"])], condition_index[key]] = 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    spearman_delta = []
    mae_delta = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(
            len(participants), np.full(len(participants), 1.0 / len(participants))
        )
        counts = weights @ observed
        if np.any(counts <= 0):
            continue
        target = (weights @ values) / counts
        indices = rng.integers(0, len(condition_keys), size=len(condition_keys))
        spearman_delta.append(
            shared.spearman(calibrated[indices], target[indices])
            - shared.spearman(ravia[indices], target[indices])
        )
        mae_delta.append(
            np.mean(np.abs(ravia[indices] - target[indices]))
            - np.mean(np.abs(calibrated[indices] - target[indices]))
        )
    if len(spearman_delta) < int(BOOTSTRAP_DRAWS * 0.95):
        raise RuntimeError("too many invalid participant-condition bootstrap draws")
    return {
        "requested_draws": BOOTSTRAP_DRAWS,
        "valid_draws": len(spearman_delta),
        "seed": BOOTSTRAP_SEED,
        "unit": "participant-cluster weights plus condition resampling",
        "calibrated_minus_ravia_spearman_95_interval": [
            float(value) for value in np.quantile(spearman_delta, [0.025, 0.975])
        ],
        "ravia_minus_calibrated_mae_95_interval": [
            float(value) for value in np.quantile(mae_delta, [0.025, 0.975])
        ],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    blind_report_path = args.blind_report.resolve(strict=True)
    pilot_path = args.pilot.resolve(strict=True)
    if args.output.resolve().exists() or args.markdown.resolve().exists():
        raise RuntimeError("refusing to overwrite crossfit calibration output")
    blind_report = json.loads(blind_report_path.read_text(encoding="utf-8"))
    if blind_report.get("blind_integrity", {}).get("pilot_sha256") != _sha256(
        pilot_path
    ):
        raise RuntimeError("blind pilot report/outcome hash mismatch")
    conditions, raw, data_audit = _load_conditions(prediction_path, pilot_path)
    target = np.asarray([row["target"] for row in conditions], dtype=float)
    nested_prediction, folds = _nested_crossfit(conditions)
    baseline_names = [
        "condition_transfer_anchored_curve",
        "strict_structure_concentration_curve",
        "frozen_ravia_global_curve",
        "structure_only_intensity",
    ]
    baseline_metrics = {
        name: _metrics(
            np.asarray([row[name] for row in conditions], dtype=float), target
        )
        for name in baseline_names
    }
    nested_metrics = _metrics(nested_prediction, target)
    all_indices = np.arange(len(conditions))
    _, full_metrics = _candidate_oof(
        conditions,
        all_indices,
        folds=OUTER_FOLDS,
        salt=FOLD_SALT + "|final-selection",
    )
    selected = _select(full_metrics)
    candidate = next(row for row in CANDIDATES if row["name"] == selected)
    x = _design(conditions, candidate["features"])
    parameters, _ = _fit(x, target, candidate["alpha"])
    ravia = np.asarray(
        [row["frozen_ravia_global_curve"] for row in conditions], dtype=float
    )
    condition_keys = [
        (str(row["molcode"]), float(row["fraction"])) for row in conditions
    ]
    bootstrap = _bootstrap(nested_prediction, ravia, raw, condition_keys)
    checks = {
        "nested_crossfit_spearman_beats_ravia": nested_metrics["spearman"]
        > baseline_metrics["frozen_ravia_global_curve"]["spearman"],
        "nested_crossfit_mae_beats_ravia": nested_metrics["mae"]
        < baseline_metrics["frozen_ravia_global_curve"]["mae"],
        "spearman_delta_bootstrap_lower_above_zero": bootstrap[
            "calibrated_minus_ravia_spearman_95_interval"
        ][0]
        > 0.0,
        "mae_improvement_bootstrap_lower_above_zero": bootstrap[
            "ravia_minus_calibrated_mae_95_interval"
        ][0]
        > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "retrospective_nested_crossfit_calibration_confirmed"
            if all(checks.values())
            else "retrospective_calibration_not_confirmed"
        ),
        "development_timing": "post_intensity_pilot_outcome",
        "source_binding": {
            "blind_predictions_sha256": _sha256(prediction_path),
            "blind_report_sha256": _sha256(blind_report_path),
            "pilot_sha256": _sha256(pilot_path),
        },
        "data": data_audit,
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "selection_metric": "inner molecule-disjoint MAE then Spearman",
            "outer_fold_audit": folds,
        },
        "nested_crossfit": nested_metrics,
        "frozen_baselines": baseline_metrics,
        "bootstrap": bootstrap,
        "portable_diagnostic_calibrator": {
            "selected_candidate": selected,
            "features": list(candidate["features"]),
            "parameters": parameters,
            "clip_range": [0.0, 100.0],
            "runtime_primary_score_weight": 0.0,
        },
        "release_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "scope": "retrospective diagnostic calibrator only",
        },
        "human_olfactory_90_percent_certified": False,
        "concentration_delta_validated": False,
        "claim_boundary": (
            "Retrospective nested molecule-disjoint calibration on the already-opened "
            "pilot. It does not preserve the blind status of model selection, does not "
            "validate concentration deltas, mixtures, recipes, or 90% olfactory accuracy."
        ),
    }
    shared.write_json(args.output.resolve(), report)
    args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown.resolve().write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    nested = report["nested_crossfit"]
    ravia = report["frozen_baselines"]["frozen_ravia_global_curve"]
    anchored = report["frozen_baselines"]["condition_transfer_anchored_curve"]
    bootstrap = report["bootstrap"]
    return "\n".join(
        [
            "# Bierling intensity 사후 nested-crossfit 보정",
            "",
            "| 평가 | Spearman | MAE |",
            "|---|---:|---:|",
            f"| Nested crossfit calibrator | {nested['spearman']:.4f} | {nested['mae']:.3f} |",
            f"| Frozen anchored curve | {anchored['spearman']:.4f} | {anchored['mae']:.3f} |",
            f"| Frozen Ravia curve | {ravia['spearman']:.4f} | {ravia['mae']:.3f} |",
            "",
            "Spearman 개선 95% 구간: "
            f"[{bootstrap['calibrated_minus_ravia_spearman_95_interval'][0]:+.4f}, "
            f"{bootstrap['calibrated_minus_ravia_spearman_95_interval'][1]:+.4f}]",
            "",
            "MAE 감소 95% 구간: "
            f"[{bootstrap['ravia_minus_calibrated_mae_95_interval'][0]:+.3f}, "
            f"{bootstrap['ravia_minus_calibrated_mae_95_interval'][1]:+.3f}]",
            "",
            "이 보정은 pilot 개봉 후의 nested molecule-disjoint 회고 분석이며 runtime weight는 0입니다.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = build(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
