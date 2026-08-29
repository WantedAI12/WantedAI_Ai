#!/usr/bin/env python
"""Richer retrospective v2 calibration with repeated nested molecule CV.

All input branches were frozen before the Bierling intensity pilot opened.
Model families are limited to portable affine, hinge, interaction, Huber, and
isotonic calibrators. Five repeated outer molecule-disjoint folds estimate
generalization; participant-cluster plus condition bootstrap gates both rank
and MAE improvement. The result remains diagnostic-only regardless of score.
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
from scripts import build_bierling_intensity_crossfit_calibration as v1  # noqa: E402


SCHEMA_VERSION = "2.0"
FOLD_SALT = "bierling-intensity-calibration-v2-repeated-nested"
OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_SEED = 20_260_829
BOOTSTRAP_DRAWS = 5_000
RAVIA_MAE = 12.960351980243308
TARGET_MAE = 9.0
HINGE_KNOTS = (20.0, 40.0, 60.0, 80.0)
BASE_FEATURES = (
    "condition_transfer_anchored_curve",
    "strict_structure_concentration_curve",
    "fixed_keller_concentration_baseline",
    "frozen_ravia_global_curve",
    "structure_only_intensity",
    "main_intensity_anchor",
    "main_anchor_available",
    "log10_fraction",
    "log10_final_fraction",
    "log10_delta",
    "absolute_log10_delta",
    "volume_ml",
    "ravia_delta",
    "strict_delta",
)
FEATURE_GROUPS = {
    "anchored": ("condition_transfer_anchored_curve",),
    "anchored_ravia": (
        "condition_transfer_anchored_curve",
        "frozen_ravia_global_curve",
    ),
    "protocol": (
        "condition_transfer_anchored_curve",
        "frozen_ravia_global_curve",
        "main_intensity_anchor",
        "log10_delta",
        "absolute_log10_delta",
        "volume_ml",
    ),
    "full": BASE_FEATURES,
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_ALPHAS = (0.0001, 0.01, 0.1)
HUBER_EPSILONS = (1.2, 1.35, 1.5)
CANDIDATES: tuple[dict[str, Any], ...] = (
    *tuple(
        {
            "name": f"ridge_{group}_{transform}_{alpha:g}",
            "algorithm": "ridge",
            "group": group,
            "transform": transform,
            "alpha": alpha,
        }
        for group, transform in (
            ("anchored", "raw"),
            ("anchored_ravia", "raw"),
            ("protocol", "raw"),
            ("protocol", "hinge"),
            ("protocol", "polynomial"),
            ("full", "interactions"),
        )
        for alpha in ALPHAS
    ),
    *tuple(
        {
            "name": f"huber_{transform}_{alpha:g}_{epsilon:g}",
            "algorithm": "huber",
            "group": "protocol",
            "transform": transform,
            "alpha": alpha,
            "epsilon": epsilon,
        }
        for transform in ("raw", "hinge")
        for alpha in HUBER_ALPHAS
        for epsilon in HUBER_EPSILONS
    ),
    {
        "name": "isotonic_anchored",
        "algorithm": "isotonic",
        "group": "anchored",
        "transform": "raw",
    },
)


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _parse_volume(value: object) -> float:
    text = str(value).strip().lower().replace(" ", "")
    if not text.endswith("ml"):
        raise ValueError(f"unsupported pilot volume: {value!r}")
    result = float(text[:-2])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"invalid pilot volume: {value!r}")
    return result


def _load_enriched(
    predictions_path: Path, pilot_path: Path
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    conditions, raw, audit = v1._load_conditions(predictions_path, pilot_path)
    prediction_document = json.loads(predictions_path.read_text(encoding="utf-8"))
    source_by_molcode = {
        row["molcode"]: row for row in prediction_document["predictions"]
    }
    volume_by_condition = {}
    volume_levels_by_condition = {}
    count_by_condition = {}
    for (molcode, fraction), group in raw.groupby(["molcode", "fraction"]):
        volume_rows = [_parse_volume(value) for value in group["volume"]]
        volumes = sorted(set(volume_rows))
        key = (str(molcode), float(fraction))
        volume_by_condition[key] = float(np.mean(volume_rows))
        volume_levels_by_condition[key] = volumes
        count_by_condition[key] = len(group)
    enriched = []
    for row in conditions:
        source = source_by_molcode[row["molcode"]]
        main_anchor = source.get("main_intensity_anchor")
        final_fraction = source.get("main_anchor_final_fraction")
        anchor_available = main_anchor is not None
        if main_anchor is None:
            main_anchor = float(row["structure_only_intensity"])
        if final_fraction is None:
            final_fraction = 0.001
        fraction = float(row["fraction"])
        log_fraction = float(np.log10(fraction))
        log_final = float(np.log10(float(final_fraction)))
        anchored = float(row["condition_transfer_anchored_curve"])
        strict = float(row["strict_structure_concentration_curve"])
        structure_only = float(row["structure_only_intensity"])
        key = (row["molcode"], fraction)
        enriched.append(
            {
                **row,
                "main_intensity_anchor": float(main_anchor),
                "main_anchor_available": float(anchor_available),
                "log10_fraction": log_fraction,
                "log10_final_fraction": log_final,
                "log10_delta": log_fraction - log_final,
                "absolute_log10_delta": abs(log_fraction - log_final),
                "volume_ml": volume_by_condition[key],
                "ravia_delta": anchored - float(main_anchor),
                "strict_delta": strict - structure_only,
                "participant_count": count_by_condition[key],
            }
        )
    if any(
        not np.all(np.isfinite([row[name] for name in BASE_FEATURES]))
        for row in enriched
    ):
        raise RuntimeError("v2 feature bank contains non-finite values")
    audit["feature_names"] = list(BASE_FEATURES)
    audit["main_anchor_fallback_molcodes"] = sorted(
        row["molcode"] for row in enriched if row["main_anchor_available"] == 0.0
    )
    audit["volume_values_ml"] = sorted(
        {value for values in volume_levels_by_condition.values() for value in values}
    )
    audit["conditions_with_multiple_volume_levels"] = sum(
        len(values) > 1 for values in volume_levels_by_condition.values()
    )
    audit["volume_feature"] = "participant-weighted condition mean milliliters"
    audit["minimum_participants_per_condition"] = min(
        row["participant_count"] for row in enriched
    )
    audit["maximum_participants_per_condition"] = max(
        row["participant_count"] for row in enriched
    )
    return enriched, raw, audit


def _raw_matrix(rows: Sequence[Mapping[str, Any]], group: str) -> np.ndarray:
    features = FEATURE_GROUPS[group]
    return np.asarray(
        [[float(row[name]) for name in features] for row in rows], dtype=float
    )


def _design(rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> np.ndarray:
    raw = _raw_matrix(rows, str(candidate["group"]))
    transform = candidate["transform"]
    anchored = np.asarray(
        [row["condition_transfer_anchored_curve"] for row in rows], dtype=float
    )[:, None]
    if transform == "raw":
        return raw
    if transform == "hinge":
        hinges = np.concatenate(
            [np.maximum(anchored - knot, 0.0) for knot in HINGE_KNOTS],
            axis=1,
        )
        return np.concatenate((raw, hinges), axis=1)
    if transform == "polynomial":
        return np.concatenate((raw, anchored**2 / 100.0, anchored**3 / 10_000.0), axis=1)
    if transform == "interactions":
        log_delta = np.asarray([row["log10_delta"] for row in rows], dtype=float)[:, None]
        main_anchor = np.asarray(
            [row["main_intensity_anchor"] for row in rows], dtype=float
        )[:, None]
        ravia = np.asarray(
            [row["frozen_ravia_global_curve"] for row in rows], dtype=float
        )[:, None]
        interactions = np.concatenate(
            (
                anchored * log_delta,
                main_anchor * log_delta,
                anchored * ravia / 100.0,
                main_anchor * ravia / 100.0,
            ),
            axis=1,
        )
        return np.concatenate((raw, interactions), axis=1)
    raise KeyError(f"unknown v2 feature transform: {transform}")


def _fit_predict(
    candidate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    target = np.asarray([row["target"] for row in training_rows], dtype=float)
    if candidate["algorithm"] == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        training_x = _design(training_rows, candidate)[:, 0]
        validation_x = _design(validation_rows, candidate)[:, 0]
        estimator = IsotonicRegression(
            increasing=True, out_of_bounds="clip", y_min=0.0, y_max=100.0
        ).fit(training_x, target)
        parameters = {
            "algorithm": "isotonic",
            "x_thresholds": [float(value) for value in estimator.X_thresholds_],
            "y_thresholds": [float(value) for value in estimator.y_thresholds_],
        }
        return parameters, np.asarray(estimator.predict(validation_x), dtype=float)

    from sklearn.preprocessing import StandardScaler

    training_x = _design(training_rows, candidate)
    validation_x = _design(validation_rows, candidate)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(training_x)
    if candidate["algorithm"] == "ridge":
        from sklearn.linear_model import Ridge

        estimator = Ridge(alpha=float(candidate["alpha"])).fit(normalized, target)
    elif candidate["algorithm"] == "huber":
        from sklearn.linear_model import HuberRegressor

        estimator = HuberRegressor(
            alpha=float(candidate["alpha"]),
            epsilon=float(candidate["epsilon"]),
            max_iter=1000,
        ).fit(normalized, target)
    else:
        raise KeyError(candidate["algorithm"])
    parameters = {
        "algorithm": candidate["algorithm"],
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in estimator.coef_],
        "intercept": float(estimator.intercept_),
    }
    prediction = ((validation_x - scaler.mean_) / scaler.scale_) @ estimator.coef_
    prediction = prediction + estimator.intercept_
    return parameters, np.clip(prediction, 0.0, 100.0)


def _portable_predict(
    parameters: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    design = _design(rows, candidate)
    if parameters["algorithm"] == "isotonic":
        return np.interp(
            design[:, 0],
            np.asarray(parameters["x_thresholds"], dtype=float),
            np.asarray(parameters["y_thresholds"], dtype=float),
        )
    mean = np.asarray(parameters["feature_mean"], dtype=float)
    scale = np.asarray(parameters["feature_scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    prediction = ((design - mean) / scale) @ coefficients + float(
        parameters["intercept"]
    )
    return np.clip(prediction, 0.0, 100.0)


def _folds(
    molcodes: Sequence[str], count: int, *, salt: str
) -> np.ndarray:
    unique = sorted(
        set(molcodes),
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(),
    )
    mapping = {value: index % count for index, value in enumerate(unique)}
    return np.asarray([mapping[value] for value in molcodes], dtype=int)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "spearman": shared.spearman(prediction, target),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def _oof_candidates(
    rows: Sequence[Mapping[str, Any]], *, folds: int, salt: str
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    molcodes = [str(row["molcode"]) for row in rows]
    assignments = _folds(molcodes, folds, salt=salt)
    target = np.asarray([row["target"] for row in rows], dtype=float)
    predictions = {
        candidate["name"]: np.full(len(rows), np.nan) for candidate in CANDIDATES
    }
    for fold in range(folds):
        training = np.flatnonzero(assignments != fold)
        validation = np.flatnonzero(assignments == fold)
        training_rows = [rows[index] for index in training]
        validation_rows = [rows[index] for index in validation]
        for candidate in CANDIDATES:
            _, values = _fit_predict(candidate, training_rows, validation_rows)
            predictions[candidate["name"]][validation] = values
    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("v2 OOF prediction is incomplete")
    return predictions, {
        name: _metrics(values, target) for name, values in predictions.items()
    }


def _candidate_complexity(candidate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> int:
    if candidate["algorithm"] == "isotonic":
        return 2
    return int(_design(rows[:1], candidate).shape[1])


def _select(
    metrics: Mapping[str, Mapping[str, float]], rows: Sequence[Mapping[str, Any]]
) -> str:
    candidates = {row["name"]: row for row in CANDIDATES}
    return min(
        metrics,
        key=lambda name: (
            metrics[name]["mae"] + 0.01 * _candidate_complexity(candidates[name], rows),
            -metrics[name]["spearman"],
            next(index for index, row in enumerate(CANDIDATES) if row["name"] == name),
        ),
    )


def _repeated_nested(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target = np.asarray([row["target"] for row in rows], dtype=float)
    molcodes = [str(row["molcode"]) for row in rows]
    repeated = np.full((OUTER_REPEATS, len(rows)), np.nan)
    audit = []
    candidates = {row["name"]: row for row in CANDIDATES}
    for repeat in range(OUTER_REPEATS):
        outer = _folds(
            molcodes, OUTER_FOLDS, salt=f"{FOLD_SALT}|outer|{repeat}"
        )
        for fold in range(OUTER_FOLDS):
            training = np.flatnonzero(outer != fold)
            validation = np.flatnonzero(outer == fold)
            inner_rows = [rows[index] for index in training]
            _, inner_metrics = _oof_candidates(
                inner_rows,
                folds=INNER_FOLDS,
                salt=f"{FOLD_SALT}|inner|{repeat}|{fold}",
            )
            selected = _select(inner_metrics, inner_rows)
            parameters, values = _fit_predict(
                candidates[selected],
                inner_rows,
                [rows[index] for index in validation],
            )
            repeated[repeat, validation] = values
            audit.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "selected_candidate": selected,
                    "training_molecules": len({molcodes[index] for index in training}),
                    "validation_molecules": len({molcodes[index] for index in validation}),
                    "inner_metrics": inner_metrics[selected],
                    "portable_parameter_sha256": shared.canonical_json_sha256(parameters),
                }
            )
    if not np.all(np.isfinite(repeated)):
        raise RuntimeError("v2 repeated nested predictions incomplete")
    averaged = repeated.mean(axis=0)
    audit.append(
        {
            "repeat_metrics": [
                _metrics(repeated[repeat], target) for repeat in range(OUTER_REPEATS)
            ]
        }
    )
    return averaged, audit


def _participant_bootstrap(
    calibrated: np.ndarray,
    ravia: np.ndarray,
    raw: Any,
    condition_keys: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    participants = sorted(raw["code"].astype(str).unique().tolist())
    p_index = {value: index for index, value in enumerate(participants)}
    c_index = {value: index for index, value in enumerate(condition_keys)}
    values = np.zeros((len(participants), len(condition_keys)), dtype=float)
    observed = np.zeros_like(values)
    for _, row in raw.iterrows():
        key = (str(row["molcode"]), float(row["fraction"]))
        values[p_index[str(row["code"])], c_index[key]] = float(row["intensive"])
        observed[p_index[str(row["code"])], c_index[key]] = 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    spearman_delta = []
    mae_delta = []
    calibrated_mae = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(
            len(participants), np.full(len(participants), 1.0 / len(participants))
        )
        counts = weights @ observed
        if np.any(counts <= 0):
            continue
        target = (weights @ values) / counts
        sampled = rng.integers(0, len(condition_keys), size=len(condition_keys))
        spearman_delta.append(
            shared.spearman(calibrated[sampled], target[sampled])
            - shared.spearman(ravia[sampled], target[sampled])
        )
        current_mae = float(np.mean(np.abs(calibrated[sampled] - target[sampled])))
        calibrated_mae.append(current_mae)
        mae_delta.append(
            float(np.mean(np.abs(ravia[sampled] - target[sampled]))) - current_mae
        )
    if len(mae_delta) < int(BOOTSTRAP_DRAWS * 0.95):
        raise RuntimeError("too many invalid v2 bootstrap draws")
    return {
        "requested_draws": BOOTSTRAP_DRAWS,
        "valid_draws": len(mae_delta),
        "seed": BOOTSTRAP_SEED,
        "unit": "participant-cluster weights plus condition resampling",
        "calibrated_mae_95_interval": [
            float(value) for value in np.quantile(calibrated_mae, [0.025, 0.975])
        ],
        "calibrated_minus_ravia_spearman_95_interval": [
            float(value) for value in np.quantile(spearman_delta, [0.025, 0.975])
        ],
        "ravia_minus_calibrated_mae_95_interval": [
            float(value) for value in np.quantile(mae_delta, [0.025, 0.975])
        ],
    }


def _fit_final(
    rows: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Any], dict[str, dict[str, float]]]:
    repeated_metrics: dict[str, list[dict[str, float]]] = {
        candidate["name"]: [] for candidate in CANDIDATES
    }
    for repeat in range(OUTER_REPEATS):
        _, metrics = _oof_candidates(
            rows,
            folds=OUTER_FOLDS,
            salt=f"{FOLD_SALT}|final-selection|{repeat}",
        )
        for name, row in metrics.items():
            repeated_metrics[name].append(row)
    averaged_metrics = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in ("spearman", "mae", "rmse")
        }
        for name, values in repeated_metrics.items()
    }
    selected = _select(averaged_metrics, rows)
    candidate = next(row for row in CANDIDATES if row["name"] == selected)
    parameters, fitted_prediction = _fit_predict(candidate, rows, rows)
    portable_prediction = _portable_predict(parameters, candidate, rows)
    maximum_delta = float(np.max(np.abs(fitted_prediction - portable_prediction)))
    if maximum_delta > 1e-10:
        raise RuntimeError("v2 portable calibration differs from fitted model")
    parameters["candidate"] = candidate
    parameters["portable_parity_maximum_absolute_error"] = maximum_delta
    return selected, parameters, averaged_metrics


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite v2 calibration output")
    predictions_path = args.predictions.resolve(strict=True)
    pilot_path = args.pilot.resolve(strict=True)
    v1_report_path = args.v1_report.resolve(strict=True)
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    if v1_report.get("source_binding", {}).get("pilot_sha256") != _sha256(pilot_path):
        raise RuntimeError("v1 calibration/pilot binding mismatch")
    rows, raw, data_audit = _load_enriched(predictions_path, pilot_path)
    target = np.asarray([row["target"] for row in rows], dtype=float)
    nested_prediction, outer_audit = _repeated_nested(rows)
    nested_metrics = _metrics(nested_prediction, target)
    ravia = np.asarray([row["frozen_ravia_global_curve"] for row in rows], dtype=float)
    anchored = np.asarray(
        [row["condition_transfer_anchored_curve"] for row in rows], dtype=float
    )
    baselines = {
        "ravia": _metrics(ravia, target),
        "frozen_anchored": _metrics(anchored, target),
        "v1_nested_crossfit": v1_report["nested_crossfit"],
    }
    condition_keys = [(row["molcode"], float(row["fraction"])) for row in rows]
    bootstrap = _participant_bootstrap(
        nested_prediction, ravia, raw, condition_keys
    )
    selected, parameters, final_selection = _fit_final(rows)
    relative_mae_reduction = 1.0 - nested_metrics["mae"] / RAVIA_MAE
    checks = {
        "nested_mae_at_most_9": nested_metrics["mae"] <= TARGET_MAE,
        "relative_mae_reduction_at_least_25_percent": relative_mae_reduction >= 0.25,
        "nested_spearman_at_least_0_50": nested_metrics["spearman"] >= 0.50,
        "spearman_gain_bootstrap_lower_above_zero": bootstrap[
            "calibrated_minus_ravia_spearman_95_interval"
        ][0]
        > 0.0,
        "mae_gain_bootstrap_lower_above_zero": bootstrap[
            "ravia_minus_calibrated_mae_95_interval"
        ][0]
        > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "v2_large_mae_reduction_confirmed"
            if all(checks.values())
            else "v2_large_mae_reduction_not_confirmed"
        ),
        "development_timing": "post_intensity_pilot_outcome_after_v1_review",
        "source_binding": {
            "blind_predictions_sha256": _sha256(predictions_path),
            "pilot_sha256": _sha256(pilot_path),
            "v1_calibration_sha256": _sha256(v1_report_path),
        },
        "data": data_audit,
        "candidate_contract": list(CANDIDATES),
        "design_contract": {
            "base_features": list(BASE_FEATURES),
            "feature_groups": {
                name: list(features) for name, features in FEATURE_GROUPS.items()
            },
            "hinge_knots": list(HINGE_KNOTS),
            "polynomial_terms": ["anchored^2/100", "anchored^3/10000"],
            "interaction_terms": [
                "anchored*log10_delta",
                "main_anchor*log10_delta",
                "anchored*ravia/100",
                "main_anchor*ravia/100",
            ],
        },
        "protocol": {
            "outer_repeats": OUTER_REPEATS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "outer_audit": outer_audit,
            "selection": "inner MAE plus 0.01-per-feature complexity penalty",
        },
        "repeated_nested_crossfit": nested_metrics,
        "relative_mae_reduction_vs_ravia": relative_mae_reduction,
        "baselines": baselines,
        "bootstrap": bootstrap,
        "final_model": {
            "selected_candidate": selected,
            "parameters": parameters,
            "selection_metrics": final_selection[selected],
            "runtime_primary_score_weight": 0.0,
            "portable_numeric_contract": True,
        },
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "v1_builder_sha256": _sha256(Path(v1.__file__).resolve()),
        },
        "large_improvement_gate": {"passed": all(checks.values()), "checks": checks},
        "human_olfactory_90_percent_certified": False,
        "concentration_delta_validated": False,
        "claim_boundary": (
            "Post-outcome repeated nested molecule-disjoint calibration on one pilot. "
            "Even a passing gate remains diagnostic pending a new external target; "
            "mixture, recipe, and 90% olfactory claims remain unauthorized."
        ),
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    nested = report["repeated_nested_crossfit"]
    ravia = report["baselines"]["ravia"]
    bootstrap = report["bootstrap"]
    return "\n".join(
        [
            "# Bierling intensity calibration v2",
            "",
            "| 평가 | Spearman | MAE |",
            "|---|---:|---:|",
            f"| Repeated nested v2 | {nested['spearman']:.4f} | {nested['mae']:.3f} |",
            f"| Frozen Ravia | {ravia['spearman']:.4f} | {ravia['mae']:.3f} |",
            "",
            f"상대 MAE 감소: {100*report['relative_mae_reduction_vs_ravia']:.2f}%",
            "",
            "MAE 감소 95% 구간: "
            f"[{bootstrap['ravia_minus_calibrated_mae_95_interval'][0]:+.3f}, "
            f"{bootstrap['ravia_minus_calibrated_mae_95_interval'][1]:+.3f}]",
            "",
            "대폭 개선 게이트: **"
            + ("PASS" if report["large_improvement_gate"]["passed"] else "FAIL")
            + "**",
            "",
            "이 모델은 post-outcome 진단 전용이며 runtime weight는 0입니다.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = build(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
