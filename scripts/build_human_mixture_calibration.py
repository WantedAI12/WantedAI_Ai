"""Build the protocol-aware Bushdid human-mixture calibration artifact.

Only the predeclared calibration partition is used to select the nuisance-
adjustment coefficient and fit the probability curve. Four-fold cross-fitted
residuals from that partition provide the conformal bound. The historical
final partition is evaluated but never used for fitting or model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.json"
DEFAULT_PREDICTIONS = ROOT / "benchmarks" / "bushdid_blind_predictions_v1.json"
DEFAULT_OUTPUT = ROOT / "fragrance_ai" / "data" / "human_mixture_calibration.json"
DEFAULT_AUDIT = ROOT / "benchmarks" / "bushdid_human_protocol_calibration_v3.json"
FOLD_SALT = "bushdid-human-protocol-calibrator-crossfit-v2"
ALPHA_MIN = -2.0
ALPHA_MAX = 0.0
ALPHA_STEP = 0.01
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_DRAWS = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _read_protocol_features(stimuli_path: Path) -> dict[str, dict[str, Any]]:
    """Read only pre-outcome stimulus construction and dilution variables."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with stimuli_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[_identifier(row.get("Stimulus", ""))].append(row)
    if not grouped or "" in grouped:
        raise RuntimeError("stimulus protocol table has missing identifiers")

    result: dict[str, dict[str, Any]] = {}
    for stimulus_id, rows in grouped.items():
        right = [row for row in rows if row.get("Answer", "").lower() == "right"]
        wrong = [row for row in rows if row.get("Answer", "").lower() == "wrong"]
        if len(rows) != 3 or len(right) != 1 or len(wrong) != 2:
            raise RuntimeError(f"stimulus {stimulus_id} is not one-right/two-wrong")
        try:
            right_dilution = float(right[0]["Stimulus dilution"])
            wrong_dilutions = tuple(
                sorted(float(row["Stimulus dilution"]) for row in wrong)
            )
            components = int(float(right[0]["Components in mixtures"]))
            overlap = float(right[0]["% mixture overlap"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"stimulus {stimulus_id} has invalid protocol values"
            ) from error
        numeric = (right_dilution, *wrong_dilutions, overlap)
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"stimulus {stimulus_id} has non-finite protocol values")
        if right_dilution <= 0.0 or any(value <= 0.0 for value in wrong_dilutions):
            raise RuntimeError(f"stimulus {stimulus_id} has non-positive dilution")
        if len(wrong_dilutions) != 2 or wrong_dilutions[0] == wrong_dilutions[1]:
            raise RuntimeError(f"stimulus {stimulus_id} wrong dilutions are invalid")
        dilution_set = sorted((right_dilution, *wrong_dilutions))
        if not np.allclose(
            dilution_set, [0.25, 0.5, 1.0], atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(f"stimulus {stimulus_id} dilution design changed")
        if any(
            int(float(row["Components in mixtures"])) != components
            or not math.isclose(float(row["% mixture overlap"]), overlap, abs_tol=1e-9)
            for row in rows
        ):
            raise RuntimeError(f"stimulus {stimulus_id} protocol rows disagree")
        wrong_log10 = np.log10(np.asarray(wrong_dilutions, dtype=float))
        result[stimulus_id] = {
            "right_dilution": right_dilution,
            "wrong_dilutions": wrong_dilutions,
            "wrong_log10_dilution_spread": float(np.std(wrong_log10)),
            "components_per_mixture": components,
            "declared_overlap_percent": overlap,
        }
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(np.asarray(left, dtype=float))
    right_rank = _average_ranks(np.asarray(right, dtype=float))
    if left_rank.std() <= 0 or right_rank.std() <= 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted pool-adjacent-violators fit over repeated x values."""

    unique = np.unique(x)
    means = np.asarray([y[x == value].mean() for value in unique], dtype=float)
    weights = np.asarray([np.sum(x == value) for value in unique], dtype=float)
    blocks: list[list[float]] = []
    for index, (value, mean, weight) in enumerate(zip(unique, means, weights)):
        blocks.append(
            [float(index), float(index), float(weight), float(mean), float(value)]
        )
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3] + 1e-15:
            right = blocks.pop()
            left = blocks.pop()
            total = left[2] + right[2]
            pooled = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], total, pooled, left[4]])
    fitted = np.empty(len(unique), dtype=float)
    for start, end, _, mean, _ in blocks:
        fitted[int(start) : int(end) + 1] = mean
    return unique, fitted


def _higher_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        raise ValueError("cannot calibrate an empty residual array")
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((len(ordered) + 1) * coverage) - 1),
    )
    return float(ordered[index])


def _protocol_score(row: Mapping[str, Any], alpha: float) -> float:
    return float(row["component_overlap_dissimilarity"]) + alpha * float(
        row["wrong_log10_dilution_spread"]
    )


def _select_alpha(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if len(rows) < 8:
        raise ValueError("protocol coefficient selection requires at least 8 rows")
    outcomes = np.asarray([row["human_correct_rate"] for row in rows], dtype=float)
    count = int(round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP)) + 1
    grid = np.linspace(ALPHA_MIN, ALPHA_MAX, count, dtype=float)
    candidates = []
    for alpha in grid:
        score = np.asarray([_protocol_score(row, float(alpha)) for row in rows])
        candidates.append(
            (_spearman(score, outcomes), -abs(float(alpha)), float(alpha))
        )
    correlation, _, selected = max(candidates)
    return selected, correlation


def _assign_crossfit_folds(rows: list[dict[str, Any]]) -> dict[str, int]:
    strata: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[
            (
                int(row["components_per_mixture"]),
                float(row["declared_overlap_percent"]),
            )
        ].append(row)
    assignments: dict[str, int] = {}
    for key, stratum in sorted(strata.items()):
        if len(stratum) != 4:
            raise RuntimeError(
                "cross-fit requires four calibration rows per stratum; "
                f"{key} has {len(stratum)}"
            )
        ordered = sorted(
            stratum,
            key=lambda row: hashlib.sha256(
                f"{FOLD_SALT}|{key}|{row['stimulus_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for fold, row in enumerate(ordered):
            assignments[str(row["stimulus_id"])] = fold
    return assignments


def _crossfit_residuals(
    rows: list[dict[str, Any]], assignments: Mapping[str, int]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    residuals: list[float] = []
    fold_audit: list[dict[str, Any]] = []
    for fold in range(4):
        train = [row for row in rows if assignments[str(row["stimulus_id"])] != fold]
        held_out = [row for row in rows if assignments[str(row["stimulus_id"])] == fold]
        alpha, train_spearman = _select_alpha(train)
        x = np.asarray([_protocol_score(row, alpha) for row in train], dtype=float)
        y = np.asarray([row["human_correct_rate"] for row in train], dtype=float)
        x_grid, y_grid = _fit_isotonic(x, y)
        predictions = np.asarray(
            [
                np.interp(_protocol_score(row, alpha), x_grid, y_grid)
                for row in held_out
            ],
            dtype=float,
        )
        outcomes = np.asarray(
            [row["human_correct_rate"] for row in held_out], dtype=float
        )
        current = np.abs(predictions - outcomes)
        residuals.extend(float(value) for value in current)
        fold_audit.append(
            {
                "fold": fold,
                "train_stimuli": len(train),
                "held_out_stimuli": len(held_out),
                "selected_dilution_spread_coefficient": alpha,
                "train_rank_spearman": train_spearman,
                "held_out_mae": float(np.mean(current)),
                "held_out_ids_sha256": hashlib.sha256(
                    _canonical_json(sorted(str(row["stimulus_id"]) for row in held_out))
                ).hexdigest(),
            }
        )
    return np.asarray(residuals, dtype=float), fold_audit


def _paired_bootstrap_interval(
    baseline: np.ndarray,
    protocol: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(outcomes), size=len(outcomes))
        differences[draw] = _spearman(protocol[indices], outcomes[indices]) - _spearman(
            baseline[indices], outcomes[indices]
        )
    return tuple(float(value) for value in np.quantile(differences, [0.025, 0.975]))


def _resolve_stimuli_path(prediction: Mapping[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    value = (
        prediction.get("dataset", {})
        .get("input_files_without_human_outcomes", {})
        .get("stimuli.csv", {})
        .get("path")
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError("sealed prediction has no stimulus protocol path")
    return Path(value).resolve()


def _attach_protocol(
    rows: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        stimulus_id = str(row["stimulus_id"])
        feature = protocol.get(stimulus_id)
        if feature is None:
            raise RuntimeError(f"stimulus {stimulus_id} is missing protocol features")
        if int(row["components_per_mixture"]) != int(
            feature["components_per_mixture"]
        ) or not math.isclose(
            float(row["declared_overlap_percent"]),
            float(feature["declared_overlap_percent"]),
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"stimulus {stimulus_id} report/protocol mismatch")
        row.update(feature)
        enriched.append(row)
    return enriched


def build(
    report_path: Path,
    *,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    stimuli_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = report_path.resolve()
    prediction_path = prediction_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    expected_prediction_hash = report.get("blind_integrity", {}).get(
        "prediction_file_sha256"
    )
    if expected_prediction_hash != sha256_file(prediction_path):
        raise RuntimeError("human report is not bound to the sealed prediction file")
    resolved_stimuli = _resolve_stimuli_path(prediction, stimuli_path)
    expected_stimulus = (
        prediction.get("dataset", {})
        .get("input_files_without_human_outcomes", {})
        .get("stimuli.csv", {})
    )
    if not resolved_stimuli.is_file():
        raise FileNotFoundError(f"missing stimulus protocol: {resolved_stimuli}")
    if sha256_file(resolved_stimuli) != expected_stimulus.get(
        "sha256"
    ) or resolved_stimuli.stat().st_size != expected_stimulus.get("bytes"):
        raise RuntimeError("stimulus protocol does not match the sealed prediction")

    protocol = _read_protocol_features(resolved_stimuli)
    rows = _attach_protocol(report["stimulus_results"], protocol)
    calibration = [row for row in rows if row["evaluation_partition"] == "calibration"]
    final = [row for row in rows if row["evaluation_partition"] == "final_test"]
    if len(calibration) != 52 or len(final) != 208:
        raise RuntimeError("registered Bushdid calibration/final partition changed")

    assignments = _assign_crossfit_folds(calibration)
    residuals, fold_audit = _crossfit_residuals(calibration, assignments)
    q95 = _higher_quantile(residuals, 0.95)
    selected_alpha, calibration_spearman = _select_alpha(calibration)
    train_x = np.asarray(
        [_protocol_score(row, selected_alpha) for row in calibration], dtype=float
    )
    train_y = np.asarray(
        [row["human_correct_rate"] for row in calibration], dtype=float
    )
    x_grid, y_grid = _fit_isotonic(train_x, train_y)

    final_baseline = np.asarray(
        [row["component_overlap_dissimilarity"] for row in final], dtype=float
    )
    final_score = np.asarray(
        [_protocol_score(row, selected_alpha) for row in final], dtype=float
    )
    final_y = np.asarray([row["human_correct_rate"] for row in final], dtype=float)
    final_prediction = np.interp(final_score, x_grid, y_grid)
    baseline_spearman = _spearman(final_baseline, final_y)
    final_spearman = _spearman(final_score, final_y)
    final_mae = float(np.mean(np.abs(final_prediction - final_y)))
    noise_ceiling = float(
        report["final_test_results"]["human_noise_ceiling"]["correlation_noise_ceiling"]
    )
    normalized = final_spearman / max(noise_ceiling, 1e-12)
    paired_interval = _paired_bootstrap_interval(final_baseline, final_score, final_y)

    artifact = {
        "schema_version": "2.0",
        "artifact_name": "bushdid_protocol_aware_human_calibration_v2",
        "endpoint": "three_alternative_odd_one_out_correct_rate",
        "source": {
            "dataset": "Bushdid 2014 raw behavior",
            "doi": "10.1126/science.1249168",
            "blind_report_sha256": sha256_file(report_path),
            "sealed_prediction_sha256": sha256_file(prediction_path),
            "stimulus_protocol_sha256": sha256_file(resolved_stimuli),
            "fit_partition": "predeclared_calibration_only",
            "historical_final_labels_used_for_parameter_selection": False,
            "development_timing": "post_unblinding_protocol_model",
        },
        "applicability_scope": {
            "matrix_ids": ["bushdid_2014_equal_presence_molecular_mixture"],
            "product_concentration_percent_range": [100.0, 100.0],
            "components_per_mixture": [10, 20, 30],
            "maximum_within_mixture_weight_cv": 1e-9,
            "required_formula_relationship": "same_size_equal_presence_component_sets",
            "study_protocol_id": "bushdid_2014_supplemental_3afc_protocol",
            "requires_registered_stimulus_table": True,
            "requires_exact_vial_dilution_design": True,
            "allowed_vial_dilutions": [0.25, 0.5, 1.0],
            "formula_projection_supported": False,
        },
        "feature_contract": {
            "component_overlap_dissimilarity_range": [0.0, 1.0],
            "dilution_feature": "population_stddev(log10(wrong_vial_dilutions))",
            "score": "component_overlap_dissimilarity + alpha * dilution_feature",
            "selected_alpha": selected_alpha,
            "alpha_grid": {
                "minimum": ALPHA_MIN,
                "maximum": ALPHA_MAX,
                "step": ALPHA_STEP,
                "selection_metric": "calibration_partition_spearman",
                "tie_break": "smallest_absolute_coefficient",
            },
        },
        "calibration": {
            "method": "protocol_score_monotonic_pava_plus_four_fold_cross_conformal",
            "fit_stimuli": len(calibration),
            "crossfit_stimuli": len(residuals),
            "fold_count": 4,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "calibration_rank_spearman": calibration_spearman,
            "x_protocol_score": [float(value) for value in x_grid],
            "predicted_correct_rate": [float(value) for value in y_grid],
            "cross_conformal_absolute_error_q95": q95,
            "folds": fold_audit,
        },
        "historical_final_evaluation": {
            "stimuli": len(final),
            "component_overlap_spearman": baseline_spearman,
            "protocol_score_spearman": final_spearman,
            "protocol_minus_overlap_spearman": final_spearman - baseline_spearman,
            "paired_stimulus_bootstrap_95_interval": list(paired_interval),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "calibrated_probability_mae": final_mae,
            "human_ceiling_normalized_rank_correlation": normalized,
            "human_similarity_90_claim_authorized": False,
            "prospective_external_validation": False,
            "claim_boundary": (
                "Retrospective protocol-aware discrimination calibration for the "
                "registered equal-presence study; not perfume smell similarity."
            ),
        },
    }
    audit = {
        "schema_version": "2.0",
        "status": "completed_retrospective_protocol_aware",
        "fit_partition": "predeclared_calibration_only",
        "fit_stimuli": len(calibration),
        "cross_conformal_stimuli": len(residuals),
        "historical_final_stimuli": len(final),
        "selected_dilution_spread_coefficient": selected_alpha,
        "r2_spearman": report["final_test_results"]["continuous_human_correct_rate"][
            "r2_spearman"
        ],
        "component_overlap_spearman": baseline_spearman,
        "protocol_aware_rank_spearman": final_spearman,
        "protocol_minus_overlap_spearman": final_spearman - baseline_spearman,
        "paired_stimulus_bootstrap_95_interval": list(paired_interval),
        "calibrated_probability_mae_percentage_points": 100.0 * final_mae,
        "cross_conformal_absolute_error_q95_percentage_points": 100.0 * q95,
        "human_ceiling_normalized_rank_correlation": normalized,
        "prospective_external_validation": False,
        "human_similarity_90_claim_authorized": False,
    }
    return artifact, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--stimuli", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    artifact, audit = build(
        args.report,
        prediction_path=args.predictions,
        stimuli_path=args.stimuli,
    )
    args.output.resolve().write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.audit.resolve().write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
