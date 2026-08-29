#!/usr/bin/env python
"""Threshold-explicit odor-activity experiment for universal intensity.

PubChem threshold annotations are sparse and heterogeneous. This v4 therefore
adds log threshold, missingness, and threshold-normalized headspace only to
portable Ridge candidates, selects exclusively on source molecule/source
holdouts, and compares a target-excluded Ma hybrid with v2. Failure keeps the
entire threshold branch at zero runtime weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import build_universal_intensity_hybrid_v2 as hybrid_v2  # noqa: E402
from scripts import build_universal_intensity_model_v1 as v1  # noqa: E402
from scripts import build_universal_intensity_transport_v3 as transport  # noqa: E402


SCHEMA_VERSION = "4.0"
FOLD_SALT = "universal-intensity-threshold-v4"
REPEATS = 5
FOLDS = 5
THRESHOLD_FEATURES = (
    "log10_odor_threshold_ppm",
    "odor_threshold_missing",
    "odor_activity_log10_proxy",
)


def _candidates() -> tuple[dict[str, Any], ...]:
    rows = []
    for base in ("transport", "transport_physical", "transport_physical_interactions"):
        for alpha in (0.1, 1.0, 10.0, 100.0, 1_000.0):
            rows.append(
                {
                    "name": f"ridge_{base}_threshold_{alpha:g}",
                    "base_feature_set": base,
                    "alpha": alpha,
                    "portable": True,
                }
            )
    return tuple(rows)


CANDIDATES = _candidates()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _load_thresholds(path: Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    records = document.get("records", [])
    if document.get("records_sha256") != shared.canonical_json_sha256(records):
        raise RuntimeError("universal threshold record hash mismatch")
    mapping: dict[str, float | None] = {}
    for row in records:
        smiles = str(row["canonical_smiles"])
        value = row.get("odor_threshold_ppm")
        mapping[smiles] = float(value) if value is not None else None
    return mapping, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "records_sha256": document["records_sha256"],
        "coverage": document["coverage"],
    }


def _attach_thresholds(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float | None]
) -> list[dict[str, Any]]:
    headspace_index = list(transport.TRANSPORT_FEATURES).index(
        "headspace_log10_ppm_proxy"
    )
    result = []
    for row in rows:
        threshold = thresholds.get(str(row["canonical_smiles"]))
        valid = threshold is not None and math.isfinite(threshold) and threshold > 0.0
        log_threshold = math.log10(float(threshold)) if valid else math.nan
        headspace = float(row["transport"][headspace_index])
        odor_activity = (
            headspace - log_threshold
            if valid and math.isfinite(headspace)
            else math.nan
        )
        result.append(
            {
                **row,
                "threshold": np.asarray(
                    [log_threshold, float(not valid), odor_activity], dtype=float
                ),
            }
        )
    return result


def _feature_names(base: str) -> list[str]:
    return [*transport._feature_names(base), *THRESHOLD_FEATURES]


def _design(rows: Sequence[Mapping[str, Any]], base: str) -> np.ndarray:
    return np.concatenate(
        (
            transport._design(rows, base),
            np.asarray([row["threshold"] for row in rows], dtype=float),
        ),
        axis=1,
    )


def _fit_predict(
    candidate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    base = str(candidate["base_feature_set"])
    training_raw = _design(training_rows, base)
    target_raw = _design(target_rows, base)
    y = np.asarray([float(row["target"]) for row in training_rows], dtype=float)
    weights = v1._sample_weights(training_rows)
    impute, coverage = transport._imputer(training_raw, weights)
    training = transport._fill(training_raw, impute)
    target = transport._fill(target_raw, impute)
    mean = np.average(training, axis=0, weights=weights)
    variance = np.average((training - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    model = Ridge(alpha=float(candidate["alpha"]))
    model.fit((training - mean) / scale, y, sample_weight=weights)
    prediction = np.clip(model.predict((target - mean) / scale), 0.0, 1.0)
    parameters = {
        "candidate": dict(candidate),
        "feature_names": _feature_names(base),
        "imputation_values": impute.tolist(),
        "training_feature_coverage": coverage.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": float(model.intercept_),
    }
    return np.asarray(prediction), parameters


def _portable_predict(
    parameters: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    candidate = parameters["candidate"]
    raw = _design(rows, str(candidate["base_feature_set"]))
    values = transport._fill(raw, np.asarray(parameters["imputation_values"]))
    mean = np.asarray(parameters["feature_mean"])
    scale = np.asarray(parameters["feature_scale"])
    coefficients = np.asarray(parameters["coefficients"])
    return np.clip(
        ((values - mean) / scale) @ coefficients + float(parameters["intercept"]),
        0.0,
        1.0,
    )


def _cv(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    pooled = {
        str(candidate["name"]): np.zeros((REPEATS, len(rows)), dtype=float)
        for candidate in CANDIDATES
    }
    molecules = [str(row["canonical_smiles"]) for row in rows]
    for repeat in range(REPEATS):
        assignments = v1._balanced_folds(
            molecules, folds=FOLDS, salt=f"{FOLD_SALT}|{repeat}"
        )
        for fold in range(FOLDS):
            train_indices = np.flatnonzero(assignments != fold)
            test_indices = np.flatnonzero(assignments == fold)
            training = [rows[index] for index in train_indices]
            testing = [rows[index] for index in test_indices]
            for candidate in CANDIDATES:
                prediction, _ = _fit_predict(candidate, training, testing)
                pooled[str(candidate["name"])][repeat, test_indices] = prediction
    return {
        name: v1._metrics(values.mean(axis=0), rows) for name, values in pooled.items()
    }


def _transfer(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for source in v1.TARGET_SOURCE_NAMES:
        testing = [row for row in rows if row["source"] == source]
        molecules = {str(row["canonical_smiles"]) for row in testing}
        training = [
            row
            for row in rows
            if row["source"] != source and str(row["canonical_smiles"]) not in molecules
        ]
        result[source] = {"candidates": {}}
        for candidate in CANDIDATES:
            prediction, _ = _fit_predict(candidate, training, testing)
            result[source]["candidates"][str(candidate["name"])] = v1._metrics(
                prediction, testing
            )
    return result


def _select(
    metrics: Mapping[str, Mapping[str, float]],
    transfer: Mapping[str, Mapping[str, Any]],
) -> str:
    return min(
        (str(row["name"]) for row in CANDIDATES),
        key=lambda name: (
            metrics[name]["mae"]
            + 0.35
            * float(
                np.mean(
                    [
                        transfer[source]["candidates"][name]["mae"]
                        for source in v1.TARGET_SOURCE_NAMES
                    ]
                )
            ),
            max(
                transfer[source]["candidates"][name]["mae"]
                for source in v1.TARGET_SOURCE_NAMES
            ),
            name,
        ),
    )


def _markdown(report: Mapping[str, Any]) -> str:
    mono = report["ma_retrospective_evaluation"]["monomolecular"]
    mixture = report["ma_retrospective_evaluation"]["binary_mixture"]
    return "\n".join(
        [
            "# Universal intensity threshold v4",
            "",
            f"- Selected: **{report['selection']['selected_candidate']}**",
            f"- Threshold hybrid component MAE: **{mono['threshold_hybrid']['mae']:.5f}**",
            f"- v2 hybrid component MAE: **{mono['v2_hybrid']['mae']:.5f}**",
            f"- Threshold hybrid mixture MAE: **{mixture['threshold_hybrid::fechner']['mae']:.5f} / 10**",
            f"- v2 hybrid mixture MAE: **{mixture['v2_hybrid::fechner']['mae']:.5f} / 10**",
            "- Threshold gate: **"
            + ("PASS" if report["retrospective_threshold_gate"]["passed"] else "FAIL")
            + "**",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite threshold v4")
    v1_report_path = args.v1_report.resolve(strict=True)
    v2_report_path = args.v2_report.resolve(strict=True)
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    if v2_report.get("source_binding", {}).get("v1_report_sha256") != _sha256(
        v1_report_path
    ):
        raise RuntimeError("threshold v4 hybrid-v2/v1 binding mismatch")
    thresholds, threshold_audit = _load_thresholds(args.thresholds)
    properties, property_audit = transport._load_physchem(args.physchem)
    keller, _ = v1._load_keller(args)
    ravia, _ = v1._load_ravia(args.ravia_root.resolve(strict=True))
    bierling, _ = v1._load_bierling(args)
    ma_targets, ma_pairs, ma_audit = v1._load_ma_targets(args)
    ma_smiles = {str(row["canonical_smiles"]) for row in ma_targets}
    training_raw = [
        row
        for row in [*keller, *ravia, *bierling]
        if str(row["canonical_smiles"]) not in ma_smiles
    ]
    all_smiles = sorted(
        {str(row["canonical_smiles"]) for row in [*training_raw, *ma_targets]}
    )
    cache = v1.build_raw_descriptor_cache(all_smiles)
    training = _attach_thresholds(
        transport._prepare_transport(v1._prepare_rows(training_raw, cache), properties),
        thresholds,
    )
    targets = _attach_thresholds(
        transport._prepare_transport(v1._prepare_rows(ma_targets, cache), properties),
        thresholds,
    )
    metrics = _cv(training)
    transfer = _transfer(training)
    selected = _select(metrics, transfer)
    candidate = next(row for row in CANDIDATES if row["name"] == selected)
    raw_target, _ = _fit_predict(candidate, training, targets)
    fitted, parameters = _fit_predict(candidate, training, training)
    parity = float(np.max(np.abs(fitted - _portable_predict(parameters, training))))
    if parity > 1e-10:
        raise RuntimeError("threshold v4 portable parity failed")
    parameters["portable_parity_maximum_absolute_error"] = parity
    humanpom = v1._ma_component_baselines(args, targets)["humanpom"]
    v1_raw = v1._portable_predict(v1_report["final_model"]["parameters"], targets)
    v2_hybrid = hybrid_v2._equal_centered_hybrid(v1_raw, humanpom)
    threshold_hybrid = hybrid_v2._equal_centered_hybrid(raw_target, humanpom)
    predictions = {
        "threshold_hybrid": threshold_hybrid,
        "threshold_raw": raw_target,
        "v2_hybrid": v2_hybrid,
        "humanpom": humanpom,
    }
    component_metrics = {
        name: v1._metrics(values, targets) for name, values in predictions.items()
    }
    mixture_metrics = v1._ma_mixture_predictions(targets, ma_pairs, predictions, args)
    component_target = np.asarray([float(row["target"]) for row in targets])
    component_bootstrap = v1._bootstrap(
        threshold_hybrid, v2_hybrid, component_target, unit="molecule"
    )
    primary_pair, baseline_pair, pair_target = hybrid_v2._pair_vectors(
        targets, ma_pairs, threshold_hybrid, v2_hybrid, args
    )
    mixture_bootstrap = v1._bootstrap(
        primary_pair, baseline_pair, pair_target, unit="mixture"
    )
    checks = {
        "ma_exact_molecule_training_leakage_zero": not bool(
            {row["canonical_smiles"] for row in training} & ma_smiles
        ),
        "source_cv_threshold_signal_positive": metrics[selected][
            "molecule_mean_spearman"
        ]
        > 0.0,
        "ma_component_mae_below_v2": component_metrics["threshold_hybrid"]["mae"]
        < component_metrics["v2_hybrid"]["mae"],
        "ma_component_spearman_not_below_v2": component_metrics["threshold_hybrid"][
            "spearman"
        ]
        >= component_metrics["v2_hybrid"]["spearman"],
        "ma_component_mae_bootstrap_lower_above_zero": component_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "ma_mixture_mae_below_v2": mixture_metrics["threshold_hybrid::fechner"][
            "mae"
        ]
        < mixture_metrics["v2_hybrid::fechner"]["mae"],
        "ma_mixture_spearman_not_below_v2": mixture_metrics[
            "threshold_hybrid::fechner"
        ]["spearman"]
        >= mixture_metrics["v2_hybrid::fechner"]["spearman"],
        "ma_mixture_mae_bootstrap_lower_above_zero": mixture_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "portable_parity_at_most_1e_10": parity <= 1e-10,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "universal_intensity_threshold_v4_retrospective_gate_passed"
            if all(checks.values())
            else "universal_intensity_threshold_v4_retrospective_gate_failed"
        ),
        "development_timing": "designed_after_ma_minnesota_and_threshold_parser_results_known",
        "source_binding": {
            "v1_report_sha256": _sha256(v1_report_path),
            "v2_report_sha256": _sha256(v2_report_path),
            "physchem": property_audit,
            "thresholds": threshold_audit,
        },
        "threshold_contract": {
            "features": list(THRESHOLD_FEATURES),
            "missing_values": "training-weighted imputation plus explicit missing flag",
            "component_identity_features": [],
            "training_threshold_rows": sum(
                math.isfinite(float(row["threshold"][0])) for row in training
            ),
            "ma_threshold_molecules": sum(
                math.isfinite(float(row["threshold"][0])) for row in targets
            ),
        },
        "candidate_contract": list(CANDIDATES),
        "source_repeated_molecule_cv": {
            "repeats": REPEATS,
            "folds": FOLDS,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "metrics": metrics,
        },
        "source_disjoint_transfer": transfer,
        "selection": {
            "selected_candidate": selected,
            "selected_cv_metrics": metrics[selected],
            "selection_used_ma_or_minnesota_labels": False,
        },
        "final_model": {
            "parameters": parameters,
            "runtime_primary_score_weight": 0.0,
        },
        "ma_data": ma_audit,
        "ma_retrospective_evaluation": {
            "monomolecular": component_metrics,
            "binary_mixture": mixture_metrics,
            "component_bootstrap_vs_v2": component_bootstrap,
            "mixture_bootstrap_vs_v2": mixture_bootstrap,
        },
        "retrospective_threshold_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "prospective_external_gate": {
            "passed": False,
            "reason": "Minnesota was already opened before v4 design",
        },
        "runtime": {"primary_score_weight": 0.0},
        "human_olfactory_90_percent_certified": False,
        "implementation": {"script_sha256": _sha256(Path(__file__).resolve())},
        "claim_boundary": (
            "Sparse heterogeneous public thresholds in a retrospective experiment. "
            "No product headspace, prospective external, or 90% authority."
        ),
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--physchem", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--keller-molecules", type=Path, required=True)
    parser.add_argument("--keller-stimuli", type=Path, required=True)
    parser.add_argument("--keller-behavior", type=Path, required=True)
    parser.add_argument("--ravia-root", type=Path, required=True)
    parser.add_argument("--bierling-predictions", type=Path, required=True)
    parser.add_argument("--bierling-pilot", type=Path, required=True)
    parser.add_argument("--ma-predictions", type=Path, required=True)
    parser.add_argument("--ma-outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = build(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selection"]["selected_candidate"],
                "ma_component": report["ma_retrospective_evaluation"]["monomolecular"],
                "ma_mixture": report["ma_retrospective_evaluation"]["binary_mixture"],
                "gate": report["retrospective_threshold_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
