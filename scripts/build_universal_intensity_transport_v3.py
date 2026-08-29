#!/usr/bin/env python
"""Transport-explicit universal intensity model and retrospective v3 hybrid.

This model adds PubChem vapor-pressure/boiling-point evidence to the identity-
free v1 model. Missing vapor pressure may be estimated only from an observed
normal boiling point through the existing Trouton-rule fallback; measurement,
fallback and missingness are separate features. Candidate selection remains
inside Keller/Ravia/Bierling molecule/source holdouts. Ma is target-excluded and
used only for retrospective comparison against hybrid v2.
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

from fragrance_ai.recommender.science import TemporalMixtureSimulator  # noqa: E402
from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import build_universal_intensity_hybrid_v2 as hybrid_v2  # noqa: E402
from scripts import build_universal_intensity_model_v1 as v1  # noqa: E402


SCHEMA_VERSION = "3.0"
FOLD_SALT = "universal-monomolecular-intensity-transport-v3"
CV_REPEATS = 5
CV_FOLDS = 5

TRANSPORT_FEATURES = (
    "log10_fraction",
    "log10_fraction_squared",
    "log10_molarity_proxy",
    "pubchem_xlogp",
    "pubchem_tpsa",
    "pubchem_complexity_log1p",
    "pubchem_hbond_donors",
    "pubchem_hbond_acceptors",
    "pubchem_rotatable_bonds",
    "boiling_point_c_scaled",
    "boiling_point_missing",
    "log10_vapor_pressure_pa",
    "vapor_pressure_measured",
    "vapor_pressure_from_boiling",
    "vapor_pressure_missing",
    "headspace_log10_ppm_proxy",
)


def _candidates() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for feature_set in (
        "transport",
        "transport_physical",
        "transport_physical_interactions",
    ):
        for alpha in (0.1, 1.0, 10.0, 100.0, 1_000.0):
            rows.append(
                {
                    "name": f"ridge_{feature_set}_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "portable": True,
                }
            )
        rows.extend(
            [
                {
                    "name": f"extra_trees_{feature_set}",
                    "algorithm": "extra_trees",
                    "feature_set": feature_set,
                    "portable": False,
                },
                {
                    "name": f"hist_gradient_boosting_{feature_set}",
                    "algorithm": "hist_gradient_boosting",
                    "feature_set": feature_set,
                    "portable": False,
                },
            ]
        )
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("transport v3 candidate names are not unique")
    return tuple(rows)


CANDIDATES = _candidates()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _load_physchem(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    records = document.get("records", [])
    if document.get("records_sha256") != shared.canonical_json_sha256(records):
        raise RuntimeError("universal physchem record hash mismatch")
    if document.get("implementation", {}).get("script_sha256") != _sha256(
        PROJECT_ROOT / "scripts" / "acquire_universal_intensity_physchem_v1.py"
    ):
        raise RuntimeError("universal physchem implementation binding changed")
    by_smiles = {str(row["canonical_smiles"]): row for row in records}
    if len(by_smiles) != len(records) or len(records) < 500:
        raise RuntimeError("universal physchem structure registry is invalid")
    return by_smiles, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "records_sha256": document["records_sha256"],
        "coverage": document["coverage"],
    }


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _transport_row(
    row: Mapping[str, Any], properties: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    smiles = str(row["canonical_smiles"])
    prop = properties.get(smiles)
    if prop is None:
        raise RuntimeError(f"transport property registry lacks structure: {smiles}")
    boiling = _finite_or_none(prop.get("boiling_point_c"))
    measured_vapor = _finite_or_none(prop.get("vapor_pressure_pa_15_30c"))
    if measured_vapor is not None and measured_vapor > 0.0:
        vapor = measured_vapor
        vapor_measured = 1.0
        vapor_from_boiling = 0.0
    elif boiling is not None:
        vapor = TemporalMixtureSimulator._boiling_point_to_vapor_pressure(boiling)
        vapor_measured = 0.0
        vapor_from_boiling = 1.0
    else:
        vapor = None
        vapor_measured = 0.0
        vapor_from_boiling = 0.0
    log_fraction = float(row["log_fraction"])
    log_vapor = math.log10(vapor) if vapor is not None and vapor > 0.0 else math.nan
    headspace = (
        log_fraction + log_vapor - math.log10(101_325.0) + 6.0
        if math.isfinite(log_vapor)
        else math.nan
    )
    values = {
        "log10_fraction": log_fraction,
        "log10_fraction_squared": log_fraction**2,
        "log10_molarity_proxy": float(row["log_molarity_proxy"]),
        "pubchem_xlogp": _finite_or_none(prop.get("xlogp")),
        "pubchem_tpsa": _finite_or_none(prop.get("tpsa")),
        "pubchem_complexity_log1p": (
            math.log1p(float(prop["complexity"]))
            if _finite_or_none(prop.get("complexity")) is not None
            and float(prop["complexity"]) >= 0.0
            else None
        ),
        "pubchem_hbond_donors": _finite_or_none(prop.get("hbond_donors")),
        "pubchem_hbond_acceptors": _finite_or_none(prop.get("hbond_acceptors")),
        "pubchem_rotatable_bonds": _finite_or_none(prop.get("rotatable_bonds")),
        "boiling_point_c_scaled": boiling / 250.0 if boiling is not None else None,
        "boiling_point_missing": float(boiling is None),
        "log10_vapor_pressure_pa": log_vapor if math.isfinite(log_vapor) else None,
        "vapor_pressure_measured": vapor_measured,
        "vapor_pressure_from_boiling": vapor_from_boiling,
        "vapor_pressure_missing": float(vapor is None),
        "headspace_log10_ppm_proxy": headspace if math.isfinite(headspace) else None,
    }
    return {
        **row,
        "transport": np.asarray(
            [float(values[name]) if values[name] is not None else math.nan for name in TRANSPORT_FEATURES],
            dtype=float,
        ),
        "transport_evidence": {
            "vapor_measured": bool(vapor_measured),
            "vapor_from_boiling": bool(vapor_from_boiling),
            "vapor_missing": vapor is None,
            "boiling_present": boiling is not None,
        },
    }


def _prepare_transport(
    rows: Sequence[Mapping[str, Any]],
    properties: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [_transport_row(row, properties) for row in rows]


def _feature_names(feature_set: str) -> list[str]:
    physical = list(v1.PHYSICAL_DESCRIPTOR_NAMES)
    if feature_set == "transport":
        return list(TRANSPORT_FEATURES)
    if feature_set == "transport_physical":
        return [*TRANSPORT_FEATURES, *physical]
    if feature_set == "transport_physical_interactions":
        return [
            *TRANSPORT_FEATURES,
            *physical,
            *(f"{name}*headspace_log10_ppm" for name in physical),
        ]
    raise KeyError(feature_set)


def _design(rows: Sequence[Mapping[str, Any]], feature_set: str) -> np.ndarray:
    transport = np.asarray([row["transport"] for row in rows], dtype=float)
    if feature_set == "transport":
        return transport
    physical = np.asarray([row["physical"] for row in rows], dtype=float)
    values = np.concatenate((transport, physical), axis=1)
    if feature_set == "transport_physical_interactions":
        headspace_index = list(TRANSPORT_FEATURES).index("headspace_log10_ppm_proxy")
        headspace = transport[:, headspace_index : headspace_index + 1]
        values = np.concatenate((values, physical * headspace), axis=1)
    return values


def _imputer(
    training: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    for column in range(training.shape[1]):
        valid = np.isfinite(training[:, column])
        if not np.any(valid):
            means.append(0.0)
        else:
            means.append(
                float(
                    np.average(training[valid, column], weights=weights[valid])
                )
            )
    return np.asarray(means, dtype=float), np.isfinite(training).mean(axis=0)


def _fill(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, means[None, :])


def _fit_predict(
    candidate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_set = str(candidate["feature_set"])
    training_raw = _design(training_rows, feature_set)
    target_raw = _design(target_rows, feature_set)
    y = np.asarray([float(row["target"]) for row in training_rows], dtype=float)
    weights = v1._sample_weights(training_rows)
    impute, training_coverage = _imputer(training_raw, weights)
    training = _fill(training_raw, impute)
    target = _fill(target_raw, impute)
    algorithm = str(candidate["algorithm"])
    if algorithm == "ridge":
        from sklearn.linear_model import Ridge

        mean = np.average(training, axis=0, weights=weights)
        variance = np.average((training - mean) ** 2, axis=0, weights=weights)
        scale = np.sqrt(np.maximum(variance, 1e-12))
        estimator = Ridge(alpha=float(candidate["alpha"]))
        estimator.fit((training - mean) / scale, y, sample_weight=weights)
        prediction = estimator.predict((target - mean) / scale)
        parameters = {
            "candidate": dict(candidate),
            "feature_names": _feature_names(feature_set),
            "imputation_values": impute.astype(float).tolist(),
            "training_feature_coverage": training_coverage.astype(float).tolist(),
            "feature_mean": mean.astype(float).tolist(),
            "feature_scale": scale.astype(float).tolist(),
            "coefficients": np.asarray(estimator.coef_, dtype=float).tolist(),
            "intercept": float(estimator.intercept_),
        }
    elif algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=4,
            max_features=0.7,
            random_state=20_260_903,
            n_jobs=-1,
        )
        estimator.fit(training, y, sample_weight=weights)
        prediction = estimator.predict(target)
        parameters = {"candidate": dict(candidate), "portable": False}
    elif algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        estimator = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=9,
            min_samples_leaf=10,
            l2_regularization=3.0,
            random_state=20_260_903,
        )
        estimator.fit(training_raw, y, sample_weight=weights)
        prediction = estimator.predict(target_raw)
        parameters = {"candidate": dict(candidate), "portable": False}
    else:
        raise KeyError(algorithm)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"non-finite transport prediction: {candidate['name']}")
    return prediction, parameters


def _portable_predict(
    parameters: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    candidate = parameters["candidate"]
    raw = _design(rows, str(candidate["feature_set"]))
    impute = np.asarray(parameters["imputation_values"], dtype=float)
    values = _fill(raw, impute)
    mean = np.asarray(parameters["feature_mean"], dtype=float)
    scale = np.asarray(parameters["feature_scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    prediction = ((values - mean) / scale) @ coefficients + float(
        parameters["intercept"]
    )
    return np.clip(prediction, 0.0, 1.0)


def _cv(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    pooled = {
        str(candidate["name"]): np.zeros((CV_REPEATS, len(rows)), dtype=float)
        for candidate in candidates
    }
    molecules = [str(row["canonical_smiles"]) for row in rows]
    for repeat in range(CV_REPEATS):
        assignments = v1._balanced_folds(
            molecules,
            folds=CV_FOLDS,
            salt=f"{FOLD_SALT}|{repeat}",
        )
        for fold in range(CV_FOLDS):
            train_indices = np.flatnonzero(assignments != fold)
            test_indices = np.flatnonzero(assignments == fold)
            training = [rows[index] for index in train_indices]
            testing = [rows[index] for index in test_indices]
            for candidate in candidates:
                prediction, _ = _fit_predict(candidate, training, testing)
                pooled[str(candidate["name"])][repeat, test_indices] = prediction
    averaged = {name: values.mean(axis=0) for name, values in pooled.items()}
    return averaged, {
        name: v1._metrics(prediction, rows) for name, prediction in averaged.items()
    }


def _source_transfer(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for source in v1.TARGET_SOURCE_NAMES:
        testing = [row for row in rows if row["source"] == source]
        molecules = {str(row["canonical_smiles"]) for row in testing}
        training = [
            row
            for row in rows
            if row["source"] != source and str(row["canonical_smiles"]) not in molecules
        ]
        result[source] = {
            "training_rows": len(training),
            "testing_rows": len(testing),
            "exact_molecule_leakage_count": 0,
            "candidates": {},
        }
        for candidate in candidates:
            prediction, _ = _fit_predict(candidate, training, testing)
            result[source]["candidates"][str(candidate["name"])] = v1._metrics(
                prediction, testing
            )
    return result


def _select(
    metrics: Mapping[str, Mapping[str, float]],
    transfer: Mapping[str, Mapping[str, Any]],
) -> str:
    portable = [str(row["name"]) for row in CANDIDATES if row["portable"]]
    return min(
        portable,
        key=lambda name: (
            float(metrics[name]["mae"])
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
            -float(metrics[name]["molecule_mean_spearman"]),
            name,
        ),
    )


def _markdown(report: Mapping[str, Any]) -> str:
    mono = report["ma_retrospective_evaluation"]["monomolecular"]
    mixture = report["ma_retrospective_evaluation"]["binary_mixture"]
    return "\n".join(
        [
            "# Universal intensity transport v3",
            "",
            f"- Selected: **{report['selection']['selected_candidate']}**",
            f"- Component transport hybrid MAE: **{mono['transport_hybrid']['mae']:.5f}**",
            f"- Component v2 hybrid MAE: **{mono['v2_hybrid']['mae']:.5f}**",
            f"- Mixture transport hybrid+Fechner MAE: **{mixture['transport_hybrid::fechner']['mae']:.5f} / 10**",
            f"- Mixture v2 hybrid+Fechner MAE: **{mixture['v2_hybrid::fechner']['mae']:.5f} / 10**",
            "- Retrospective transport gate: **"
            + ("PASS" if report["retrospective_transport_gate"]["passed"] else "FAIL")
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
        raise RuntimeError("refusing to overwrite universal transport v3")
    v1_report_path = args.v1_report.resolve(strict=True)
    v2_report_path = args.v2_report.resolve(strict=True)
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    if v2_report.get("source_binding", {}).get("v1_report_sha256") != _sha256(
        v1_report_path
    ):
        raise RuntimeError("universal hybrid v2/v1 binding mismatch")
    properties, property_audit = _load_physchem(args.physchem.resolve(strict=True))

    keller_rows, _ = v1._load_keller(args)
    ravia_rows, _ = v1._load_ravia(args.ravia_root.resolve(strict=True))
    bierling_rows, _ = v1._load_bierling(args)
    ma_targets, ma_pairs, ma_audit = v1._load_ma_targets(args)
    ma_smiles = {str(row["canonical_smiles"]) for row in ma_targets}
    training_raw = [
        row
        for row in [*keller_rows, *ravia_rows, *bierling_rows]
        if str(row["canonical_smiles"]) not in ma_smiles
    ]
    all_smiles = sorted(
        {str(row["canonical_smiles"]) for row in [*training_raw, *ma_targets]}
    )
    descriptor_cache = v1.build_raw_descriptor_cache(all_smiles)
    training_base = v1._prepare_rows(training_raw, descriptor_cache)
    target_base = v1._prepare_rows(ma_targets, descriptor_cache)
    training = _prepare_transport(training_base, properties)
    targets = _prepare_transport(target_base, properties)

    cv_prediction, cv_metrics = _cv(training, CANDIDATES)
    transfer = _source_transfer(training, CANDIDATES)
    selected = _select(cv_metrics, transfer)
    candidate = next(row for row in CANDIDATES if row["name"] == selected)
    raw_target, _ = _fit_predict(candidate, training, targets)
    fitted, parameters = _fit_predict(candidate, training, training)
    parity = float(np.max(np.abs(fitted - _portable_predict(parameters, training))))
    if parity > 1e-10:
        raise RuntimeError("transport v3 portable parity failed")
    parameters["portable_parity_maximum_absolute_error"] = parity

    humanpom = v1._ma_component_baselines(args, target_base)["humanpom"]
    v1_raw = v1._portable_predict(v1_report["final_model"]["parameters"], target_base)
    v2_hybrid = hybrid_v2._equal_centered_hybrid(v1_raw, humanpom)
    transport_hybrid = hybrid_v2._equal_centered_hybrid(raw_target, humanpom)
    component_predictions = {
        "transport_hybrid": transport_hybrid,
        "transport_raw": raw_target,
        "v2_hybrid": v2_hybrid,
        "humanpom": humanpom,
    }
    component_metrics = {
        name: v1._metrics(values, targets)
        for name, values in component_predictions.items()
    }
    mixture_metrics = v1._ma_mixture_predictions(
        targets, ma_pairs, component_predictions, args
    )
    component_target = np.asarray([float(row["target"]) for row in targets])
    component_bootstrap = v1._bootstrap(
        transport_hybrid,
        v2_hybrid,
        component_target,
        unit="molecule",
    )
    primary_pair, baseline_pair, pair_target = hybrid_v2._pair_vectors(
        targets, ma_pairs, transport_hybrid, v2_hybrid, args
    )
    mixture_bootstrap = v1._bootstrap(
        primary_pair,
        baseline_pair,
        pair_target,
        unit="mixture",
    )
    checks = {
        "ma_exact_molecule_training_leakage_zero": not bool(
            {row["canonical_smiles"] for row in training} & ma_smiles
        ),
        "selected_transport_model_has_source_cv_signal": cv_metrics[selected][
            "molecule_mean_spearman"
        ]
        > 0.0,
        "ma_component_mae_below_v2": component_metrics["transport_hybrid"]["mae"]
        < component_metrics["v2_hybrid"]["mae"],
        "ma_component_spearman_not_below_v2": component_metrics["transport_hybrid"][
            "spearman"
        ]
        >= component_metrics["v2_hybrid"]["spearman"],
        "ma_component_mae_bootstrap_lower_above_zero": component_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "ma_mixture_fechner_mae_below_v2": mixture_metrics[
            "transport_hybrid::fechner"
        ]["mae"]
        < mixture_metrics["v2_hybrid::fechner"]["mae"],
        "ma_mixture_fechner_spearman_not_below_v2": mixture_metrics[
            "transport_hybrid::fechner"
        ]["spearman"]
        >= mixture_metrics["v2_hybrid::fechner"]["spearman"],
        "ma_mixture_mae_bootstrap_lower_above_zero": mixture_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "portable_numeric_parity_at_most_1e_10": parity <= 1e-10,
    }
    evidence = {
        "training_vapor_measured_rows": sum(
            row["transport_evidence"]["vapor_measured"] for row in training
        ),
        "training_vapor_from_boiling_rows": sum(
            row["transport_evidence"]["vapor_from_boiling"] for row in training
        ),
        "training_vapor_missing_rows": sum(
            row["transport_evidence"]["vapor_missing"] for row in training
        ),
        "ma_vapor_measured_molecules": sum(
            row["transport_evidence"]["vapor_measured"] for row in targets
        ),
        "ma_vapor_from_boiling_molecules": sum(
            row["transport_evidence"]["vapor_from_boiling"] for row in targets
        ),
        "ma_vapor_missing_molecules": sum(
            row["transport_evidence"]["vapor_missing"] for row in targets
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "universal_intensity_transport_v3_retrospective_gate_passed"
            if all(checks.values())
            else "universal_intensity_transport_v3_retrospective_gate_failed"
        ),
        "development_timing": "designed_after_ma_and_hybrid_v2_results_were_known",
        "source_binding": {
            "v1_report_sha256": _sha256(v1_report_path),
            "v2_report_sha256": _sha256(v2_report_path),
            "physchem": property_audit,
            "ma_predictions_sha256": _sha256(args.ma_predictions.resolve(strict=True)),
            "ma_outcome_sha256": _sha256(args.ma_outcome.resolve(strict=True)),
        },
        "transport_contract": {
            "features": list(TRANSPORT_FEATURES),
            "boiling_to_vapor_fallback": "ScientificDigitalTwin Trouton-rule prior",
            "missing_values": "training-weighted mean plus explicit missing flags",
            "headspace_proxy": "log10(fraction*vapor_pressure/101325*1e6)",
            "component_or_ingredient_identity_features": [],
            "evidence_counts": evidence,
        },
        "candidate_contract": list(CANDIDATES),
        "source_repeated_molecule_cv": {
            "repeats": CV_REPEATS,
            "folds": CV_FOLDS,
            "fold_salt_sha256": hashlib.sha256(FOLD_SALT.encode()).hexdigest(),
            "metrics": cv_metrics,
        },
        "source_disjoint_transfer": transfer,
        "selection": {
            "selected_candidate": selected,
            "selected_cv_metrics": cv_metrics[selected],
            "selection_used_ma_labels": False,
        },
        "final_model": {
            "parameters": parameters,
            "runtime_primary_score_weight": 0.0,
            "runtime_status": "retrospective_validation_gated_diagnostic_only",
        },
        "ma_data": ma_audit,
        "ma_retrospective_evaluation": {
            "monomolecular": component_metrics,
            "binary_mixture": mixture_metrics,
            "component_bootstrap_vs_v2": component_bootstrap,
            "mixture_bootstrap_vs_v2": mixture_bootstrap,
        },
        "retrospective_transport_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "prospective_external_gate": {
            "passed": False,
            "reason": "no outcome-unseen post-v3 external intensity target",
        },
        "runtime": {"primary_score_weight": 0.0},
        "human_olfactory_90_percent_certified": False,
        "implementation": {"script_sha256": _sha256(Path(__file__).resolve())},
        "claim_boundary": (
            "Retrospective target-excluded transport experiment. PubChem properties "
            "are public molecular evidence, not product headspace measurements. The "
            "model was designed after Ma was known and has zero runtime weight."
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
                "ma_component": report["ma_retrospective_evaluation"][
                    "monomolecular"
                ],
                "ma_mixture": report["ma_retrospective_evaluation"][
                    "binary_mixture"
                ],
                "gate": report["retrospective_transport_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
