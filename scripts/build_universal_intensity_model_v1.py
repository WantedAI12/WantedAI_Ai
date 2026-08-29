#!/usr/bin/env python
"""Build a target-excluded universal monomolecular intensity model.

Keller 2016, Ravia 2020 and Bierling 2025 supply normalized human intensity
ratings at recorded dilution conditions.  Every exact Ma 2021 molecule is
removed before model selection or fitting.  Candidate selection uses repeated
molecule-disjoint CV and source-transfer diagnostics only; Ma monomolecular and
binary-mixture outcomes are read once for retrospective external evaluation.

This v1 deliberately uses molecule-intrinsic RDKit descriptors and a common
concentration proxy, never component IDs.  A passing report remains diagnostic
because the model was designed after the Ma outcome became known.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender.concentration_response import (  # noqa: E402
    FrozenConcentrationResponse,
)
from fragrance_ai.research.r2_physsim import (  # noqa: E402
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    canonical_smiles,
    descriptor_contract,
)
from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_bierling_intensity_pilot_benchmark as bierling  # noqa: E402
from scripts import blind_ma_2021_binary_mixture_benchmark as ma  # noqa: E402
from scripts import build_bierling_intensity_crossfit_calibration as bierling_crossfit  # noqa: E402


SCHEMA_VERSION = "1.0"
FOLD_SALT = "universal-monomolecular-intensity-v1"
CV_REPEATS = 5
CV_FOLDS = 5
BOOTSTRAP_SEED = 20_260_901
BOOTSTRAP_DRAWS = 10_000
TARGET_SOURCE_NAMES = ("keller_2016", "ravia_2020", "bierling_2025")

RAVIA_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
RAVIA_SOURCE_CONTRACT = {
    "molecules.csv": {
        "bytes": 25_686,
        "sha256": "ea03779465f0637d2993ba2c5224106e6575149499d11df92c61c800b09a6594",
    },
    "stimuli.csv": {
        "bytes": 41_983,
        "sha256": "49122b86eee2b48ad1fc9690c1dcbd418e2e0eb2175bbe0f8f735da513ea2cf7",
    },
    "behavior_1.csv": {
        "bytes": 7_897,
        "sha256": "1434107c9b97151549f848cb22d92742d3b17477e34e530bbc736dcc3029b223",
    },
    "manifest.toml": {
        "bytes": 795,
        "sha256": "8a4968ab727da3bf802ead9e2a906bfd55e4da99990c2c73d7fa7240d1a12cd7",
    },
}

PHYSICAL_DESCRIPTOR_NAMES = (
    "MaxAbsEStateIndex",
    "MinEStateIndex",
    "qed",
    "SPS",
    "MolWt",
    "ExactMolWt",
    "MaxPartialCharge",
    "MinPartialCharge",
    "FpDensityMorgan2",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "TPSA",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_C_O",
    "fr_C_S",
    "fr_aldehyde",
    "fr_ester",
    "fr_ether",
    "fr_ketone",
    "fr_lactone",
    "fr_sulfide",
)


def _candidate_contract() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1_000.0):
        rows.extend(
            [
                {
                    "name": f"ridge_concentration_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": "concentration",
                    "alpha": alpha,
                    "portable": True,
                },
                {
                    "name": f"ridge_physical_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": "physical",
                    "alpha": alpha,
                    "portable": True,
                },
                {
                    "name": f"ridge_physical_interactions_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": "physical_interactions",
                    "alpha": alpha,
                    "portable": True,
                },
                {
                    "name": f"ridge_rdkit_interactions_{alpha:g}",
                    "algorithm": "ridge",
                    "feature_set": "rdkit_interactions",
                    "alpha": alpha,
                    "portable": True,
                },
            ]
        )
    for feature_set in ("physical", "physical_interactions", "rdkit"):
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
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("universal intensity candidate names are not unique")
    return tuple(rows)


CANDIDATES = _candidate_contract()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _verify_source(path: Path, expected: Mapping[str, Any], name: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise RuntimeError(f"source contract changed: {name}")
    return actual


def _load_ravia(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {
        name: _verify_source(root / name, contract, f"ravia/{name}")
        for name, contract in RAVIA_SOURCE_CONTRACT.items()
    }
    with (root / "molecules.csv").open(encoding="utf-8-sig", newline="") as handle:
        cid_to_smiles = {
            str(row["CID"]).strip(): canonical_smiles(str(row["IsomericSMILES"]).strip())
            for row in csv.DictReader(handle)
            if row.get("CID") and row.get("IsomericSMILES")
        }
    with (root / "stimuli.csv").open(encoding="utf-8-sig", newline="") as handle:
        stimuli = {str(row["Stimulus"]).strip(): row for row in csv.DictReader(handle)}
    rows = []
    with (root / "behavior_1.csv").open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            if not str(source.get("Intensity", "")).strip():
                continue
            stimulus = stimuli.get(str(source.get("Stimulus", "")).strip())
            if stimulus is None or stimulus.get("Type") != "mono-molecule":
                continue
            raw_dilution = str(stimulus.get(f"Dilution{source['Dilution #']}", "")).strip()
            try:
                fraction = float(raw_dilution.rstrip("%")) / 100.0
                target = float(source["Intensity"]) / 100.0
            except ValueError:
                continue
            smiles = cid_to_smiles.get(str(stimulus.get("CID", "")).strip())
            if (
                smiles is None
                or not 0.0 < fraction <= 1.0
                or not 0.0 <= target <= 1.0
            ):
                continue
            rows.append(
                {
                    "source": "ravia_2020",
                    "canonical_smiles": smiles,
                    "concentration_fraction": fraction,
                    "target": target,
                    "condition_id": f"ravia:{source['Stimulus']}:{source['Dilution #']}",
                }
            )
    if len(rows) < 250 or len({row["canonical_smiles"] for row in rows}) < 65:
        raise RuntimeError("too few Ravia intensity rows")
    return rows, {
        "commit": RAVIA_COMMIT,
        "doi": "10.1038/s41586-020-2891-7",
        "sources": sources,
        "rows": len(rows),
        "molecules": len({row["canonical_smiles"] for row in rows}),
    }


def _load_keller(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        "molecules.csv": args.keller_molecules.resolve(strict=True),
        "stimuli.csv": args.keller_stimuli.resolve(strict=True),
        "behavior.csv": args.keller_behavior.resolve(strict=True),
    }
    for name, expected in shared.KELLER_SOURCE_CONTRACT.items():
        _verify_source(paths[name], expected, f"keller/{name}")
    source_rows, audit = bierling._load_keller_intensity(
        paths["molecules.csv"], paths["stimuli.csv"], paths["behavior.csv"]
    )
    rows = [
        {
            "source": "keller_2016",
            "canonical_smiles": str(row["canonical_smiles"]),
            "concentration_fraction": float(row["concentration_fraction"]),
            "target": float(row["intensity"]) / 100.0,
            "condition_id": f"keller:{row['stimulus']}",
        }
        for row in source_rows
    ]
    return rows, {
        **audit,
        "doi": "10.1186/s12868-016-0287-2",
        "sources": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def _load_bierling(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_path = args.bierling_predictions.resolve(strict=True)
    pilot_path = args.bierling_pilot.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    source_by_code = {str(row["molcode"]): row for row in predictions["predictions"]}
    conditions, _, audit = bierling_crossfit._load_conditions(predictions_path, pilot_path)
    rows = []
    for condition in conditions:
        source = source_by_code.get(str(condition["molcode"]))
        if source is None:
            raise RuntimeError("Bierling condition lacks frozen structure")
        rows.append(
            {
                "source": "bierling_2025",
                "canonical_smiles": str(source["canonical_smiles"]),
                "concentration_fraction": float(condition["fraction"]),
                "target": float(condition["target"]) / 100.0,
                "condition_id": (
                    f"bierling:{condition['molcode']}:{float(condition['fraction']):.12g}"
                ),
            }
        )
    return rows, {
        **audit,
        "doi": "10.1038/s41597-025-04644-2",
        "predictions_sha256": _sha256(predictions_path),
        "pilot_sha256": _sha256(pilot_path),
    }


def _load_ma_targets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions_path = args.ma_predictions.resolve(strict=True)
    outcome_path = args.ma_outcome.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    _, individual, parser_audit = ma._load_outcome(outcome_path)
    metadata_by_name = {
        str(row["normalized_odorant"]): row for row in predictions["target_odorants"]
    }
    participant_molecule: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, source in individual.iterrows():
        for name_column, intensity_column in (("odor A", "IA"), ("odor B", "IB")):
            normalized = ma.normalize_name(source[name_column])
            metadata = metadata_by_name.get(normalized)
            if metadata is None:
                raise RuntimeError(f"Ma monomolecular target name is unmapped: {source[name_column]}")
            participant_molecule[
                (str(source["subject_id"]), str(metadata["cas"]))
            ].append(float(source[intensity_column]))
    molecule_participant_means: dict[str, list[float]] = defaultdict(list)
    for (_, cas), values in participant_molecule.items():
        molecule_participant_means[cas].append(float(np.mean(values)))

    target_rows = []
    for metadata in predictions["target_odorants"]:
        cas = str(metadata["cas"])
        values = molecule_participant_means.get(cas, [])
        if len(values) < 10:
            raise RuntimeError(f"too few Ma participant means for {cas}")
        fraction_proxy = float(metadata["concentration_mg_ml"]) / 1_000.0
        target_rows.append(
            {
                "source": "ma_2021",
                "canonical_smiles": str(metadata["canonical_smiles"]),
                "cas": cas,
                "concentration_fraction": fraction_proxy,
                "target": float(np.mean(values)) / 10.0,
                "participant_count": len(values),
                "condition_id": f"ma:{cas}",
            }
        )

    scored_rows, _ = ma._scored_rows(individual, predictions)
    pair_rows = ma._aggregate(scored_rows, "pair")
    if len(target_rows) != 72 or len(pair_rows) != 198:
        raise RuntimeError("Ma target counts changed")
    return target_rows, pair_rows, {
        **parser_audit,
        "monomolecular_targets": len(target_rows),
        "binary_mixture_targets": len(pair_rows),
        "prediction_sha256": _sha256(predictions_path),
        "outcome_sha256": _sha256(outcome_path),
        "concentration_proxy": "mg/mL divided by 1000 as approximate liquid fraction",
        "aggregation": "within-participant molecule mean, then equal-participant mean",
    }


def _descriptor_indices() -> tuple[list[str], np.ndarray]:
    names = [name for name, _ in descriptor_contract()]
    missing = sorted(set(PHYSICAL_DESCRIPTOR_NAMES) - set(names))
    if missing:
        raise RuntimeError(f"physical RDKit descriptor contract changed: {missing}")
    return names, np.asarray([names.index(name) for name in PHYSICAL_DESCRIPTOR_NAMES], dtype=int)


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]], raw_cache: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    descriptor_names, physical_indices = _descriptor_indices()
    molwt_index = descriptor_names.index("MolWt")
    prepared = []
    for row in rows:
        smiles = str(row["canonical_smiles"])
        descriptors = np.asarray(raw_cache[smiles], dtype=float)
        fraction = float(row["concentration_fraction"])
        molecular_weight = max(1.0, float(descriptors[molwt_index]))
        log_fraction = math.log10(float(np.clip(fraction, 1e-9, 1.0)))
        molarity_proxy = max(1e-12, 1_000.0 * fraction / molecular_weight)
        prepared.append(
            {
                **row,
                "descriptors": descriptors,
                "physical": descriptors[physical_indices],
                "log_fraction": log_fraction,
                "log_molarity_proxy": math.log10(molarity_proxy),
                "scaffold": bemis_murcko_scaffold(smiles),
            }
        )
    return prepared


def _feature_names(feature_set: str) -> list[str]:
    concentration = [
        "log10_fraction",
        "log10_fraction_squared",
        "log10_molarity_proxy",
        "log10_molarity_proxy_squared",
    ]
    physical = list(PHYSICAL_DESCRIPTOR_NAMES)
    if feature_set == "concentration":
        return concentration
    if feature_set == "physical":
        return [*physical, *concentration]
    if feature_set == "physical_interactions":
        return [
            *physical,
            *concentration,
            *(f"{name}*log10_fraction" for name in physical),
            "balanced_volatility_proxy",
        ]
    descriptor_names = [name for name, _ in descriptor_contract()]
    if feature_set == "rdkit":
        return [*descriptor_names, *concentration]
    if feature_set == "rdkit_interactions":
        return [
            *descriptor_names,
            *concentration,
            *(f"{name}*log10_fraction" for name in descriptor_names),
        ]
    raise KeyError(feature_set)


def _design(rows: Sequence[Mapping[str, Any]], feature_set: str) -> np.ndarray:
    concentration = np.asarray(
        [
            [
                float(row["log_fraction"]),
                float(row["log_fraction"]) ** 2,
                float(row["log_molarity_proxy"]),
                float(row["log_molarity_proxy"]) ** 2,
            ]
            for row in rows
        ],
        dtype=float,
    )
    if feature_set == "concentration":
        return concentration
    if feature_set.startswith("physical"):
        descriptors = np.asarray([row["physical"] for row in rows], dtype=float)
    else:
        descriptors = np.asarray([row["descriptors"] for row in rows], dtype=float)
    values = np.concatenate((descriptors, concentration), axis=1)
    if feature_set.endswith("interactions"):
        log_fraction = concentration[:, :1]
        interactions = descriptors * log_fraction
        values = np.concatenate((values, interactions), axis=1)
        if feature_set == "physical_interactions":
            # A bounded transport proxy: lipophilicity minus molecular-size burden,
            # multiplied by available liquid fraction. It is descriptor-derived,
            # not a measured vapor pressure.
            names = list(PHYSICAL_DESCRIPTOR_NAMES)
            logp = descriptors[:, names.index("MolLogP")]
            molwt = descriptors[:, names.index("MolWt")]
            transport = np.tanh(logp / 4.0 - molwt / 500.0) * np.exp(
                np.clip(log_fraction[:, 0], -9.0, 0.0)
            )
            values = np.concatenate((values, transport[:, None]), axis=1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite universal intensity features: {feature_set}")
    return values


def _sample_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    source_counts = Counter(str(row["source"]) for row in rows)
    molecule_source_counts = Counter(
        (str(row["source"]), str(row["canonical_smiles"])) for row in rows
    )
    weights = np.asarray(
        [
            1.0
            / source_counts[str(row["source"])]
            / molecule_source_counts[(str(row["source"]), str(row["canonical_smiles"]))]
            for row in rows
        ],
        dtype=float,
    )
    return weights / weights.mean()


def _fit_predict(
    candidate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_set = str(candidate["feature_set"])
    training = _design(training_rows, feature_set)
    target = _design(target_rows, feature_set)
    y = np.asarray([float(row["target"]) for row in training_rows], dtype=float)
    weights = _sample_weights(training_rows)
    algorithm = str(candidate["algorithm"])
    if algorithm == "ridge":
        mean = np.average(training, axis=0, weights=weights)
        variance = np.average((training - mean) ** 2, axis=0, weights=weights)
        scale = np.sqrt(np.maximum(variance, 1e-12))
        x_train = (training - mean) / scale
        x_target = (target - mean) / scale
        from sklearn.linear_model import Ridge

        estimator = Ridge(alpha=float(candidate["alpha"]))
        estimator.fit(x_train, y, sample_weight=weights)
        prediction = estimator.predict(x_target)
        parameters = {
            "candidate": dict(candidate),
            "feature_names": _feature_names(feature_set),
            "feature_mean": mean.astype(float).tolist(),
            "feature_scale": scale.astype(float).tolist(),
            "coefficients": np.asarray(estimator.coef_, dtype=float).tolist(),
            "intercept": float(estimator.intercept_),
        }
    elif algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            max_features=0.65,
            random_state=20_260_901,
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
            random_state=20_260_901,
        )
        estimator.fit(training, y, sample_weight=weights)
        prediction = estimator.predict(target)
        parameters = {"candidate": dict(candidate), "portable": False}
    else:
        raise KeyError(algorithm)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"non-finite universal intensity prediction: {candidate['name']}")
    return prediction, parameters


def _portable_predict(parameters: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    candidate = parameters["candidate"]
    design = _design(rows, str(candidate["feature_set"]))
    mean = np.asarray(parameters["feature_mean"], dtype=float)
    scale = np.asarray(parameters["feature_scale"], dtype=float)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    prediction = ((design - mean) / scale) @ coefficients + float(parameters["intercept"])
    return np.clip(prediction, 0.0, 1.0)


def _balanced_folds(values: Sequence[str], *, folds: int, salt: str) -> np.ndarray:
    unique = sorted(set(values), key=lambda value: hashlib.sha256(f"{salt}|{value}".encode()).digest())
    assignment = {value: index % folds for index, value in enumerate(unique)}
    return np.asarray([assignment[value] for value in values], dtype=int)


def _metrics(prediction: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)
    target = np.asarray([float(row["target"]) for row in rows], dtype=float)
    weights = _sample_weights(rows)
    error = predicted - target
    molecule_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction_value in zip(rows, predicted, strict=True):
        molecule_values[str(row["canonical_smiles"])].append(
            (float(prediction_value), float(row["target"]))
        )
    molecule_prediction = [float(np.mean([value[0] for value in values])) for values in molecule_values.values()]
    molecule_target = [float(np.mean([value[1] for value in values])) for values in molecule_values.values()]
    return {
        "mae": float(np.average(np.abs(error), weights=weights)),
        "rmse": float(np.sqrt(np.average(error**2, weights=weights))),
        "spearman": shared.spearman(predicted, target),
        "molecule_mean_spearman": shared.spearman(molecule_prediction, molecule_target),
        "rows": len(rows),
        "molecules": len(molecule_values),
    }


def _repeated_molecule_cv(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    pooled = {
        str(candidate["name"]): np.zeros((CV_REPEATS, len(rows)), dtype=float)
        for candidate in candidates
    }
    molecules = [str(row["canonical_smiles"]) for row in rows]
    for repeat in range(CV_REPEATS):
        assignments = _balanced_folds(
            molecules,
            folds=CV_FOLDS,
            salt=f"{FOLD_SALT}|molecule-cv|{repeat}",
        )
        for fold in range(CV_FOLDS):
            train_indices = np.flatnonzero(assignments != fold)
            test_indices = np.flatnonzero(assignments == fold)
            training = [rows[index] for index in train_indices]
            testing = [rows[index] for index in test_indices]
            if {row["canonical_smiles"] for row in training} & {
                row["canonical_smiles"] for row in testing
            }:
                raise RuntimeError("universal intensity molecule CV leakage")
            for candidate in candidates:
                prediction, _ = _fit_predict(candidate, training, testing)
                pooled[str(candidate["name"])][repeat, test_indices] = prediction
    averaged = {name: values.mean(axis=0) for name, values in pooled.items()}
    return averaged, {name: _metrics(prediction, rows) for name, prediction in averaged.items()}


def _source_transfer(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in TARGET_SOURCE_NAMES:
        testing = [row for row in rows if row["source"] == source]
        test_molecules = {str(row["canonical_smiles"]) for row in testing}
        training = [
            row
            for row in rows
            if row["source"] != source and str(row["canonical_smiles"]) not in test_molecules
        ]
        if len(training) < 100 or len(testing) < 30:
            raise RuntimeError(f"source-transfer partition too small: {source}")
        result[source] = {
            "training_rows": len(training),
            "testing_rows": len(testing),
            "testing_molecules": len(test_molecules),
            "exact_molecule_leakage_count": 0,
            "candidates": {},
        }
        for candidate in candidates:
            prediction, _ = _fit_predict(candidate, training, testing)
            result[source]["candidates"][str(candidate["name"])] = _metrics(
                prediction, testing
            )
    return result


def _select_candidate(
    cv_metrics: Mapping[str, Mapping[str, float]],
    transfer: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    by_name = {str(row["name"]): row for row in candidates}
    portable = [name for name, row in by_name.items() if bool(row["portable"])]
    return min(
        portable,
        key=lambda name: (
            float(cv_metrics[name]["mae"])
            + 0.35
            * float(
                np.mean(
                    [
                        transfer[source]["candidates"][name]["mae"]
                        for source in TARGET_SOURCE_NAMES
                    ]
                )
            ),
            max(
                float(transfer[source]["candidates"][name]["mae"])
                for source in TARGET_SOURCE_NAMES
            ),
            -float(cv_metrics[name]["molecule_mean_spearman"]),
            name,
        ),
    )


def _ma_component_baselines(
    args: argparse.Namespace, target_rows: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    document = json.loads(args.ma_predictions.read_text(encoding="utf-8"))
    by_cas: dict[str, list[float]] = defaultdict(list)
    for pair in document["predictions"]:
        track = pair["end_to_end"]["primary"]
        by_cas[str(pair["component_a"]["cas"])].append(
            float(track["component_a_intensity"]) / 10.0
        )
        by_cas[str(pair["component_b"]["cas"])].append(
            float(track["component_b_intensity"]) / 10.0
        )
    humanpom = []
    ravia = []
    adapter = FrozenConcentrationResponse()
    for row in target_rows:
        values = by_cas[str(row["cas"])]
        if not values or max(values) - min(values) > 1e-8:
            raise RuntimeError(f"inconsistent frozen HumanPOM prediction for {row['cas']}")
        humanpom.append(float(np.mean(values)))
        ravia.append(adapter.intensity(float(row["concentration_fraction"]))[0] / 100.0)
    return {
        "humanpom": np.asarray(humanpom, dtype=float),
        "ravia_global": np.asarray(ravia, dtype=float),
        "constant_training_mean": np.full(len(target_rows), np.nan, dtype=float),
    }


def _ma_mixture_predictions(
    target_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    component_predictions: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    target_document = json.loads(args.ma_predictions.read_text(encoding="utf-8"))
    by_cas_index = {str(row["cas"]): index for index, row in enumerate(target_rows)}
    pair_by_id = {str(row["pair_id"]): row for row in target_document["predictions"]}
    scale = float(target_document["model"]["fechner_scale_target_units_per_natural_log"])
    target = np.asarray([float(row["target_iab"]) for row in pair_rows], dtype=float)
    result: dict[str, dict[str, float]] = {}
    for model_name, values in component_predictions.items():
        strongest = []
        fechner = []
        for row in pair_rows:
            pair = pair_by_id[str(row["unit_id"])]
            first = float(values[by_cas_index[str(pair["component_a"]["cas"])]] * 10.0)
            second = float(values[by_cas_index[str(pair["component_b"]["cas"])]] * 10.0)
            strongest.append(max(first, second))
            fechner.append(ma._fechner_pool(first, second, scale))
        for operator, prediction in (("strongest", strongest), ("fechner", fechner)):
            result[f"{model_name}::{operator}"] = ma._metrics(prediction, target)
    return result


def _bootstrap(
    primary: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    *,
    unit: str,
) -> dict[str, Any]:
    generator = np.random.default_rng(BOOTSTRAP_SEED + (1 if unit == "mixture" else 0))
    mae_gain = []
    spearman_gain = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = generator.integers(0, len(target), len(target))
        observed = target[selected]
        model = primary[selected]
        reference = baseline[selected]
        mae_gain.append(
            float(np.mean(np.abs(reference - observed)) - np.mean(np.abs(model - observed)))
        )
        spearman_gain.append(
            shared.spearman(model, observed) - shared.spearman(reference, observed)
        )
    return {
        "unit": unit,
        "seed": BOOTSTRAP_SEED + (1 if unit == "mixture" else 0),
        "draws": BOOTSTRAP_DRAWS,
        "baseline_minus_primary_mae_95_interval": [
            float(value) for value in np.quantile(mae_gain, [0.025, 0.975])
        ],
        "primary_minus_baseline_spearman_95_interval": [
            float(value) for value in np.quantile(spearman_gain, [0.025, 0.975])
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    selected = report["selection"]["selected_candidate"]
    ma_component = report["ma_external_evaluation"]["monomolecular"]
    ma_mixture = report["ma_external_evaluation"]["binary_mixture"]
    return "\n".join(
        [
            "# Universal monomolecular intensity v1",
            "",
            f"- Selected source-only model: **{selected}**",
            f"- Ma component MAE: **{ma_component['universal']['mae']:.4f}**",
            f"- Ma HumanPOM MAE: **{ma_component['humanpom']['mae']:.4f}**",
            f"- Ma universal+max mixture MAE: **{ma_mixture['universal::strongest']['mae']:.4f} / 10**",
            f"- Ma HumanPOM+max mixture MAE: **{ma_mixture['humanpom::strongest']['mae']:.4f} / 10**",
            "- External integration gate: **"
            + ("PASS" if report["retrospective_external_gate"]["passed"] else "FAIL")
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
        raise RuntimeError("refusing to overwrite universal intensity outputs")
    keller_rows, keller_audit = _load_keller(args)
    ravia_rows, ravia_audit = _load_ravia(args.ravia_root.resolve(strict=True))
    bierling_rows, bierling_audit = _load_bierling(args)
    ma_targets, ma_pairs, ma_audit = _load_ma_targets(args)
    ma_smiles = {str(row["canonical_smiles"]) for row in ma_targets}
    training_raw = [*keller_rows, *ravia_rows, *bierling_rows]
    exact_overlap = [row for row in training_raw if row["canonical_smiles"] in ma_smiles]
    training_raw = [row for row in training_raw if row["canonical_smiles"] not in ma_smiles]
    all_smiles = sorted(
        {str(row["canonical_smiles"]) for row in [*training_raw, *ma_targets]}
    )
    raw_cache = build_raw_descriptor_cache(all_smiles)
    training = _prepare_rows(training_raw, raw_cache)
    targets = _prepare_rows(ma_targets, raw_cache)
    cv_prediction, cv_metrics = _repeated_molecule_cv(training, CANDIDATES)
    transfer = _source_transfer(training, CANDIDATES)
    selected = _select_candidate(cv_metrics, transfer, CANDIDATES)
    selected_candidate = next(row for row in CANDIDATES if row["name"] == selected)
    target_prediction, parameters = _fit_predict(selected_candidate, training, targets)
    fitted, final_parameters = _fit_predict(selected_candidate, training, training)
    portable = _portable_predict(final_parameters, training)
    parity = float(np.max(np.abs(fitted - portable)))
    if parity > 1e-10:
        raise RuntimeError("universal intensity portable parameter parity failed")
    final_parameters["portable_parity_maximum_absolute_error"] = parity
    target_values = np.asarray([float(row["target"]) for row in targets], dtype=float)
    baselines = _ma_component_baselines(args, targets)
    baselines["constant_training_mean"] = np.full(
        len(targets),
        float(np.average([row["target"] for row in training], weights=_sample_weights(training))),
    )
    component_metrics = {
        "universal": _metrics(target_prediction, targets),
        **{name: _metrics(values, targets) for name, values in baselines.items()},
    }
    component_predictions = {"universal": target_prediction, **baselines}
    mixture_metrics = _ma_mixture_predictions(
        targets, ma_pairs, component_predictions, args
    )
    component_bootstrap = _bootstrap(
        target_prediction,
        baselines["humanpom"],
        target_values,
        unit="molecule",
    )

    # Reconstruct the pair-level vectors for the selected comparison.
    target_document = json.loads(args.ma_predictions.read_text(encoding="utf-8"))
    pair_lookup = {str(row["pair_id"]): row for row in target_document["predictions"]}
    index_by_cas = {str(row["cas"]): index for index, row in enumerate(targets)}
    primary_pair = []
    baseline_pair = []
    pair_target = np.asarray([float(row["target_iab"]) for row in ma_pairs], dtype=float)
    for row in ma_pairs:
        pair = pair_lookup[str(row["unit_id"])]
        first_index = index_by_cas[str(pair["component_a"]["cas"])]
        second_index = index_by_cas[str(pair["component_b"]["cas"])]
        primary_pair.append(
            max(target_prediction[first_index], target_prediction[second_index]) * 10.0
        )
        baseline_pair.append(
            max(baselines["humanpom"][first_index], baselines["humanpom"][second_index])
            * 10.0
        )
    mixture_bootstrap = _bootstrap(
        np.asarray(primary_pair),
        np.asarray(baseline_pair),
        pair_target,
        unit="mixture",
    )
    checks = {
        "ma_exact_molecule_training_leakage_zero": not bool(
            {row["canonical_smiles"] for row in training} & ma_smiles
        ),
        "source_cv_selected_model_beats_concentration_mae": cv_metrics[selected]["mae"]
        < cv_metrics["ridge_concentration_100"]["mae"],
        "source_cv_selected_model_beats_concentration_molecule_rank": cv_metrics[selected][
            "molecule_mean_spearman"
        ]
        > cv_metrics["ridge_concentration_100"]["molecule_mean_spearman"],
        "ma_component_mae_beats_humanpom": component_metrics["universal"]["mae"]
        < component_metrics["humanpom"]["mae"],
        "ma_component_spearman_beats_humanpom": component_metrics["universal"]["spearman"]
        > component_metrics["humanpom"]["spearman"],
        "ma_component_mae_bootstrap_lower_above_zero": component_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "ma_mixture_mae_beats_humanpom": mixture_metrics["universal::strongest"]["mae"]
        < mixture_metrics["humanpom::strongest"]["mae"],
        "ma_mixture_spearman_beats_humanpom": mixture_metrics[
            "universal::strongest"
        ]["spearman"]
        > mixture_metrics["humanpom::strongest"]["spearman"],
        "ma_mixture_mae_bootstrap_lower_above_zero": mixture_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "portable_numeric_parity_at_most_1e_10": parity <= 1e-10,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "universal_intensity_v1_retrospective_external_gate_passed"
            if all(checks.values())
            else "universal_intensity_v1_retrospective_external_gate_failed"
        ),
        "development_timing": "designed_after_ma_2021_outcome_was_known",
        "source_contract": {
            "keller": keller_audit,
            "ravia": ravia_audit,
            "bierling": bierling_audit,
            "ma_evaluation": ma_audit,
            "ma_exact_overlap_rows_excluded": len(exact_overlap),
            "ma_exact_overlap_molecules_excluded": len(
                {row["canonical_smiles"] for row in exact_overlap}
            ),
            "training_rows_after_target_exclusion": len(training),
            "training_molecules_after_target_exclusion": len(
                {row["canonical_smiles"] for row in training}
            ),
        },
        "feature_contract": {
            "identity_features": [],
            "physical_rdkit_descriptors": list(PHYSICAL_DESCRIPTOR_NAMES),
            "rdkit_descriptor_count": len(descriptor_contract()),
            "concentration_features": [
                "log10_fraction",
                "log10_fraction_squared",
                "log10_molarity_proxy",
                "log10_molarity_proxy_squared",
            ],
            "ma_fraction_proxy": "concentration_mg_ml / 1000",
            "measured_vapor_pressure_used": False,
            "measured_boiling_point_used": False,
            "measured_odor_threshold_used": False,
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
            "rule": "portable minimum source-CV MAE plus 0.35 mean source-transfer MAE",
        },
        "final_model": {
            "parameters": final_parameters,
            "runtime_primary_score_weight": 0.0,
            "runtime_status": "retrospective_validation_gated_diagnostic_only",
        },
        "ma_external_evaluation": {
            "monomolecular": component_metrics,
            "binary_mixture": mixture_metrics,
            "component_bootstrap": component_bootstrap,
            "mixture_bootstrap": mixture_bootstrap,
            "all_ma_molecules_exact_target_excluded": True,
        },
        "retrospective_external_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "human_olfactory_90_percent_certified": False,
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Target-excluded post-outcome research evaluation. Ma labels did not enter "
            "fit or automated selection, but the model family was designed after Ma was "
            "known. No production weight or 90% olfactory claim is authorized."
        ),
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
                "source_cv": report["selection"]["selected_cv_metrics"],
                "ma_component": report["ma_external_evaluation"]["monomolecular"],
                "ma_mixture": report["ma_external_evaluation"]["binary_mixture"],
                "gate": report["retrospective_external_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
