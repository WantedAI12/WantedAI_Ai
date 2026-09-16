"""Measured-stock anchored, concentration-aware molecular profile prediction.

Research-only CPU model. Molecular fallback is a prediction, not a registry
identity match. Interpolation and extrapolation never cross molecule/solvent
boundaries. A measured solvent control can be replayed only at its exact stock
condition; it is never assigned an invented molecular graph.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np

from .perception_validation import fit_ridge, group_folds, predict_ridge, profile_errors
from .kernel_profiles import predict_component_regressor

SCHEMA = "conditional-molecular-profile/v2"
FP_SIZE = 1024
CARRIERS = ("nt", "pg", "dep", "paraffin oil", "mineral oil", "90% ethanol", "99% ethanol", "other")
PHYSICAL = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
            "RingCount", "NumAromaticRings", "NumAliphaticRings", "FractionCSP3", "HeavyAtomCount",
            "BertzCT", "LabuteASA", "HallKierAlpha", "NumSaturatedRings", "NOCount")
ALPHAS = (10.0, 100.0, 1000.0)


@lru_cache(maxsize=8192)
def _molecular_descriptor_tuple(smiles: str):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid molecular structure")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE, includeChirality=True)
    fingerprint = generator.GetFingerprint(mol)
    bits = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    physical = [float(getattr(Descriptors, name)(mol)) for name in PHYSICAL]
    if not np.isfinite(physical).all():
        raise ValueError("nonfinite molecular descriptors")
    return Chem.MolToSmiles(mol, isomericSmiles=True), tuple(np.flatnonzero(bits).tolist()), tuple(physical)


def molecule_features(smiles: str, native: dict | None = None) -> dict:
    """Reuse immutable chemical descriptors; retain caller-owned native data.

    Descriptor calculation is independent of dose, language and observed
    native profiles. Returning fresh lists prevents cache mutation by callers.
    """
    canonical,bits,physical = _molecular_descriptor_tuple(smiles)
    if native is not None and (len(native['profile']) != 19 or not np.isfinite(native['profile']).all()):
        raise ValueError('native profile feature contract changed')
    return {'canonical_smiles':canonical, 'fingerprint_bits':list(bits), 'physical':list(physical), 'native':native}


def features_for(key: tuple[str, str, str], molecules: dict, family: str = "molecular") -> np.ndarray | None:
    if family not in ("native", "molecular"):
        raise ValueError("unknown feature family")
    item = molecules.get(key[0])
    if item is None or int(key[0]) < 0:
        return None
    dilution = float(key[1])
    if not math.isfinite(dilution) or not 0 < dilution <= 1:
        raise ValueError("invalid stock dilution")
    log_dose = math.log10(dilution)
    carrier = key[2] if key[2] in CARRIERS else "other"
    onehot = np.asarray([float(carrier == name) for name in CARRIERS])
    native = item["native"]
    if family == "native":
        if native is None:
            return None
        profile = np.asarray(native["profile"])
        return np.r_[profile, profile * log_dose, log_dose, math.log1p(native["odor_impact"]), onehot]
    physical = np.asarray(item["physical"])
    fp = np.zeros(FP_SIZE)
    fp[item["fingerprint_bits"]] = 1.0
    profile = np.asarray(native["profile"]) if native is not None else np.zeros(19)
    return np.r_[fp, physical, physical * log_dose, profile, profile * log_dose,
                 float(native is not None), log_dose, log_dose * log_dose, onehot]


def feature_matrix(keys: list[tuple], molecules: dict, family: str) -> np.ndarray:
    values = [features_for(key, molecules, family) for key in keys]
    width = next((len(value) for value in values if value is not None), None)
    if width is None:
        raise ValueError("no molecules supported by this feature family")
    return np.asarray([value if value is not None else np.full(width, np.nan) for value in values])


def fit_conditional(x: np.ndarray, y: np.ndarray, keys: list[tuple], alpha: float, *, anchored: bool) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(keys) != len(y) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("invalid conditional training arrays")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate stock conditions must be aggregated before fitting")
    available = np.isfinite(x).all(axis=1)
    regressor = fit_ridge(x[available], y[available], alpha)
    fitted = np.full(y.shape, np.nan)
    fitted[available] = predict_ridge(regressor, x[available])
    anchors = [{"key": list(key), "profile": row.tolist(),
                "base_prediction": base.tolist() if np.isfinite(base).all() else None}
               for key, row, base in zip(keys, y, fitted)] if anchored else []
    return {"schema": SCHEMA, "regressor": regressor, "anchors": anchors,
            "feature_width": x.shape[1], "profile_width": y.shape[1],
            "anchored": bool(anchored), "extrapolation_decay_decades": 1.0,
            "actual_human_accuracy_90_authorized": False}


def predict_conditional(model: dict, x: np.ndarray, keys: list[tuple]) -> tuple[np.ndarray, list[dict]]:
    x = np.asarray(x, dtype=float)
    if model.get("schema") != SCHEMA or x.ndim != 2 or len(x) != len(keys) or x.shape[1] != model["feature_width"]:
        raise ValueError("conditional prediction contract mismatch")
    if model.get("actual_human_accuracy_90_authorized") is not False:
        raise ValueError("research model cannot authorize a human accuracy claim")
    available = np.isfinite(x).all(axis=1)
    prediction = np.full((len(x), model["profile_width"]), np.nan)
    prediction[available] = predict_component_regressor(model["regressor"], x[available])
    exact, families = {}, defaultdict(list)
    for anchor in model["anchors"]:
        key = tuple(anchor["key"])
        profile = np.asarray(anchor["profile"], dtype=float)
        if (key in exact or profile.shape != (model["profile_width"],)
                or not np.isfinite(profile).all() or np.any(profile < 0)):
            raise ValueError("invalid or duplicate measured anchor")
        exact[key] = anchor
        families[(key[0], key[2])].append(anchor)
    status = []
    for index, key in enumerate(keys):
        detail = {"basis": "molecular_prediction" if available[index] else "unsupported_structure",
                  "anchor_distance_decades": None, "measured_at_requested_stock": False}
        if key in exact:
            prediction[index] = exact[key]["profile"]
            detail.update(basis="measured_solvent_control" if int(key[0]) < 0 else "measured_stock",
                          anchor_distance_decades=0.0, measured_at_requested_stock=True)
        elif available[index] and families.get((key[0], key[2])):
            ordered = sorted(families[(key[0], key[2])], key=lambda row: float(row["key"][1]))
            logs = np.log10([float(row["key"][1]) for row in ordered])
            query = math.log10(float(key[1]))
            nearest = int(np.argmin(np.abs(logs - query)))
            distance = float(abs(logs[nearest] - query))
            detail["anchor_distance_decades"] = distance
            if len(ordered) >= 2 and logs[0] < query < logs[-1]:
                upper = int(np.searchsorted(logs, query))
                weight = (query - logs[upper - 1]) / (logs[upper] - logs[upper - 1])
                prediction[index] = (1 - weight) * np.asarray(ordered[upper - 1]["profile"]) + weight * np.asarray(ordered[upper]["profile"])
                detail["basis"] = "same_molecule_solvent_log_dose_interpolation"
            else:
                anchor = ordered[nearest]
                if anchor["base_prediction"] is not None:
                    residual = np.asarray(anchor["profile"]) - np.asarray(anchor["base_prediction"])
                    prediction[index] = np.maximum(0.0, prediction[index] + math.exp(-distance) * residual)
                    detail["basis"] = "same_molecule_solvent_extrapolation_proxy"
        status.append(detail)
    return prediction, status


def choose_conditional(x: np.ndarray, y: np.ndarray, keys: list[tuple], groups: list[str],
                       *, anchored: bool, folds: int = 3) -> tuple[dict, dict]:
    assigned = group_folds(groups, folds)
    scores = []
    for alpha in ALPHAS:
        prediction = np.full(y.shape, np.nan)
        for fold in range(folds):
            train = assigned != fold
            model = fit_conditional(x[train], y[train], [key for key, keep in zip(keys, train) if keep], alpha, anchored=anchored)
            prediction[~train], _ = predict_conditional(model, x[~train], [key for key, keep in zip(keys, ~train) if keep])
        errors = profile_errors(prediction, y)
        scores.append({"alpha": alpha, "selection_loss_with_abstention_penalty": float(np.mean(np.nan_to_num(errors["cosine_distance"], nan=1.0)))})
    chosen = min(scores, key=lambda row: (row["selection_loss_with_abstention_penalty"], row["alpha"]))
    return fit_conditional(x, y, keys, chosen["alpha"], anchored=anchored), {
        "inner_group_folds": folds, "candidates": scores, "selected_alpha": chosen["alpha"],
        "selection_scope": "inner_training_groups_only"}
