#!/usr/bin/env python
"""Train/validate molecular coverage and stock-conditional profiles on CPU.

Only provided SINGLE and MIXTURE TRAINING outcomes may be read. The previously
scored leaderboard and official final test are prediction-only: their outcomes
are denied by a Python file audit hook. V1 sealed code/artifacts are unchanged.
All reported scores are nested-CV development results, not a fresh blind test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_human_mixture_profiles as v1  # noqa: E402
from fragrance_ai.research.conditional_profiles import (  # noqa: E402
    ALPHAS, CARRIERS, FP_SIZE, PHYSICAL, choose_conditional, feature_matrix,
    molecule_features, predict_conditional,
)
from fragrance_ai.research.perception_validation import (  # noqa: E402
    fit_ridge, group_folds, paired_profile_comparison, predict_ridge,
    profile_errors, summarize_profiles,
)

SCHEMA = "conditional-profile-training-only/v2"
INPUTS = {name: binding for name, binding in v1.SOURCE_FILES.items() if name != "target_outcomes"}
INPUTS.update({
    "task1_training": ("data/Task1/TASK1_training_All.csv", "eec99654bb7702e4d13520816e43fabd5797fdab"),
    "final_test_ids": ("data/Task2/TASK2_Test_set_Submission_form.csv", "5e2c14b966b38706201c7caa76cefe1929358513"),
})
NATIVE_FEATURE_ARTIFACT_SHA = "3487f3d8839ee4a017c4e4ce771974899297d000db2f45f5127ad5b761a4b0f1"
FROZEN_V1_CODE = {
    "scripts/benchmark_human_mixture_profiles.py": "4952feeaec0e8744f3870f7e44f06f31374f4626d47239c74954cb3d8c71c363",
    "fragrance_ai/research/perception_validation.py": "81cccffe7c1e0146b97411230b46ac99f6bd7fe7925474e6db2be983973055a3",
}
MODEL_SPECS = {"native": ("native", False), "molecular": ("molecular", False), "anchored": ("molecular", True)}


def install_training_only_guard(source: Path) -> list[str]:
    allowed = {str((source / relative).resolve()).casefold() for relative, _ in INPUTS.values()}
    root = source.resolve()
    denied_attempts = []

    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        normalized = str(path).casefold()
        # Also deny copies elsewhere. Submission templates are metadata only.
        name = path.name.casefold()
        parts = {part.casefold() for part in path.parts}
        forbidden = ("actualvalue" in name or "human_mixture_profiles_v1.json" == name
                     or ("human_mixture_profiles_v1" in parts and name in ("report.json", "predictions.json")))
        if (path.is_relative_to(root) and normalized not in allowed) or forbidden:
            denied_attempts.append(str(path))
            raise PermissionError("training-only run: forbidden outcome/source access")
    sys.addaudithook(hook)
    return denied_attempts


def verify_inputs(source: Path) -> dict:
    bindings = {}
    for name, (relative, expected) in INPUTS.items():
        raw = (source / relative).read_bytes()
        candidates = (raw, raw.replace(b"\r\n", b"\n"))
        if expected not in {hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest() for value in candidates}:
            raise ValueError(f"upstream training/metadata file drift: {relative}")
        bindings[name] = {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "blob": expected}
    for relative, expected in FROZEN_V1_CODE.items():
        if v1.sha(ROOT / relative) != expected:
            raise ValueError("sealed V1 implementation was changed")
    return bindings


def extended_observations(source: Path) -> tuple[list[tuple], np.ndarray, dict, dict]:
    _, stimuli, _, _ = v1.load_design(source)
    task1 = {row["stimulus"]: v1.condition(row["molecule"], row["dilution"], row["solvent"])
             for row in v1.rows(source / INPUTS["task1_stimuli"][0])}
    records = {}
    for row in v1.rows(source / INPUTS["single"][0]):
        stimulus = row["stimulus"]
        key = task1[stimulus] if stimulus in task1 else stimuli[stimulus][0]
        records[stimulus] = (key, v1.vector(row))
    duplicates = 0
    new_rows = v1.unique_rows(v1.rows(source / INPUTS["task1_training"][0]), "stimulus")
    for stimulus, row in new_rows.items():
        key, profile = task1[stimulus], v1.vector(row)
        if stimulus in records:
            if records[stimulus][0] != key or not np.array_equal(records[stimulus][1], profile):
                raise ValueError(f"conflicting duplicate stimulus: {stimulus}")
            duplicates += 1
        else:
            records[stimulus] = (key, profile)
    grouped = defaultdict(list)
    for key, profile in records.values():
        grouped[key].append(profile)
    keys = sorted(grouped)
    y = np.asarray([np.mean(grouped[key], axis=0) for key in keys])
    return keys, y, stimuli, {"observations_after_stimulus_deduplication": len(records),
                             "cross_file_duplicate_stimuli_removed": duplicates, "stock_conditions": len(keys),
                             "molecule_or_control_ids": len({key[0] for key in keys}),
                             "same_condition_repeat_groups": sum(len(value) > 1 for value in grouped.values()),
                             "participant_repeats_available": False}


def molecular_bank(source: Path, native_path: Path) -> dict:
    if v1.sha(native_path) != NATIVE_FEATURE_ARTIFACT_SHA:
        raise ValueError("V1 native feature artifact drift")
    # Only immutable ingredient profiles are reused, NOT the mixture regressor.
    native = json.loads(native_path.read_text(encoding="utf-8"))["cid_profiles"]
    bank = {}
    for row in v1.rows(source / INPUTS["molecules"][0]):
        cid = str(int(row["molecule"]))
        if cid in bank:
            raise ValueError("duplicate molecular identity")
        if int(cid) < 0:
            continue
        bank[cid] = molecule_features(row["SMILES"], native.get(cid))
    return bank


def nested_component_cv(keys: list[tuple], y: np.ndarray, matrices: dict, track: str) -> dict:
    groups = [key[0] if track == "molecule_disjoint" else json.dumps(key) for key in keys]
    outer = group_folds(groups, 5)
    predictions, selections, bases = {}, {}, {}
    for name, (family, anchored) in MODEL_SPECS.items():
        prediction = np.full(y.shape, np.nan)
        selection, basis = [], Counter()
        for fold in range(5):
            training = outer != fold
            train_keys = [key for key, keep in zip(keys, training) if keep]
            test_keys = [key for key, keep in zip(keys, ~training) if keep]
            model, audit = choose_conditional(matrices[family][training], y[training], train_keys,
                                             [group for group, keep in zip(groups, training) if keep], anchored=anchored)
            prediction[~training], status = predict_conditional(model, matrices[family][~training], test_keys)
            basis.update(row["basis"] for row in status)
            selection.append({"outer_fold": fold, "held_out": int((~training).sum()), **audit})
        predictions[name], selections[name], bases[name] = prediction, selection, dict(basis)
    results = {name: summarize_profiles(prediction, y, groups) for name, prediction in predictions.items()}
    return {"scope": "nested training-development CV; not an independent external human validation", "track": track,
            "outer_group_folds": 5, "groups": len(set(groups)), "conditions": len(keys),
            "results": results, "selections": selections, "prediction_basis": bases,
            "molecular_vs_native_on_common": paired_profile_comparison(predictions["native"], predictions["molecular"], y, groups),
            "anchored_vs_molecular_on_common": paired_profile_comparison(predictions["molecular"], predictions["anchored"], y, groups)}


def blend_arrays(ids: list[str], stimuli: dict, stock_predictions: dict, stock_status: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    means, features, metadata = [], [], []
    width = 3 * len(v1.ENDPOINTS) + 4
    for stimulus in ids:
        keys = stimuli[stimulus]
        profiles = np.asarray([stock_predictions[key] for key in keys])
        status = [stock_status[key] for key in keys]
        available = np.isfinite(profiles).all()
        means.append(profiles.mean(axis=0) if available else np.full(len(v1.ENDPOINTS), np.nan))
        features.append(v1.blend_features(profiles, keys) if available else np.full(width, np.nan))
        metadata.append({"stimulus": stimulus, "component_count": len(keys), "formula_key": v1.formula_key(keys),
                         "component_basis": dict(Counter(row["basis"] for row in status)), "predicted": bool(available)})
    return np.asarray(features), np.asarray(means), metadata


def fit_blend(x: np.ndarray, y: np.ndarray, mean: np.ndarray, groups: list[str]) -> tuple[dict, dict]:
    folds, present = group_folds(groups, 3), np.isfinite(x).all(axis=1)
    choices = [{"alpha": None, "weight": 0.0,
                "loss": float(np.mean(np.nan_to_num(profile_errors(mean, y)["cosine_distance"], nan=1.0)))}]
    for alpha in ALPHAS:
        oof = np.full(y.shape, np.nan)
        for fold in range(3):
            training, test = (folds != fold) & present, (folds == fold) & present
            regressor = fit_ridge(x[training], y[training], alpha)
            oof[test] = predict_ridge(regressor, x[test])
        for weight in (0.5, 1.0):
            mixed = (1 - weight) * mean + weight * oof
            choices.append({"alpha": alpha, "weight": weight,
                            "loss": float(np.mean(np.nan_to_num(profile_errors(mixed, y)["cosine_distance"], nan=1.0)))})
    selected = min(choices, key=lambda row: (row["loss"], row["weight"], row["alpha"] or 0))
    regressor = fit_ridge(x[present], y[present], selected["alpha"]) if selected["alpha"] is not None else None
    return {"regressor": regressor, "weight": selected["weight"]}, {"selection": selected, "inner_choices": choices}


def predict_blend(model: dict, x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    if model["regressor"] is None:
        return mean.copy()
    available = np.isfinite(x).all(axis=1)
    prediction = np.full(mean.shape, np.nan)
    prediction[available] = (1 - model["weight"]) * mean[available] + model["weight"] * predict_ridge(model["regressor"], x[available])
    return prediction


def run(args: argparse.Namespace) -> dict:
    from rdkit import rdBase

    output, source = args.output.resolve(), args.source.resolve()
    if output.exists():
        raise FileExistsError("new development run directory required")
    denied = install_training_only_guard(source)
    inputs = verify_inputs(source)
    keys, y, stimuli, data_audit = extended_observations(source)
    bank = molecular_bank(source, args.native_features)
    matrices = {family: feature_matrix(keys, bank, family) for family in ("native", "molecular")}
    target_ids = {name: list(v1.unique_rows(v1.rows(source / INPUTS[name][0]), "stimulus")) for name in ("target_ids", "final_test_ids")}
    protected_compositions = {v1.formula_key(stimuli[stimulus]) for ids in target_ids.values() for stimulus in ids}
    training = v1.rows(source / INPUTS["training"][0])
    excluded = [row["stimulus"] for row in training if v1.formula_key(stimuli[row["stimulus"]]) in protected_compositions]
    training = [row for row in training if row["stimulus"] not in excluded]
    mix_ids = [row["stimulus"] for row in training]
    mix_y = np.asarray([v1.vector(row) for row in training])
    mix_groups = [v1.formula_key(stimuli[stimulus]) for stimulus in mix_ids]
    outer_mix = group_folds(mix_groups, 5)
    code = {relative: v1.sha(ROOT / relative) for relative in (
        "scripts/train_conditional_profiles_v2.py", "fragrance_ai/research/conditional_profiles.py", *FROZEN_V1_CODE)}
    protocol = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "rdkit": rdBase.rdkitVersion,
                                "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
                "code": code, "endpoints": list(v1.ENDPOINTS), "alphas": list(ALPHAS),
                "molecular_features": {"morgan_bits": FP_SIZE, "radius": 2, "include_chirality": True, "physical": list(PHYSICAL), "carriers": list(CARRIERS)},
                "outer_folds": 5, "inner_folds": 3, "blend_weights": [0.0, 0.5, 1.0],
                "source_overlap_policy": "remove train compositions matching either leaderboard or final-test metadata",
                "target_outcomes_policy": "forbidden: do not read, hash, evaluate or select on leaderboard/final outcomes",
                "scope": "nested training-development validation; experimental single-stock prediction and mixture calibration",
                "actual_human_accuracy_90_authorized": False, "runtime_promotion_allowed": False}
    output.mkdir(parents=True)
    v1.write_new(output / "protocol.json", protocol)
    print("Protocol fixed; component molecule-disjoint nested CV", flush=True)
    component = nested_component_cv(keys, y, matrices, "molecule_disjoint")
    print("Component stock-condition nested CV", flush=True)
    conditional = nested_component_cv(keys, y, matrices, "stock_condition_disjoint")
    all_stock_keys = sorted({key for values in stimuli.values() for key in values})
    stock_features = {family: feature_matrix(all_stock_keys, bank, family) for family in matrices}
    component_models, blend_models, mix_results, target_predictions = {}, {}, {}, {}
    mix_pred, selections, coverage = {}, {}, {}
    for name, (family, anchored) in MODEL_SPECS.items():
        print(f"Mixture nested CV: {name}", flush=True)
        model, selection = choose_conditional(matrices[family], y, keys, [key[0] for key in keys], anchored=anchored)
        stock_pred, status = predict_conditional(model, stock_features[family], all_stock_keys)
        pred_map, status_map = dict(zip(all_stock_keys, stock_pred)), dict(zip(all_stock_keys, status))
        x, mean, meta = blend_arrays(mix_ids, stimuli, pred_map, status_map)
        prediction, outer_selections = np.full(mix_y.shape, np.nan), []
        for fold in range(5):
            train = outer_mix != fold
            fitted, chosen = fit_blend(x[train], mix_y[train], mean[train], [group for group, keep in zip(mix_groups, train) if keep])
            prediction[~train] = predict_blend(fitted, x[~train], mean[~train])
            outer_selections.append({"outer_fold": fold, **chosen})
        fitted, chosen = fit_blend(x, mix_y, mean, mix_groups)
        component_models[name], blend_models[name] = model, fitted
        mix_results[name], mix_pred[name] = summarize_profiles(prediction, mix_y, mix_groups), prediction
        selections[name] = {"component_training_selection": selection, "outer_blend_selections": outer_selections, "final_training_only_blend_selection": chosen}
        coverage[name] = {"training_profiles": len(meta), "training_predictable": sum(row["predicted"] for row in meta)}
        for dataset, ids in target_ids.items():
            tx, tm, metadata = blend_arrays(ids, stimuli, pred_map, status_map)
            predictions = predict_blend(fitted, tx, tm)
            target_predictions.setdefault(dataset, {})[name] = [
                {**row, "profile": value.tolist() if np.isfinite(value).all() else None}
                for row, value in zip(metadata, predictions)]
    private_model = {"schema": SCHEMA, "molecular_bank": bank, "component_models": component_models,
                     "blend_models": blend_models, "profile_dimensions": list(v1.ENDPOINTS),
                     "runtime_promotion_allowed": False, "data_redistribution_authorized": False}
    # Verify JSON replay through the actual prediction functions, not serialization alone.
    replay = json.loads(json.dumps(private_model, allow_nan=False))
    for name, (family, _) in MODEL_SPECS.items():
        a, a_status = predict_conditional(component_models[name], stock_features[family], all_stock_keys)
        b, b_status = predict_conditional(replay["component_models"][name], stock_features[family], all_stock_keys)
        if not np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True) or a_status != b_status:
            raise ValueError("portable component replay mismatch")
        ax, am, _ = blend_arrays(mix_ids, stimuli, dict(zip(all_stock_keys, a)), dict(zip(all_stock_keys, a_status)))
        bx, bm, _ = blend_arrays(mix_ids, stimuli, dict(zip(all_stock_keys, b)), dict(zip(all_stock_keys, b_status)))
        if not np.allclose(predict_blend(blend_models[name], ax, am), predict_blend(replay["blend_models"][name], bx, bm),
                           atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError("portable end-to-end mixture replay mismatch")
    v1.write_new(output / "models.json", private_model)
    v1.write_new(output / "unscored_target_predictions.json", {"scope": "predictions only; no target outcomes accessed", "datasets": target_predictions})
    report = {"schema": SCHEMA, "status": "completed_training_only_nested_cv", "data_audit": data_audit,
              "training_mixtures": len(training), "training_composition_groups": len(set(mix_groups)),
              "protected_composition_train_rows_removed": excluded,
              "component_molecule_disjoint": component, "component_stock_condition_disjoint": conditional,
              "mixture_nested_cv": {"results": mix_results, "selections": selections, "coverage": coverage,
                                    "anchored_vs_native": paired_profile_comparison(mix_pred["native"], mix_pred["anchored"], mix_y, mix_groups),
                                    "anchored_vs_molecular": paired_profile_comparison(mix_pred["molecular"], mix_pred["anchored"], mix_y, mix_groups)},
              "unscored_target_coverage": {dataset: {name: {"total": len(values), "predicted": sum(row["profile"] is not None for row in values)} for name, values in models.items()} for dataset, models in target_predictions.items()},
              "outcome_read_attempts": denied, "target_outcomes_scored": False,
              "portable_component_replay_max_tolerance": 1e-12,
              "portable_end_to_end_mixture_replay_max_tolerance": 1e-12,
              "actual_human_accuracy_90_authorized": False, "runtime_promotion_allowed": False,
              "protocol_sha256": v1.sha(output / "protocol.json"), "models_sha256": v1.sha(output / "models.json")}
    v1.write_new(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, default=ROOT / ".benchmarks/human_mixture_profiles_v1/models.json")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "data_audit": report["data_audit"],
                      "unscored_target_coverage": report["unscored_target_coverage"],
                      "mixture_comparison": report["mixture_nested_cv"]["anchored_vs_native"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
