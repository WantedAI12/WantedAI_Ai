#!/usr/bin/env python
"""Frozen external RATA ablation of the exact V5 runtime ingredient profiles.

prepare never parses leaderboard outcomes. score verifies the complete seal,
then opens the public outcomes once. They were already cached in this project:
this is a retrospective, training-selected external diagnostic, not a new
prospective blind study or a generated-perfume validation. Data/model outputs
stay in a local research directory and are not shipped in the runtime wheel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragrance_ai.research.perception_validation import (  # noqa: E402
    choose_ridge, paired_profile_comparison, predict_ridge, summarize_profiles,
)

SCHEMA = "human-mixture-profile-ablation/v1"
UPSTREAM = "https://github.com/Satarifard/Olfactory-Mixtures-Prediction-2025"
COMMIT = "1aff2ea1764b8434e703a0eb8c27302ec2910f60"
SOURCE_FILES = {
    "components": ("data/Task2/TASK2_Component_definition.csv", "30456b8dc067451a692f84b66eae58e5263c34f9"),
    "stimuli": ("data/Task2/TASK2_Stimulus_definition.csv", "8e42664ff2772279cdc95e823e107a614d8d2cdd"),
    "task1_stimuli": ("data/Task1/TASK1_Stimulus_definition.csv", "85db7860957a3f1a311ce90abaf859ceac67164d"),
    "single": ("data/Task2/Task2_single_RATA_fixed_V1.csv", "c5229e4fbc1c9fcb33f00b3f7938eb014ea5c6ae"),
    "training": ("data/Task2/TASK2_Train_mixture_Dataset.csv", "ae620fb253feb09f933fb3338ac3659c949d5079"),
    "target_ids": ("data/Task2/TASK2_Leaderboard_set_Submission_form.csv", "4d826c0aa04c667883936999d9097544603f9fc2"),
    "target_outcomes": ("Task2/Data/TASK2_Leaderboard_ActualValue.csv", "3bd04f1264e6c36851d4e523728f30f5863ced56"),
    "molecules": ("data/Task2/CID.csv", "3146f181a7dbdb45520907db1f91db2b6eea4248"),
}
ENDPOINTS = tuple("Green,Cucumber,Herbal,Mint,Woody,Pine,Floral,Powdery,Fruity,Citrus,Tropical,Berry,Peach,Sweet,Caramellic,Vanilla,BrownSpice,Smoky,Burnt,Roasted,Grainy,Meaty,Nutty,Fatty,Coconut,Waxy,Dairy,Buttery,Cheesy,Sour,Fermented,Sulfurous,Garlic.Onion,Earthy,Mushroom,Musty,Ammonia,Fishy,Fecal,Rotten.Decay,Rubber,Phenolic,Animal,Medicinal,Cooling,Sharp,Chlorine,Alcoholic,Plastic,Ozone,Metallic".split(","))
# The organizer's release discussion documents missing Ozone/Metallic labels.
# Fix the 49-term primary scoring axis BEFORE outcomes; never pick axes by score.
EVALUATED_ENDPOINTS = tuple(name for name in ENDPOINTS if name not in ("Ozone", "Metallic"))
SCENT_DIMENSIONS = tuple("citrus,fresh,clean,green,aquatic,floral,rose,white_floral,fruity,spicy,aromatic,woody,amber,musky,gourmand,powdery,smoky,leathery,earthy".split(","))
CARRIERS = ("nt", "pg", "dep", "paraffin oil", "mineral oil", "90% ethanol", "99% ethanol", "other")
METHODS = ("training_mean", "measured_exact_mean", "measured_nearest_stock_proxy",
           "native_v5_component_mean", "native_v5_blend_calibrated")
PAIRED_CONTRASTS = {
    "component_profile_gap_same_mean_aggregator": ("measured_exact_mean", "native_v5_component_mean"),
    "nearest_stock_proxy_vs_native_same_aggregator": ("measured_nearest_stock_proxy", "native_v5_component_mean"),
    "blend_calibration_gain": ("native_v5_component_mean", "native_v5_blend_calibrated"),
    "candidate_vs_training_mean": ("training_mean", "native_v5_blend_calibrated"),
    "candidate_vs_exact_stock_baseline": ("measured_exact_mean", "native_v5_blend_calibrated"),
}
WHEEL_HASH = "02509fb1806b6247294595dfaa2042dc1cb75e1d08b51e230522ce1a6a093dd6"
CATALOG_HASH = "bf2f2cf333c75fcf0f3f9165c12c686823fcaa47b7b6e3e320e473cf8c816d4a"
REGISTRY_HASH = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path: Path, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(raw)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"duplicate/empty columns in {path.name}")
        result = list(reader)
    if any(None in row for row in result):
        raise ValueError(f"malformed CSV row in {path.name}")
    return result


def unique_rows(records: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for row in records:
        key = row[field].strip()
        if not key or key in result:
            raise ValueError(f"duplicate/empty {field}: {key}")
        result[key] = row
    return result


def condition(cid: str, dilution: str, solvent: str) -> tuple[str, str, str]:
    numeric = Decimal(dilution)
    if not numeric.is_finite() or not Decimal(0) < numeric <= Decimal(1):
        raise ValueError("invalid stock dilution")
    carrier = solvent.strip().lower()
    carrier = {"1,2-propanediol": "pg", "po": "paraffin oil", "mo": "mineral oil"}.get(carrier, carrier)
    if not carrier:
        raise ValueError("missing solvent")
    return (str(int(cid)), str(numeric.normalize()), carrier)


def formula_key(components: list[tuple[str, str, str]]) -> str:
    if not components:
        raise ValueError("empty mixture")
    # Preserve relative multiplicity, but A+B and A+A+B+B are one physical
    # equal-volume composition and MUST NOT leak across folds.
    fractions = [(key, Fraction(count, len(components))) for key, count in sorted(Counter(components).items())]
    return json.dumps([(key, fraction.numerator, fraction.denominator) for key, fraction in fractions], separators=(",", ":"))


def vector(row: dict, *, optional_release_columns: bool = False) -> np.ndarray:
    values = []
    for name in ENDPOINTS:
        value = row.get(name, "")
        if optional_release_columns and name not in EVALUATED_ENDPOINTS and str(value).strip().lower() in ("", "na", "nan"):
            values.append(np.nan)
        else:
            values.append(float(value))
    result = np.asarray(values, dtype=float)
    required = np.asarray([name in EVALUATED_ENDPOINTS if optional_release_columns else True for name in ENDPOINTS])
    if not np.isfinite(result[required]).all() or np.isinf(result).any() or np.any(result < 0):
        raise ValueError(f"invalid/missing human profile for {row.get('stimulus')}")
    return result


def source_contract(source: Path) -> dict:
    result = {}
    for name, (relative, expected_blob) in SOURCE_FILES.items():
        path = source / relative
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        # Windows git checkouts may translate line endings. SHA256 still binds
        # the exact bytes consumed, while blob identity binds the upstream file.
        lf = raw.replace(b"\r\n", b"\n")
        lf_blob = hashlib.sha1(b"blob " + str(len(lf)).encode() + b"\0" + lf).hexdigest()
        if expected_blob not in (blob, lf_blob):
            raise ValueError(f"upstream source drift: {relative}")
        result[name] = {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "upstream_git_blob": expected_blob, "outcome_parsed": name not in ("target_outcomes",)}
    return result


def load_design(source: Path) -> tuple[dict, dict, dict, dict]:
    components = {key: condition(row["CID"], row["dilution"], row["solvent"])
                  for key, row in unique_rows(rows(source / SOURCE_FILES["components"][0]), "id").items()}
    stimuli = {}
    for key, row in unique_rows(rows(source / SOURCE_FILES["stimuli"][0]), "id").items():
        ids = [part.strip() for part in row["components"].split(";")]
        if any(part not in components for part in ids):
            raise ValueError(f"undefined component in {key}")
        stimuli[key] = [components[part] for part in ids]
    task1 = {key: condition(row["molecule"], row["dilution"], row["solvent"])
             for key, row in unique_rows(rows(source / SOURCE_FILES["task1_stimuli"][0]), "stimulus").items()}
    observed, audit = defaultdict(list), {"ignored_renumbered_single_component_ids": [], "single_rows": 0}
    for row in rows(source / SOURCE_FILES["single"][0]):
        stimulus = row["stimulus"]
        if stimulus in task1:
            key = task1[stimulus]
        else:
            if stimulus not in stimuli or len(stimuli[stimulus]) != 1:
                raise ValueError(f"single lacks an unambiguous stimulus definition: {stimulus}")
            key = stimuli[stimulus][0]
            if row["components"] in components and components[row["components"]] != key:
                audit["ignored_renumbered_single_component_ids"].append(stimulus)
        if key[:2] != condition(row["molecule"], row["dilution"], key[2])[:2]:
            raise ValueError(f"single CID/dilution conflict: {stimulus}")
        observed[key].append(vector(row))
        audit["single_rows"] += 1
    measured = {key: np.mean(value, axis=0) for key, value in observed.items()}
    audit["unique_measured_stock_conditions"] = len(measured)
    audit["replicated_stock_conditions"] = sum(len(value) > 1 for value in observed.values())
    audit["repeats_have_independent_panel_ids"] = False
    audit["join_rule"] = "same-version stimulus to component; assert CID+dilution; never use renumbered single.components"
    return components, stimuli, measured, audit


def native_profiles(registry: Path, snapshot: Path, wheel: Path) -> tuple[dict, dict]:
    """Read the exact runtime snapshot; no research labels change these vectors."""
    import gzip
    import zipfile
    from rdkit import Chem

    for path, expected in ((registry, REGISTRY_HASH), (snapshot, CATALOG_HASH), (wheel, WHEEL_HASH)):
        if sha(path) != expected:
            raise ValueError(f"V5 artifact binding mismatch: {path.name}")
    with zipfile.ZipFile(wheel) as archive:
        for relative in ("fragrance_ai/recommender/models.py", "fragrance_ai/recommender/registry_activation.py"):
            if archive.read(relative) != (ROOT / relative).read_bytes():
                # Source line endings may differ without changing executable code.
                if archive.read(relative).replace(b"\r\n", b"\n") != (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"):
                    raise ValueError("current runtime source differs from pinned V5 wheel")
    payload = json.loads(gzip.decompress(snapshot.read_bytes()))
    if payload["wheel_sha256"] != WHEEL_HASH or payload["registry_sha256"] != REGISTRY_HASH:
        raise ValueError("runtime snapshot provenance mismatch")
    catalog = {row["ingredient_id"]: row for row in payload["ingredients"]}
    con = sqlite3.connect(registry.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        raw = con.execute("SELECT registry_id, canonical_smiles FROM ingredients").fetchall()
        linked = dict(con.execute("SELECT linked_registry_id, ingredient_id FROM formulation_materials WHERE linked_registry_id IS NOT NULL"))
    finally:
        con.close()
    result = {}
    for registry_id, smiles in raw:
        ingredient = catalog.get(linked.get(registry_id, "")) or catalog.get("registry_" + registry_id.split(":", 1)[-1])
        if ingredient is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        profile = np.asarray([ingredient["profile"].get(key, 0.0) for key in SCENT_DIMENSIONS])
        if profile.sum() > 0:
            result[canonical] = {"profile": profile.tolist(), "ingredient_id": ingredient["ingredient_id"],
                                 "odor_impact": ingredient["odor_impact"]}
    return result, {"wheel_sha256": WHEEL_HASH, "catalog_sha256": CATALOG_HASH,
                    "registry_sha256": REGISTRY_HASH, "native_profile_dimensions": list(SCENT_DIMENSIONS),
                    "scope": "exact V5 ingredient-profile features plus a research-only 51-term adapter; not the recipe API"}


def feature(key: tuple[str, str, str], cid_profiles: dict) -> np.ndarray | None:
    item = cid_profiles.get(key[0])
    if item is None:
        return None
    profile = np.asarray(item["profile"], dtype=float)
    log_stock = math.log10(float(key[1]))
    carrier = key[2] if key[2] in CARRIERS else "other"
    return np.r_[profile, profile * log_stock, log_stock, math.log1p(item["odor_impact"]),
                 [float(carrier == name) for name in CARRIERS]]


def measured_mean(keys: list[tuple], measured: dict, *, nearest: bool = False) -> tuple[np.ndarray | None, int]:
    profiles, substitutions = [], 0
    for key in keys:
        if key in measured:
            profiles.append(measured[key])
            continue
        choices = [(abs(math.log10(float(key[1]) / float(other[1]))), other)
                   for other in measured if other[0] == key[0] and other[2] == key[2]] if nearest else []
        choices.sort()
        if not choices or choices[0][0] > 1.0 + 1e-12:
            return None, substitutions
        profiles.append(measured[choices[0][1]])
        substitutions += 1
    return np.mean(profiles, axis=0), substitutions


def blend_features(profiles: np.ndarray, keys: list[tuple]) -> np.ndarray:
    log_stock = np.log10([float(key[1]) for key in keys])
    return np.r_[profiles.mean(axis=0), profiles.max(axis=0), profiles.std(axis=0),
                 len(keys), math.log(len(keys)), log_stock.mean(), log_stock.std()]


def predict_methods(keys: list[tuple], measured: dict, cid_profiles: dict, component_model: dict,
                    mixture_model: dict | None, train_mean: np.ndarray) -> tuple[dict, dict, np.ndarray | None]:
    exact, _ = measured_mean(keys, measured)
    nearest, substitutions = measured_mean(keys, measured, nearest=True)
    features = [feature(key, cid_profiles) for key in keys]
    blended = native = corrected = None
    if all(value is not None for value in features):
        component_prediction = predict_ridge(component_model, np.asarray(features))
        native = component_prediction.mean(axis=0)
        blended = blend_features(component_prediction, keys)
        if mixture_model is not None:
            corrected = predict_ridge(mixture_model, blended[None, :])[0]
    predictions = dict(zip(METHODS, (train_mean, exact, nearest, native, corrected)))
    return ({name: value.tolist() if value is not None else None for name, value in predictions.items()},
            {"exact_measured_components": sum(key in measured for key in keys), "component_count": len(keys),
             "nearest_proxy_substitutions": substitutions, "native_components": sum(value is not None for value in features),
             "mixed_stock_carriers": len({key[2] for key in keys}) > 1,
             "contains_solvent_pseudo_cid": any(int(key[0]) < 0 for key in keys)}, blended)


def prepare(args: argparse.Namespace) -> dict:
    output, source = args.output.resolve(), args.source.resolve()
    if output.exists():
        raise FileExistsError("prepare requires a fresh run directory")
    sources = source_contract(source)
    _, stimuli, measured, audit = load_design(source)
    target_rows = rows(source / SOURCE_FILES["target_ids"][0])
    target_ids = list(unique_rows(target_rows, "stimulus"))
    target_groups = {formula_key(stimuli[key]) for key in target_ids}
    train_rows = rows(source / SOURCE_FILES["training"][0])
    unique_rows(train_rows, "stimulus")
    excluded = [row["stimulus"] for row in train_rows if formula_key(stimuli[row["stimulus"]]) in target_groups]
    train_rows = [row for row in train_rows if row["stimulus"] not in excluded]
    training_y = np.asarray([vector(row) for row in train_rows])
    native, runtime = native_profiles(args.registry, args.catalog, args.wheel)
    from rdkit import Chem
    cid_profiles = {}
    for row in rows(source / SOURCE_FILES["molecules"][0]):
        if not row["SMILES"].strip():
            continue
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is not None:
            profile = native.get(Chem.MolToSmiles(mol, isomericSmiles=True))
            if profile is not None:
                cid_profiles[str(int(row["molecule"]))] = profile
    conditions = sorted(key for key in measured if feature(key, cid_profiles) is not None)
    x = np.asarray([feature(key, cid_profiles) for key in conditions])
    y = np.asarray([measured[key] for key in conditions])
    component_model, component_selection = choose_ridge(x, y, [key[0] for key in conditions])
    mean = training_y.mean(axis=0)
    training_x, training_labels, groups = [], [], []
    training_coverage = Counter()
    for row, label in zip(train_rows, training_y):
        keys = stimuli[row["stimulus"]]
        predictions, _, blended = predict_methods(keys, measured, cid_profiles, component_model, None, mean)
        for name, value in predictions.items():
            training_coverage[name] += value is not None
        if blended is not None:
            training_x.append(blended)
            training_labels.append(label)
            groups.append(formula_key(keys))
    mixture_model, mixture_selection = choose_ridge(np.asarray(training_x), np.asarray(training_labels), groups)
    training_coverage["native_v5_blend_calibrated"] = len(training_x)
    records = []
    for stimulus in target_ids:
        keys = stimuli[stimulus]
        predictions, coverage, _ = predict_methods(keys, measured, cid_profiles, component_model, mixture_model, mean)
        records.append({"stimulus": stimulus, "formula_key": formula_key(keys), "predictions": predictions, "coverage": coverage})
    # Reproducible portable models are exported before opening any target values.
    models = {"component": component_model, "mixture": mixture_model,
              "mean": mean.tolist(), "cid_profiles": cid_profiles,
              "component_selection": component_selection, "mixture_selection": mixture_selection}
    audit.update({"source": UPSTREAM, "commit": COMMIT, "source_files": sources,
                  "source_root": str(source), "target_outcomes_previously_cached": True,
                  "complete_prior_outcome_exposure_audit_available": False,
                  "target_ids": len(target_ids), "target_unique_compositions": len(target_groups),
                  "training_rows_after_excluding_target_formulas": len(train_rows),
                  "training_formulas_excluded_as_target_duplicates": excluded,
                  "training_coverage": dict(training_coverage), "native_training_stock_conditions": len(conditions),
                  "blend_calibration_training_rows": len(training_x),
                  "training_unique_compositions": len(set(formula_key(stimuli[row["stimulus"]]) for row in train_rows)),
                  "target_stock_exact_baseline_profiles": sum(row["predictions"]["measured_exact_mean"] is not None for row in records),
                  "target_mixed_carrier_profiles": sum(row["coverage"]["mixed_stock_carriers"] for row in records),
                  "native_training_molecules": len(set(key[0] for key in conditions)),
                  "corrected_synapse_metadata_download": "anonymous_403_not_bypassed",
                  "participant_replicates_available": False, "data_redistribution_authorized": False})
    protocol = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime, "endpoints": list(ENDPOINTS), "evaluated_endpoints": list(EVALUATED_ENDPOINTS),
                "missing_axis_policy": "49-term primary fixed before outcomes; Ozone and Metallic excluded per organizer release discussion 12207",
                "primary_predictor": METHODS[-1],
                "methods": list(METHODS), "cosine_tolerance": 0.10,
                "paired_contrasts": PAIRED_CONTRASTS,
                "tolerance_scope": "predeclared engineering diagnostic, not an empirical human discrimination threshold",
                "blend_protocol": "equal volumes of pre-diluted stocks; stock dilution is not multiplied into measured RATA twice",
                "carrier_feature_order": list(CARRIERS),
                "measured_baseline_scope": "exact pre-mix stock condition, not exact post-mix concentration/headspace",
                "domain": "new mixture profiles from known molecule families, not molecule-disjoint or temporal/formula-generation validation",
                "outcome_exposure": "public_outcomes_previously_cached; current-run predictions precede scoring",
                "certification_90_allowed": False, "runtime_promotion_allowed": False,
                "scope_blockers": ["no independent participant/retest data for tolerance and uncertainty",
                                   "prior public outcome exposure cannot be excluded",
                                   "research calibration adapter is not deployed recipe inference",
                                   "updated organizer metadata not anonymously downloadable",
                                   "data license not established for runtime redistribution"]}
    output.mkdir(parents=True)
    write_new(output / "protocol.json", protocol)
    write_new(output / "source_audit.json", audit)
    write_new(output / "models.json", models)
    write_new(output / "predictions.json", {"schema": SCHEMA, "records": records})
    bindings = {name: sha(output / name) for name in ("protocol.json", "source_audit.json", "models.json", "predictions.json")}
    code = {str(Path(__file__).resolve()): sha(Path(__file__)),
            str(ROOT / "fragrance_ai/research/perception_validation.py"): sha(ROOT / "fragrance_ai/research/perception_validation.py")}
    write_new(output / "seal.json", {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                                    "files": bindings, "code": code, "independent_timestamp": False})
    return {"stage": "prepared_without_parsing_target_outcomes", "target_profiles": len(records),
            "training_coverage": dict(training_coverage), "native_training_conditions": len(conditions),
            "protocol_sha256": bindings["protocol.json"], "predictions_sha256": bindings["predictions.json"]}


def verify_seal(output: Path) -> tuple[dict, dict, dict, dict]:
    seal = json.loads((output / "seal.json").read_text(encoding="utf-8"))
    expected_files = {"protocol.json", "source_audit.json", "models.json", "predictions.json"}
    expected_code = {str(Path(__file__).resolve()), str(ROOT / "fragrance_ai/research/perception_validation.py")}
    if seal["schema"] != SCHEMA or set(seal["files"]) != expected_files or set(seal["code"]) != expected_code:
        raise ValueError("unknown seal schema")
    for name, expected in seal["files"].items():
        if Path(name).name != name or sha(output / name) != expected:
            raise ValueError(f"sealed artifact changed: {name}")
    for path, expected in seal["code"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"sealed code changed: {path}")
    docs = [json.loads((output / name).read_text(encoding="utf-8"))
            for name in ("protocol.json", "source_audit.json", "models.json", "predictions.json")]
    source = Path(docs[1]["source_root"])
    for binding in docs[1]["source_files"].values():
        if sha(source / binding["path"]) != binding["sha256"]:
            raise ValueError("sealed upstream source changed")
    return tuple(docs)


def score(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if (output / "outcomes_opened.json").exists():
        raise FileExistsError("outcomes have already been opened; preserve this run")
    protocol, audit, models, saved = verify_seal(output)
    source = Path(audit["source_root"])
    _, stimuli, measured, _ = load_design(source)
    # Portable replay must agree before target labels are accessed.
    for row in saved["records"]:
        replay, _, _ = predict_methods(stimuli[row["stimulus"]], measured, models["cid_profiles"],
                                      models["component"], models["mixture"], np.asarray(models["mean"]))
        for name in METHODS:
            if (replay[name] is None) != (row["predictions"][name] is None):
                raise ValueError("portable replay coverage mismatch")
            if replay[name] is not None and not np.allclose(replay[name], row["predictions"][name], atol=1e-12, rtol=0):
                raise ValueError("portable model prediction mismatch")
    write_new(output / "outcomes_opened.json", {"opened_at": datetime.now(timezone.utc).isoformat(),
                                               "seal_sha256": sha(output / "seal.json")})
    outcomes = unique_rows(rows(source / SOURCE_FILES["target_outcomes"][0]), "stimulus")
    if set(outcomes) != {row["stimulus"] for row in saved["records"]}:
        raise ValueError("target outcome population changed; no rows silently dropped")
    raw_y = np.asarray([vector(outcomes[row["stimulus"]], optional_release_columns=True) for row in saved["records"]])
    selected_axes = [ENDPOINTS.index(name) for name in protocol["evaluated_endpoints"]]
    y = raw_y[:, selected_axes]
    groups = [row["formula_key"] for row in saved["records"]]
    predictions, results = {}, {}
    for name in METHODS:
        pred = np.asarray([row["predictions"][name] if row["predictions"][name] is not None else [np.nan] * len(ENDPOINTS)
                           for row in saved["records"]])
        pred = pred[:, selected_axes]
        predictions[name] = pred
        results[name] = summarize_profiles(pred, y, groups, protocol["cosine_tolerance"])
    comparisons = {}
    for contrast, (baseline, candidate) in protocol["paired_contrasts"].items():
        comparisons[contrast] = {"baseline": baseline, "candidate": candidate,
                                **paired_profile_comparison(predictions[baseline], predictions[candidate], y, groups)}
    report = {"schema": SCHEMA, "status": "completed_retrospective_external_profile_evaluation",
              "seal_sha256": sha(output / "seal.json"), "protocol_sha256": sha(output / "protocol.json"),
              "predictions_sha256": sha(output / "predictions.json"), "runtime": protocol["runtime"],
              "outcomes": {"profiles": len(y), "predicted_endpoints": len(ENDPOINTS), "evaluated_endpoints": len(selected_axes),
                           "excluded_axes": [name for name in ENDPOINTS if name not in protocol["evaluated_endpoints"]],
                           "missing_axes_in_actual_release": [name for index, name in enumerate(ENDPOINTS) if not np.isfinite(raw_y[:, index]).all()],
                           "sha256": audit["source_files"]["target_outcomes"]["sha256"]},
              "results": results, "paired_comparisons": comparisons,
              "by_mixture_size": {
                  str(size): {name: summarize_profiles(pred[[row["coverage"]["component_count"] == size for row in saved["records"]]],
                                                      y[[row["coverage"]["component_count"] == size for row in saved["records"]]],
                                                      [group for group, row in zip(groups, saved["records"]) if row["coverage"]["component_count"] == size],
                                                      protocol["cosine_tolerance"])
                              for name, pred in predictions.items()}
                  for size in sorted({row["coverage"]["component_count"] for row in saved["records"]})},
              "portable_model_replay_max_tolerance": 1e-12,
              "actual_human_accuracy_90_authorized": False, "runtime_promotion_allowed": False,
              "scope_blockers": protocol["scope_blockers"]}
    write_new(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--wheel", type=Path, default=ROOT / "dist/phase-runtime-v5/perfumery_ai_core-1.4.0-py3-none-any.whl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "dist/phase-runtime-v5/runtime_catalog_v2.json.gz")
    parser.add_argument("--registry", type=Path, default=ROOT / "benchmarks/industrial_ingredient_registry_v1.db")
    args = parser.parse_args()
    if args.stage == "prepare" and args.source is None:
        parser.error("prepare requires --source")
    print(json.dumps(prepare(args) if args.stage == "prepare" else score(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
