#!/usr/bin/env python
"""Blind, human-grounded evaluation of the frozen R2 ensemble on Bushdid 2014.

The ``predict`` command deliberately has no behavior-file argument.  It reads
only molecular structures and stimulus composition, writes model predictions,
and seals their digest.  The separate ``score`` command verifies that seal
before opening the human behavioral outcomes.

This benchmark measures transfer to previously unused human discrimination
labels.  It is not molecule- or scaffold-disjoint because Bushdid molecules
occur in the frozen model's declared training/pretraining sources, and it does
not measure a newly generated perfume formula by smelling it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREDICTION_SCHEMA_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"
PARTITION_SALT = "bushdid-human-blind-v1-calibration-20pct"
CHANCE_CORRECT = 1.0 / 3.0
DEFAULT_BOOTSTRAP_DRAWS = 20_000
PRIMARY_DATASET_CITATION = {
    "title": "Humans can discriminate more than one trillion olfactory stimuli",
    "doi": "10.1126/science.1249168",
    "use_boundary": (
        "Only the 26-subject raw odd-one-out trial outcomes are evaluated; "
        "the disputed extrapolation to the number of discriminable odors is not used."
    ),
}


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _as_identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_molecules(path: Path) -> dict[str, str]:
    from fragrance_ai.research.r2_physsim import canonical_smiles

    result: dict[str, str] = {}
    for row in _read_csv(path):
        cid = _as_identifier(row.get("CID", ""))
        smiles = str(row.get("IsomericSMILES", "")).strip()
        if not cid or not smiles:
            continue
        canonical = canonical_smiles(smiles)
        previous = result.get(cid)
        if previous is not None and previous != canonical:
            raise RuntimeError(f"CID {cid} maps to multiple structures")
        result[cid] = canonical
    if not result:
        raise RuntimeError("Bushdid molecule table contains no usable structures")
    return result


def _component_ids(row: dict[str, str]) -> tuple[str, ...]:
    values = []
    for index in range(1, 31):
        value = _as_identifier(row.get(f"Molecule {index}", ""))
        if value and value != "0":
            values.append(value)
    return tuple(values)


def _partition_stimuli(stimuli: list[dict[str, Any]]) -> None:
    """Assign 20% calibration within structural strata without behavior labels."""

    strata: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for stimulus in stimuli:
        if stimulus["components_per_mixture"] <= 1:
            stimulus["evaluation_partition"] = "control"
            continue
        key = (
            int(stimulus["components_per_mixture"]),
            float(stimulus["declared_overlap_percent"]),
        )
        strata[key].append(stimulus)
    for key, rows in strata.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{PARTITION_SALT}|{key}|{row['stimulus_id']}".encode()
            ).hexdigest(),
        )
        calibration_count = max(1, int(math.ceil(len(ranked) * 0.20)))
        for index, row in enumerate(ranked):
            row["evaluation_partition"] = (
                "calibration" if index < calibration_count else "final_test"
            )


def load_bushdid_stimuli(
    molecules_path: Path,
    stimuli_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    molecules = _load_molecules(molecules_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(stimuli_path):
        grouped[_as_identifier(row.get("Stimulus", ""))].append(row)
    if not grouped or "" in grouped:
        raise RuntimeError("Bushdid stimulus table has missing identifiers")

    stimuli: list[dict[str, Any]] = []
    for stimulus_id in sorted(grouped, key=lambda value: int(value)):
        rows = grouped[stimulus_id]
        right = [row for row in rows if row.get("Answer", "").lower() == "right"]
        wrong = [row for row in rows if row.get("Answer", "").lower() == "wrong"]
        if len(rows) != 3 or len(right) != 1 or len(wrong) != 2:
            raise RuntimeError(f"stimulus {stimulus_id} is not one-right/two-wrong")
        right_ids = _component_ids(right[0])
        wrong_ids = _component_ids(wrong[0])
        if wrong_ids != _component_ids(wrong[1]):
            raise RuntimeError(f"stimulus {stimulus_id} wrong mixtures differ")
        component_count = int(float(right[0]["Components in mixtures"]))
        if len(right_ids) != component_count or len(wrong_ids) != component_count:
            raise RuntimeError(f"stimulus {stimulus_id} component count mismatch")
        missing = (set(right_ids) | set(wrong_ids)) - set(molecules)
        if missing:
            raise RuntimeError(
                f"stimulus {stimulus_id} has unmapped CIDs: {sorted(missing)}"
            )
        shared = len(set(right_ids) & set(wrong_ids))
        shared_fraction = shared / component_count
        declared_overlap = float(right[0]["% mixture overlap"])
        if not math.isclose(
            shared_fraction * 100.0,
            declared_overlap,
            abs_tol=0.011,
        ):
            raise RuntimeError(f"stimulus {stimulus_id} overlap metadata mismatch")
        if any(
            int(float(row["Components in mixtures"])) != component_count
            or not math.isclose(
                float(row["% mixture overlap"]), declared_overlap, abs_tol=1e-9
            )
            for row in rows
        ):
            raise RuntimeError(f"stimulus {stimulus_id} metadata differs by vial")
        stimuli.append(
            {
                "stimulus_id": stimulus_id,
                "components_per_mixture": component_count,
                "components_that_differ": int(
                    float(right[0]["Components that differ"])
                ),
                "declared_overlap_percent": declared_overlap,
                "shared_component_fraction": shared_fraction,
                "right_cids": right_ids,
                "wrong_cids": wrong_ids,
                "right_smiles": tuple(molecules[value] for value in right_ids),
                "wrong_smiles": tuple(molecules[value] for value in wrong_ids),
                "right_dilution": float(right[0]["Stimulus dilution"]),
                "wrong_dilutions": tuple(
                    float(row["Stimulus dilution"]) for row in wrong
                ),
            }
        )
    _partition_stimuli(stimuli)
    return molecules, stimuli


def _verify_wheel_binding(
    wheel: Path,
    data_dir: Path,
) -> dict[str, Any]:
    source_files = {
        "fragrance_ai/__init__.py": PROJECT_ROOT / "fragrance_ai" / "__init__.py",
        "fragrance_ai/research/r2_physsim.py": (
            PROJECT_ROOT / "fragrance_ai" / "research" / "r2_physsim.py"
        ),
        "fragrance_ai/data/physsim_r2_manifest.json": (
            data_dir / "physsim_r2_manifest.json"
        ),
        "fragrance_ai/data/physsim_r2_ensemble_manifest.json": (
            data_dir / "physsim_r2_ensemble_manifest.json"
        ),
        "fragrance_ai/data/physsim_r2_checkpoint.pt": (
            data_dir / "physsim_r2_checkpoint.pt"
        ),
        "fragrance_ai/data/physsim_r2_checkpoint_seed_20260713.pt": (
            data_dir / "physsim_r2_checkpoint_seed_20260713.pt"
        ),
    }
    verified: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        version_lines = [
            line.split(":", 1)[1].strip()
            for line in metadata.splitlines()
            if line.startswith("Version:")
        ]
        if len(version_lines) != 1:
            raise RuntimeError("wheel version metadata is missing or ambiguous")
        for member, source in source_files.items():
            if member not in names:
                raise RuntimeError(f"wheel is missing frozen model resource: {member}")
            wheel_bytes = archive.read(member)
            source_bytes = source.read_bytes()
            if wheel_bytes != source_bytes:
                raise RuntimeError(f"source/wheel model binding mismatch: {member}")
            verified[member] = hashlib.sha256(wheel_bytes).hexdigest()
    return {
        "path": str(wheel.resolve()),
        "sha256": sha256_file(wheel),
        "bytes": wheel.stat().st_size,
        "package_version": version_lines[0],
        "source_resources_byte_identical": True,
        "verified_resources": verified,
    }


def _verify_declared_model_sources(
    model_source_root: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, specification in manifest["source_files"].items():
        path = model_source_root / relative
        digest = sha256_file(path)
        if digest != str(specification["sha256"]):
            raise RuntimeError(f"frozen model source hash mismatch: {relative}")
        if path.stat().st_size != int(specification["bytes"]):
            raise RuntimeError(f"frozen model source size mismatch: {relative}")
        verified[relative] = digest
    if any("bushdid_2014" in name.lower() for name in verified):
        raise RuntimeError("Bushdid data unexpectedly occurs in frozen model sources")
    return verified


def _canonical_structures(path: Path) -> set[str]:
    from fragrance_ai.research.r2_physsim import canonical_smiles

    values: set[str] = set()
    for row in _read_csv(path):
        raw = str(row.get("IsomericSMILES", "")).strip()
        if not raw:
            continue
        try:
            values.add(canonical_smiles(raw))
        except ValueError:
            continue
    return values


def _model_overlap_audit(
    bushdid_molecules: Iterable[str],
    model_source_root: Path,
) -> dict[str, Any]:
    from fragrance_ai.research.r2_physsim import (
        bemis_murcko_scaffold,
        load_snitz_pairs,
    )

    bushdid = set(bushdid_molecules)
    snitz_pairs = load_snitz_pairs(model_source_root / "dream_mixture")
    snitz_used = {value for pair in snitz_pairs for value in pair.molecules}
    pretraining: set[str] = set()
    for archive in (
        "leffingwell",
        "goodscents",
        "flavornet",
        "aromadb",
        "ifra_2019",
        "flavordb",
    ):
        pretraining.update(
            _canonical_structures(
                model_source_root / "pyrfume_all" / archive / "molecules.csv"
            )
        )
    bushdid_scaffolds = {bemis_murcko_scaffold(value) for value in bushdid}
    snitz_scaffolds = {bemis_murcko_scaffold(value) for value in snitz_used}
    pretraining_scaffolds = {
        bemis_murcko_scaffold(value) for value in pretraining
    }
    all_seen = snitz_used | pretraining
    all_seen_scaffolds = snitz_scaffolds | pretraining_scaffolds
    return {
        "behavior_labels_used_by_frozen_model": False,
        "behavior_label_blind_external_transfer": True,
        "bushdid_molecules": len(bushdid),
        "snitz_training_molecules": len(snitz_used),
        "descriptor_pretraining_molecules": len(pretraining),
        "exact_overlap_with_snitz_training": len(bushdid & snitz_used),
        "exact_overlap_with_descriptor_pretraining": len(bushdid & pretraining),
        "exact_unseen_from_all_declared_model_sources": len(bushdid - all_seen),
        "bushdid_scaffolds": len(bushdid_scaffolds),
        "scaffold_overlap_with_snitz_training": len(
            bushdid_scaffolds & snitz_scaffolds
        ),
        "scaffold_overlap_with_descriptor_pretraining": len(
            bushdid_scaffolds & pretraining_scaffolds
        ),
        "scaffold_unseen_from_all_declared_model_sources": len(
            bushdid_scaffolds - all_seen_scaffolds
        ),
        "molecule_disjoint": not bool(bushdid & all_seen),
        "scaffold_disjoint": not bool(bushdid_scaffolds & all_seen_scaffolds),
        "interpretation": (
            "Human discrimination outcomes are unused external labels, but the "
            "chemicals/scaffolds are not novel to all declared model sources."
        ),
    }


def _load_frozen_ensemble(data_dir: Path):
    import torch

    from fragrance_ai.research.r2_physsim import R2PhysSimCore

    manifest_path = data_dir / "physsim_r2_ensemble_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    release = manifest.get("release_gate", {})
    checks = release.get("checks", {})
    if not release.get("passed") or not checks or not all(checks.values()):
        raise RuntimeError("frozen ensemble release gate is not satisfied")
    members = manifest.get("members", [])
    if len(members) != 2:
        raise RuntimeError("blind benchmark requires the frozen two-member ensemble")
    weights = np.asarray([float(member["weight"]) for member in members])
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("invalid frozen ensemble weights")

    models = []
    means = []
    standard_deviations = []
    descriptor_names = []
    for member in members:
        path = data_dir / str(member["file"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(member["sha256"]):
            raise RuntimeError("frozen checkpoint hash mismatch")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if int(payload["training"]["model_seed"]) != int(member["model_seed"]):
            raise RuntimeError("frozen checkpoint seed mismatch")
        model = R2PhysSimCore()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
        means.append(np.asarray(payload["normalizer"]["mean"], dtype=np.float32))
        standard_deviations.append(
            np.asarray(payload["normalizer"]["std"], dtype=np.float32)
        )
        descriptor_names.append(tuple(payload["normalizer"]["descriptor_names"]))
    if not all(np.array_equal(means[0], value) for value in means[1:]):
        raise RuntimeError("ensemble normalizer means differ")
    if not all(
        np.array_equal(standard_deviations[0], value)
        for value in standard_deviations[1:]
    ):
        raise RuntimeError("ensemble normalizer standard deviations differ")
    if not all(descriptor_names[0] == value for value in descriptor_names[1:]):
        raise RuntimeError("ensemble descriptor contracts differ")
    return (
        models,
        weights,
        means[0],
        standard_deviations[0],
        manifest,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _normalized_descriptors(
    smiles: Iterable[str],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> dict[str, np.ndarray]:
    from fragrance_ai.research.r2_physsim import smiles_to_descriptors

    result: dict[str, np.ndarray] = {}
    for value in sorted(set(smiles)):
        raw = smiles_to_descriptors(value).astype(np.float64)
        standardized = (raw - mean.astype(np.float64)) / standard_deviation.astype(
            np.float64
        )
        result[value] = np.clip(
            np.nan_to_num(
                standardized,
                nan=0.0,
                posinf=100.0,
                neginf=-100.0,
            ),
            -100.0,
            100.0,
        ).astype(np.float32)
    return result


def _padded_mixtures(
    stimuli: Sequence[dict[str, Any]],
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = max(int(row["components_per_mixture"]) for row in stimuli)
    descriptor_width = len(next(iter(cache.values())))
    right = np.zeros((len(stimuli), width, descriptor_width), dtype=np.float32)
    wrong = np.zeros_like(right)
    right_mask = np.zeros((len(stimuli), width), dtype=np.float32)
    wrong_mask = np.zeros_like(right_mask)
    for row_index, stimulus in enumerate(stimuli):
        for column, smiles in enumerate(stimulus["right_smiles"]):
            right[row_index, column] = cache[smiles]
            right_mask[row_index, column] = 1.0
        for column, smiles in enumerate(stimulus["wrong_smiles"]):
            wrong[row_index, column] = cache[smiles]
            wrong_mask[row_index, column] = 1.0
    return right, right_mask, wrong, wrong_mask


def _ensemble_predictions(
    models: Sequence[Any],
    weights: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
    wrong: np.ndarray,
    wrong_mask: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    member_rows = []
    maximum_symmetry_delta = 0.0
    with torch.inference_mode():
        for model in models:
            values = []
            swapped_values = []
            for start in range(0, len(right), batch_size):
                stop = min(start + batch_size, len(right))
                right_batch = torch.from_numpy(right[start:stop])
                right_mask_batch = torch.from_numpy(right_mask[start:stop])
                wrong_batch = torch.from_numpy(wrong[start:stop])
                wrong_mask_batch = torch.from_numpy(wrong_mask[start:stop])
                values.extend(
                    model(
                        right_batch,
                        right_mask_batch,
                        wrong_batch,
                        wrong_mask_batch,
                    )
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )
                swapped_values.extend(
                    model(
                        wrong_batch,
                        wrong_mask_batch,
                        right_batch,
                        right_mask_batch,
                    )
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )
            current = np.asarray(values, dtype=float)
            swapped = np.asarray(swapped_values, dtype=float)
            maximum_symmetry_delta = max(
                maximum_symmetry_delta,
                float(np.max(np.abs(current - swapped))),
            )
            member_rows.append(current)
    members = np.vstack(member_rows).T
    ensemble = members @ weights
    return ensemble, members, maximum_symmetry_delta


def build_blind_predictions(
    *,
    dataset_root: Path,
    model_source_root: Path,
    data_dir: Path,
    wheel: Path,
    batch_size: int,
    torch_threads: int,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    molecules_path = dataset_root / "molecules.csv"
    stimuli_path = dataset_root / "stimuli.csv"
    molecules, stimuli = load_bushdid_stimuli(molecules_path, stimuli_path)

    model_manifest = json.loads(
        (data_dir / "physsim_r2_manifest.json").read_text(encoding="utf-8")
    )
    verified_model_sources = _verify_declared_model_sources(
        model_source_root, model_manifest
    )
    wheel_binding = _verify_wheel_binding(wheel, data_dir)
    models, weights, mean, std, ensemble_manifest, ensemble_manifest_hash = (
        _load_frozen_ensemble(data_dir)
    )
    cache = _normalized_descriptors(molecules.values(), mean, std)
    right, right_mask, wrong, wrong_mask = _padded_mixtures(stimuli, cache)
    ensemble, member_predictions, symmetry_delta = _ensemble_predictions(
        models,
        weights,
        right,
        right_mask,
        wrong,
        wrong_mask,
        batch_size=batch_size,
    )
    if symmetry_delta > 1e-6:
        raise RuntimeError("frozen R2 pair symmetry failed")

    disagreement_limit = float(
        ensemble_manifest["uncertainty"]["maximum_member_disagreement"]
    )
    neutral_similarity = float(
        ensemble_manifest["calibration"]["neutral_similarity_percent"]
    ) / 100.0
    prediction_rows = []
    for index, stimulus in enumerate(stimuli):
        member_values = member_predictions[index]
        disagreement = float(np.max(member_values) - np.min(member_values))
        descriptors = np.vstack(
            [cache[value] for value in (*stimulus["right_smiles"], *stimulus["wrong_smiles"])]
        )
        prediction_rows.append(
            {
                "stimulus_id": stimulus["stimulus_id"],
                "evaluation_partition": stimulus["evaluation_partition"],
                "components_per_mixture": stimulus["components_per_mixture"],
                "components_that_differ": stimulus["components_that_differ"],
                "declared_overlap_percent": stimulus["declared_overlap_percent"],
                "component_overlap_similarity": stimulus[
                    "shared_component_fraction"
                ],
                "component_overlap_dissimilarity": 1.0
                - stimulus["shared_component_fraction"],
                "r2_similarity": float(ensemble[index]),
                "r2_dissimilarity": float(1.0 - ensemble[index]),
                "member_predictions": [float(value) for value in member_values],
                "member_disagreement": disagreement,
                "within_frozen_disagreement_gate": disagreement
                <= disagreement_limit,
                "descriptor_z_within_8_fraction": float(
                    np.mean(np.abs(descriptors) <= 8.0)
                ),
                "frozen_neutral_similarity_prediction": (
                    "discriminable"
                    if float(ensemble[index]) <= neutral_similarity
                    else "not_discriminable"
                ),
            }
        )
    prediction_hash = hashlib.sha256(
        _canonical_json(prediction_rows)
    ).hexdigest()
    overlap_audit = _model_overlap_audit(molecules.values(), model_source_root)
    partition_counts = {
        name: sum(row["evaluation_partition"] == name for row in prediction_rows)
        for name in ("calibration", "final_test", "control")
    }
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blind_predictions_sealed_before_human_outcomes",
        "blind_contract": {
            "target_behavior_file_read": False,
            "predict_command_accepts_target_behavior_path": False,
            "human_outcomes_used_for_partitioning": False,
            "human_outcomes_used_for_model_or_threshold_selection": False,
            "partition_rule": (
                "20% calibration within each composition-size/overlap stratum "
                "using SHA-256 ordering; remaining 80% is final_test"
            ),
            "partition_salt": PARTITION_SALT,
            "prediction_rows_sha256": prediction_hash,
        },
        "dataset": {
            "name": "Bushdid 2014",
            "citation": PRIMARY_DATASET_CITATION,
            "input_files_without_human_outcomes": {
                "molecules.csv": {
                    "path": str(molecules_path.resolve()),
                    "sha256": sha256_file(molecules_path),
                    "bytes": molecules_path.stat().st_size,
                },
                "stimuli.csv": {
                    "path": str(stimuli_path.resolve()),
                    "sha256": sha256_file(stimuli_path),
                    "bytes": stimuli_path.stat().st_size,
                },
            },
            "molecules": len(molecules),
            "stimuli": len(stimuli),
            "partition_counts": partition_counts,
        },
        "frozen_model": {
            "wheel": wheel_binding,
            "ensemble_manifest_sha256": ensemble_manifest_hash,
            "member_weights": [float(value) for value in weights],
            "neutral_similarity_threshold": neutral_similarity,
            "maximum_member_disagreement": disagreement_limit,
            "declared_source_hashes_verified": verified_model_sources,
            "model_overlap_audit": overlap_audit,
            "prediction_symmetry_max_abs_delta": symmetry_delta,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_threads": torch_threads,
            "deterministic_algorithms": True,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
        "predictions": prediction_rows,
        "claim_boundary": (
            "Behavior-label-blind external human discrimination transfer. "
            "Not molecule/scaffold-disjoint and not a sensory test of generated recipes."
        ),
    }


def write_prediction_and_seal(
    prediction: dict[str, Any],
    output: Path,
    seal_output: Path,
) -> dict[str, Any]:
    _atomic_write(
        output,
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
    )
    seal = {
        "schema_version": "1.0",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "prediction_file": output.name,
        "prediction_file_sha256": sha256_file(output),
        "prediction_file_bytes": output.stat().st_size,
        "prediction_rows_sha256": prediction["blind_contract"][
            "prediction_rows_sha256"
        ],
        "frozen_wheel_sha256": prediction["frozen_model"]["wheel"]["sha256"],
        "target_behavior_file_read_before_seal": False,
    }
    _atomic_write(
        seal_output,
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n",
    )
    return seal


def verify_prediction_seal(
    prediction_path: Path,
    seal_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("prediction_file") != prediction_path.name:
        raise RuntimeError("prediction seal filename mismatch")
    if sha256_file(prediction_path) != seal.get("prediction_file_sha256"):
        raise RuntimeError("prediction seal SHA-256 mismatch")
    if prediction_path.stat().st_size != int(seal["prediction_file_bytes"]):
        raise RuntimeError("prediction seal size mismatch")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction.get("status") != "blind_predictions_sealed_before_human_outcomes":
        raise RuntimeError("prediction artifact is not blind-sealed")
    contract = prediction.get("blind_contract", {})
    if (
        contract.get("target_behavior_file_read") is not False
        or contract.get("human_outcomes_used_for_partitioning") is not False
        or contract.get("human_outcomes_used_for_model_or_threshold_selection")
        is not False
    ):
        raise RuntimeError("prediction artifact violates blind contract")
    rows_hash = hashlib.sha256(
        _canonical_json(prediction["predictions"])
    ).hexdigest()
    if rows_hash != seal.get("prediction_rows_sha256") or rows_hash != contract.get(
        "prediction_rows_sha256"
    ):
        raise RuntimeError("sealed prediction-row digest mismatch")
    return prediction, seal


def _read_behavior_matrix(
    path: Path,
    stimulus_ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    rows = _read_csv(path)
    subjects = sorted(
        {_as_identifier(row.get("Subject", "")) for row in rows},
        key=lambda value: int(value),
    )
    if not subjects or "" in subjects:
        raise RuntimeError("human behavior table has missing subjects")
    subject_index = {value: index for index, value in enumerate(subjects)}
    stimulus_index = {value: index for index, value in enumerate(stimulus_ids)}
    matrix = np.full((len(stimulus_ids), len(subjects)), np.nan, dtype=float)
    for row in rows:
        stimulus = _as_identifier(row.get("Stimulus", ""))
        subject = _as_identifier(row.get("Subject", ""))
        if stimulus not in stimulus_index:
            raise RuntimeError(f"human outcome has unknown stimulus {stimulus}")
        raw = str(row.get("Correct", "")).strip().lower()
        if raw not in {"true", "false"}:
            raise RuntimeError("human outcome Correct must be True or False")
        location = (stimulus_index[stimulus], subject_index[subject])
        if np.isfinite(matrix[location]):
            raise RuntimeError("duplicate subject/stimulus human outcome")
        matrix[location] = 1.0 if raw == "true" else 0.0
    if not np.isfinite(matrix).all():
        raise RuntimeError("human behavior matrix is incomplete")
    return matrix, subjects


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(p_values)
    ordered = p_values[order]
    thresholds = alpha * np.arange(1, len(ordered) + 1) / len(ordered)
    accepted = np.flatnonzero(ordered <= thresholds)
    result = np.zeros(len(ordered), dtype=bool)
    if len(accepted):
        result[p_values <= ordered[accepted[-1]]] = True
    return result


def _rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    left_rank = rankdata(left, axis=1, method="average")
    right_rank = rankdata(right, axis=1, method="average")
    left_centered = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    return np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0,
    )


def _two_way_spearman_bootstrap(
    human_correct: np.ndarray,
    model_score: np.ndarray,
    baseline_score: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model_values = []
    baseline_values = []
    stimulus_count, subject_count = human_correct.shape
    chunk = 250
    for offset in range(0, draws, chunk):
        count = min(chunk, draws - offset)
        subject_indices = rng.integers(
            0, subject_count, size=(count, subject_count)
        )
        stimulus_indices = rng.integers(
            0, stimulus_count, size=(count, stimulus_count)
        )
        resampled_rates = human_correct[:, subject_indices].mean(axis=2).T
        resampled_rates = np.take_along_axis(
            resampled_rates, stimulus_indices, axis=1
        )
        model_values.append(
            _rowwise_spearman(model_score[stimulus_indices], resampled_rates)
        )
        baseline_values.append(
            _rowwise_spearman(baseline_score[stimulus_indices], resampled_rates)
        )
    model = np.concatenate(model_values)
    baseline = np.concatenate(baseline_values)
    finite = np.isfinite(model) & np.isfinite(baseline)
    if int(finite.sum()) < draws * 0.999:
        raise RuntimeError("too many non-finite two-way bootstrap draws")
    model = model[finite]
    baseline = baseline[finite]
    difference = model - baseline
    return {
        "model": model,
        "baseline": baseline,
        "difference": difference,
        "model_95_interval": [
            float(value) for value in np.quantile(model, [0.025, 0.975])
        ],
        "baseline_95_interval": [
            float(value) for value in np.quantile(baseline, [0.025, 0.975])
        ],
        "model_minus_baseline_95_interval": [
            float(value) for value in np.quantile(difference, [0.025, 0.975])
        ],
        "fraction_model_not_above_baseline": float(np.mean(difference <= 0.0)),
        "draws": draws,
        "seed": seed,
        "resampling_unit": "crossed subjects and stimuli",
    }


def _split_half_reliability(
    human_correct: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subject_count = human_correct.shape[1]
    half = subject_count // 2
    if half < 2:
        raise RuntimeError("split-half reliability needs at least four subjects")
    values = []
    chunk = 250
    for offset in range(0, draws, chunk):
        count = min(chunk, draws - offset)
        permutations = np.argsort(
            rng.random((count, subject_count)), axis=1
        )
        first = human_correct[:, permutations[:, :half]].mean(axis=2).T
        second = human_correct[:, permutations[:, half : 2 * half]].mean(
            axis=2
        ).T
        correlation = _rowwise_spearman(first, second)
        corrected = np.divide(
            2.0 * correlation,
            1.0 + correlation,
            out=np.full_like(correlation, np.nan),
            where=np.abs(1.0 + correlation) > 1e-12,
        )
        values.append(corrected)
    result = np.concatenate(values)
    return result[np.isfinite(result)]


def _rowwise_auc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    positives = labels.sum(axis=1)
    negatives = labels.shape[1] - positives
    ranks = rankdata(scores, axis=1, method="average")
    positive_rank_sum = np.sum(ranks * labels, axis=1)
    return (
        positive_rank_sum - positives * (positives + 1.0) / 2.0
    ) / (positives * negatives)


def _stratified_auc_bootstrap(
    labels: np.ndarray,
    model_score: np.ndarray,
    baseline_score: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    positive_indices = np.flatnonzero(labels)
    negative_indices = np.flatnonzero(~labels)
    if not len(positive_indices) or not len(negative_indices):
        raise RuntimeError("AUC requires both discriminable classes")
    rng = np.random.default_rng(seed)
    model_values = []
    baseline_values = []
    chunk = 250
    for offset in range(0, draws, chunk):
        count = min(chunk, draws - offset)
        positives = positive_indices[
            rng.integers(
                0,
                len(positive_indices),
                size=(count, len(positive_indices)),
            )
        ]
        negatives = negative_indices[
            rng.integers(
                0,
                len(negative_indices),
                size=(count, len(negative_indices)),
            )
        ]
        indices = np.concatenate((positives, negatives), axis=1)
        current_labels = np.concatenate(
            (
                np.ones_like(positives, dtype=float),
                np.zeros_like(negatives, dtype=float),
            ),
            axis=1,
        )
        model_values.append(_rowwise_auc(current_labels, model_score[indices]))
        baseline_values.append(
            _rowwise_auc(current_labels, baseline_score[indices])
        )
    model = np.concatenate(model_values)
    baseline = np.concatenate(baseline_values)
    difference = model - baseline
    return {
        "model_95_interval": [
            float(value) for value in np.quantile(model, [0.025, 0.975])
        ],
        "baseline_95_interval": [
            float(value) for value in np.quantile(baseline, [0.025, 0.975])
        ],
        "model_minus_baseline_95_interval": [
            float(value) for value in np.quantile(difference, [0.025, 0.975])
        ],
        "fraction_model_not_above_baseline": float(np.mean(difference <= 0.0)),
        "draws": draws,
        "seed": seed,
        "resampling_unit": "stimuli, stratified by human FDR label",
    }


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires observations")
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        fraction * (1.0 - fraction) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [center - margin, center + margin]


def _classification_metrics(
    labels: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    true_positive = int(np.sum(labels & predicted))
    true_negative = int(np.sum(~labels & ~predicted))
    false_positive = int(np.sum(~labels & predicted))
    false_negative = int(np.sum(labels & ~predicted))
    total = len(labels)
    accuracy = (true_positive + true_negative) / total
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    return {
        "accuracy": accuracy,
        "accuracy_95_wilson_interval": _wilson_interval(
            true_positive + true_negative, total
        ),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def _partial_spearman(
    model_score: np.ndarray,
    human_rate: np.ndarray,
    controls: np.ndarray,
) -> float:
    from scipy.stats import rankdata

    model_rank = rankdata(model_score)
    human_rank = rankdata(human_rate)
    design = np.column_stack(
        (np.ones(len(controls)), *(rankdata(controls[:, i]) for i in range(controls.shape[1])))
    )
    model_residual = model_rank - design @ np.linalg.lstsq(
        design, model_rank, rcond=None
    )[0]
    human_residual = human_rank - design @ np.linalg.lstsq(
        design, human_rank, rcond=None
    )[0]
    denominator = float(
        np.linalg.norm(model_residual) * np.linalg.norm(human_residual)
    )
    return (
        float(np.dot(model_residual, human_residual) / denominator)
        if denominator > 0
        else 0.0
    )


def _calibrated_rate_error(
    calibration_score: np.ndarray,
    calibration_rate: np.ndarray,
    final_score: np.ndarray,
    final_rate: np.ndarray,
    final_subject_outcomes: np.ndarray,
) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss

    calibrator = IsotonicRegression(
        y_min=CHANCE_CORRECT,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    calibrator.fit(calibration_score, calibration_rate)
    probability = np.asarray(calibrator.predict(final_score), dtype=float)
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    residual = probability - final_rate
    repeated_probability = np.repeat(
        probability[:, None], final_subject_outcomes.shape[1], axis=1
    )
    return {
        "mapping": "isotonic fitted on pre-sealed calibration partition only",
        "final_mean_absolute_error": float(np.mean(np.abs(residual))),
        "final_mean_absolute_error_percentage_points": float(
            100.0 * np.mean(np.abs(residual))
        ),
        "final_root_mean_squared_error": float(
            np.sqrt(np.mean(residual * residual))
        ),
        "final_root_mean_squared_error_percentage_points": float(
            100.0 * np.sqrt(np.mean(residual * residual))
        ),
        "final_mean_bias_percentage_points": float(100.0 * np.mean(residual)),
        "individual_trial_brier_score": float(
            brier_score_loss(
                final_subject_outcomes.ravel(), repeated_probability.ravel()
            )
        ),
        "individual_trial_log_loss": float(
            log_loss(
                final_subject_outcomes.ravel(), repeated_probability.ravel(), labels=[0, 1]
            )
        ),
        "predicted_correct_rate_min": float(np.min(probability)),
        "predicted_correct_rate_max": float(np.max(probability)),
    }


def score_blind_predictions(
    *,
    prediction_path: Path,
    seal_path: Path,
    behavior_path: Path,
    bootstrap_draws: int,
) -> dict[str, Any]:
    from scipy.stats import binomtest, pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    started = time.perf_counter()
    # Seal verification intentionally completes before the human file is opened.
    prediction, seal = verify_prediction_seal(prediction_path, seal_path)
    seal_verified_at = datetime.now(timezone.utc).isoformat()
    rows = prediction["predictions"]
    stimulus_ids = [str(row["stimulus_id"]) for row in rows]
    human_correct, subjects = _read_behavior_matrix(behavior_path, stimulus_ids)
    behavior_opened_at = datetime.now(timezone.utc).isoformat()
    human_rate = human_correct.mean(axis=1)
    correct_count = human_correct.sum(axis=1).astype(int)
    trial_count = human_correct.shape[1]
    p_values = np.asarray(
        [
            binomtest(
                int(count),
                trial_count,
                p=CHANCE_CORRECT,
                alternative="greater",
            ).pvalue
            for count in correct_count
        ],
        dtype=float,
    )
    fdr_labels = _benjamini_hochberg(p_values)
    nominal_labels = p_values < 0.05

    partition = np.asarray([row["evaluation_partition"] for row in rows])
    calibration_mask = partition == "calibration"
    final_mask = partition == "final_test"
    control_mask = partition == "control"
    if not calibration_mask.any() or not final_mask.any() or not control_mask.any():
        raise RuntimeError("sealed evaluation partitions are incomplete")

    model_score = np.asarray([row["r2_dissimilarity"] for row in rows])
    baseline_score = np.asarray(
        [row["component_overlap_dissimilarity"] for row in rows]
    )
    model_similarity = 1.0 - model_score
    neutral_threshold = float(
        prediction["frozen_model"]["neutral_similarity_threshold"]
    )
    fixed_prediction = model_similarity <= neutral_threshold

    final_human_rate = human_rate[final_mask]
    final_model = model_score[final_mask]
    final_baseline = baseline_score[final_mask]
    final_correct = human_correct[final_mask]
    final_fdr = fdr_labels[final_mask]
    final_nominal = nominal_labels[final_mask]
    model_spearman = float(spearmanr(final_model, final_human_rate).statistic)
    baseline_spearman = float(
        spearmanr(final_baseline, final_human_rate).statistic
    )
    model_pearson = float(pearsonr(final_model, final_human_rate).statistic)
    baseline_pearson = float(
        pearsonr(final_baseline, final_human_rate).statistic
    )
    model_auc = float(roc_auc_score(final_fdr, final_model))
    baseline_auc = float(roc_auc_score(final_fdr, final_baseline))
    model_average_precision = float(
        average_precision_score(final_fdr, final_model)
    )
    baseline_average_precision = float(
        average_precision_score(final_fdr, final_baseline)
    )

    crossed = _two_way_spearman_bootstrap(
        final_correct,
        final_model,
        final_baseline,
        draws=bootstrap_draws,
        seed=20260822,
    )
    reliability = _split_half_reliability(
        final_correct,
        draws=bootstrap_draws,
        seed=20260823,
    )
    positive_reliability = reliability[reliability > 0.0]
    if len(positive_reliability) < bootstrap_draws * 0.95:
        raise RuntimeError("human split-half reliability is too often non-positive")
    reliability_point = float(np.median(positive_reliability))
    noise_ceiling = math.sqrt(reliability_point)
    normalized_point = model_spearman / noise_ceiling
    rng = np.random.default_rng(20260824)
    model_bootstrap = crossed["model"]
    count = min(len(model_bootstrap), len(positive_reliability))
    reliability_sample = rng.choice(
        positive_reliability, size=count, replace=len(positive_reliability) < count
    )
    normalized_samples = model_bootstrap[:count] / np.sqrt(reliability_sample)
    normalized_interval = [
        float(value)
        for value in np.quantile(normalized_samples, [0.025, 0.975])
    ]

    auc_bootstrap = _stratified_auc_bootstrap(
        final_fdr,
        final_model,
        final_baseline,
        draws=bootstrap_draws,
        seed=20260825,
    )
    model_calibration = _calibrated_rate_error(
        model_score[calibration_mask],
        human_rate[calibration_mask],
        final_model,
        final_human_rate,
        final_correct,
    )
    baseline_calibration = _calibrated_rate_error(
        baseline_score[calibration_mask],
        human_rate[calibration_mask],
        final_baseline,
        final_human_rate,
        final_correct,
    )
    controls = np.column_stack(
        (
            final_baseline,
            np.asarray(
                [row["components_per_mixture"] for row in rows], dtype=float
            )[final_mask],
        )
    )
    partial = _partial_spearman(final_model, final_human_rate, controls)
    fixed_classification = _classification_metrics(
        final_fdr, fixed_prediction[final_mask]
    )

    stimulus_results = []
    for index, row in enumerate(rows):
        stimulus_results.append(
            {
                "stimulus_id": row["stimulus_id"],
                "evaluation_partition": row["evaluation_partition"],
                "components_per_mixture": row["components_per_mixture"],
                "declared_overlap_percent": row["declared_overlap_percent"],
                "r2_similarity": row["r2_similarity"],
                "r2_dissimilarity": row["r2_dissimilarity"],
                "component_overlap_dissimilarity": row[
                    "component_overlap_dissimilarity"
                ],
                "human_correct_count": int(correct_count[index]),
                "human_trials": trial_count,
                "human_correct_rate": float(human_rate[index]),
                "above_chance_one_sided_p": float(p_values[index]),
                "nominally_discriminable": bool(nominal_labels[index]),
                "fdr_discriminable": bool(fdr_labels[index]),
            }
        )

    gate = {
        "definition": (
            "model correlation divided by sqrt(Spearman-Brown-corrected "
            "split-half reliability) on the sealed final partition"
        ),
        "threshold": 0.90,
        "point_estimate": normalized_point,
        "bootstrap_95_interval": normalized_interval,
        "passed": normalized_interval[0] >= 0.90,
        "claim_boundary": (
            "Approximate human-ceiling-normalized profile agreement, not a "
            "percentage of end-to-end perfume smell equivalence."
        ),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "blind_integrity": {
            "prediction_seal_verified_before_behavior_open": True,
            "prediction_file_sha256": seal["prediction_file_sha256"],
            "prediction_rows_sha256": seal["prediction_rows_sha256"],
            "prediction_sealed_at": seal["sealed_at"],
            "seal_verified_at": seal_verified_at,
            "behavior_opened_at": behavior_opened_at,
            "human_outcomes_used_for_frozen_predictions": False,
            "human_outcomes_used_for_partitioning": False,
            "calibration_labels_used_only_for_post_hoc_probability_mapping": True,
            "final_test_labels_used_for_tuning": False,
        },
        "dataset": {
            "name": "Bushdid 2014",
            "citation": PRIMARY_DATASET_CITATION,
            "behavior_file": {
                "path": str(behavior_path.resolve()),
                "sha256": sha256_file(behavior_path),
                "bytes": behavior_path.stat().st_size,
            },
            "subjects": len(subjects),
            "stimuli_total": len(rows),
            "calibration_mixture_stimuli": int(calibration_mask.sum()),
            "final_test_mixture_stimuli": int(final_mask.sum()),
            "control_stimuli": int(control_mask.sum()),
            "trials_per_stimulus": trial_count,
            "final_test_fdr_discriminable": int(final_fdr.sum()),
            "final_test_nominally_discriminable": int(final_nominal.sum()),
        },
        "model_scope": prediction["frozen_model"]["model_overlap_audit"],
        "final_test_results": {
            "continuous_human_correct_rate": {
                "r2_spearman": model_spearman,
                "r2_spearman_two_way_bootstrap_95_interval": crossed[
                    "model_95_interval"
                ],
                "component_overlap_spearman": baseline_spearman,
                "component_overlap_spearman_two_way_bootstrap_95_interval": crossed[
                    "baseline_95_interval"
                ],
                "r2_minus_component_overlap_spearman": model_spearman
                - baseline_spearman,
                "paired_difference_95_interval": crossed[
                    "model_minus_baseline_95_interval"
                ],
                "r2_pearson": model_pearson,
                "component_overlap_pearson": baseline_pearson,
                "r2_partial_spearman_controlling_overlap_and_mixture_size": partial,
                "two_way_bootstrap": {
                    key: value
                    for key, value in crossed.items()
                    if key not in {"model", "baseline", "difference"}
                },
            },
            "fdr_discriminability": {
                "human_label": (
                    "one-sided exact binomial versus 1/3 chance, Benjamini-Hochberg "
                    "FDR 0.05 across all 264 sealed stimuli"
                ),
                "r2_roc_auc": model_auc,
                "r2_average_precision": model_average_precision,
                "component_overlap_roc_auc": baseline_auc,
                "component_overlap_average_precision": baseline_average_precision,
                "r2_minus_component_overlap_auc": model_auc - baseline_auc,
                "paired_stratified_bootstrap": auc_bootstrap,
                "frozen_neutral_threshold_classification": fixed_classification,
            },
            "post_seal_calibrated_human_rate_error": {
                "r2": model_calibration,
                "component_overlap": baseline_calibration,
                "boundary": (
                    "The isotonic mapping uses only the pre-sealed 20% calibration "
                    "partition; frozen R2 rankings are never refit."
                ),
            },
            "human_noise_ceiling": {
                "method": (
                    "20,000 random subject split-halves, Spearman-Brown correction; "
                    "correlation ceiling approximated as sqrt(reliability)"
                ),
                "corrected_reliability_median": reliability_point,
                "corrected_reliability_95_interval": [
                    float(value)
                    for value in np.quantile(
                        positive_reliability, [0.025, 0.975]
                    )
                ],
                "correlation_noise_ceiling": noise_ceiling,
            },
            "human_ceiling_90_percent_gate": gate,
        },
        "controls": {
            "human_correct_rate_mean": float(np.mean(human_rate[control_mask])),
            "r2_dissimilarity_mean": float(np.mean(model_score[control_mask])),
            "excluded_from_primary_metrics": True,
        },
        "stimulus_results": stimulus_results,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "bootstrap_draws": bootstrap_draws,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
        "claim_boundary": (
            "Actual human odd-one-out measurements are the ground truth. This "
            "is behavior-label-blind transfer on chemically seen molecules, not "
            "molecule-disjoint validation and not direct smelling of generated recipes."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    result = report["final_test_results"]
    continuous = result["continuous_human_correct_rate"]
    discrimination = result["fdr_discriminability"]
    calibrated = result["post_seal_calibrated_human_rate_error"]
    gate = result["human_ceiling_90_percent_gate"]
    dataset = report["dataset"]
    scope = report["model_scope"]
    return "\n".join(
        [
            "# Bushdid 인간 후각 블라인드 외부 검증",
            "",
            f"- 상태: `{report['status']}`",
            "- 예측 봉인 후 인간 행동값 공개: `PASS`",
            f"- 인간 참가자: `{dataset['subjects']}`명",
            f"- 보정/최종/대조 자극: `{dataset['calibration_mixture_stimuli']}` / "
            f"`{dataset['final_test_mixture_stimuli']}` / `{dataset['control_stimuli']}`",
            "",
            "## 봉인된 최종 시험 결과",
            "",
            "| 지표 | R2 | 성분 겹침 기준선 | R2-기준선 |",
            "|---|---:|---:|---:|",
            (
                "| 인간 정답률 Spearman | "
                f"{continuous['r2_spearman']:.4f} | "
                f"{continuous['component_overlap_spearman']:.4f} | "
                f"{continuous['r2_minus_component_overlap_spearman']:+.4f} |"
            ),
            (
                "| 인간 판별 FDR-label ROC-AUC | "
                f"{discrimination['r2_roc_auc']:.4f} | "
                f"{discrimination['component_overlap_roc_auc']:.4f} | "
                f"{discrimination['r2_minus_component_overlap_auc']:+.4f} |"
            ),
            (
                "| 보정 후 인간 정답률 MAE(%p) | "
                f"{calibrated['r2']['final_mean_absolute_error_percentage_points']:.3f} | "
                f"{calibrated['component_overlap']['final_mean_absolute_error_percentage_points']:.3f} | "
                "- |"
            ),
            "",
            "- R2 Spearman 95% 구간: "
            f"`[{continuous['r2_spearman_two_way_bootstrap_95_interval'][0]:+.4f}, "
            f"{continuous['r2_spearman_two_way_bootstrap_95_interval'][1]:+.4f}]`",
            "- R2-기준선 Spearman 차이 95% 구간: "
            f"`[{continuous['paired_difference_95_interval'][0]:+.4f}, "
            f"{continuous['paired_difference_95_interval'][1]:+.4f}]`",
            f"- 인간 ceiling 대비 점수: `{gate['point_estimate']:.4f}`; 95% 구간 "
            f"`[{gate['bootstrap_95_interval'][0]:+.4f}, "
            f"{gate['bootstrap_95_interval'][1]:+.4f}]`",
            f"- 인간 ceiling 90% 게이트: `{'PASS' if gate['passed'] else 'FAIL'}`",
            "",
            "## 검증 범위",
            "",
            "- Bushdid 행동 라벨은 모델 학습·선택·분할에 사용되지 않았습니다.",
            f"- 전체 선언 모델 원천에서 처음 보는 분자: "
            f"`{scope['exact_unseen_from_all_declared_model_sources']}`개",
            f"- 전체 선언 모델 원천에서 처음 보는 scaffold: "
            f"`{scope['scaffold_unseen_from_all_declared_model_sources']}`개",
            "- 따라서 실제 인간 실측값 기반 행동-label-blind 전이 검증이지만, "
            "분자/scaffold-disjoint 검증은 아닙니다.",
            "- 생성 레시피를 사람이 직접 맡은 종단 관능 정확도 시험도 아닙니다.",
            "",
            f"> {report['claim_boundary']}",
            "",
        ]
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="predict from structures/stimuli only and seal before human outcomes",
    )
    predict.add_argument("--dataset-root", type=Path, required=True)
    predict.add_argument("--model-source-root", type=Path, required=True)
    predict.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "fragrance_ai" / "data"
    )
    predict.add_argument("--wheel", type=Path, required=True)
    predict.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "bushdid_blind_predictions_v1.json",
    )
    predict.add_argument(
        "--seal-output",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "bushdid_blind_prediction_seal_v1.json"
        ),
    )
    predict.add_argument("--batch-size", type=positive_int, default=16)
    predict.add_argument("--torch-threads", type=positive_int, default=4)

    score = subparsers.add_parser(
        "score", help="verify prediction seal, then reveal and score human outcomes"
    )
    score.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "bushdid_blind_predictions_v1.json",
    )
    score.add_argument(
        "--seal",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "bushdid_blind_prediction_seal_v1.json"
        ),
    )
    score.add_argument("--behavior", type=Path, required=True)
    score.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.json"
        ),
    )
    score.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            PROJECT_ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.md"
        ),
    )
    score.add_argument(
        "--bootstrap-draws", type=positive_int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    args = parser.parse_args()

    if args.command == "predict":
        prediction = build_blind_predictions(
            dataset_root=args.dataset_root.resolve(),
            model_source_root=args.model_source_root.resolve(),
            data_dir=args.data_dir.resolve(),
            wheel=args.wheel.resolve(),
            batch_size=args.batch_size,
            torch_threads=args.torch_threads,
        )
        seal = write_prediction_and_seal(
            prediction, args.output.resolve(), args.seal_output.resolve()
        )
        print(
            json.dumps(
                {
                    "status": prediction["status"],
                    "prediction_rows": len(prediction["predictions"]),
                    "partition_counts": prediction["dataset"]["partition_counts"],
                    "prediction_file_sha256": seal["prediction_file_sha256"],
                    "target_behavior_file_read": False,
                    "output": str(args.output.resolve()),
                    "seal_output": str(args.seal_output.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = score_blind_predictions(
        prediction_path=args.predictions.resolve(),
        seal_path=args.seal.resolve(),
        behavior_path=args.behavior.resolve(),
        bootstrap_draws=args.bootstrap_draws,
    )
    _atomic_write(
        args.output.resolve(),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(args.markdown_output.resolve(), render_markdown(report))
    result = report["final_test_results"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "blind_integrity": report["blind_integrity"],
                "dataset": report["dataset"],
                "continuous_human_correct_rate": result[
                    "continuous_human_correct_rate"
                ],
                "fdr_discriminability": result["fdr_discriminability"],
                "post_seal_calibrated_human_rate_error": result[
                    "post_seal_calibrated_human_rate_error"
                ],
                "human_ceiling_90_percent_gate": result[
                    "human_ceiling_90_percent_gate"
                ],
                "output": str(args.output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
