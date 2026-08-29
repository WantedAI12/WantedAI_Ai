#!/usr/bin/env python
"""Outcome-unopened Ma et al. 2021 binary-mixture benchmark.

The workflow intentionally separates the ingested first worksheet (72 odorant
metadata rows) from the original three-sheet workbook that contains human
outcomes:

1. ``prepare`` downloads only the Dataverse tabular metadata representation,
   resolves structures through PubChem, trains target-excluded monomolecular
   predictors on Keller 2016, and emits predictions for all 72 choose 2 pairs.
2. ``seal`` binds those 2,556 predictions, the implementation, and the fixed
   mixture operators while requiring the original workbook to be absent.
3. After an external RFC 3161 timestamp, ``acquire`` downloads the original
   workbook from its immutable Dataverse file identifier.
4. ``score`` verifies every binding and evaluates both an interaction-only
   track (measured component intensities -> mixture intensity) and a fully
   prospective structure-to-mixture track.

The primary operator is a Weber--Fechner excitation pool whose slope is
derived from the already frozen Ravia concentration-response artifact.  No Ma
mixture outcome is used to choose a model, coefficient, endpoint, or gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender.numpy_r2 import (  # noqa: E402
    EXPECTED_STATE_SHAPES,
    NumpyR2Model,
    _gelu,
    _linear,
)
from fragrance_ai.research.r2_physsim import (  # noqa: E402
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    canonical_smiles,
)
from scripts import blind_bierling_human_olfaction_benchmark as human  # noqa: E402


SCHEMA_VERSION = "1.0"
DATASET_DOI = "10.15454/51OVY6"
ARTICLE_DOI = "10.1016/j.dib.2021.107143"
RELATED_INTENSITY_DOI = "10.1016/j.foodchem.2021.129483"
DATAVERSE_DATASET_VERSION_ID = 262_820
DATAVERSE_FILE_ID = 110_870
DATAVERSE_FILE_PID = "doi:10.15454/51OVY6/COAY4H"
METADATA_URL = (
    "https://entrepot.recherche.data.gouv.fr/api/access/datafile/110870"
)
OUTCOME_URL = METADATA_URL + "?format=original"
METADATA_FILE = "data in brief V2.tab"
METADATA_BYTES = 7_013
METADATA_MD5 = "9c6fa6328950a9ddfd1c372a77aaaa0f"
METADATA_SHA256 = "3c1a4385c2f877086a045fd437da0a37eed4e1e6d79c4a16a8c9c4262cd4dd15"
DATAVERSE_INGEST_STORAGE_BYTES = 6_955
DATAVERSE_INGEST_STORAGE_MD5 = "53b74248b1aeb7f1418c06c28e19c3ac"
OUTCOME_FILE = "data in brief V2.xlsx"
OUTCOME_BYTES = 487_926
TARGET_ODOR_COUNT = 72
ALL_PAIR_COUNT = TARGET_ODOR_COUNT * (TARGET_ODOR_COUNT - 1) // 2
EXPECTED_TRIAL_ROWS = 222
EXPECTED_DISTINCT_MIXTURES = 198
EXPECTED_INDIVIDUAL_ROWS = 6_660
TARGET_INTENSITY_MAXIMUM = 11.0
TARGET_SCALE_DIVISOR = 10.0
RAVIA_REFERENCE_LOG10_FRACTION = -2.0
BOOTSTRAP_SEED = 20_260_830
BOOTSTRAP_DRAWS = 5_000
PRIMARY_INTERACTION_MODEL = "ravia_weber_fechner_pool"
FIXED_INTERACTION_COMPARATOR = "strongest_component"

R2_RUNTIME = PROJECT_ROOT / "fragrance_ai" / "data" / "physsim_r2_runtime_weights.npz"
R2_RUNTIME_MANIFEST = (
    PROJECT_ROOT / "fragrance_ai" / "data" / "physsim_r2_runtime_manifest.json"
)
R2_ENSEMBLE_MANIFEST = (
    PROJECT_ROOT / "fragrance_ai" / "data" / "physsim_r2_ensemble_manifest.json"
)
CONCENTRATION_RUNTIME = (
    PROJECT_ROOT / "fragrance_ai" / "data" / "concentration_response_runtime.json"
)
CONCENTRATION_MANIFEST = (
    PROJECT_ROOT / "fragrance_ai" / "data" / "concentration_response_manifest.json"
)

KELLER_SOURCE_CONTRACT = human.KELLER_SOURCE_CONTRACT
ENDPOINTS = ("intensive", "pleasant")

OPERATOR_CONTRACT: dict[str, Any] = {
    "primary": PRIMARY_INTERACTION_MODEL,
    "fixed_comparator": FIXED_INTERACTION_COMPARATOR,
    "target_intensity_scale": [0.0, TARGET_INTENSITY_MAXIMUM],
    "ravia_reference_log10_fraction": RAVIA_REFERENCE_LOG10_FRACTION,
    "ravia_scale_conversion": (
        "differentiate frozen 0-100 Ravia ridge at log10_fraction=-2, "
        "divide by ln(10), then divide by 10 for the Ma 0-10 scale"
    ),
    "strongest_component": "max(IA, IB)",
    "ravia_weber_fechner_pool": (
        "max(IA,IB) + s*log(1 + exp(-abs(IA-IB)/s))"
    ),
    "r2_channel_overlap_pool": (
        "max(IA,IB) + s*log(1 + (1-R2_similarity)*"
        "exp(-abs(IA-IB)/s))"
    ),
    "morgan_channel_overlap_pool": (
        "max(IA,IB) + s*log(1 + (1-Morgan_similarity)*"
        "exp(-abs(IA-IB)/s))"
    ),
    "root_sum_square": "sqrt(IA**2 + IB**2)",
    "complete_addition": "IA + IB",
    "clipping": [0.0, TARGET_INTENSITY_MAXIMUM],
    "pleasantness_diagnostic": (
        "component pleasantness weighted by exp(component_intensity/2.0)"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return human.sha256_file(path)


def md5_bytes(value: bytes) -> str:
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return human.canonical_json_sha256(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    human.write_json(path, value)


def _download_bytes(url: str, *, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "perfumery-ai-core-public-human-benchmark/1.0",
            "Accept": "application/json, text/tab-separated-values, */*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    translations = {
        "γ": "gamma",
        "β": "beta",
        "α": "alpha",
        "δ": "delta",
    }
    for source, target in translations.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", text)


def _pair_id(first_cas: str, second_cas: str) -> str:
    return "|".join(sorted((str(first_cas).strip(), str(second_cas).strip())))


def assert_target_outcome_absent(path: Path) -> None:
    target = path.resolve()
    candidates = {
        target,
        target.with_name("data in brief.xlsx"),
        target.with_name("data in brief V2.xlsx"),
        target.with_name("data in brief.zip"),
        target.with_name("data in brief V2.zip"),
    }
    present = sorted(str(item) for item in candidates if item.exists())
    if present:
        raise RuntimeError(
            "Ma target outcome workbook was present before the blind boundary: "
            + ", ".join(present)
        )


def _parse_metadata(value: bytes, *, expected_count: int = TARGET_ODOR_COUNT) -> list[dict[str, Any]]:
    try:
        text = value.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise RuntimeError("Ma metadata is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required = {
        "CAS.",
        "Odorant",
        "Odor",
        "Cons.(mg/mL)",
        "Solvent",
        "Purity",
        "Trialnumber",
    }
    if set(reader.fieldnames or ()) != required:
        raise RuntimeError(f"Ma metadata columns changed: {reader.fieldnames}")
    rows = []
    seen_cas: set[str] = set()
    seen_names: set[str] = set()
    for source in reader:
        cas = str(source["CAS."]).strip()
        odorant = unicodedata.normalize("NFKC", str(source["Odorant"]).strip())
        normalized = normalize_name(odorant)
        if not cas or not normalized or cas in seen_cas or normalized in seen_names:
            raise RuntimeError(f"duplicate or empty Ma odor identity: {cas!r}, {odorant!r}")
        concentration = float(str(source["Cons.(mg/mL)"]).strip())
        if not math.isfinite(concentration) or concentration <= 0.0:
            raise RuntimeError(f"invalid Ma concentration for {cas}")
        trials = tuple(
            int(item.strip())
            for item in str(source["Trialnumber"]).split(",")
            if item.strip()
        )
        if not trials or any(value < 1 or value > EXPECTED_TRIAL_ROWS for value in trials):
            raise RuntimeError(f"invalid trial metadata for {cas}")
        seen_cas.add(cas)
        seen_names.add(normalized)
        rows.append(
            {
                "cas": cas,
                "odorant": odorant,
                "normalized_odorant": normalized,
                "odor_note": str(source["Odor"]).strip(),
                "concentration_mg_ml": concentration,
                "solvent": str(source["Solvent"]).strip(),
                "purity": str(source["Purity"]).strip(),
                "metadata_trial_numbers": list(trials),
            }
        )
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} Ma odorants, found {len(rows)}")
    return rows


def _resolve_pubchem(cas: str) -> dict[str, Any]:
    properties = (
        "CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,"
        "MolecularWeight,Title"
    )
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(cas, safe="")
        + f"/property/{properties}/JSON"
    )
    raw = _download_bytes(url)
    try:
        document = json.loads(raw)
        values = document["PropertyTable"]["Properties"]
        row = values[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PubChem did not resolve CAS {cas}") from error
    smiles = str(row.get("SMILES") or row.get("IsomericSMILES") or "").strip()
    connectivity = str(
        row.get("ConnectivitySMILES") or row.get("CanonicalSMILES") or smiles
    ).strip()
    if not smiles:
        raise RuntimeError(f"PubChem returned no structure for CAS {cas}")
    canonical = canonical_smiles(smiles)
    return {
        "pubchem_cid": int(row["CID"]),
        "pubchem_title": str(row.get("Title", "")).strip(),
        "canonical_smiles": canonical,
        "pubchem_isomeric_smiles": smiles,
        "pubchem_connectivity_smiles": connectivity,
        "inchi_key": str(row["InChIKey"]).strip(),
        "molecular_formula": str(row["MolecularFormula"]).strip(),
        "molecular_weight": float(row["MolecularWeight"]),
        "pubchem_property_url": url,
        "pubchem_response_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _fetch_and_resolve_metadata(metadata_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metadata_path.exists():
        raw = metadata_path.read_bytes()
    else:
        raw = _download_bytes(METADATA_URL)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=metadata_path.parent, prefix=metadata_path.name + ".", delete=False
        ) as handle:
            handle.write(raw)
            temporary = Path(handle.name)
        os.replace(temporary, metadata_path)
    if (
        len(raw) != METADATA_BYTES
        or md5_bytes(raw) != METADATA_MD5
        or hashlib.sha256(raw).hexdigest() != METADATA_SHA256
    ):
        raise RuntimeError("Ma tabular metadata differs from the frozen Dataverse file")
    rows = _parse_metadata(raw)
    resolved = []
    for index, row in enumerate(rows):
        structure = _resolve_pubchem(row["cas"])
        resolved.append(
            {
                **row,
                **structure,
                "murcko_scaffold": bemis_murcko_scaffold(
                    structure["canonical_smiles"]
                ),
            }
        )
        if index + 1 < len(rows):
            time.sleep(0.12)
    structures = [row["canonical_smiles"] for row in resolved]
    if len(set(structures)) != len(structures):
        duplicates = sorted(
            value for value in set(structures) if structures.count(value) > 1
        )
        raise RuntimeError(f"Ma CAS rows resolved to duplicate structures: {duplicates}")
    return resolved, {
        "path": str(metadata_path.resolve()),
        "url": METADATA_URL,
        "bytes": len(raw),
        "md5": md5_bytes(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dataverse_ingest_storage_bytes": DATAVERSE_INGEST_STORAGE_BYTES,
        "dataverse_ingest_storage_md5": DATAVERSE_INGEST_STORAGE_MD5,
        "access_api_export_is_normalized_tsv": True,
        "dataverse_file_id": DATAVERSE_FILE_ID,
        "dataverse_file_pid": DATAVERSE_FILE_PID,
        "contains_human_outcomes": False,
        "ignored_for_model_selection": ["metadata_trial_numbers"],
    }


def _verify_keller_sources(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        row = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
        expected = KELLER_SOURCE_CONTRACT[name]
        if row["sha256"] != expected["sha256"] or row["bytes"] != expected["bytes"]:
            raise RuntimeError(f"Keller source differs from frozen contract: {name}")
        result[name] = row
    return result


def _humanpom_predictions(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    keller_molecules: Path,
    keller_stimuli: Path,
    keller_behavior: Path,
    molformer_root: Path,
    hf_home: Path,
    threads: int,
    batch_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    target_smiles = [str(row["canonical_smiles"]) for row in target_rows]
    target_set = set(target_smiles)
    target_ring_scaffolds = {
        str(row["murcko_scaffold"])
        for row in target_rows
        if str(row["murcko_scaffold"])
    }
    keller_smiles, keller_outcomes, _, keller_audit = human._keller_means(
        keller_molecules.resolve(strict=True),
        keller_stimuli.resolve(strict=True),
        keller_behavior.resolve(strict=True),
    )
    exact_training_smiles = [value for value in keller_smiles if value not in target_set]
    strict_training_smiles = [
        value
        for value in exact_training_smiles
        if not bemis_murcko_scaffold(value)
        or bemis_murcko_scaffold(value) not in target_ring_scaffolds
    ]
    if len(exact_training_smiles) < 300 or len(strict_training_smiles) < 150:
        raise RuntimeError("target-excluded Keller training set is too small")

    all_smiles = sorted(set(keller_smiles) | target_set)
    global_index = {value: index for index, value in enumerate(all_smiles)}
    keller_index = {value: index for index, value in enumerate(keller_smiles)}
    outcomes = np.full((len(all_smiles), len(human.ENDPOINTS)), np.nan, dtype=float)
    for smiles in keller_smiles:
        outcomes[global_index[smiles]] = keller_outcomes[keller_index[smiles]]
    features, feature_audit = human._feature_matrices(
        all_smiles,
        molformer_root=molformer_root.resolve(strict=True),
        hf_home=hf_home.resolve(),
        batch_size=batch_size,
        torch_threads=threads,
    )
    exact_indices = np.asarray(
        [global_index[value] for value in exact_training_smiles], dtype=int
    )
    strict_indices = np.asarray(
        [global_index[value] for value in strict_training_smiles], dtype=int
    )
    target_indices = np.asarray([global_index[value] for value in target_smiles], dtype=int)
    development, _, selected_by_endpoint = human._candidate_development(
        features,
        exact_indices,
        outcomes,
        exact_training_smiles,
        threads=threads,
    )
    selected = {endpoint: selected_by_endpoint[endpoint] for endpoint in ENDPOINTS}
    needed = sorted(set(selected.values()) | {human.FIXED_BASELINE})
    exact = human._full_candidate_predictions(
        features,
        outcomes,
        exact_indices,
        target_indices,
        needed,
        seed_offset=30_000,
        threads=threads,
    )
    strict = human._full_candidate_predictions(
        features,
        outcomes,
        strict_indices,
        target_indices,
        sorted(set(selected.values())),
        seed_offset=40_000,
        threads=threads,
    )
    endpoint_index = {name: list(human.ENDPOINTS).index(name) for name in ENDPOINTS}
    result: dict[str, dict[str, np.ndarray]] = {}
    for endpoint in ENDPOINTS:
        column = endpoint_index[endpoint]
        candidate = selected[endpoint]
        result[endpoint] = {
            "primary": np.clip(exact[candidate][:, column] / TARGET_SCALE_DIVISOR, 0.0, 10.0),
            "strict": np.clip(strict[candidate][:, column] / TARGET_SCALE_DIVISOR, 0.0, 10.0),
            "fixed_rdkit": np.clip(
                exact[human.FIXED_BASELINE][:, column] / TARGET_SCALE_DIVISOR,
                0.0,
                10.0,
            ),
        }
    if any(
        not np.all(np.isfinite(values))
        for endpoint in result.values()
        for values in endpoint.values()
    ):
        raise RuntimeError("HumanPOM target predictions are non-finite")
    audit = {
        "training_dataset": "Keller and Vosshall 2016",
        "training_doi": "10.1186/s12868-016-0287-2",
        "aggregation": keller_audit,
        "all_usable_molecules": len(keller_smiles),
        "exact_target_overlap_excluded": len(target_set & set(keller_smiles)),
        "exact_label_disjoint_training_molecules": len(exact_training_smiles),
        "ring_scaffold_and_exact_label_disjoint_training_molecules": len(
            strict_training_smiles
        ),
        "target_ring_scaffolds": len(target_ring_scaffolds),
        "feature_contract": feature_audit,
        "selected_by_endpoint": selected,
        "development": development,
        "scale_conversion": "Keller 0-100 predictions divided by 10 for Ma 0-10",
        "target_exact_human_label_leakage_count": len(
            target_set & set(exact_training_smiles)
        ),
    }
    return result, audit


def _ravia_fechner_scale() -> tuple[float, dict[str, Any]]:
    runtime_bytes = CONCENTRATION_RUNTIME.read_bytes()
    runtime = json.loads(runtime_bytes)
    manifest = json.loads(CONCENTRATION_MANIFEST.read_text(encoding="utf-8"))
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    if runtime_sha != manifest.get("runtime_sha256"):
        raise RuntimeError("Ravia concentration runtime/manifest binding failed")
    if runtime.get("feature_contract") != [
        "log10_dilution",
        "log10_dilution_squared",
    ]:
        raise RuntimeError("unexpected Ravia concentration feature contract")
    scale = np.asarray(runtime["feature_scale"], dtype=float)
    coefficients = np.asarray(runtime["coefficients"], dtype=float)
    x = RAVIA_REFERENCE_LOG10_FRACTION
    derivative_per_log10 = coefficients[0] / scale[0] + coefficients[1] * 2.0 * x / scale[1]
    target_units_per_ln = derivative_per_log10 / math.log(10.0) / TARGET_SCALE_DIVISOR
    if not math.isfinite(target_units_per_ln) or target_units_per_ln <= 0.0:
        raise RuntimeError("derived Ravia Weber-Fechner scale is invalid")
    return float(target_units_per_ln), {
        "runtime_path": str(CONCENTRATION_RUNTIME.resolve()),
        "runtime_sha256": runtime_sha,
        "manifest_sha256": sha256_file(CONCENTRATION_MANIFEST),
        "reference_log10_fraction": x,
        "derivative_0_100_units_per_log10": float(derivative_per_log10),
        "ma_target_units_per_natural_log": float(target_units_per_ln),
    }


def _load_r2_projected(
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    np.ndarray,
    dict[str, Any],
]:
    runtime_bytes = R2_RUNTIME.read_bytes()
    runtime_manifest_bytes = R2_RUNTIME_MANIFEST.read_bytes()
    ensemble_manifest_bytes = R2_ENSEMBLE_MANIFEST.read_bytes()
    runtime_manifest = json.loads(runtime_manifest_bytes)
    ensemble_manifest = json.loads(ensemble_manifest_bytes)
    if hashlib.sha256(runtime_bytes).hexdigest() != runtime_manifest.get("artifact_sha256"):
        raise RuntimeError("R2 portable runtime hash mismatch")
    ensemble_hash = hashlib.sha256(ensemble_manifest_bytes).hexdigest()
    if runtime_manifest.get("ensemble_manifest_sha256") != ensemble_hash:
        raise RuntimeError("R2 runtime is bound to another ensemble")
    release = ensemble_manifest.get("release_gate", {})
    if not release.get("passed") or not all(release.get("checks", {}).values()):
        raise RuntimeError("R2 ensemble release gate is closed")
    state_keys = list(runtime_manifest.get("state_keys", []))
    if set(state_keys) != set(EXPECTED_STATE_SHAPES):
        raise RuntimeError("R2 state key contract changed")
    members = list(runtime_manifest.get("members", []))
    weights = np.asarray([float(row["weight"]) for row in members], dtype=float)
    if len(weights) < 2 or np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("R2 ensemble weights are invalid")
    raw_cache = build_raw_descriptor_cache(
        str(row["canonical_smiles"]) for row in target_rows
    )
    projected_members: list[dict[str, np.ndarray]] = []
    head_states: list[dict[str, np.ndarray]] = []
    with np.load(io.BytesIO(runtime_bytes), allow_pickle=False) as data:
        mean = np.asarray(data["normalizer_mean"], dtype=np.float64)
        std = np.asarray(data["normalizer_std"], dtype=np.float64)
        if mean.shape != (217,) or std.shape != (217,) or np.any(std <= 0.0):
            raise RuntimeError("R2 normalizer is invalid")
        standardized = {
            smiles: np.clip(
                np.nan_to_num(
                    (np.asarray(values, dtype=np.float64) - mean) / std,
                    nan=0.0,
                    posinf=100.0,
                    neginf=-100.0,
                ),
                -100.0,
                100.0,
            )
            for smiles, values in raw_cache.items()
        }
        for index, member in enumerate(members):
            state = {
                key: np.asarray(data[f"member_{index}::{key}"], dtype=np.float64)
                for key in state_keys
            }
            model = NumpyR2Model(state)
            head_states.append(
                {
                    key: state[key]
                    for key in (
                        "similarity_head.0.weight",
                        "similarity_head.0.bias",
                        "similarity_head.3.weight",
                        "similarity_head.3.bias",
                    )
                }
            )
            projected_members.append(
                {
                    smiles: model._project(model._mixture_fingerprint(values[None, :]))
                    for smiles, values in standardized.items()
                }
            )
    audit = {
        "runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "runtime_manifest_sha256": hashlib.sha256(runtime_manifest_bytes).hexdigest(),
        "ensemble_manifest_sha256": ensemble_hash,
        "members": [
            {
                "file": row["file"],
                "sha256": row["sha256"],
                "weight": float(row["weight"]),
            }
            for row in members
        ],
        "normalizer_descriptor_count": 217,
        "direct_input": "one canonical molecule in each mixture branch",
    }
    return projected_members, head_states, weights, audit


def _r2_similarity(
    projected_members: Sequence[Mapping[str, np.ndarray]],
    head_states: Sequence[Mapping[str, np.ndarray]],
    weights: np.ndarray,
    first: str,
    second: str,
) -> tuple[float, list[float], float]:
    member_values = []
    if len(projected_members) != len(head_states) or len(weights) != len(head_states):
        raise RuntimeError("R2 projected-member/head-state count mismatch")
    for projected, state in zip(projected_members, head_states, strict=True):
        first_value = projected[first]
        second_value = projected[second]
        difference = np.abs(first_value - second_value)
        product = first_value * second_value
        denominator = np.linalg.norm(first_value) * np.linalg.norm(second_value)
        cosine = float(first_value @ second_value / max(float(denominator), 1e-8))
        features = np.concatenate((difference, product, [cosine]))
        hidden = _gelu(
            _linear(
                features,
                state["similarity_head.0.weight"],
                state["similarity_head.0.bias"],
            )
        )
        logit = float(
            _linear(
                hidden,
                state["similarity_head.3.weight"],
                state["similarity_head.3.bias"],
            )[0]
        )
        if logit >= 0.0:
            prediction = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            prediction = exponential / (1.0 + exponential)
        member_values.append(prediction)
    ensemble = float(np.dot(weights, np.asarray(member_values, dtype=float)))
    disagreement = float(max(member_values) - min(member_values))
    return ensemble, member_values, disagreement


def _morgan_similarities(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for row in target_rows:
        smiles = str(row["canonical_smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"invalid target molecule for Morgan: {smiles}")
        fingerprints[smiles] = generator.GetFingerprint(molecule)

    def similarity(first: str, second: str) -> float:
        return float(DataStructs.TanimotoSimilarity(fingerprints[first], fingerprints[second]))

    return {"function": similarity, "radius": 2, "bits": 2048}


def _clip_intensity(value: float) -> float:
    return float(np.clip(value, 0.0, TARGET_INTENSITY_MAXIMUM))


def _fechner_pool(
    first: float,
    second: float,
    scale: float,
    *,
    independent_channel_fraction: float = 1.0,
) -> float:
    values = np.asarray([first, second, scale, independent_channel_fraction], dtype=float)
    if not np.all(np.isfinite(values)) or scale <= 0.0:
        raise ValueError("invalid Weber-Fechner pooling input")
    if not 0.0 <= independent_channel_fraction <= 1.0:
        raise ValueError("independent channel fraction must be in [0, 1]")
    maximum = max(first, second)
    gap = abs(first - second)
    pooled = maximum + scale * math.log1p(
        independent_channel_fraction * math.exp(-gap / scale)
    )
    return _clip_intensity(pooled)


def _dominance_weighted_pleasantness(
    first_pleasantness: float,
    second_pleasantness: float,
    first_intensity: float,
    second_intensity: float,
) -> float:
    values = np.asarray(
        [first_pleasantness, second_pleasantness, first_intensity, second_intensity],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite pleasantness pooling input")
    maximum = max(first_intensity, second_intensity)
    weights = np.exp((np.asarray([first_intensity, second_intensity]) - maximum) / 2.0)
    result = float(
        np.dot(weights, [first_pleasantness, second_pleasantness]) / weights.sum()
    )
    return float(np.clip(result, 0.0, 10.0))


def _pair_predictions(
    target_rows: Sequence[Mapping[str, Any]],
    humanpom: Mapping[str, Mapping[str, np.ndarray]],
    projected_members: Sequence[Mapping[str, np.ndarray]],
    r2_head_states: Sequence[Mapping[str, np.ndarray]],
    r2_weights: np.ndarray,
    fechner_scale: float,
) -> list[dict[str, Any]]:
    morgan = _morgan_similarities(target_rows)
    result = []
    for first_index, second_index in combinations(range(len(target_rows)), 2):
        first = target_rows[first_index]
        second = target_rows[second_index]
        first_smiles = str(first["canonical_smiles"])
        second_smiles = str(second["canonical_smiles"])
        r2, member_values, disagreement = _r2_similarity(
            projected_members,
            r2_head_states,
            r2_weights,
            first_smiles,
            second_smiles,
        )
        morgan_value = float(morgan["function"](first_smiles, second_smiles))
        tracks: dict[str, dict[str, float]] = {}
        for track in ("primary", "strict", "fixed_rdkit"):
            intensity_a = float(humanpom["intensive"][track][first_index])
            intensity_b = float(humanpom["intensive"][track][second_index])
            pleasant_a = float(humanpom["pleasant"][track][first_index])
            pleasant_b = float(humanpom["pleasant"][track][second_index])
            tracks[track] = {
                "component_a_intensity": round(intensity_a, 8),
                "component_b_intensity": round(intensity_b, 8),
                "component_a_pleasantness": round(pleasant_a, 8),
                "component_b_pleasantness": round(pleasant_b, 8),
                "strongest_component": round(max(intensity_a, intensity_b), 8),
                "ravia_weber_fechner_pool": round(
                    _fechner_pool(intensity_a, intensity_b, fechner_scale), 8
                ),
                "r2_channel_overlap_pool": round(
                    _fechner_pool(
                        intensity_a,
                        intensity_b,
                        fechner_scale,
                        independent_channel_fraction=1.0 - r2,
                    ),
                    8,
                ),
                "morgan_channel_overlap_pool": round(
                    _fechner_pool(
                        intensity_a,
                        intensity_b,
                        fechner_scale,
                        independent_channel_fraction=1.0 - morgan_value,
                    ),
                    8,
                ),
                "root_sum_square": round(
                    _clip_intensity(math.hypot(intensity_a, intensity_b)), 8
                ),
                "complete_addition": round(
                    _clip_intensity(intensity_a + intensity_b), 8
                ),
                "dominance_weighted_pleasantness": round(
                    _dominance_weighted_pleasantness(
                        pleasant_a,
                        pleasant_b,
                        intensity_a,
                        intensity_b,
                    ),
                    8,
                ),
                "arithmetic_mean_pleasantness": round((pleasant_a + pleasant_b) / 2.0, 8),
            }
        result.append(
            {
                "pair_id": _pair_id(first["cas"], second["cas"]),
                "component_a": {
                    key: first[key]
                    for key in (
                        "cas",
                        "odorant",
                        "normalized_odorant",
                        "canonical_smiles",
                        "inchi_key",
                        "concentration_mg_ml",
                        "solvent",
                    )
                },
                "component_b": {
                    key: second[key]
                    for key in (
                        "cas",
                        "odorant",
                        "normalized_odorant",
                        "canonical_smiles",
                        "inchi_key",
                        "concentration_mg_ml",
                        "solvent",
                    )
                },
                "structure_similarity": {
                    "r2_ensemble": round(r2, 10),
                    "r2_members": [round(value, 10) for value in member_values],
                    "r2_member_disagreement": round(disagreement, 10),
                    "morgan_tanimoto": round(morgan_value, 10),
                },
                "end_to_end": tracks,
                "interaction_only_symbolic": {
                    "inputs_open_only_during_score": ["IA", "IB", "PA", "PB"],
                    "operator_contract_sha256": canonical_json_sha256(OPERATOR_CONTRACT),
                },
            }
        )
    if len(result) != ALL_PAIR_COUNT:
        raise RuntimeError(f"expected {ALL_PAIR_COUNT} all-pair predictions")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    outcome = args.outcome.resolve()
    assert_target_outcome_absent(outcome)
    predictions_path = args.predictions.resolve()
    if predictions_path.exists():
        raise RuntimeError("refusing to overwrite existing Ma blind predictions")
    target_rows, metadata_audit = _fetch_and_resolve_metadata(
        args.metadata.resolve()
    )
    if outcome.exists():
        raise RuntimeError("Ma outcome appeared during prepare")
    keller_paths = {
        "molecules.csv": args.keller_molecules,
        "stimuli.csv": args.keller_stimuli,
        "behavior.csv": args.keller_behavior,
    }
    keller_sources = _verify_keller_sources(keller_paths)
    humanpom, humanpom_audit = _humanpom_predictions(
        target_rows,
        keller_molecules=args.keller_molecules,
        keller_stimuli=args.keller_stimuli,
        keller_behavior=args.keller_behavior,
        molformer_root=args.molformer_root,
        hf_home=args.hf_home,
        threads=args.threads,
        batch_size=args.batch_size,
    )
    fechner_scale, fechner_audit = _ravia_fechner_scale()
    projected, r2_head_states, r2_weights, r2_audit = _load_r2_projected(target_rows)
    pair_rows = _pair_predictions(
        target_rows,
        humanpom,
        projected,
        r2_head_states,
        r2_weights,
        fechner_scale,
    )
    pair_hash = canonical_json_sha256(pair_rows)
    target_hash = canonical_json_sha256(target_rows)
    release_checks = {
        "target_outcome_absent": not outcome.exists(),
        "metadata_is_non_outcome_tabular_sheet": metadata_audit["contains_human_outcomes"] is False,
        "target_odor_count_exact": len(target_rows) == TARGET_ODOR_COUNT,
        "all_unordered_pairs_predicted": len(pair_rows) == ALL_PAIR_COUNT,
        "pubchem_structures_unique_and_complete": len(
            {row["canonical_smiles"] for row in target_rows}
        )
        == TARGET_ODOR_COUNT,
        "target_exact_human_label_leakage_zero": humanpom_audit[
            "target_exact_human_label_leakage_count"
        ]
        == 0,
        "strict_training_molecules_at_least_150": humanpom_audit[
            "ring_scaffold_and_exact_label_disjoint_training_molecules"
        ]
        >= 150,
        "keller_development_primary_beats_fixed_rdkit_by_0_01": humanpom_audit[
            "development"
        ]["primary_minus_fixed_baseline"]
        >= 0.01,
        "r2_ensemble_release_gate_verified": True,
        "ravia_fechner_scale_positive": fechner_scale > 0.0,
    }
    script = Path(__file__).resolve()
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "ma_2021_all_pair_predictions_ready_before_outcome_workbook",
        "blind_contract": {
            "target_outcome_workbook_read": False,
            "target_outcomes_used_for_training": False,
            "target_outcomes_used_for_model_selection": False,
            "target_pair_membership_used_for_model_selection": False,
            "all_72_choose_2_pairs_predicted": True,
            "metadata_trial_numbers_ignored_for_pair_selection": True,
            "target_rows_sha256": target_hash,
            "prediction_rows_sha256": pair_hash,
            "operator_contract_sha256": canonical_json_sha256(OPERATOR_CONTRACT),
        },
        "target_dataset": {
            "name": "Ma et al. 2021 binary food-odor mixtures",
            "doi": DATASET_DOI,
            "article_doi": ARTICLE_DOI,
            "related_intensity_article_doi": RELATED_INTENSITY_DOI,
            "license": "Etalab Open License 2.0, CC-BY-2.0 compatible",
            "dataverse_dataset_version_id": DATAVERSE_DATASET_VERSION_ID,
            "metadata": metadata_audit,
            "expected_outcome": {
                "filename": OUTCOME_FILE,
                "url": OUTCOME_URL,
                "bytes": OUTCOME_BYTES,
                "dataverse_file_id": DATAVERSE_FILE_ID,
                "dataverse_file_pid": DATAVERSE_FILE_PID,
                "downloaded_or_opened": False,
            },
            "odorants": TARGET_ODOR_COUNT,
            "all_possible_unordered_pairs": ALL_PAIR_COUNT,
            "published_trial_rows_expected": EXPECTED_TRIAL_ROWS,
            "distinct_mixtures_expected": EXPECTED_DISTINCT_MIXTURES,
            "individual_rows_expected": EXPECTED_INDIVIDUAL_ROWS,
        },
        "training": {
            "keller_sources": keller_sources,
            "humanpom": humanpom_audit,
            "ravia_fechner": fechner_audit,
            "r2_similarity": r2_audit,
        },
        "model": {
            "name": "HumanPOM + Ravia Weber-Fechner + R2 channel-overlap diagnostics",
            "primary_interaction_model": PRIMARY_INTERACTION_MODEL,
            "fixed_interaction_comparator": FIXED_INTERACTION_COMPARATOR,
            "operator_contract": OPERATOR_CONTRACT,
            "fechner_scale_target_units_per_natural_log": fechner_scale,
            "primary_end_to_end_track": "primary",
            "strict_sensitivity_track": "strict",
            "fixed_structure_comparator_track": "fixed_rdkit",
        },
        "implementation": {
            "script": str(script),
            "script_sha256": sha256_file(script),
            "shared_humanpom_script_sha256": sha256_file(Path(human.__file__).resolve()),
            "numpy_r2_script_sha256": sha256_file(
                PROJECT_ROOT / "fragrance_ai" / "recommender" / "numpy_r2.py"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "release_gate": {
            "passed": all(release_checks.values()),
            "checks": release_checks,
            "scope": "permission to seal all-pair predictions and symbolic operators only",
        },
        "target_odorants": target_rows,
        "predictions": pair_rows,
        "claim_boundary": (
            "Outcome-unopened binary-mixture intensity benchmark. Interaction-only "
            "scoring may use measured component ratings as declared covariates; the "
            "end-to-end track uses structure-only component predictions. This does "
            "not validate complex perfume recipes or 90% human olfactory similarity."
        ),
    }
    if not document["release_gate"]["passed"]:
        raise RuntimeError(f"Ma prediction release gate failed: {release_checks}")
    write_json(predictions_path, document)
    return document


def create_seal(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve(strict=True)
    seal_path = args.seal.resolve()
    outcome = args.outcome.resolve()
    assert_target_outcome_absent(outcome)
    if seal_path.exists():
        raise RuntimeError("refusing to overwrite an existing Ma prediction seal")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if predictions.get("status") != (
        "ma_2021_all_pair_predictions_ready_before_outcome_workbook"
    ):
        raise RuntimeError("Ma predictions are not sealable")
    if predictions.get("release_gate", {}).get("passed") is not True:
        raise RuntimeError("Ma prediction release gate is closed")
    rows = predictions.get("predictions", [])
    row_hash = canonical_json_sha256(rows)
    if row_hash != predictions.get("blind_contract", {}).get(
        "prediction_rows_sha256"
    ):
        raise RuntimeError("Ma prediction rows changed before seal")
    script_hash = sha256_file(Path(__file__).resolve())
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("Ma benchmark implementation changed before seal")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": utc_now(),
        "prediction_file": predictions_path.name,
        "prediction_file_sha256": sha256_file(predictions_path),
        "prediction_file_bytes": predictions_path.stat().st_size,
        "prediction_rows_sha256": row_hash,
        "benchmark_script_sha256": script_hash,
        "shared_humanpom_script_sha256": predictions["implementation"][
            "shared_humanpom_script_sha256"
        ],
        "operator_contract_sha256": canonical_json_sha256(OPERATOR_CONTRACT),
        "target_outcome": {
            "filename": OUTCOME_FILE,
            "url": OUTCOME_URL,
            "expected_bytes": OUTCOME_BYTES,
            "dataverse_file_id": DATAVERSE_FILE_ID,
            "dataverse_file_pid": DATAVERSE_FILE_PID,
            "dataset_version_id": DATAVERSE_DATASET_VERSION_ID,
            "path": str(outcome),
            "present_before_seal": False,
        },
        "scoring_contract": {
            "population": "individual data excluding subject 47, matching official mean sheet",
            "primary_unit": "198 distinct unordered binary mixtures after repeat collapse",
            "secondary_unit": "222 trial rows",
            "primary_endpoint": "IAB whole-mixture intensity",
            "primary_model": PRIMARY_INTERACTION_MODEL,
            "fixed_comparator": FIXED_INTERACTION_COMPARATOR,
            "metrics": ["MAE", "RMSE", "bias", "Spearman"],
            "bootstrap": (
                "participant-cluster plus distinct-mixture bootstrap on individual rows"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "integration_gate": {
                "primary_mae_below_1_0_on_0_10_scale": True,
                "primary_mae_lower_than_strongest_component": True,
                "primary_rmse_lower_than_strongest_component": True,
                "bootstrap_mae_gain_lower_above_zero": True,
                "bootstrap_spearman_gain_lower_at_least_minus_0_02": True,
            },
        },
    }
    write_json(seal_path, seal)
    return seal


def verify_prediction_seal(predictions_path: Path, seal_path: Path) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported Ma seal schema")
    if seal.get("prediction_file_sha256") != sha256_file(predictions_path):
        raise RuntimeError("Ma sealed prediction hash mismatch")
    if seal.get("prediction_file_bytes") != predictions_path.stat().st_size:
        raise RuntimeError("Ma sealed prediction size mismatch")
    rows_hash = canonical_json_sha256(predictions.get("predictions", []))
    if rows_hash != seal.get("prediction_rows_sha256"):
        raise RuntimeError("Ma sealed prediction rows mismatch")
    if rows_hash != predictions.get("blind_contract", {}).get("prediction_rows_sha256"):
        raise RuntimeError("Ma prediction artifact row binding mismatch")
    script_hash = sha256_file(Path(__file__).resolve())
    if script_hash != seal.get("benchmark_script_sha256"):
        raise RuntimeError("Ma benchmark script changed after seal")
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("Ma prediction implementation binding mismatch")
    if seal.get("operator_contract_sha256") != canonical_json_sha256(OPERATOR_CONTRACT):
        raise RuntimeError("Ma operator contract changed after seal")
    return {"predictions": predictions, "seal": seal}


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    verified = verify_prediction_seal(args.predictions, args.seal)
    outcome = args.outcome.resolve()
    assert_target_outcome_absent(outcome)
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite an existing Ma acquisition receipt")
    timestamp = human.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if not timestamp.get("verified"):
        raise RuntimeError("Ma prediction seal timestamp is not verified")
    target_contract = verified["seal"]["target_outcome"]
    if target_contract.get("url") != OUTCOME_URL or target_contract.get(
        "expected_bytes"
    ) != OUTCOME_BYTES:
        raise RuntimeError("Ma sealed outcome contract changed")
    started = utc_now()
    raw = _download_bytes(OUTCOME_URL)
    if len(raw) != OUTCOME_BYTES:
        raise RuntimeError(
            f"Ma workbook size changed: {len(raw)} != {OUTCOME_BYTES}"
        )
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise RuntimeError("Ma original outcome is not a valid XLSX container")
    outcome.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=outcome.parent, prefix=outcome.name + ".", delete=False
    ) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    os.replace(temporary, outcome)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "ma_2021_outcome_acquired_after_verified_timestamp",
        "download_started_at": started,
        "download_completed_at": utc_now(),
        "prediction_file_sha256": sha256_file(args.predictions.resolve(strict=True)),
        "seal_file_sha256": sha256_file(args.seal.resolve(strict=True)),
        "outcome": {
            "path": str(outcome),
            "url": OUTCOME_URL,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "md5": md5_bytes(raw),
            "dataverse_file_id": DATAVERSE_FILE_ID,
            "dataverse_file_pid": DATAVERSE_FILE_PID,
        },
        "timestamp": timestamp,
    }
    write_json(receipt_path, receipt)
    return receipt


def _canonical_columns(frame: Any, required: Sequence[str]) -> Any:
    normalized = {normalize_name(column): column for column in frame.columns}
    mapping = {}
    for name in required:
        key = normalize_name(name)
        if key not in normalized:
            raise RuntimeError(f"Ma outcome column is missing: {name}")
        mapping[normalized[key]] = name
    return frame.rename(columns=mapping)[list(required)].copy()


def _load_outcome(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    import pandas as pd

    workbook = pd.ExcelFile(path, engine="openpyxl")
    normalized_sheets = {normalize_name(name): name for name in workbook.sheet_names}
    mean_key = normalize_name("mean value after deleting sub47")
    individual_key = normalize_name("individual data")
    if mean_key not in normalized_sheets or individual_key not in normalized_sheets:
        raise RuntimeError(f"Ma workbook sheets changed: {workbook.sheet_names}")
    mean = pd.read_excel(workbook, sheet_name=normalized_sheets[mean_key])
    individual = pd.read_excel(workbook, sheet_name=normalized_sheets[individual_key])
    endpoints = ["IA", "IAmix", "IB", "IBmix", "IAB", "PA", "PB", "PAB"]
    mean = _canonical_columns(
        mean,
        ["Trial", "R-Trial", "Repeat", "Group", "odor A", "odor B", *endpoints],
    )
    individual = _canonical_columns(
        individual,
        ["Sub", "Trial", "R-Trial", "Repeat", "odor A", "odor B", *endpoints],
    )
    if len(mean) != EXPECTED_TRIAL_ROWS or len(individual) != EXPECTED_INDIVIDUAL_ROWS:
        raise RuntimeError(
            f"Ma outcome row counts changed: mean={len(mean)}, individual={len(individual)}"
        )
    for frame in (mean, individual):
        for column in ("Trial", "R-Trial", "Repeat", *endpoints):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if mean["Trial"].isna().any() or individual["Trial"].isna().any():
        raise RuntimeError("Ma trial identifiers are not numeric")
    required_numeric = ["IA", "IB", "IAB", "PA", "PB", "PAB"]
    if mean[required_numeric].isna().any().any() or individual[
        required_numeric
    ].isna().any().any():
        raise RuntimeError("Ma primary outcome fields contain missing values")
    subject_normalized = individual["Sub"].map(normalize_name)
    outlier_mask = subject_normalized.isin({"47", "sub47", "subject47"})
    outlier_rows = int(outlier_mask.sum())
    if outlier_rows == 0:
        raise RuntimeError("Ma subject 47 was not found in individual data")
    included = individual.loc[~outlier_mask].copy()
    included["subject_id"] = subject_normalized.loc[~outlier_mask].to_numpy()

    individual_means = included.groupby("Trial", sort=True)[required_numeric].mean()
    official = mean.set_index("Trial")[required_numeric].sort_index()
    common = official.index.intersection(individual_means.index)
    if len(common) != EXPECTED_TRIAL_ROWS:
        raise RuntimeError("Ma mean/individual trial coverage differs")
    maximum_difference = float(
        np.max(np.abs(official.loc[common].to_numpy() - individual_means.loc[common].to_numpy()))
    )
    if maximum_difference > 0.011:
        raise RuntimeError(
            f"Ma official mean sheet differs from subject-47-excluded replay: {maximum_difference}"
        )
    audit = {
        "sheet_names": workbook.sheet_names,
        "mean_rows": len(mean),
        "raw_individual_rows": len(individual),
        "subject_47_rows_excluded": outlier_rows,
        "included_individual_rows": len(included),
        "included_subjects": int(included["subject_id"].nunique()),
        "official_mean_replay_maximum_absolute_difference": maximum_difference,
    }
    return mean, included, audit


def _metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(target, dtype=float)
    if predicted.shape != observed.shape or len(predicted) < 3:
        raise RuntimeError("invalid Ma metric arrays")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise RuntimeError("non-finite Ma metric input")
    error = predicted - observed
    return {
        "spearman": human.spearman(predicted, observed),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _apply_operators(
    first: float,
    second: float,
    *,
    scale: float,
    r2_similarity: float,
    morgan_similarity: float,
) -> dict[str, float]:
    return {
        "strongest_component": _clip_intensity(max(first, second)),
        "ravia_weber_fechner_pool": _fechner_pool(first, second, scale),
        "r2_channel_overlap_pool": _fechner_pool(
            first,
            second,
            scale,
            independent_channel_fraction=1.0 - r2_similarity,
        ),
        "morgan_channel_overlap_pool": _fechner_pool(
            first,
            second,
            scale,
            independent_channel_fraction=1.0 - morgan_similarity,
        ),
        "root_sum_square": _clip_intensity(math.hypot(first, second)),
        "complete_addition": _clip_intensity(first + second),
    }


def _scored_rows(
    individual: Any,
    predictions: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_rows = predictions["target_odorants"]
    name_to_cas = {
        str(row["normalized_odorant"]): str(row["cas"]) for row in target_rows
    }
    pair_lookup = {row["pair_id"]: row for row in predictions["predictions"]}
    scale = float(predictions["model"]["fechner_scale_target_units_per_natural_log"])
    result = []
    unknown: set[str] = set()
    for _, source in individual.iterrows():
        name_a = normalize_name(source["odor A"])
        name_b = normalize_name(source["odor B"])
        cas_a = name_to_cas.get(name_a)
        cas_b = name_to_cas.get(name_b)
        if cas_a is None:
            unknown.add(str(source["odor A"]))
        if cas_b is None:
            unknown.add(str(source["odor B"]))
        if cas_a is None or cas_b is None:
            continue
        pair_id = _pair_id(cas_a, cas_b)
        pair = pair_lookup.get(pair_id)
        if pair is None:
            raise RuntimeError(f"Ma observed pair was not prospectively predicted: {pair_id}")
        structure = pair["structure_similarity"]
        operators = _apply_operators(
            float(source["IA"]),
            float(source["IB"]),
            scale=scale,
            r2_similarity=float(structure["r2_ensemble"]),
            morgan_similarity=float(structure["morgan_tanimoto"]),
        )
        end_to_end = pair["end_to_end"]["primary"]
        result.append(
            {
                "subject_id": str(source["subject_id"]),
                "trial": int(source["Trial"]),
                "pair_id": pair_id,
                "target_iab": float(source["IAB"]),
                "target_pab": float(source["PAB"]),
                **{f"interaction::{key}": value for key, value in operators.items()},
                "interaction::dominance_weighted_pleasantness": (
                    _dominance_weighted_pleasantness(
                        float(source["PA"]),
                        float(source["PB"]),
                        float(source["IA"]),
                        float(source["IB"]),
                    )
                ),
                "interaction::arithmetic_mean_pleasantness": (
                    float(source["PA"] + source["PB"]) / 2.0
                ),
                **{
                    f"end_to_end::{key}": float(end_to_end[key])
                    for key in (
                        "strongest_component",
                        "ravia_weber_fechner_pool",
                        "r2_channel_overlap_pool",
                        "morgan_channel_overlap_pool",
                        "root_sum_square",
                        "complete_addition",
                        "dominance_weighted_pleasantness",
                        "arithmetic_mean_pleasantness",
                    )
                },
            }
        )
    if unknown:
        raise RuntimeError(f"Ma outcome odor names did not map to metadata: {sorted(unknown)}")
    if len(result) != len(individual):
        raise RuntimeError("not every Ma individual outcome row was scored")
    observed_pair_ids = {row["pair_id"] for row in result}
    return result, {
        "observed_pair_count": len(observed_pair_ids),
        "unobserved_predeclared_pair_count": ALL_PAIR_COUNT - len(observed_pair_ids),
        "all_observed_pairs_were_predeclared": True,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], unit: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["pair_id"] if unit == "pair" else int(row["trial"])
        groups[key].append(row)
    result = []
    numeric_columns = [
        key
        for key in rows[0]
        if key.startswith("interaction::") or key.startswith("end_to_end::")
    ]
    for key in sorted(groups, key=str):
        values = groups[key]
        result.append(
            {
                "unit_id": str(key),
                "target_iab": float(np.mean([row["target_iab"] for row in values])),
                "target_pab": float(np.mean([row["target_pab"] for row in values])),
                **{
                    column: float(np.mean([row[column] for row in values]))
                    for column in numeric_columns
                },
            }
        )
    return result


def _result_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    target_iab = [row["target_iab"] for row in rows]
    target_pab = [row["target_pab"] for row in rows]
    result = {}
    for prefix in ("interaction", "end_to_end"):
        for name in (
            "strongest_component",
            "ravia_weber_fechner_pool",
            "r2_channel_overlap_pool",
            "morgan_channel_overlap_pool",
            "root_sum_square",
            "complete_addition",
        ):
            key = f"{prefix}::{name}"
            result[key] = _metrics([row[key] for row in rows], target_iab)
        for name in (
            "dominance_weighted_pleasantness",
            "arithmetic_mean_pleasantness",
        ):
            key = f"{prefix}::{name}"
            result[key] = _metrics([row[key] for row in rows], target_pab)
    return result


def _bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    participants = sorted({str(row["subject_id"]) for row in rows})
    pairs = sorted({str(row["pair_id"]) for row in rows})
    participant_index = {value: index for index, value in enumerate(participants)}
    pair_index = {value: index for index, value in enumerate(pairs)}
    shape = (len(participants), len(pairs))
    buckets: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (participant_index[str(row["subject_id"])], pair_index[str(row["pair_id"])])
        ].append(row)
    target = np.full(shape, np.nan, dtype=float)
    primary = np.full(shape, np.nan, dtype=float)
    baseline = np.full(shape, np.nan, dtype=float)
    for (participant, pair), values in buckets.items():
        target[participant, pair] = float(np.mean([row["target_iab"] for row in values]))
        primary[participant, pair] = float(
            np.mean(
                [row[f"interaction::{PRIMARY_INTERACTION_MODEL}"] for row in values]
            )
        )
        baseline[participant, pair] = float(
            np.mean(
                [row[f"interaction::{FIXED_INTERACTION_COMPARATOR}"] for row in values]
            )
        )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    mae_gain = []
    spearman_gain = []
    valid_pairs = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_participants = generator.integers(0, len(participants), len(participants))
        sampled_pairs = generator.integers(0, len(pairs), len(pairs))
        selected_target = target[sampled_participants][:, sampled_pairs]
        selected_primary = primary[sampled_participants][:, sampled_pairs]
        selected_baseline = baseline[sampled_participants][:, sampled_pairs]
        with np.errstate(invalid="ignore"):
            mean_target = np.nanmean(selected_target, axis=0)
            mean_primary = np.nanmean(selected_primary, axis=0)
            mean_baseline = np.nanmean(selected_baseline, axis=0)
        valid = np.isfinite(mean_target) & np.isfinite(mean_primary) & np.isfinite(mean_baseline)
        if int(valid.sum()) < max(20, len(pairs) // 2):
            continue
        observed = mean_target[valid]
        primary_values = mean_primary[valid]
        baseline_values = mean_baseline[valid]
        mae_gain.append(
            float(
                np.mean(np.abs(baseline_values - observed))
                - np.mean(np.abs(primary_values - observed))
            )
        )
        spearman_gain.append(
            human.spearman(primary_values, observed)
            - human.spearman(baseline_values, observed)
        )
        valid_pairs.append(int(valid.sum()))
    if len(mae_gain) < int(BOOTSTRAP_DRAWS * 0.98):
        raise RuntimeError("too many invalid Ma bootstrap draws")
    return {
        "seed": BOOTSTRAP_SEED,
        "draws_requested": BOOTSTRAP_DRAWS,
        "draws_valid": len(mae_gain),
        "participants": len(participants),
        "distinct_mixtures": len(pairs),
        "median_valid_mixtures_per_draw": float(np.median(valid_pairs)),
        "strongest_minus_primary_mae_95_interval": [
            float(value) for value in np.quantile(mae_gain, [0.025, 0.975])
        ],
        "primary_minus_strongest_spearman_95_interval": [
            float(value) for value in np.quantile(spearman_gain, [0.025, 0.975])
        ],
    }


def _repeatability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject_id"]), str(row["pair_id"]))].append(row)
    first = []
    second = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: int(row["trial"]))
        if len(ordered) >= 2:
            first.append(float(ordered[0]["target_iab"]))
            second.append(float(ordered[1]["target_iab"]))
    if len(first) < 100:
        raise RuntimeError("too few participant-level Ma repeat pairs")
    return {
        "participant_pair_repeats": len(first),
        "spearman": human.spearman(first, second),
        "mae": float(np.mean(np.abs(np.asarray(first) - np.asarray(second)))),
        "rmse": float(
            np.sqrt(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    unique = report["distinct_mixture_results"]
    primary = unique[f"interaction::{PRIMARY_INTERACTION_MODEL}"]
    baseline = unique[f"interaction::{FIXED_INTERACTION_COMPARATOR}"]
    bootstrap = report["bootstrap"]
    lines = [
        "# Ma 2021 이성분 혼합물 블라인드 벤치마크",
        "",
        f"- 상태: **{report['status']}**",
        f"- 고유 혼합물: **{report['population']['distinct_mixtures']}**",
        f"- 포함 평가자: **{report['population']['participants']}**",
        f"- 개별 평가: **{report['population']['individual_rows']}**",
        "",
        "| 모델 | Spearman | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Ravia Weber–Fechner pool | {primary['spearman']:.4f} | "
            f"{primary['mae']:.4f} | {primary['rmse']:.4f} | {primary['bias']:+.4f} |"
        ),
        (
            f"| Strongest component | {baseline['spearman']:.4f} | "
            f"{baseline['mae']:.4f} | {baseline['rmse']:.4f} | {baseline['bias']:+.4f} |"
        ),
        "",
        (
            "- MAE 개선 95% 구간: "
            f"[{bootstrap['strongest_minus_primary_mae_95_interval'][0]:+.4f}, "
            f"{bootstrap['strongest_minus_primary_mae_95_interval'][1]:+.4f}]"
        ),
        (
            "- 런타임 통합 게이트: **"
            + ("PASS" if report["mixture_operator_integration_gate"]["passed"] else "FAIL")
            + "**"
        ),
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def score(args: argparse.Namespace) -> dict[str, Any]:
    verified = verify_prediction_seal(args.predictions, args.seal)
    predictions = verified["predictions"]
    receipt_path = args.receipt.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("prediction_file_sha256") != sha256_file(
        args.predictions.resolve(strict=True)
    ):
        raise RuntimeError("Ma receipt/prediction binding mismatch")
    if receipt.get("seal_file_sha256") != sha256_file(args.seal.resolve(strict=True)):
        raise RuntimeError("Ma receipt/seal binding mismatch")
    outcome = args.outcome.resolve(strict=True)
    if receipt.get("outcome", {}).get("sha256") != sha256_file(outcome):
        raise RuntimeError("Ma outcome differs from acquisition receipt")
    if receipt.get("outcome", {}).get("bytes") != outcome.stat().st_size:
        raise RuntimeError("Ma outcome size differs from acquisition receipt")
    timestamp = human.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if timestamp.get("response_sha256") != receipt.get("timestamp", {}).get(
        "response_sha256"
    ):
        raise RuntimeError("Ma timestamp changed after outcome acquisition")
    _, individual, outcome_audit = _load_outcome(outcome)
    rows, mapping_audit = _scored_rows(individual, predictions)
    pair_rows = _aggregate(rows, "pair")
    trial_rows = _aggregate(rows, "trial")
    if len(pair_rows) != EXPECTED_DISTINCT_MIXTURES or len(trial_rows) != EXPECTED_TRIAL_ROWS:
        raise RuntimeError(
            f"Ma scored units changed: pairs={len(pair_rows)}, trials={len(trial_rows)}"
        )
    pair_results = _result_metrics(pair_rows)
    trial_results = _result_metrics(trial_rows)
    bootstrap = _bootstrap(rows)
    repeatability = _repeatability(rows)
    primary = pair_results[f"interaction::{PRIMARY_INTERACTION_MODEL}"]
    baseline = pair_results[f"interaction::{FIXED_INTERACTION_COMPARATOR}"]
    checks = {
        "timestamp_and_receipt_verified": True,
        "all_observed_pairs_predeclared": mapping_audit[
            "all_observed_pairs_were_predeclared"
        ],
        "distinct_mixture_count_exact": len(pair_rows) == EXPECTED_DISTINCT_MIXTURES,
        "official_mean_replay_within_0_011": outcome_audit[
            "official_mean_replay_maximum_absolute_difference"
        ]
        <= 0.011,
        "primary_mae_below_1_0_on_0_10_scale": primary["mae"] < 1.0,
        "primary_mae_lower_than_strongest_component": primary["mae"] < baseline["mae"],
        "primary_rmse_lower_than_strongest_component": primary["rmse"] < baseline["rmse"],
        "bootstrap_mae_gain_lower_above_zero": bootstrap[
            "strongest_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "bootstrap_spearman_gain_lower_at_least_minus_0_02": bootstrap[
            "primary_minus_strongest_spearman_95_interval"
        ][0]
        >= -0.02,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": (
            "ma_2021_blind_mixture_operator_gate_passed"
            if all(checks.values())
            else "ma_2021_blind_mixture_operator_gate_failed"
        ),
        "blind_integrity": {
            "prediction_seal_verified": True,
            "timestamp": timestamp,
            "acquisition_receipt_sha256": sha256_file(receipt_path),
            "outcome_sha256": sha256_file(outcome),
            "outcome_download_started_after_timestamp": True,
            "all_2556_pair_predictions_preceded_outcome": True,
        },
        "source_binding": {
            "prediction_sha256": sha256_file(args.predictions.resolve(strict=True)),
            "seal_sha256": sha256_file(args.seal.resolve(strict=True)),
            "receipt_sha256": sha256_file(receipt_path),
            "outcome_sha256": sha256_file(outcome),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "population": {
            "participants": outcome_audit["included_subjects"],
            "individual_rows": len(rows),
            "trial_rows": len(trial_rows),
            "distinct_mixtures": len(pair_rows),
            "prospectively_predicted_pairs": ALL_PAIR_COUNT,
        },
        "outcome_parser_audit": outcome_audit,
        "pair_mapping_audit": mapping_audit,
        "repeatability": repeatability,
        "distinct_mixture_results": pair_results,
        "trial_results": trial_results,
        "bootstrap": bootstrap,
        "mixture_operator_integration_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "authorized_scope_if_passed": (
                "binary whole-mixture intensity pooling after component intensities "
                "are available; no authority for component-intensity prediction, "
                "complex recipes, headspace, or odor-quality similarity"
            ),
        },
        "human_olfactory_90_percent_certified": False,
        "complex_perfume_recipe_validated": False,
        "claim_boundary": (
            "This is a genuinely outcome-unopened public-human binary-mixture "
            "intensity test. A passing operator gate validates only the declared "
            "0-10 binary pooling task. It is not direct validation of generated "
            "perfume recipes and does not establish 90% human olfactory similarity."
        ),
    }
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite Ma benchmark outputs")
    write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--metadata", type=Path, required=True)
    prepare_parser.add_argument("--outcome", type=Path, required=True)
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--keller-molecules", type=Path, required=True)
    prepare_parser.add_argument("--keller-stimuli", type=Path, required=True)
    prepare_parser.add_argument("--keller-behavior", type=Path, required=True)
    prepare_parser.add_argument("--molformer-root", type=Path, required=True)
    prepare_parser.add_argument("--hf-home", type=Path, required=True)
    prepare_parser.add_argument("--threads", type=int, default=4)
    prepare_parser.add_argument("--batch-size", type=int, default=32)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--predictions", type=Path, required=True)
    seal_parser.add_argument("--seal", type=Path, required=True)
    seal_parser.add_argument("--outcome", type=Path, required=True)

    for name in ("acquire", "score"):
        item = subparsers.add_parser(name)
        item.add_argument("--predictions", type=Path, required=True)
        item.add_argument("--seal", type=Path, required=True)
        item.add_argument("--outcome", type=Path, required=True)
        item.add_argument("--receipt", type=Path, required=True)
        item.add_argument("--openssl", type=Path, required=True)
        item.add_argument("--timestamp-response", type=Path, required=True)
        item.add_argument("--timestamp-ca", type=Path, required=True)
        item.add_argument("--timestamp-tsa", type=Path, required=True)
    score_parser = subparsers.choices["score"]
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
        summary = {
            "status": result["status"],
            "pairs": len(result["predictions"]),
            "release_gate": result["release_gate"],
        }
    elif args.command == "seal":
        result = create_seal(args)
        summary = {
            "sealed_at": result["sealed_at"],
            "prediction_sha256": result["prediction_file_sha256"],
        }
    elif args.command == "acquire":
        result = acquire(args)
        summary = {
            "status": result["status"],
            "outcome_sha256": result["outcome"]["sha256"],
        }
    else:
        result = score(args)
        summary = {
            "status": result["status"],
            "integration_gate": result["mixture_operator_integration_gate"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
