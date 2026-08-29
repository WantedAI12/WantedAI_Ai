#!/usr/bin/env python
"""Outcome-unopened Minnesota odor-intensity matching benchmark.

Five target structures are excluded from Keller/Ravia/Bierling before source-
only candidate selection and fitting. Curves over a frozen ppm grid and match
concentrations relative to 5 ppm acetylpropionyl are written before any of the
three raw outcome CSVs are downloaded. The seal is externally timestamped;
acquire verifies repository MD5s; score estimates participant match points on
the 150 mm scale without changing the frozen curves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    bemis_murcko_scaffold,
    build_raw_descriptor_cache,
    canonical_smiles,
)
from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_ma_2021_binary_mixture_benchmark as ma  # noqa: E402
from scripts import build_universal_intensity_model_v1 as universal  # noqa: E402


SCHEMA_VERSION = "1.0"
DATASET_DOI = "10.13020/D68591"
ARTICLE_DOI = "10.1111/joss.12460"
ITEM_HANDLE = "11299/182551"
ITEM_UUID = "7fc0dbdf-d49c-4c20-be9b-4d0c89474666"
STANDARD_CODE = "AP"
STANDARD_PPM = 5.0
LINE_CENTER_MM = 75.0
GRID_LOG10_PPM = np.linspace(-4.0, 4.0, 321)
GRID_PPM = np.power(10.0, GRID_LOG10_PPM)
BOOTSTRAP_SEED = 20_260_902
BOOTSTRAP_DRAWS = 10_000

COMPOUNDS: tuple[dict[str, Any], ...] = (
    {
        "code": "BA",
        "name": "butyric acid",
        "cid": 264,
        "canonical_smiles": "CCCC(=O)O",
        "inchi_key": "FERIUCNNQQJTOY-UHFFFAOYSA-N",
    },
    {
        "code": "DD",
        "name": "delta-decalactone",
        "cid": 12_810,
        "canonical_smiles": "CCCCCC1CCCC(=O)O1",
        "inchi_key": "GHBSPIPJMLAMEP-UHFFFAOYSA-N",
    },
    {
        "code": "F",
        "name": "furaneol",
        "cid": 19_309,
        "canonical_smiles": "CC1C(=O)C(=C(O1)C)O",
        "inchi_key": "INAXVXBDKKUCGI-UHFFFAOYSA-N",
    },
    {
        "code": "M",
        "name": "methional",
        "cid": 18_635,
        "canonical_smiles": "CSCCC=O",
        "inchi_key": "CLUWOWRTHNNBBU-UHFFFAOYSA-N",
    },
    {
        "code": STANDARD_CODE,
        "name": "acetylpropionyl (2,3-pentanedione)",
        "cid": 11_747,
        "canonical_smiles": "CCC(=O)C(=O)C",
        "inchi_key": "TZMFJUDUGYTVRY-UHFFFAOYSA-N",
    },
)

FILES: tuple[dict[str, Any], ...] = (
    {
        "key": "original",
        "uuid": "f3d28e6b-c4dc-4747-8e1f-63e1d5d1f2cb",
        "filename": "Intensity Matching-raw data.csv",
        "md5": "7164354617f22911c5d41a23a1485ad1",
        "expected_rows": 820,
    },
    {
        "key": "retest",
        "uuid": "3ba1a5b4-6004-4fa8-b4ea-72f4b59d83ee",
        "filename": "Intensity Matching-raw data-retest.csv",
        "md5": "d9ffe801575307b901386b06cbb42f88",
        "expected_rows": 360,
    },
    {
        "key": "new_recruits",
        "uuid": "ad98c0c9-34ea-4bce-8b9d-c4d32e87aeb9",
        "filename": "Intensity Matching-raw data-new recruits.csv",
        "md5": "8be2acad0d7687c2bae9e0e27e71dd42",
        "expected_rows": 560,
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _md5_bytes(value: bytes) -> str:
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    shared.write_json(path, value)


def _outcome_paths(directory: Path) -> dict[str, Path]:
    return {str(row["key"]): directory / str(row["filename"]) for row in FILES}


def assert_outcomes_absent(directory: Path) -> None:
    present = [str(path) for path in _outcome_paths(directory.resolve()).values() if path.exists()]
    if present:
        raise RuntimeError("Minnesota outcomes exist before permitted acquisition: " + ", ".join(present))


def _source_training(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keller, keller_audit = universal._load_keller(args)
    ravia, ravia_audit = universal._load_ravia(args.ravia_root.resolve(strict=True))
    bierling, bierling_audit = universal._load_bierling(args)
    target_smiles = {canonical_smiles(str(row["canonical_smiles"])) for row in COMPOUNDS}
    raw = [*keller, *ravia, *bierling]
    overlap = [row for row in raw if row["canonical_smiles"] in target_smiles]
    retained = [row for row in raw if row["canonical_smiles"] not in target_smiles]
    if {row["canonical_smiles"] for row in retained} & target_smiles:
        raise RuntimeError("Minnesota target molecule leaked into source training")
    return retained, {
        "keller": keller_audit,
        "ravia": ravia_audit,
        "bierling": bierling_audit,
        "target_exact_overlap_rows_excluded": len(overlap),
        "target_exact_overlap_molecules_excluded": len(
            {row["canonical_smiles"] for row in overlap}
        ),
        "retained_rows": len(retained),
        "retained_molecules": len({row["canonical_smiles"] for row in retained}),
    }


def _humanpom(
    args: argparse.Namespace, target_rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions, audit = ma._humanpom_predictions(
        target_rows,
        keller_molecules=args.keller_molecules,
        keller_stimuli=args.keller_stimuli,
        keller_behavior=args.keller_behavior,
        molformer_root=args.molformer_root,
        hf_home=args.hf_home,
        threads=args.threads,
        batch_size=args.batch_size,
    )
    return np.asarray(predictions["intensive"]["primary"], dtype=float) / 10.0, audit


def _monotonic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("curve must be a finite vector")
    return np.maximum.accumulate(np.clip(values, 0.0, 1.0))


def _interpolate_curve(log_ppm: float, curve: np.ndarray) -> float:
    return float(np.interp(log_ppm, GRID_LOG10_PPM, curve))


def _match_log_ppm(curve: np.ndarray, target_score: float) -> tuple[float, str]:
    curve = _monotonic(curve)
    if target_score <= curve[0]:
        return float(GRID_LOG10_PPM[0]), "lower_grid_censored"
    if target_score >= curve[-1]:
        return float(GRID_LOG10_PPM[-1]), "upper_grid_censored"
    upper = int(np.searchsorted(curve, target_score, side="left"))
    lower = max(0, upper - 1)
    y0, y1 = float(curve[lower]), float(curve[upper])
    x0, x1 = float(GRID_LOG10_PPM[lower]), float(GRID_LOG10_PPM[upper])
    if y1 <= y0 + 1e-12:
        return x1, "flat_segment"
    fraction = (target_score - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0), "interpolated"


def _curves(
    args: argparse.Namespace,
    training_raw: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any], dict[str, Any]]:
    target_rows = [
        {
            **row,
            "canonical_smiles": canonical_smiles(str(row["canonical_smiles"])),
            "murcko_scaffold": bemis_murcko_scaffold(str(row["canonical_smiles"])),
        }
        for row in COMPOUNDS
    ]
    all_smiles = sorted(
        {str(row["canonical_smiles"]) for row in [*training_raw, *target_rows]}
    )
    cache = build_raw_descriptor_cache(all_smiles)
    training = universal._prepare_rows(training_raw, cache)
    portable_candidates = [row for row in universal.CANDIDATES if row["portable"]]
    _, cv_metrics = universal._repeated_molecule_cv(training, portable_candidates)
    transfer = universal._source_transfer(training, portable_candidates)
    selected = universal._select_candidate(
        cv_metrics, transfer, portable_candidates
    )
    selected_candidate = next(row for row in portable_candidates if row["name"] == selected)

    grid_rows = []
    for log_ppm, ppm in zip(GRID_LOG10_PPM, GRID_PPM, strict=True):
        for target in target_rows:
            grid_rows.append(
                {
                    "source": "minnesota_2016_preoutcome_grid",
                    "canonical_smiles": str(target["canonical_smiles"]),
                    "concentration_fraction": float(ppm / 1_000_000.0),
                    "target": 0.0,
                    "condition_id": f"{target['code']}:{log_ppm:.6f}",
                }
            )
    prepared_grid = universal._prepare_rows(grid_rows, cache)
    raw_grid, _ = universal._fit_predict(
        selected_candidate, training, prepared_grid
    )
    _, parameters = universal._fit_predict(selected_candidate, training, training)
    parity = float(
        np.max(
            np.abs(
                universal._portable_predict(parameters, training)
                - universal._fit_predict(selected_candidate, training, training)[0]
            )
        )
    )
    if parity > 1e-10:
        raise RuntimeError("Minnesota source model portable parity failed")
    parameters["portable_parity_maximum_absolute_error"] = parity
    raw_matrix = raw_grid.reshape(len(GRID_LOG10_PPM), len(target_rows))
    humanpom, humanpom_audit = _humanpom(args, target_rows)
    centered_humanpom = humanpom - float(humanpom.mean())
    hybrid_matrix = np.empty_like(raw_matrix)
    for index, row in enumerate(raw_matrix):
        hybrid_matrix[index] = (
            float(row.mean())
            + 0.5 * centered_humanpom
            + 0.5 * (row - float(row.mean()))
        )
    hybrid_matrix = np.clip(hybrid_matrix, 0.0, 1.0)
    curves: dict[str, dict[str, list[float]]] = {}
    for compound_index, target in enumerate(target_rows):
        curves[str(target["code"])] = {
            "universal_raw": _monotonic(raw_matrix[:, compound_index]).tolist(),
            "concentration_preserving_equal_hybrid": _monotonic(
                hybrid_matrix[:, compound_index]
            ).tolist(),
        }
    model_audit = {
        "selected_candidate": selected,
        "selected_cv_metrics": cv_metrics[selected],
        "source_transfer": {
            source: transfer[source]["candidates"][selected]
            for source in universal.TARGET_SOURCE_NAMES
        },
        "selection_used_minnesota_outcomes": False,
        "parameters": parameters,
    }
    return curves, model_audit, humanpom_audit


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve()
    outcome_dir = args.outcome_dir.resolve()
    assert_outcomes_absent(outcome_dir)
    if predictions_path.exists():
        raise RuntimeError("refusing to overwrite Minnesota blind predictions")
    training_raw, source_audit = _source_training(args)
    curves, model_audit, humanpom_audit = _curves(args, training_raw)
    standard_curve = np.asarray(
        curves[STANDARD_CODE]["concentration_preserving_equal_hybrid"], dtype=float
    )
    standard_score = _interpolate_curve(math.log10(STANDARD_PPM), standard_curve)
    matches = []
    for compound in COMPOUNDS:
        if compound["code"] == STANDARD_CODE:
            continue
        row = {"code": compound["code"], "name": compound["name"]}
        for model_name in (
            "universal_raw",
            "concentration_preserving_equal_hybrid",
        ):
            reference_curve = np.asarray(curves[STANDARD_CODE][model_name], dtype=float)
            reference_score = _interpolate_curve(
                math.log10(STANDARD_PPM), reference_curve
            )
            log_match, status = _match_log_ppm(
                np.asarray(curves[str(compound["code"])][model_name], dtype=float),
                reference_score,
            )
            row[model_name] = {
                "predicted_log10_match_ppm": log_match,
                "predicted_match_ppm": 10.0**log_match,
                "status": status,
            }
        row["constant_5ppm_baseline"] = {
            "predicted_log10_match_ppm": math.log10(STANDARD_PPM),
            "predicted_match_ppm": STANDARD_PPM,
            "status": "fixed_baseline",
        }
        matches.append(row)
    prediction_rows_hash = shared.canonical_json_sha256(matches)
    script = Path(__file__).resolve()
    checks = {
        "outcomes_absent": not any(path.exists() for path in _outcome_paths(outcome_dir).values()),
        "five_structures_fixed": len(COMPOUNDS) == 5,
        "four_target_matches_predicted": len(matches) == 4,
        "target_exact_molecule_training_leakage_zero": source_audit[
            "target_exact_overlap_molecules_excluded"
        ]
        >= 1,
        "source_only_selection": model_audit["selection_used_minnesota_outcomes"] is False,
        "portable_parity_at_most_1e_10": model_audit["parameters"][
            "portable_parity_maximum_absolute_error"
        ]
        <= 1e-10,
        "all_match_predictions_finite": all(
            math.isfinite(float(row[model]["predicted_log10_match_ppm"]))
            for row in matches
            for model in (
                "universal_raw",
                "concentration_preserving_equal_hybrid",
                "constant_5ppm_baseline",
            )
        ),
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "minnesota_intensity_matching_predictions_ready_before_outcomes",
        "blind_contract": {
            "raw_outcome_files_opened": False,
            "outcome_values_used_for_training": False,
            "outcome_values_used_for_model_selection": False,
            "prediction_rows_sha256": prediction_rows_hash,
            "grid_sha256": shared.canonical_json_sha256(GRID_LOG10_PPM.tolist()),
        },
        "dataset": {
            "doi": DATASET_DOI,
            "article_doi": ARTICLE_DOI,
            "item_handle": ITEM_HANDLE,
            "item_uuid": ITEM_UUID,
            "protocol_metadata_source": "repository readme only",
            "standard": "5 ppm acetylpropionyl",
            "line_scale_mm": [0.0, 150.0],
            "same_intensity_mark_mm": LINE_CENTER_MM,
            "expected_outcomes": [
                {
                    **row,
                    "url": (
                        "https://conservancy.umn.edu/server/api/core/bitstreams/"
                        f"{row['uuid']}/content"
                    ),
                    "downloaded_or_opened": False,
                }
                for row in FILES
            ],
        },
        "compounds": list(COMPOUNDS),
        "grid": {
            "log10_ppm": GRID_LOG10_PPM.tolist(),
            "ppm_minimum": float(GRID_PPM[0]),
            "ppm_maximum": float(GRID_PPM[-1]),
            "points": len(GRID_PPM),
        },
        "training": source_audit,
        "model": {
            **model_audit,
            "humanpom": humanpom_audit,
            "primary": "concentration_preserving_equal_hybrid",
            "fixed_baseline": "constant_5ppm_baseline",
            "hybrid_formula": (
                "mean(universal_at_c) + 0.5*(HumanPOM-mean(HumanPOM)) + "
                "0.5*(universal_at_c-mean(universal_at_c))"
            ),
            "design_informed_by_minnesota_outcomes": False,
        },
        "curves": curves,
        "standard_primary_score_at_5ppm": standard_score,
        "predictions": matches,
        "implementation": {
            "script_sha256": _sha256(script),
            "universal_builder_sha256": _sha256(Path(universal.__file__).resolve()),
            "shared_humanpom_sha256": _sha256(Path(shared.__file__).resolve()),
        },
        "release_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "scope": "permission to seal concentration-match predictions only",
        },
        "claim_boundary": (
            "Outcome-unopened prediction of four concentration match points relative "
            "to one 5 ppm standard. Not perfume-mixture similarity or human 90%."
        ),
    }
    if not document["release_gate"]["passed"]:
        raise RuntimeError(f"Minnesota prediction release gate failed: {checks}")
    _write_json(predictions_path, document)
    return document


def create_seal(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve(strict=True)
    seal_path = args.seal.resolve()
    outcome_dir = args.outcome_dir.resolve()
    assert_outcomes_absent(outcome_dir)
    if seal_path.exists():
        raise RuntimeError("refusing to overwrite Minnesota seal")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if predictions.get("release_gate", {}).get("passed") is not True:
        raise RuntimeError("Minnesota prediction release gate is closed")
    row_hash = shared.canonical_json_sha256(predictions.get("predictions", []))
    if row_hash != predictions.get("blind_contract", {}).get("prediction_rows_sha256"):
        raise RuntimeError("Minnesota prediction rows changed before seal")
    script_hash = _sha256(Path(__file__).resolve())
    if script_hash != predictions.get("implementation", {}).get("script_sha256"):
        raise RuntimeError("Minnesota implementation changed before seal")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": utc_now(),
        "prediction_file_sha256": _sha256(predictions_path),
        "prediction_file_bytes": predictions_path.stat().st_size,
        "prediction_rows_sha256": row_hash,
        "benchmark_script_sha256": script_hash,
        "outcome_directory": str(outcome_dir),
        "target_files": [
            {
                **row,
                "url": (
                    "https://conservancy.umn.edu/server/api/core/bitstreams/"
                    f"{row['uuid']}/content"
                ),
                "present_before_seal": False,
            }
            for row in FILES
        ],
        "scoring_contract": {
            "primary_endpoint": "participant-specific log10 ppm at 75 mm",
            "primary_model": "concentration_preserving_equal_hybrid",
            "fixed_baseline": "constant_5ppm_baseline",
            "participant_rule": "retest supersedes original judge-compound rows",
            "condition_rule": "replicate mean at compound/judge/concentration",
            "match_rule": "monotone intensity interpolation at 75 mm",
            "primary_metric": "four-compound mean absolute log10 concentration error",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
    }
    _write_json(seal_path, seal)
    return seal


def verify_seal(predictions_path: Path, seal_path: Path) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("prediction_file_sha256") != _sha256(predictions_path):
        raise RuntimeError("Minnesota sealed prediction hash mismatch")
    if seal.get("prediction_file_bytes") != predictions_path.stat().st_size:
        raise RuntimeError("Minnesota sealed prediction size mismatch")
    rows_hash = shared.canonical_json_sha256(predictions.get("predictions", []))
    if rows_hash != seal.get("prediction_rows_sha256"):
        raise RuntimeError("Minnesota sealed rows mismatch")
    script_hash = _sha256(Path(__file__).resolve())
    if script_hash != seal.get("benchmark_script_sha256"):
        raise RuntimeError("Minnesota script changed after seal")
    return {"predictions": predictions, "seal": seal}


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "perfumery-ai-core-minnesota-blind/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    verify_seal(args.predictions, args.seal)
    outcome_dir = args.outcome_dir.resolve()
    assert_outcomes_absent(outcome_dir)
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite Minnesota receipt")
    timestamp = shared.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if not timestamp.get("verified"):
        raise RuntimeError("Minnesota timestamp verification failed")
    started = utc_now()
    acquired = []
    for contract in FILES:
        url = (
            "https://conservancy.umn.edu/server/api/core/bitstreams/"
            f"{contract['uuid']}/content"
        )
        raw = _download(url)
        if _md5_bytes(raw) != contract["md5"]:
            raise RuntimeError(f"Minnesota repository MD5 changed: {contract['filename']}")
        text = raw.decode("utf-8-sig")
        row_count = sum(1 for _ in csv.DictReader(text.splitlines()))
        if row_count != contract["expected_rows"]:
            raise RuntimeError(f"Minnesota row count changed: {contract['filename']}")
        path = outcome_dir / str(contract["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            handle.write(raw)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        acquired.append(
            {
                "key": contract["key"],
                "path": str(path),
                "url": url,
                "bytes": len(raw),
                "md5": _md5_bytes(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": row_count,
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "minnesota_outcomes_acquired_after_verified_timestamp",
        "download_started_at": started,
        "download_completed_at": utc_now(),
        "prediction_sha256": _sha256(args.predictions.resolve(strict=True)),
        "seal_sha256": _sha256(args.seal.resolve(strict=True)),
        "timestamp": timestamp,
        "files": acquired,
    }
    _write_json(receipt_path, receipt)
    return receipt


def _normalize_column(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _load_outcomes(
    directory: Path, receipt: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_by_key = {str(row["key"]): row for row in FILES}
    frames: dict[str, list[dict[str, Any]]] = {}
    for file_record in receipt.get("files", []):
        key = str(file_record["key"])
        contract = contract_by_key[key]
        path = directory / str(contract["filename"])
        if _sha256(path) != file_record["sha256"]:
            raise RuntimeError(f"Minnesota outcome differs from receipt: {key}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            normalized = {_normalize_column(name): name for name in reader.fieldnames or []}
            required = {name: normalized.get(name) for name in ("compound", "judge", "conc", "intensity")}
            if any(value is None for value in required.values()):
                raise RuntimeError(f"Minnesota columns changed: {key}")
            rows = []
            for source in reader:
                rows.append(
                    {
                        "compound": str(source[required["compound"]]).strip().upper(),
                        "judge": str(source[required["judge"]]).strip(),
                        "concentration_ppm": float(source[required["conc"]]),
                        "intensity_mm": float(source[required["intensity"]]),
                        "file_key": key,
                    }
                )
            frames[key] = rows
    if set(frames) != set(contract_by_key):
        raise RuntimeError("Minnesota acquisition receipt is incomplete")
    return [*frames["original"], *frames["retest"], *frames["new_recruits"]], {
        "raw_rows": sum(len(values) for values in frames.values()),
        "file_rows": {key: len(values) for key, values in frames.items()},
    }


def _participant_matches(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retested = {
        (str(row["judge"]), str(row["compound"]))
        for row in rows
        if row["file_key"] == "retest"
    }
    selected = [
        row
        for row in rows
        if not (
            row["file_key"] == "original"
            and (str(row["judge"]), str(row["compound"])) in retested
        )
    ]
    conditions: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in selected:
        code = str(row["compound"]).upper()
        if code not in {str(item["code"]) for item in COMPOUNDS if item["code"] != STANDARD_CODE}:
            raise RuntimeError(f"unknown Minnesota compound code: {code}")
        concentration = float(row["concentration_ppm"])
        intensity = float(row["intensity_mm"])
        if concentration <= 0.0 or not 0.0 <= intensity <= 150.0:
            raise RuntimeError("Minnesota outcome value outside documented range")
        conditions[(str(row["judge"]), code, concentration)].append(intensity)
    by_participant: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (judge, code, concentration), values in conditions.items():
        by_participant[(judge, code)].append(
            (math.log10(concentration), float(np.mean(values)))
        )
    matches = []
    censored = Counter()
    for (judge, code), values in sorted(by_participant.items()):
        ordered = sorted(values)
        x = np.asarray([value[0] for value in ordered], dtype=float)
        y = _monotonic(np.asarray([value[1] / 150.0 for value in ordered])) * 150.0
        if LINE_CENTER_MM <= y[0]:
            match = float(x[0])
            status = "lower_observed_censored"
        elif LINE_CENTER_MM >= y[-1]:
            match = float(x[-1])
            status = "upper_observed_censored"
        else:
            upper = int(np.searchsorted(y, LINE_CENTER_MM, side="left"))
            lower = upper - 1
            if y[upper] <= y[lower] + 1e-12:
                match = float(x[upper])
                status = "flat_segment"
            else:
                fraction = (LINE_CENTER_MM - y[lower]) / (y[upper] - y[lower])
                match = float(x[lower] + fraction * (x[upper] - x[lower]))
                status = "interpolated"
        censored[status] += 1
        matches.append(
            {
                "judge": judge,
                "compound": code,
                "observed_log10_match_ppm": match,
                "observed_match_ppm": 10.0**match,
                "status": status,
                "conditions": len(values),
            }
        )
    return matches, {
        "rows_after_retest_supersession": len(selected),
        "retested_judge_compounds": len(retested),
        "participant_compound_matches": len(matches),
        "status_counts": dict(sorted(censored.items())),
        "judges": len({row["judge"] for row in matches}),
    }


def _score_models(
    matches: Sequence[Mapping[str, Any]], predictions: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_by_code = {str(row["code"]): row for row in predictions["predictions"]}
    primary_matches = [row for row in matches if row["status"] == "interpolated"]
    if len(primary_matches) < 20:
        raise RuntimeError("too few uncensored Minnesota participant matches")
    model_names = (
        "concentration_preserving_equal_hybrid",
        "universal_raw",
        "constant_5ppm_baseline",
    )
    results = {}
    for name in model_names:
        errors = []
        predicted = []
        observed = []
        for row in primary_matches:
            value = float(prediction_by_code[str(row["compound"])][name]["predicted_log10_match_ppm"])
            target = float(row["observed_log10_match_ppm"])
            predicted.append(value)
            observed.append(target)
            errors.append(value - target)
        results[name] = {
            "log10_mae": float(np.mean(np.abs(errors))),
            "log10_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "bias": float(np.mean(errors)),
            "spearman": shared.spearman(predicted, observed),
            "within_factor_2": float(np.mean(np.abs(errors) <= math.log10(2.0))),
            "within_factor_3": float(np.mean(np.abs(errors) <= math.log10(3.0))),
            "participant_matches": len(errors),
        }
    by_compound = {}
    for code in sorted(prediction_by_code):
        values = [
            float(row["observed_log10_match_ppm"])
            for row in primary_matches
            if row["compound"] == code
        ]
        by_compound[code] = {
            "participants": len(values),
            "observed_median_log10_ppm": float(np.median(values)) if values else None,
            "observed_median_ppm": float(10.0 ** np.median(values)) if values else None,
            "predictions": {
                name: float(prediction_by_code[code][name]["predicted_match_ppm"])
                for name in model_names
            },
        }
    return results, {"uncensored": len(primary_matches), "by_compound": by_compound}


def _bootstrap_matches(
    matches: Sequence[Mapping[str, Any]], predictions: Mapping[str, Any]
) -> dict[str, Any]:
    primary = [row for row in matches if row["status"] == "interpolated"]
    by_code: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        by_code[str(row["compound"])].append(float(row["observed_log10_match_ppm"]))
    pred = {str(row["code"]): row for row in predictions["predictions"]}
    codes = sorted(by_code)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    mae_gain = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected_codes = generator.choice(codes, len(codes), replace=True)
        primary_errors = []
        baseline_errors = []
        for code in selected_codes:
            values = by_code[str(code)]
            sampled = generator.choice(values, len(values), replace=True)
            target = float(np.median(sampled))
            primary_value = float(
                pred[str(code)]["concentration_preserving_equal_hybrid"][
                    "predicted_log10_match_ppm"
                ]
            )
            baseline_value = math.log10(STANDARD_PPM)
            primary_errors.append(abs(primary_value - target))
            baseline_errors.append(abs(baseline_value - target))
        mae_gain.append(float(np.mean(baseline_errors) - np.mean(primary_errors)))
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "baseline_minus_primary_log10_mae_95_interval": [
            float(value) for value in np.quantile(mae_gain, [0.025, 0.975])
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    results = report["results"]
    return "\n".join(
        [
            "# Minnesota intensity matching blind benchmark",
            "",
            "| Model | log10 MAE | Spearman | within 2x |",
            "|---|---:|---:|---:|",
            *[
                f"| {name} | {row['log10_mae']:.4f} | {row['spearman']:.4f} | {row['within_factor_2']:.3f} |"
                for name, row in results.items()
            ],
            "",
            "- External concentration gate: **"
            + ("PASS" if report["external_concentration_gate"]["passed"] else "FAIL")
            + "**",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def score(args: argparse.Namespace) -> dict[str, Any]:
    verified = verify_seal(args.predictions, args.seal)
    predictions = verified["predictions"]
    receipt_path = args.receipt.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("prediction_sha256") != _sha256(args.predictions.resolve(strict=True)):
        raise RuntimeError("Minnesota receipt/prediction binding mismatch")
    if receipt.get("seal_sha256") != _sha256(args.seal.resolve(strict=True)):
        raise RuntimeError("Minnesota receipt/seal binding mismatch")
    timestamp = shared.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if timestamp.get("response_sha256") != receipt.get("timestamp", {}).get("response_sha256"):
        raise RuntimeError("Minnesota timestamp changed after acquisition")
    raw_rows, parser_audit = _load_outcomes(args.outcome_dir.resolve(strict=True), receipt)
    matches, match_audit = _participant_matches(raw_rows)
    results, population = _score_models(matches, predictions)
    bootstrap = _bootstrap_matches(matches, predictions)
    primary = results["concentration_preserving_equal_hybrid"]
    baseline = results["constant_5ppm_baseline"]
    checks = {
        "seal_timestamp_receipt_verified": True,
        "raw_rows_exact": parser_audit["raw_rows"] == 1_740,
        "primary_uncensored_matches_at_least_20": population["uncensored"] >= 20,
        "primary_log10_mae_below_constant_5ppm": primary["log10_mae"]
        < baseline["log10_mae"],
        "primary_log10_rmse_below_constant_5ppm": primary["log10_rmse"]
        < baseline["log10_rmse"],
        "primary_spearman_above_constant_5ppm": primary["spearman"]
        > baseline["spearman"],
        "bootstrap_mae_gain_lower_above_zero": bootstrap[
            "baseline_minus_primary_log10_mae_95_interval"
        ][0]
        > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": (
            "minnesota_blind_external_concentration_gate_passed"
            if all(checks.values())
            else "minnesota_blind_external_concentration_gate_failed"
        ),
        "blind_integrity": {
            "prediction_sha256": _sha256(args.predictions.resolve(strict=True)),
            "seal_sha256": _sha256(args.seal.resolve(strict=True)),
            "receipt_sha256": _sha256(receipt_path),
            "timestamp": timestamp,
            "outcomes_acquired_after_timestamp": True,
        },
        "dataset": {
            "doi": DATASET_DOI,
            "item_handle": ITEM_HANDLE,
            "parser": parser_audit,
            "matches": match_audit,
            "population": population,
        },
        "results": results,
        "bootstrap": bootstrap,
        "external_concentration_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "scope": "four-compound concentration matching against 5 ppm standard",
        },
        "mixture_intensity_external_gate": {
            "passed": False,
            "reason": "dataset supplies concentration matching, not whole-mixture intensity ratings",
        },
        "runtime_primary_score_weight": 0.0,
        "human_olfactory_90_percent_certified": False,
        "implementation": {"script_sha256": _sha256(Path(__file__).resolve())},
        "claim_boundary": (
            "Outcome-unopened external concentration-matching test on four compounds. "
            "It does not validate binary/complex mixture intensity or 90% perfume similarity."
        ),
    }
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite Minnesota score outputs")
    _write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--outcome-dir", type=Path, required=True)
    prepare_parser.add_argument("--keller-molecules", type=Path, required=True)
    prepare_parser.add_argument("--keller-stimuli", type=Path, required=True)
    prepare_parser.add_argument("--keller-behavior", type=Path, required=True)
    prepare_parser.add_argument("--ravia-root", type=Path, required=True)
    prepare_parser.add_argument("--bierling-predictions", type=Path, required=True)
    prepare_parser.add_argument("--bierling-pilot", type=Path, required=True)
    prepare_parser.add_argument("--molformer-root", type=Path, required=True)
    prepare_parser.add_argument("--hf-home", type=Path, required=True)
    prepare_parser.add_argument("--threads", type=int, default=4)
    prepare_parser.add_argument("--batch-size", type=int, default=32)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--predictions", type=Path, required=True)
    seal_parser.add_argument("--seal", type=Path, required=True)
    seal_parser.add_argument("--outcome-dir", type=Path, required=True)

    for name in ("acquire", "score"):
        item = subparsers.add_parser(name)
        item.add_argument("--predictions", type=Path, required=True)
        item.add_argument("--seal", type=Path, required=True)
        item.add_argument("--outcome-dir", type=Path, required=True)
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
        value = prepare(args)
        summary = {"status": value["status"], "predictions": value["predictions"]}
    elif args.command == "seal":
        value = create_seal(args)
        summary = {"sealed_at": value["sealed_at"], "prediction_sha256": value["prediction_file_sha256"]}
    elif args.command == "acquire":
        value = acquire(args)
        summary = {"status": value["status"], "files": value["files"]}
    else:
        value = score(args)
        summary = {
            "status": value["status"],
            "results": value["results"],
            "gate": value["external_concentration_gate"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
