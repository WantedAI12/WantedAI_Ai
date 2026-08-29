#!/usr/bin/env python
"""Calibrate a headspace response exponent on molecule-held-out Keller data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "concentration-headspace-calibration/v1"
SPLIT_SALT = "keller-concentration-headspace-v1"
BOOTSTRAP_SEED = 20_260_828
BOOTSTRAP_DRAWS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def split_fold(cid: int) -> int:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{int(cid)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5


def load_pairs(connection: sqlite3.Connection) -> list[dict[str, float | int]]:
    rows = connection.execute(
        "SELECT sc.cid, s.concentration_value, so.value "
        "FROM stimuli s JOIN stimulus_components sc "
        "USING(dataset, stimulus_id, variant) "
        "JOIN sensory_observations so "
        "ON so.dataset=s.dataset AND so.stimulus_a=s.stimulus_id "
        "WHERE s.dataset='keller_2016' AND so.endpoint='intensity' "
        "AND s.concentration_unit='fraction_v_v'"
    ).fetchall()
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for cid, concentration, intensity in rows:
        grouped[int(cid)].append((float(concentration), float(intensity)))
    pairs = []
    for cid in sorted(grouped):
        values = sorted(grouped[cid])
        if len(values) != 2:
            raise RuntimeError(f"Keller CID {cid} does not have exactly two levels")
        (low_concentration, low_intensity), (
            high_concentration,
            high_intensity,
        ) = values
        if not (
            0.0 < low_concentration < high_concentration
            and 0.0 <= low_intensity <= 100.0
            and 0.0 <= high_intensity <= 100.0
        ):
            raise RuntimeError(f"invalid Keller concentration pair for CID {cid}")
        exponent = (
            math.log1p(high_intensity) - math.log1p(low_intensity)
        ) / (math.log(high_concentration) - math.log(low_concentration))
        pairs.append(
            {
                "cid": cid,
                "low_concentration": low_concentration,
                "low_intensity": low_intensity,
                "high_concentration": high_concentration,
                "high_intensity": high_intensity,
                "molecule_exponent": exponent,
                "fold": split_fold(cid),
            }
        )
    if len(pairs) != 480:
        raise RuntimeError(f"expected 480 Keller concentration pairs, found {len(pairs)}")
    return pairs


def predict(
    rows: Sequence[Mapping[str, float | int]], exponent: float, direction: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = []
    candidate = []
    baseline = []
    for row in rows:
        low_c = float(row["low_concentration"])
        high_c = float(row["high_concentration"])
        low_i = float(row["low_intensity"])
        high_i = float(row["high_intensity"])
        if direction == "low_to_high":
            source_c, source_i, target_c, target_i = low_c, low_i, high_c, high_i
        elif direction == "high_to_low":
            source_c, source_i, target_c, target_i = high_c, high_i, low_c, low_i
        else:
            raise ValueError(f"unsupported concentration direction: {direction}")
        estimate = math.expm1(
            math.log1p(source_i) + exponent * (math.log(target_c) - math.log(source_c))
        )
        target.append(target_i)
        candidate.append(float(np.clip(estimate, 0.0, 100.0)))
        baseline.append(source_i)
    return np.asarray(target), np.asarray(candidate), np.asarray(baseline)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "n": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "pearson": float(np.corrcoef(prediction, target)[0, 1]),
    }


def paired_bootstrap(
    target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(target)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        index = rng.integers(0, count, count)
        gains[draw] = np.mean(np.abs(baseline[index] - target[index])) - np.mean(
            np.abs(candidate[index] - target[index])
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "baseline_minus_candidate_mae_95_interval": [
            float(np.percentile(gains, 2.5)),
            float(np.percentile(gains, 97.5)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.db",
    )
    parser.add_argument(
        "--hub-report",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "concentration_headspace_calibration_v1.json",
    )
    args = parser.parse_args()
    hub = args.hub.expanduser().resolve(strict=True)
    hub_report_path = args.hub_report.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    hub_report = json.loads(hub_report_path.read_text(encoding="utf-8"))
    if hub_report.get("schema") != "headspace-sensory-hub/v1":
        raise RuntimeError("unsupported headspace sensory hub report")
    if hub_report.get("database", {}).get("sha256") != sha256(hub):
        raise RuntimeError("headspace sensory hub hash does not match its report")
    connection = sqlite3.connect(hub.as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("headspace sensory hub integrity check failed")
        pairs = load_pairs(connection)
    finally:
        connection.close()
    training = [row for row in pairs if int(row["fold"]) != 0]
    holdout = [row for row in pairs if int(row["fold"]) == 0]
    exponent = float(np.median([float(row["molecule_exponent"]) for row in training]))
    distribution = np.asarray(
        [float(row["molecule_exponent"]) for row in pairs], dtype=float
    )
    directions: dict[str, Any] = {}
    checks = []
    for direction in ("low_to_high", "high_to_low"):
        target, candidate, baseline = predict(holdout, exponent, direction)
        bootstrap = paired_bootstrap(target, candidate, baseline)
        candidate_metrics = metrics(candidate, target)
        baseline_metrics = metrics(baseline, target)
        passed = bool(
            candidate_metrics["mae"] < baseline_metrics["mae"]
            and bootstrap["baseline_minus_candidate_mae_95_interval"][0] > 0.0
        )
        checks.append(passed)
        directions[direction] = {
            "candidate": candidate_metrics,
            "unchanged_intensity_baseline": baseline_metrics,
            "paired_molecule_bootstrap": bootstrap,
            "passed": passed,
        }
    gate = bool(all(checks) and 0.0 < exponent < 1.0)
    report = {
        "schema": SCHEMA,
        "status": (
            "molecule_holdout_transfer_passed_research_only"
            if gate
            else "molecule_holdout_transfer_failed"
        ),
        "source": {
            "hub_path": str(hub),
            "hub_sha256": sha256(hub),
            "hub_report_sha256": sha256(hub_report_path),
            "dataset": "Pyrfume processed Keller et al. 2016",
            "solvent": "paraffin oil",
            "intensity_unit": "rating_0_100",
        },
        "split": {
            "method": "sha256_cid_five_way_fold_zero_holdout",
            "salt": SPLIT_SALT,
            "training_molecules": len(training),
            "holdout_molecules": len(holdout),
            "molecule_overlap": 0,
            "selection_used_holdout_outcomes": False,
        },
        "calibration": {
            "equation": "log1p(I2)=log1p(I1)+exponent*ln(C2/C1)",
            "response_exponent": exponent,
            "estimator": "training_molecule_median",
            "molecule_exponent_quantiles": {
                name: float(np.quantile(distribution, quantile))
                for name, quantile in (
                    ("p05", 0.05),
                    ("p25", 0.25),
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p95", 0.95),
                )
            },
        },
        "holdout": directions,
        "gates": {
            "molecule_holdout_transfer": {
                "checks": {
                    "response_exponent_between_zero_and_one": 0.0 < exponent < 1.0,
                    "low_to_high_mae_improved_with_positive_ci": checks[0],
                    "high_to_low_mae_improved_with_positive_ci": checks[1],
                },
                "passed": gate,
            },
            "production": {
                "passed": False,
                "runtime_primary_score_weight": 0.0,
            },
        },
        "runtime_primary_score_weight": 0.0,
        "implementation": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "allow_pickle": False,
        },
        "claim_boundary": {
            "retrospective_public_human_intensity_transfer": True,
            "headspace_gc_ms_measurement": False,
            "multi_component_formula_validation": False,
            "natural_language_recipe_accuracy_measured": False,
            "human_olfactory_90_percent_certified": False,
        },
    }
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
