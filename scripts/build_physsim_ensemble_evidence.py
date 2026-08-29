#!/usr/bin/env python
"""Build pre-registered two-seed ensemble, uncertainty and OOD evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_physsim_r2 import metric_summary  # noqa: E402


PROTOCOLS = ("molecule_disjoint", "scaffold_disjoint")
CONFIG = "all_low_lr"
MODEL_SEEDS = (20260715, 20260713)
# A deliberately coarse grid limits development-set tuning.  The final reports
# are not read until one of these eleven candidates has been selected.
SEED_15_WEIGHT_GRID = tuple(index / 10.0 for index in range(11))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration(report: dict, protocol: str) -> dict:
    return report[protocol]["configurations"][CONFIG]


def _aligned(report_a: dict, report_b: dict, protocol: str) -> None:
    first = _configuration(report_a, protocol)["repeats"]
    second = _configuration(report_b, protocol)["repeats"]
    if len(first) != len(second):
        raise RuntimeError(f"{protocol}: repeat count mismatch")
    for left, right in zip(first, second):
        if (
            left["split_seed"] != right["split_seed"]
            or left["held_out_sha256"] != right["held_out_sha256"]
            or left["targets"] != right["targets"]
        ):
            raise RuntimeError(f"{protocol}: predictions are not partition-aligned")


def _quantile_higher(values: np.ndarray, probability: float) -> float:
    probability = min(1.0, math.ceil((len(values) + 1) * probability) / len(values))
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def _phase(
    first: dict,
    second: dict,
    protocol: str,
    member_weights: dict[int, float],
) -> dict:
    _aligned(first, second, protocol)
    first_rows = _configuration(first, protocol)["repeats"]
    second_rows = _configuration(second, protocol)["repeats"]
    pooled_predictions = []
    pooled_targets = []
    pooled_disagreement = []
    rows = []
    for row_15, row_13 in zip(first_rows, second_rows):
        p15 = np.asarray(row_15["predictions"], dtype=float)
        p13 = np.asarray(row_13["predictions"], dtype=float)
        targets = np.asarray(row_15["targets"], dtype=float)
        predictions = (
            member_weights[20260715] * p15
            + member_weights[20260713] * p13
        )
        disagreement = np.abs(p15 - p13)
        metrics = metric_summary(predictions, targets)
        rows.append(
            {
                "repeat": row_15["repeat"],
                "split_seed": row_15["split_seed"],
                "held_out_sha256": row_15["held_out_sha256"],
                "metrics": metrics,
                "member_disagreement_mean": float(np.mean(disagreement)),
                "member_disagreement_p95": float(np.percentile(disagreement, 95)),
                "predictions": predictions.tolist(),
                "targets": targets.tolist(),
            }
        )
        pooled_predictions.extend(predictions.tolist())
        pooled_targets.extend(targets.tolist())
        pooled_disagreement.extend(disagreement.tolist())
    fold_spearman = [row["metrics"]["spearman"] for row in rows]
    return {
        "repeats": rows,
        "pooled": metric_summary(pooled_predictions, pooled_targets),
        "fold_mean_spearman": float(np.mean(fold_spearman)),
        "fold_std_spearman": float(np.std(fold_spearman, ddof=1)),
        "member_disagreement_mean": float(np.mean(pooled_disagreement)),
        "member_disagreement_p95": float(np.percentile(pooled_disagreement, 95)),
        "pooled_predictions": pooled_predictions,
        "pooled_targets": pooled_targets,
        "pooled_member_disagreement": pooled_disagreement,
    }


def _select_member_weights(
    development_seed_15: dict,
    development_seed_13: dict,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    """Select one fixed ensemble on development data only.

    The objective gives equal weight to pooled and fold-mean Spearman for both
    strict protocols.  It therefore cannot solve molecule-disjoint instability
    by silently sacrificing scaffold-disjoint transfer.  Candidate resolution
    is intentionally only 0.1 to avoid fitting a precise weight to 149 repeated
    development observations.
    """

    candidates: list[dict[str, object]] = []
    for seed_15_weight in SEED_15_WEIGHT_GRID:
        weights = {
            20260715: float(seed_15_weight),
            20260713: float(1.0 - seed_15_weight),
        }
        protocol_metrics: dict[str, dict[str, float]] = {}
        objective_terms: list[float] = []
        for protocol in PROTOCOLS:
            section = _phase(
                development_seed_15,
                development_seed_13,
                protocol,
                weights,
            )
            metrics = {
                "pooled_spearman": float(section["pooled"]["spearman"]),
                "fold_mean_spearman": float(section["fold_mean_spearman"]),
                "minimum_fold_spearman": float(
                    min(row["metrics"]["spearman"] for row in section["repeats"])
                ),
            }
            protocol_metrics[protocol] = metrics
            objective_terms.extend(
                (metrics["pooled_spearman"], metrics["fold_mean_spearman"])
            )
        candidates.append(
            {
                "seed_20260715_weight": float(seed_15_weight),
                "seed_20260713_weight": float(1.0 - seed_15_weight),
                "balanced_mean_spearman": float(np.mean(objective_terms)),
                "worst_reported_spearman": float(
                    min(
                        value
                        for metrics in protocol_metrics.values()
                        for value in metrics.values()
                    )
                ),
                "protocol_metrics": protocol_metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["balanced_mean_spearman"]),
            float(row["worst_reported_spearman"]),
            -abs(float(row["seed_20260715_weight"]) - 0.5),
        ),
    )
    weights = {
        20260715: float(selected["seed_20260715_weight"]),
        20260713: float(selected["seed_20260713_weight"]),
    }
    return weights, candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--development-seed-15", type=Path, default=root / "benchmarks/physsim_r2_transfer_development_calibration.json")
    parser.add_argument("--development-seed-13", type=Path, default=root / "benchmarks/physsim_r2_transfer_development_seed_20260713.json")
    parser.add_argument("--final-seed-15", type=Path, default=root / "benchmarks/physsim_r2_transfer_final_strict.json")
    parser.add_argument("--final-seed-13", type=Path, default=root / "benchmarks/physsim_r2_transfer_final_seed_20260713.json")
    parser.add_argument("--strong-baselines", type=Path, default=root / "benchmarks/physsim_r2_strong_baselines.json")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/physsim_r2_ensemble_validation.json")
    args = parser.parse_args()

    paths = {
        "development_seed_20260715": args.development_seed_15,
        "development_seed_20260713": args.development_seed_13,
        "final_seed_20260715": args.final_seed_15,
        "final_seed_20260713": args.final_seed_13,
        "strong_baselines": args.strong_baselines,
    }
    # The final reports are deliberately left unopened until selection is
    # complete.  This makes the no-final-label selection claim executable,
    # rather than merely relying on a convention inside the objective.
    documents = {
        name: json.loads(paths[name].read_text(encoding="utf-8"))
        for name in (
            "development_seed_20260715",
            "development_seed_20260713",
        )
    }
    for name, expected_seed in (
        ("development_seed_20260715", 20260715),
        ("development_seed_20260713", 20260713),
    ):
        if documents[name]["model_seed"] != expected_seed:
            raise RuntimeError(f"{name}: seed contract mismatch")
    member_weights, selection_candidates = _select_member_weights(
        documents["development_seed_20260715"],
        documents["development_seed_20260713"],
    )
    documents.update(
        {
            name: json.loads(paths[name].read_text(encoding="utf-8"))
            for name in (
                "final_seed_20260715",
                "final_seed_20260713",
                "strong_baselines",
            )
        }
    )
    for name, expected_seed in (
        ("final_seed_20260715", 20260715),
        ("final_seed_20260713", 20260713),
    ):
        if documents[name]["model_seed"] != expected_seed:
            raise RuntimeError(f"{name}: seed contract mismatch")

    phases = {}
    for phase in ("development", "final"):
        first = documents[f"{phase}_seed_20260715"]
        second = documents[f"{phase}_seed_20260713"]
        phases[phase] = {
            protocol: _phase(first, second, protocol, member_weights)
            for protocol in PROTOCOLS
        }

    dev_errors = []
    dev_disagreements = []
    for protocol in PROTOCOLS:
        section = phases["development"][protocol]
        dev_errors.extend(
            np.abs(np.asarray(section["pooled_predictions"]) - np.asarray(section["pooled_targets"])).tolist()
        )
        dev_disagreements.extend(section["pooled_member_disagreement"])
    conformal_q95 = _quantile_higher(np.asarray(dev_errors, dtype=float), 0.95)
    disagreement_p95 = _quantile_higher(np.asarray(dev_disagreements, dtype=float), 0.95)

    strong = documents["strong_baselines"]
    selected_baseline = strong["selection"]["selected"]
    comparisons = {}
    gate = True
    for protocol in PROTOCOLS:
        final = phases["final"][protocol]
        baseline = strong["phases"]["final"][protocol]["summary"][selected_baseline]
        single = _configuration(documents["final_seed_20260715"], protocol)
        predictions = np.asarray(final["pooled_predictions"])
        targets = np.asarray(final["pooled_targets"])
        coverage = float(np.mean(np.abs(predictions - targets) <= conformal_q95))
        comparisons[protocol] = {
            "ensemble_pooled_spearman": final["pooled"]["spearman"],
            "seed_20260715_pooled_spearman": single["pooled_model"]["spearman"],
            "selected_strong_baseline": selected_baseline,
            "strong_baseline_pooled_spearman": baseline["pooled"]["spearman"],
            "ensemble_vs_strong_delta": final["pooled"]["spearman"] - baseline["pooled"]["spearman"],
            "ensemble_fold_mean_spearman": final["fold_mean_spearman"],
            "strong_baseline_fold_mean_spearman": baseline["fold_mean_spearman"],
            "fold_mean_delta": final["fold_mean_spearman"] - baseline["fold_mean_spearman"],
            "development_calibrated_interval_coverage": coverage,
        }
        passed = comparisons[protocol]["ensemble_vs_strong_delta"] >= 0.01 and comparisons[protocol]["fold_mean_delta"] >= 0.01
        comparisons[protocol]["passed"] = passed
        gate = gate and passed

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "selection_contract": {
            "member_weights": {
                str(key): value for key, value in member_weights.items()
            },
            "weights_selected_on": (
                "development-only balanced mean of pooled and fold-mean "
                "Spearman across molecule/scaffold strict protocols"
            ),
            "weight_grid": list(SEED_15_WEIGHT_GRID),
            "selection_candidates": selection_candidates,
            "final_labels_used_for_selection": False,
            "configuration": CONFIG,
        },
        "source_files": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "phases": phases,
        "uncertainty_calibration": {
            "method": "split-conformal absolute residual plus two-seed disagreement OOD gate",
            "calibration_phase": "development_only",
            "absolute_error_q95": conformal_q95,
            "maximum_member_disagreement": disagreement_p95,
            "behavior_outside_gate": "learned R2 weight is zero",
        },
        "final_comparison": comparisons,
        "release_gate_passed": gate,
        "claim_boundary": "Uncertainty is calibrated for historical mixture-pair labels, not human text-to-formula olfactory equivalence.",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "weights": payload["selection_contract"]["member_weights"],
        "uncertainty": payload["uncertainty_calibration"],
        "comparison": comparisons,
        "gate": gate,
        "output": str(args.output),
    }, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
