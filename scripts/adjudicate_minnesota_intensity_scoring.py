#!/usr/bin/env python
"""Correct Minnesota scoring to the sealed equal-compound primary metric.

The frozen parent averaged all uncensored participant-compound rows, which
weights compounds by their available participant counts. The seal predeclared
the mean absolute log10 error across four compounds. This additive adjudicator
keeps predictions and outcomes unchanged, computes participant errors within
each compound, then gives the four compounds equal weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_minnesota_intensity_matching_benchmark as parent  # noqa: E402


SCHEMA_VERSION = "1.0"
MODEL_NAMES = (
    "concentration_preserving_equal_hybrid",
    "universal_raw",
    "constant_5ppm_baseline",
)


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _compound_balanced_metrics(
    matches: Sequence[Mapping[str, Any]], predictions: Mapping[str, Any]
) -> dict[str, dict[str, float | int | bool]]:
    primary = [row for row in matches if row["status"] == "interpolated"]
    by_code: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        by_code[str(row["compound"])].append(
            float(row["observed_log10_match_ppm"])
        )
    prediction_by_code = {
        str(row["code"]): row for row in predictions["predictions"]
    }
    if set(by_code) != set(prediction_by_code):
        raise RuntimeError("Minnesota corrected score lacks one or more compounds")
    results = {}
    for name in MODEL_NAMES:
        compound_mae = []
        compound_mse = []
        compound_bias = []
        compound_prediction = []
        compound_target = []
        pooled_prediction = []
        pooled_target = []
        for code in sorted(by_code):
            predicted = float(
                prediction_by_code[code][name]["predicted_log10_match_ppm"]
            )
            observed = np.asarray(by_code[code], dtype=float)
            error = predicted - observed
            compound_mae.append(float(np.mean(np.abs(error))))
            compound_mse.append(float(np.mean(error**2)))
            compound_bias.append(float(np.mean(error)))
            compound_prediction.append(predicted)
            compound_target.append(float(np.median(observed)))
            pooled_prediction.extend([predicted] * len(observed))
            pooled_target.extend(observed.tolist())
        results[name] = {
            "log10_mae": float(np.mean(compound_mae)),
            "log10_rmse": float(np.sqrt(np.mean(compound_mse))),
            "bias": float(np.mean(compound_bias)),
            "compound_median_spearman": shared.spearman(
                compound_prediction, compound_target
            ),
            "participant_level_spearman": shared.spearman(
                pooled_prediction, pooled_target
            ),
            "compounds": len(by_code),
            "participant_matches": len(primary),
            "compound_balanced": True,
        }
    return results


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Minnesota intensity scoring adjudication",
            "",
            "| Model | compound-balanced log10 MAE | compound rank |",
            "|---|---:|---:|",
            *[
                f"| {name} | {row['log10_mae']:.4f} | {row['compound_median_spearman']:.4f} |"
                for name, row in report["corrected_results"].items()
            ],
            "",
            "- Corrected external gate: **"
            + ("PASS" if report["corrected_external_gate"]["passed"] else "FAIL")
            + "**",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite Minnesota scoring adjudication")
    predictions_path = args.predictions.resolve(strict=True)
    seal_path = args.seal.resolve(strict=True)
    receipt_path = args.receipt.resolve(strict=True)
    parent_report_path = args.parent_report.resolve(strict=True)
    verified = parent.verify_seal(predictions_path, seal_path)
    predictions = verified["predictions"]
    seal = verified["seal"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if parent_report.get("blind_integrity", {}).get("receipt_sha256") != _sha256(
        receipt_path
    ):
        raise RuntimeError("Minnesota parent report/receipt binding mismatch")
    if seal.get("scoring_contract", {}).get("primary_metric") != (
        "four-compound mean absolute log10 concentration error"
    ):
        raise RuntimeError("Minnesota sealed primary metric changed")
    raw_rows, parser_audit = parent._load_outcomes(
        args.outcome_dir.resolve(strict=True), receipt
    )
    matches, match_audit = parent._participant_matches(raw_rows)
    results = _compound_balanced_metrics(matches, predictions)
    bootstrap = parent._bootstrap_matches(matches, predictions)
    primary = results["concentration_preserving_equal_hybrid"]
    baseline = results["constant_5ppm_baseline"]
    checks = {
        "sealed_metric_replayed_with_equal_compound_weight": True,
        "repository_actual_rows_1630": parser_audit["raw_rows"] == 1_630,
        "four_compounds_scored": primary["compounds"] == 4,
        "primary_mae_below_constant_5ppm": primary["log10_mae"]
        < baseline["log10_mae"],
        "primary_rmse_below_constant_5ppm": primary["log10_rmse"]
        < baseline["log10_rmse"],
        "primary_compound_rank_above_constant": primary["compound_median_spearman"]
        > baseline["compound_median_spearman"],
        "bootstrap_mae_gain_lower_above_zero": bootstrap[
            "baseline_minus_primary_log10_mae_95_interval"
        ][0]
        > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "minnesota_compound_balanced_external_gate_passed"
            if all(checks.values())
            else "minnesota_compound_balanced_external_gate_failed"
        ),
        "source_binding": {
            "predictions_sha256": _sha256(predictions_path),
            "seal_sha256": _sha256(seal_path),
            "receipt_sha256": _sha256(receipt_path),
            "parent_report_sha256": _sha256(parent_report_path),
            "parent_script_sha256": _sha256(Path(parent.__file__).resolve()),
        },
        "adjudication": {
            "developed_after_outcomes_opened": True,
            "reason": (
                "parent pooled participant-compound rows despite sealed equal-compound metric"
            ),
            "prediction_values_changed": False,
            "outcome_values_changed": False,
            "participant_match_algorithm_changed": False,
            "metric_weighting_changed_to_match_seal": True,
            "parent_gate_preserved": parent_report.get(
                "external_concentration_gate", {}
            ).get("passed"),
        },
        "data": {"parser": parser_audit, "matches": match_audit},
        "parent_results": parent_report.get("results", {}),
        "corrected_results": results,
        "bootstrap": bootstrap,
        "corrected_external_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "mixture_intensity_external_gate": {
            "passed": False,
            "reason": "no whole-mixture intensity endpoint",
        },
        "runtime_primary_score_weight": 0.0,
        "human_olfactory_90_percent_certified": False,
        "implementation": {"script_sha256": _sha256(Path(__file__).resolve())},
        "claim_boundary": (
            "Post-outcome metric-weighting correction only. The corrected Minnesota "
            "concentration gate remains separate from mixture/perfume similarity."
        ),
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--outcome-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = adjudicate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "results": report["corrected_results"],
                "gate": report["corrected_external_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
