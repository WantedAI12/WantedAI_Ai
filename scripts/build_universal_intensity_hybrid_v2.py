#!/usr/bin/env python
"""Retrospective scale-anchor repair for universal intensity v1.

Universal v1 improved Ma molecule ranking but was biased low because the three
training protocols and the Ma anchored scale have different absolute levels.
This v2 keeps the frozen v1 molecular signal, anchors its cohort mean to the
target-excluded HumanPOM cohort mean, and averages the two centered signals at
fixed equal weight.  Ma outcomes are used only for retrospective evaluation,
not in the numeric transform, but the transform was designed after seeing v1's
Ma failure and therefore receives zero runtime weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_ma_2021_binary_mixture_benchmark as ma  # noqa: E402
from scripts import build_universal_intensity_model_v1 as v1  # noqa: E402


SCHEMA_VERSION = "2.0"
UNIVERSAL_WEIGHT = 0.5
HUMANPOM_WEIGHT = 0.5


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _equal_centered_hybrid(
    universal: np.ndarray, humanpom: np.ndarray
) -> np.ndarray:
    universal = np.asarray(universal, dtype=float)
    humanpom = np.asarray(humanpom, dtype=float)
    if universal.shape != humanpom.shape or universal.ndim != 1 or len(universal) < 3:
        raise ValueError("hybrid inputs must be equal non-trivial vectors")
    if not np.all(np.isfinite(universal)) or not np.all(np.isfinite(humanpom)):
        raise ValueError("hybrid inputs must be finite")
    anchored = (
        float(humanpom.mean())
        + HUMANPOM_WEIGHT * (humanpom - float(humanpom.mean()))
        + UNIVERSAL_WEIGHT * (universal - float(universal.mean()))
    )
    return np.clip(anchored, 0.0, 1.0)


def _pair_vectors(
    target_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    primary: np.ndarray,
    baseline: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    document = json.loads(args.ma_predictions.read_text(encoding="utf-8"))
    pair_lookup = {str(row["pair_id"]): row for row in document["predictions"]}
    index_by_cas = {str(row["cas"]): index for index, row in enumerate(target_rows)}
    scale = float(document["model"]["fechner_scale_target_units_per_natural_log"])
    primary_pair = []
    baseline_pair = []
    for row in pair_rows:
        pair = pair_lookup[str(row["unit_id"])]
        first = index_by_cas[str(pair["component_a"]["cas"])]
        second = index_by_cas[str(pair["component_b"]["cas"])]
        primary_pair.append(
            ma._fechner_pool(
                float(primary[first] * 10.0),
                float(primary[second] * 10.0),
                scale,
            )
        )
        baseline_pair.append(
            ma._fechner_pool(
                float(baseline[first] * 10.0),
                float(baseline[second] * 10.0),
                scale,
            )
        )
    target = np.asarray([float(row["target_iab"]) for row in pair_rows], dtype=float)
    return np.asarray(primary_pair), np.asarray(baseline_pair), target


def _markdown(report: Mapping[str, Any]) -> str:
    component = report["ma_retrospective_evaluation"]["monomolecular"]
    mixture = report["ma_retrospective_evaluation"]["binary_mixture"]
    return "\n".join(
        [
            "# Universal intensity hybrid v2",
            "",
            f"- Component hybrid MAE: **{component['hybrid_equal']['mae']:.5f}**",
            f"- Component HumanPOM MAE: **{component['humanpom']['mae']:.5f}**",
            f"- Mixture hybrid+Fechner MAE: **{mixture['hybrid_equal::fechner']['mae']:.5f} / 10**",
            f"- Mixture HumanPOM+Fechner MAE: **{mixture['humanpom::fechner']['mae']:.5f} / 10**",
            "- Retrospective repair gate: **"
            + ("PASS" if report["retrospective_repair_gate"]["passed"] else "FAIL")
            + "**",
            "- Prospective external gate: **FAIL (not run)**",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite universal intensity hybrid v2")
    v1_path = args.v1_report.resolve(strict=True)
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    if v1_report.get("implementation", {}).get("script_sha256") != _sha256(
        Path(v1.__file__).resolve()
    ):
        raise RuntimeError("universal intensity v1 implementation binding changed")
    if v1_report.get("final_model", {}).get("parameters", {}).get(
        "portable_parity_maximum_absolute_error"
    ) != 0.0:
        raise RuntimeError("universal intensity v1 portable parity is not exact")

    target_rows, pair_rows, ma_audit = v1._load_ma_targets(args)
    raw_cache = v1.build_raw_descriptor_cache(
        [str(row["canonical_smiles"]) for row in target_rows]
    )
    prepared = v1._prepare_rows(target_rows, raw_cache)
    universal = v1._portable_predict(
        v1_report["final_model"]["parameters"], prepared
    )
    humanpom = v1._ma_component_baselines(args, prepared)["humanpom"]
    hybrid = _equal_centered_hybrid(universal, humanpom)
    component_metrics = {
        "hybrid_equal": v1._metrics(hybrid, prepared),
        "universal_raw": v1._metrics(universal, prepared),
        "humanpom": v1._metrics(humanpom, prepared),
    }
    mixture_metrics = v1._ma_mixture_predictions(
        prepared,
        pair_rows,
        {
            "hybrid_equal": hybrid,
            "universal_raw": universal,
            "humanpom": humanpom,
        },
        args,
    )
    target_component = np.asarray([float(row["target"]) for row in prepared])
    component_bootstrap = v1._bootstrap(
        hybrid,
        humanpom,
        target_component,
        unit="molecule",
    )
    primary_pair, baseline_pair, pair_target = _pair_vectors(
        prepared, pair_rows, hybrid, humanpom, args
    )
    mixture_bootstrap = v1._bootstrap(
        primary_pair,
        baseline_pair,
        pair_target,
        unit="mixture",
    )
    checks = {
        "equal_weights_sum_to_one": abs(UNIVERSAL_WEIGHT + HUMANPOM_WEIGHT - 1.0)
        <= 1e-12,
        "transform_uses_no_ma_outcome_values": True,
        "ma_component_mae_below_humanpom": component_metrics["hybrid_equal"]["mae"]
        < component_metrics["humanpom"]["mae"],
        "ma_component_spearman_above_humanpom": component_metrics["hybrid_equal"][
            "spearman"
        ]
        > component_metrics["humanpom"]["spearman"],
        "ma_component_mae_bootstrap_lower_above_zero": component_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "ma_mixture_fechner_mae_below_humanpom": mixture_metrics[
            "hybrid_equal::fechner"
        ]["mae"]
        < mixture_metrics["humanpom::fechner"]["mae"],
        "ma_mixture_fechner_spearman_above_humanpom": mixture_metrics[
            "hybrid_equal::fechner"
        ]["spearman"]
        > mixture_metrics["humanpom::fechner"]["spearman"],
        "ma_mixture_mae_bootstrap_lower_above_zero": mixture_bootstrap[
            "baseline_minus_primary_mae_95_interval"
        ][0]
        > 0.0,
        "ma_mixture_spearman_bootstrap_lower_at_least_minus_0_05": mixture_bootstrap[
            "primary_minus_baseline_spearman_95_interval"
        ][0]
        >= -0.05,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "universal_intensity_hybrid_v2_retrospective_repair_passed"
            if all(checks.values())
            else "universal_intensity_hybrid_v2_retrospective_repair_failed"
        ),
        "development_timing": "designed_after_v1_ma_external_failure_was_known",
        "source_binding": {
            "v1_report_sha256": _sha256(v1_path),
            "v1_script_sha256": _sha256(Path(v1.__file__).resolve()),
            "ma_predictions_sha256": _sha256(args.ma_predictions.resolve(strict=True)),
            "ma_outcome_sha256": _sha256(args.ma_outcome.resolve(strict=True)),
        },
        "hybrid_contract": {
            "formula": (
                "mean(HumanPOM) + 0.5*(HumanPOM-mean(HumanPOM)) + "
                "0.5*(Universal-mean(Universal))"
            ),
            "humanpom_weight": HUMANPOM_WEIGHT,
            "universal_weight": UNIVERSAL_WEIGHT,
            "component_or_ingredient_identity_features": [],
            "ma_outcomes_used_in_numeric_transform": False,
            "design_informed_by_ma_outcome": True,
        },
        "ma_data": ma_audit,
        "ma_retrospective_evaluation": {
            "monomolecular": component_metrics,
            "binary_mixture": mixture_metrics,
            "component_bootstrap": component_bootstrap,
            "mixture_bootstrap": mixture_bootstrap,
        },
        "retrospective_repair_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "prospective_external_gate": {
            "passed": False,
            "reason": "no outcome-unseen post-v2 external intensity target",
        },
        "runtime": {
            "primary_score_weight": 0.0,
            "status": "retrospective_validation_gated_diagnostic_only",
        },
        "human_olfactory_90_percent_certified": False,
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Retrospective target-excluded repair. Numeric inference does not use Ma "
            "labels, but the equal-weight hybrid was designed after inspecting Ma v1. "
            "It requires a new outcome-unseen external target before runtime promotion."
        ),
    }
    shared.write_json(output, report)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", type=Path, required=True)
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
                "monomolecular": report["ma_retrospective_evaluation"][
                    "monomolecular"
                ],
                "binary_mixture": report["ma_retrospective_evaluation"][
                    "binary_mixture"
                ],
                "retrospective_gate": report["retrospective_repair_gate"],
                "prospective_gate": report["prospective_external_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
