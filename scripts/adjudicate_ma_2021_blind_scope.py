#!/usr/bin/env python
"""Bind a scope-only adjudication to the frozen Ma 2021 blind artifacts.

The row-level workbook was not accessed until after the RFC 3161 seal, but the
paper's aggregate findings were already public and were reviewed during source
selection.  This adjudicator therefore narrows the study label from a fully
outcome-naive prospective blind test to a row-level-outcome-unopened,
publication-summary-aware test.  It cannot alter predictions, ratings, metrics,
or gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402


SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _markdown(value: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Ma 2021 블라인드 범위 판정",
            "",
            "- 판정: **row-level outcome-unopened / publication-summary-aware**",
            "- 완전 outcome-naive prospective blind: **아님**",
            "- 예측·평정·지표·게이트 변경: **없음**",
            "",
            (
                "원본 3-sheet Excel의 행 단위 IA/IB/IAB 값은 RFC 3161 봉인 뒤 "
                "처음 취득했지만, 관련 논문 초록의 집계 결과는 source selection "
                "중 확인되었다. 따라서 행 단위 라벨 누출은 없으나 publication-level "
                "outcome awareness는 존재한다."
            ),
            "",
            value["claim_boundary"],
            "",
        ]
    )


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    if output.exists() or markdown.exists():
        raise RuntimeError("refusing to overwrite Ma scope adjudication")
    paths = {
        "predictions": args.predictions.resolve(strict=True),
        "seal": args.seal.resolve(strict=True),
        "timestamp": args.timestamp.resolve(strict=True),
        "receipt": args.receipt.resolve(strict=True),
        "blind_report": args.blind_report.resolve(strict=True),
    }
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    report = json.loads(paths["blind_report"].read_text(encoding="utf-8"))
    if seal.get("prediction_file_sha256") != _sha256(paths["predictions"]):
        raise RuntimeError("Ma adjudication prediction/seal binding mismatch")
    if report.get("source_binding", {}).get("seal_sha256") != _sha256(paths["seal"]):
        raise RuntimeError("Ma adjudication report/seal binding mismatch")
    if report.get("source_binding", {}).get("receipt_sha256") != _sha256(
        paths["receipt"]
    ):
        raise RuntimeError("Ma adjudication report/receipt binding mismatch")
    if report.get("blind_integrity", {}).get("timestamp", {}).get(
        "response_sha256"
    ) != _sha256(paths["timestamp"]):
        raise RuntimeError("Ma adjudication timestamp binding mismatch")
    if receipt.get("outcome", {}).get("sha256") != report.get(
        "source_binding", {}
    ).get("outcome_sha256"):
        raise RuntimeError("Ma adjudication outcome binding mismatch")
    if len(predictions.get("predictions", [])) != 2556:
        raise RuntimeError("Ma adjudication expected all 2556 prospective pair rows")
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "ma_2021_blind_scope_narrowed_without_metric_change",
        "source_binding": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "adjudication": {
            "row_level_outcome_workbook_opened_before_seal": False,
            "all_2556_pair_predictions_fixed_before_row_level_outcomes": True,
            "publication_aggregate_findings_available_before_seal": True,
            "publication_aggregate_findings_reviewed_during_source_selection": True,
            "fully_outcome_naive_prospective_blind": False,
            "approved_study_label": (
                "row-level outcome-unopened, publication-summary-aware external test"
            ),
            "raw_prediction_rows_changed": False,
            "human_rating_rows_changed": False,
            "metric_values_changed": False,
            "gate_values_changed": False,
        },
        "publication_level_information": {
            "data_article_doi": "10.1016/j.dib.2021.107143",
            "intensity_article_doi": "10.1016/j.foodchem.2021.129483",
            "examples_of_published_aggregate_findings": [
                "whole-mixture intensity equaled the strongest component in 73.9% of cases",
                "partial addition was reported in 21.7% of cases",
            ],
            "use_in_code_parameter_fitting": False,
            "design_awareness_cannot_be_reversed_after_review": True,
        },
        "authoritative_gate_status": {
            "parent_blind_integration_gate_passed": report.get(
                "mixture_operator_integration_gate", {}
            ).get("passed"),
            "runtime_integration_authorized": False,
            "human_olfactory_90_percent_certified": False,
        },
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "This scope adjudication preserves the useful row-level label-unopened "
            "test while prohibiting the stronger claim that the study was fully "
            "outcome-naive. It creates no new performance evidence or runtime authority."
        ),
    }
    shared.write_json(output, value)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(value), encoding="utf-8")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--timestamp", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    value = adjudicate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": value["status"],
                "approved_study_label": value["adjudication"][
                    "approved_study_label"
                ],
                "metric_values_changed": value["adjudication"][
                    "metric_values_changed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
