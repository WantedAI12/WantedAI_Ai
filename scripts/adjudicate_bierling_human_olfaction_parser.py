#!/usr/bin/env python
"""Post-outcome parser/availability adjudication for the frozen benchmark.

The externally timestamped parent scorer correctly fixed predictions,
population, endpoints, metrics, uncertainty, and target-file identity, but its
first scoring call stopped before computing any statistic because it used CSV
delimiter inference and two variable-dictionary spellings rather than the
record's actual R-export headers.  This additive adjudicator changes exactly:

* delimiter inference -> the observed and source-declared semicolon delimiter;
* ``fruit`` -> frozen endpoint ``fruity``;
* ``ammonia/urinous`` -> variable-dictionary endpoint ``ammonia/urinuos``.

It monkeypatches only the one pandas read of the exact outcome file, delegates
all metric calculations to the byte-frozen parent, and records that the public
behavior file contains no row for one predeclared odor (``4Isoprop``).  That
odor remains unscored: the original 74-odor gate stays failed while a separate
73-measured-odor adjudicated gate is reported.  These corrections were
developed after the target file was opened and do not turn the result into a
fully presealed scorer evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as parent  # noqa: E402


PARENT_SCRIPT_SHA256 = (
    "f4fc960456b2415969cd3543d5c244ff0f364e1deb11f20410391d1fe43f2632"
)
PREDICTION_SHA256 = "e35a5694676c30070b1ae1077a6c23f6e434a46c743d09a62f17fea2784029d8"
SEAL_SHA256 = "d4bed13fce00849f1b0beaac925a17ffb98410b36151c5ef0235a129ade9708e"
RECEIPT_SHA256 = "ff56be89a53c01f12514db9efc2f098491a87bc74d30573884dff86d62ab09c2"
OUTCOME_SHA256 = "4e7ec47089cfc43df3e008ed558ffd1ee05d23f51c364e5e2538ce247ef163a4"
OUTCOME_HEADER_SHA256 = (
    "8034b3c68419a80f744ae8b0fe503533607d31c68a2dfd2abb13a2b44d63c6fd"
)
COLUMN_ALIASES = {
    "fruit": "fruity",
    "ammonia/urinous": "ammonia/urinuos",
}
UNSCORED_ZERO_ROW_TARGET = "4Isoprop"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_inputs(args: argparse.Namespace) -> list[str]:
    expected = {
        args.predictions.resolve(strict=True): PREDICTION_SHA256,
        args.seal.resolve(strict=True): SEAL_SHA256,
        args.receipt.resolve(strict=True): RECEIPT_SHA256,
        args.outcome.resolve(strict=True): OUTCOME_SHA256,
        Path(parent.__file__).resolve(strict=True): PARENT_SCRIPT_SHA256,
    }
    mismatches = [
        str(path) for path, digest in expected.items() if _sha256(path) != digest
    ]
    if mismatches:
        raise RuntimeError("frozen adjudication input hash mismatch: " + ",".join(mismatches))
    outcome = args.outcome.resolve(strict=True)
    with outcome.open("rb") as handle:
        header_bytes = handle.readline().rstrip(b"\r\n")
    if hashlib.sha256(header_bytes).hexdigest() != OUTCOME_HEADER_SHA256:
        raise RuntimeError("Bierling outcome header differs from adjudicated header")
    columns = header_bytes.decode("utf-8").split(";")
    columns[0] = columns[0].removeprefix("\ufeff")
    if len(columns) != len(set(columns)):
        raise RuntimeError("Bierling outcome header contains duplicate columns")
    if "inclusion" not in columns:
        raise RuntimeError("semicolon parsing no longer exposes inclusion")
    if any(source not in columns for source in COLUMN_ALIASES):
        raise RuntimeError("adjudicated source column is missing")
    if any(target in columns for target in COLUMN_ALIASES.values()):
        raise RuntimeError("canonical endpoint already exists beside its alias")
    return columns


def score_with_parser_adjudication(args: argparse.Namespace) -> dict[str, Any]:
    source_columns = _validate_frozen_inputs(args)
    outcome = args.outcome.resolve(strict=True)
    final_report = args.report.resolve()
    final_markdown = args.markdown.resolve()
    if final_report.exists() or final_markdown.exists():
        raise RuntimeError("refusing to overwrite an existing adjudicated report")
    final_report.parent.mkdir(parents=True, exist_ok=True)
    final_markdown.parent.mkdir(parents=True, exist_ok=True)
    original_read_csv = pd.read_csv
    original_verify_prediction_seal = parent.verify_prediction_seal

    def adjudicated_read_csv(path: Any, *read_args: Any, **read_kwargs: Any) -> Any:
        candidate = Path(path).resolve(strict=True)
        if candidate != outcome:
            return original_read_csv(path, *read_args, **read_kwargs)
        if read_args:
            raise RuntimeError("parent outcome reader unexpectedly used positional options")
        if read_kwargs != {"sep": None, "engine": "python"}:
            raise RuntimeError("parent outcome read contract changed")
        frame = original_read_csv(candidate, sep=";", low_memory=False)
        if frame.columns.tolist() != source_columns:
            raise RuntimeError("parsed Bierling columns differ from the frozen header")
        if UNSCORED_ZERO_ROW_TARGET in set(frame["molcode"].astype(str)):
            raise RuntimeError("the adjudicated zero-row target now has behavior rows")
        return frame.rename(columns=COLUMN_ALIASES)

    def adjudicated_verify_prediction_seal(
        predictions_path: Path, seal_path: Path
    ) -> dict[str, Any]:
        verified = original_verify_prediction_seal(predictions_path, seal_path)
        result = copy.deepcopy(verified)
        rows = result["predictions"]["predictions"]
        omitted = [
            row for row in rows if row.get("molcode") == UNSCORED_ZERO_ROW_TARGET
        ]
        if len(rows) != 74 or len(omitted) != 1:
            raise RuntimeError("frozen 74-odor prediction set changed")
        result["predictions"]["predictions"] = [
            row for row in rows if row.get("molcode") != UNSCORED_ZERO_ROW_TARGET
        ]
        return result

    pd.read_csv = adjudicated_read_csv
    parent.verify_prediction_seal = adjudicated_verify_prediction_seal
    try:
        with tempfile.TemporaryDirectory(
            prefix="bierling-parser-adjudication-", dir=final_report.parent
        ) as temporary:
            temporary_root = Path(temporary)
            parent_args = argparse.Namespace(**vars(args))
            parent_args.report = temporary_root / "parent-report.json"
            parent_args.markdown = temporary_root / "parent-report.md"
            report = parent.score_outcome(parent_args)
            parent_markdown = parent_args.markdown.read_text(encoding="utf-8")
    finally:
        pd.read_csv = original_read_csv
        parent.verify_prediction_seal = original_verify_prediction_seal

    measured_checks = dict(report["improvement_gate"]["checks"])
    measured_checks.pop("all_74_target_odors_scored", None)
    measured_checks["public_behavior_73_of_74_scored"] = (
        report["dataset"]["population"]["odors"] == 73
    )
    measured_checks["zero_row_target_left_unscored"] = True
    report["measured_odor_improvement_gate"] = {
        "passed": all(measured_checks.values()),
        "checks": measured_checks,
        "scope": "73 odors with rows in the public behavior file",
    }
    report["status"] = (
        "blind_predictions_post_outcome_parser_availability_adjudicated_improvement"
        if report["measured_odor_improvement_gate"]["passed"]
        else "blind_predictions_post_outcome_adjudication_no_improvement"
    )

    adjudicator_path = Path(__file__).resolve()
    report["parser_adjudication"] = {
        "status": "post_outcome_parser_and_zero_row_availability_adjudication",
        "developed_after_target_file_opened": True,
        "first_parent_score_computed_statistics": False,
        "parent_failure": (
            "delimiter inference hid inclusion and R-export used fruit plus "
            "ammonia/urinous instead of variable-dictionary names"
        ),
        "changes": {
            "delimiter": "semicolon",
            "utf8_bom_removed_from_first_header_token": True,
            "column_aliases": COLUMN_ALIASES,
            "model_predictions_changed": False,
            "population_filter_changed": False,
            "endpoint_set_changed": False,
            "metric_or_bootstrap_changed": False,
            "scored_target_set_changed": True,
            "unscored_zero_row_target": UNSCORED_ZERO_ROW_TARGET,
            "original_74_odor_gate_preserved_failed": True,
        },
        "parent_script_sha256": PARENT_SCRIPT_SHA256,
        "adjudicator_script_sha256": _sha256(adjudicator_path),
        "outcome_header_sha256": OUTCOME_HEADER_SHA256,
        "claim_boundary": (
            "Predictions were genuinely outcome-unopened and externally timestamped; "
            "the exact CSV parser aliases were adjudicated post-outcome."
        ),
    }
    parent.write_json(final_report, report)
    final_markdown.write_text(
        parent_markdown
        + "\n## Parser adjudication\n\n"
        + "예측·모델·모집단·22개 endpoint·통계량은 외부 시각 전에 고정됐습니다. "
        + "다만 첫 채점은 통계 계산 전 CSV delimiter와 `fruit`, "
        + "`ammonia/urinous` 열 이름에서 중단됐고, 이 세 가지 읽기 규칙만 "
        + "결과 파일 개봉 후 별도 adjudicator로 수정했습니다. 또한 자극표의 "
        + "`4Isoprop`은 행동 파일에 한 행도 없어 미측정으로 남겼습니다. 원래 "
        + "74개 게이트는 실패로 보존하고 실측 가능한 73개 보조 게이트를 "
        + "분리했습니다. 따라서 예측은 "
        + "outcome-unopened이지만 scorer 전체가 완전 사전 봉인됐다고 표현하지 않습니다.\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--openssl", type=Path, required=True)
    parser.add_argument("--timestamp-response", type=Path, required=True)
    parser.add_argument("--timestamp-ca", type=Path, required=True)
    parser.add_argument("--timestamp-tsa", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = score_with_parser_adjudication(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "parser_adjudication": report["parser_adjudication"],
                "improvement_gate": report["improvement_gate"],
                "measured_odor_improvement_gate": report[
                    "measured_odor_improvement_gate"
                ],
                "results": report["results"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
