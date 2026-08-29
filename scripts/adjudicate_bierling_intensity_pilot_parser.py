#!/usr/bin/env python
"""Post-outcome one-column parser adjudication for the intensity pilot.

The frozen parent stopped before computing a statistic because the Zenodo CSV
uses ``intensity`` while the accompanying variable dictionary and parent score
contract use ``intensive``.  The file also contains 12 literal zero slider
endpoints although the dictionary says 1--100. This additive script verifies
those exact facts, keeps the zeros unchanged, relaxes only the validation bound
to 0--100, delegates every calculation to the unchanged parent, and marks both
corrections as post-outcome.
"""

from __future__ import annotations

import argparse
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

from scripts import blind_bierling_intensity_pilot_benchmark as parent  # noqa: E402


PARENT_SHA256 = "0eca50312fbe722f03b7dc4ce1f773e67d99e0f63bffd289b6b3d601d50d2561"
PREDICTION_SHA256 = "65f4a191f9408971f0944a602e25f58d810d1521e2b5504ffca2139ddeb39aaa"
SEAL_SHA256 = "03d08229aaca551d7d1df7b99af0c641af2328a48d8e037a5c7cac4578130a8c"
RECEIPT_SHA256 = "79d5d94f8b56a3baa93729be35aa297be10aedb882445dc1adade4a4e3055a44"
PILOT_SHA256 = "cc09e9f3f198d610a49f92b4da157e93ff8fded512f82641516bf98dc36827ab"
HEADER_SHA256 = "dae251644f2a4372ff2eff9c40709b629c289cd1b5e507e65351b58ffcef5486"
COLUMN_ALIASES = {"intensity": "intensive"}
EXPECTED_ZERO_RATINGS = 12
EXPECTED_REPEATED_ANCHOR_ROWS = 4
EXPECTED_REPEATED_ANCHOR_GROUPS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(args: argparse.Namespace) -> list[str]:
    contracts = {
        args.predictions.resolve(strict=True): PREDICTION_SHA256,
        args.seal.resolve(strict=True): SEAL_SHA256,
        args.receipt.resolve(strict=True): RECEIPT_SHA256,
        args.pilot.resolve(strict=True): PILOT_SHA256,
        Path(parent.__file__).resolve(strict=True): PARENT_SHA256,
    }
    mismatches = [
        str(path) for path, expected in contracts.items() if _sha256(path) != expected
    ]
    if mismatches:
        raise RuntimeError("frozen intensity adjudication hash mismatch: " + ",".join(mismatches))
    with args.pilot.resolve(strict=True).open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
    if hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise RuntimeError("intensity pilot header changed")
    columns = header.decode("utf-8").split(";")
    columns[0] = columns[0].removeprefix("\ufeff")
    if columns[-1] == "":
        columns[-1] = f"Unnamed: {len(columns) - 1}"
    if len(columns) != len(set(columns)):
        raise RuntimeError("intensity pilot header contains duplicate columns")
    if "intensity" not in columns or "intensive" in columns:
        raise RuntimeError("intensity alias adjudication is no longer exact")
    return columns


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    source_columns = _validate(args)
    pilot = args.pilot.resolve(strict=True)
    report_path = args.report.resolve()
    markdown_path = args.markdown.resolve()
    if report_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite intensity adjudication output")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    original_read_csv = pd.read_csv
    original_series_lt = pd.Series.__lt__

    def adjudicated_read_csv(path: Any, *read_args: Any, **read_kwargs: Any) -> Any:
        candidate = Path(path).resolve(strict=True)
        if candidate != pilot:
            return original_read_csv(path, *read_args, **read_kwargs)
        if read_args or read_kwargs != {"sep": ";", "low_memory": False}:
            raise RuntimeError("parent intensity CSV read contract changed")
        frame = original_read_csv(candidate, sep=";", low_memory=False)
        if frame.columns.tolist() != source_columns:
            raise RuntimeError("parsed intensity columns differ from frozen header")
        numeric = pd.to_numeric(frame["intensity"], errors="coerce")
        if (
            numeric.isna().any()
            or int((numeric == 0.0).sum()) != EXPECTED_ZERO_RATINGS
            or float(numeric.min()) != 0.0
            or float(numeric.max()) != 100.0
        ):
            raise RuntimeError("intensity pilot zero-endpoint contract changed")
        duplicate_key = ["code", "molcode", "concentration", "volume", "cas"]
        repeated = frame[frame.duplicated(duplicate_key, keep=False)]
        if (
            len(repeated) != EXPECTED_REPEATED_ANCHOR_ROWS
            or repeated.groupby(duplicate_key).ngroups
            != EXPECTED_REPEATED_ANCHOR_GROUPS
            or set(repeated["code"].astype(str)) != {"18"}
            or set(repeated["molcode"].astype(str)) != {"Benzyl", "Decan"}
            or set(repeated["odor_group"].astype(str)) != {"1", "2"}
        ):
            raise RuntimeError("intensity pilot repeated-anchor contract changed")
        frame["intensity"] = numeric
        aggregations = {
            "odor_group": "first",
            "intensity": "mean",
            source_columns[-1]: "first",
        }
        frame = frame.groupby(duplicate_key, as_index=False, sort=False).agg(
            aggregations
        )
        frame = frame[source_columns]
        if frame.duplicated(duplicate_key).any() or len(frame) != 964:
            raise RuntimeError("repeated-anchor collapse failed")
        return frame.rename(columns=COLUMN_ALIASES)

    def adjudicated_series_lt(series: Any, other: Any) -> Any:
        if series.name == "intensive" and other == 1:
            return original_series_lt(series, 0)
        return original_series_lt(series, other)

    pd.read_csv = adjudicated_read_csv
    pd.Series.__lt__ = adjudicated_series_lt
    try:
        with tempfile.TemporaryDirectory(
            prefix="bierling-intensity-adjudication-", dir=report_path.parent
        ) as temporary:
            temporary_root = Path(temporary)
            parent_args = argparse.Namespace(**vars(args))
            parent_args.report = temporary_root / "parent-report.json"
            parent_args.markdown = temporary_root / "parent-report.md"
            report = parent.score(parent_args)
            markdown = parent_args.markdown.read_text(encoding="utf-8")
    finally:
        pd.read_csv = original_read_csv
        pd.Series.__lt__ = original_series_lt

    script_path = Path(__file__).resolve()
    report["parser_adjudication"] = {
        "status": "post_outcome_single_column_alias_adjudication",
        "developed_after_pilot_opened": True,
        "first_parent_score_computed_statistics": False,
        "column_aliases": COLUMN_ALIASES,
        "trailing_empty_column_ignored": True,
        "accepted_zero_slider_endpoint_rows": EXPECTED_ZERO_RATINGS,
        "rating_values_changed": False,
        "adjudicated_valid_range": [0, 100],
        "repeated_anchor_rows_collapsed": EXPECTED_REPEATED_ANCHOR_ROWS,
        "repeated_anchor_participant_conditions": EXPECTED_REPEATED_ANCHOR_GROUPS,
        "repeated_anchor_aggregation": "within_participant_condition_mean",
        "predictions_changed": False,
        "conditions_or_ratings_changed": False,
        "metrics_or_bootstrap_changed": False,
        "parent_script_sha256": PARENT_SHA256,
        "adjudicator_script_sha256": _sha256(script_path),
        "header_sha256": HEADER_SHA256,
        "claim_boundary": (
            "Curves were outcome-unopened and externally timestamped; the exact "
            "intensity column alias was adjudicated after opening the pilot."
        ),
    }
    parent._write_json(report_path, report)
    markdown_path.write_text(
        markdown
        + "\n## Parser adjudication\n\n"
        + "첫 채점은 통계 계산 전에 공개 CSV의 `intensity`와 변수사전의 "
        + "`intensive` 차이에서 중단됐습니다. 결과 개봉 후 이 한 열 별칭만 "
        + "수정했습니다. 변수사전의 1--100 표기와 달리 공개 파일에 있는 12개 "
        + "0점은 값을 바꾸지 않고 slider endpoint로 허용했습니다. 곡선·조건·"
        + "평가값·지표·bootstrap은 변경하지 않았습니다.\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--openssl", type=Path, required=True)
    parser.add_argument("--timestamp-response", type=Path, required=True)
    parser.add_argument("--timestamp-ca", type=Path, required=True)
    parser.add_argument("--timestamp-tsa", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main() -> int:
    report = adjudicate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "population": report["population"],
                "results": report["results"],
                "condition_transfer_gate": report[
                    "condition_transfer_improvement_gate"
                ],
                "strict_external_gate": report["strict_external_gate"],
                "parser_adjudication": report["parser_adjudication"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
