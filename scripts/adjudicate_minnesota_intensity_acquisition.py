#!/usr/bin/env python
"""Complete Minnesota acquisition after a documented readme row-count mismatch.

The frozen parent acquired and verified the original 820-row file, then stopped
before writing a receipt because the repository retest file contains 250 rows
while its readme says 360. This additive adjudicator preserves the parent code,
predictions and seal; it verifies all repository MD5s, records actual CSV row
counts, and writes the acquisition receipt consumed by the frozen scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_minnesota_intensity_matching_benchmark as parent  # noqa: E402


SCHEMA_VERSION = "1.0"
ACTUAL_ROW_COUNTS = {"original": 820, "retest": 250, "new_recruits": 560}


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _md5(value: bytes) -> str:
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "perfumery-ai-core-minnesota-adjudication/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite Minnesota adjudicated receipt")
    verified = parent.verify_seal(args.predictions, args.seal)
    timestamp = shared.verify_rfc3161_timestamp(
        openssl=args.openssl,
        seal_path=args.seal,
        response_path=args.timestamp_response,
        ca_path=args.timestamp_ca,
        tsa_path=args.timestamp_tsa,
    )
    if not timestamp.get("verified"):
        raise RuntimeError("Minnesota adjudication timestamp verification failed")
    outcome_dir = args.outcome_dir.resolve()
    outcome_dir.mkdir(parents=True, exist_ok=True)
    adjudication_started = datetime.now(timezone.utc).isoformat()
    original_path = outcome_dir / str(parent.FILES[0]["filename"])
    original_download_time = (
        datetime.fromtimestamp(original_path.stat().st_ctime, timezone.utc).isoformat()
        if original_path.is_file()
        else adjudication_started
    )
    acquired = []
    for contract in parent.FILES:
        key = str(contract["key"])
        url = (
            "https://conservancy.umn.edu/server/api/core/bitstreams/"
            f"{contract['uuid']}/content"
        )
        path = outcome_dir / str(contract["filename"])
        if path.is_file():
            raw = path.read_bytes()
            acquisition = (
                "verified_parent_partial_file"
                if key == "original"
                else "verified_existing_adjudicator_file"
            )
        else:
            raw = _download(url)
            acquisition = "downloaded_by_additive_adjudicator"
        if _md5(raw) != contract["md5"]:
            raise RuntimeError(f"Minnesota repository MD5 changed: {contract['filename']}")
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if set(reader.fieldnames or ()) != {"compound", "judge", "conc", "intensity"}:
            raise RuntimeError(f"Minnesota columns changed: {contract['filename']}")
        if len(rows) != ACTUAL_ROW_COUNTS[key]:
            raise RuntimeError(f"Minnesota actual row contract changed: {key}")
        if not path.is_file():
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", delete=False
            ) as handle:
                handle.write(raw)
                temporary = Path(handle.name)
            os.replace(temporary, path)
        acquired.append(
            {
                "key": key,
                "path": str(path),
                "url": url,
                "bytes": len(raw),
                "md5": _md5(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(rows),
                "acquisition": acquisition,
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "minnesota_outcomes_acquired_with_row_count_adjudication",
        "download_started_at": original_download_time,
        "download_completed_at": datetime.now(timezone.utc).isoformat(),
        "adjudication_started_at": adjudication_started,
        "prediction_sha256": _sha256(args.predictions.resolve(strict=True)),
        "seal_sha256": _sha256(args.seal.resolve(strict=True)),
        "timestamp": timestamp,
        "files": acquired,
        "acquisition_adjudication": {
            "developed_after_outcome_acquisition_started": True,
            "parent_stopped_before_receipt": True,
            "parent_verified_original_file_before_stop": True,
            "readme_expected_rows": {
                str(row["key"]): int(row["expected_rows"]) for row in parent.FILES
            },
            "repository_actual_rows": ACTUAL_ROW_COUNTS,
            "retest_readme_minus_actual_rows": 110,
            "prediction_rows_changed": False,
            "outcome_rows_changed": False,
            "scoring_contract_changed": False,
            "parent_script_sha256": _sha256(Path(parent.__file__).resolve()),
            "adjudicator_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "sealed_prediction_sha256": verified["seal"]["prediction_file_sha256"],
    }
    shared.write_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--outcome-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--openssl", type=Path, required=True)
    parser.add_argument("--timestamp-response", type=Path, required=True)
    parser.add_argument("--timestamp-ca", type=Path, required=True)
    parser.add_argument("--timestamp-tsa", type=Path, required=True)
    return parser


def main() -> int:
    value = adjudicate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": value["status"],
                "files": value["files"],
                "adjudication": value["acquisition_adjudication"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
