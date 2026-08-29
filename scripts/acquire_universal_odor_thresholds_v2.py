#!/usr/bin/env python
"""Acquire and normalize PubChem odor-threshold annotations.

This additive registry fixes two limitations of the legacy parser: bracketed
units such as ``[ppm]`` and strings containing both ``Odor low`` and ``Odor
high``. Low/detection thresholds are prioritized using local context; missing
records remain null. Raw response hashes are retained through the shared cache.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import blind_minnesota_intensity_matching_benchmark as minnesota  # noqa: E402
from scripts import acquire_universal_intensity_physchem_v1 as physchem  # noqa: E402


SCHEMA_VERSION = "2.0"
NUMBER = r"\d+(?:\.\d+)?(?:\s*[Xx×]\s*10\s*\^?\s*[+-]?\d+|[Ee][+-]?\d+)?"
THRESHOLD_PATTERN = re.compile(
    rf"({NUMBER})\s*\]?\s*\[?\s*(ppm|ppb|mg\s*/\s*(?:m\^?3|cu\s*m))\s*\]?",
    flags=re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _number(value: str) -> float | None:
    cleaned = value.replace("×", "X").replace(" ", "")
    cleaned = re.sub(r"([0-9.]+)[Xx]10\^?([+-]?\d+)", r"\1e\2", cleaned)
    try:
        numeric = float(cleaned)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def parse_threshold_ppm(strings: Sequence[str], molecular_weight: float) -> float | None:
    if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError("molecular weight must be positive and finite")
    candidates: list[tuple[int, float]] = []
    for text in strings:
        lowered = str(text).casefold()
        for match in THRESHOLD_PATTERN.finditer(str(text)):
            value = _number(match.group(1))
            if value is None or value <= 0.0:
                continue
            unit = re.sub(r"\s+", "", match.group(2).casefold())
            if unit == "ppb":
                value /= 1_000.0
            elif unit.startswith("mg/"):
                value = value * 24.45 / molecular_weight
            if not 1e-12 <= value <= 1_000_000.0:
                continue
            before = lowered[max(0, match.start() - 35) : match.start()]
            after = lowered[match.end() : min(len(lowered), match.end() + 35)]
            if re.search(r"\b(?:odor\s*)?low\b", after[:20]):
                score = 6
            elif re.search(r"\b(?:odor\s*)?high\b", after[:20]):
                score = 1
            elif re.search(r"\blow\b", before[-25:]):
                score = 6
            elif re.search(r"\bhigh\b", before[-25:]):
                score = 1
            elif "detection" in before or "threshold" in before:
                score = 5
            else:
                score = 3
            candidates.append((score, value))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    values = [value for score, value in candidates if score == best]
    return float(math.exp(statistics.median(math.log(value) for value in values)))


def _registry(physchem_document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "canonical_smiles": str(row["canonical_smiles"]),
            "preferred_cid": int(row["preferred_cid"]),
            "molecular_weight": float(row["molecular_weight"]),
            "sources": list(row["sources"]),
        }
        for row in physchem_document.get("records", [])
        if row.get("preferred_cid") is not None and row.get("molecular_weight") is not None
    ]
    by_cid = {int(row["preferred_cid"]): row for row in rows}
    for target in minnesota.COMPOUNDS:
        cid = int(target["cid"])
        entry = by_cid.setdefault(
            cid,
            {
                "canonical_smiles": str(target["canonical_smiles"]),
                "preferred_cid": cid,
                "molecular_weight": float(target.get("molecular_weight", 0.0) or 0.0),
                "sources": ["minnesota_2016_target"],
            },
        )
        if "minnesota_2016_target" not in entry["sources"]:
            entry["sources"] = [*entry["sources"], "minnesota_2016_target"]
    # Minnesota constants keep molecular weight in the PubChem discovery
    # metadata only in the parent script comments, so use the official fixed map.
    weights = {264: 88.11, 12_810: 170.25, 19_309: 128.13, 18_635: 104.17, 11_747: 100.12}
    for cid, weight in weights.items():
        by_cid[cid]["molecular_weight"] = weight
    return [by_cid[cid] for cid in sorted(by_cid)]


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite universal threshold registry")
    physchem_path = args.physchem.resolve(strict=True)
    document = json.loads(physchem_path.read_text(encoding="utf-8"))
    if document.get("records_sha256") != shared.canonical_json_sha256(
        document.get("records", [])
    ):
        raise RuntimeError("threshold acquisition physchem binding mismatch")
    registry = _registry(document)
    cache = args.cache.resolve()
    limiter = physchem._RateLimiter(args.delay_seconds)

    def enrich(row: Mapping[str, Any]) -> dict[str, Any]:
        strings, audit = physchem._annotation(
            int(row["preferred_cid"]),
            "Odor Threshold",
            cache,
            limiter=limiter,
        )
        threshold = parse_threshold_ppm(strings, float(row["molecular_weight"]))
        return {
            **row,
            "odor_threshold_ppm": threshold,
            "annotation_audit": audit,
            "annotation_string_count": len(strings),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(enrich, registry)):
            rows.append(row)
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "progress": index + 1,
                            "total": len(registry),
                            "thresholds": sum(
                                item["odor_threshold_ppm"] is not None for item in rows
                            ),
                        }
                    ),
                    flush=True,
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "universal_odor_threshold_registry_acquired",
        "source_binding": {
            "physchem_path": str(physchem_path),
            "physchem_sha256": _sha256(physchem_path),
            "physchem_records_sha256": document["records_sha256"],
            "cache": str(cache),
        },
        "parser_contract": {
            "bracketed_units": True,
            "local_low_high_context": True,
            "mg_m3_to_ppm_at_25c": "ppm = mg/m3 * 24.45 / molecular_weight",
            "selection": "highest local-context score then log-space median",
        },
        "coverage": {
            "structures": len(rows),
            "thresholds": sum(row["odor_threshold_ppm"] is not None for row in rows),
            "missing": sum(row["odor_threshold_ppm"] is None for row in rows),
            "minnesota_targets": {
                row["preferred_cid"]: row["odor_threshold_ppm"]
                for row in rows
                if "minnesota_2016_target" in row["sources"]
            },
        },
        "records_sha256": shared.canonical_json_sha256(rows),
        "records": rows,
        "claim_boundary": (
            "Normalized public odor-threshold annotations with heterogeneous source "
            "methods. Not a complete threshold database, lot measurement, or human "
            "similarity label. Missing values remain null."
        ),
        "implementation": {"script_sha256": _sha256(Path(__file__).resolve())},
    }
    shared.write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physchem", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.22)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    report = acquire(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage": report["coverage"],
                "records_sha256": report["records_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
