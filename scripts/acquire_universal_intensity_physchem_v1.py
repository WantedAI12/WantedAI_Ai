#!/usr/bin/env python
"""Acquire traceable PubChem transport properties for olfactory intensity data.

The registry merges exact structures from Keller, Ravia, Bierling and Ma,
resolves PubChem numeric properties in batches, and caches raw PUG View
responses for vapor pressure and boiling point. Missing annotations stay null.
Every normalized value retains the URL and SHA-256 of its raw response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import canonical_smiles  # noqa: E402
from scripts import blind_bierling_human_olfaction_benchmark as shared  # noqa: E402
from scripts import build_universal_intensity_model_v1 as universal  # noqa: E402
from scripts.enrich_pubchem import (  # noqa: E402
    parse_boiling_point_c,
    parse_vapor_pressure_pa,
)


SCHEMA_VERSION = "1.0"
USER_AGENT = "perfumery-ai-core-universal-intensity-physchem/1.0"
PROPERTY_NAMES = (
    "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
    "RotatableBondCount,Complexity"
)
HEADINGS = ("Vapor Pressure", "Boiling Point")


class _RateLimiter:
    def __init__(self, minimum_interval_seconds: float):
        self.minimum_interval = max(0.0, float(minimum_interval_seconds))
        self._lock = threading.Lock()
        self._last_start = 0.0

    def wait(self) -> None:
        with self._lock:
            remaining = self.minimum_interval - (time.monotonic() - self._last_start)
            if remaining > 0.0:
                time.sleep(remaining)
            self._last_start = time.monotonic()


def _sha256(path: Path) -> str:
    return shared.sha256_file(path)


def _identifier(value: object) -> str:
    text = str(value).strip().replace("\u00a0", "")
    return text[:-2] if text.endswith(".0") else text


def _request(url: str, *, attempts: int = 4) -> tuple[bytes | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), "ok"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None, "not_found"
            last_error = error
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"PubChem request failed: {url}") from last_error


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Two exact structures can resolve to the same preferred CID. Parallel
        # workers may therefore finish the same immutable cache object together.
        # Keep the first complete winner; never replace a partial file.
        if path.is_file() and path.stat().st_size > 0:
            temporary.unlink(missing_ok=True)
        else:
            raise


def _registry(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    skipped_structures: list[dict[str, str]] = []

    def add(
        smiles: str,
        *,
        source: str,
        cid: object = "",
        cas: object = "",
    ) -> None:
        raw = str(smiles).strip()
        if not raw or raw.casefold() in {"na", "nan", "none", "null"}:
            skipped_structures.append(
                {"source": source, "raw_structure": raw, "reason": "missing_structure"}
            )
            return
        try:
            canonical = canonical_smiles(raw)
        except ValueError:
            skipped_structures.append(
                {"source": source, "raw_structure": raw, "reason": "invalid_structure"}
            )
            return
        row = records.setdefault(
            canonical,
            {
                "canonical_smiles": canonical,
                "cids": set(),
                "cas_numbers": set(),
                "sources": set(),
            },
        )
        identifier = _identifier(cid)
        if identifier and identifier.isdigit():
            row["cids"].add(int(identifier))
        cas_value = str(cas).strip()
        if cas_value and cas_value.lower() not in {"nan", "none"}:
            row["cas_numbers"].add(cas_value)
        row["sources"].add(source)

    keller_path = args.keller_molecules.resolve(strict=True)
    universal._verify_source(
        keller_path,
        shared.KELLER_SOURCE_CONTRACT["molecules.csv"],
        "keller/molecules.csv",
    )
    with keller_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            smiles = str(row.get("CanonicalSMILES", "")).strip()
            if smiles:
                add(smiles, source="keller_2016", cid=row.get("CID", ""))

    ravia_path = args.ravia_molecules.resolve(strict=True)
    universal._verify_source(
        ravia_path,
        universal.RAVIA_SOURCE_CONTRACT["molecules.csv"],
        "ravia/molecules.csv",
    )
    with ravia_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            smiles = str(row.get("IsomericSMILES", "")).strip()
            if smiles:
                add(smiles, source="ravia_2020", cid=row.get("CID", ""))

    bierling_path = args.bierling_predictions.resolve(strict=True)
    bierling = json.loads(bierling_path.read_text(encoding="utf-8"))
    for row in bierling.get("predictions", []):
        add(
            str(row["canonical_smiles"]),
            source="bierling_2025",
            cid=row.get("cid", ""),
            cas=row.get("cas", ""),
        )

    ma_path = args.ma_predictions.resolve(strict=True)
    ma = json.loads(ma_path.read_text(encoding="utf-8"))
    for row in ma.get("target_odorants", []):
        add(
            str(row["canonical_smiles"]),
            source="ma_2021",
            cid=row.get("pubchem_cid", ""),
            cas=row.get("cas", ""),
        )

    result = []
    conflicting_cids = 0
    for smiles, row in sorted(records.items()):
        cids = sorted(row["cids"])
        if len(cids) > 1:
            conflicting_cids += 1
        result.append(
            {
                "canonical_smiles": smiles,
                "preferred_cid": cids[0] if cids else None,
                "all_cids": cids,
                "cas_numbers": sorted(row["cas_numbers"]),
                "sources": sorted(row["sources"]),
            }
        )
    if len(result) < 500:
        raise RuntimeError(f"too few unique olfactory structures: {len(result)}")
    return result, {
        "structures": len(result),
        "with_cid": sum(row["preferred_cid"] is not None for row in result),
        "without_cid": sum(row["preferred_cid"] is None for row in result),
        "structures_with_multiple_cids": conflicting_cids,
        "skipped_structures": skipped_structures,
        "source_files": {
            "keller_molecules": {
                "path": str(keller_path),
                "sha256": _sha256(keller_path),
                "bytes": keller_path.stat().st_size,
            },
            "ravia_molecules": {
                "path": str(ravia_path),
                "sha256": _sha256(ravia_path),
                "bytes": ravia_path.stat().st_size,
                "commit": universal.RAVIA_COMMIT,
            },
            "bierling_predictions": {
                "path": str(bierling_path),
                "sha256": _sha256(bierling_path),
                "bytes": bierling_path.stat().st_size,
            },
            "ma_predictions": {
                "path": str(ma_path),
                "sha256": _sha256(ma_path),
                "bytes": ma_path.stat().st_size,
            },
        },
    }


def _property_batches(
    registry: Sequence[Mapping[str, Any]], cache: Path
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    cids = sorted(
        int(row["preferred_cid"])
        for row in registry
        if row.get("preferred_cid") is not None
    )
    properties: dict[int, dict[str, Any]] = {}
    batch_hashes = []
    for batch_index, start in enumerate(range(0, len(cids), 75)):
        batch = cids[start : start + 75]
        joined = ",".join(str(value) for value in batch)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{joined}/property/{PROPERTY_NAMES}/JSON"
        )
        path = cache / "properties" / f"batch_{batch_index:03d}.json"
        if path.is_file():
            raw = path.read_bytes()
        else:
            raw, status = _request(url)
            if status != "ok" or raw is None:
                raise RuntimeError(f"PubChem property batch unavailable: {batch_index}")
            _atomic_write(path, raw)
        payload = json.loads(raw)
        for row in payload.get("PropertyTable", {}).get("Properties", []):
            properties[int(row["CID"])] = row
        batch_hashes.append(
            {
                "file": str(path.resolve()),
                "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "cid_count": len(batch),
            }
        )
    missing = sorted(set(cids) - set(properties))
    return properties, {
        "batches": batch_hashes,
        "resolved_cids": len(properties),
        "missing_cids": missing,
    }


def _annotation_strings(payload: object) -> list[str]:
    strings: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for marked in value.get("StringWithMarkup", []):
                if isinstance(marked, dict) and marked.get("String"):
                    strings.append(str(marked["String"]))
            if value.get("Number") and value.get("Unit"):
                strings.append(f"{value['Number']} {value['Unit']}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return list(dict.fromkeys(strings))


def _annotation(
    cid: int,
    heading: str,
    cache: Path,
    *,
    limiter: _RateLimiter,
) -> tuple[list[str], dict[str, Any]]:
    slug = heading.lower().replace(" ", "_")
    path = cache / "annotations" / str(cid) / f"{slug}.json"
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        f"?heading={urllib.parse.quote(heading)}"
    )
    if path.is_file():
        envelope = json.loads(path.read_text(encoding="utf-8"))
    else:
        limiter.wait()
        raw, status = _request(url)
        envelope = {
            "status": status,
            "url": url,
            "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
            "raw_bytes": len(raw) if raw else 0,
            "payload": json.loads(raw) if raw else None,
        }
        _atomic_write(
            path,
            (json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    strings = (
        _annotation_strings(envelope.get("payload"))
        if envelope.get("status") == "ok"
        else []
    )
    audit = {
        "cache_file": str(path.resolve()),
        "cache_sha256": _sha256(path),
        "url": url,
        "status": envelope.get("status"),
        "raw_sha256": envelope.get("raw_sha256"),
        "raw_bytes": envelope.get("raw_bytes"),
        "string_count": len(strings),
    }
    return strings, audit


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite universal physchem output")
    registry, registry_audit = _registry(args)
    cache = args.cache.resolve()
    properties, property_audit = _property_batches(registry, cache)
    limiter = _RateLimiter(args.delay_seconds)

    def enrich(source: Mapping[str, Any]) -> dict[str, Any]:
        cid = source.get("preferred_cid")
        annotations: dict[str, list[str]] = {}
        annotation_audit = {}
        if cid is not None:
            for heading in HEADINGS:
                values, audit = _annotation(
                    int(cid),
                    heading,
                    cache,
                    limiter=limiter,
                )
                annotations[heading] = values
                annotation_audit[heading] = audit
        property_row = properties.get(int(cid), {}) if cid is not None else {}
        vapor = parse_vapor_pressure_pa(annotations.get("Vapor Pressure", []))
        boiling = parse_boiling_point_c(annotations.get("Boiling Point", []))
        return {
            **source,
            "molecular_weight": (
                float(property_row["MolecularWeight"])
                if property_row.get("MolecularWeight") is not None
                else None
            ),
            "xlogp": (
                float(property_row["XLogP"])
                if property_row.get("XLogP") is not None
                else None
            ),
            "tpsa": (
                float(property_row["TPSA"])
                if property_row.get("TPSA") is not None
                else None
            ),
            "hbond_donors": property_row.get("HBondDonorCount"),
            "hbond_acceptors": property_row.get("HBondAcceptorCount"),
            "rotatable_bonds": property_row.get("RotatableBondCount"),
            "complexity": property_row.get("Complexity"),
            "vapor_pressure_pa_15_30c": vapor,
            "boiling_point_c": boiling,
            "annotation_audit": annotation_audit,
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
                        "vapor": sum(
                            row["vapor_pressure_pa_15_30c"] is not None for row in rows
                        ),
                        "boiling": sum(row["boiling_point_c"] is not None for row in rows),
                    }
                ),
                flush=True,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "universal_intensity_physchem_acquired",
        "source": {
            "provider": "PubChem PUG REST and PUG View",
            "provider_url": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "retrieval_cache": str(cache),
            "registry": registry_audit,
            "property_batches": property_audit,
        },
        "coverage": {
            "structures": len(rows),
            "cid": sum(row["preferred_cid"] is not None for row in rows),
            "molecular_weight": sum(row["molecular_weight"] is not None for row in rows),
            "xlogp": sum(row["xlogp"] is not None for row in rows),
            "vapor_pressure": sum(
                row["vapor_pressure_pa_15_30c"] is not None for row in rows
            ),
            "boiling_point": sum(row["boiling_point_c"] is not None for row in rows),
        },
        "records_sha256": shared.canonical_json_sha256(rows),
        "records": rows,
        "claim_boundary": (
            "Public physicochemical annotations and normalized literature values. "
            "Not lot-specific measurements, headspace measurements, odor thresholds, "
            "or human olfactory outcomes. Missing values remain null."
        ),
        "implementation": {
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    shared.write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keller-molecules", type=Path, required=True)
    parser.add_argument("--ravia-molecules", type=Path, required=True)
    parser.add_argument("--bierling-predictions", type=Path, required=True)
    parser.add_argument("--ma-predictions", type=Path, required=True)
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
