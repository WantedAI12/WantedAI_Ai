#!/usr/bin/env python
"""Import published gas-phase odor thresholds with explicit lineage.

The Abraham et al. values are historical human odor detection thresholds in
ppmv.  They improve headspace weighting where structures match exactly, but
they are never counted as validation of a generated formula or as evidence of
90% olfactory similarity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
import urllib.request
from datetime import date
from pathlib import Path

from rdkit import Chem


BASE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/abraham_2012"
ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3278675/"
DOI = "10.1093/chemse/bjr094"
LICENSE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/LICENSE"
OVERRIDE_SMILES = {
    "cyclamen_aldehyde": "CC(C)C1=CC=C(C=C1)CC(C)C=O",
    "undecavertol": "CCCCCC(/C(=C/CC)/C)O",
    "timberol": "CCCC(CCC1C(CCCC1(C)C)C)O",
    "cedramber": "C[C@@H]1CC[C@@H]2[C@]13CC[C@@]([C@H](C3)C2(C)C)(C)OC",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download(name: str) -> bytes:
    request = urllib.request.Request(
        f"{BASE_URL}/{name}",
        headers={"User-Agent": "perfumery-ai-core-threshold-import/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule, isomericSmiles=False, canonical=True
    )


def load_abraham() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    behavior_bytes = download("behavior.csv")
    molecule_bytes = download("molecules.csv")
    manifest_bytes = download("manifest.toml")
    behavior = {
        str(row["Stimulus"]): row
        for row in csv.DictReader(io.StringIO(behavior_bytes.decode("utf-8")))
    }
    records: dict[str, dict[str, object]] = {}
    for row in csv.DictReader(io.StringIO(molecule_bytes.decode("utf-8"))):
        cid = str(row["CID"])
        source = behavior.get(cid)
        structure = canonical(str(row.get("IsomericSMILES", "")))
        if source is None or structure is None:
            continue
        log_inverse = float(source["Log (1/ODT)"])
        records[structure] = {
            "pubchem_cid": int(cid),
            "name": row.get("name") or row.get("IUPACName"),
            "smiles": row.get("IsomericSMILES"),
            "log10_inverse_odt_ppmv": log_inverse,
            "odor_detection_threshold_ppmv": 10.0 ** (-log_inverse),
            "duplicates_averaged": int(float(source.get("Duplicates") or 0)),
        }
    lineage = {
        "behavior.csv": {
            "url": f"{BASE_URL}/behavior.csv",
            "sha256": sha256_bytes(behavior_bytes),
            "bytes": len(behavior_bytes),
        },
        "molecules.csv": {
            "url": f"{BASE_URL}/molecules.csv",
            "sha256": sha256_bytes(molecule_bytes),
            "bytes": len(molecule_bytes),
        },
        "manifest.toml": {
            "url": f"{BASE_URL}/manifest.toml",
            "sha256": sha256_bytes(manifest_bytes),
            "bytes": len(manifest_bytes),
        },
    }
    return records, lineage


def catalog_structures(
    catalog_path: Path, comptox_db: Path
) -> dict[str, tuple[str, str]]:
    catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
    ingredients = catalog_data.get("ingredients", catalog_data)
    names = {
        item["ingredient_id"]: item["name"]
        for item in ingredients
        if item.get("formulation_ready")
    }
    structures: dict[str, str] = {}
    with sqlite3.connect(comptox_db) as connection:
        for ingredient_id, smiles in connection.execute(
            """
            SELECT ingredient_id, COALESCE(qsar_ready_smiles, smiles)
            FROM chemicals
            WHERE COALESCE(qsar_ready_smiles, smiles) IS NOT NULL
            """
        ):
            if ingredient_id in names:
                structures[str(ingredient_id)] = str(smiles)
    for ingredient_id, smiles in OVERRIDE_SMILES.items():
        if ingredient_id in names:
            structures.setdefault(ingredient_id, smiles)
    return {
        ingredient_id: (names[ingredient_id], structure)
        for ingredient_id, structure in structures.items()
    }


def natural_components(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for material in data["materials"]:
        for component in material["components"]:
            if component.get("smiles"):
                records.append(
                    {
                        "ingredient_id": material["ingredient_id"],
                        "component_name": component["name"],
                        "cas_number": component.get("cas_number"),
                        "smiles": component["smiles"],
                        "composition_percent": component["percent"],
                    }
                )
    return records


def update_scientific_db(
    path: Path, matches: list[dict[str, object]], *, dry_run: bool
) -> int:
    updated = 0
    source_ref = (
        f"{ARTICLE_URL}; processed archive {BASE_URL}/behavior.csv; DOI:{DOI}"
    )
    with sqlite3.connect(path) as connection:
        for match in matches:
            if match["match_scope"] != "catalog_ingredient":
                continue
            row = connection.execute(
                "SELECT odor_threshold_ppm, source_ref FROM molecular_properties WHERE ingredient_id = ?",
                (match["ingredient_id"],),
            ).fetchone()
            if row is None or row[0] is not None:
                continue
            references = str(row[1] or "")
            references = f"{references}; {source_ref}" if references else source_ref
            if not dry_run:
                connection.execute(
                    """
                    UPDATE molecular_properties
                    SET odor_threshold_ppm = ?, source_ref = ?, verified_on = ?
                    WHERE ingredient_id = ?
                    """,
                    (
                        float(match["odor_detection_threshold_ppmv"]),
                        references,
                        date.today().isoformat(),
                        match["ingredient_id"],
                    ),
                )
            updated += 1
        if dry_run:
            connection.rollback()
        else:
            connection.commit()
    return updated


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "fragrance_ai" / "data" / "safe_ingredient_catalog.json",
    )
    parser.add_argument(
        "--comptox-db",
        type=Path,
        default=root / "fragrance_ai" / "data" / "epa_comptox_extract.db",
    )
    parser.add_argument(
        "--scientific-db",
        type=Path,
        default=root / "fragrance_ai" / "data" / "scientific_properties.db",
    )
    parser.add_argument(
        "--natural-compositions",
        type=Path,
        default=root / "fragrance_ai" / "data" / "natural_material_compositions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "fragrance_ai" / "data" / "odor_threshold_registry.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source, lineage = load_abraham()
    matches: list[dict[str, object]] = []
    for ingredient_id, (name, smiles) in catalog_structures(
        args.catalog, args.comptox_db
    ).items():
        structure = canonical(smiles)
        if structure is None or structure not in source:
            continue
        matches.append(
            {
                "match_scope": "catalog_ingredient",
                "ingredient_id": ingredient_id,
                "name": name,
                **source[structure],
            }
        )
    for component in natural_components(args.natural_compositions):
        structure = canonical(str(component["smiles"]))
        if structure is None or structure not in source:
            continue
        matches.append(
            {
                "match_scope": "natural_material_component",
                **component,
                **source[structure],
            }
        )
    matches.sort(
        key=lambda item: (
            str(item["match_scope"]),
            str(item["ingredient_id"]),
            str(item.get("component_name", "")),
        )
    )
    updated = update_scientific_db(
        args.scientific_db, matches, dry_run=args.dry_run
    )
    registry = {
        "schema_version": "1.0",
        "source": {
            "title": "An Algorithm for 353 Odor Detection Thresholds in Humans",
            "doi": DOI,
            "article_url": ARTICLE_URL,
            "processed_archive": "Pyrfume Public Data Archive / abraham_2012",
            "archive_license": "MIT",
            "archive_license_url": LICENSE_URL,
            "measurement": "historical human gas-phase odor detection threshold",
            "unit": "ppmv",
            "conversion": "ODT_ppmv = 10 ** (-Log10(1/ODT))",
            "known_source_self_consistency": "approximately 0.66 log unit as reported by the article",
            "retrieved_on": date.today().isoformat(),
            "files": lineage,
        },
        "claim_boundary": (
            "Historical compound-level threshold prior only; excluded from generated-formula "
            "human validation and from any 90% olfactory-equivalence claim."
        ),
        "source_record_count": len(source),
        "matched_record_count": len(matches),
        "catalog_match_count": sum(
            item["match_scope"] == "catalog_ingredient" for item in matches
        ),
        "natural_component_match_count": sum(
            item["match_scope"] == "natural_material_component" for item in matches
        ),
        "scientific_db_rows_updated": updated,
        "dry_run": args.dry_run,
        "matches": matches,
    }
    if not args.dry_run:
        args.output.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({key: registry[key] for key in (
        "source_record_count", "matched_record_count", "catalog_match_count",
        "natural_component_match_count", "scientific_db_rows_updated", "dry_run"
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
