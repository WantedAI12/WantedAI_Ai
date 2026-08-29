#!/usr/bin/env python
"""Build the frozen catalog-to-R2 molecular descriptor registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.r2_physsim import (  # noqa: E402
    descriptor_contract,
    sha256_file,
    sha256_json,
    smiles_to_descriptors,
)


PUBCHEM_OVERRIDES = {
    "cyclamen_aldehyde": {
        "smiles": "CC(C)C1=CC=C(C=C1)CC(C)C=O",
        "source": "https://pubchem.ncbi.nlm.nih.gov/compound/517827",
    },
    "undecavertol": {
        "smiles": "CCCCCC(/C(=C/CC)/C)O",
        "source": "https://pubchem.ncbi.nlm.nih.gov/compound/6441135",
    },
    "timberol": {
        "smiles": "CCCC(CCC1C(CCCC1(C)C)C)O",
        "source": "https://pubchem.ncbi.nlm.nih.gov/compound/116699",
    },
    "cedramber": {
        "smiles": "C[C@@H]1CC[C@@H]2[C@]13CC[C@@]([C@H](C3)C2(C)C)(C)OC",
        "source": "https://pubchem.ncbi.nlm.nih.gov/compound/11085796",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    data = PROJECT_ROOT / "fragrance_ai" / "data"
    parser.add_argument("--catalog", type=Path, default=data / "safe_ingredient_catalog.json")
    parser.add_argument("--comptox-db", type=Path, default=data / "epa_comptox_extract.db")
    parser.add_argument("--natural", type=Path, default=data / "natural_material_compositions.json")
    parser.add_argument("--output", type=Path, default=data / "r2_ingredient_components.npz")
    parser.add_argument("--manifest", type=Path, default=data / "r2_ingredient_components_manifest.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    catalog_data = json.loads(args.catalog.read_text(encoding="utf-8"))
    ready = {
        item["ingredient_id"]: item
        for item in catalog_data["ingredients"]
        if item.get("formulation_ready")
    }
    natural_data = json.loads(args.natural.read_text(encoding="utf-8"))
    natural = {
        item["ingredient_id"]: item for item in natural_data["materials"]
    }
    comptox: dict[str, dict[str, str]] = {}
    with sqlite3.connect(args.comptox_db) as connection:
        for ingredient_id, smiles, dtxsid in connection.execute(
            """
            SELECT ingredient_id, COALESCE(qsar_ready_smiles, smiles), dtxsid
            FROM chemicals
            WHERE COALESCE(qsar_ready_smiles, smiles) IS NOT NULL
            """
        ):
            comptox[str(ingredient_id)] = {
                "smiles": str(smiles),
                "source": f"https://comptox.epa.gov/dashboard/chemical/details/{dtxsid}",
            }

    rows: list[dict[str, object]] = []
    coverage: dict[str, float] = {}
    for ingredient_id, item in ready.items():
        if ingredient_id in natural:
            material = natural[ingredient_id]
            known_percent = 0.0
            for component in material["components"]:
                smiles = component.get("smiles")
                if not smiles:
                    continue
                percent = float(component["percent"])
                known_percent += percent
                rows.append(
                    {
                        "ingredient_id": ingredient_id,
                        "ingredient_name": item["name"],
                        "component_name": component["name"],
                        "cas_number": component.get("cas_number") or "",
                        "smiles": smiles,
                        "material_fraction": percent / 100.0,
                        "evidence_class": material["evidence_class"],
                        "source": material["source"]["url"],
                    }
                )
            coverage[ingredient_id] = min(100.0, known_percent)
            continue
        source = comptox.get(ingredient_id) or PUBCHEM_OVERRIDES.get(ingredient_id)
        if source is None:
            raise RuntimeError(f"no exact molecular structure for {ingredient_id}")
        rows.append(
            {
                "ingredient_id": ingredient_id,
                "ingredient_name": item["name"],
                "component_name": item["name"],
                "cas_number": item.get("cas_number") or "",
                "smiles": source["smiles"],
                "material_fraction": 1.0,
                "evidence_class": (
                    "epa_comptox_structure"
                    if ingredient_id in comptox
                    else "pubchem_structure_override"
                ),
                "source": source["source"],
            }
        )
        coverage[ingredient_id] = 100.0

    rows.sort(
        key=lambda row: (
            str(row["ingredient_id"]),
            -float(row["material_fraction"]),
            str(row["component_name"]),
        )
    )
    descriptors = np.asarray(
        [smiles_to_descriptors(str(row["smiles"])) for row in rows],
        dtype=np.float32,
    )
    arrays = {
        "ingredient_ids": np.asarray([row["ingredient_id"] for row in rows], dtype="U64"),
        "ingredient_names": np.asarray([row["ingredient_name"] for row in rows], dtype="U128"),
        "component_names": np.asarray([row["component_name"] for row in rows], dtype="U160"),
        "cas_numbers": np.asarray([row["cas_number"] for row in rows], dtype="U32"),
        "smiles": np.asarray([row["smiles"] for row in rows], dtype="U300"),
        "material_fractions": np.asarray([row["material_fraction"] for row in rows], dtype=np.float32),
        "evidence_classes": np.asarray([row["evidence_class"] for row in rows], dtype="U80"),
        "sources": np.asarray([row["source"] for row in rows], dtype="U300"),
        "descriptors": descriptors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    descriptor_names = [name for name, _ in descriptor_contract()]
    manifest = {
        "schema_version": "1.0",
        "generated_on": date.today().isoformat(),
        "artifact_file": args.output.name,
        "artifact_sha256": sha256_file(args.output),
        "descriptor_count": len(descriptor_names),
        "descriptor_contract_sha256": sha256_json(descriptor_names),
        "formulation_ready_ingredient_count": len(ready),
        "covered_ingredient_count": len(coverage),
        "component_row_count": len(rows),
        "ingredient_composition_coverage_percent": coverage,
        "source_files": {
            args.catalog.name: sha256_file(args.catalog),
            args.comptox_db.name: sha256_file(args.comptox_db),
            args.natural.name: sha256_file(args.natural),
        },
        "claim_boundary": (
            "Synthetic structures are exact registry structures. Natural-material "
            "components are published reference compositions, not measurements of "
            "the operator's supplied lot."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "artifact": str(args.output),
        "sha256": manifest["artifact_sha256"],
        "ingredients": len(coverage),
        "component_rows": len(rows),
        "descriptor_shape": list(descriptors.shape),
        "minimum_composition_coverage_percent": min(coverage.values()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
