#!/usr/bin/env python
"""Derive UVCB descriptor centroids from published natural-oil compositions.

Only molecular descriptor centroids are derived. Vapor pressure and odor
threshold remain unset unless measured for the actual material, so the
headspace engine continues to expose and penalize those missing endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.recommender.science import (  # noqa: E402
    MolecularProperties,
    ScientificPropertyStore,
)


def build_parser() -> argparse.ArgumentParser:
    data = PROJECT_ROOT / "fragrance_ai" / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compositions",
        type=Path,
        default=data / "natural_material_compositions.json",
    )
    parser.add_argument(
        "--db", type=Path, default=data / "scientific_properties.db"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(args.compositions.read_text(encoding="utf-8"))
    rows: list[MolecularProperties] = []
    summaries: list[dict[str, object]] = []
    for material in data["materials"]:
        values: list[list[float]] = []
        weights: list[float] = []
        for component in material["components"]:
            smiles = component.get("smiles")
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            values.append(
                [
                    Descriptors.MolWt(molecule),
                    Crippen.MolLogP(molecule),
                    rdMolDescriptors.CalcTPSA(molecule),
                    Lipinski.NumHDonors(molecule),
                    Lipinski.NumHAcceptors(molecule),
                    Lipinski.NumRotatableBonds(molecule),
                    Descriptors.BertzCT(molecule),
                ]
            )
            weights.append(float(component["percent"]))
        weight_array = np.asarray(weights, dtype=float)
        weight_array /= weight_array.sum()
        centroid = weight_array @ np.asarray(values, dtype=float)
        source = material["source"]
        source_ref = (
            f"composition-derived UVCB descriptor centroid; {source['url']}; "
            f"evidence={material['evidence_class']}; "
            "not a lot-specific physicochemical measurement"
        )
        row = MolecularProperties(
            ingredient_id=material["ingredient_id"],
            cas_number=material["material_cas"],
            molecular_weight=float(centroid[0]),
            xlogp=float(centroid[1]),
            tpsa=float(centroid[2]),
            hbond_donors=int(round(float(centroid[3]))),
            hbond_acceptors=int(round(float(centroid[4]))),
            rotatable_bonds=int(round(float(centroid[5]))),
            complexity=float(centroid[6]),
            vapor_pressure_pa_25c=None,
            boiling_point_c=None,
            odor_threshold_ppm=None,
            source_ref=source_ref,
            verified_on=date.today().isoformat(),
        )
        rows.append(row)
        summaries.append(
            {
                "ingredient_id": row.ingredient_id,
                "known_component_percent": round(sum(weights), 4),
                "apparent_molecular_weight": round(row.molecular_weight, 4),
                "apparent_xlogp": round(float(row.xlogp or 0.0), 4),
                "vapor_pressure_imported": False,
                "odor_threshold_imported": False,
            }
        )
    if not args.dry_run:
        store = ScientificPropertyStore(args.db)
        try:
            for row in rows:
                store.upsert(row)
        finally:
            store.close()
    print(json.dumps({
        "dry_run": args.dry_run,
        "records": len(rows),
        "rows": summaries,
        "claim_boundary": "composition-derived descriptors only; vapor and threshold left missing",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
