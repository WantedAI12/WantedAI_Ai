"""Compile every qualified local observation, not a selected recipe subset.

The immutable research hub is read-only. No fitted imputer is imported. Exact
isomeric identity, endpoint units, source records and conflicts are retained.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from rdkit import Chem
from rdkit.Chem import Descriptors

SCHEMA = "physical-evidence-index/v2"
HUB_SHA = "8696893581786f61992c934a56df342a99a72975bcfad232067d9c5f433433b8"
OPERA_SHA = "84a51d3615f61c6d752a0d0cb1254fa73ff00c9a3103f1830c695864d2ff1b7c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(smiles):
    if not isinstance(smiles, str) or not smiles or "." in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return None
    return (
        Chem.MolToInchiKey(mol),
        Chem.MolToSmiles(mol, isomericSmiles=True),
        float(Descriptors.MolWt(mol)),
    )


def compile_index(hub, legacy):
    if sha(hub) != HUB_SHA:
        raise ValueError(
            "inspected research hub hash required; requalify changed sources"
        )
    old = json.loads(Path(legacy).read_text(encoding="utf-8"))
    if (
        old.get("schema") != "physical-evidence-index/v1"
        or old.get("human_recipe_validation") is not False
    ):
        raise ValueError("qualified legacy threshold index required")
    candidates, excluded, structures = defaultdict(list), [], {}
    with sqlite3.connect(Path(hub).resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute(
            "SELECT * FROM physchem_observations ORDER BY observation_id"
        ):
            field = {
                "log10_vapor_pressure_mmhg": "vapor_pressure_pa_25c",
                "boiling_point_c": "boiling_point_c",
            }.get(row["endpoint"])
            if field is None:
                continue
            ident = identity(row["canonical_smiles"])
            expected_unit = (
                "log10(mmHg)" if field == "vapor_pressure_pa_25c" else "degC"
            )
            prefix = "OPERA_VP/" if field == "vapor_pressure_pa_25c" else "OPERA_BP/"
            if (
                ident is None
                or ident[0] != row["inchi_key"]
                or row["unit"] != expected_unit
                or not row["source_path"].startswith(prefix)
            ):
                excluded.append(
                    {
                        "observation_id": row["observation_id"],
                        "reason": "identity_or_endpoint_unit_or_source_mismatch",
                    }
                )
                continue
            value = float(row["value"])
            if not math.isfinite(value) or (
                field == "vapor_pressure_pa_25c" and not -30 < value < 10
            ):
                excluded.append(
                    {
                        "observation_id": row["observation_id"],
                        "reason": "invalid_numeric_domain",
                    }
                )
                continue
            value = (
                133.322387415 * 10.0**value
                if field == "vapor_pressure_pa_25c"
                else value
            )
            if field == "boiling_point_c" and value <= -273.15:
                raise ValueError("boiling point below absolute zero")
            key, graph, mw = ident
            structures[key] = {"graph": graph, "molecular_weight": mw}
            candidates[key, field].append(
                {
                    "value": value,
                    "unit": "Pa" if field == "vapor_pressure_pa_25c" else "degC",
                    "reference_temperature_k": 298.15
                    if field == "vapor_pressure_pa_25c"
                    else None,
                    "reference_pressure_pa": 101325.0
                    if field == "boiling_point_c"
                    else None,
                    "source_ref": "doi:10.23645/epacomptox.5588512.v1;"
                    + row["source_path"]
                    + ";observation:"
                    + str(row["observation_id"]),
                    "source_split": row["split"],
                    "source_archive_sha256": OPERA_SHA,
                    "source_class": "curated_experimental_property_not_OPERA_model_prediction",
                    "condition_source": "https://comptox.epa.gov/dashboard-api/ccdapp1/qmrfdata/file/by-modelid/30"
                    if field == "vapor_pressure_pa_25c"
                    else "OPERA_normal_boiling_point_endpoint",
                    "aggregation": "source_value",
                }
            )
    for graph, row in old["by_structure"].items():
        ident = identity(graph)
        if (
            ident is None
            or row.get("unit") != "ppmv"
            or row.get("identity_basis") != "exact_isomeric_graph"
        ):
            continue
        key, canonical, mw = ident
        if abs(mw - row["molecular_weight"]) > max(0.1, 0.005 * mw):
            continue
        structures[key] = {"graph": canonical, "molecular_weight": mw}
        candidates[key, "odor_threshold_ppm"].append(
            {
                "value": row["odor_threshold_ppm"],
                "unit": "ppmv",
                "reference_temperature_k": None,
                "source_ref": row["source_ref"],
                "source_class": "published_gas_detection_threshold_not_formula_similarity",
                "aggregation": "source_value",
            }
        )
    by_key, conflicts = {}, []
    for (key, field), rows in sorted(candidates.items()):
        values = [row["value"] for row in rows]
        if any(
            not math.isfinite(v) or (v <= 0 and field != "boiling_point_c")
            for v in values
        ):
            raise ValueError("invalid normalized physical evidence")
        # Conflicting measurements are not silently averaged into a fake value.
        tolerance = 1e-8 * max(1.0, max(map(abs, values)))
        if max(values) - min(values) > tolerance:
            conflicts.append({"inchi_key": key, "field": field, "observations": rows})
            continue
        record = by_key.setdefault(key, {**structures[key], "properties": {}})
        record["properties"][field] = {**rows[0], "observations": rows}
    return {
        "schema": SCHEMA,
        "by_inchikey": by_key,
        "legacy_by_cas": old["by_cas"],
        "identity_policy": "exact_full_InChIKey_plus_isomeric_graph_and_molecular_weight;no_name_or_skeleton_join",
        "sources": {
            "hub_sha256": HUB_SHA,
            "legacy_sha256": sha(legacy),
            "opera_archive_sha256": OPERA_SHA,
        },
        "conflicts_excluded": conflicts,
        "invalid_identity_or_units": excluded,
        "counts": {
            field: sum(field in r["properties"] for r in by_key.values())
            for field in (
                "vapor_pressure_pa_25c",
                "boiling_point_c",
                "odor_threshold_ppm",
            )
        },
        "human_recipe_validation": False,
        "imputer_predictions_imported": False,
        "data_rows_selected_by_recipe_outcome": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("new immutable output required")
    result = compile_index(a.hub, a.legacy)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sha256": sha(a.output),
                "counts": result["counts"],
                "conflicts": len(result["conflicts_excluded"]),
                "rejected": len(result["invalid_identity_or_units"]),
            }
        )
    )


if __name__ == "__main__":
    main()
