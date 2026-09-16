"""Hash-pinned whole-source property lookup, shared by all scientific consumers."""

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .physical_evidence import FIELDS, _graph


@lru_cache(maxsize=4)
def load(path, digest, size, mtime):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("physical evidence snapshot changed")
    doc = json.loads(raw)
    if (
        doc.get("schema") != "physical-evidence-index/v2"
        or doc.get("human_recipe_validation") is not False
        or doc.get("imputer_predictions_imported") is not False
        or not isinstance(doc.get("by_inchikey"), dict)
    ):
        raise ValueError("qualified independent physical evidence required")
    for key, row in doc["by_inchikey"].items():
        if (
            not row.get("graph")
            or not math.isfinite(row.get("molecular_weight", float("nan")))
            or row["molecular_weight"] <= 0
        ):
            raise ValueError("physical evidence identity required")
        if set(row["properties"]) - set(FIELDS):
            raise ValueError("unknown physical endpoint")
        for field, record in row["properties"].items():
            value = record.get("value")
            unit = {
                "vapor_pressure_pa_25c": "Pa",
                "boiling_point_c": "degC",
                "odor_threshold_ppm": "ppmv",
            }[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= (-273.15 if field == "boiling_point_c" else 0)
                or record.get("unit") != unit
                or not record.get("source_ref")
                or not record.get("observations")
            ):
                raise ValueError("invalid physical evidence value/unit/provenance")
            if (
                field == "vapor_pressure_pa_25c"
                and record.get("reference_temperature_k") != 298.15
            ):
                raise ValueError("vapor pressure must be qualified at 298.15 K")
    return doc


@lru_cache(maxsize=32768)
def full_key(graph):
    from rdkit import Chem

    return Chem.MolToInchiKey(Chem.MolFromSmiles(graph))


def enrich(ingredients, properties, pair):
    path, digest = pair
    stat = Path(path).stat()
    document = load(str(path), digest, stat.st_size, stat.st_mtime_ns)
    result = dict(properties)
    for item in ingredients:
        prior = result.get(item.ingredient_id)
        graph = _graph(item.structure_smiles)
        if prior is None or graph is None:
            continue
        record = document["by_inchikey"].get(full_key(graph))
        if (
            record is None
            or graph != record["graph"]
            or abs(prior.molecular_weight - record["molecular_weight"])
            > max(0.1, 0.005 * prior.molecular_weight)
        ):
            continue
        additions = {
            field: value["value"]
            for field, value in record["properties"].items()
            if getattr(prior, field) is None
        }
        if additions:
            refs = ";".join(
                record["properties"][field]["source_ref"] for field in sorted(additions)
            )
            result[item.ingredient_id] = replace(
                prior,
                **additions,
                source_ref=prior.source_ref + ";physical-v76:" + refs,
            )
    return result


def configured_pair():
    from .local_runtime import local_profile

    profile = local_profile()
    return profile.get("physical_evidence") if profile else None
