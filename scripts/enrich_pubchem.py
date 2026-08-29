"""Enrich catalog chemicals from official PubChem PUG REST/PUG View records.

The importer keeps only values whose units and conditions can be normalized.
Conflicting annotations are reduced with a median, and unavailable fields stay
NULL.  No value is silently replaced by a made-up zero.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.recommender.science import MolecularProperties, ScientificPropertyStore  # noqa: E402


PROPERTY_NAMES = (
    "MolecularWeight,XLogP,TPSA,HBondDonorCount,"
    "HBondAcceptorCount,RotatableBondCount,Complexity"
)
USER_AGENT = "perfumery-ai-core/0.7 (public scientific-data enrichment)"
NUMBER = r"\d+(?:\.\d+)?(?:\s*[Xx×]\s*10\s*\^?\s*[+-]?\d+|[Ee][+-]?\d+)?"


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch(cas_number: str) -> tuple[dict, str]:
    encoded = urllib.parse.quote(cas_number, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/"
        f"property/{PROPERTY_NAMES}/JSON"
    )
    payload = _request_json(url)
    return payload["PropertyTable"]["Properties"][0], url


def fetch_annotations(cid: int, heading: str) -> tuple[list[str], str]:
    encoded_heading = urllib.parse.quote(heading)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        f"?heading={encoded_heading}"
    )
    payload = _request_json(url)
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
    return list(dict.fromkeys(strings)), url


def _number(value: str) -> float | None:
    cleaned = value.replace("×", "X").replace(" ", "")
    cleaned = re.sub(r"([0-9.]+)[Xx]10\^?([+-]?\d+)", r"\1e\2", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _temperature_c(text: str) -> float | None:
    match = re.search(
        rf"(?:\bat\b|@)\s*({NUMBER})\s*(?:°|º|deg|∑)?\s*([CF])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = _number(match.group(1))
    if value is None:
        return None
    return (value - 32.0) * 5.0 / 9.0 if match.group(2).upper() == "F" else value


def parse_vapor_pressure_pa(strings: list[str]) -> float | None:
    candidates: list[tuple[int, float]] = []
    unit_factors = {
        "pa": 1.0,
        "kpa": 1_000.0,
        "hpa": 100.0,
        "mmhg": 133.322368,
        "torr": 133.322368,
        "atm": 101_325.0,
    }
    value_before_unit = re.compile(
        rf"({NUMBER})\s*\]?\s*(mm\s*hg|mmhg|kpa|hpa|pa|torr|atm)\b",
        flags=re.IGNORECASE,
    )
    unit_before_value = re.compile(
        rf"\b(mm\s*hg|mmhg|kpa|hpa|pa|torr|atm)\b[^:;]{{0,40}}:\s*({NUMBER})",
        flags=re.IGNORECASE,
    )
    for text in strings:
        temperature = _temperature_c(text)
        if temperature is not None and not 15.0 <= temperature <= 30.0:
            continue
        matches = [(match.group(1), match.group(2)) for match in value_before_unit.finditer(text)]
        matches.extend(
            (match.group(2), match.group(1)) for match in unit_before_value.finditer(text)
        )
        for raw_value, raw_unit in matches:
            value = _number(raw_value)
            if value is None or value <= 0:
                continue
            unit = re.sub(r"\s+", "", raw_unit.lower())
            pressure = value * unit_factors[unit]
            if not 1e-10 <= pressure <= 101_325.0:
                continue
            score = 4 if temperature is not None and 24.0 <= temperature <= 26.0 else (
                3 if temperature is not None else 2
            )
            candidates.append((score, pressure))
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    values = [value for score, value in candidates if score == best_score]
    return float(statistics.median(values))


def parse_boiling_point_c(strings: list[str]) -> float | None:
    candidates: list[float] = []
    pattern = re.compile(rf"({NUMBER})\s*(?:°|º|deg|∑)?\s*([CF])\b", re.IGNORECASE)
    for text in strings:
        for match in pattern.finditer(text):
            value = _number(match.group(1))
            if value is None:
                continue
            if match.group(2).upper() == "F":
                value = (value - 32.0) * 5.0 / 9.0
            if 20.0 <= value <= 650.0:
                candidates.append(value)
    return float(statistics.median(candidates)) if candidates else None


def parse_odor_threshold_ppm(strings: list[str], molecular_weight: float) -> float | None:
    candidates: list[tuple[int, float]] = []
    pattern = re.compile(
        rf"({NUMBER})\s*\]?\s*(ppm|ppb|mg\s*/\s*(?:m\^?3|cu\s*m))\b",
        flags=re.IGNORECASE,
    )
    for text in strings:
        lowered = text.lower()
        for match in pattern.finditer(text):
            value = _number(match.group(1))
            if value is None or value <= 0:
                continue
            unit = re.sub(r"\s+", "", match.group(2).lower())
            if unit == "ppb":
                value /= 1_000.0
            elif unit.startswith("mg/"):
                value = value * 24.45 / molecular_weight
            if not 1e-12 <= value <= 1_000_000.0:
                continue
            score = 3 if "low" in lowered or "detection" in lowered else 2
            candidates.append((score, value))
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    values = sorted(value for score, value in candidates if score == best_score)
    return float(math.exp(statistics.median([math.log(value) for value in values])))


def _optional_float(item: dict, key: str) -> float | None:
    return float(item[key]) if item.get(key) is not None else None


def _optional_int(item: dict, key: str) -> int | None:
    return int(item[key]) if item.get(key) is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--ingredient-id", action="append")
    parser.add_argument("--delay-seconds", type=float, default=0.15)
    args = parser.parse_args()
    catalog = IngredientCatalog.load_builtin()
    selected = set(args.ingredient_id or [])
    store = ScientificPropertyStore(args.db)
    imported = 0
    vapor_count = 0
    boiling_count = 0
    threshold_count = 0
    skipped: list[str] = []
    for ingredient in catalog.ingredients:
        if selected and ingredient.ingredient_id not in selected:
            continue
        if not ingredient.cas_number:
            skipped.append(f"{ingredient.ingredient_id}:missing_cas")
            continue
        try:
            item, descriptor_url = fetch(ingredient.cas_number)
            cid = int(item["CID"])
            molecular_weight = float(item["MolecularWeight"])
            source_urls = [descriptor_url]
            annotations: dict[str, list[str]] = {}
            for heading in ("Vapor Pressure", "Boiling Point", "Odor Threshold"):
                try:
                    values, url = fetch_annotations(cid, heading)
                    annotations[heading] = values
                    source_urls.append(url)
                except Exception:
                    annotations[heading] = []
                time.sleep(max(0.0, args.delay_seconds))
            vapor_pressure = parse_vapor_pressure_pa(annotations["Vapor Pressure"])
            boiling_point = parse_boiling_point_c(annotations["Boiling Point"])
            odor_threshold = parse_odor_threshold_ppm(
                annotations["Odor Threshold"], molecular_weight
            )
            store.upsert(
                MolecularProperties(
                    ingredient_id=ingredient.ingredient_id,
                    cas_number=ingredient.cas_number,
                    molecular_weight=molecular_weight,
                    xlogp=_optional_float(item, "XLogP"),
                    tpsa=_optional_float(item, "TPSA"),
                    hbond_donors=_optional_int(item, "HBondDonorCount"),
                    hbond_acceptors=_optional_int(item, "HBondAcceptorCount"),
                    rotatable_bonds=_optional_int(item, "RotatableBondCount"),
                    complexity=_optional_float(item, "Complexity"),
                    vapor_pressure_pa_25c=vapor_pressure,
                    boiling_point_c=boiling_point,
                    odor_threshold_ppm=odor_threshold,
                    source_ref="; ".join(source_urls),
                    verified_on=date.today().isoformat(),
                )
            )
            imported += 1
            vapor_count += vapor_pressure is not None
            boiling_count += boiling_point is not None
            threshold_count += odor_threshold is not None
        except Exception as error:
            skipped.append(f"{ingredient.ingredient_id}:{type(error).__name__}:{error}")
    store.close()
    print(
        json.dumps(
            {
                "imported": imported,
                "vapor_pressure_records": vapor_count,
                "boiling_point_records": boiling_count,
                "odor_threshold_records": threshold_count,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
