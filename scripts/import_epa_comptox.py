"""Download and compact official EPA CompTox bulk data for the safe catalog.

The bulk archives remain in a cache directory.  Only rows matching catalog CAS
numbers or their DSSTox substance identifiers are written to the packaged
SQLite extract.  Human sensory/panel data are not read by this pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.epa_comptox import EPACompToxStore  # noqa: E402


EPA_DOWNLOAD_PAGE = (
    "https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data"
)
OPEN_DATA_LICENSE_NOTE = (
    "US EPA CompTox open data: free of copyright restrictions and available for "
    "commercial and non-commercial use; attribution and source/version lineage retained"
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    dataset_name: str
    dataset_version: str
    file_name: str
    origin_uri: str
    download_uri: str
    expected_bytes: int
    expected_sha256: str | None = None
    optional: bool = False


SOURCES = (
    SourceSpec(
        "epa_dsstox_2025_12_csv",
        "Distributed Structure-Searchable Toxicity Database (DSSTox)",
        "December 2025 / Figshare version 8",
        "DSSTox_CCD_dump_12092025_CSVs.zip",
        "https://doi.org/10.23645/epacomptox.5588566.v8",
        "https://clowder.edap-cluster.com/api/files/69529775e4b0731a616efc4b?key=",
        289_824_966,
        "66fc9d4d3bda053ab60ee01e2ff0ed4ba030339588c40711b6939faede6cf4c3",
    ),
    SourceSpec(
        "epa_cpdat_v4_0",
        "Chemical and Products Database (CPDat)",
        "4.0 / Figshare version 5",
        "cpdat_v4.0.zip",
        "https://doi.org/10.23645/epacomptox.5352997.v5",
        "https://ndownloader.figshare.com/files/53538266",
        106_473_580,
        "53cd5b49210a6f3790be0c33485654de7461b4d1fdeea9fdc312668c82e93d15",
    ),
    SourceSpec(
        "epa_toxrefdb_v3_0_pod",
        "Toxicity Reference Database (ToxRefDB) POD",
        "3.0",
        "toxrefdb_3_0_pod.csv",
        "https://doi.org/10.23645/epacomptox.6062545.v5",
        "https://clowder.edap-cluster.com/api/files/688cbadae4b02565bc3f8c07?key=",
        27_730_838,
        "69dd2da06ffc6a98a1bed684c347d8b2f31e6d35d607d4f2f83ebb6845a708d5",
    ),
    SourceSpec(
        "epa_toxrefdb_v3_0_studies",
        "Toxicity Reference Database (ToxRefDB) study-chemical summary",
        "3.0",
        "toxrefdb_3_0_study_chem_summary.xlsx",
        "https://doi.org/10.23645/epacomptox.6062545.v5",
        "https://clowder.edap-cluster.com/api/files/689c9b5fe4b025654d12f40c?key=",
        1_975_486,
        "55a21431fcd5b2864eaa51822f94b4c26797a230b552c12da2a30421531b7918",
    ),
    SourceSpec(
        "epa_toxvaldb_v9_7_0_inputs",
        "Toxicity Values Database (ToxValDB) GitHub input files",
        "9.7.0 / Figshare version 11",
        "toxvaldb_v9.7.0_github_input_files.zip",
        "https://doi.org/10.23645/epacomptox.20394501.v11",
        "https://clowder.edap-cluster.com/api/files/68d3fa82e4b02565fc7dedb2?key=",
        1_416_929_280,
        "f5a125e036b5978140f4e4c413f2a8439789778ea01cee00d1d63bb2d4d567da",
        optional=True,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: SourceSpec, destination: Path, max_download_bytes: int) -> None:
    if spec.expected_bytes > max_download_bytes:
        raise ValueError(
            f"{spec.file_name} is {spec.expected_bytes} bytes, above --max-download-bytes"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing == spec.expected_bytes:
        if spec.expected_sha256 and sha256(destination) != spec.expected_sha256:
            raise ValueError(f"checksum mismatch for cached {spec.file_name}")
        return
    if existing > spec.expected_bytes:
        raise ValueError(f"cached file is larger than expected: {destination}")
    headers = {"User-Agent": "perfumery-ai-core/0.6 EPA-open-data-import"}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    request = urllib.request.Request(spec.download_uri, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(mode) as output:
        if existing and getattr(response, "status", None) != 206:
            output.close()
            destination.unlink(missing_ok=True)
            return download(spec, destination, max_download_bytes)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    if destination.stat().st_size != spec.expected_bytes:
        raise ValueError(
            f"incomplete download for {spec.file_name}: "
            f"{destination.stat().st_size}/{spec.expected_bytes}"
        )
    if spec.expected_sha256 and sha256(destination) != spec.expected_sha256:
        raise ValueError(f"checksum mismatch for downloaded {spec.file_name}")


def clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "na", "n/a", "nan", "none", "null"} else text


def numeric(value: object) -> float | None:
    text = clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def casrn(value: object) -> str | None:
    text = clean(value)
    return text if text and CAS_PATTERN.fullmatch(text) else None


def catalog_maps(catalog_path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_cas: dict[str, str] = {}
    by_id: dict[str, dict] = {}
    for row in payload["ingredients"]:
        ingredient_id = str(row["ingredient_id"])
        by_id[ingredient_id] = row
        normalized_cas = casrn(row.get("cas_number"))
        if normalized_cas:
            by_cas[normalized_cas] = ingredient_id
    return by_cas, by_id


def zip_csv_rows(archive: Path, suffix: str):
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"expected exactly one *{suffix} in {archive}, found {names}")
        with bundle.open(names[0]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                yield from csv.DictReader(text)


def xlsx_first_sheet_rows(path: Path):
    """Yield first-sheet rows using only XLSX's documented ZIP/XML structure."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as bundle:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in bundle.namelist():
            root = ElementTree.parse(bundle.open("xl/sharedStrings.xml")).getroot()
            for item in root.findall(f"{namespace}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ElementTree.parse(bundle.open(sheet_name)).getroot()
        matrix: list[list[object]] = []
        for row in root.iter(f"{namespace}row"):
            values: list[object] = []
            for cell in row.findall(f"{namespace}c"):
                reference = cell.attrib.get("r", "A1")
                letters = "".join(character for character in reference if character.isalpha())
                column = 0
                for letter in letters:
                    column = column * 26 + ord(letter.upper()) - 64
                while len(values) < column:
                    values.append(None)
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{namespace}v")
                value: object = value_node.text if value_node is not None else None
                if cell_type == "s" and value is not None:
                    value = shared[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{namespace}t")
                    )
                elif cell_type == "b" and value is not None:
                    value = value == "1"
                values[column - 1] = value
            matrix.append(values)
        if not matrix:
            return
        headers: list[str] = []
        seen: dict[str, int] = {}
        for index, raw_header in enumerate(matrix[0]):
            header = clean(raw_header) or f"column_{index + 1}"
            seen[header] = seen.get(header, 0) + 1
            headers.append(header if seen[header] == 1 else f"{header}_{seen[header]}")
        for values in matrix[1:]:
            values.extend([None] * (len(headers) - len(values)))
            yield dict(zip(headers, values))


def _xlsx_shared_strings(bundle: zipfile.ZipFile) -> list[str]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    if "xl/sharedStrings.xml" not in bundle.namelist():
        return []
    strings: list[str] = []
    with bundle.open("xl/sharedStrings.xml") as handle:
        for _event, element in ElementTree.iterparse(handle, events=("end",)):
            if element.tag == f"{namespace}si":
                strings.append(
                    "".join(node.text or "" for node in element.iter(f"{namespace}t"))
                )
                element.clear()
    return strings


def _xlsx_sheet_paths(bundle: zipfile.ZipFile) -> list[tuple[str, str]]:
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_doc_ns = (
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    )
    rel_pkg_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    try:
        workbook = ElementTree.parse(bundle.open("xl/workbook.xml")).getroot()
        relationships = ElementTree.parse(
            bundle.open("xl/_rels/workbook.xml.rels")
        ).getroot()
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{rel_pkg_ns}Relationship")
        }
        result: list[tuple[str, str]] = []
        for sheet in workbook.iter(f"{main_ns}sheet"):
            target = targets.get(sheet.attrib.get(f"{rel_doc_ns}id", ""))
            if not target:
                continue
            normalized = target.replace("\\", "/").lstrip("/")
            if not normalized.startswith("xl/"):
                normalized = "xl/" + normalized
            result.append((sheet.attrib.get("name", normalized), normalized))
        if result:
            return result
        return [
            (Path(name).stem, name)
            for name in bundle.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
    except (KeyError, ElementTree.ParseError):
        return [
            (Path(name).stem, name)
            for name in bundle.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]


def _column_index(reference: str) -> int:
    column = 0
    for letter in (character for character in reference if character.isalpha()):
        column = column * 26 + ord(letter.upper()) - 64
    return max(0, column - 1)


def xlsx_sheet_rows(path: Path):
    """Stream every worksheet as ``(sheet, row_number, values)`` tuples."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as bundle:
        shared = _xlsx_shared_strings(bundle)
        for sheet_name, sheet_path in _xlsx_sheet_paths(bundle):
            if sheet_path not in bundle.namelist():
                continue
            with bundle.open(sheet_path) as handle:
                for _event, row in ElementTree.iterparse(handle, events=("end",)):
                    if row.tag != f"{namespace}row":
                        continue
                    values: list[object] = []
                    for cell in row.findall(f"{namespace}c"):
                        column = _column_index(cell.attrib.get("r", "A1"))
                        while len(values) <= column:
                            values.append(None)
                        cell_type = cell.attrib.get("t")
                        value_node = cell.find(f"{namespace}v")
                        value: object = value_node.text if value_node is not None else None
                        if cell_type == "s" and value is not None:
                            index = int(value)
                            value = shared[index] if index < len(shared) else value
                        elif cell_type == "inlineStr":
                            value = "".join(
                                node.text or ""
                                for node in cell.iter(f"{namespace}t")
                            )
                        elif cell_type == "b" and value is not None:
                            value = value == "1"
                        values[column] = value
                    yield sheet_name, int(row.attrib.get("r", "0") or 0), values
                    row.clear()


def insert_source(
    connection: sqlite3.Connection,
    spec: SourceSpec,
    path: Path | None,
    record_count: int,
    status: str,
) -> None:
    connection.execute(
        """INSERT OR REPLACE INTO source_files VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            spec.source_id,
            spec.dataset_name,
            spec.dataset_version,
            spec.file_name,
            spec.origin_uri,
            spec.download_uri,
            sha256(path) if path and path.exists() else "",
            path.stat().st_size if path and path.exists() else spec.expected_bytes,
            date.today().isoformat(),
            record_count,
            status,
            OPEN_DATA_LICENSE_NOTE,
        ),
    )


def ingest_dsstox(
    connection: sqlite3.Connection,
    path: Path,
    by_cas: dict[str, str],
    catalog: dict[str, dict],
) -> tuple[int, dict[str, str]]:
    rows: list[tuple] = []
    dtxsid_to_ingredient: dict[str, str] = {}
    for row in zip_csv_rows(path, "/DSSToxCCDdump.csv"):
        normalized_cas = casrn(row.get("CASRN"))
        ingredient_id = by_cas.get(normalized_cas or "")
        dtxsid = clean(row.get("DTXSID"))
        if not ingredient_id or not normalized_cas or not dtxsid:
            continue
        item = catalog[ingredient_id]
        rows.append(
            (
                ingredient_id,
                str(item["name"]),
                normalized_cas,
                dtxsid,
                clean(row.get("DTXCID")),
                clean(row.get("PREFERRED_NAME")),
                clean(row.get("INCHIKEY")),
                clean(row.get("IUPAC_NAME")),
                clean(row.get("SMILES")),
                clean(row.get("MOLECULAR_FORMULA")),
                numeric(row.get("AVERAGE_MASS")),
                numeric(row.get("MONOISOTOPIC_MASS")),
                clean(row.get("QSAR_READY_SMILES")),
                clean(row.get("MS_READY_SMILES")),
            )
        )
        dtxsid_to_ingredient[dtxsid] = ingredient_id
    connection.executemany(
        "INSERT OR REPLACE INTO chemicals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows), dtxsid_to_ingredient


CPDAT_SUFFIXES = {
    "functional_use": "/cpdat_v4.0_functional_use_data.csv",
    "list_presence": "/cpdat_v4.0_list_presence_data.csv",
    "product_composition": "/cpdat_v4.0_product_composition_data.csv",
}


def ingest_cpdat(
    connection: sqlite3.Connection,
    path: Path,
    by_cas: dict[str, str],
    dtxsid_to_ingredient: dict[str, str],
) -> int:
    sql = """INSERT INTO cpdat_observations (
        record_type, ingredient_id, dtxsid, curated_casrn, curated_chemical_name,
        data_source, data_source_url, data_document_id, data_document_title,
        data_document_url, data_document_date, organization, raw_functional_use,
        function_category, product_id, product_title, puc_kind, puc_general_category,
        puc_product_family, puc_product_type, lower_weight_fraction,
        upper_weight_fraction, central_weight_fraction, component_name, keyword_set
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    total = 0
    batch: list[tuple] = []
    for record_type, suffix in CPDAT_SUFFIXES.items():
        for row in zip_csv_rows(path, suffix):
            dtxsid = clean(row.get("dtxsid"))
            curated_cas = casrn(row.get("curated_casrn"))
            raw_cas = casrn(row.get("raw_casrn"))
            ingredient_id = (
                by_cas.get(curated_cas or "")
                or by_cas.get(raw_cas or "")
                or dtxsid_to_ingredient.get(dtxsid or "")
            )
            if not ingredient_id:
                continue
            batch.append(
                (
                    record_type,
                    ingredient_id,
                    dtxsid,
                    curated_cas or raw_cas,
                    clean(row.get("curated_chemical_name")) or clean(row.get("raw_chemical_name")),
                    clean(row.get("data_source")),
                    clean(row.get("data_source_url")),
                    clean(row.get("cpdat_data_document_id")),
                    clean(row.get("data_document_title")),
                    clean(row.get("data_document_url")),
                    clean(row.get("data_document_date")),
                    clean(row.get("organization")),
                    clean(row.get("raw_functional_use")),
                    clean(row.get("function_category")),
                    clean(row.get("cpdat_product_id")),
                    clean(row.get("product_title")),
                    clean(row.get("puc_kind")),
                    clean(row.get("puc_general_category")),
                    clean(row.get("puc_product_family")),
                    clean(row.get("puc_product_type")),
                    numeric(row.get("lower_weight_fraction")),
                    numeric(row.get("upper_weight_fraction")),
                    numeric(row.get("central_weight_fraction")),
                    clean(row.get("component_name")),
                    clean(row.get("keyword_set")),
                )
            )
            if len(batch) >= 2_000:
                connection.executemany(sql, batch)
                total += len(batch)
                batch.clear()
        if batch:
            connection.executemany(sql, batch)
            total += len(batch)
            batch.clear()
        connection.commit()
    return total


def ingest_toxref_studies(
    connection: sqlite3.Connection,
    path: Path,
    by_cas: dict[str, str],
    dtxsid_to_ingredient: dict[str, str],
) -> int:
    rows: list[tuple] = []
    for row in xlsx_first_sheet_rows(path):
        dtxsid = clean(row.get("dsstox_substance_id"))
        normalized_cas = casrn(row.get("casrn"))
        ingredient_id = (
            dtxsid_to_ingredient.get(dtxsid or "") or by_cas.get(normalized_cas or "")
        )
        if not ingredient_id or not dtxsid:
            continue
        rows.append(
            (
                ingredient_id,
                clean(row.get("chemical_id")),
                dtxsid,
                normalized_cas,
                clean(row.get("preferred_name")),
                clean(row.get("study_id")),
                clean(row.get("study_source_id")),
                clean(row.get("study_citation")),
                clean(row.get("study_year")),
                clean(row.get("study_source")),
                clean(row.get("study_type")),
                clean(row.get("study_type_guideline")),
                clean(row.get("species")),
                clean(row.get("strain_group")),
                clean(row.get("strain")),
                clean(row.get("admin_route")),
                clean(row.get("admin_method")),
                numeric(row.get("dose_start")),
                clean(row.get("dose_start_unit")),
                numeric(row.get("dose_end")),
                clean(row.get("dose_end_unit")),
                clean(row.get("study_comment")),
                clean(row.get("guideline_id")),
                clean(row.get("processed")),
            )
        )
    connection.executemany(
        """INSERT INTO toxref_studies (
        ingredient_id, chemical_id, dtxsid, casrn, preferred_name, study_id,
        study_source_id, study_citation, study_year, study_source, study_type,
        study_type_guideline, species, strain_group, strain, admin_route,
        admin_method, dose_start, dose_start_unit, dose_end, dose_end_unit,
        study_comment, guideline_id, processed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def ingest_toxref_pods(
    connection: sqlite3.Connection,
    path: Path,
    dtxsid_to_ingredient: dict[str, str],
) -> int:
    rows: list[tuple] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dtxsid = clean(row.get("dsstox_substance_id"))
            ingredient_id = dtxsid_to_ingredient.get(dtxsid or "")
            if not ingredient_id or not dtxsid:
                continue
            rows.append(
                (
                    ingredient_id,
                    clean(row.get("study_id")),
                    clean(row.get("study_type")),
                    clean(row.get("preferred_name")),
                    dtxsid,
                    clean(row.get("toxval_study_source_id")),
                    clean(row.get("toxval_effect_list")),
                    clean(row.get("dose_level")),
                    clean(row.get("calc_pod_type")),
                    clean(row.get("qualifier")),
                    numeric(row.get("mg_kg_day_value")),
                    clean(row.get("admin_route")),
                    clean(row.get("admin_method")),
                    clean(row.get("vehicle")),
                    clean(row.get("species")),
                    clean(row.get("strain_group")),
                    clean(row.get("strain")),
                    numeric(row.get("dose_start")),
                    clean(row.get("dose_start_unit")),
                    numeric(row.get("dose_end")),
                    clean(row.get("dose_end_unit")),
                    clean(row.get("study_year")),
                    clean(row.get("study_citation")),
                )
            )
    connection.executemany(
        """INSERT INTO toxref_pods (
        ingredient_id, study_id, study_type, preferred_name, dtxsid,
        toxval_study_source_id, toxval_effect_list, dose_level, calc_pod_type,
        qualifier, mg_kg_day_value, admin_route, admin_method, vehicle, species,
        strain_group, strain, dose_start, dose_start_unit, dose_end, dose_end_unit,
        study_year, study_citation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def _normalized_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _identifier_header(value: object) -> bool:
    header = _normalized_header(value)
    return (
        header in {
            "cas", "casrn", "casno", "casnumber", "casregistrynumber",
            "chemicalcas", "chemicalcasrn", "dtxsid", "dsstoxsubstanceid",
        }
        or header.startswith("dtxsid")
        or header.startswith("dsstoxsubstance")
    )


def _deduplicated_headers(values: list[object]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(values):
        header = clean(value) or f"column_{index + 1}"
        seen[header] = seen.get(header, 0) + 1
        headers.append(header if seen[header] == 1 else f"{header}_{seen[header]}")
    return headers


def _record_field(record: dict[str, object], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalized_header(key): clean(value) for key, value in record.items()}
    for candidate in candidates:
        if normalized.get(candidate):
            return normalized[candidate]
    for key, value in normalized.items():
        if value and any(candidate in key for candidate in candidates):
            return value
    return None


CAS_FIND_PATTERN = re.compile(r"(?<!\d)(\d{2,7}-\d{2}-\d)(?!\d)")
DTXSID_FIND_PATTERN = re.compile(r"DTXSID\d+", re.IGNORECASE)


def _matched_ingredients(
    values: object,
    by_cas: dict[str, str],
    dtxsid_to_ingredient: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    ingredients: set[str] = set()
    found_cas: set[str] = set()
    found_dtxsid: set[str] = set()
    for value in values:
        text = clean(value)
        if not text:
            continue
        for match in CAS_FIND_PATTERN.finditer(text):
            identifier = match.group(1)
            ingredient_id = by_cas.get(identifier)
            if ingredient_id:
                ingredients.add(ingredient_id)
                found_cas.add(identifier)
        for match in DTXSID_FIND_PATTERN.finditer(text):
            identifier = match.group(0).upper()
            ingredient_id = dtxsid_to_ingredient.get(identifier)
            if ingredient_id:
                ingredients.add(ingredient_id)
                found_dtxsid.add(identifier)
    return ingredients, found_cas, found_dtxsid


def ingest_toxval(
    connection: sqlite3.Connection,
    path: Path,
    by_cas: dict[str, str],
    catalog: dict[str, dict],
    dtxsid_to_ingredient: dict[str, str],
) -> tuple[int, int, int]:
    """Extract direct CAS/DTXSID matches from every ToxValDB input workbook.

    ToxValDB v9.7.0's downloadable ``GitHub Input Files`` archive contains
    heterogeneous source workbooks rather than one normalized table.  This
    scanner therefore preserves each directly matched source row as JSON and
    does not infer joins through undocumented source-specific identifiers.
    """
    sql = """INSERT INTO toxval_observations (
        ingredient_id, source_table, source_record_id, dtxsid, casrn,
        preferred_name, effect_category, effect_text, study_type, species,
        value_numeric, value_unit, raw_record_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    total = 0
    scanned = 0
    failed = 0
    csv_scanned = 0
    csv_failed = 0
    xls_scanned = 0
    xls_failed = 0
    xls_unparsed = 0
    failures: list[str] = []
    batch: list[tuple] = []
    ingredient_to_dtxsid = {
        ingredient_id: dtxsid
        for dtxsid, ingredient_id in dtxsid_to_ingredient.items()
    }

    def queue_record(
        record: dict[str, object], source_table: str, source_record_id: str
    ) -> None:
        ingredient_ids, found_cas, found_dtxsids = _matched_ingredients(
            record.values(), by_cas, dtxsid_to_ingredient
        )
        if not ingredient_ids:
            return
        compact_record = {
            key: value for key, value in record.items() if clean(value) is not None
        }
        raw_json = json.dumps(
            compact_record, ensure_ascii=False, sort_keys=True, default=str
        )
        preferred_name = _record_field(
            record,
            (
                "preferredname", "curatedchemicalname", "chemicalname",
                "substancename", "compoundname", "name",
            ),
        )
        effect_category = _record_field(
            record,
            (
                "effectcategory", "effecttype", "toxvaltype",
                "endpointcategory", "endpoint",
            ),
        )
        effect_text = _record_field(
            record, ("criticaleffect", "toxvaleffect", "effectdescription", "effect")
        )
        study_type = _record_field(
            record, ("studytype", "studytypedetail", "studycategory")
        )
        species = _record_field(record, ("speciescommon", "testspecies", "species"))
        raw_value = _record_field(
            record,
            (
                "toxvalnumeric", "effectlevelvalue", "podvalue",
                "dosevalue", "numericvalue",
            ),
        )
        value_unit = _record_field(
            record, ("toxvalunits", "effectlevelunits", "doseunits", "units")
        )
        for ingredient_id in sorted(ingredient_ids):
            catalog_cas = casrn(catalog[ingredient_id].get("cas_number"))
            direct_dtxsids = sorted(
                dtxsid for dtxsid in found_dtxsids
                if dtxsid_to_ingredient.get(dtxsid) == ingredient_id
            )
            mapped_dtxsid = direct_dtxsids[0] if direct_dtxsids else (
                ingredient_to_dtxsid.get(ingredient_id)
                if catalog_cas in found_cas else None
            )
            batch.append(
                (
                    ingredient_id,
                    source_table,
                    source_record_id,
                    mapped_dtxsid,
                    catalog_cas,
                    preferred_name,
                    effect_category,
                    effect_text,
                    study_type,
                    species,
                    numeric(raw_value),
                    value_unit,
                    raw_json,
                )
            )
    with zipfile.ZipFile(path) as outer, tempfile.TemporaryDirectory(
        prefix="toxvaldb_scan_"
    ) as temporary:
        workbooks = [
            info for info in outer.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".xlsx")
        ]
        for index, info in enumerate(workbooks, start=1):
            temporary_path = Path(temporary) / f"workbook_{index:04d}.xlsx"
            try:
                with outer.open(info) as source, temporary_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                scanned += 1
                active_sheet: str | None = None
                headers: list[str] | None = None
                header_row = 0
                header_search_exhausted = False
                for sheet, row_number, values in xlsx_sheet_rows(temporary_path):
                    if sheet != active_sheet:
                        active_sheet = sheet
                        headers = None
                        header_row = 0
                        header_search_exhausted = False
                    if headers is None:
                        if row_number <= 200 and any(_identifier_header(value) for value in values):
                            headers = _deduplicated_headers(values)
                            header_row = row_number
                        elif row_number > 200:
                            header_search_exhausted = True
                        continue
                    if header_search_exhausted or row_number <= header_row:
                        continue
                    values.extend([None] * (len(headers) - len(values)))
                    record = dict(zip(headers, values[:len(headers)]))
                    queue_record(record, f"{info.filename}#{sheet}", str(row_number))
                    if len(batch) >= 500:
                        connection.executemany(sql, batch)
                        total += len(batch)
                        batch.clear()
                if batch:
                    connection.executemany(sql, batch)
                    total += len(batch)
                    batch.clear()
                if index % 25 == 0:
                    connection.commit()
                    print(
                        f"ToxValDB workbooks: {index}/{len(workbooks)}, matched rows: {total}",
                        file=sys.stderr,
                    )
            except Exception as error:  # one malformed upstream workbook must not abort the release
                failed += 1
                failures.append(f"{info.filename}: {type(error).__name__}: {error}")
            finally:
                temporary_path.unlink(missing_ok=True)
        csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
        csv_files = [
            info for info in outer.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".csv")
        ]
        for info in csv_files:
            try:
                csv_scanned += 1
                with outer.open(info) as binary:
                    with io.TextIOWrapper(
                        binary, encoding="utf-8-sig", errors="replace", newline=""
                    ) as text:
                        reader = csv.DictReader(text)
                        if not reader.fieldnames or not any(
                            _identifier_header(header) for header in reader.fieldnames
                        ):
                            continue
                        for row_number, record in enumerate(reader, start=2):
                            queue_record(
                                dict(record), f"{info.filename}#csv", str(row_number)
                            )
                            if len(batch) >= 500:
                                connection.executemany(sql, batch)
                                total += len(batch)
                                batch.clear()
                if batch:
                    connection.executemany(sql, batch)
                    total += len(batch)
                    batch.clear()
            except Exception as error:  # preserve other source files if one CSV is malformed
                csv_failed += 1
                failures.append(f"{info.filename}: {type(error).__name__}: {error}")
        xls_files = [
            info for info in outer.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".xls")
        ]
        try:
            import xlrd  # type: ignore[import-not-found]
        except ImportError:
            xlrd = None
        if xlrd is None:
            xls_unparsed = len(xls_files)
            failures.extend(
                f"{info.filename}: xlrd optional data dependency is not installed"
                for info in xls_files
            )
        else:
            for info in xls_files:
                try:
                    book = xlrd.open_workbook(file_contents=outer.read(info), on_demand=True)
                    xls_scanned += 1
                    for sheet in book.sheets():
                        headers: list[str] | None = None
                        header_row = -1
                        for row_index in range(sheet.nrows):
                            values = sheet.row_values(row_index)
                            if headers is None:
                                if row_index < 200 and any(
                                    _identifier_header(value) for value in values
                                ):
                                    headers = _deduplicated_headers(values)
                                    header_row = row_index
                                elif row_index >= 200:
                                    break
                                continue
                            if row_index <= header_row:
                                continue
                            values.extend([None] * (len(headers) - len(values)))
                            queue_record(
                                dict(zip(headers, values[:len(headers)])),
                                f"{info.filename}#{sheet.name}",
                                str(row_index + 1),
                            )
                            if len(batch) >= 500:
                                connection.executemany(sql, batch)
                                total += len(batch)
                                batch.clear()
                    book.release_resources()
                    if batch:
                        connection.executemany(sql, batch)
                        total += len(batch)
                        batch.clear()
                except Exception as error:
                    xls_failed += 1
                    failures.append(f"{info.filename}: {type(error).__name__}: {error}")
    if batch:
        connection.executemany(sql, batch)
        total += len(batch)
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_workbooks_scanned', ?)",
        (str(scanned),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_workbooks_failed', ?)",
        (str(failed),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_csv_files_scanned', ?)",
        (str(csv_scanned),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_csv_files_failed', ?)",
        (str(csv_failed),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_xls_files_scanned', ?)",
        (str(xls_scanned),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_xls_files_failed', ?)",
        (str(xls_failed),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_xls_files_unparsed', ?)",
        (str(xls_unparsed),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('toxval_scan_failures', ?)",
        (json.dumps(failures, ensure_ascii=False),),
    )
    return total, scanned, failed + csv_failed + xls_failed + xls_unparsed


def build_extract(
    cache_dir: Path,
    output: Path,
    include_toxval: bool,
) -> dict[str, int | str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    connection = sqlite3.connect(output)
    EPACompToxStore.initialize(connection)
    by_cas, catalog = catalog_maps(
        ROOT / "fragrance_ai" / "data" / "safe_ingredient_catalog.json"
    )
    spec = {item.source_id: item for item in SOURCES}
    paths = {item.source_id: cache_dir / item.file_name for item in SOURCES}

    dsstox_count, dtxsid_map = ingest_dsstox(
        connection, paths["epa_dsstox_2025_12_csv"], by_cas, catalog
    )
    insert_source(
        connection, spec["epa_dsstox_2025_12_csv"],
        paths["epa_dsstox_2025_12_csv"], dsstox_count, "catalog_filtered",
    )
    cpdat_count = ingest_cpdat(
        connection, paths["epa_cpdat_v4_0"], by_cas, dtxsid_map
    )
    insert_source(
        connection, spec["epa_cpdat_v4_0"], paths["epa_cpdat_v4_0"],
        cpdat_count, "catalog_filtered",
    )
    study_count = ingest_toxref_studies(
        connection, paths["epa_toxrefdb_v3_0_studies"], by_cas, dtxsid_map
    )
    insert_source(
        connection, spec["epa_toxrefdb_v3_0_studies"],
        paths["epa_toxrefdb_v3_0_studies"], study_count, "catalog_filtered",
    )
    pod_count = ingest_toxref_pods(
        connection, paths["epa_toxrefdb_v3_0_pod"], dtxsid_map
    )
    insert_source(
        connection, spec["epa_toxrefdb_v3_0_pod"],
        paths["epa_toxrefdb_v3_0_pod"], pod_count, "catalog_filtered",
    )

    toxval_spec = spec["epa_toxvaldb_v9_7_0_inputs"]
    toxval_path = paths[toxval_spec.source_id]
    if include_toxval and toxval_path.exists():
        toxval_count, _workbooks_scanned, workbooks_failed = ingest_toxval(
            connection, toxval_path, by_cas, catalog, dtxsid_map
        )
        status = (
            "catalog_filtered_direct_identifier_rows"
            if workbooks_failed == 0
            else "catalog_filtered_with_parse_failures"
        )
        insert_source(connection, toxval_spec, toxval_path, toxval_count, status)
    else:
        insert_source(connection, toxval_spec, None, 0, "metadata_only_size_guard")
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('catalog_cas_count', ?)",
        (str(len(by_cas)),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('human_sensory_data_included', '0')"
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('allowed_use', ?)",
        (
            "Identifier/structure linkage, product-use evidence, non-human toxicology "
            "screening and model-feature research only",
        ),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('prohibited_use', ?)",
        (
            "Human odor-similarity certification, automatic safety approval, IFRA/legal "
            "compliance, supplier qualification or commercial release",
        ),
    )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()
    store = EPACompToxStore(output)
    result = store.stats()
    store.close()
    result.update({"output": str(output.resolve()), "sha256": sha256(output)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a catalog-filtered EPA CompTox open-data extract"
    )
    parser.add_argument(
        "--cache-dir", default=str(ROOT / ".cache" / "epa_comptox")
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "fragrance_ai" / "data" / "epa_comptox_extract.db"),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--include-toxval", action="store_true")
    parser.add_argument("--max-download-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    selected = [source for source in SOURCES if args.include_toxval or not source.optional]
    if not args.skip_download:
        for source in selected:
            download(source, cache_dir / source.file_name, args.max_download_bytes)
    for source in selected:
        path = cache_dir / source.file_name
        if not path.exists() or path.stat().st_size != source.expected_bytes:
            parser.error(f"missing or incomplete source file: {path}")
    if args.download_only:
        print(json.dumps({
            "download_page": EPA_DOWNLOAD_PAGE,
            "files": [
                {
                    "file": source.file_name,
                    "bytes": (cache_dir / source.file_name).stat().st_size,
                    "sha256": sha256(cache_dir / source.file_name),
                }
                for source in selected
            ],
        }, ensure_ascii=False, indent=2))
        return
    result = build_extract(cache_dir, Path(args.output), args.include_toxval)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
