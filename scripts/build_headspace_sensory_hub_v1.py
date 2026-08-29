#!/usr/bin/env python
"""Build a source-bound concentration, headspace, and sensory research hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import inchi


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "headspace-sensory-hub/v1"
PYRFUME_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
OPERA_URI = (
    "https://gaftp.epa.gov/COMPTOX/NCCT_Publication_Data/"
    "Williams_A/Opera_Model_Paper/S1.zip"
)
OPERA_DOI = "10.23645/epacomptox.5588512.v1"
OPERA_SHA256 = "84a51d3615f61c6d752a0d0cb1254fa73ff00c9a3103f1830c695864d2ff1b7c"
OPERA_BYTES = 15_303_706
PUBCHEM_SUPPLEMENT_SHA256 = (
    "87f34f2c7758de7aeded1a6db786c0ba7a2778c0899850294fcec294bef1609c"
)
PYRFUME_FILES = {
    "LICENSE": "fe7593299c9a8eebd15067f2befdac884c523d01a9017883a18aa4a6542d3142",
    "keller_2016/manifest.toml": "2281ff593fef1c5f0ed5226c9b76d56b0fa674f00c6454cc1c6cdf3b7e9ebd9a",
    "keller_2016/molecules.csv": "5f862b6ea62dd3ee0a15138605ea2567d06a527263284348a5982844f9bec81f",
    "keller_2016/stimuli.csv": "49c1a6bfb8c1665a31e13f515add95a95291f29416be1bb0b54d3317efb6665b",
    "keller_2016/behavior.csv": "ab31c17c475c8b471706822f89aaf9e148bebe5f50f33ff03ebc42b32ac94df8",
    "ravia_2020/manifest.toml": "9de2e67e29b4630b582978ff535a8bacdf08eab943b97d277db7818a6aa2084c",
    "ravia_2020/molecules.csv": "176a503411a3f14bff549d98ac68d88e9325fc2b489034384ee4f9a553793cda",
    "ravia_2020/stimuli.csv": "9cae42bb1a8d387b0b2a388a9d442fd1709f7d3cf40381a5d140a6ff08e372c0",
    "ravia_2020/behavior_1.csv": "644d69e3a5246738558255b54139d680b1f725525529f8cb19d59bd5b02a7e90",
    "ravia_2020/behavior_2.csv": "c24cb423dc88514a1d5425a57fca37d592a8d1dc1254aa758fb960ceb0cd1117",
    "ravia_2020/behavior_3.csv": "3ba1b6da75c0d2428199cccb0412a1812db145943a60bb32bbf4ecd3c5764d13",
    "ma_2021/manifest.toml": "b4f0827e32a5df50d58ba996e3354ff9b8b911b0db2fc212b9923c88f554893c",
    "ma_2021/molecules.csv": "f78215d755113fdfc034b10ca6419eecc5c424db9f6abfa24ad6ebc5ab7fc970",
    "ma_2021/stimuli.csv": "85ef675c43b670eab17c6e727a9cda19a0c3c51adc4e35cdffa8668c1c30ad78",
    "ma_2021/behavior.csv": "75556c1bb393381644d84ec7f59fdb75658b7463db972f1e81b77559af11b456",
    "bushdid_2014/manifest.toml": "38f117c4eadc77012b30232c8735ab8c95f3b64920d8ae23192ad000c05136e7",
    "bushdid_2014/molecules.csv": "84f19e8813953187156b92e4c34cf380622a2ec5307c25eedc9c100a7544906c",
    "bushdid_2014/stimuli.csv": "b84afd27b9a77abeff0813c1834efed27558a8573b8e39c85219235def4f8d87",
    "bushdid_2014/behavior.csv": "6aa3b7314591db378bd1b3c414a310f5682804ed4629eb6bb77a953c99ae5e36",
    "snitz_2013/manifest.toml": "3f2823e9dd877a31a8ba055e189d59cb1a888355c75fcd6a2a2cf8c4ffddcfe3",
    "snitz_2013/molecules.csv": "d8d34819ec7857b87d9399c7846b894da2fa727ba7a18ae554a0e2f49ab17920",
    "snitz_2013/stimuli.csv": "8a6b08f5b1c6f2ea84b125adcc32c3584b584bad56549db950f53890b8635937",
    "snitz_2013/behavior.csv": "d931f21c501158e045d9af450b432c549ade432695e627dd3bd2dbcdee750396",
    "abraham_2012/manifest.toml": "d0f261348b543c7a9a018ddf7a650fc3b0bea72ee879fafcd8679c0e4ffb68d2",
    "abraham_2012/molecules.csv": "d575918f6bdd03c1ba32bc2bbdcac24a5e0eebece54a172360c9cc32d653e000",
    "abraham_2012/stimuli.csv": "6ed2095e5f863b2fafe4df8cc232e5942f58798e28c39422bfc087aa0f597762",
    "abraham_2012/behavior.csv": "db36689c889c88f7df7df2643cfb79721d12b456ddc38368b97674dc742f7ab5",
}
ARCHIVES = (
    "keller_2016",
    "ravia_2020",
    "ma_2021",
    "bushdid_2014",
    "snitz_2013",
    "abraham_2012",
)
OPERA_ENDPOINTS = (
    (
        "log10_vapor_pressure_mmhg",
        "LogVP",
        "log10(mmHg)",
        "OPERA_VP/TR_VP_2034.sdf",
        "train",
    ),
    (
        "log10_vapor_pressure_mmhg",
        "LogVP",
        "log10(mmHg)",
        "OPERA_VP/TST_VP_679.sdf",
        "test",
    ),
    (
        "log10_henry_atm_m3_mol",
        "LogHL",
        "log10(atm*m^3/mol)",
        "OPERA_HL/TR_HL_441.sdf",
        "train",
    ),
    (
        "log10_henry_atm_m3_mol",
        "LogHL",
        "log10(atm*m^3/mol)",
        "OPERA_HL/TST_HL_150.sdf",
        "test",
    ),
    (
        "boiling_point_c",
        "BP",
        "degC",
        "OPERA_BP/TR_BP_4077.sdf",
        "train",
    ),
    (
        "boiling_point_c",
        "BP",
        "degC",
        "OPERA_BP/TST_BP_1358.sdf",
        "test",
    ),
    (
        "log10_octanol_air_partition",
        "LogKOA",
        "log10(Koa)",
        "OPERA_KOA/TR_KOA_202.sdf",
        "train",
    ),
    (
        "log10_octanol_air_partition",
        "LogKOA",
        "log10(Koa)",
        "OPERA_KOA/TST_KOA_68.sdf",
        "test",
    ),
    (
        "log10_octanol_water_partition",
        "LogP",
        "log10(Kow)",
        "OPERA_LogP/TR_LogP_10537.sdf",
        "train",
    ),
    (
        "log10_octanol_water_partition",
        "LogP",
        "log10(Kow)",
        "OPERA_LogP/TST_LogP_3513.sdf",
        "test",
    ),
    (
        "log10_water_solubility_mol_l",
        "LogMolar",
        "log10(mol/L)",
        "OPERA_WS/TR_WS_3158.sdf",
        "train",
    ),
    (
        "log10_water_solubility_mol_l",
        "LogMolar",
        "log10(mol/L)",
        "OPERA_WS/TST_WS_1066.sdf",
        "test",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "nan", "none", "null", "n/a"} else text


def first_clean(*values: Any) -> str | None:
    """Return the first real string without treating pandas NaN as truthy."""
    for value in values:
        result = clean(value)
        if result is not None:
            return result
    return None


def binary(value: Any) -> float:
    """Parse an explicit binary field and reject ambiguous truthiness."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        number = finite(value)
        if number in (0.0, 1.0):
            return float(number)
    text = clean(value)
    if text is not None:
        normalized = text.casefold()
        if normalized in {"true", "yes", "correct", "1"}:
            return 1.0
        if normalized in {"false", "no", "incorrect", "0"}:
            return 0.0
    raise ValueError(f"expected an explicit binary value, received {value!r}")


def parse_dilution(value: Any) -> tuple[float | None, str | None, str]:
    """Normalize an explicit Ravia dilution while retaining its raw token."""
    text = clean(value)
    if text is None or text == "-":
        return None, None, "missing"
    if text.casefold() == "solid":
        return None, "solid_material", "non_numeric_solid"
    if text.endswith("%"):
        numeric = finite(text[:-1])
        status = "parsed_percent"
        result = None if numeric is None else numeric / 100.0
    else:
        result = finite(text)
        status = "parsed_fraction"
    if result is None or not 0.0 <= result <= 1.0:
        raise ValueError(f"invalid dilution token: {value!r}")
    return result, "fraction_v_v", status


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()


def verify_sources(
    pyrfume_root: Path, opera_zip: Path, pubchem_supplement: Path
) -> dict[str, Any]:
    if git_commit(pyrfume_root) != PYRFUME_COMMIT:
        raise RuntimeError("unsupported Pyrfume data commit")
    changed = [
        relative
        for relative, expected in PYRFUME_FILES.items()
        if not (pyrfume_root / relative).is_file()
        or sha256(pyrfume_root / relative) != expected
    ]
    if changed:
        raise RuntimeError("Pyrfume source bytes changed: " + ", ".join(changed))
    if (
        not opera_zip.is_file()
        or opera_zip.stat().st_size != OPERA_BYTES
        or sha256(opera_zip) != OPERA_SHA256
    ):
        raise RuntimeError("EPA OPERA archive bytes changed")
    if (
        not pubchem_supplement.is_file()
        or sha256(pubchem_supplement) != PUBCHEM_SUPPLEMENT_SHA256
    ):
        raise RuntimeError("PubChem molecule supplement bytes changed")
    with zipfile.ZipFile(opera_zip) as bundle:
        missing = [path for *_prefix, path, _split in OPERA_ENDPOINTS if path not in bundle.namelist()]
    if missing:
        raise RuntimeError("EPA OPERA archive is missing endpoints: " + ", ".join(missing))
    return {
        "pyrfume_repository": "https://github.com/pyrfume/pyrfume-data",
        "pyrfume_commit": PYRFUME_COMMIT,
        "pyrfume_license": "MIT repository license; original-paper rights not inferred",
        "pyrfume_files": {
            relative: {
                "sha256": expected,
                "bytes": (pyrfume_root / relative).stat().st_size,
            }
            for relative, expected in sorted(PYRFUME_FILES.items())
        },
        "opera_uri": OPERA_URI,
        "opera_doi": OPERA_DOI,
        "opera_license": "CC0",
        "opera_sha256": OPERA_SHA256,
        "opera_bytes": OPERA_BYTES,
        "pubchem_supplement": {
            "path": str(pubchem_supplement.resolve()),
            "sha256": PUBCHEM_SUPPLEMENT_SHA256,
            "bytes": pubchem_supplement.stat().st_size,
            "source": "PubChem PUG REST",
        },
    }


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE hub_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE source_files (
            source_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            path TEXT NOT NULL,
            data_class TEXT NOT NULL,
            license TEXT NOT NULL,
            origin_uri TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
            bytes INTEGER NOT NULL CHECK(bytes > 0)
        );
        CREATE TABLE molecules (
            cid INTEGER PRIMARY KEY,
            inchi_key TEXT NOT NULL,
            inchi_key_skeleton TEXT NOT NULL,
            canonical_smiles TEXT NOT NULL,
            molecular_weight REAL,
            preferred_name TEXT,
            casrn TEXT
        );
        CREATE TABLE molecule_sources (
            cid INTEGER NOT NULL REFERENCES molecules(cid),
            dataset TEXT NOT NULL,
            PRIMARY KEY (cid, dataset)
        );
        CREATE TABLE physchem_observations (
            observation_id INTEGER PRIMARY KEY,
            endpoint TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'test')),
            inchi_key TEXT NOT NULL,
            inchi_key_skeleton TEXT NOT NULL,
            casrn TEXT,
            dtxsid TEXT,
            canonical_smiles TEXT,
            preferred_name TEXT,
            source_path TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE molecule_physchem_links (
            cid INTEGER NOT NULL REFERENCES molecules(cid),
            observation_id INTEGER NOT NULL REFERENCES physchem_observations(observation_id),
            PRIMARY KEY (cid, observation_id)
        );
        CREATE TABLE stimuli (
            dataset TEXT NOT NULL,
            stimulus_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            stimulus_type TEXT NOT NULL,
            solvent TEXT,
            concentration_value REAL,
            concentration_unit TEXT,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (dataset, stimulus_id, variant)
        );
        CREATE TABLE stimulus_components (
            dataset TEXT NOT NULL,
            stimulus_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            position INTEGER NOT NULL,
            cid INTEGER NOT NULL REFERENCES molecules(cid),
            liquid_concentration_value REAL,
            liquid_concentration_unit TEXT,
            PRIMARY KEY (dataset, stimulus_id, variant, position),
            FOREIGN KEY (dataset, stimulus_id, variant)
                REFERENCES stimuli(dataset, stimulus_id, variant)
        );
        CREATE TABLE stimulus_dilutions (
            dataset TEXT NOT NULL,
            stimulus_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            dilution_index INTEGER NOT NULL CHECK(dilution_index > 0),
            normalized_value REAL,
            normalized_unit TEXT,
            raw_value TEXT,
            parse_status TEXT NOT NULL,
            PRIMARY KEY (dataset, stimulus_id, variant, dilution_index),
            FOREIGN KEY (dataset, stimulus_id, variant)
                REFERENCES stimuli(dataset, stimulus_id, variant)
        );
        CREATE TABLE sensory_observations (
            observation_id INTEGER PRIMARY KEY,
            dataset TEXT NOT NULL,
            stimulus_a TEXT,
            stimulus_b TEXT,
            subject_id TEXT,
            replicate TEXT,
            endpoint TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            aggregation_level TEXT NOT NULL,
            sample_size INTEGER,
            metadata_json TEXT NOT NULL
        );
        CREATE INDEX idx_physchem_endpoint ON physchem_observations(endpoint);
        CREATE INDEX idx_physchem_inchi ON physchem_observations(inchi_key);
        CREATE INDEX idx_links_cid ON molecule_physchem_links(cid);
        CREATE INDEX idx_stimulus_components_cid ON stimulus_components(cid);
        CREATE INDEX idx_stimulus_dilutions_lookup
            ON stimulus_dilutions(dataset, stimulus_id, dilution_index);
        CREATE INDEX idx_sensory_dataset_endpoint
            ON sensory_observations(dataset, endpoint);
        """
    )


def insert_sources(
    connection: sqlite3.Connection,
    pyrfume_root: Path,
    opera_zip: Path,
    pubchem_supplement: Path,
) -> None:
    connection.executemany(
        "INSERT INTO hub_metadata VALUES (?, ?)",
        (
            ("schema", SCHEMA),
            ("pyrfume_commit", PYRFUME_COMMIT),
            ("opera_sha256", OPERA_SHA256),
            ("pubchem_supplement_sha256", PUBCHEM_SUPPLEMENT_SHA256),
        ),
    )
    for relative, digest in sorted(PYRFUME_FILES.items()):
        dataset = relative.split("/", 1)[0] if "/" in relative else "pyrfume"
        connection.execute(
            "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"pyrfume:{relative}",
                dataset,
                relative,
                "manifest_or_processed_human_data",
                "Pyrfume repository MIT; original-paper rights not inferred",
                f"https://github.com/pyrfume/pyrfume-data/tree/{PYRFUME_COMMIT}/{relative}",
                digest,
                (pyrfume_root / relative).stat().st_size,
            ),
        )
    connection.execute(
        "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "epa_opera:S1.zip",
            "epa_opera_2018",
            "S1.zip",
            "measured_physicochemical_training_and_test_data",
            "CC0",
            OPERA_URI,
            OPERA_SHA256,
            opera_zip.stat().st_size,
        ),
    )
    connection.execute(
        "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "pubchem:headspace_molecule_supplement_v1",
            "pubchem",
            pubchem_supplement.name,
            "chemical_identifier_and_structure_supplement",
            "U.S. government database factual record; source attribution retained",
            "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            PUBCHEM_SUPPLEMENT_SHA256,
            pubchem_supplement.stat().st_size,
        ),
    )


def load_molecules(
    connection: sqlite3.Connection,
    pyrfume_root: Path,
    pubchem_supplement: Path,
) -> tuple[dict[str, list[int]], dict[int, dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    sources: dict[int, set[str]] = defaultdict(set)
    for archive in ARCHIVES:
        frame = pd.read_csv(pyrfume_root / archive / "molecules.csv")
        smiles_column = (
            "CanonicalSMILES"
            if "CanonicalSMILES" in frame.columns
            else "IsomericSMILES"
        )
        for _, row in frame.iterrows():
            cid_value = finite(row.get("CID"))
            raw_smiles = clean(row.get(smiles_column))
            if cid_value is None or raw_smiles is None:
                continue
            cid = int(cid_value)
            molecule = Chem.MolFromSmiles(raw_smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES for {archive} CID {cid}")
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            inchi_key = inchi.MolToInchiKey(molecule)
            inchi_key_skeleton = inchi_key.split("-", 1)[0]
            candidate = {
                "cid": cid,
                "inchi_key": inchi_key,
                "inchi_key_skeleton": inchi_key_skeleton,
                "canonical_smiles": canonical_smiles,
                "molecular_weight": finite(row.get("MolecularWeight")),
                "preferred_name": first_clean(
                    row.get("OdorName"),
                    row.get("Odorant name"),
                    row.get("IUPACName"),
                    row.get("name"),
                ),
                "casrn": first_clean(row.get("CAS"), row.get("C.A.S.")),
            }
            if (
                cid in records
                and records[cid]["inchi_key_skeleton"] != inchi_key_skeleton
            ):
                raise ValueError(f"conflicting structures for PubChem CID {cid}")
            if cid not in records:
                records[cid] = candidate
            else:
                if (
                    records[cid]["inchi_key"].endswith("UHFFFAOYSA-N")
                    and not inchi_key.endswith("UHFFFAOYSA-N")
                ):
                    records[cid]["inchi_key"] = inchi_key
                    records[cid]["canonical_smiles"] = canonical_smiles
                for field in ("molecular_weight", "preferred_name", "casrn"):
                    if records[cid].get(field) is None and candidate.get(field) is not None:
                        records[cid][field] = candidate[field]
            sources[cid].add(archive)
    supplement = json.loads(pubchem_supplement.read_text(encoding="utf-8"))
    if supplement.get("schema") != "pubchem-headspace-molecule-supplement/v1":
        raise ValueError("unsupported PubChem supplement schema")
    for row in supplement.get("records", []):
        cid = int(row["cid"])
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None:
            raise ValueError(f"invalid PubChem supplement SMILES for CID {cid}")
        inchi_key = inchi.MolToInchiKey(molecule)
        if inchi_key != row["inchi_key"]:
            raise ValueError(f"PubChem supplement InChIKey mismatch for CID {cid}")
        candidate = {
            "cid": cid,
            "inchi_key": inchi_key,
            "inchi_key_skeleton": inchi_key.split("-", 1)[0],
            "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
            "molecular_weight": float(row["molecular_weight"]),
            "preferred_name": str(row["iupac_name"]),
            "casrn": str(row["casrn"]),
        }
        if cid in records and records[cid]["inchi_key_skeleton"] != candidate[
            "inchi_key_skeleton"
        ]:
            raise ValueError(f"PubChem supplement conflicts for CID {cid}")
        records[cid] = candidate
        sources[cid].add("pubchem_supplement")
    for cid in sorted(records):
        row = records[cid]
        connection.execute(
            "INSERT INTO molecules VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                row["inchi_key"],
                row["inchi_key_skeleton"],
                row["canonical_smiles"],
                row["molecular_weight"],
                row["preferred_name"],
                row["casrn"],
            ),
        )
        connection.executemany(
            "INSERT INTO molecule_sources VALUES (?, ?)",
            [(cid, archive) for archive in sorted(sources[cid])],
        )
    by_inchi: dict[str, list[int]] = defaultdict(list)
    for cid, row in records.items():
        by_inchi[row["inchi_key_skeleton"]].append(cid)
    return {key: sorted(value) for key, value in by_inchi.items()}, records


def load_physchem(
    connection: sqlite3.Connection,
    opera_zip: Path,
    cids_by_inchi: Mapping[str, list[int]],
) -> None:
    observation_id = 0
    with zipfile.ZipFile(opera_zip) as bundle:
        for endpoint, property_name, unit, source_path, split in OPERA_ENDPOINTS:
            supplier = Chem.ForwardSDMolSupplier(
                bundle.open(source_path), sanitize=False, removeHs=False
            )
            for molecule in supplier:
                if molecule is None or not molecule.HasProp(property_name):
                    continue
                value = finite(molecule.GetProp(property_name))
                key = clean(
                    molecule.GetProp("InChI Key_QSARr")
                    if molecule.HasProp("InChI Key_QSARr")
                    else None
                )
                if value is None or key is None:
                    continue
                key_skeleton = key.split("-", 1)[0]
                observation_id += 1
                properties = molecule.GetPropsAsDict()
                metadata = {
                    str(name): str(raw)
                    for name, raw in properties.items()
                    if "Reference" in str(name) or "Temperature" in str(name)
                }
                connection.execute(
                    "INSERT INTO physchem_observations VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        observation_id,
                        endpoint,
                        value,
                        unit,
                        split,
                        key,
                        key_skeleton,
                        first_clean(
                            properties.get("CAS"), properties.get("source_casrn")
                        ),
                        clean(properties.get("dsstox_substance_id")),
                        first_clean(
                            properties.get("Canonical_QSARr"), properties.get("SMILES")
                        ),
                        first_clean(
                            properties.get("preferred_name"), properties.get("NAME")
                        ),
                        source_path,
                        canonical_json(metadata),
                    ),
                )
                connection.executemany(
                    "INSERT INTO molecule_physchem_links VALUES (?, ?)",
                    [
                        (cid, observation_id)
                        for cid in cids_by_inchi.get(key_skeleton, [])
                    ],
                )


def insert_stimulus(
    connection: sqlite3.Connection,
    *,
    dataset: str,
    stimulus_id: str,
    variant: str,
    stimulus_type: str,
    components: Iterable[int],
    solvent: str | None = None,
    concentration_value: float | None = None,
    concentration_unit: str | None = None,
    component_concentrations: Mapping[int, tuple[float | None, str | None]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    component_list = [int(cid) for cid in components]
    if (
        (not component_list and stimulus_type != "blank")
        or len(component_list) != len(set(component_list))
    ):
        raise ValueError(f"invalid stimulus components: {dataset}/{stimulus_id}/{variant}")
    connection.execute(
        "INSERT INTO stimuli VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dataset,
            stimulus_id,
            variant,
            stimulus_type,
            solvent,
            concentration_value,
            concentration_unit,
            canonical_json(dict(metadata or {})),
        ),
    )
    rows = []
    for position, cid in enumerate(component_list):
        concentration = (component_concentrations or {}).get(cid, (None, None))
        rows.append(
            (
                dataset,
                stimulus_id,
                variant,
                position,
                cid,
                concentration[0],
                concentration[1],
            )
        )
    connection.executemany(
        "INSERT INTO stimulus_components VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )


def load_stimuli(
    connection: sqlite3.Connection,
    pyrfume_root: Path,
    molecules: Mapping[int, Mapping[str, Any]],
) -> dict[str, str]:
    stimulus_aliases: dict[str, str] = {}
    keller = pd.read_csv(pyrfume_root / "keller_2016" / "stimuli.csv")
    keller_molecules = pd.read_csv(
        pyrfume_root / "keller_2016" / "molecules.csv"
    )
    keller_cas_to_cid = {
        str(row["CAS"]).strip(): int(row["CID"])
        for _, row in keller_molecules.iterrows()
        if pd.notna(row.get("CAS")) and pd.notna(row.get("CID"))
    }
    for _, row in keller.sort_values("Stimulus").iterrows():
        raw_identifier = str(row["CIDs"]).strip()
        cid = (
            int(float(raw_identifier))
            if raw_identifier.replace(".", "", 1).isdigit()
            else keller_cas_to_cid[raw_identifier]
        )
        if cid not in molecules:
            raise ValueError(f"Keller stimulus has unresolved molecule: {raw_identifier}")
        insert_stimulus(
            connection,
            dataset="keller_2016",
            stimulus_id=str(int(row["Stimulus"])),
            variant="base",
            stimulus_type="single_molecule",
            components=[cid],
            solvent=clean(row.get("Solvent")),
            concentration_value=float(row["Concentration"]),
            concentration_unit="fraction_v_v",
            component_concentrations={
                cid: (float(row["Concentration"]), "fraction_v_v")
            },
            metadata={"ratio": clean(row.get("Ratio"))},
        )
    ravia = pd.read_csv(pyrfume_root / "ravia_2020" / "stimuli.csv")
    for _, row in ravia.sort_values("Stimulus").iterrows():
        raw_components = str(row["CID"]).strip()
        components = (
            []
            if raw_components.casefold() == "blank"
            else [int(part) for part in raw_components.split(";")]
        )
        stimulus_type = (
            "blank"
            if not components
            else str(row["Type"]).replace("mixuture", "mixture")
        )
        insert_stimulus(
            connection,
            dataset="ravia_2020",
            stimulus_id=str(int(row["Stimulus"])),
            variant="base",
            stimulus_type=stimulus_type,
            components=components,
            concentration_value=None,
            concentration_unit=None,
            metadata={
                "experiment": clean(row.get("Experiment")),
                "dilutions": [
                    clean(row.get("Dilution1")),
                    clean(row.get("Dilution2")),
                    clean(row.get("Dilution3")),
                ],
                "mixture_id": clean(row.get("ID")),
                "compared_to_id": clean(row.get("Compared To ID")),
            },
        )
        dilution_rows = []
        for dilution_index in (1, 2, 3):
            raw = clean(row.get(f"Dilution{dilution_index}"))
            value, unit, status = parse_dilution(raw)
            dilution_rows.append(
                (
                    "ravia_2020",
                    str(int(row["Stimulus"])),
                    "base",
                    dilution_index,
                    value,
                    unit,
                    raw,
                    status,
                )
            )
        connection.executemany(
            "INSERT INTO stimulus_dilutions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            dilution_rows,
        )
    ma = pd.read_csv(pyrfume_root / "ma_2021" / "stimuli.csv")
    for _, row in ma.sort_values("Stimulus").iterrows():
        cid = int(row["CID"])
        insert_stimulus(
            connection,
            dataset="ma_2021",
            stimulus_id=str(int(row["Stimulus"])),
            variant="base",
            stimulus_type="single_molecule_stock",
            components=[cid],
            solvent=clean(row.get("Solvent")),
            concentration_value=float(row["Concentration (mg/mL)"]),
            concentration_unit="mg_per_ml",
            component_concentrations={
                cid: (float(row["Concentration (mg/mL)"]), "mg_per_ml")
            },
            metadata={"purity": clean(row.get("Purity"))},
        )
    bushdid_molecules = pd.read_csv(
        pyrfume_root / "bushdid_2014" / "molecules.csv"
    ).set_index("CID")
    bushdid = pd.read_csv(pyrfume_root / "bushdid_2014" / "stimuli.csv")
    molecule_columns = [name for name in bushdid.columns if name.startswith("Molecule ")]
    for index, row in bushdid.iterrows():
        components = [
            int(row[name])
            for name in molecule_columns
            if pd.notna(row[name]) and int(row[name]) > 0
        ]
        dilution = float(row["Stimulus dilution"])
        component_concentrations = {}
        solvents = set()
        for cid in components:
            stock = bushdid_molecules.loc[cid]
            percent = finite(stock.get("% odorant"))
            if percent is not None:
                component_concentrations[cid] = (
                    percent / 100.0 * dilution,
                    "estimated_stock_fraction_times_stimulus_dilution",
                )
            solvent = clean(stock.get("Solvent"))
            if solvent:
                solvents.add(solvent)
        variant = f"{index}:{row['Answer']}:{dilution:g}"
        insert_stimulus(
            connection,
            dataset="bushdid_2014",
            stimulus_id=str(int(row["Stimulus"])),
            variant=variant,
            stimulus_type="mixture_triangle_variant",
            components=components,
            solvent=";".join(sorted(solvents)) or None,
            concentration_value=dilution,
            concentration_unit="relative_stimulus_dilution",
            component_concentrations=component_concentrations,
            metadata={
                "answer": str(row["Answer"]),
                "components_in_mixture": int(row["Components in mixtures"]),
                "components_that_differ": int(row["Components that differ"]),
                "mixture_overlap_percent": float(row["% mixture overlap"]),
            },
        )
    snitz = pd.read_csv(pyrfume_root / "snitz_2013" / "behavior.csv")
    unique_mixtures = set()
    for column in ("StimulusA", "StimulusB"):
        unique_mixtures.update(str(value) for value in snitz[column])
    for raw in sorted(unique_mixtures):
        components = [int(part) for part in raw.split(",")]
        signature = hashlib.sha256(raw.encode("ascii")).hexdigest()[:20]
        stimulus_id = f"mixture:{signature}"
        stimulus_aliases[raw] = stimulus_id
        insert_stimulus(
            connection,
            dataset="snitz_2013",
            stimulus_id=stimulus_id,
            variant="base",
            stimulus_type=(
                "single_molecule" if len(components) == 1 else "mixture"
            ),
            components=components,
            metadata={"component_signature": raw},
        )
    abraham = pd.read_csv(pyrfume_root / "abraham_2012" / "stimuli.csv")
    for _, row in abraham.sort_values("Stimulus").iterrows():
        cid = int(row["CID"])
        insert_stimulus(
            connection,
            dataset="abraham_2012",
            stimulus_id=str(int(row["Stimulus"])),
            variant="base",
            stimulus_type="single_molecule_threshold",
            components=[cid],
        )
    return stimulus_aliases


def insert_sensory(
    connection: sqlite3.Connection,
    *,
    dataset: str,
    endpoint: str,
    value: Any,
    unit: str,
    stimulus_a: str | None = None,
    stimulus_b: str | None = None,
    subject_id: Any = None,
    replicate: Any = None,
    aggregation_level: str = "raw_row",
    sample_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    numeric = finite(value)
    if numeric is None:
        return
    connection.execute(
        "INSERT INTO sensory_observations "
        "(dataset, stimulus_a, stimulus_b, subject_id, replicate, endpoint, "
        "value, unit, aggregation_level, sample_size, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dataset,
            stimulus_a,
            stimulus_b,
            clean(subject_id),
            clean(replicate),
            endpoint,
            numeric,
            unit,
            aggregation_level,
            sample_size,
            canonical_json(dict(metadata or {})),
        ),
    )


def load_sensory(
    connection: sqlite3.Connection,
    pyrfume_root: Path,
    snitz_aliases: Mapping[str, str],
) -> None:
    keller = pd.read_csv(
        pyrfume_root / "keller_2016" / "behavior.csv",
        usecols=["Stimulus", "Subject", "MeasurementValue", "Value"],
        low_memory=False,
    )
    keller = keller[
        keller["MeasurementValue"].isin(
            {
                "HOW STRONG IS THE SMELL?",
                "HOW PLEASANT IS THE SMELL?",
                "HOW FAMILIAR IS THE SMELL?",
            }
        )
    ].copy()
    keller["numeric"] = pd.to_numeric(keller["Value"], errors="coerce")
    keller = keller.dropna(subset=["numeric"])
    endpoint_map = {
        "HOW STRONG IS THE SMELL?": "intensity",
        "HOW PLEASANT IS THE SMELL?": "pleasantness",
        "HOW FAMILIAR IS THE SMELL?": "familiarity",
    }
    grouped = keller.groupby(["Stimulus", "MeasurementValue"])["numeric"].agg(
        ["mean", "std", "median", "count"]
    )
    for (stimulus, measurement), row in grouped.sort_index().iterrows():
        insert_sensory(
            connection,
            dataset="keller_2016",
            stimulus_a=str(int(stimulus)),
            endpoint=endpoint_map[str(measurement)],
            value=row["mean"],
            unit="rating_0_100",
            aggregation_level="stimulus_mean",
            sample_size=int(row["count"]),
            metadata={
                "std": finite(row["std"]),
                "median": finite(row["median"]),
            },
        )
    ravia_intensity = pd.read_csv(
        pyrfume_root / "ravia_2020" / "behavior_1.csv"
    )
    for _, row in ravia_intensity.iterrows():
        insert_sensory(
            connection,
            dataset="ravia_2020",
            stimulus_a=str(int(row["Stimulus"])),
            replicate=row["Dilution #"],
            endpoint="intensity",
            value=row["Intensity"],
            unit="rating_0_100",
            aggregation_level="published_aggregate",
            metadata={"dilution_number": int(row["Dilution #"])},
        )
    ravia_similarity = pd.read_csv(
        pyrfume_root / "ravia_2020" / "behavior_2.csv"
    )
    for _, row in ravia_similarity.iterrows():
        insert_sensory(
            connection,
            dataset="ravia_2020",
            stimulus_a=str(int(row["Stimulus 1"])),
            stimulus_b=str(int(row["Stimulus 2"])),
            endpoint="similarity",
            value=row["RatedSimilarity"],
            unit="rating_0_100",
            aggregation_level="published_aggregate",
        )
    ravia_discrimination = pd.read_csv(
        pyrfume_root / "ravia_2020" / "behavior_3.csv"
    )
    for _, row in ravia_discrimination.iterrows():
        shared = {
            "dataset": "ravia_2020",
            "stimulus_a": str(int(row["Stimulus 1"])),
            "stimulus_b": str(int(row["Stimulus 2"])),
            "subject_id": row["Subject"],
            "replicate": row["TrialOrderInSession"],
            "aggregation_level": "participant_trial",
            "metadata": {
                "choice": finite(row.get("Choice")),
                "correct_choice_on_screen": finite(
                    row.get("CorrectChoiceOnScreen")
                ),
            },
        }
        insert_sensory(
            connection,
            endpoint="discrimination_correct",
            value=row["IsCorrect"],
            unit="binary",
            **shared,
        )
        insert_sensory(
            connection,
            endpoint="discrimination_confidence",
            value=row["Confidence"],
            unit="fraction",
            **shared,
        )
    ma = pd.read_csv(pyrfume_root / "ma_2021" / "behavior.csv")
    ma_endpoints = {
        "IA": "intensity_component_a_alone",
        "IAmix": "intensity_component_a_in_mixture",
        "IB": "intensity_component_b_alone",
        "IBmix": "intensity_component_b_in_mixture",
        "IAB": "intensity_binary_mixture",
        "PA": "pleasantness_component_a",
        "PB": "pleasantness_component_b",
        "PAB": "pleasantness_binary_mixture",
    }
    for _, row in ma.iterrows():
        for column, endpoint in ma_endpoints.items():
            insert_sensory(
                connection,
                dataset="ma_2021",
                stimulus_a=str(int(row["Stimulus A"])),
                stimulus_b=str(int(row["Stimulus B"])),
                subject_id=row["Subject"],
                replicate=row["Rep"],
                endpoint=endpoint,
                value=row[column],
                unit="rating_0_10",
                aggregation_level="participant_trial",
            )
    bushdid = pd.read_csv(
        pyrfume_root / "bushdid_2014" / "behavior.csv"
    )
    for _, row in bushdid.iterrows():
        insert_sensory(
            connection,
            dataset="bushdid_2014",
            stimulus_a=str(int(row["Stimulus"])),
            subject_id=row["Subject"],
            endpoint="triangle_test_correct",
            value=binary(row["Correct"]),
            unit="binary",
            aggregation_level="participant_trial",
        )
    snitz = pd.read_csv(pyrfume_root / "snitz_2013" / "behavior.csv")
    for _, row in snitz.iterrows():
        insert_sensory(
            connection,
            dataset="snitz_2013",
            stimulus_a=snitz_aliases[str(row["StimulusA"])],
            stimulus_b=snitz_aliases[str(row["StimulusB"])],
            endpoint="similarity",
            value=row["Similarity"],
            unit="rating_0_100",
            aggregation_level="published_aggregate",
            metadata={"pair": int(row["Pair"])},
        )
    abraham = pd.read_csv(
        pyrfume_root / "abraham_2012" / "behavior.csv"
    )
    for _, row in abraham.iterrows():
        insert_sensory(
            connection,
            dataset="abraham_2012",
            stimulus_a=str(int(row["Stimulus"])),
            endpoint="log10_inverse_odor_detection_threshold",
            value=row["Log (1/ODT)"],
            unit="log10_inverse_ppmv",
            aggregation_level="published_value",
            metadata={
                "training_flag": finite(row.get("Train")),
                "derived_odor_detection_threshold_ppmv": 10.0
                ** (-float(row["Log (1/ODT)"])),
                "scale_note": "ODT values are on the paper's Nagata-aligned ppmv scale",
            },
        )


def scalar(connection: sqlite3.Connection, query: str, parameters=()) -> int:
    return int(connection.execute(query, parameters).fetchone()[0])


def build_report(
    connection: sqlite3.Connection,
    output: Path,
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint_rows = {
        endpoint: {
            "observations": count,
            "linked_molecules": scalar(
                connection,
                "SELECT COUNT(DISTINCT l.cid) FROM molecule_physchem_links l "
                "JOIN physchem_observations p ON p.observation_id=l.observation_id "
                "WHERE p.endpoint=?",
                (endpoint,),
            ),
        }
        for endpoint, count in connection.execute(
            "SELECT endpoint, COUNT(*) FROM physchem_observations "
            "GROUP BY endpoint ORDER BY endpoint"
        )
    }
    sensory_rows = {
        dataset: {
            "observations": count,
            "endpoints": scalar(
                connection,
                "SELECT COUNT(DISTINCT endpoint) FROM sensory_observations "
                "WHERE dataset=?",
                (dataset,),
            ),
        }
        for dataset, count in connection.execute(
            "SELECT dataset, COUNT(*) FROM sensory_observations "
            "GROUP BY dataset ORDER BY dataset"
        )
    }
    stimuli_rows = {
        dataset: count
        for dataset, count in connection.execute(
            "SELECT dataset, COUNT(*) FROM stimuli GROUP BY dataset ORDER BY dataset"
        )
    }
    report = {
        "schema": SCHEMA,
        "source": dict(source_audit),
        "software": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "rdkit_version": rdBase.rdkitVersion,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "sqlite_version": sqlite3.sqlite_version,
        },
        "counts": {
            "source_files": scalar(connection, "SELECT COUNT(*) FROM source_files"),
            "molecules": scalar(connection, "SELECT COUNT(*) FROM molecules"),
            "molecule_source_links": scalar(
                connection, "SELECT COUNT(*) FROM molecule_sources"
            ),
            "physchem_observations": scalar(
                connection, "SELECT COUNT(*) FROM physchem_observations"
            ),
            "molecules_with_physchem": scalar(
                connection,
                "SELECT COUNT(DISTINCT cid) FROM molecule_physchem_links",
            ),
            "stimuli": scalar(connection, "SELECT COUNT(*) FROM stimuli"),
            "stimulus_components": scalar(
                connection, "SELECT COUNT(*) FROM stimulus_components"
            ),
            "stimulus_dilutions": scalar(
                connection, "SELECT COUNT(*) FROM stimulus_dilutions"
            ),
            "sensory_observations": scalar(
                connection, "SELECT COUNT(*) FROM sensory_observations"
            ),
        },
        "physchem_endpoints": endpoint_rows,
        "stimuli_by_dataset": stimuli_rows,
        "sensory_by_dataset": sensory_rows,
        "database": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "allow_pickle": False,
            "packaged_in_wheel": False,
        },
        "claim_boundary": {
            "research_data_hub": True,
            "measured_physchem_is_not_mixture_headspace_measurement": True,
            "pyrfume_processed_data_is_not_new_prospective_evidence": True,
            "human_olfactory_90_percent_certified": False,
            "commercial_runtime_weight": 0.0,
        },
    }
    return report


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyrfume-root", type=Path, required=True)
    parser.add_argument("--opera-zip", type=Path, required=True)
    parser.add_argument(
        "--pubchem-supplement",
        type=Path,
        default=ROOT
        / "benchmarks"
        / "source_data"
        / "pubchem_headspace_molecules_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.db",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "benchmarks" / "headspace_sensory_hub_v1.json",
    )
    args = parser.parse_args()
    pyrfume_root = args.pyrfume_root.expanduser().resolve(strict=True)
    opera_zip = args.opera_zip.expanduser().resolve(strict=True)
    pubchem_supplement = args.pubchem_supplement.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    source_audit = verify_sources(pyrfume_root, opera_zip, pubchem_supplement)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA page_size = 4096")
            create_schema(connection)
            insert_sources(
                connection, pyrfume_root, opera_zip, pubchem_supplement
            )
            cids_by_inchi, molecules = load_molecules(
                connection, pyrfume_root, pubchem_supplement
            )
            load_physchem(connection, opera_zip, cids_by_inchi)
            aliases = load_stimuli(connection, pyrfume_root, molecules)
            load_sensory(connection, pyrfume_root, aliases)
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("headspace sensory hub integrity check failed")
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError(
                    f"headspace sensory hub foreign-key failures: {foreign_key_errors[:3]}"
                )
            connection.execute("VACUUM")
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(f"file:{output.as_posix()}?mode=ro", uri=True)
    try:
        report = build_report(connection, output, source_audit)
    finally:
        connection.close()
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
