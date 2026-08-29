"""Read-only access to the catalog-filtered EPA CompTox extract.

The extract preserves official identifiers, product-use observations and
non-human toxicology references for catalog materials.  These records are
screening and provenance evidence only: they never constitute an IFRA limit,
supplier qualification, commercial release approval or human-olfaction truth.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .sqlite_lifecycle import SQLiteConnectionOwner


SCHEMA_VERSION = "1.0"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_files (
    source_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    file_name TEXT NOT NULL,
    origin_uri TEXT NOT NULL,
    download_uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    retrieved_on TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    materialization_status TEXT NOT NULL,
    license_note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chemicals (
    ingredient_id TEXT NOT NULL,
    catalog_name TEXT NOT NULL,
    catalog_casrn TEXT NOT NULL,
    dtxsid TEXT NOT NULL,
    dtxcid TEXT,
    preferred_name TEXT,
    inchikey TEXT,
    iupac_name TEXT,
    smiles TEXT,
    molecular_formula TEXT,
    average_mass REAL,
    monoisotopic_mass REAL,
    qsar_ready_smiles TEXT,
    ms_ready_smiles TEXT,
    PRIMARY KEY (ingredient_id, dtxsid)
);
CREATE TABLE IF NOT EXISTS cpdat_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT NOT NULL,
    ingredient_id TEXT NOT NULL,
    dtxsid TEXT,
    curated_casrn TEXT,
    curated_chemical_name TEXT,
    data_source TEXT,
    data_source_url TEXT,
    data_document_id TEXT,
    data_document_title TEXT,
    data_document_url TEXT,
    data_document_date TEXT,
    organization TEXT,
    raw_functional_use TEXT,
    function_category TEXT,
    product_id TEXT,
    product_title TEXT,
    puc_kind TEXT,
    puc_general_category TEXT,
    puc_product_family TEXT,
    puc_product_type TEXT,
    lower_weight_fraction REAL,
    upper_weight_fraction REAL,
    central_weight_fraction REAL,
    component_name TEXT,
    keyword_set TEXT
);
CREATE TABLE IF NOT EXISTS toxref_studies (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingredient_id TEXT NOT NULL,
    chemical_id TEXT,
    dtxsid TEXT NOT NULL,
    casrn TEXT,
    preferred_name TEXT,
    study_id TEXT,
    study_source_id TEXT,
    study_citation TEXT,
    study_year TEXT,
    study_source TEXT,
    study_type TEXT,
    study_type_guideline TEXT,
    species TEXT,
    strain_group TEXT,
    strain TEXT,
    admin_route TEXT,
    admin_method TEXT,
    dose_start REAL,
    dose_start_unit TEXT,
    dose_end REAL,
    dose_end_unit TEXT,
    study_comment TEXT,
    guideline_id TEXT,
    processed TEXT
);
CREATE TABLE IF NOT EXISTS toxref_pods (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingredient_id TEXT NOT NULL,
    study_id TEXT,
    study_type TEXT,
    preferred_name TEXT,
    dtxsid TEXT NOT NULL,
    toxval_study_source_id TEXT,
    toxval_effect_list TEXT,
    dose_level TEXT,
    calc_pod_type TEXT,
    qualifier TEXT,
    mg_kg_day_value REAL,
    admin_route TEXT,
    admin_method TEXT,
    vehicle TEXT,
    species TEXT,
    strain_group TEXT,
    strain TEXT,
    dose_start REAL,
    dose_start_unit TEXT,
    dose_end REAL,
    dose_end_unit TEXT,
    study_year TEXT,
    study_citation TEXT
);
CREATE TABLE IF NOT EXISTS toxval_observations (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingredient_id TEXT NOT NULL,
    source_table TEXT NOT NULL,
    source_record_id TEXT,
    dtxsid TEXT,
    casrn TEXT,
    preferred_name TEXT,
    effect_category TEXT,
    effect_text TEXT,
    study_type TEXT,
    species TEXT,
    value_numeric REAL,
    value_unit TEXT,
    raw_record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_epa_chemicals_cas ON chemicals(catalog_casrn);
CREATE INDEX IF NOT EXISTS ix_epa_chemicals_dtxsid ON chemicals(dtxsid);
CREATE INDEX IF NOT EXISTS ix_epa_cpdat_ingredient ON cpdat_observations(ingredient_id);
CREATE INDEX IF NOT EXISTS ix_epa_cpdat_type ON cpdat_observations(record_type);
CREATE INDEX IF NOT EXISTS ix_epa_toxref_study_ingredient ON toxref_studies(ingredient_id);
CREATE INDEX IF NOT EXISTS ix_epa_toxref_pod_ingredient ON toxref_pods(ingredient_id);
CREATE INDEX IF NOT EXISTS ix_epa_toxval_ingredient ON toxval_observations(ingredient_id);
"""


class EPACompToxStore(SQLiteConnectionOwner):
    """Query a compact EPA extract without loading bulk source files."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent.parent / "data" / "epa_comptox_extract.db"
        )
        if self.path.exists():
            self.connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro", uri=True
            )
        else:
            self.connection = sqlite3.connect(":memory:")
            self.initialize(self.connection)

    @staticmethod
    def initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )

    def material_summary(self, ingredient_id: str) -> dict[str, int | str | None]:
        chemical = self.connection.execute(
            "SELECT catalog_casrn, dtxsid FROM chemicals WHERE ingredient_id = ? "
            "ORDER BY dtxsid LIMIT 1",
            (ingredient_id,),
        ).fetchone()
        return {
            "ingredient_id": ingredient_id,
            "casrn": chemical[0] if chemical else None,
            "dtxsid": chemical[1] if chemical else None,
            "cpdat_rows": int(self.connection.execute(
                "SELECT COUNT(*) FROM cpdat_observations WHERE ingredient_id = ?",
                (ingredient_id,),
            ).fetchone()[0]),
            "toxref_studies": int(self.connection.execute(
                "SELECT COUNT(*) FROM toxref_studies WHERE ingredient_id = ?",
                (ingredient_id,),
            ).fetchone()[0]),
            "toxref_pods": int(self.connection.execute(
                "SELECT COUNT(*) FROM toxref_pods WHERE ingredient_id = ?",
                (ingredient_id,),
            ).fetchone()[0]),
            "toxval_rows": int(self.connection.execute(
                "SELECT COUNT(*) FROM toxval_observations WHERE ingredient_id = ?",
                (ingredient_id,),
            ).fetchone()[0]),
        }

    def stats(self) -> dict[str, int | str]:
        metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))

        def count(table: str) -> int:
            return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "epa_comptox_schema_version": metadata.get("schema_version", "unknown"),
            "epa_source_files": count("source_files"),
            "epa_catalog_ingredients_matched": int(self.connection.execute(
                "SELECT COUNT(DISTINCT ingredient_id) FROM chemicals"
            ).fetchone()[0]),
            "epa_dsstox_mappings": count("chemicals"),
            "epa_cpdat_rows": count("cpdat_observations"),
            "epa_toxref_studies": count("toxref_studies"),
            "epa_toxref_pods": count("toxref_pods"),
            "epa_toxval_rows": count("toxval_observations"),
        }

    def close(self) -> None:
        self.connection.close()
