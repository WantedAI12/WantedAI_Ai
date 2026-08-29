"""Read-only access to the non-human fragrance data lineage hub.

The hub keeps conflicting observations rather than silently overwriting them.
Only reference-formula rows with an allowed provenance tier may influence the
historical co-occurrence prior; synthetic training assets and unverified
regulatory rows remain searchable lineage, never release evidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .sqlite_lifecycle import SQLiteConnectionOwner


SCHEMA_VERSION = "1.0"
REFERENCE_TIERS = {"curated_engineering", "unverified_reference", "published_reference"}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hub_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS data_sources (
    source_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    trust_tier TEXT NOT NULL,
    allowed_use TEXT NOT NULL,
    prohibited_use TEXT NOT NULL,
    origin_uri TEXT NOT NULL,
    local_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    record_count INTEGER,
    refreshed_on TEXT NOT NULL,
    license_note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS material_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    canonical_ingredient_id TEXT,
    material_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    cas_number TEXT,
    property_name TEXT NOT NULL,
    value_text TEXT,
    value_numeric REAL,
    unit TEXT,
    evidence_class TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
);
CREATE TABLE IF NOT EXISTS formula_references (
    source_id TEXT NOT NULL,
    formula_ref TEXT NOT NULL,
    formula_name TEXT NOT NULL,
    provenance_tier TEXT NOT NULL,
    reference_only INTEGER NOT NULL CHECK(reference_only = 1),
    PRIMARY KEY (source_id, formula_ref),
    FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
);
CREATE TABLE IF NOT EXISTS formula_notes (
    source_id TEXT NOT NULL,
    formula_ref TEXT NOT NULL,
    material_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    pyramid TEXT,
    amount REAL,
    FOREIGN KEY (source_id, formula_ref) REFERENCES formula_references(source_id, formula_ref)
);
CREATE TABLE IF NOT EXISTS regulatory_observations (
    source_id TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    cas_number TEXT,
    market_or_category TEXT NOT NULL,
    rule_status TEXT NOT NULL,
    limit_value REAL,
    limit_unit TEXT,
    effective_on TEXT,
    evidence_class TEXT NOT NULL,
    reference_only INTEGER NOT NULL CHECK(reference_only = 1),
    PRIMARY KEY (source_id, source_record_id),
    FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
);
CREATE INDEX IF NOT EXISTS ix_material_canonical ON material_observations(canonical_ingredient_id);
CREATE INDEX IF NOT EXISTS ix_material_cas ON material_observations(cas_number);
CREATE INDEX IF NOT EXISTS ix_material_name ON material_observations(normalized_name);
CREATE INDEX IF NOT EXISTS ix_formula_notes_ref ON formula_notes(source_id, formula_ref);
"""


class NonHumanDataHub(SQLiteConnectionOwner):
    """Query the packaged, provenance-aware non-human data hub."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent.parent / "data" / "nonhuman_data_hub.db"
        )
        if self.path.exists():
            self.connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro", uri=True
            )
        else:
            self.connection = sqlite3.connect(":memory:")
            self.connection.executescript(SCHEMA_SQL)

    @staticmethod
    def initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT OR REPLACE INTO hub_metadata VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )

    def additional_reference_formulas(self) -> list[tuple[str, str, str, str]]:
        """Return non-synthetic note rows allowed as a weak co-occurrence prior."""
        placeholders = ",".join("?" for _ in REFERENCE_TIERS)
        return [
            (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
            for row in self.connection.execute(
                f"""SELECT n.source_id, n.formula_ref, n.material_name, r.formula_name
                FROM formula_notes n
                JOIN formula_references r
                  ON r.source_id = n.source_id AND r.formula_ref = n.formula_ref
                WHERE r.reference_only = 1 AND r.provenance_tier IN ({placeholders})
                ORDER BY n.source_id, n.formula_ref""",
                tuple(sorted(REFERENCE_TIERS)),
            )
        ]

    def material_evidence(self, ingredient_id: str) -> list[dict]:
        columns = (
            "source_id", "property_name", "value_text", "value_numeric", "unit",
            "evidence_class", "cas_number",
        )
        rows = self.connection.execute(
            f"SELECT {','.join(columns)} FROM material_observations "
            "WHERE canonical_ingredient_id = ? ORDER BY source_id, property_name",
            (ingredient_id,),
        ).fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def stats(self) -> dict[str, int | str]:
        def count(table: str) -> int:
            return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        metadata = dict(self.connection.execute("SELECT key, value FROM hub_metadata"))
        epa_source_rows = self.connection.execute(
            "SELECT COALESCE(SUM(record_count), 0) FROM data_sources WHERE source_id LIKE 'epa_%'"
        ).fetchone()[0]
        return {
            "nonhuman_hub_schema_version": metadata.get("schema_version", "unknown"),
            "nonhuman_data_sources": count("data_sources"),
            "nonhuman_material_observations": count("material_observations"),
            "nonhuman_reference_formulas": count("formula_references"),
            "nonhuman_reference_note_rows": count("formula_notes"),
            "nonhuman_regulatory_reference_rows": count("regulatory_observations"),
            "synthetic_training_assets_quarantined": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM data_sources WHERE trust_tier = 'synthetic_training'"
                ).fetchone()[0]
            ),
            "human_validation_assets_excluded": int(
                metadata.get("human_validation_assets_excluded", "0")
            ),
            "epa_open_data_sources": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM data_sources WHERE source_id LIKE 'epa_%'"
                ).fetchone()[0]
            ),
            "epa_source_rows_connected": int(epa_source_rows or 0),
            "epa_material_observations": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM material_observations WHERE source_id LIKE 'epa_%'"
                ).fetchone()[0]
            ),
            "epa_catalog_ingredients_matched": int(
                self.connection.execute(
                    """SELECT COUNT(DISTINCT canonical_ingredient_id)
                    FROM material_observations
                    WHERE source_id LIKE 'epa_%' AND canonical_ingredient_id IS NOT NULL"""
                ).fetchone()[0]
            ),
        }

    def close(self) -> None:
        self.connection.close()
