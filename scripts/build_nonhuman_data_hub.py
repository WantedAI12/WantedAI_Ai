"""Build the provenance-aware non-human fragrance data hub.

Human sensory, expert evaluation, feedback and validation assets are excluded.
Synthetic datasets are fingerprinted and quarantined as training-only lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.data_hub import NonHumanDataHub  # noqa: E402


HUMAN_TERMS = ("human", "sensory", "expert", "feedback", "validation", "panel")
DATA_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".csv", ".json"}


def normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HubBuilder:
    def __init__(self, workspace: Path, output: Path):
        self.workspace = workspace.resolve()
        self.output = output.resolve()
        self.known_paths: set[Path] = set()
        self.excluded_human_paths: set[Path] = set()
        self.catalog_path = ROOT / "fragrance_ai" / "data" / "safe_ingredient_catalog.json"
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.catalog_items = payload["ingredients"]
        self.by_cas: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        for item in self.catalog_items:
            ingredient_id = str(item["ingredient_id"])
            if item.get("cas_number"):
                self.by_cas[str(item["cas_number"])] = ingredient_id
            for name in (item.get("name"), *(item.get("aliases") or [])):
                if name:
                    self.by_name[normalized(str(name))] = ingredient_id

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        self.connection = sqlite3.connect(output)
        self.connection.execute("PRAGMA foreign_keys = ON")
        NonHumanDataHub.initialize(self.connection)

    def canonical(self, name: str, cas_number: str | None = None) -> str | None:
        if cas_number and cas_number in self.by_cas:
            return self.by_cas[cas_number]
        return self.by_name.get(normalized(name))

    def source(
        self,
        source_id: str,
        path: Path,
        source_kind: str,
        trust_tier: str,
        allowed_use: str,
        prohibited_use: str,
        record_count: int | None = None,
        origin_uri: str = "",
        license_note: str = "Local workspace asset; upstream rights not established",
    ) -> None:
        path = path.resolve()
        if not path.exists():
            return
        self.known_paths.add(path)
        relative = (
            path.relative_to(self.workspace).as_posix()
            if path.is_relative_to(self.workspace)
            else path.name
        )
        refreshed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()
        self.connection.execute(
            """INSERT OR REPLACE INTO data_sources VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                path.name,
                source_kind,
                trust_tier,
                allowed_use,
                prohibited_use,
                origin_uri or f"workspace://{relative}",
                relative,
                sha256(path),
                path.stat().st_size,
                record_count,
                refreshed,
                license_note,
            ),
        )

    def virtual_source(
        self,
        source_id: str,
        display_name: str,
        source_kind: str,
        trust_tier: str,
        allowed_use: str,
        prohibited_use: str,
        origin_uri: str,
        record_count: int | None,
        license_note: str,
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO data_sources VALUES
            (?, ?, ?, ?, ?, ?, ?, '', '', 0, ?, ?, ?)""",
            (
                source_id, display_name, source_kind, trust_tier, allowed_use,
                prohibited_use, origin_uri, record_count,
                datetime.now(timezone.utc).date().isoformat(), license_note,
            ),
        )

    def observation(
        self,
        source_id: str,
        record_id: str,
        name: str,
        cas_number: str | None,
        property_name: str,
        value: object,
        unit: str | None,
        evidence_class: str,
    ) -> None:
        if value is None or value == "":
            return
        numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        text_value = None if numeric is not None else (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else str(value)
        )
        self.connection.execute(
            """INSERT INTO material_observations
            (source_id, source_record_id, canonical_ingredient_id, material_name,
             normalized_name, cas_number, property_name, value_text, value_numeric,
             unit, evidence_class)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                record_id,
                self.canonical(name, cas_number),
                name,
                normalized(name),
                cas_number,
                property_name,
                text_value,
                numeric,
                unit,
                evidence_class,
            ),
        )

    def ingest_catalog(self) -> None:
        self.source(
            "core_safe_catalog",
            self.catalog_path,
            "formulation_catalog",
            "curated_engineering",
            "Candidate generation and conservative prototype screening",
            "Supplier, regulatory or measured-property claims",
            len(self.catalog_items),
        )
        fields = {
            "pyramid": None,
            "profile": None,
            "price_per_kg": "USD_estimate/kg",
            "availability": "0-1 engineering score",
            "rarity": None,
            "risk_tier": None,
            "odor_impact": None,
            "max_concentrate_percent": "%",
            "formulation_ready": None,
            "blocked": None,
        }
        for item in self.catalog_items:
            for field, unit in fields.items():
                self.observation(
                    "core_safe_catalog", str(item["ingredient_id"]), str(item["name"]),
                    item.get("cas_number"), field, item.get(field), unit, "curated_engineering",
                )

    def ingest_core_auxiliary(self) -> None:
        data = ROOT / "fragrance_ai" / "data"
        self.source(
            "operator_supplier_registry", data / "supplier_registry.json",
            "supplier_registry", "operator_required",
            "Verified operator records when populated",
            "Qualification from the bundled empty registry", 0,
        )
        self.source(
            "supplier_import_template", data / "supplier_materials_template.csv",
            "schema_template", "schema_only", "Supplier import schema",
            "Any material, price, stock or document claim", 0,
        )
        self.source(
            "scientific_import_template", data / "scientific_properties_template.csv",
            "schema_template", "schema_only", "Scientific-property import schema",
            "Any measured-property claim", 0,
        )

    def ingest_ifra_reference(self, csv_path: Path | None = None) -> None:
        if csv_path is None:
            self.virtual_source(
                "ifra_transparency_2025_index",
                "IFRA 2025 Transparency List",
                "official_reference_index",
                "official_reference_index",
                "Global fragrance-palette scope and identifier lookup",
                "Formulation readiness, legal compliance, price, inventory or safety approval",
                "https://ifrafragrance.org/transparency-list",
                3691,
                "Official IFRA online reference; row export not bundled",
            )
            return
        rows: list[dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                lowered = {str(key).casefold(): str(value).strip() for key, value in row.items()}
                name = lowered.get("principal name") or lowered.get("name") or ""
                cas = lowered.get("cas n°") or lowered.get("cas number") or lowered.get("cas") or ""
                if name:
                    rows.append({"name": name, "cas": cas})
        self.source(
            "ifra_transparency_2025_rows", csv_path, "official_reference_rows",
            "official_reference_index", "Identifier and palette-scope lookup",
            "Formulation readiness, legal compliance or safety approval", len(rows),
            "https://ifrafragrance.org/transparency-list",
            "Operator-provided export from official IFRA Transparency List",
        )
        for index, row in enumerate(rows):
            self.observation(
                "ifra_transparency_2025_rows", str(index), row["name"], row["cas"] or None,
                "transparency_list_presence", True, None, "official_reference_index",
            )

    def ingest_science(self) -> None:
        path = ROOT / "fragrance_ai" / "data" / "scientific_properties.db"
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            rows = list(source.execute("SELECT * FROM molecular_properties ORDER BY ingredient_id"))
        self.source(
            "pubchem_descriptors",
            path,
            "molecular_properties",
            "mixed_public_descriptor",
            "PubChem molecular descriptors and published-composition-derived UVCB descriptor centroids with row-level source references",
            "Vapor, odor-threshold, supplier or human-olfaction claims",
            len(rows),
            "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "US NCBI PubChem public service plus cited GC-MS/ISO composition references; observe row source_ref and claim boundary",
        )
        property_units = {
            "molecular_weight": "g/mol", "xlogp": None, "tpsa": "A^2",
            "hbond_donors": "count", "hbond_acceptors": "count",
            "rotatable_bonds": "count", "complexity": None,
            "vapor_pressure_pa_25c": "Pa", "boiling_point_c": "degC",
            "odor_threshold_ppm": "ppm", "source_ref": None, "verified_on": None,
        }
        for row in rows:
            for field, unit in property_units.items():
                # Compound odor thresholds are historical human sensory
                # endpoints even when stored beside physicochemical values.
                # The non-human hub must not ingest or reclassify them.
                if field == "odor_threshold_ppm":
                    continue
                evidence = (
                    "lineage"
                    if field in {"source_ref", "verified_on"}
                    else "composition_derived_reference_descriptor"
                    if "composition-derived" in str(row["source_ref"]).casefold()
                    else "official_descriptor"
                )
                self.observation(
                    "pubchem_descriptors", str(row["ingredient_id"]), str(row["ingredient_id"]),
                    row["cas_number"], field, row[field], unit,
                    evidence,
                )

    def ingest_epa_comptox(self) -> None:
        path = ROOT / "fragrance_ai" / "data" / "epa_comptox_extract.db"
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            source_files = list(source.execute("SELECT * FROM source_files ORDER BY source_id"))
            chemicals = list(source.execute("SELECT * FROM chemicals ORDER BY ingredient_id, dtxsid"))
            cpdat = list(source.execute(
                """SELECT ingredient_id, record_type, COUNT(*) AS row_count,
                          GROUP_CONCAT(DISTINCT function_category) AS function_categories,
                          GROUP_CONCAT(DISTINCT puc_general_category) AS general_categories,
                          GROUP_CONCAT(DISTINCT puc_product_family) AS product_families,
                          GROUP_CONCAT(DISTINCT puc_product_type) AS product_types,
                          MIN(lower_weight_fraction) AS min_weight_fraction,
                          MAX(upper_weight_fraction) AS max_weight_fraction
                   FROM cpdat_observations
                   GROUP BY ingredient_id, record_type
                   ORDER BY ingredient_id, record_type"""
            ))
            toxref_studies = list(source.execute(
                """SELECT ingredient_id, COUNT(*) AS row_count,
                          GROUP_CONCAT(DISTINCT study_type) AS study_types,
                          GROUP_CONCAT(DISTINCT species) AS species,
                          GROUP_CONCAT(DISTINCT admin_route) AS routes
                   FROM toxref_studies GROUP BY ingredient_id ORDER BY ingredient_id"""
            ))
            toxref_pods = list(source.execute(
                """SELECT ingredient_id, COUNT(*) AS row_count,
                          GROUP_CONCAT(DISTINCT calc_pod_type) AS pod_types,
                          GROUP_CONCAT(DISTINCT species) AS species,
                          GROUP_CONCAT(DISTINCT admin_route) AS routes
                   FROM toxref_pods GROUP BY ingredient_id ORDER BY ingredient_id"""
            ))
            toxval = list(source.execute(
                """SELECT ingredient_id, COUNT(*) AS row_count,
                          COUNT(DISTINCT source_table) AS source_tables,
                          GROUP_CONCAT(DISTINCT effect_category) AS effect_categories,
                          GROUP_CONCAT(DISTINCT study_type) AS study_types,
                          GROUP_CONCAT(DISTINCT species) AS species
                   FROM toxval_observations GROUP BY ingredient_id ORDER BY ingredient_id"""
            ))

        source_kinds = {
            "epa_dsstox_2025_12_csv": "official_chemical_identifier_structure",
            "epa_cpdat_v4_0": "official_product_use_and_composition",
            "epa_toxrefdb_v3_0_pod": "official_nonhuman_toxicology_pod",
            "epa_toxrefdb_v3_0_studies": "official_nonhuman_toxicology_studies",
            "epa_toxvaldb_v9_7_0_inputs": "official_toxicology_source_rows",
        }
        for row in source_files:
            self.source(
                str(row["source_id"]), path,
                source_kinds.get(str(row["source_id"]), "official_comptox_open_data"),
                "official_open_data",
                "Identifier/structure linkage, product-use evidence and toxicology screening",
                "Human odor-similarity certification, automatic safety approval, IFRA/legal "
                "compliance, supplier qualification or commercial release",
                int(row["record_count"]),
                str(row["origin_uri"]),
                str(row["license_note"]),
            )

        catalog = {str(item["ingredient_id"]): item for item in self.catalog_items}

        def item_for(ingredient_id: str) -> tuple[str, str | None]:
            item = catalog[ingredient_id]
            return str(item["name"]), item.get("cas_number")

        def values(value: object) -> list[str]:
            if value is None:
                return []
            return sorted({part.strip() for part in str(value).split(",") if part.strip()})

        dsstox_fields = (
            "dtxsid", "dtxcid", "preferred_name", "inchikey", "iupac_name", "smiles",
            "molecular_formula", "average_mass", "monoisotopic_mass",
            "qsar_ready_smiles", "ms_ready_smiles",
        )
        for row in chemicals:
            ingredient_id = str(row["ingredient_id"])
            name, cas = item_for(ingredient_id)
            for field in dsstox_fields:
                self.observation(
                    "epa_dsstox_2025_12_csv", str(row["dtxsid"]), name, cas,
                    field, row[field], "g/mol" if field in {"average_mass", "monoisotopic_mass"} else None,
                    "official_identifier_structure",
                )

        for row in cpdat:
            ingredient_id = str(row["ingredient_id"])
            name, cas = item_for(ingredient_id)
            record_type = str(row["record_type"])
            record_id = f"{ingredient_id}:{record_type}"
            for property_name, value, unit in (
                (f"cpdat_{record_type}_row_count", int(row["row_count"]), "rows"),
                (f"cpdat_{record_type}_function_categories", values(row["function_categories"]), None),
                (f"cpdat_{record_type}_general_categories", values(row["general_categories"]), None),
                (f"cpdat_{record_type}_product_families", values(row["product_families"]), None),
                (f"cpdat_{record_type}_product_types", values(row["product_types"]), None),
                (f"cpdat_{record_type}_min_weight_fraction", row["min_weight_fraction"], "fraction"),
                (f"cpdat_{record_type}_max_weight_fraction", row["max_weight_fraction"], "fraction"),
            ):
                self.observation(
                    "epa_cpdat_v4_0", record_id, name, cas, property_name, value, unit,
                    "official_product_use_reference",
                )

        for row in toxref_studies:
            ingredient_id = str(row["ingredient_id"])
            name, cas = item_for(ingredient_id)
            for property_name, value in (
                ("toxref_study_rows", int(row["row_count"])),
                ("toxref_study_types", values(row["study_types"])),
                ("toxref_species", values(row["species"])),
                ("toxref_admin_routes", values(row["routes"])),
            ):
                self.observation(
                    "epa_toxrefdb_v3_0_studies", ingredient_id, name, cas,
                    property_name, value, "rows" if property_name.endswith("rows") else None,
                    "official_nonhuman_toxicology_reference",
                )

        for row in toxref_pods:
            ingredient_id = str(row["ingredient_id"])
            name, cas = item_for(ingredient_id)
            for property_name, value in (
                ("toxref_pod_rows", int(row["row_count"])),
                ("toxref_pod_types", values(row["pod_types"])),
                ("toxref_pod_species", values(row["species"])),
                ("toxref_pod_admin_routes", values(row["routes"])),
            ):
                self.observation(
                    "epa_toxrefdb_v3_0_pod", ingredient_id, name, cas,
                    property_name, value, "rows" if property_name.endswith("rows") else None,
                    "official_nonhuman_toxicology_reference",
                )

        for row in toxval:
            ingredient_id = str(row["ingredient_id"])
            name, cas = item_for(ingredient_id)
            for property_name, value in (
                ("toxval_direct_match_rows", int(row["row_count"])),
                ("toxval_source_tables", int(row["source_tables"])),
                ("toxval_effect_categories", values(row["effect_categories"])),
                ("toxval_study_types", values(row["study_types"])),
                ("toxval_species", values(row["species"])),
            ):
                self.observation(
                    "epa_toxvaldb_v9_7_0_inputs", ingredient_id, name, cas,
                    property_name, value,
                    "rows" if property_name.endswith(("rows", "tables")) else None,
                    "official_toxicology_source_reference",
                )

    def ingest_reference_corpus(self) -> None:
        path = ROOT / "fragrance_ai" / "data" / "reference_fragrances.db"
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            perfumes = int(source.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0])
            notes = int(source.execute("SELECT COUNT(*) FROM perfume_notes").fetchone()[0])
            ingredient_rows = source.execute(
                "SELECT id, name, category, volatility, price_per_kg, ifra_limit, odor_strength, cas_number, origin FROM ingredients"
            ).fetchall()
        self.source(
            "workspace_reference_10k", path, "historical_note_corpus",
            "unverified_reference", "Weak note co-occurrence prior only",
            "Sensory truth, market coverage, price or regulatory authority", notes,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO hub_metadata VALUES ('base_reference_perfumes', ?)",
            (str(perfumes),),
        )
        for row in ingredient_rows:
            record_id, name, category, volatility, price, ifra, strength, cas, origin = row
            for field, value, unit in (
                ("category", category, None), ("volatility", volatility, "legacy score"),
                ("price_per_kg", price, "legacy currency/kg"),
                ("ifra_limit", ifra, "legacy unspecified"),
                ("odor_strength", strength, "legacy score"), ("origin", origin, None),
            ):
                self.observation(
                    "workspace_reference_10k", str(record_id), str(name), cas,
                    field, value, unit, "unverified_reference",
                )

    def ingest_legacy_ingredient_db(self, source_id: str, path: Path, trust: str) -> None:
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            rows = list(source.execute("SELECT * FROM ingredients"))
        self.source(
            source_id, path, "legacy_material_properties", trust,
            "Conflict-preserving engineering reference only",
            "Measured-property, supplier, safety or regulatory claims", len(rows),
        )
        ignored = {"id", "name", "cas_number", "description"}
        for row in rows:
            name = str(row["name"])
            cas = row["cas_number"] if "cas_number" in row.keys() else None
            for field in row.keys():
                if field in ignored:
                    continue
                self.observation(
                    source_id, str(row["id"]), name, cas, field, row[field], None, trust,
                )
            if "description" in row.keys():
                self.observation(
                    source_id, str(row["id"]), name, cas, "description",
                    row["description"], None, trust,
                )

    def ingest_newss_recipes(self) -> None:
        path = self.workspace / "Newss" / "data" / "databases" / "fragrance_ai.db"
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            recipe_rows = source.execute("SELECT id, name FROM recipes ORDER BY id").fetchall()
            note_rows = source.execute(
                """SELECT ri.recipe_id, n.name, ri.note_position, ri.percentage
                FROM recipe_ingredients ri JOIN fragrance_notes n ON n.id = ri.note_id
                ORDER BY ri.recipe_id, ri.id"""
            ).fetchall()
            note_catalog = source.execute(
                "SELECT id, name, note_type, fragrance_family, intensity, longevity, sillage FROM fragrance_notes"
            ).fetchall()
        self.source(
            "newss_recipe_reference", path, "recipe_and_note_reference",
            "unverified_reference", "Weak formula-note co-occurrence prior",
            "Exact commercial formulas, sensory truth or intellectual-property claims",
            len(note_rows),
        )
        for recipe_id, name in recipe_rows:
            self.connection.execute(
                "INSERT INTO formula_references VALUES (?, ?, ?, ?, 1)",
                ("newss_recipe_reference", str(recipe_id), str(name), "unverified_reference"),
            )
        registered = {str(recipe_id) for recipe_id, _name in recipe_rows}
        for recipe_id in sorted({str(row[0]) for row in note_rows} - registered):
            self.connection.execute(
                "INSERT INTO formula_references VALUES (?, ?, ?, ?, 1)",
                (
                    "newss_recipe_reference",
                    recipe_id,
                    f"orphan_reference:{recipe_id}",
                    "unverified_reference",
                ),
            )
        for recipe_id, name, pyramid, amount in note_rows:
            self.connection.execute(
                "INSERT INTO formula_notes VALUES (?, ?, ?, ?, ?, ?)",
                ("newss_recipe_reference", str(recipe_id), str(name), normalized(str(name)), pyramid, amount),
            )
        for note_id, name, note_type, family, intensity, longevity, sillage in note_catalog:
            for field, value in (
                ("note_type", note_type), ("fragrance_family", family),
                ("intensity", intensity), ("longevity", longevity), ("sillage", sillage),
            ):
                self.observation(
                    "newss_recipe_reference", str(note_id), str(name), None,
                    field, value, None, "unverified_reference",
                )

    def ingest_olfactory_dna(self) -> None:
        path = self.workspace / "Newss" / "olfactory_dna.db"
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            formulas = list(source.execute("SELECT id, name, recipe FROM existing_fragrances"))
            ingredients = list(source.execute("SELECT * FROM ingredients"))
        self.source(
            "newss_olfactory_reference", path, "legacy_formula_reference",
            "unverified_reference", "Weak formula-note co-occurrence prior",
            "Exact commercial formulas, measured properties or regulatory claims",
            len(formulas),
        )
        for row in formulas:
            ref = str(row["id"])
            self.connection.execute(
                "INSERT INTO formula_references VALUES (?, ?, ?, ?, 1)",
                ("newss_olfactory_reference", ref, str(row["name"]), "unverified_reference"),
            )
            recipe = json.loads(row["recipe"])
            for pyramid, materials in recipe.items():
                for name, amount in materials.items():
                    self.connection.execute(
                        "INSERT INTO formula_notes VALUES (?, ?, ?, ?, ?, ?)",
                        ("newss_olfactory_reference", ref, name, normalized(name), pyramid, amount),
                    )
        for row in ingredients:
            for field in ("category", "odor_family", "volatility", "intensity", "price_per_kg", "ifra_limit"):
                self.observation(
                    "newss_olfactory_reference", str(row["id"]), str(row["name"]),
                    row["cas_number"], field, row[field], None, "unverified_reference",
                )

    def ingest_synthetic_recipes(self) -> None:
        path = self.workspace / "ai project" / "ai_perfume" / "perfume_ai.db"
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            rows = list(source.execute("SELECT * FROM recipes"))
        self.source(
            "ai_perfume_synthetic_recipes", path, "synthetic_recipe_reference",
            "synthetic_training", "Offline training/robustness experiments only",
            "Historical, measured, human or commercial formula evidence", len(rows),
        )
        for row in rows:
            ref = str(row["id"])
            self.connection.execute(
                "INSERT INTO formula_references VALUES (?, ?, ?, ?, 1)",
                ("ai_perfume_synthetic_recipes", ref, str(row["name"]), "synthetic_training"),
            )
            for pyramid, column in (("top", "top_notes"), ("heart", "middle_notes"), ("base", "base_notes")):
                for name in json.loads(row[column]):
                    self.connection.execute(
                        "INSERT INTO formula_notes VALUES (?, ?, ?, ?, ?, NULL)",
                        ("ai_perfume_synthetic_recipes", ref, str(name), normalized(str(name)), pyramid),
                    )

    def ingest_initial_json(self) -> None:
        folder = self.workspace / "Newss" / "data" / "initial"
        for path in sorted(folder.glob("*fragrance_ingredients.json")) if folder.exists() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("ingredients", [])
            source_id = "initial_" + normalized(path.stem)
            self.source(
                source_id, path, "legacy_material_catalog", "unverified_reference",
                "Alias, family and descriptive reference",
                "Supplier, measured-property, safety or regulatory claims", len(rows),
            )
            for index, row in enumerate(rows):
                name = str(row.get("english_name") or row.get("name") or row.get("korean_name") or "")
                if not name:
                    continue
                cas = row.get("cas_number")
                for field in (
                    "korean_name", "category", "fragrance_family", "note_type", "description",
                    "origin", "intensity", "longevity", "sillage", "price_range",
                    "safety_rating", "allergen_info", "blending_guidelines", "molecular_formula",
                ):
                    self.observation(
                        source_id, str(index), name, cas, field, row.get(field), None,
                        "unverified_reference",
                    )

    def ingest_regulatory_reference(self) -> None:
        path = self.workspace / "ai project" / "fragrance_database.db"
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            rows = source.execute(
                """SELECT cas_number, category, max_concentration_ppm, restriction_status,
                          effective_date, MAX(last_updated)
                FROM ifra_restrictions
                GROUP BY cas_number, category, max_concentration_ppm, restriction_status, effective_date"""
            ).fetchall()
        self.source(
            "legacy_ifra_rows", path, "regulatory_reference", "unverified_regulatory_reference",
            "Conflict and regression diagnostics only",
            "IFRA compliance, legal safety or commercial release decisions", len(rows),
        )
        for index, row in enumerate(rows):
            cas, category, limit, status, effective, _updated = row
            self.connection.execute(
                "INSERT INTO regulatory_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    "legacy_ifra_rows", str(index), cas, category, status, limit, "ppm",
                    effective, "unverified_regulatory_reference",
                ),
            )

    def register_remaining_assets(self) -> None:
        roots = (
            self.workspace / "Newss" / "data",
            self.workspace / "ai project" / "ai_perfume" / "data",
            self.workspace / "ai project" / "ai_perfume" / "generated_recipes",
        )
        for folder in roots:
            if not folder.exists():
                continue
            for path in sorted(item for item in folder.rglob("*") if item.is_file()):
                if path.suffix.casefold() not in DATA_EXTENSIONS or path.resolve() in self.known_paths:
                    continue
                relative = path.relative_to(self.workspace).as_posix()
                lowered = relative.casefold()
                if any(term in lowered for term in HUMAN_TERMS):
                    self.excluded_human_paths.add(path.resolve())
                    continue
                synthetic = any(
                    term in lowered for term in ("generated_recipe", "training", "200k", "movie", "advanced")
                )
                trust = "synthetic_training" if synthetic else "unverified_reference"
                kind = "synthetic_training_asset" if synthetic else "unverified_workspace_asset"
                source_id = "asset_" + hashlib.sha256(relative.encode()).hexdigest()[:16]
                self.source(
                    source_id, path, kind, trust,
                    "Offline training only" if synthetic else "Searchable lineage/reference only",
                    "Measured, supplier, regulatory, human or commercial evidence",
                )

    def build(self, ifra_csv: Path | None = None) -> dict[str, int | str]:
        self.ingest_catalog()
        self.ingest_core_auxiliary()
        self.ingest_ifra_reference(ifra_csv)
        self.ingest_science()
        self.ingest_epa_comptox()
        self.ingest_reference_corpus()
        self.ingest_legacy_ingredient_db(
            "legacy_moga_properties", ROOT / "fragrance_ai" / "data" / "moga_ingredients.db",
            "unverified_engineering",
        )
        self.ingest_legacy_ingredient_db(
            "newss_production_properties", self.workspace / "Newss" / "data" / "fragrance_production.db",
            "unverified_engineering",
        )
        self.ingest_newss_recipes()
        self.ingest_olfactory_dna()
        self.ingest_synthetic_recipes()
        self.ingest_initial_json()
        self.ingest_regulatory_reference()
        self.register_remaining_assets()
        for path in self.workspace.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in DATA_EXTENSIONS:
                continue
            relative_path = path.relative_to(self.workspace)
            ignored_parts = {".venv", ".cache", ".git", "build", "dist", "__pycache__"}
            if any(part.casefold() in ignored_parts for part in relative_path.parts):
                continue
            lowered = relative_path.as_posix().casefold()
            if any(term in lowered for term in HUMAN_TERMS):
                self.excluded_human_paths.add(path.resolve())
        forbidden = self.connection.execute(
            """SELECT local_path FROM data_sources
            WHERE lower(local_path) LIKE '%human%'
               OR lower(local_path) LIKE '%sensory%'
               OR lower(local_path) LIKE '%expert%'
               OR lower(local_path) LIKE '%feedback%'
               OR lower(local_path) LIKE '%validation%'
               OR lower(local_path) LIKE '%panel%'"""
        ).fetchall()
        if forbidden:
            raise RuntimeError(f"human validation assets entered non-human hub: {forbidden}")
        self.connection.execute(
            "INSERT OR REPLACE INTO hub_metadata VALUES ('human_validation_assets_excluded', ?)",
            (str(len(self.excluded_human_paths)),),
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO hub_metadata VALUES ('builder_version', '1.2')"
        )
        self.connection.commit()
        self.connection.execute("VACUUM")
        self.connection.close()
        hub = NonHumanDataHub(self.output)
        result = hub.stats()
        hub.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=str(ROOT.parent))
    parser.add_argument(
        "--output",
        default=str(ROOT / "fragrance_ai" / "data" / "nonhuman_data_hub.db"),
    )
    parser.add_argument("--ifra-csv", help="Operator-provided official IFRA Transparency CSV")
    args = parser.parse_args()
    output = Path(args.output)
    result = HubBuilder(Path(args.workspace_root), output).build(
        Path(args.ifra_csv) if args.ifra_csv else None
    )
    result.update({"output": str(output.resolve()), "sha256": sha256(output)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
