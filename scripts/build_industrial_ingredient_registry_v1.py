#!/usr/bin/env python
"""Build a source-pinned industrial-scale ingredient reference registry.

Broad molecule and odor-descriptor coverage is reference-only.  The builder
copies the existing safe catalog into a separate formulation table and never
promotes a molecule merely because it appears in a public odor archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import normalize_name  # noqa: E402


SCHEMA = "industrial-ingredient-registry-v1.2"
ARCHIVES = (
    "leffingwell",
    "goodscents",
    "flavornet",
    "aromadb",
    "ifra_2019",
    "flavordb",
)
EXPECTED_REFERENCE_MOLECULES = 29_240


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or any(None in row for row in rows):
        raise ValueError(f"invalid source CSV: {path}")
    return rows


def _canonical_smiles(value: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _registry_id(smiles: str) -> str:
    return "mol:" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]


def _descriptor_tokens(value: Any) -> Iterable[str]:
    for token in re.split(r"[;,|/]", str(value or "")):
        cleaned = " ".join(token.casefold().strip().split())
        if cleaned and cleaned not in {"nan", "none", "null", "0"}:
            yield cleaned[:160]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size=4096;
        PRAGMA foreign_keys=ON;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;

        CREATE TABLE registry_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE source_files (
            source_id TEXT NOT NULL,
            file_kind TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256)=64),
            bytes INTEGER NOT NULL CHECK(bytes>0),
            license_status TEXT NOT NULL,
            redistribution_allowed INTEGER NOT NULL CHECK(redistribution_allowed IN (0,1)),
            PRIMARY KEY (source_id, file_kind)
        );
        CREATE TABLE ingredients (
            registry_id TEXT PRIMARY KEY,
            canonical_smiles TEXT NOT NULL UNIQUE,
            preferred_name TEXT NOT NULL,
            iupac_name TEXT,
            molecular_weight REAL,
            representative_cid TEXT,
            source_count INTEGER NOT NULL DEFAULT 0,
            descriptor_count INTEGER NOT NULL DEFAULT 0,
            reference_only INTEGER NOT NULL DEFAULT 1 CHECK(reference_only=1)
        );
        CREATE TABLE ingredient_sources (
            registry_id TEXT NOT NULL REFERENCES ingredients(registry_id),
            source_id TEXT NOT NULL,
            source_cid TEXT NOT NULL,
            source_name TEXT,
            PRIMARY KEY (registry_id, source_id, source_cid)
        );
        CREATE TABLE ingredient_identifiers (
            registry_id TEXT NOT NULL REFERENCES ingredients(registry_id),
            identifier_type TEXT NOT NULL,
            identifier_value TEXT NOT NULL,
            source_id TEXT NOT NULL,
            PRIMARY KEY (identifier_type, identifier_value, source_id)
        );
        CREATE TABLE ingredient_names (
            registry_id TEXT NOT NULL REFERENCES ingredients(registry_id),
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            source_id TEXT NOT NULL,
            PRIMARY KEY (registry_id, normalized_name, source_id)
        );
        CREATE TABLE odor_descriptors (
            registry_id TEXT NOT NULL REFERENCES ingredients(registry_id),
            source_id TEXT NOT NULL,
            descriptor TEXT NOT NULL,
            normalized_descriptor TEXT NOT NULL,
            PRIMARY KEY (registry_id, source_id, normalized_descriptor)
        );
        CREATE TABLE formulation_materials (
            ingredient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            cas_number TEXT,
            formulation_tier TEXT NOT NULL CHECK(formulation_tier IN
                ('prototype_safe_active','prototype_conditional_active',
                 'formulation_metadata_only','reference_blocked')),
            risk_tier INTEGER NOT NULL,
            availability REAL NOT NULL,
            price_per_kg REAL NOT NULL,
            max_concentrate_percent REAL NOT NULL,
            linked_registry_id TEXT REFERENCES ingredients(registry_id),
            promotion_evidence TEXT NOT NULL
        );
        CREATE TABLE promotion_candidates (
            registry_id TEXT PRIMARY KEY REFERENCES ingredients(registry_id),
            evidence_score INTEGER NOT NULL,
            source_count INTEGER NOT NULL,
            descriptor_count INTEGER NOT NULL,
            molecular_weight REAL,
            ifra_reference INTEGER NOT NULL CHECK(ifra_reference IN (0,1)),
            promotion_status TEXT NOT NULL CHECK(promotion_status IN
                ('evidence_pending','structural_review_required')),
            required_evidence TEXT NOT NULL
        );
        CREATE TABLE safety_screening (
            registry_id TEXT PRIMARY KEY REFERENCES ingredients(registry_id),
            screening_status TEXT NOT NULL CHECK(screening_status IN
                ('evidence_pending','structural_review_required')),
            structural_alert_count INTEGER NOT NULL,
            structural_alerts_json TEXT NOT NULL,
            has_cas INTEGER NOT NULL CHECK(has_cas IN (0,1)),
            ifra_reference INTEGER NOT NULL CHECK(ifra_reference IN (0,1)),
            source_count INTEGER NOT NULL,
            descriptor_count INTEGER NOT NULL,
            molecular_weight REAL,
            required_evidence TEXT NOT NULL
        );
        """
    )


def _insert_name(
    connection: sqlite3.Connection,
    registry_id: str,
    value: str,
    source_id: str,
) -> None:
    name = " ".join(str(value or "").strip().split())[:300]
    normalized = normalize_name(name)
    if name and normalized:
        connection.execute(
            "INSERT OR IGNORE INTO ingredient_names VALUES (?,?,?,?)",
            (registry_id, name, normalized, source_id),
        )


def _load_molecules(
    connection: sqlite3.Connection,
    source_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    source_lookup: dict[str, dict[str, str]] = {}
    invalid_counts: dict[str, int] = {}
    for archive in ARCHIVES:
        lookup: dict[str, str] = {}
        invalid = 0
        for row in _rows(source_root / archive / "molecules.csv"):
            canonical = _canonical_smiles(row.get("IsomericSMILES", ""))
            if canonical is None:
                invalid += 1
                continue
            registry_id = _registry_id(canonical)
            cid = str(row.get("CID", "")).strip()
            name = str(row.get("name", "")).strip()
            iupac = str(row.get("IUPACName", "")).strip()
            try:
                molecular_weight = float(row["MolecularWeight"])
            except (KeyError, TypeError, ValueError):
                molecular_weight = None
            connection.execute(
                """
                INSERT INTO ingredients(
                    registry_id, canonical_smiles, preferred_name, iupac_name,
                    molecular_weight, representative_cid
                ) VALUES (?,?,?,?,?,?)
                ON CONFLICT(registry_id) DO UPDATE SET
                    preferred_name = CASE
                        WHEN length(excluded.preferred_name) > 0
                         AND (length(ingredients.preferred_name) = 0
                          OR length(excluded.preferred_name) < length(ingredients.preferred_name))
                        THEN excluded.preferred_name ELSE ingredients.preferred_name END,
                    iupac_name = COALESCE(NULLIF(ingredients.iupac_name,''), excluded.iupac_name),
                    molecular_weight = COALESCE(ingredients.molecular_weight, excluded.molecular_weight),
                    representative_cid = COALESCE(NULLIF(ingredients.representative_cid,''), excluded.representative_cid)
                """,
                (
                    registry_id,
                    canonical,
                    name or iupac,
                    iupac or None,
                    molecular_weight,
                    cid or None,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO ingredient_sources VALUES (?,?,?,?)",
                (registry_id, archive, cid, name or None),
            )
            _insert_name(connection, registry_id, name, archive)
            _insert_name(connection, registry_id, iupac, archive)
            if cid:
                lookup[cid] = registry_id
        source_lookup[archive] = lookup
        invalid_counts[archive] = invalid
    return source_lookup, invalid_counts


_STRUCTURAL_ALERT_SMARTS = {
    "peroxide": "[O;X1,X2]-[O;X1,X2]",
    "azide": "[N-]=[N+]=N",
    "isocyanate": "N=C=O",
    "acid_halide": "[C;X3](=[O;X1])[F,Cl,Br,I]",
    "epoxide": "[O;r3]1[C;r3][C;r3]1",
    "diazo": "[N-]=[N+]=[C,N]",
}
_STRUCTURAL_ALERT_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in _STRUCTURAL_ALERT_SMARTS.items()
}
if any(pattern is None for pattern in _STRUCTURAL_ALERT_PATTERNS.values()):
    raise RuntimeError("invalid structural alert SMARTS contract")
_ALLOWED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
_REQUIRED_PROMOTION_EVIDENCE = (
    "identity_cas_structure",
    "supplier_sku_lot",
    "coa",
    "sds",
    "ifra_limit_or_certificate",
    "regulatory_status",
    "market_category_rule_pack",
    "toxicology_assessment",
    "quantitative_allergen",
    "price_availability",
    "concentration_cap",
    "odor_profile",
    "formulation_spec",
    "expert_signoff",
)


def _valid_cas_number(value: str) -> bool:
    match = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", value)
    if match is None:
        return False
    body = match.group(1) + match.group(2)
    checksum = (
        sum(
            multiplier * int(digit)
            for multiplier, digit in enumerate(reversed(body), start=1)
        )
        % 10
    )
    return checksum == int(match.group(3))


def _structural_alerts(
    canonical_smiles: str, molecular_weight: float | None
) -> list[str]:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return ["invalid_structure"]
    alerts = []
    if "." in canonical_smiles:
        alerts.append("multi_fragment")
    if any(
        atom.GetAtomicNum() not in _ALLOWED_ATOMIC_NUMBERS
        for atom in molecule.GetAtoms()
    ):
        alerts.append("metal_or_unsupported_element")
    if molecular_weight is None:
        alerts.append("molecular_weight_missing")
    elif not 30.0 <= molecular_weight <= 500.0:
        alerts.append("molecular_weight_outside_screening_range")
    for name, pattern in _STRUCTURAL_ALERT_PATTERNS.items():
        if pattern is None:  # guarded by the module-level contract above
            raise RuntimeError(f"invalid structural alert pattern: {name}")
        if molecule.HasSubstructMatch(pattern):
            alerts.append(name)
    return sorted(set(alerts))


def _populate_identifiers_and_screening(
    connection: sqlite3.Connection,
    source_root: Path,
    lookups: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    cas_to_cid = {
        str(key): str(value)
        for key, value in json.loads(
            (source_root / "goodscents" / "cas_to_cid.json").read_text(encoding="utf-8")
        ).items()
    }
    cas_registry: set[str] = set()
    invalid_cas = 0
    for cas_number, cid in sorted(cas_to_cid.items()):
        if not _valid_cas_number(cas_number):
            invalid_cas += 1
            continue
        registry_id = lookups["goodscents"].get(cid)
        if registry_id is None:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO ingredient_identifiers VALUES (?,?,?,?)",
            (registry_id, "CAS", cas_number, "goodscents"),
        )
        cas_registry.add(registry_id)
    ifra_registry = set(lookups["ifra_2019"].values())
    required_evidence = json.dumps(
        _REQUIRED_PROMOTION_EVIDENCE,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    screenable = 0
    structural_review = 0
    for row in connection.execute(
        """
        SELECT registry_id, canonical_smiles, molecular_weight,
               source_count, descriptor_count
        FROM ingredients ORDER BY registry_id
        """
    ):
        registry_id = str(row[0])
        alerts = _structural_alerts(str(row[1]), row[2])
        if alerts:
            status = "structural_review_required"
            structural_review += 1
        else:
            status = "evidence_pending"
            screenable += 1
        connection.execute(
            "INSERT INTO safety_screening VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                registry_id,
                status,
                len(alerts),
                json.dumps(alerts, separators=(",", ":")),
                int(registry_id in cas_registry),
                int(registry_id in ifra_registry),
                int(row[3]),
                int(row[4]),
                row[2],
                required_evidence,
            ),
        )
    return {
        "safety_screened": screenable + structural_review,
        "evidence_pending": screenable,
        "structural_review_required": structural_review,
        "with_cas": len(cas_registry),
        "invalid_cas_identifiers_rejected": invalid_cas,
        "ifra_reference": len(ifra_registry),
    }


def _add_descriptor(
    connection: sqlite3.Connection,
    registry_id: str | None,
    source_id: str,
    descriptor: str,
) -> None:
    if registry_id is None:
        return
    normalized = normalize_name(descriptor)
    if not normalized:
        return
    connection.execute(
        "INSERT OR IGNORE INTO odor_descriptors VALUES (?,?,?,?)",
        (registry_id, source_id, descriptor[:160], normalized[:160]),
    )


def _load_descriptors(
    connection: sqlite3.Connection,
    source_root: Path,
    lookups: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    unmatched: dict[str, int] = {}
    for archive in ARCHIVES:
        missing = 0
        rows = _rows(source_root / archive / "behavior.csv")
        cas_to_cid: dict[str, str] = {}
        if archive == "goodscents":
            cas_to_cid = {
                str(key): str(value)
                for key, value in json.loads(
                    (source_root / archive / "cas_to_cid.json").read_text(
                        encoding="utf-8"
                    )
                ).items()
            }
        for row in rows:
            stimulus = str(row.get("Stimulus", "")).strip()
            source_cid = cas_to_cid.get(stimulus, stimulus)
            registry_id = lookups[archive].get(source_cid)
            if registry_id is None:
                missing += 1
                continue
            if archive == "leffingwell":
                for column, value in row.items():
                    if column != "Stimulus" and str(value).strip() == "1":
                        _add_descriptor(connection, registry_id, archive, column)
            elif archive == "goodscents":
                for value in _descriptor_tokens(row.get("Descriptors")):
                    _add_descriptor(connection, registry_id, archive, value)
            elif archive == "flavornet":
                for value in _descriptor_tokens(row.get("Descriptors")):
                    _add_descriptor(connection, registry_id, archive, value)
            elif archive == "aromadb":
                value = row.get("Filtered Descriptors") or row.get("Raw Descriptors")
                for token in _descriptor_tokens(value):
                    _add_descriptor(connection, registry_id, archive, token)
            elif archive == "ifra_2019":
                for column in ("Descriptor 1", "Descriptor 2", "Descriptor 3"):
                    for token in _descriptor_tokens(row.get(column)):
                        _add_descriptor(connection, registry_id, archive, token)
            elif archive == "flavordb":
                for column in ("Odor Percepts", "Flavor Percepts"):
                    for token in _descriptor_tokens(row.get(column)):
                        _add_descriptor(connection, registry_id, archive, token)
        unmatched[archive] = missing
    return unmatched


def _link_formulation_catalog(
    connection: sqlite3.Connection,
    safe_catalog_path: Path,
    source_root: Path,
    lookups: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    catalog = json.loads(safe_catalog_path.read_text(encoding="utf-8"))
    cas_to_cid = {
        str(key): str(value)
        for key, value in json.loads(
            (source_root / "goodscents" / "cas_to_cid.json").read_text(encoding="utf-8")
        ).items()
    }
    linked = 0
    active = 0
    conditional = 0
    for item in catalog["ingredients"]:
        if bool(item.get("formulation_ready")) and not bool(item.get("blocked")):
            if int(item["risk_tier"]) <= 1 and float(item["availability"]) >= 0.75:
                tier = "prototype_safe_active"
                active += 1
            elif int(item["risk_tier"]) == 2 and float(item["availability"]) >= 0.75:
                tier = "prototype_conditional_active"
                conditional += 1
            else:
                tier = "formulation_metadata_only"
        else:
            tier = "reference_blocked"
        cas_number = str(item.get("cas_number") or "").strip()
        linked_registry_id = None
        cid = cas_to_cid.get(cas_number)
        if cid:
            linked_registry_id = lookups["goodscents"].get(cid)
        if linked_registry_id is None:
            normalized_names = {
                normalize_name(str(item.get("name", ""))),
                *{normalize_name(str(alias)) for alias in item.get("aliases", [])},
            }
            normalized_names.discard("")
            if normalized_names:
                placeholders = ",".join("?" for _ in normalized_names)
                candidates = connection.execute(
                    f"""
                    SELECT registry_id, COUNT(*) AS matches
                    FROM ingredient_names
                    WHERE normalized_name IN ({placeholders})
                    GROUP BY registry_id
                    ORDER BY matches DESC, registry_id
                    LIMIT 2
                    """,
                    tuple(sorted(normalized_names)),
                ).fetchall()
                if len(candidates) == 1 or (
                    len(candidates) > 1 and candidates[0][1] > candidates[1][1]
                ):
                    linked_registry_id = str(candidates[0][0])
        if linked_registry_id is not None:
            linked += 1
        connection.execute(
            "INSERT INTO formulation_materials VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                item["ingredient_id"],
                item["name"],
                cas_number or None,
                tier,
                int(item["risk_tier"]),
                float(item["availability"]),
                float(item["price_per_kg"]),
                float(item["max_concentrate_percent"]),
                linked_registry_id,
                (
                    "curated prototype metadata; supplier SKU/lot COA/SDS/IFRA "
                    "evidence is still required for qualified/commercial promotion"
                ),
            ),
        )
    return {
        "catalog_materials": len(catalog["ingredients"]),
        "prototype_safe_active": active,
        "prototype_conditional_active": conditional,
        "prototype_active_total": active + conditional,
        "molecularly_linked": linked,
    }


def build(
    *,
    source_root: Path,
    source_manifest_path: Path,
    safe_catalog_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    declared = manifest["source_files"]
    source_evidence = []
    for archive in ARCHIVES:
        for filename, kind in (
            ("molecules.csv", "molecules"),
            ("behavior.csv", "behavior"),
        ):
            relative = f"pyrfume_all/{archive}/{filename}"
            path = source_root / archive / filename
            evidence = declared[relative]
            if (
                sha256_file(path) != evidence["sha256"]
                or path.stat().st_size != evidence["bytes"]
            ):
                raise RuntimeError(f"industrial registry source changed: {relative}")
            source_evidence.append(
                {
                    "source_id": archive,
                    "file_kind": kind,
                    "path": str(path.resolve()),
                    "sha256": evidence["sha256"],
                    "bytes": evidence["bytes"],
                }
            )
    cas_map_path = source_root / "goodscents" / "cas_to_cid.json"
    source_evidence.append(
        {
            "source_id": "goodscents",
            "file_kind": "cas_to_cid",
            "path": str(cas_map_path.resolve()),
            "sha256": sha256_file(cas_map_path),
            "bytes": cas_map_path.stat().st_size,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".db", dir=output_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        _create_schema(connection)
        connection.executemany(
            "INSERT INTO source_files VALUES (?,?,?,?,?,?,?)",
            [
                (
                    item["source_id"],
                    item["file_kind"],
                    item["path"],
                    item["sha256"],
                    item["bytes"],
                    "source_specific_unverified_workspace_research",
                    0,
                )
                for item in source_evidence
            ],
        )
        lookups, invalid_molecules = _load_molecules(connection, source_root)
        descriptor_unmatched = _load_descriptors(connection, source_root, lookups)
        formulation = _link_formulation_catalog(
            connection, safe_catalog_path, source_root, lookups
        )
        connection.executescript(
            """
            UPDATE ingredients SET source_count = (
                SELECT COUNT(*) FROM ingredient_sources s
                WHERE s.registry_id = ingredients.registry_id
            );
            UPDATE ingredients SET descriptor_count = (
                SELECT COUNT(*) FROM odor_descriptors d
                WHERE d.registry_id = ingredients.registry_id
            );
            CREATE INDEX idx_ingredient_names_normalized
                ON ingredient_names(normalized_name, registry_id);
            CREATE INDEX idx_odor_descriptors_normalized
                ON odor_descriptors(normalized_descriptor, registry_id);
            CREATE INDEX idx_ingredient_sources_source
                ON ingredient_sources(source_id, source_cid);
            """
        )
        screening = _populate_identifiers_and_screening(
            connection, source_root, lookups
        )
        connection.executescript(
            """
            CREATE INDEX idx_ingredient_identifiers_value
                ON ingredient_identifiers(identifier_type, identifier_value);
            CREATE INDEX idx_safety_screening_status
                ON safety_screening(screening_status, registry_id);
            INSERT INTO promotion_candidates(
                registry_id, evidence_score, source_count, descriptor_count,
                molecular_weight, ifra_reference, promotion_status, required_evidence
            )
            SELECT i.registry_id,
                   10 * i.source_count + MIN(i.descriptor_count, 50)
                       + CASE WHEN s.ifra_reference = 1 THEN 100 ELSE 0 END
                       + CASE WHEN s.has_cas = 1 THEN 20 ELSE 0 END
                       - CASE WHEN s.structural_alert_count > 0 THEN 50 ELSE 0 END,
                   i.source_count, i.descriptor_count, i.molecular_weight,
                   s.ifra_reference, s.screening_status, s.required_evidence
            FROM ingredients i
            JOIN safety_screening s ON s.registry_id = i.registry_id
            WHERE NOT EXISTS (
                SELECT 1 FROM formulation_materials f
                WHERE f.linked_registry_id = i.registry_id
            );
            CREATE INDEX idx_promotion_candidates_score
                ON promotion_candidates(evidence_score DESC, registry_id);
            INSERT INTO registry_metadata VALUES
                ('schema', 'industrial-ingredient-registry-v1.2'),
                ('reference_is_formulation_permission', 'false'),
                ('all_reference_molecules_have_safety_disposition', 'true'),
                ('all_unlinked_molecules_have_promotion_path', 'true'),
                ('commercial_promotion_requires_supplier_evidence', 'true'),
                ('promotion_requires_independent_signature', 'true');
            """
        )
        counts = {
            "reference_molecules": connection.execute(
                "SELECT COUNT(*) FROM ingredients"
            ).fetchone()[0],
            "source_links": connection.execute(
                "SELECT COUNT(*) FROM ingredient_sources"
            ).fetchone()[0],
            "names": connection.execute(
                "SELECT COUNT(*) FROM ingredient_names"
            ).fetchone()[0],
            "descriptor_assertions": connection.execute(
                "SELECT COUNT(*) FROM odor_descriptors"
            ).fetchone()[0],
            "molecules_with_descriptors": connection.execute(
                "SELECT COUNT(*) FROM ingredients WHERE descriptor_count > 0"
            ).fetchone()[0],
            "promotion_candidates_total": connection.execute(
                "SELECT COUNT(*) FROM promotion_candidates"
            ).fetchone()[0],
            "promotion_evidence_pending": connection.execute(
                "SELECT COUNT(*) FROM promotion_candidates "
                "WHERE promotion_status='evidence_pending'"
            ).fetchone()[0],
            "promotion_structural_review_required": connection.execute(
                "SELECT COUNT(*) FROM promotion_candidates "
                "WHERE promotion_status='structural_review_required'"
            ).fetchone()[0],
            "high_priority_candidates": connection.execute(
                """
                SELECT COUNT(*) FROM promotion_candidates p
                JOIN ingredients i ON i.registry_id = p.registry_id
                WHERE i.source_count >= 2
                  AND i.descriptor_count >= 1
                  AND i.molecular_weight BETWEEN 50.0 AND 350.0
                  AND i.canonical_smiles NOT LIKE '%.%'
                  AND p.ifra_reference = 1
                """
            ).fetchone()[0],
        }
        counts.update(screening)
        if int(counts["reference_molecules"]) != EXPECTED_REFERENCE_MOLECULES:
            raise RuntimeError(
                "industrial reference molecule count changed: "
                f"{counts['reference_molecules']}"
            )
        if formulation["prototype_safe_active"] != 29:
            raise RuntimeError("prototype-safe active material count changed")
        if formulation["prototype_conditional_active"] != 5:
            raise RuntimeError("prototype-conditional active material count changed")
        if int(counts["safety_screened"]) != EXPECTED_REFERENCE_MOLECULES:
            raise RuntimeError("not every reference molecule was safety screened")
        expected_queue = EXPECTED_REFERENCE_MOLECULES - int(
            formulation["molecularly_linked"]
        )
        if int(counts["promotion_candidates_total"]) != expected_queue:
            raise RuntimeError("not every unlinked molecule received a promotion path")
        connection.commit()
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("industrial registry foreign keys failed")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("industrial registry integrity check failed")
        connection.execute("VACUUM")
        connection.close()
        connection = None
        os.replace(temporary, output_path)
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "industrial_scale_reference_registry_built",
        "database": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "integrity_check": "ok",
        },
        "counts": {**counts, **formulation},
        "source_binding": source_evidence,
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256_file(source_manifest_path),
        },
        "safe_catalog": {
            "path": str(safe_catalog_path.resolve()),
            "sha256": sha256_file(safe_catalog_path),
        },
        "invalid_molecule_rows": invalid_molecules,
        "unmatched_descriptor_rows": descriptor_unmatched,
        "tier_contract": {
            "reference_molecules_are_searchable": True,
            "reference_molecules_are_formula_eligible": False,
            "all_reference_molecules_have_safety_screening": True,
            "all_unlinked_molecules_have_promotion_path": True,
            "prototype_safe_active_materials": formulation["prototype_safe_active"],
            "prototype_conditional_active_materials": formulation[
                "prototype_conditional_active"
            ],
            "prototype_active_total": formulation["prototype_active_total"],
            "conditional_activation_requires": [
                "explicit max_risk_tier >= 2",
                "catalog concentration cap",
                "allergen and oxidation warning review",
            ],
            "reference_promotion_candidates": counts["promotion_candidates_total"],
            "reference_evidence_pending_candidates": counts[
                "promotion_evidence_pending"
            ],
            "reference_structural_review_required": counts[
                "promotion_structural_review_required"
            ],
            "required_promotion_evidence": list(_REQUIRED_PROMOTION_EVIDENCE),
            "qualified_or_commercial_materials": 0,
            "promotion_requires": [
                "supplier SKU and lot",
                "COA, SDS and applicable IFRA certificate",
                "quantitative allergen evidence",
                "market and product-category rule pack",
                "allowlisted independent Ed25519 signature",
            ],
            "promotion_signature_scope": [
                "exact registry database SHA-256",
                "registry molecule ID and canonical structure",
                "target formulation tier",
                "market and product category",
                "every evidence file SHA-256",
            ],
            "runtime_auto_activation": {
                "enabled": True,
                "requires_signed_formulation_spec": True,
                "merges_into_default_formula_pool_on_service_start": True,
                "rechecks_market_category_expiry_risk_and_concentration": True,
            },
        },
        "distribution": {
            "workspace_research_only": True,
            "included_in_wheel": False,
            "source_specific_license_review_required": True,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "All 29,240 molecules receive a deterministic safety disposition, and "
            "every unlinked molecule receives an evidence-gated promotion route. "
            "Screening is not a safety certificate: policy-blocked materials remain "
            "blocked, while the 29 default-safe and 5 risk-tier-2 conditional "
            "materials are the static active pool. A reference material is added "
            "automatically at service start only after the required identity, supplier, "
            "regulatory, toxicology, concentration, odor, formulation spec and signed "
            "evidence passes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\Users\user\Desktop\Game\server\data\pom_data\pyrfume_all"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "fragrance_ai" / "data" / "physsim_r2_manifest.json",
    )
    parser.add_argument(
        "--safe-catalog",
        type=Path,
        default=ROOT / "fragrance_ai" / "data" / "safe_ingredient_catalog.json",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "industrial_ingredient_registry_v1.json",
    )
    args = parser.parse_args()
    result = build(
        source_root=args.source_root.resolve(strict=True),
        source_manifest_path=args.source_manifest.resolve(strict=True),
        safe_catalog_path=args.safe_catalog.resolve(strict=True),
        output_path=args.database.resolve(),
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
