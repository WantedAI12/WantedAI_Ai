"""Read-only access to the workspace industrial ingredient registry.

The registry deliberately separates broad reference coverage from materials
that have enough safety and supply metadata to enter formula optimization.
Search results never promote a reference molecule into a formulation tier.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_trust import EvidenceTrustRoot, sha256_file
from .catalog import normalize_name


INDUSTRIAL_REGISTRY_SCHEMA = "industrial-ingredient-registry-v1.2"
SAFETY_DOSSIER_ARTIFACT_TYPE = "ingredient-safety-dossier/v1"
FORMULATION_TIERS = {
    "prototype_conditional_active",
    "prototype_safe_active",
}


@dataclass(frozen=True)
class IndustrialIngredientRecord:
    registry_id: str
    preferred_name: str
    canonical_smiles: str
    molecular_weight: float | None
    source_count: int
    descriptor_count: int
    formulation_tier: str
    formulation_material_id: str | None
    cas_number: str | None


@dataclass(frozen=True)
class SafetyScreeningRecord:
    registry_id: str
    screening_status: str
    structural_alerts: tuple[str, ...]
    has_cas: bool
    ifra_reference: bool
    source_count: int
    descriptor_count: int
    molecular_weight: float | None
    required_evidence: tuple[str, ...]


@dataclass(frozen=True)
class SafetyPromotionDecision:
    registry_id: str
    current_tier: str
    requested_tier: str
    dossier_complete: bool
    independent_signature_verified: bool
    eligible_tier: str | None
    missing_evidence: tuple[str, ...]
    blocking_alerts: tuple[str, ...]
    decision_reason: str
    artifact_id: str | None = None
    signer_id: str | None = None
    issued_at: str | None = None
    expires_at: str | None = None


class IndustrialIngredientRegistry:
    """Query a hash-audited registry without modifying its bytes."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("industrial ingredient registry must be a file")
        self._connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        integrity = self._connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            self._connection.close()
            raise ValueError("industrial ingredient registry integrity check failed")
        schema = self._connection.execute(
            "SELECT value FROM registry_metadata WHERE key='schema'"
        ).fetchone()
        if schema is None or schema[0] != INDUSTRIAL_REGISTRY_SCHEMA:
            self._connection.close()
            raise ValueError("unsupported industrial ingredient registry schema")
        try:
            self.sha256 = sha256_file(self.path)
        except Exception:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "IndustrialIngredientRegistry":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def stats(self) -> dict[str, int]:
        row = self._connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM ingredients) AS reference_molecules,
                (SELECT COUNT(*) FROM ingredient_sources) AS source_links,
                (SELECT COUNT(*) FROM odor_descriptors) AS descriptor_assertions,
                (SELECT COUNT(*) FROM formulation_materials) AS formulation_materials,
                (SELECT COUNT(*) FROM formulation_materials
                    WHERE formulation_tier = 'prototype_safe_active') AS active_safe,
                (SELECT COUNT(*) FROM formulation_materials
                    WHERE formulation_tier = 'prototype_conditional_active') AS active_conditional,
                (SELECT COUNT(*) FROM formulation_materials
                    WHERE formulation_tier IN
                    ('prototype_safe_active','prototype_conditional_active')) AS prototype_active_total,
                (SELECT COUNT(*) FROM formulation_materials
                    WHERE linked_registry_id IS NOT NULL) AS molecularly_linked_materials,
                (SELECT COUNT(*) FROM safety_screening) AS safety_screened,
                (SELECT COUNT(*) FROM safety_screening
                    WHERE screening_status = 'evidence_pending') AS screening_evidence_pending,
                (SELECT COUNT(*) FROM safety_screening
                    WHERE screening_status = 'structural_review_required') AS structural_review_required,
                (SELECT COUNT(*) FROM safety_screening
                    WHERE has_cas = 1) AS molecules_with_cas,
                (SELECT COUNT(*) FROM safety_screening
                    WHERE ifra_reference = 1) AS ifra_reference_molecules,
                (SELECT COUNT(*) FROM promotion_candidates) AS promotion_candidates_total,
                (SELECT COUNT(*) FROM promotion_candidates
                    WHERE promotion_status = 'evidence_pending') AS promotion_evidence_pending,
                (SELECT COUNT(*) FROM promotion_candidates
                    WHERE promotion_status = 'structural_review_required')
                    AS promotion_structural_review_required,
                (SELECT COUNT(*) FROM promotion_candidates p
                    JOIN ingredients i ON i.registry_id = p.registry_id
                    WHERE i.source_count >= 2 AND i.descriptor_count >= 1
                      AND i.molecular_weight BETWEEN 50.0 AND 350.0
                      AND i.canonical_smiles NOT LIKE '%.%'
                      AND p.ifra_reference = 1) AS high_priority_candidates
            """
        ).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    @staticmethod
    def _record(row: sqlite3.Row) -> IndustrialIngredientRecord:
        return IndustrialIngredientRecord(
            registry_id=str(row["registry_id"]),
            preferred_name=str(row["preferred_name"] or ""),
            canonical_smiles=str(row["canonical_smiles"] or ""),
            molecular_weight=(
                None
                if row["molecular_weight"] is None
                else float(row["molecular_weight"])
            ),
            source_count=int(row["source_count"]),
            descriptor_count=int(row["descriptor_count"]),
            formulation_tier=str(row["formulation_tier"]),
            formulation_material_id=(
                None
                if row["formulation_material_id"] is None
                else str(row["formulation_material_id"])
            ),
            cas_number=(None if row["cas_number"] is None else str(row["cas_number"])),
        )

    def get(self, registry_id: str) -> IndustrialIngredientRecord | None:
        row = self._connection.execute(
            """
            SELECT i.registry_id, i.preferred_name, i.canonical_smiles,
                   i.molecular_weight, i.source_count, i.descriptor_count,
                   COALESCE(f.formulation_tier, 'reference_only') AS formulation_tier,
                   f.ingredient_id AS formulation_material_id,
                   COALESCE(f.cas_number, (
                       SELECT x.identifier_value FROM ingredient_identifiers x
                       WHERE x.registry_id = i.registry_id
                         AND x.identifier_type = 'CAS'
                       ORDER BY x.identifier_value LIMIT 1
                   )) AS cas_number
            FROM ingredients i
            LEFT JOIN formulation_materials f ON f.linked_registry_id = i.registry_id
            WHERE i.registry_id = ?
            ORDER BY CASE f.formulation_tier
                WHEN 'prototype_safe_active' THEN 0
                WHEN 'prototype_conditional_active' THEN 1 ELSE 2 END,
                f.ingredient_id
            LIMIT 1
            """,
            (str(registry_id),),
        ).fetchone()
        return None if row is None else self._record(row)

    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        formulation_only: bool = False,
    ) -> list[IndustrialIngredientRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            raise ValueError(
                "industrial catalog search limit must be between 1 and 500"
            )
        normalized = normalize_name(query)
        if not normalized:
            raise ValueError("industrial catalog search query is required")
        formulation_rows = self._connection.execute(
            """
            SELECT f.ingredient_id, f.name,
                   COALESCE(f.cas_number, (
                       SELECT x.identifier_value FROM ingredient_identifiers x
                       WHERE x.registry_id = f.linked_registry_id
                         AND x.identifier_type = 'CAS'
                       ORDER BY x.identifier_value LIMIT 1
                   )) AS cas_number,
                   f.formulation_tier,
                   f.linked_registry_id, i.preferred_name, i.canonical_smiles,
                   i.molecular_weight, i.source_count, i.descriptor_count
            FROM formulation_materials f
            LEFT JOIN ingredients i ON i.registry_id = f.linked_registry_id
            ORDER BY CASE f.formulation_tier
                WHEN 'prototype_safe_active' THEN 0
                WHEN 'prototype_conditional_active' THEN 1 ELSE 2 END,
                f.ingredient_id
            """
        ).fetchall()
        formulation_hits: list[IndustrialIngredientRecord] = []
        for row in formulation_rows:
            searchable = normalize_name(
                " ".join(
                    str(value or "")
                    for value in (
                        row["ingredient_id"],
                        row["name"],
                        row["cas_number"],
                        row["preferred_name"],
                    )
                )
            )
            if normalized not in searchable:
                continue
            formulation_hits.append(
                IndustrialIngredientRecord(
                    registry_id=str(
                        row["linked_registry_id"] or f"material:{row['ingredient_id']}"
                    ),
                    preferred_name=str(row["name"]),
                    canonical_smiles=str(row["canonical_smiles"] or ""),
                    molecular_weight=(
                        None
                        if row["molecular_weight"] is None
                        else float(row["molecular_weight"])
                    ),
                    source_count=int(row["source_count"] or 0),
                    descriptor_count=int(row["descriptor_count"] or 0),
                    formulation_tier=str(row["formulation_tier"]),
                    formulation_material_id=str(row["ingredient_id"]),
                    cas_number=(
                        None if row["cas_number"] is None else str(row["cas_number"])
                    ),
                )
            )
        if normalized.isdigit() and 5 <= len(normalized) <= 10:
            cas_rows = self._connection.execute(
                """
                SELECT i.registry_id, i.preferred_name, i.canonical_smiles,
                       i.molecular_weight, i.source_count, i.descriptor_count,
                       COALESCE(f.formulation_tier, 'reference_only')
                           AS formulation_tier,
                       f.ingredient_id AS formulation_material_id,
                       x.identifier_value AS cas_number
                FROM ingredient_identifiers x
                JOIN ingredients i ON i.registry_id = x.registry_id
                LEFT JOIN formulation_materials f
                    ON f.linked_registry_id = i.registry_id
                WHERE x.identifier_type = 'CAS'
                  AND replace(x.identifier_value, '-', '') = ?
                ORDER BY CASE COALESCE(f.formulation_tier, 'reference_only')
                    WHEN 'prototype_safe_active' THEN 0
                    WHEN 'prototype_conditional_active' THEN 1
                    WHEN 'formulation_metadata_only' THEN 2
                    ELSE 3 END,
                    i.registry_id
                LIMIT ?
                """,
                (normalized, limit),
            ).fetchall()
            linked_hits = {
                row.registry_id
                for row in formulation_hits
                if row.formulation_material_id is not None
                and not row.registry_id.startswith("material:")
            }
            reference_hits = [
                self._record(row)
                for row in cas_rows
                if str(row["registry_id"]) not in linked_hits
                and (not formulation_only or row["formulation_material_id"] is not None)
            ]
            return [*formulation_hits, *reference_hits][:limit]
        rows = self._connection.execute(
            """
            SELECT i.registry_id, i.preferred_name, i.canonical_smiles,
                   i.molecular_weight, i.source_count, i.descriptor_count,
                   COALESCE(f.formulation_tier, 'reference_only') AS formulation_tier,
                   f.ingredient_id AS formulation_material_id,
                   COALESCE(f.cas_number, (
                       SELECT x.identifier_value FROM ingredient_identifiers x
                       WHERE x.registry_id = i.registry_id
                         AND x.identifier_type = 'CAS'
                       ORDER BY x.identifier_value LIMIT 1
                   )) AS cas_number,
                   MIN(CASE
                       WHEN n.normalized_name = ? THEN 0
                       WHEN n.normalized_name LIKE ? THEN 1
                       WHEN d.normalized_descriptor = ? THEN 2
                       WHEN d.normalized_descriptor LIKE ? THEN 3
                       ELSE 4
                   END) AS match_rank
            FROM ingredients i
            LEFT JOIN ingredient_names n ON n.registry_id = i.registry_id
            LEFT JOIN odor_descriptors d ON d.registry_id = i.registry_id
            LEFT JOIN formulation_materials f ON f.linked_registry_id = i.registry_id
            WHERE (n.normalized_name LIKE ? OR d.normalized_descriptor LIKE ?)
            GROUP BY i.registry_id, f.ingredient_id
            ORDER BY match_rank,
                     CASE COALESCE(f.formulation_tier, 'reference_only')
                         WHEN 'prototype_safe_active' THEN 0
                         WHEN 'prototype_conditional_active' THEN 1
                         WHEN 'formulation_metadata_only' THEN 2
                         ELSE 3
                     END,
                     i.source_count DESC,
                     i.descriptor_count DESC,
                     i.registry_id
            LIMIT ?
            """,
            (
                normalized,
                normalized + "%",
                normalized,
                normalized + "%",
                "%" + normalized + "%",
                "%" + normalized + "%",
                limit,
            ),
        ).fetchall()
        linked_hits = {
            row.registry_id
            for row in formulation_hits
            if row.formulation_material_id is not None
            and not row.registry_id.startswith("material:")
        }
        reference_hits = [
            self._record(row)
            for row in rows
            if str(row["registry_id"]) not in linked_hits
            and (not formulation_only or row["formulation_material_id"] is not None)
        ]
        return [*formulation_hits, *reference_hits][:limit]

    def formulation_materials(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT ingredient_id, name, cas_number, formulation_tier,
                   risk_tier, availability, price_per_kg,
                   max_concentrate_percent, linked_registry_id
            FROM formulation_materials
            ORDER BY CASE formulation_tier
                WHEN 'prototype_safe_active' THEN 0
                WHEN 'prototype_conditional_active' THEN 1 ELSE 2 END,
                ingredient_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def safety_screening(self, registry_id: str) -> SafetyScreeningRecord | None:
        row = self._connection.execute(
            """
            SELECT registry_id, screening_status, structural_alerts_json,
                   has_cas, ifra_reference, source_count, descriptor_count,
                   molecular_weight, required_evidence
            FROM safety_screening WHERE registry_id = ?
            """,
            (str(registry_id),),
        ).fetchone()
        if row is None:
            return None
        alerts = json.loads(str(row["structural_alerts_json"]))
        required = json.loads(str(row["required_evidence"]))
        if not isinstance(alerts, list) or not all(isinstance(x, str) for x in alerts):
            raise ValueError("invalid structural-alert payload in industrial registry")
        if not isinstance(required, list) or not all(
            isinstance(x, str) for x in required
        ):
            raise ValueError(
                "invalid promotion-evidence payload in industrial registry"
            )
        return SafetyScreeningRecord(
            registry_id=str(row["registry_id"]),
            screening_status=str(row["screening_status"]),
            structural_alerts=tuple(alerts),
            has_cas=bool(row["has_cas"]),
            ifra_reference=bool(row["ifra_reference"]),
            source_count=int(row["source_count"]),
            descriptor_count=int(row["descriptor_count"]),
            molecular_weight=(
                None
                if row["molecular_weight"] is None
                else float(row["molecular_weight"])
            ),
            required_evidence=tuple(required),
        )

    def _promotion_context(
        self,
        registry_id: str,
        target_tier: str,
    ) -> tuple[IndustrialIngredientRecord, SafetyScreeningRecord, tuple[str, ...]]:
        if target_tier not in FORMULATION_TIERS:
            raise ValueError("unsupported formulation promotion tier")
        ingredient = self.get(registry_id)
        screening = self.safety_screening(registry_id)
        if ingredient is None or screening is None:
            raise ValueError("unknown industrial registry molecule")
        if ingredient.formulation_tier == "reference_blocked":
            raise ValueError("policy-blocked material cannot be promoted")
        required = list(screening.required_evidence)
        if screening.structural_alerts:
            required.append("structural_review_signoff")
        if target_tier == "prototype_safe_active":
            required.append("low_risk_signoff")
        return ingredient, screening, tuple(required)

    def evaluate_safety_promotion(
        self,
        registry_id: str,
        evidence_labels: Iterable[str],
        *,
        target_tier: str = "prototype_conditional_active",
    ) -> SafetyPromotionDecision:
        """Check dossier completeness without granting formulation permission.

        This is safe for UI/API preflight.  Even a complete label set remains
        ineligible until :meth:`verify_safety_promotion` verifies the real files
        and an independently allowlisted Ed25519 signature.
        """

        ingredient, screening, required = self._promotion_context(
            registry_id, target_tier
        )
        if isinstance(evidence_labels, (str, bytes)):
            raise ValueError("promotion evidence labels must be an iterable of labels")
        supplied = {
            str(label).strip() for label in evidence_labels if str(label).strip()
        }
        missing = tuple(label for label in required if label not in supplied)
        blocking_alerts = (
            screening.structural_alerts
            if screening.structural_alerts and "structural_review_signoff" in missing
            else ()
        )
        reason = (
            "missing_required_evidence" if missing else "independent_signature_required"
        )
        return SafetyPromotionDecision(
            registry_id=registry_id,
            current_tier=ingredient.formulation_tier,
            requested_tier=target_tier,
            dossier_complete=not missing,
            independent_signature_verified=False,
            eligible_tier=None,
            missing_evidence=missing,
            blocking_alerts=blocking_alerts,
            decision_reason=reason,
        )

    def verify_safety_promotion(
        self,
        registry_id: str,
        *,
        target_tier: str,
        market: str,
        product_category: str,
        envelope: Mapping[str, Any],
        artifact_paths: Mapping[str, str | Path],
        trust_root: EvidenceTrustRoot,
        as_of: date | datetime | None = None,
    ) -> SafetyPromotionDecision:
        """Verify a scoped, signed dossier and return a fail-closed tier decision."""

        ingredient, screening, required = self._promotion_context(
            registry_id, target_tier
        )
        market = str(market).strip()
        product_category = str(product_category).strip()
        if not market or not product_category:
            raise ValueError("promotion market and product category are required")
        missing = tuple(label for label in required if label not in artifact_paths)
        if missing:
            return SafetyPromotionDecision(
                registry_id=registry_id,
                current_tier=ingredient.formulation_tier,
                requested_tier=target_tier,
                dossier_complete=False,
                independent_signature_verified=False,
                eligible_tier=None,
                missing_evidence=missing,
                blocking_alerts=(
                    screening.structural_alerts
                    if "structural_review_signoff" in missing
                    else ()
                ),
                decision_reason="missing_required_evidence",
            )
        expected_scope = {
            "registry_schema": INDUSTRIAL_REGISTRY_SCHEMA,
            "registry_sha256": self.sha256,
            "registry_id": registry_id,
            "canonical_smiles": ingredient.canonical_smiles,
            "target_tier": target_tier,
            "market": market,
            "product_category": product_category,
        }
        verified = trust_root.verify(
            envelope,
            artifact_paths,
            expected_artifact_type=SAFETY_DOSSIER_ARTIFACT_TYPE,
            expected_scope=expected_scope,
            allowed_roles={"toxicologist", "regulatory_safety_officer"},
            as_of=as_of,
        )
        return SafetyPromotionDecision(
            registry_id=registry_id,
            current_tier=ingredient.formulation_tier,
            requested_tier=target_tier,
            dossier_complete=True,
            independent_signature_verified=True,
            eligible_tier=target_tier,
            missing_evidence=(),
            blocking_alerts=(),
            decision_reason="signed_dossier_verified",
            artifact_id=verified.artifact_id,
            signer_id=verified.signer_id,
            issued_at=verified.issued_at,
            expires_at=verified.expires_at,
        )

    def promotion_candidates(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("promotion candidate limit must be between 1 and 1000")
        if status is not None and status not in {
            "evidence_pending",
            "structural_review_required",
        }:
            raise ValueError("unsupported promotion candidate status")
        rows = self._connection.execute(
            """
            SELECT p.registry_id, i.preferred_name, i.canonical_smiles,
                   p.evidence_score, p.source_count, p.descriptor_count,
                   p.molecular_weight, p.ifra_reference,
                   p.promotion_status, p.required_evidence
            FROM promotion_candidates p
            JOIN ingredients i ON i.registry_id = p.registry_id
            WHERE (? IS NULL OR p.promotion_status = ?)
            ORDER BY p.evidence_score DESC, p.source_count DESC,
                     p.descriptor_count DESC, p.registry_id
            LIMIT ?
            """,
            (status, status, limit),
        ).fetchall()
        return [dict(row) for row in rows]
