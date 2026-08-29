"""Canonical, file-verifiable scope for a commercial fragrance release.

An external approval is only meaningful when it is tied to *all* of the
things that can change the finished product: the as-supplied formula,
supplier lots and source documents, product use, and the rule/data/model
versions used to make the decision.  This module is deliberately independent
from the approval database so the same canonical scope is used at signing and
at release time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import RecipeConstraints, RecipeLine
from .supplier import SupplierRegistry


RELEASE_SPEC_SCHEMA = "perfumery-commercial-release-spec/v1"
REQUIRED_SUPPLIER_DOCUMENTS = (
    "coa",
    "sds",
    "ifra_certificate",
    "allergen_statement",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Return the only serialization permitted for scope and signature IDs."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"document is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"release scope requires {field}")
    return text


def _document_payload(document: Mapping[str, Any], label: str) -> dict[str, str]:
    """Resolve a document and verify a supplied expected digest, if present."""
    path = _clean_text(document.get("path"), f"{label}.path")
    calculated = sha256_file(path)
    expected = str(document.get("sha256", "")).strip().lower()
    if expected and (not _SHA256.fullmatch(expected) or expected != calculated):
        raise ValueError(f"{label}: supplied sha256 does not match document bytes")
    # The canonical scope intentionally contains only the digest.  The path is
    # retained outside the signed ID by the caller and is never trusted as
    # evidence by itself.
    return {"sha256": calculated}


@dataclass(frozen=True)
class ReleaseSpec:
    """Validated canonical release scope and its SHA-256 identifier."""

    payload: dict[str, Any]
    release_spec_id: str
    # Paths never participate in the ID or signature.  They are retained only
    # so the system can re-hash the exact source documents immediately before
    # accepting an approval and immediately before allowing manufacture.
    document_paths: tuple[tuple[str, str], ...] = ()

    @classmethod
    def build(
        cls,
        lines: Sequence[RecipeLine],
        constraints: RecipeConstraints,
        supplier_registry: SupplierRegistry,
        *,
        rule_pack_version: str,
        data_version: str,
        model_version: str,
        as_of: date | None = None,
    ) -> "ReleaseSpec":
        """Build a scope only from current supplier lots and real documents.

        ``constraints.commercial_supplier_evidence`` is keyed by ingredient ID.
        Every entry must name the same supplier/SKU/lot selected by the
        qualified registry offer and provide real COA, SDS, IFRA and allergen
        document paths.  Bare hashes are deliberately rejected.
        """
        if not lines:
            raise ValueError("release scope requires at least one formula line")
        as_of = as_of or date.today()
        market = _clean_text(constraints.target_region, "market_region").upper()
        product = {
            "concentration_percent": round(float(constraints.product_concentration_percent), 6),
            "category": _clean_text(constraints.product_category, "product_category"),
            "market_region": market,
            "base_id": _clean_text(
                constraints.commercial_product_base_id, "commercial_product_base_id"
            ),
            "packaging_id": _clean_text(
                constraints.commercial_packaging_id, "commercial_packaging_id"
            ),
        }
        if not 0 < product["concentration_percent"] <= 100:
            raise ValueError("product concentration must be in (0, 100]")

        scope_lines = []
        evidence = constraints.commercial_supplier_evidence
        supplier_materials = []
        document_paths: list[tuple[str, str]] = []
        seen: set[str] = set()
        for line in sorted(lines, key=lambda item: item.ingredient_id):
            ingredient_id = _clean_text(line.ingredient_id, "ingredient_id")
            if ingredient_id in seen:
                raise ValueError(f"release scope has duplicate ingredient: {ingredient_id}")
            seen.add(ingredient_id)
            supplied = evidence.get(ingredient_id)
            if not isinstance(supplied, Mapping):
                raise ValueError(f"release scope missing supplier evidence for {ingredient_id}")
            offer_candidates = [
                offer for offer in supplier_registry.records
                if offer.ingredient_id == ingredient_id
                and offer.supplier == str(supplied.get("supplier", "")).strip()
                and offer.sku == str(supplied.get("sku", "")).strip()
                and offer.lot_number == str(supplied.get("lot_number", "")).strip()
            ]
            if len(offer_candidates) != 1:
                raise ValueError(
                    f"{ingredient_id}: supplier/SKU/lot is not an exact registered offer"
                )
            offer = offer_candidates[0]
            if not offer.lot_number:
                raise ValueError(f"{ingredient_id}: supplier offer has no lot number")
            offer_assessment = supplier_registry.assess_offer(
                offer,
                constraints,
                as_of,
                active_ifra_amendment="51",
            )
            if not offer_assessment.qualified:
                details = "; ".join(
                    (*offer_assessment.reasons, *offer_assessment.missing_documents)
                )
                raise ValueError(
                    f"{ingredient_id}: supplier lot is not currently qualified: {details}"
                )
            if abs(
                float(line.active_strength_percent)
                - float(offer.active_strength_percent)
            ) > 1e-6:
                raise ValueError(
                    f"{ingredient_id}: formula active strength does not match supplier lot"
                )
            if str(line.carrier or "") != str(offer.carrier or ""):
                raise ValueError(f"{ingredient_id}: formula carrier does not match supplier lot")
            expected_finished = (
                float(line.concentrate_percent) * product["concentration_percent"] / 100.0
            )
            if abs(float(line.finished_product_percent) - expected_finished) > 1e-6:
                raise ValueError(
                    f"{ingredient_id}: finished-product percentage does not match product concentration"
                )
            documents = supplied.get("documents")
            if not isinstance(documents, Mapping):
                raise ValueError(f"{ingredient_id}: release scope requires supplier documents")
            document_hashes = {
                name: _document_payload(
                    documents.get(name, {}), f"{ingredient_id}.{name}"
                )
                for name in REQUIRED_SUPPLIER_DOCUMENTS
            }
            document_paths.extend(
                (
                    f"{ingredient_id}.{name}",
                    str(Path(str(documents[name]["path"])).expanduser().resolve(strict=True)),
                )
                for name in REQUIRED_SUPPLIER_DOCUMENTS
            )
            scope_lines.append(
                {
                    "ingredient_id": ingredient_id,
                    "concentrate_percent": round(float(line.concentrate_percent), 6),
                    "finished_product_percent": round(float(line.finished_product_percent), 6),
                    "active_strength_percent": round(float(line.active_strength_percent), 6),
                    "carrier": str(line.carrier or ""),
                }
            )
            supplier_materials.append(
                {
                    "ingredient_id": ingredient_id,
                    "supplier": offer.supplier,
                    "sku": offer.sku,
                    "lot_number": offer.lot_number,
                    "active_strength_percent": round(float(offer.active_strength_percent), 6),
                    "carrier": str(offer.carrier or ""),
                    "document_hashes": document_hashes,
                }
            )

        total = sum(item["concentrate_percent"] for item in scope_lines)
        if abs(total - 100.0) > 0.05:
            raise ValueError("release scope formula must sum to 100%")
        payload = {
            "schema": RELEASE_SPEC_SCHEMA,
            "formula": {"lines": scope_lines},
            "finished_product": product,
            "supplier_materials": supplier_materials,
            "versions": {
                "rule_pack": _clean_text(rule_pack_version, "rule_pack_version"),
                "data": _clean_text(data_version, "data_version"),
                "model": _clean_text(model_version, "model_version"),
            },
        }
        verified = cls.from_payload(payload)
        return cls(verified.payload, verified.release_spec_id, tuple(document_paths))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReleaseSpec":
        """Create an ID from an already validated canonical payload.

        This parser intentionally does not accept unverified document paths;
        operational code must use :meth:`build` to create a signable scope.
        It is useful for auditing a stored immutable payload.
        """
        normalized = json.loads(canonical_json(dict(payload)).decode("utf-8"))
        if normalized.get("schema") != RELEASE_SPEC_SCHEMA:
            raise ValueError("unsupported release scope schema")
        required = {"formula", "finished_product", "supplier_materials", "versions"}
        if not required <= set(normalized):
            raise ValueError("release scope is incomplete")
        if set(normalized) != {"schema", *required}:
            raise ValueError("release scope contains unsupported top-level fields")
        formula = normalized["formula"]
        product = normalized["finished_product"]
        materials = normalized["supplier_materials"]
        versions = normalized["versions"]
        if not isinstance(formula, dict) or not isinstance(formula.get("lines"), list):
            raise ValueError("release scope formula lines are invalid")
        if not isinstance(product, dict) or set(product) != {
            "concentration_percent",
            "category",
            "market_region",
            "base_id",
            "packaging_id",
        }:
            raise ValueError("finished-product scope is malformed")
        if not isinstance(materials, list) or not isinstance(versions, dict):
            raise ValueError("release scope sections are invalid")
        line_ids: set[str] = set()
        total = 0.0
        expected_line_fields = {
            "ingredient_id",
            "concentrate_percent",
            "finished_product_percent",
            "active_strength_percent",
            "carrier",
        }
        for line in formula["lines"]:
            if not isinstance(line, dict) or set(line) != expected_line_fields:
                raise ValueError("release scope formula line is invalid")
            ingredient = _clean_text(line.get("ingredient_id"), "formula ingredient_id")
            if ingredient in line_ids:
                raise ValueError("release scope has duplicate formula ingredient")
            line_ids.add(ingredient)
            for name in (
                "concentrate_percent",
                "finished_product_percent",
                "active_strength_percent",
            ):
                try:
                    value = float(line[name])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"release scope requires numeric formula {name}") from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"release scope formula {name} cannot be negative")
                if name == "concentrate_percent" and value <= 0:
                    raise ValueError("release scope formula percentages must be positive")
                if name == "active_strength_percent" and not 0 < value <= 100:
                    raise ValueError("release scope active strength is invalid")
                if name == "concentrate_percent":
                    total += value
        if not line_ids or abs(total - 100.0) > 0.05:
            raise ValueError("release scope formula must be nonempty and sum to 100%")
        for name in ("category", "market_region", "base_id", "packaging_id"):
            _clean_text(product.get(name), f"finished_product.{name}")
        try:
            concentration = float(product["concentration_percent"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("release scope requires product concentration") from error
        if not math.isfinite(concentration) or not 0 < concentration <= 100:
            raise ValueError("release scope product concentration is invalid")
        for line in formula["lines"]:
            expected_finished = (
                float(line["concentrate_percent"]) * concentration / 100.0
            )
            if abs(float(line["finished_product_percent"]) - expected_finished) > 1e-6:
                raise ValueError(
                    "release scope finished-product percentage is inconsistent"
                )
        material_ids: set[str] = set()
        expected_material_fields = {
            "ingredient_id",
            "supplier",
            "sku",
            "lot_number",
            "active_strength_percent",
            "carrier",
            "document_hashes",
        }
        formula_by_id = {
            str(line["ingredient_id"]): line for line in formula["lines"]
        }
        for material in materials:
            if not isinstance(material, dict) or set(material) != expected_material_fields:
                raise ValueError("release scope supplier material is invalid")
            ingredient = _clean_text(material.get("ingredient_id"), "supplier ingredient_id")
            if ingredient in material_ids:
                raise ValueError("release scope has duplicate supplier material")
            material_ids.add(ingredient)
            for name in ("supplier", "sku", "lot_number"):
                _clean_text(material.get(name), f"supplier_material.{name}")
            hashes = material.get("document_hashes")
            if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_SUPPLIER_DOCUMENTS):
                raise ValueError("release scope has incomplete supplier document hashes")
            for name, item in hashes.items():
                if (
                    not isinstance(item, dict)
                    or set(item) != {"sha256"}
                    or not _SHA256.fullmatch(str(item.get("sha256", "")))
                ):
                    raise ValueError(f"release scope has invalid {name} document hash")
            formula_line = formula_by_id.get(ingredient)
            strength = float(material["active_strength_percent"])
            if (
                formula_line is None
                or not math.isfinite(strength)
                or abs(strength - float(formula_line["active_strength_percent"])) > 1e-6
            ):
                raise ValueError("supplier strength does not match formula")
            if str(material.get("carrier") or "") != str(
                formula_line.get("carrier") or ""
            ):
                raise ValueError("supplier carrier does not match formula")
        if material_ids != line_ids or len(materials) != len(line_ids):
            raise ValueError("release scope supplier materials do not match formula")
        if set(versions) != {"rule_pack", "data", "model"}:
            raise ValueError("release scope artifact versions are malformed")
        for name in ("rule_pack", "data", "model"):
            _clean_text(versions.get(name), f"versions.{name}")
        digest = hashlib.sha256(canonical_json(normalized)).hexdigest()
        return cls(normalized, "sha256:" + digest)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.payload)

    def verify_bound_documents(self) -> None:
        """Re-hash every bound supplier document against the signed scope."""
        expected = {
            f"{material['ingredient_id']}.{name}": value["sha256"]
            for material in self.payload["supplier_materials"]
            for name, value in material["document_hashes"].items()
        }
        provided = dict(self.document_paths)
        if set(provided) != set(expected):
            raise ValueError("release scope has no complete verified supplier document paths")
        for label, expected_sha in expected.items():
            actual = sha256_file(provided[label])
            if actual != expected_sha:
                raise ValueError(f"{label}: document bytes changed after scope creation")
