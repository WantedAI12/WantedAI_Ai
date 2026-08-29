"""Supplier evidence, price, inventory, and document qualification.

The built-in registry is intentionally empty.  Real supplier quotations and
certificates must be imported by the operator; the code never fabricates them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .catalog import normalize_name
from .models import Ingredient, RecipeConstraints


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "available"}


@dataclass(frozen=True)
class SupplierMaterial:
    ingredient_id: str
    supplier: str
    sku: str
    cas_number: str | None
    price_per_kg: float
    currency: str
    moq_kg: float
    in_stock: bool
    lead_time_days: int
    density_g_ml: float | None
    active_strength_percent: float
    carrier: str | None
    regions: tuple[str, ...]
    ifra_amendment: str | None
    ifra_certificate_valid_until: str | None
    sds_valid_until: str | None
    coa_available: bool
    allergen_statement_valid_until: str | None
    allergen_fractions: dict[str, float]
    lot_number: str | None = None
    source_file: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "SupplierMaterial":
        fractions = row.get("allergen_fractions", {})
        if isinstance(fractions, str):
            fractions = json.loads(fractions) if fractions.strip() else {}
        regions = row.get("regions", ())
        if isinstance(regions, str):
            regions = tuple(part.strip().upper() for part in regions.split("|") if part.strip())
        return cls(
            ingredient_id=str(row.get("ingredient_id", "")).strip(),
            supplier=str(row.get("supplier", "")).strip(),
            sku=str(row.get("sku", "")).strip(),
            cas_number=str(row.get("cas_number", "")).strip() or None,
            price_per_kg=float(row.get("price_per_kg", 0)),
            currency=str(row.get("currency", "")).strip().upper(),
            moq_kg=float(row.get("moq_kg", 0)),
            in_stock=_bool(row.get("in_stock", False)),
            lead_time_days=int(row.get("lead_time_days", 0)),
            density_g_ml=_optional_float(row.get("density_g_ml")),
            active_strength_percent=float(row.get("active_strength_percent", 100)),
            carrier=str(row.get("carrier", "")).strip() or None,
            regions=tuple(regions),
            ifra_amendment=str(row.get("ifra_amendment", "")).strip() or None,
            ifra_certificate_valid_until=str(row.get("ifra_certificate_valid_until", "")).strip() or None,
            sds_valid_until=str(row.get("sds_valid_until", "")).strip() or None,
            coa_available=_bool(row.get("coa_available", False)),
            allergen_statement_valid_until=str(row.get("allergen_statement_valid_until", "")).strip() or None,
            allergen_fractions={str(key): float(value) for key, value in dict(fractions).items()},
            lot_number=str(row.get("lot_number", "")).strip() or None,
            source_file=str(row.get("source_file", "")).strip() or None,
        )

    def validate(self) -> None:
        if not self.ingredient_id or not self.supplier or not self.sku:
            raise ValueError("ingredient_id, supplier, and sku are required")
        if self.price_per_kg <= 0 or not self.currency:
            raise ValueError(f"{self.ingredient_id}: positive price and currency are required")
        if not 0 < self.active_strength_percent <= 100:
            raise ValueError(f"{self.ingredient_id}: active strength must be in (0, 100]")
        if self.density_g_ml is not None and self.density_g_ml <= 0:
            raise ValueError(f"{self.ingredient_id}: density must be positive")
        if any(not 0 <= value <= 1 for value in self.allergen_fractions.values()):
            raise ValueError(f"{self.ingredient_id}: allergen fractions must be between 0 and 1")
        for value in (
            self.ifra_certificate_valid_until,
            self.sds_valid_until,
            self.allergen_statement_valid_until,
        ):
            _parse_date(value)


@dataclass(frozen=True)
class OfferAssessment:
    qualified: bool
    reasons: tuple[str, ...]
    missing_documents: tuple[str, ...]
    offer: SupplierMaterial | None


class SupplierRegistry:
    """Validated, operator-provided supplier offers and regulatory evidence."""

    def __init__(self, records: Iterable[SupplierMaterial] = (), metadata: dict | None = None):
        self.records = list(records)
        self.metadata = metadata or {}
        self._by_id: dict[str, list[SupplierMaterial]] = {}
        self._by_cas: dict[str, list[SupplierMaterial]] = {}
        for record in self.records:
            record.validate()
            self._by_id.setdefault(normalize_name(record.ingredient_id), []).append(record)
            if record.cas_number:
                self._by_cas.setdefault(normalize_name(record.cas_number), []).append(record)

    @classmethod
    def load_builtin(cls) -> "SupplierRegistry":
        path = Path(__file__).resolve().parent.parent / "data" / "supplier_registry.json"
        if not path.exists():
            return cls(metadata={"registry_status": "no_operator_data"})
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            (SupplierMaterial.from_mapping(item) for item in payload.get("records", [])),
            payload.get("metadata", {}),
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "SupplierRegistry":
        source = Path(path)
        records: list[SupplierMaterial] = []
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["source_file"] = str(source)
                records.append(SupplierMaterial.from_mapping(row))
        return cls(records, {"source": str(source), "imported_records": len(records)})

    def to_json(self, path: str | Path) -> None:
        payload = {"metadata": self.metadata, "records": [asdict(record) for record in self.records]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def offers_for(self, ingredient: Ingredient) -> list[SupplierMaterial]:
        records = list(self._by_id.get(normalize_name(ingredient.ingredient_id), []))
        if ingredient.cas_number:
            records.extend(self._by_cas.get(normalize_name(ingredient.cas_number), []))
        unique: dict[tuple[str, str], SupplierMaterial] = {
            (record.supplier, record.sku): record for record in records
        }
        return list(unique.values())

    def assess_offer(
        self,
        offer: SupplierMaterial,
        constraints: RecipeConstraints,
        as_of: date,
        active_ifra_amendment: str = "51",
    ) -> OfferAssessment:
        reasons: list[str] = []
        missing: list[str] = []
        region = constraints.target_region.upper()
        if region not in offer.regions and "GLOBAL" not in offer.regions:
            reasons.append(f"target region {region} is not covered")
        if not offer.in_stock:
            reasons.append("not in stock")
        if offer.lead_time_days > constraints.max_supplier_lead_time_days:
            reasons.append("lead time exceeds limit")
        if offer.moq_kg > constraints.max_supplier_moq_kg:
            reasons.append("MOQ exceeds limit")
        if offer.price_per_kg > constraints.max_ingredient_price_per_kg:
            reasons.append("supplier price exceeds limit")

        if offer.ifra_amendment != active_ifra_amendment:
            missing.append(f"IFRA certificate for amendment {active_ifra_amendment}")
        if (_parse_date(offer.ifra_certificate_valid_until) or date.min) < as_of:
            missing.append("current IFRA certificate")
        if (_parse_date(offer.sds_valid_until) or date.min) < as_of:
            missing.append("current SDS")
        if not offer.coa_available:
            missing.append("lot COA")
        if (_parse_date(offer.allergen_statement_valid_until) or date.min) < as_of:
            missing.append("current quantitative allergen statement")
        if not offer.allergen_fractions:
            missing.append("quantitative allergen composition")
        if offer.density_g_ml is None:
            missing.append("density specification")

        return OfferAssessment(not reasons and not missing, tuple(reasons), tuple(missing), offer)

    def best_assessment(
        self,
        ingredient: Ingredient,
        constraints: RecipeConstraints,
        as_of: date,
        active_ifra_amendment: str = "51",
    ) -> OfferAssessment:
        offers = self.offers_for(ingredient)
        if not offers:
            return OfferAssessment(
                False,
                ("no supplier offer",),
                ("supplier quotation and regulatory document pack",),
                None,
            )
        assessments = [
            self.assess_offer(offer, constraints, as_of, active_ifra_amendment)
            for offer in offers
        ]
        qualified = [item for item in assessments if item.qualified]
        pool = qualified or assessments
        return min(pool, key=lambda item: item.offer.price_per_kg if item.offer else float("inf"))

    @staticmethod
    def overlay(ingredient: Ingredient, offer: SupplierMaterial) -> Ingredient:
        return replace(
            ingredient,
            price_per_kg=offer.price_per_kg,
            availability=1.0 if offer.in_stock else 0.0,
            currency=offer.currency,
            density_g_ml=offer.density_g_ml,
            active_strength_percent=offer.active_strength_percent,
            carrier=offer.carrier,
            data_source=f"supplier:{offer.supplier}:{offer.sku}",
            data_verified_on=date.today().isoformat(),
        )

    def stats(self) -> dict[str, int | str]:
        return {
            "supplier_records": len(self.records),
            "supplier_count": len({record.supplier for record in self.records}),
            "supplier_registry_status": str(self.metadata.get("registry_status", "operator_provided")),
        }
