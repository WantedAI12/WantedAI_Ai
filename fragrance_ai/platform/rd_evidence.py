"""Server-pinned review evidence and deterministic material change assessment.

An operator's reviewed records are evidence inputs, never a regulatory certificate.
HTTP callers can select versions but cannot supply or promote review records.
"""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def content_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class EvidenceRecord(Input):
    ingredient_id: str = Field(min_length=1, max_length=300)
    cas_number: str = Field(min_length=1, max_length=100)
    region: Literal["EU", "KR", "US"]
    product_category: str = Field(min_length=1, max_length=80)
    source_reference: str = Field(min_length=1, max_length=1000)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_on: date
    valid_until: date
    rule_version: str = Field(min_length=1, max_length=200)
    frameworks: dict[str, Literal["supported", "restricted", "prohibited", "unknown"]]
    maximum_finished_product_percent: float = Field(strict=True, ge=0, le=100)
    supplier: str = Field(min_length=1, max_length=200)
    sku: str = Field(min_length=1, max_length=200)
    quote_reference: str = Field(min_length=1, max_length=1000)
    quote_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quoted_on: date
    quote_valid_until: date
    currency: Literal["USD"]
    price_per_kg: float = Field(strict=True, gt=0, le=1e7)
    available_kg: float = Field(strict=True, ge=0, le=1e9)
    minimum_order_kg: float = Field(strict=True, ge=0, le=1e9)
    lead_time_days: int = Field(strict=True, ge=0, le=3650)

    @model_validator(mode="after")
    def dates(self):
        if self.valid_until < self.reviewed_on or self.quote_valid_until < self.quoted_on:
            raise ValueError("evidence validity must not end before its source date")
        return self


class Snapshot(Input):
    version: str = Field(min_length=1, max_length=200)
    effective_on: date
    records: list[EvidenceRecord] = Field(max_length=100000)

    @model_validator(mode="after")
    def unique_records(self):
        keys = [(r.ingredient_id, r.region, r.product_category) for r in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate material/region/product evidence")
        return self


class Bundle(Input):
    schema_version: Literal["rd-evidence-1"]
    active_version: str
    snapshots: list[Snapshot] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def versions(self):
        versions = [s.version for s in self.snapshots]
        if len(versions) != len(set(versions)) or self.active_version not in versions:
            raise ValueError("evidence versions must be unique and include active_version")
        return self


class EvidencePolicy(Input):
    finished_batch_mass_g: float = Field(strict=True, gt=0, le=1e12)
    maximum_lead_time_days: int = Field(strict=True, ge=0, le=3650)
    maximum_purchase_cost_usd: float = Field(strict=True, gt=0, le=1e12)


class FormulaLine(Input):
    ingredient_id: str = Field(min_length=1, max_length=300)
    concentrate_percent: float = Field(strict=True, gt=0, le=100)


class EvidenceAssessment(Input):
    lines: list[FormulaLine] = Field(min_length=1, max_length=50000)
    target_region: Literal["EU", "KR", "US"]
    product_category: str = Field(min_length=1, max_length=80)
    product_concentration_percent: float = Field(strict=True, gt=0, le=30)
    max_formula_cost_per_kg: float = Field(strict=True, gt=0, le=1e7)
    max_ingredient_price_per_kg: float = Field(default=300., strict=True, gt=0, le=1e7)
    policy: EvidencePolicy

    @model_validator(mode="after")
    def composition(self):
        if len({r.ingredient_id for r in self.lines}) != len(self.lines):
            raise ValueError("formula material IDs must be unique")
        if abs(sum(r.concentrate_percent for r in self.lines) - 100) > .001 + 1e-9:
            raise ValueError("concentrate percentages must sum to 100")
        return self


class EvidenceStore:
    def __init__(self, path=None, digest=None):
        self.path = Path(path).resolve() if path else None
        self.digest = digest
        self.bundle = None
        if bool(path) != bool(digest):
            raise ValueError("evidence path and SHA256 must be configured together")
        if path:
            raw = self._read()
            self.bundle = Bundle.model_validate_json(raw)

    @classmethod
    def configured(cls):
        return cls(os.environ.get("PERFUMERY_AI_RD_EVIDENCE_PATH"),
                   os.environ.get("PERFUMERY_AI_RD_EVIDENCE_SHA256"))

    def _read(self):
        try:
            if self.path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("evidence bundle exceeds 16 MiB")
            raw = self.path.read_bytes()
        except OSError as error:
            raise ValueError("registered evidence bundle is unavailable") from error
        if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != self.digest:
            raise ValueError("evidence bundle changed or SHA256 mismatch")
        return raw

    def assert_current(self):
        if self.path:
            self._read()

    def contract(self):
        return {"bundle_sha256": self.digest,
                "active_version": self.bundle.active_version if self.bundle else None,
                "source_kind": "operator_reviewed_pinned_records",
                "document_authenticity_independently_verified": False,
                "live_inventory_verified": False}

    def snapshot(self, version, as_of):
        if self.bundle is None:
            return None
        for snapshot in self.bundle.snapshots:
            if snapshot.version == version:
                if snapshot.effective_on > as_of:
                    raise ValueError("future evidence snapshot is not usable")
                return snapshot
        raise ValueError("unknown evidence snapshot version")

    def assess(self, request, catalog, *, version=None, as_of=None):
        self.assert_current()
        as_of = as_of or date.today()
        version = version or (self.bundle.active_version if self.bundle else None)
        snapshot = self.snapshot(version, as_of)
        records = {(r.ingredient_id, r.region, r.product_category): r for r in snapshot.records} if snapshot else {}
        ingredients = {r.ingredient_id: r for r in catalog.ingredients}
        required = {"EU": ("IFRA", "EU_REACH"), "KR": ("IFRA", "K_REACH"), "US": ("IFRA", "FDA")}[request.target_region]
        rows, blockers = [], []
        cost, purchase_cost, all_priced = 0., 0., True
        for line in request.lines:
            row = {"ingredient_id": line.ingredient_id, "blockers": [], "evidence": None}
            material = ingredients.get(line.ingredient_id)
            record = records.get((line.ingredient_id, request.target_region, request.product_category))
            if material is None:
                row["blockers"].append("unknown_material")
            if record is None:
                row["blockers"].append("missing_scoped_evidence")
                all_priced = False
            else:
                row["evidence"] = record.model_dump(mode="json")
                if material is None or not material.cas_number or record.cas_number != material.cas_number:
                    row["blockers"].append("material_identity_mismatch")
                if not record.reviewed_on <= as_of <= record.valid_until:
                    row["blockers"].append("regulatory_evidence_not_current")
                for framework in required:
                    if record.frameworks.get(framework) not in ("supported", "restricted"):
                        row["blockers"].append("framework_not_supported:" + framework)
                finished_percent = line.concentrate_percent * request.product_concentration_percent / 100
                if finished_percent > record.maximum_finished_product_percent + 1e-9:
                    row["blockers"].append("reviewed_use_limit_exceeded")
                if not record.quoted_on <= as_of <= record.quote_valid_until:
                    row["blockers"].append("quote_not_current")
                required_kg = request.policy.finished_batch_mass_g / 1000 * finished_percent / 100
                order_kg = max(required_kg, record.minimum_order_kg)
                if record.available_kg + 1e-12 < order_kg:
                    row["blockers"].append("insufficient_recorded_stock")
                if record.lead_time_days > request.policy.maximum_lead_time_days:
                    row["blockers"].append("lead_time_exceeded")
                if record.price_per_kg > request.max_ingredient_price_per_kg:
                    row["blockers"].append("ingredient_price_limit_exceeded")
                cost += line.concentrate_percent / 100 * record.price_per_kg
                purchase_cost += order_kg * record.price_per_kg
                row.update(required_kg=required_kg, purchase_kg=order_kg,
                           purchase_cost_usd=order_kg * record.price_per_kg,
                           finished_product_percent=finished_percent)
            blockers.extend({"ingredient_id": line.ingredient_id, "reason": reason} for reason in row["blockers"])
            rows.append(row)
        if all_priced:
            if cost > request.max_formula_cost_per_kg + 1e-9:
                blockers.append({"ingredient_id": None, "reason": "reviewed_formula_cost_exceeded"})
            if purchase_cost > request.policy.maximum_purchase_cost_usd + 1e-9:
                blockers.append({"ingredient_id": None, "reason": "purchase_budget_exceeded"})
        result = {"schema_version": "rd-evidence-assessment-1", "snapshot_version": version,
                  "evaluated_on": as_of.isoformat(), "input_id": content_id(request.model_dump(mode="json")),
                  "evidence_contract": self.contract(), "required_frameworks": list(required),
                  "status": "blocked" if blockers else "supported_by_registered_evidence",
                  "gate_passed": not blockers, "blockers": blockers, "materials": rows,
                  "quoted_formula_cost_usd_per_kg": cost if all_priced else None,
                  "purchase_cost_usd": purchase_cost if all_priced else None,
                  "manufacturing_approval": False,
                  "scope": "registered_evidence_screen_requires_existing_safety_and_scientific_gates"}
        result["result_id"] = content_id(result)
        return result

    def change_impact(self, request, catalog, previous_version):
        if self.bundle is None:
            raise ValueError("change impact requires registered evidence snapshots")
        if previous_version == self.bundle.active_version:
            raise ValueError("previous and active evidence versions must differ")
        before_snapshot = self.snapshot(previous_version, date.today())
        after_snapshot = self.snapshot(self.bundle.active_version, date.today())
        if before_snapshot.effective_on > after_snapshot.effective_on:
            raise ValueError("previous evidence snapshot is newer than the active snapshot")
        before = self.assess(request, catalog, version=previous_version)
        after = self.assess(request, catalog)
        changes = []
        for old, new in zip(before["materials"], after["materials"]):
            left, right = old["evidence"] or {}, new["evidence"] or {}
            changed = [{"field": key, "previous": left.get(key), "current": right.get(key)}
                       for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
            if changed or old["blockers"] != new["blockers"]:
                changes.append({"ingredient_id": new["ingredient_id"], "changes": changed,
                                "previous_blockers": old["blockers"], "current_blockers": new["blockers"]})
        result = {"schema_version": "rd-change-impact-1", "before": before, "after": after,
                  "changes": changes, "affected_material_count": len(changes),
                  "review_required": bool(changes) or not after["gate_passed"],
                  "state_changed": False, "scope": "same_formula_and_policy_both_snapshots_evaluated_today"}
        result["result_id"] = content_id(result)
        return result
