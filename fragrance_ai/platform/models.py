"""Immutable records shared by workspace-store implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectRecord:
    tenant_id: str
    project_id: str
    name: str
    description: str
    created_by: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormulaVersionRecord:
    tenant_id: str
    project_id: str
    formula_id: str
    version_id: str
    version_number: int
    parent_version_id: str | None
    change_kind: str
    change_note: str
    created_by: str
    created_at: str
    payload: dict[str, Any]
    content_sha256: str

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_payload:
            result.pop("payload")
        return result


@dataclass(frozen=True)
class FormulaRecord:
    tenant_id: str
    project_id: str
    formula_id: str
    name: str
    kind: str
    created_by: str
    created_at: str
    updated_at: str
    latest_version: FormulaVersionRecord

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["latest_version"] = self.latest_version.to_dict(
            include_payload=include_payload
        )
        return result


@dataclass(frozen=True)
class JobRecord:
    tenant_id: str
    job_id: str
    kind: str
    status: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    created_by: str
    created_at: str
    updated_at: str
    available_at: str
    lease_owner: str | None
    lease_expires_at: str | None
    attempts: int
    max_attempts: int

    @property
    def terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
