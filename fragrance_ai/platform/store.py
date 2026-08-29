"""Tenant-scoped workspace storage and durable inference queue.

SQLite is the single-node development backend. PostgreSQL is selected lazily
for horizontally scaled deployments so the core package retains a NumPy-only
installation path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from .models import FormulaRecord, FormulaVersionRecord, JobRecord, ProjectRecord


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_JOB_KINDS = frozenset({"recipe.generate", "formula.revise", "accord.generate"})
_FORMULA_KINDS = frozenset({"formula", "accord"})
_CHANGE_KINDS = frozenset({"generated", "natural_language_revision", "manual_edit"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a valid identifier")
    return value


def _bounded_text(value: str, field: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    cleaned = value.strip()
    if (not empty and not cleaned) or len(cleaned) > maximum:
        qualifier = "possibly empty " if empty else "non-empty "
        raise ValueError(f"{field} must be {qualifier}text of at most {maximum} characters")
    return cleaned


class WorkspaceStore(Protocol):
    backend: str
    horizontally_scalable: bool

    def ping(self) -> bool: ...

    def close(self) -> None: ...

    def create_project(
        self, *, tenant_id: str, name: str, description: str, actor_id: str
    ) -> ProjectRecord: ...

    def list_projects(self, *, tenant_id: str, limit: int = 100) -> list[ProjectRecord]: ...

    def get_project(self, *, tenant_id: str, project_id: str) -> ProjectRecord | None: ...

    def list_formulas(
        self, *, tenant_id: str, project_id: str, limit: int = 100
    ) -> list[FormulaRecord]: ...

    def create_formula(
        self,
        *,
        tenant_id: str,
        project_id: str,
        name: str,
        kind: str,
        payload: dict[str, Any],
        actor_id: str,
        change_note: str,
        source_job_id: str | None = None,
    ) -> FormulaRecord: ...

    def append_formula_version(
        self,
        *,
        tenant_id: str,
        project_id: str,
        formula_id: str,
        expected_parent_version_id: str,
        change_kind: str,
        change_note: str,
        payload: dict[str, Any],
        actor_id: str,
        source_job_id: str | None = None,
    ) -> FormulaVersionRecord: ...

    def get_formula(
        self, *, tenant_id: str, project_id: str, formula_id: str
    ) -> FormulaRecord | None: ...

    def list_formula_versions(
        self, *, tenant_id: str, project_id: str, formula_id: str, limit: int = 100
    ) -> list[FormulaVersionRecord]: ...

    def get_formula_version(
        self,
        *,
        tenant_id: str,
        project_id: str,
        formula_id: str,
        version_id: str,
    ) -> FormulaVersionRecord | None: ...

    def enqueue_job(
        self,
        *,
        tenant_id: str,
        kind: str,
        payload: dict[str, Any],
        actor_id: str,
        max_attempts: int = 3,
    ) -> JobRecord: ...

    def get_job(self, *, tenant_id: str, job_id: str) -> JobRecord | None: ...

    def list_jobs(self, *, tenant_id: str, limit: int = 100) -> list[JobRecord]: ...

    def claim_job(self, *, worker_id: str, lease_seconds: int = 300) -> JobRecord | None: ...

    def renew_job_lease(
        self, *, job_id: str, worker_id: str, lease_seconds: int = 300
    ) -> bool: ...

    def complete_job(
        self, *, job_id: str, worker_id: str, result: dict[str, Any]
    ) -> JobRecord: ...

    def fail_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        retry_delay_seconds: int = 5,
    ) -> JobRecord: ...


class SqliteWorkspaceStore:
    """Durable single-node store with tenant predicates on every read/write."""

    backend = "sqlite"
    horizontally_scalable = False

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, project_id)
                );
                CREATE TABLE IF NOT EXISTS formulas (
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    formula_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, formula_id),
                    FOREIGN KEY (tenant_id, project_id)
                        REFERENCES projects(tenant_id, project_id)
                );
                CREATE INDEX IF NOT EXISTS ix_formulas_project
                    ON formulas(tenant_id, project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS formula_versions (
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    formula_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    parent_version_id TEXT,
                    change_kind TEXT NOT NULL,
                    change_note TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, version_id),
                    UNIQUE (tenant_id, formula_id, version_number),
                    FOREIGN KEY (tenant_id, formula_id)
                        REFERENCES formulas(tenant_id, formula_id)
                );
                CREATE INDEX IF NOT EXISTS ix_formula_versions_formula
                    ON formula_versions(tenant_id, formula_id, version_number DESC);
                CREATE TABLE IF NOT EXISTS inference_jobs (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, job_id),
                    UNIQUE (job_id)
                );
                CREATE INDEX IF NOT EXISTS ix_jobs_claim
                    ON inference_jobs(status, available_at, created_at);
                CREATE TABLE IF NOT EXISTS job_effects (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    effect_type TEXT NOT NULL,
                    formula_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, job_id),
                    FOREIGN KEY (tenant_id, job_id)
                        REFERENCES inference_jobs(tenant_id, job_id),
                    FOREIGN KEY (tenant_id, formula_id)
                        REFERENCES formulas(tenant_id, formula_id),
                    FOREIGN KEY (tenant_id, version_id)
                        REFERENCES formula_versions(tenant_id, version_id)
                );
                """
            )

    @staticmethod
    def _project(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(**dict(row))

    @staticmethod
    def _version(row: sqlite3.Row) -> FormulaVersionRecord:
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        return FormulaVersionRecord(**value)

    @staticmethod
    def _job(row: sqlite3.Row) -> JobRecord:
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        raw_result = value.pop("result_json")
        value["result"] = json.loads(raw_result) if raw_result is not None else None
        return JobRecord(**value)

    def _formula_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> FormulaRecord:
        latest = connection.execute(
            """
            SELECT * FROM formula_versions
            WHERE tenant_id=? AND formula_id=?
            ORDER BY version_number DESC LIMIT 1
            """,
            (row["tenant_id"], row["formula_id"]),
        ).fetchone()
        if latest is None:
            raise RuntimeError("formula exists without a version")
        return FormulaRecord(**dict(row), latest_version=self._version(latest))

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def close(self) -> None:
        # Connections are operation-scoped so there is no shared handle.
        return None

    def create_project(
        self, *, tenant_id: str, name: str, description: str, actor_id: str
    ) -> ProjectRecord:
        tenant = _identifier(tenant_id, "tenant_id")
        actor = _identifier(actor_id, "actor_id")
        project_id = "prj_" + uuid.uuid4().hex
        now = utc_now()
        record = ProjectRecord(
            tenant,
            project_id,
            _bounded_text(name, "name", 160),
            _bounded_text(description, "description", 4000, empty=True),
            actor,
            now,
            now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)",
                tuple(record.__dict__.values()),
            )
            connection.commit()
        return record

    def list_projects(self, *, tenant_id: str, limit: int = 100) -> list[ProjectRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        bounded = min(max(int(limit), 1), 500)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects WHERE tenant_id=? ORDER BY updated_at DESC LIMIT ?",
                (tenant, bounded),
            ).fetchall()
        return [self._project(row) for row in rows]

    def get_project(self, *, tenant_id: str, project_id: str) -> ProjectRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE tenant_id=? AND project_id=?",
                (tenant, project),
            ).fetchone()
        return self._project(row) if row else None

    def list_formulas(
        self, *, tenant_id: str, project_id: str, limit: int = 100
    ) -> list[FormulaRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        bounded = min(max(int(limit), 1), 500)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM formulas
                WHERE tenant_id=? AND project_id=?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (tenant, project, bounded),
            ).fetchall()
            return [self._formula_from_row(connection, row) for row in rows]

    def create_formula(
        self,
        *,
        tenant_id: str,
        project_id: str,
        name: str,
        kind: str,
        payload: dict[str, Any],
        actor_id: str,
        change_note: str,
        source_job_id: str | None = None,
    ) -> FormulaRecord:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        actor = _identifier(actor_id, "actor_id")
        source_job = (
            _identifier(source_job_id, "source_job_id")
            if source_job_id is not None
            else None
        )
        if kind not in _FORMULA_KINDS:
            raise ValueError("unsupported formula kind")
        if not isinstance(payload, dict):
            raise ValueError("formula payload must be an object")
        formula_id = "frm_" + uuid.uuid4().hex
        version_id = "ver_" + uuid.uuid4().hex
        now = utc_now()
        payload_json = _canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        formula_values = (
            tenant,
            project,
            formula_id,
            _bounded_text(name, "name", 160),
            kind,
            actor,
            now,
            now,
        )
        version_values = (
            tenant,
            project,
            formula_id,
            version_id,
            1,
            None,
            "generated",
            _bounded_text(change_note, "change_note", 2000, empty=True),
            actor,
            now,
            payload_json,
            digest,
        )
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                exists = connection.execute(
                    "SELECT 1 FROM projects WHERE tenant_id=? AND project_id=?",
                    (tenant, project),
                ).fetchone()
                if not exists:
                    raise KeyError("project not found")
                if source_job is not None:
                    effect = connection.execute(
                        """
                        SELECT * FROM job_effects
                        WHERE tenant_id=? AND job_id=?
                        """,
                        (tenant, source_job),
                    ).fetchone()
                    if effect is not None:
                        if effect["effect_type"] != "formula.create":
                            raise RuntimeError("job effect kind conflict")
                        existing = connection.execute(
                            """
                            SELECT * FROM formulas
                            WHERE tenant_id=? AND project_id=? AND formula_id=?
                            """,
                            (tenant, project, effect["formula_id"]),
                        ).fetchone()
                        if existing is None:
                            raise RuntimeError("job effect references a missing formula")
                        connection.commit()
                        return self._formula_from_row(connection, existing)
                connection.execute(
                    "INSERT INTO formulas VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    formula_values,
                )
                connection.execute(
                    "INSERT INTO formula_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    version_values,
                )
                if source_job is not None:
                    connection.execute(
                        """
                        INSERT INTO job_effects
                        (tenant_id, job_id, effect_type, formula_id, version_id, created_at)
                        VALUES (?, ?, 'formula.create', ?, ?, ?)
                        """,
                        (tenant, source_job, formula_id, version_id, now),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        version = FormulaVersionRecord(
            tenant,
            project,
            formula_id,
            version_id,
            1,
            None,
            "generated",
            version_values[7],
            actor,
            now,
            json.loads(payload_json),
            digest,
        )
        return FormulaRecord(
            tenant,
            project,
            formula_id,
            formula_values[3],
            kind,
            actor,
            now,
            now,
            version,
        )

    def append_formula_version(
        self,
        *,
        tenant_id: str,
        project_id: str,
        formula_id: str,
        expected_parent_version_id: str,
        change_kind: str,
        change_note: str,
        payload: dict[str, Any],
        actor_id: str,
        source_job_id: str | None = None,
    ) -> FormulaVersionRecord:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        expected = _identifier(expected_parent_version_id, "expected_parent_version_id")
        actor = _identifier(actor_id, "actor_id")
        source_job = (
            _identifier(source_job_id, "source_job_id")
            if source_job_id is not None
            else None
        )
        if change_kind not in _CHANGE_KINDS:
            raise ValueError("unsupported change kind")
        if not isinstance(payload, dict):
            raise ValueError("formula payload must be an object")
        version_id = "ver_" + uuid.uuid4().hex
        now = utc_now()
        payload_json = _canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if source_job is not None:
                    effect = connection.execute(
                        """
                        SELECT * FROM job_effects
                        WHERE tenant_id=? AND job_id=?
                        """,
                        (tenant, source_job),
                    ).fetchone()
                    if effect is not None:
                        if (
                            effect["effect_type"] != "formula.version"
                            or effect["formula_id"] != formula
                        ):
                            raise RuntimeError("job effect kind conflict")
                        existing = connection.execute(
                            """
                            SELECT * FROM formula_versions
                            WHERE tenant_id=? AND formula_id=? AND version_id=?
                            """,
                            (tenant, formula, effect["version_id"]),
                        ).fetchone()
                        if existing is None:
                            raise RuntimeError("job effect references a missing version")
                        connection.commit()
                        return self._version(existing)
                latest = connection.execute(
                    """
                    SELECT version_id, version_number FROM formula_versions
                    WHERE tenant_id=? AND project_id=? AND formula_id=?
                    ORDER BY version_number DESC LIMIT 1
                    """,
                    (tenant, project, formula),
                ).fetchone()
                if latest is None:
                    raise KeyError("formula not found")
                if latest["version_id"] != expected:
                    raise RuntimeError("formula version conflict")
                version_number = int(latest["version_number"]) + 1
                note = _bounded_text(change_note, "change_note", 2000, empty=True)
                connection.execute(
                    """
                    INSERT INTO formula_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant,
                        project,
                        formula,
                        version_id,
                        version_number,
                        expected,
                        change_kind,
                        note,
                        actor,
                        now,
                        payload_json,
                        digest,
                    ),
                )
                connection.execute(
                    """
                    UPDATE formulas SET updated_at=?
                    WHERE tenant_id=? AND project_id=? AND formula_id=?
                    """,
                    (now, tenant, project, formula),
                )
                connection.execute(
                    "UPDATE projects SET updated_at=? WHERE tenant_id=? AND project_id=?",
                    (now, tenant, project),
                )
                if source_job is not None:
                    connection.execute(
                        """
                        INSERT INTO job_effects
                        (tenant_id, job_id, effect_type, formula_id, version_id, created_at)
                        VALUES (?, ?, 'formula.version', ?, ?, ?)
                        """,
                        (tenant, source_job, formula, version_id, now),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return FormulaVersionRecord(
            tenant,
            project,
            formula,
            version_id,
            version_number,
            expected,
            change_kind,
            note,
            actor,
            now,
            json.loads(payload_json),
            digest,
        )

    def get_formula(
        self, *, tenant_id: str, project_id: str, formula_id: str
    ) -> FormulaRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM formulas
                WHERE tenant_id=? AND project_id=? AND formula_id=?
                """,
                (tenant, project, formula),
            ).fetchone()
            return self._formula_from_row(connection, row) if row else None

    def list_formula_versions(
        self, *, tenant_id: str, project_id: str, formula_id: str, limit: int = 100
    ) -> list[FormulaVersionRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        bounded = min(max(int(limit), 1), 500)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM formula_versions
                WHERE tenant_id=? AND project_id=? AND formula_id=?
                ORDER BY version_number DESC LIMIT ?
                """,
                (tenant, project, formula, bounded),
            ).fetchall()
        return [self._version(row) for row in rows]

    def get_formula_version(
        self,
        *,
        tenant_id: str,
        project_id: str,
        formula_id: str,
        version_id: str,
    ) -> FormulaVersionRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        version = _identifier(version_id, "version_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM formula_versions
                WHERE tenant_id=? AND project_id=? AND formula_id=? AND version_id=?
                """,
                (tenant, project, formula, version),
            ).fetchone()
        return self._version(row) if row else None

    def enqueue_job(
        self,
        *,
        tenant_id: str,
        kind: str,
        payload: dict[str, Any],
        actor_id: str,
        max_attempts: int = 3,
    ) -> JobRecord:
        tenant = _identifier(tenant_id, "tenant_id")
        actor = _identifier(actor_id, "actor_id")
        if kind not in _JOB_KINDS:
            raise ValueError("unsupported job kind")
        if not isinstance(payload, dict):
            raise ValueError("job payload must be an object")
        if not 1 <= int(max_attempts) <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        job_id = "job_" + uuid.uuid4().hex
        now = utc_now()
        record = JobRecord(
            tenant,
            job_id,
            kind,
            "queued",
            json.loads(_canonical_json(payload)),
            None,
            None,
            actor,
            now,
            now,
            now,
            None,
            None,
            0,
            int(max_attempts),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO inference_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tenant_id,
                    record.job_id,
                    record.kind,
                    record.status,
                    _canonical_json(record.payload),
                    None,
                    None,
                    record.created_by,
                    record.created_at,
                    record.updated_at,
                    record.available_at,
                    None,
                    None,
                    0,
                    record.max_attempts,
                ),
            )
            connection.commit()
        return record

    def get_job(self, *, tenant_id: str, job_id: str) -> JobRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        job = _identifier(job_id, "job_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_jobs WHERE tenant_id=? AND job_id=?",
                (tenant, job),
            ).fetchone()
        return self._job(row) if row else None

    def list_jobs(self, *, tenant_id: str, limit: int = 100) -> list[JobRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        bounded = min(max(int(limit), 1), 500)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM inference_jobs
                WHERE tenant_id=?
                ORDER BY created_at DESC, job_id DESC LIMIT ?
                """,
                (tenant, bounded),
            ).fetchall()
        return [self._job(row) for row in rows]

    def claim_job(self, *, worker_id: str, lease_seconds: int = 300) -> JobRecord | None:
        worker = _identifier(worker_id, "worker_id")
        if not 10 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 10 and 3600")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_until = (now_dt + timedelta(seconds=int(lease_seconds))).isoformat()
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT tenant_id, job_id FROM inference_jobs
                    WHERE attempts < max_attempts AND (
                        (status='queued' AND available_at <= ?)
                        OR (status='running' AND lease_expires_at < ?)
                    )
                    ORDER BY created_at, job_id LIMIT 1
                    """,
                    (now, now),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                changed = connection.execute(
                    """
                    UPDATE inference_jobs
                    SET status='running', lease_owner=?, lease_expires_at=?,
                        attempts=attempts+1, updated_at=?, error_code=NULL
                    WHERE tenant_id=? AND job_id=? AND attempts < max_attempts
                    """,
                    (worker, lease_until, now, row["tenant_id"], row["job_id"]),
                ).rowcount
                if changed != 1:
                    connection.rollback()
                    return None
                claimed = connection.execute(
                    "SELECT * FROM inference_jobs WHERE tenant_id=? AND job_id=?",
                    (row["tenant_id"], row["job_id"]),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(claimed)

    def renew_job_lease(
        self, *, job_id: str, worker_id: str, lease_seconds: int = 300
    ) -> bool:
        job = _identifier(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id")
        if not 10 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 10 and 3600")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_until = (now_dt + timedelta(seconds=int(lease_seconds))).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE inference_jobs
                SET lease_expires_at=?, updated_at=?
                WHERE job_id=? AND status='running' AND lease_owner=?
                  AND lease_expires_at>=?
                """,
                (lease_until, now, job, worker, now),
            ).rowcount
            connection.commit()
        return changed == 1

    def complete_job(
        self, *, job_id: str, worker_id: str, result: dict[str, Any]
    ) -> JobRecord:
        job = _identifier(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id")
        if not isinstance(result, dict):
            raise ValueError("job result must be an object")
        now = utc_now()
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                changed = connection.execute(
                    """
                    UPDATE inference_jobs
                    SET status='succeeded', result_json=?, error_code=NULL,
                        updated_at=?, lease_owner=NULL, lease_expires_at=NULL
                    WHERE job_id=? AND status='running' AND lease_owner=?
                    """,
                    (_canonical_json(result), now, job, worker),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("job lease is not owned by this worker")
                row = connection.execute(
                    "SELECT * FROM inference_jobs WHERE job_id=?", (job,)
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(row)

    def fail_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        retry_delay_seconds: int = 5,
    ) -> JobRecord:
        job = _identifier(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id")
        code = _identifier(error_code, "error_code")
        if not 0 <= int(retry_delay_seconds) <= 3600:
            raise ValueError("retry delay must be between 0 and 3600 seconds")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        available = (now_dt + timedelta(seconds=int(retry_delay_seconds))).isoformat()
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """
                    SELECT attempts, max_attempts FROM inference_jobs
                    WHERE job_id=? AND status='running' AND lease_owner=?
                    """,
                    (job, worker),
                ).fetchone()
                if current is None:
                    raise RuntimeError("job lease is not owned by this worker")
                terminal = int(current["attempts"]) >= int(current["max_attempts"])
                status = "failed" if terminal else "queued"
                connection.execute(
                    """
                    UPDATE inference_jobs
                    SET status=?, error_code=?, updated_at=?, available_at=?,
                        lease_owner=NULL, lease_expires_at=NULL
                    WHERE job_id=? AND lease_owner=?
                    """,
                    (status, code, now, available, job, worker),
                )
                row = connection.execute(
                    "SELECT * FROM inference_jobs WHERE job_id=?", (job,)
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(row)


def workspace_store_from_env() -> WorkspaceStore:
    """Build a storage backend without silently downgrading production."""

    database_url = os.environ.get("PERFUMERY_AI_DATABASE_URL", "").strip()
    environment = os.environ.get("PERFUMERY_AI_ENV", "development").strip().lower()
    if database_url.startswith(("postgresql://", "postgres://")):
        from .postgres import PostgresWorkspaceStore

        return PostgresWorkspaceStore(database_url)
    if database_url and not database_url.startswith("sqlite:///"):
        raise RuntimeError("PERFUMERY_AI_DATABASE_URL must use postgresql:// or sqlite:///")
    if environment == "production":
        raise RuntimeError("production requires a PostgreSQL PERFUMERY_AI_DATABASE_URL")
    path = (
        database_url.removeprefix("sqlite:///")
        if database_url
        else os.environ.get("PERFUMERY_AI_WORKSPACE_DB", "perfumery-workspace.db")
    )
    return SqliteWorkspaceStore(path)
