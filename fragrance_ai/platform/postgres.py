"""PostgreSQL workspace, queue, and distributed rate-limit backend."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import FormulaRecord, FormulaVersionRecord, JobRecord, ProjectRecord
from .store import (
    _CHANGE_KINDS,
    _FORMULA_KINDS,
    _JOB_KINDS,
    _bounded_text,
    _canonical_json,
    _identifier,
    _payload_hash,
    utc_now,
)


def _driver():
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PostgreSQL requires psycopg and psycopg-pool; "
            "install perfumery-ai-core[commercial]"
        ) from error
    return psycopg, dict_row, ConnectionPool


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS perfumery_projects (
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, project_id)
);
CREATE TABLE IF NOT EXISTS perfumery_formulas (
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    formula_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, formula_id),
    FOREIGN KEY (tenant_id, project_id)
        REFERENCES perfumery_projects(tenant_id, project_id)
);
CREATE INDEX IF NOT EXISTS ix_perfumery_formulas_project
    ON perfumery_formulas(tenant_id, project_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS perfumery_formula_versions (
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    formula_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    parent_version_id TEXT,
    change_kind TEXT NOT NULL,
    change_note TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY (tenant_id, version_id),
    UNIQUE (tenant_id, formula_id, version_number),
    FOREIGN KEY (tenant_id, formula_id)
        REFERENCES perfumery_formulas(tenant_id, formula_id)
);
CREATE INDEX IF NOT EXISTS ix_perfumery_formula_versions_formula
    ON perfumery_formula_versions(tenant_id, formula_id, version_number DESC);
CREATE TABLE IF NOT EXISTS perfumery_inference_jobs (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    result_json JSONB,
    error_code TEXT,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    attempts INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, job_id)
);
CREATE INDEX IF NOT EXISTS ix_perfumery_jobs_claim
    ON perfumery_inference_jobs(status, available_at, created_at);
CREATE TABLE IF NOT EXISTS perfumery_job_effects (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    effect_type TEXT NOT NULL,
    formula_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, job_id),
    FOREIGN KEY (tenant_id, job_id)
        REFERENCES perfumery_inference_jobs(tenant_id, job_id),
    FOREIGN KEY (tenant_id, formula_id)
        REFERENCES perfumery_formulas(tenant_id, formula_id),
    FOREIGN KEY (tenant_id, version_id)
        REFERENCES perfumery_formula_versions(tenant_id, version_id)
);
CREATE TABLE IF NOT EXISTS perfumery_rate_limit_buckets (
    identity_hash TEXT NOT NULL,
    bucket_start BIGINT NOT NULL,
    request_count INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (identity_hash, bucket_start)
);
CREATE INDEX IF NOT EXISTS ix_perfumery_rate_limit_expiry
    ON perfumery_rate_limit_buckets(bucket_start);
"""


class PostgresWorkspaceStore:
    """Horizontally safe persistence using transactions and SKIP LOCKED."""

    backend = "postgresql"
    horizontally_scalable = True

    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 12):
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("database_url must be PostgreSQL")
        if not 1 <= min_size <= max_size <= 100:
            raise ValueError("invalid PostgreSQL pool size")
        psycopg, dict_row, connection_pool = _driver()
        self._jsonb = psycopg.types.json.Jsonb
        self.pool = connection_pool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=True,
        )
        self._initialize()

    def _initialize(self) -> None:
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(POSTGRES_SCHEMA)
            connection.commit()

    @staticmethod
    def _project(row: dict[str, Any]) -> ProjectRecord:
        return ProjectRecord(
            tenant_id=str(row["tenant_id"]),
            project_id=str(row["project_id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            created_by=str(row["created_by"]),
            created_at=_iso(row["created_at"]) or "",
            updated_at=_iso(row["updated_at"]) or "",
        )

    @staticmethod
    def _version(row: dict[str, Any]) -> FormulaVersionRecord:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FormulaVersionRecord(
            tenant_id=str(row["tenant_id"]),
            project_id=str(row["project_id"]),
            formula_id=str(row["formula_id"]),
            version_id=str(row["version_id"]),
            version_number=int(row["version_number"]),
            parent_version_id=(
                str(row["parent_version_id"]) if row["parent_version_id"] else None
            ),
            change_kind=str(row["change_kind"]),
            change_note=str(row["change_note"]),
            created_by=str(row["created_by"]),
            created_at=_iso(row["created_at"]) or "",
            payload=dict(payload),
            content_sha256=str(row["content_sha256"]),
        )

    @staticmethod
    def _job(row: dict[str, Any]) -> JobRecord:
        payload = row["payload_json"]
        result = row["result_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(result, str):
            result = json.loads(result)
        return JobRecord(
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            payload=dict(payload),
            result=dict(result) if result is not None else None,
            error_code=str(row["error_code"]) if row["error_code"] else None,
            created_by=str(row["created_by"]),
            created_at=_iso(row["created_at"]) or "",
            updated_at=_iso(row["updated_at"]) or "",
            available_at=_iso(row["available_at"]) or "",
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] else None,
            lease_expires_at=_iso(row["lease_expires_at"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
        )

    def _formula_from_joined_row(self, row: dict[str, Any]) -> FormulaRecord:
        latest = row.get("latest_version")
        if isinstance(latest, str):
            latest = json.loads(latest)
        if not isinstance(latest, dict):
            raise RuntimeError("formula exists without a version")
        return FormulaRecord(
            tenant_id=str(row["tenant_id"]),
            project_id=str(row["project_id"]),
            formula_id=str(row["formula_id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            created_by=str(row["created_by"]),
            created_at=_iso(row["created_at"]) or "",
            updated_at=_iso(row["updated_at"]) or "",
            latest_version=self._version(latest),
        )

    def ping(self) -> bool:
        try:
            with self.pool.connection() as connection:
                return connection.execute("SELECT 1 AS ok").fetchone()["ok"] == 1
        except Exception:
            return False

    def close(self) -> None:
        self.pool.close()

    def create_project(
        self, *, tenant_id: str, name: str, description: str, actor_id: str
    ) -> ProjectRecord:
        tenant = _identifier(tenant_id, "tenant_id")
        actor = _identifier(actor_id, "actor_id")
        now = utc_now()
        record = ProjectRecord(
            tenant,
            "prj_" + uuid.uuid4().hex,
            _bounded_text(name, "name", 160),
            _bounded_text(description, "description", 4000, empty=True),
            actor,
            now,
            now,
        )
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO perfumery_projects
                (tenant_id, project_id, name, description, created_by, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                tuple(record.__dict__.values()),
            )
            connection.commit()
        return record

    def list_projects(self, *, tenant_id: str, limit: int = 100) -> list[ProjectRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        bounded = min(max(int(limit), 1), 500)
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM perfumery_projects
                WHERE tenant_id=%s ORDER BY updated_at DESC LIMIT %s
                """,
                (tenant, bounded),
            ).fetchall()
        return [self._project(row) for row in rows]

    def get_project(self, *, tenant_id: str, project_id: str) -> ProjectRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM perfumery_projects
                WHERE tenant_id=%s AND project_id=%s
                """,
                (tenant, project),
            ).fetchone()
        return self._project(row) if row else None

    def list_formulas(
        self, *, tenant_id: str, project_id: str, limit: int = 100
    ) -> list[FormulaRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        bounded = min(max(int(limit), 1), 500)
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT formulas.*, to_jsonb(latest) AS latest_version
                FROM perfumery_formulas AS formulas
                JOIN LATERAL (
                    SELECT * FROM perfumery_formula_versions AS versions
                    WHERE versions.tenant_id=formulas.tenant_id
                      AND versions.formula_id=formulas.formula_id
                    ORDER BY versions.version_number DESC
                    LIMIT 1
                ) AS latest ON TRUE
                WHERE formulas.tenant_id=%s AND formulas.project_id=%s
                ORDER BY formulas.updated_at DESC LIMIT %s
                """,
                (tenant, project, bounded),
            ).fetchall()
        return [self._formula_from_joined_row(row) for row in rows]

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
        note = _bounded_text(change_note, "change_note", 2000, empty=True)
        digest = _payload_hash(payload)
        clean_payload = json.loads(_canonical_json(payload))
        formula_name = _bounded_text(name, "name", 160)
        with self.pool.connection() as connection:
            try:
                exists = connection.execute(
                    """
                    SELECT 1 FROM perfumery_projects
                    WHERE tenant_id=%s AND project_id=%s FOR UPDATE
                    """,
                    (tenant, project),
                ).fetchone()
                if not exists:
                    raise KeyError("project not found")
                if source_job is not None:
                    locked_job = connection.execute(
                        """
                        SELECT job_id FROM perfumery_inference_jobs
                        WHERE tenant_id=%s AND job_id=%s FOR UPDATE
                        """,
                        (tenant, source_job),
                    ).fetchone()
                    if locked_job is None:
                        raise KeyError("source job not found")
                    effect = connection.execute(
                        """
                        SELECT * FROM perfumery_job_effects
                        WHERE tenant_id=%s AND job_id=%s
                        """,
                        (tenant, source_job),
                    ).fetchone()
                    if effect is not None:
                        if effect["effect_type"] != "formula.create":
                            raise RuntimeError("job effect kind conflict")
                        existing = connection.execute(
                            """
                            SELECT formulas.*, to_jsonb(latest) AS latest_version
                            FROM perfumery_formulas AS formulas
                            JOIN LATERAL (
                                SELECT * FROM perfumery_formula_versions AS versions
                                WHERE versions.tenant_id=formulas.tenant_id
                                  AND versions.formula_id=formulas.formula_id
                                ORDER BY versions.version_number DESC LIMIT 1
                            ) AS latest ON TRUE
                            WHERE formulas.tenant_id=%s AND formulas.project_id=%s
                              AND formulas.formula_id=%s
                            """,
                            (tenant, project, effect["formula_id"]),
                        ).fetchone()
                        if existing is None:
                            raise RuntimeError("job effect references a missing formula")
                        connection.commit()
                        return self._formula_from_joined_row(existing)
                connection.execute(
                    """
                    INSERT INTO perfumery_formulas
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (tenant, project, formula_id, formula_name, kind, actor, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO perfumery_formula_versions
                    VALUES (%s, %s, %s, %s, 1, NULL, 'generated', %s, %s, %s, %s, %s)
                    """,
                    (
                        tenant,
                        project,
                        formula_id,
                        version_id,
                        note,
                        actor,
                        now,
                        self._jsonb(clean_payload),
                        digest,
                    ),
                )
                if source_job is not None:
                    connection.execute(
                        """
                        INSERT INTO perfumery_job_effects
                        (tenant_id, job_id, effect_type, formula_id, version_id, created_at)
                        VALUES (%s, %s, 'formula.create', %s, %s, %s)
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
            note,
            actor,
            now,
            clean_payload,
            digest,
        )
        return FormulaRecord(
            tenant,
            project,
            formula_id,
            formula_name,
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
        note = _bounded_text(change_note, "change_note", 2000, empty=True)
        clean_payload = json.loads(_canonical_json(payload))
        digest = _payload_hash(clean_payload)
        with self.pool.connection() as connection:
            try:
                if source_job is not None:
                    locked_job = connection.execute(
                        """
                        SELECT job_id FROM perfumery_inference_jobs
                        WHERE tenant_id=%s AND job_id=%s FOR UPDATE
                        """,
                        (tenant, source_job),
                    ).fetchone()
                    if locked_job is None:
                        raise KeyError("source job not found")
                    effect = connection.execute(
                        """
                        SELECT * FROM perfumery_job_effects
                        WHERE tenant_id=%s AND job_id=%s
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
                            SELECT * FROM perfumery_formula_versions
                            WHERE tenant_id=%s AND formula_id=%s AND version_id=%s
                            """,
                            (tenant, formula, effect["version_id"]),
                        ).fetchone()
                        if existing is None:
                            raise RuntimeError("job effect references a missing version")
                        connection.commit()
                        return self._version(existing)
                locked_formula = connection.execute(
                    """
                    SELECT formula_id FROM perfumery_formulas
                    WHERE tenant_id=%s AND project_id=%s AND formula_id=%s
                    FOR UPDATE
                    """,
                    (tenant, project, formula),
                ).fetchone()
                if locked_formula is None:
                    raise KeyError("formula not found")
                latest = connection.execute(
                    """
                    SELECT version_id, version_number
                    FROM perfumery_formula_versions
                    WHERE tenant_id=%s AND project_id=%s AND formula_id=%s
                    ORDER BY version_number DESC LIMIT 1
                    """,
                    (tenant, project, formula),
                ).fetchone()
                if latest is None:
                    raise KeyError("formula not found")
                if latest["version_id"] != expected:
                    raise RuntimeError("formula version conflict")
                number = int(latest["version_number"]) + 1
                connection.execute(
                    """
                    INSERT INTO perfumery_formula_versions
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tenant,
                        project,
                        formula,
                        version_id,
                        number,
                        expected,
                        change_kind,
                        note,
                        actor,
                        now,
                        self._jsonb(clean_payload),
                        digest,
                    ),
                )
                connection.execute(
                    """
                    UPDATE perfumery_formulas SET updated_at=%s
                    WHERE tenant_id=%s AND project_id=%s AND formula_id=%s
                    """,
                    (now, tenant, project, formula),
                )
                connection.execute(
                    """
                    UPDATE perfumery_projects SET updated_at=%s
                    WHERE tenant_id=%s AND project_id=%s
                    """,
                    (now, tenant, project),
                )
                if source_job is not None:
                    connection.execute(
                        """
                        INSERT INTO perfumery_job_effects
                        (tenant_id, job_id, effect_type, formula_id, version_id, created_at)
                        VALUES (%s, %s, 'formula.version', %s, %s, %s)
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
            number,
            expected,
            change_kind,
            note,
            actor,
            now,
            clean_payload,
            digest,
        )

    def get_formula(
        self, *, tenant_id: str, project_id: str, formula_id: str
    ) -> FormulaRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT formulas.*, to_jsonb(latest) AS latest_version
                FROM perfumery_formulas AS formulas
                JOIN LATERAL (
                    SELECT * FROM perfumery_formula_versions AS versions
                    WHERE versions.tenant_id=formulas.tenant_id
                      AND versions.formula_id=formulas.formula_id
                    ORDER BY versions.version_number DESC
                    LIMIT 1
                ) AS latest ON TRUE
                WHERE formulas.tenant_id=%s AND formulas.project_id=%s
                  AND formulas.formula_id=%s
                """,
                (tenant, project, formula),
            ).fetchone()
        return self._formula_from_joined_row(row) if row else None

    def list_formula_versions(
        self, *, tenant_id: str, project_id: str, formula_id: str, limit: int = 100
    ) -> list[FormulaVersionRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        project = _identifier(project_id, "project_id")
        formula = _identifier(formula_id, "formula_id")
        bounded = min(max(int(limit), 1), 500)
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM perfumery_formula_versions
                WHERE tenant_id=%s AND project_id=%s AND formula_id=%s
                ORDER BY version_number DESC LIMIT %s
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
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM perfumery_formula_versions
                WHERE tenant_id=%s AND project_id=%s AND formula_id=%s AND version_id=%s
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
        now = utc_now()
        record = JobRecord(
            tenant,
            "job_" + uuid.uuid4().hex,
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
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO perfumery_inference_jobs
                VALUES (%s, %s, %s, 'queued', %s, NULL, NULL, %s, %s, %s,
                        %s, NULL, NULL, 0, %s)
                """,
                (
                    tenant,
                    record.job_id,
                    kind,
                    self._jsonb(record.payload),
                    actor,
                    now,
                    now,
                    now,
                    record.max_attempts,
                ),
            )
            connection.commit()
        return record

    def get_job(self, *, tenant_id: str, job_id: str) -> JobRecord | None:
        tenant = _identifier(tenant_id, "tenant_id")
        job = _identifier(job_id, "job_id")
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM perfumery_inference_jobs
                WHERE tenant_id=%s AND job_id=%s
                """,
                (tenant, job),
            ).fetchone()
        return self._job(row) if row else None

    def list_jobs(self, *, tenant_id: str, limit: int = 100) -> list[JobRecord]:
        tenant = _identifier(tenant_id, "tenant_id")
        bounded = min(max(int(limit), 1), 500)
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM perfumery_inference_jobs
                WHERE tenant_id=%s
                ORDER BY created_at DESC, job_id DESC LIMIT %s
                """,
                (tenant, bounded),
            ).fetchall()
        return [self._job(row) for row in rows]

    def claim_job(self, *, worker_id: str, lease_seconds: int = 300) -> JobRecord | None:
        worker = _identifier(worker_id, "worker_id")
        if not 10 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 10 and 3600")
        with self.pool.connection() as connection:
            try:
                row = connection.execute(
                    """
                    WITH candidate AS (
                        SELECT tenant_id, job_id
                        FROM perfumery_inference_jobs
                        WHERE attempts < max_attempts AND (
                            (status='queued' AND available_at <= NOW())
                            OR (status='running' AND lease_expires_at < NOW())
                        )
                        ORDER BY created_at, job_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE perfumery_inference_jobs AS jobs
                    SET status='running', lease_owner=%s,
                        lease_expires_at=NOW() + (%s * INTERVAL '1 second'),
                        attempts=jobs.attempts + 1, updated_at=NOW(), error_code=NULL
                    FROM candidate
                    WHERE jobs.tenant_id=candidate.tenant_id
                      AND jobs.job_id=candidate.job_id
                    RETURNING jobs.*
                    """,
                    (worker, int(lease_seconds)),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(row) if row else None

    def complete_job(
        self, *, job_id: str, worker_id: str, result: dict[str, Any]
    ) -> JobRecord:
        job = _identifier(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id")
        if not isinstance(result, dict):
            raise ValueError("job result must be an object")
        with self.pool.connection() as connection:
            try:
                row = connection.execute(
                    """
                    UPDATE perfumery_inference_jobs
                    SET status='succeeded', result_json=%s, error_code=NULL,
                        updated_at=NOW(), lease_owner=NULL, lease_expires_at=NULL
                    WHERE job_id=%s AND status='running' AND lease_owner=%s
                    RETURNING *
                    """,
                    (self._jsonb(json.loads(_canonical_json(result))), job, worker),
                ).fetchone()
                if row is None:
                    raise RuntimeError("job lease is not owned by this worker")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(row)

    def renew_job_lease(
        self, *, job_id: str, worker_id: str, lease_seconds: int = 300
    ) -> bool:
        job = _identifier(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id")
        if not 10 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 10 and 3600")
        with self.pool.connection() as connection:
            try:
                row = connection.execute(
                    """
                    UPDATE perfumery_inference_jobs
                    SET lease_expires_at=NOW() + (%s * INTERVAL '1 second'),
                        updated_at=NOW()
                    WHERE job_id=%s AND status='running' AND lease_owner=%s
                      AND lease_expires_at>=NOW()
                    RETURNING job_id
                    """,
                    (int(lease_seconds), job, worker),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return row is not None

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
        with self.pool.connection() as connection:
            try:
                row = connection.execute(
                    """
                    UPDATE perfumery_inference_jobs
                    SET status=CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                        error_code=%s, updated_at=NOW(),
                        available_at=NOW() + (%s * INTERVAL '1 second'),
                        lease_owner=NULL, lease_expires_at=NULL
                    WHERE job_id=%s AND status='running' AND lease_owner=%s
                    RETURNING *
                    """,
                    (code, int(retry_delay_seconds), job, worker),
                ).fetchone()
                if row is None:
                    raise RuntimeError("job lease is not owned by this worker")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._job(row)


class PostgresFixedWindowRateLimiter:
    """Atomic distributed limiter shared by all API replicas."""

    def __init__(
        self,
        store: PostgresWorkspaceStore,
        *,
        requests: int = 60,
        window_seconds: int = 60,
    ):
        if requests <= 0 or not 1 <= int(window_seconds) <= 3600:
            raise ValueError("invalid rate-limit values")
        self.store = store
        self.requests = int(requests)
        self.window_seconds = int(window_seconds)

    def allow(self, identity: str, now: float | None = None) -> bool:
        import hashlib
        import time

        current = time.time() if now is None else float(now)
        bucket = int(current // self.window_seconds) * self.window_seconds
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        with self.store.pool.connection() as connection:
            try:
                connection.execute(
                    """
                    DELETE FROM perfumery_rate_limit_buckets
                    WHERE bucket_start < %s
                    """,
                    (bucket - self.window_seconds,),
                )
                row = connection.execute(
                    """
                    INSERT INTO perfumery_rate_limit_buckets
                        (identity_hash, bucket_start, request_count, updated_at)
                    VALUES (%s, %s, 1, NOW())
                    ON CONFLICT (identity_hash, bucket_start)
                    DO UPDATE SET request_count=perfumery_rate_limit_buckets.request_count + 1,
                                  updated_at=NOW()
                    WHERE perfumery_rate_limit_buckets.request_count < %s
                    RETURNING request_count
                    """,
                    (identity_hash, bucket, self.requests),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return row is not None and int(row["request_count"]) <= self.requests
