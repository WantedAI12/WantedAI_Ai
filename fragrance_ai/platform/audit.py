"""Authenticated audit-log selection for single-node and PostgreSQL services."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..recommender.audit_log import (
    GENESIS_HASH,
    AppendOnlyAuditLog,
    AuditEvent,
)


class AuditLog(Protocol):
    def append(
        self,
        *,
        actor_id: str,
        actor_role: str,
        event_type: str,
        scope_id: str,
        payload: dict[str, Any],
        occurred_at: str | None = None,
    ) -> AuditEvent: ...

    def verify(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


POSTGRES_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS perfumery_audit_events (
    sequence BIGINT PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL,
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    event_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE,
    event_mac TEXT NOT NULL
);
CREATE OR REPLACE FUNCTION perfumery_reject_audit_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'perfumery audit events are append-only';
END;
$$;
DROP TRIGGER IF EXISTS perfumery_audit_no_update ON perfumery_audit_events;
CREATE TRIGGER perfumery_audit_no_update
BEFORE UPDATE ON perfumery_audit_events
FOR EACH ROW EXECUTE FUNCTION perfumery_reject_audit_mutation();
DROP TRIGGER IF EXISTS perfumery_audit_no_delete ON perfumery_audit_events;
CREATE TRIGGER perfumery_audit_no_delete
BEFORE DELETE ON perfumery_audit_events
FOR EACH ROW EXECUTE FUNCTION perfumery_reject_audit_mutation();
"""


def _canonical_occurred_at(value: str | None) -> str:
    """Return one UTC representation so database round-trips preserve hashes."""

    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError("occurred_at must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("occurred_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("occurred_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


class PostgresAuditLog:
    """Globally ordered HMAC-authenticated chain safe across API replicas."""

    def __init__(self, database_url: str, *, signing_key: bytes):
        if len(signing_key) < 32:
            raise ValueError("PostgreSQL audit signing key must contain at least 32 bytes")
        try:
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
            from psycopg_pool import ConnectionPool
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "PostgreSQL audit requires perfumery-ai-core[commercial]"
            ) from error
        self._signing_key = signing_key
        self._jsonb = Jsonb
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=8,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=True,
        )
        with self.pool.connection() as connection:
            connection.execute(POSTGRES_AUDIT_SCHEMA)
            connection.commit()

    def append(
        self,
        *,
        actor_id: str,
        actor_role: str,
        event_type: str,
        scope_id: str,
        payload: dict[str, Any],
        occurred_at: str | None = None,
    ) -> AuditEvent:
        required = (actor_id, actor_role, event_type, scope_id)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise ValueError("actor, role, event type and scope are required")
        occurred = _canonical_occurred_at(occurred_at)
        payload_json = AppendOnlyAuditLog._canonical_payload(payload)
        with self.pool.connection() as connection:
            try:
                # One short transaction serializes the chain head across all replicas.
                connection.execute("SELECT pg_advisory_xact_lock(734829104)")
                last = connection.execute(
                    """
                    SELECT sequence, event_hash FROM perfumery_audit_events
                    ORDER BY sequence DESC LIMIT 1
                    """
                ).fetchone()
                sequence = int(last["sequence"]) + 1 if last else 1
                previous = str(last["event_hash"]) if last else GENESIS_HASH
                event_hash = AppendOnlyAuditLog._hash_fields(
                    sequence,
                    occurred,
                    actor_id,
                    actor_role,
                    event_type,
                    scope_id,
                    payload_json,
                    previous,
                )
                event_mac = hmac.new(
                    self._signing_key,
                    event_hash.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO perfumery_audit_events
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sequence,
                        occurred,
                        actor_id,
                        actor_role,
                        event_type,
                        scope_id,
                        self._jsonb(json.loads(payload_json)),
                        previous,
                        event_hash,
                        event_mac,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return AuditEvent(
            sequence,
            occurred,
            actor_id,
            actor_role,
            event_type,
            scope_id,
            json.loads(payload_json),
            previous,
            event_hash,
            event_mac,
        )

    def verify(self) -> dict[str, Any]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM perfumery_audit_events ORDER BY sequence"
            ).fetchall()
        previous = GENESIS_HASH
        failures: list[str] = []
        for expected, row in enumerate(rows, start=1):
            sequence = int(row["sequence"])
            if sequence != expected:
                failures.append(f"sequence gap at {expected}")
            payload = row["payload_json"]
            payload_json = AppendOnlyAuditLog._canonical_payload(
                json.loads(payload) if isinstance(payload, str) else payload
            )
            if str(row["previous_hash"]) != previous:
                failures.append(f"previous hash mismatch at {sequence}")
            occurred_at = row["occurred_at"]
            occurred = (
                occurred_at.astimezone(timezone.utc).isoformat()
                if isinstance(occurred_at, datetime)
                else str(occurred_at)
            )
            calculated = AppendOnlyAuditLog._hash_fields(
                sequence,
                occurred,
                str(row["actor_id"]),
                str(row["actor_role"]),
                str(row["event_type"]),
                str(row["scope_id"]),
                payload_json,
                str(row["previous_hash"]),
            )
            if calculated != str(row["event_hash"]):
                failures.append(f"event hash mismatch at {sequence}")
            expected_mac = hmac.new(
                self._signing_key,
                str(row["event_hash"]).encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected_mac, str(row["event_mac"])):
                failures.append(f"event MAC mismatch at {sequence}")
            previous = str(row["event_hash"])
        return {
            "passed": not failures,
            "events": len(rows),
            "head_hash": previous,
            "failures": failures,
            "authenticated_chain": True,
            "backend": "postgresql",
            "retention_boundary": (
                "Export signed head hashes to an independently controlled WORM sink; "
                "database superusers remain outside the application trust boundary."
            ),
        }

    def close(self) -> None:
        self.pool.close()


def _signing_key_from_env(*, required: bool) -> bytes | None:
    encoded = os.environ.get("PERFUMERY_AI_AUDIT_HMAC_KEY", "").strip()
    if not encoded:
        if required:
            raise RuntimeError("PERFUMERY_AI_AUDIT_HMAC_KEY is required in production")
        return None
    try:
        key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as error:
        raise RuntimeError("PERFUMERY_AI_AUDIT_HMAC_KEY must be URL-safe base64") from error
    if len(key) < 32:
        raise RuntimeError("PERFUMERY_AI_AUDIT_HMAC_KEY must decode to at least 32 bytes")
    return key


def audit_log_from_env() -> AuditLog:
    environment = os.environ.get("PERFUMERY_AI_ENV", "development").strip().lower()
    production = environment == "production"
    key = _signing_key_from_env(required=production)
    database_url = os.environ.get("PERFUMERY_AI_DATABASE_URL", "").strip()
    if database_url.startswith(("postgresql://", "postgres://")):
        if key is None:
            raise RuntimeError("PostgreSQL audit requires PERFUMERY_AI_AUDIT_HMAC_KEY")
        return PostgresAuditLog(database_url, signing_key=key)
    if production:
        raise RuntimeError("production audit requires PostgreSQL")
    path = Path(os.environ.get("PERFUMERY_AI_AUDIT_DB", "perfumery-audit.db"))
    return AppendOnlyAuditLog(path, signing_key=key)
