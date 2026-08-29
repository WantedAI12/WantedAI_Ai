from __future__ import annotations

import inspect

import pytest

from fragrance_ai.platform.audit import POSTGRES_AUDIT_SCHEMA, audit_log_from_env
from fragrance_ai.platform.postgres import (
    POSTGRES_SCHEMA,
    PostgresWorkspaceStore,
    _driver,
)


def test_postgres_schema_binds_tenant_keys_and_json_payloads():
    normalized = " ".join(POSTGRES_SCHEMA.split())
    assert "PRIMARY KEY (tenant_id, project_id)" in normalized
    assert "PRIMARY KEY (tenant_id, formula_id)" in normalized
    assert "PRIMARY KEY (tenant_id, version_id)" in normalized
    assert "FOREIGN KEY (tenant_id, project_id)" in normalized
    assert "FOREIGN KEY (tenant_id, formula_id)" in normalized
    assert "payload_json JSONB NOT NULL" in normalized
    assert "UNIQUE (tenant_id, formula_id, version_number)" in normalized
    assert "CREATE TABLE IF NOT EXISTS perfumery_job_effects" in normalized
    assert "PRIMARY KEY (tenant_id, job_id)" in normalized


def test_postgres_queue_claim_and_heartbeat_are_lease_guarded():
    claim_source = inspect.getsource(PostgresWorkspaceStore.claim_job)
    heartbeat_source = inspect.getsource(PostgresWorkspaceStore.renew_job_lease)
    completion_source = inspect.getsource(PostgresWorkspaceStore.complete_job)
    assert "FOR UPDATE SKIP LOCKED" in claim_source
    assert "lease_expires_at < NOW()" in claim_source
    assert "status='running' AND lease_owner=%s" in heartbeat_source
    assert "lease_expires_at>=NOW()" in heartbeat_source
    assert "status='running' AND lease_owner=%s" in completion_source


def test_postgres_version_append_serializes_on_formula_parent_and_lists_without_n_plus_one():
    append_source = inspect.getsource(PostgresWorkspaceStore.append_formula_version)
    list_source = inspect.getsource(PostgresWorkspaceStore.list_formulas)
    formula_lock = append_source.index("SELECT formula_id FROM perfumery_formulas")
    latest_read = append_source.index("SELECT version_id, version_number", formula_lock)
    assert formula_lock < latest_read
    assert "FOR UPDATE" in append_source[formula_lock:latest_read]
    assert "JOIN LATERAL" in list_source
    assert "_formula(connection" not in list_source
    assert "perfumery_job_effects" in append_source


def test_postgres_driver_and_json_adapter_are_installed():
    psycopg, dict_row, connection_pool = _driver()
    assert psycopg.types.json.Jsonb is not None
    assert callable(dict_row)
    assert connection_pool.__name__ == "ConnectionPool"


def test_postgres_audit_schema_is_append_only_and_globally_ordered():
    normalized = " ".join(POSTGRES_AUDIT_SCHEMA.split())
    assert "sequence BIGINT PRIMARY KEY" in normalized
    assert "payload_json JSONB NOT NULL" in normalized
    assert "event_hash TEXT NOT NULL UNIQUE" in normalized
    assert "BEFORE UPDATE" in normalized
    assert "BEFORE DELETE" in normalized


def test_production_audit_refuses_missing_hmac_key(monkeypatch):
    monkeypatch.setenv("PERFUMERY_AI_ENV", "production")
    monkeypatch.setenv(
        "PERFUMERY_AI_DATABASE_URL",
        "postgresql://perfumery:password@localhost/perfumery",
    )
    monkeypatch.delenv("PERFUMERY_AI_AUDIT_HMAC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required in production"):
        audit_log_from_env()
