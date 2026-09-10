import sqlite3

import pytest

from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog


def test_hash_chained_audit_log_is_append_only_and_verifiable(tmp_path):
    log = AppendOnlyAuditLog(tmp_path / "audit.db")
    first = log.append(
        actor_id="operator-1",
        actor_role="formulator",
        event_type="recipe.requested",
        scope_id="request:1",
        payload={"brief_sha256": "a" * 64},
        occurred_at="2026-07-28T00:00:00+00:00",
    )
    second = log.append(
        actor_id="operator-1",
        actor_role="formulator",
        event_type="recipe.completed",
        scope_id="request:1",
        payload={"status": "prototype_ready"},
        occurred_at="2026-07-28T00:00:01+00:00",
    )
    assert second.previous_hash == first.event_hash
    assert log.verify()["passed"]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.connection.execute("DELETE FROM audit_events")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.connection.execute(
            "UPDATE audit_events SET actor_id='attacker' WHERE sequence=1"
        )
    log.close()


def test_audit_log_requires_actor_and_scope(tmp_path):
    log = AppendOnlyAuditLog(tmp_path / "audit.db")
    with pytest.raises(ValueError):
        log.append(
            actor_id="",
            actor_role="formulator",
            event_type="recipe.requested",
            scope_id="request:1",
            payload={},
        )
    log.close()


def test_authenticated_audit_chain_detects_mac_tampering(tmp_path):
    key = b"k" * 32
    log = AppendOnlyAuditLog(tmp_path / "signed-audit.db", signing_key=key)
    event = log.append(
        actor_id="operator-1",
        actor_role="formulator",
        event_type="formula.saved",
        scope_id="tenant:t:formula:f",
        payload={"tenant_id": "t"},
    )
    assert event.event_mac
    verified = log.verify()
    assert verified["passed"]
    assert verified["authenticated_chain"] is True
    # Bypass the application trigger only to emulate an offline file rewrite.
    log.connection.execute("DROP TRIGGER audit_events_no_update")
    log.connection.execute(
        "UPDATE audit_events SET event_mac=? WHERE sequence=1", ("0" * 64,)
    )
    assert log.verify()["passed"] is False
    log.close()
