"""Append-only, hash-chained operational audit events.

The log is an application integrity control, not a substitute for a managed
WORM store.  SQLite triggers prevent accidental mutation and the hash chain
detects offline edits when copied to an independently retained sink.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sqlite_lifecycle import SQLiteConnectionOwner


GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    occurred_at: str
    actor_id: str
    actor_role: str
    event_type: str
    scope_id: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str
    event_mac: str = ""


class AppendOnlyAuditLog(SQLiteConnectionOwner):
    """Thread-safe append-only SQLite log with deterministic chain validation."""

    def __init__(
        self,
        path: str | Path,
        *,
        signing_key: bytes | None = None,
    ):
        self.path = str(path)
        if signing_key is not None and len(signing_key) < 32:
            raise ValueError("audit signing key must contain at least 32 bytes")
        self._signing_key = signing_key
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                event_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE,
                event_mac TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(audit_events)")
        }
        if "event_mac" not in columns:
            self.connection.execute(
                "ALTER TABLE audit_events ADD COLUMN event_mac TEXT NOT NULL DEFAULT ''"
            )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events are append-only');
            END
            """
        )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events are append-only');
            END
            """
        )
        self.connection.commit()

    @staticmethod
    def _canonical_payload(payload: dict[str, Any]) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def _hash_fields(
        cls,
        sequence: int,
        occurred_at: str,
        actor_id: str,
        actor_role: str,
        event_type: str,
        scope_id: str,
        payload_json: str,
        previous_hash: str,
    ) -> str:
        canonical = json.dumps(
            {
                "sequence": sequence,
                "occurred_at": occurred_at,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "event_type": event_type,
                "scope_id": scope_id,
                "payload_json": payload_json,
                "previous_hash": previous_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

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
        if not all(value and value.strip() for value in required):
            raise ValueError("actor, role, event type and scope are required")
        occurred_at = occurred_at or datetime.now(timezone.utc).isoformat()
        payload_json = self._canonical_payload(payload)
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                last = self.connection.execute(
                    "SELECT sequence, event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = int(last[0]) + 1 if last else 1
                previous_hash = str(last[1]) if last else GENESIS_HASH
                event_hash = self._hash_fields(
                    sequence,
                    occurred_at,
                    actor_id,
                    actor_role,
                    event_type,
                    scope_id,
                    payload_json,
                    previous_hash,
                )
                event_mac = (
                    hmac.new(
                        self._signing_key,
                        event_hash.encode("ascii"),
                        hashlib.sha256,
                    ).hexdigest()
                    if self._signing_key is not None
                    else ""
                )
                self.connection.execute(
                    """
                    INSERT INTO audit_events
                    (sequence, occurred_at, actor_id, actor_role, event_type,
                     scope_id, payload_json, previous_hash, event_hash, event_mac)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence,
                        occurred_at,
                        actor_id,
                        actor_role,
                        event_type,
                        scope_id,
                        payload_json,
                        previous_hash,
                        event_hash,
                        event_mac,
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return AuditEvent(
            sequence,
            occurred_at,
            actor_id,
            actor_role,
            event_type,
            scope_id,
            json.loads(payload_json),
            previous_hash,
            event_hash,
            event_mac,
        )

    def verify(self) -> dict[str, Any]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT sequence, occurred_at, actor_id, actor_role, event_type,
                       scope_id, payload_json, previous_hash, event_hash, event_mac
                FROM audit_events ORDER BY sequence
                """
            ).fetchall()
        previous = GENESIS_HASH
        failures: list[str] = []
        for expected_sequence, row in enumerate(rows, start=1):
            sequence = int(row[0])
            if sequence != expected_sequence:
                failures.append(f"sequence gap at {expected_sequence}")
            if str(row[7]) != previous:
                failures.append(f"previous hash mismatch at {sequence}")
            calculated = self._hash_fields(
                sequence,
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
            )
            if calculated != str(row[8]):
                failures.append(f"event hash mismatch at {sequence}")
            if self._signing_key is not None:
                expected_mac = hmac.new(
                    self._signing_key,
                    str(row[8]).encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
                if not str(row[9]) or not hmac.compare_digest(expected_mac, str(row[9])):
                    failures.append(f"event MAC mismatch at {sequence}")
            previous = str(row[8])
        return {
            "passed": not failures,
            "events": len(rows),
            "head_hash": previous,
            "failures": failures,
            "authenticated_chain": self._signing_key is not None,
            "retention_boundary": (
                "Copy the head hash and database snapshot to an independently "
                "controlled immutable store; a local administrator can replace both."
            ),
        }

    def close(self) -> None:
        with self._lock:
            self.connection.close()
