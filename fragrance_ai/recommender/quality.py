"""Formula fingerprints and fail-closed physical-quality evidence.

The historical ``record`` interface remains for audit migration only.  It is
deliberately unable to approve a formulation.  Release-quality evidence must
be an independently signed artifact whose protocol and report bytes are
re-hashed before *every* assessment.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .artifact_trust import EvidenceTrustRoot
from .manufacturing import REQUIRED_STABILITY_TESTS
from .models import RecipeLine
from .sqlite_lifecycle import SQLiteConnectionOwner


_SHA256_SCOPE = re.compile(r"^sha256:[0-9a-f]{64}$")
QUALITY_ARTIFACT_TYPE = "quality_stability_report/v1"
QUALITY_SIGNER_ROLES = frozenset({"quality_authority", "quality_laboratory", "qa_reviewer"})


def formula_fingerprint(lines: list[RecipeLine]) -> str:
    canonical = [
        {
            "ingredient_id": line.ingredient_id,
            "concentrate_percent": round(float(line.concentrate_percent), 4),
        }
        for line in sorted(lines, key=lambda item: item.ingredient_id)
    ]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def release_spec_fingerprint(
    lines: list[RecipeLine],
    constraints,
    supplier_registry,
    *,
    rule_pack_version: str,
    data_version: str,
    model_version: str,
    as_of=None,
) -> str:
    """Return the immutable commercial scope ID for a current product run."""
    from .release_spec import ReleaseSpec

    return ReleaseSpec.build(
        lines,
        constraints,
        supplier_registry,
        rule_pack_version=rule_pack_version,
        data_version=data_version,
        model_version=model_version,
        as_of=as_of,
    ).release_spec_id


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    status: str
    passed_tests: tuple[str, ...]
    missing_tests: tuple[str, ...]
    failed_tests: tuple[str, ...]
    verification_failures: tuple[str, ...] = ()
    evidence_source: str = "verified_external_evidence_only"


class QualityEvidenceStore(SQLiteConnectionOwner):
    """SQLite index for quality evidence; signatures and files remain authoritative.

    ``trusted_signers`` must be an independent allowlist managed outside the
    evidence database.  Passing no root is valid for R&D/read-only migration,
    but it trusts no quality report and therefore cannot pass an assessment.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        trusted_signers: EvidenceTrustRoot | Mapping[str, Any] | None = None,
    ):
        self.path = str(path)
        self.trust_root = (
            trusted_signers
            if isinstance(trusted_signers, EvidenceTrustRoot)
            else EvidenceTrustRoot(trusted_signers)
        )
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS quality_results (
                formula_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('pass', 'fail')),
                completed_on TEXT NOT NULL,
                protocol_ref TEXT NOT NULL,
                report_ref TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                migration_status TEXT NOT NULL DEFAULT 'legacy_unverified',
                PRIMARY KEY (formula_id, test_name, completed_on, report_ref)
            );
            CREATE TABLE IF NOT EXISTS verified_quality_results (
                release_spec_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('pass', 'fail')),
                completed_on TEXT NOT NULL,
                protocol_path TEXT NOT NULL,
                report_path TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                PRIMARY KEY (release_spec_id, test_name, completed_on, artifact_id)
            );
            CREATE TABLE IF NOT EXISTS quality_artifact_revocations (
                artifact_id TEXT PRIMARY KEY,
                revoked_on TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            """
        )
        columns = {str(row[1]) for row in self.connection.execute("PRAGMA table_info(quality_results)")}
        if "migration_status" not in columns:
            self.connection.execute(
                "ALTER TABLE quality_results ADD COLUMN migration_status TEXT NOT NULL DEFAULT 'legacy_unverified'"
            )
        self.connection.commit()

    @staticmethod
    def _validate_release_spec_id(value: str) -> str:
        scope = str(value or "").strip().lower()
        if not _SHA256_SCOPE.fullmatch(scope):
            raise ValueError("release_spec_id must be a SHA-256 commercial release scope ID")
        return scope

    @staticmethod
    def _validate_test(test_name: str, result: str) -> tuple[str, str]:
        if test_name not in REQUIRED_STABILITY_TESTS:
            raise ValueError(f"unknown quality test: {test_name}")
        if result not in {"pass", "fail"}:
            raise ValueError("result must be pass or fail")
        return test_name, result

    def record(
        self,
        formula_id: str,
        test_name: str,
        result: str,
        completed_on: date,
        protocol_ref: str,
        report_ref: str,
        notes: str = "",
    ) -> None:
        """Store an unsigned historical row for audit, never as approval evidence."""
        self._validate_test(test_name, result)
        if not str(formula_id).startswith("sha256:"):
            raise ValueError("formula_id must be a SHA-256 fingerprint")
        if not str(protocol_ref).strip() or not str(report_ref).strip():
            raise ValueError("protocol_ref and report_ref are required")
        self.connection.execute(
            """INSERT INTO quality_results
            (formula_id, test_name, result, completed_on, protocol_ref, report_ref, notes, migration_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'legacy_unverified')""",
            (
                formula_id,
                test_name,
                result,
                completed_on.isoformat(),
                protocol_ref,
                report_ref,
                notes,
            ),
        )
        self.connection.commit()

    def record_verified(
        self,
        release_spec_id: str,
        test_name: str,
        result: str,
        completed_on: date,
        protocol_path: str | Path,
        report_path: str | Path,
        signed_envelope: Mapping[str, Any],
        *,
        as_of: date | None = None,
    ) -> None:
        """Verify and index a real QA protocol/report pair for one exact release.

        The envelope must sign both byte hashes and the release scope, test,
        result, and completed date.  A claimed hash, URL, or a database row is
        insufficient.
        """
        scope = self._validate_release_spec_id(release_spec_id)
        test_name, result = self._validate_test(test_name, result)
        if not isinstance(completed_on, date):
            raise ValueError("completed_on must be a date")
        verified = self.trust_root.verify(
            signed_envelope,
            {"protocol": protocol_path, "report": report_path},
            expected_artifact_type=QUALITY_ARTIFACT_TYPE,
            expected_scope={
                "release_spec_id": scope,
                "test_name": test_name,
                "result": result,
                "completed_on": completed_on.isoformat(),
            },
            allowed_roles=QUALITY_SIGNER_ROLES,
            as_of=as_of,
            local_revocations=self._local_revocations(),
        )
        self.connection.execute(
            """INSERT INTO verified_quality_results
            (release_spec_id, test_name, result, completed_on, protocol_path, report_path, envelope_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(release_spec_id, test_name, completed_on, artifact_id)
            DO UPDATE SET result=excluded.result, protocol_path=excluded.protocol_path,
                          report_path=excluded.report_path, envelope_json=excluded.envelope_json""",
            (
                scope,
                test_name,
                result,
                completed_on.isoformat(),
                str(Path(protocol_path).expanduser().resolve(strict=True)),
                str(Path(report_path).expanduser().resolve(strict=True)),
                json.dumps(dict(signed_envelope), ensure_ascii=False, sort_keys=True),
                verified.artifact_id,
            ),
        )
        self.connection.commit()

    def revoke(self, artifact_id: str, revoked_on: date, reason: str) -> None:
        """Revoke an indexed artifact; it will fail all later assessments."""
        if not str(artifact_id).strip() or not str(reason).strip():
            raise ValueError("artifact_id and revocation reason are required")
        self.connection.execute(
            """INSERT INTO quality_artifact_revocations VALUES (?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET revoked_on=excluded.revoked_on, reason=excluded.reason""",
            (str(artifact_id).strip(), revoked_on.isoformat(), str(reason).strip()),
        )
        self.connection.commit()

    def _local_revocations(self) -> dict[str, date]:
        return {
            str(identifier): date.fromisoformat(str(revoked_on))
            for identifier, revoked_on in self.connection.execute(
                "SELECT artifact_id, revoked_on FROM quality_artifact_revocations"
            )
        }

    def assess(
        self, release_spec_id: str, as_of: date | None = None
    ) -> QualityAssessment:
        """Re-verify each current report before allowing an all-test pass.

        A raw formula fingerprint does not contain the product/base/packaging
        commercial scope and is therefore intentionally treated as legacy.
        """
        try:
            scope = self._validate_release_spec_id(release_spec_id)
        except ValueError:
            legacy_rows = self.connection.execute(
                "SELECT 1 FROM quality_results WHERE formula_id = ? LIMIT 1", (release_spec_id,)
            ).fetchone()
            return QualityAssessment(
                False,
                "legacy_unverified" if legacy_rows else "release_scope_required",
                (),
                tuple(sorted(REQUIRED_STABILITY_TESTS)),
                (),
                ("unsigned legacy quality rows cannot approve a commercial scope",),
                "legacy_unverified" if legacy_rows else "no_verified_evidence",
            )

        rows = self.connection.execute(
            """SELECT test_name, result, completed_on, protocol_path, report_path,
                       envelope_json, artifact_id
               FROM verified_quality_results WHERE release_spec_id = ?
               ORDER BY test_name, completed_on DESC, artifact_id DESC""",
            (scope,),
        ).fetchall()
        latest: dict[str, str] = {}
        failures: list[str] = []
        local_revocations = self._local_revocations()
        for test_name, result, completed, protocol_path, report_path, envelope_json, artifact_id in rows:
            name = str(test_name)
            if name in latest:
                continue
            # The newest claimed evidence decides the status of a test.  Do
            # not silently fall back to an older report if a current one is
            # altered, revoked, expired, or signed by an untrusted party.
            try:
                envelope = json.loads(str(envelope_json))
                self.trust_root.verify(
                    envelope,
                    {"protocol": str(protocol_path), "report": str(report_path)},
                    expected_artifact_type=QUALITY_ARTIFACT_TYPE,
                    expected_scope={
                        "release_spec_id": scope,
                        "test_name": name,
                        "result": str(result),
                        "completed_on": str(completed),
                    },
                    allowed_roles=QUALITY_SIGNER_ROLES,
                    as_of=as_of,
                    local_revocations=local_revocations,
                )
                if date.fromisoformat(str(completed)) > (as_of or date.today()):
                    raise ValueError("quality test completion is in the future")
                latest[name] = str(result)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                latest[name] = "invalid"
                failures.append(f"{name}/{artifact_id}: {error}")

        passed = sorted(name for name, result in latest.items() if result == "pass")
        failed = sorted(name for name, result in latest.items() if result in {"fail", "invalid"})
        missing = sorted(set(REQUIRED_STABILITY_TESTS) - set(latest))
        complete = not failed and not missing and len(passed) == len(REQUIRED_STABILITY_TESTS)
        legacy_rows = self.connection.execute(
            "SELECT 1 FROM quality_results WHERE formula_id = ? LIMIT 1", (scope,)
        ).fetchone()
        if complete:
            status = "verified_passed"
        elif failures:
            status = "verified_evidence_invalid"
        elif not rows:
            status = "legacy_unverified" if legacy_rows else "verified_evidence_missing"
        else:
            status = "verified_incomplete_or_failed"
        return QualityAssessment(
            passed=complete,
            status=status,
            passed_tests=tuple(passed),
            missing_tests=tuple(missing),
            failed_tests=tuple(failed),
            verification_failures=tuple(failures),
            evidence_source="legacy_unverified" if legacy_rows and not complete else "verified_external_evidence_only",
        )

    def close(self) -> None:
        self.connection.close()
