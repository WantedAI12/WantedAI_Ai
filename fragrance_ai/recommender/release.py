"""Cryptographically verified external evidence for commercial release.

The previous ``RegulatorySignoff`` API accepted strings and a claimed hash.
It remains import-compatible solely to preserve historical records, but those
records are explicitly legacy and can never make a product manufacturing-ready.
"""

from __future__ import annotations

import base64
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .release_spec import ReleaseSpec, canonical_json, sha256_file
from .sqlite_lifecycle import SQLiteConnectionOwner


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SIGNATURE_SCHEMA = "perfumery-commercial-release-signature/v1"


@dataclass(frozen=True)
class RegulatorySignoff:
    """Deprecated unsigned evidence shape.

    ``CommercialReleaseStore.record`` persists it only as ``legacy_unverified``
    for migration/audit.  It never passes :meth:`CommercialReleaseStore.assess`.
    """

    formula_id: str
    market_region: str
    approver_role: str
    organization: str
    approved_on: str
    valid_until: str
    report_ref: str
    report_sha256: str


@dataclass(frozen=True)
class VerifiedRegulatorySignoff:
    """An Ed25519-signed approval bound to one :class:`ReleaseSpec`."""

    release_spec_id: str
    market_region: str
    approver_role: str
    organization: str
    signer_id: str
    approved_on: str
    valid_until: str
    report_ref: str
    report_path: str
    report_sha256: str
    signature: str | bytes


@dataclass(frozen=True)
class ReleaseEvidenceAssessment:
    passed: bool
    status: str
    valid_signoffs: int
    missing: tuple[str, ...]
    scope_verified: bool = False
    verification_failures: tuple[str, ...] = ()


def signing_payload(signoff: VerifiedRegulatorySignoff) -> bytes:
    """Canonical bytes a registered signer must sign with Ed25519."""
    return canonical_json(
        {
            "schema": SIGNATURE_SCHEMA,
            "release_spec_id": signoff.release_spec_id,
            "market_region": signoff.market_region.strip().upper(),
            "approver_role": signoff.approver_role.strip(),
            "organization": signoff.organization.strip(),
            "signer_id": signoff.signer_id.strip(),
            "approved_on": signoff.approved_on,
            "valid_until": signoff.valid_until,
            "report_ref": signoff.report_ref.strip(),
            "report_sha256": signoff.report_sha256.strip().lower(),
        }
    )


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _signature_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return base64.b64decode(str(value), validate=True)
    except Exception as error:  # noqa: BLE001 - normalize external input errors
        raise ValueError("signature must be base64-encoded Ed25519 bytes") from error


def _public_key(value: Any):
    """Load a public key without making cryptography a base dependency."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "Ed25519 verification requires the commercial cryptography dependency"
        ) from error

    candidate = value.get("public_key") if isinstance(value, Mapping) else value
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped.startswith("-----BEGIN"):
            key = serialization.load_pem_public_key(stripped.encode("ascii"))
            if not isinstance(key, Ed25519PublicKey):
                raise ValueError("trusted signer key must be Ed25519")
            return key
        try:
            candidate = bytes.fromhex(stripped)
        except ValueError:
            try:
                candidate = base64.b64decode(stripped, validate=True)
            except Exception as error:  # noqa: BLE001
                raise ValueError("trusted signer key must be PEM, hex, or base64") from error
    if not isinstance(candidate, bytes) or len(candidate) != 32:
        raise ValueError("trusted Ed25519 public key must be 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(candidate)


def _signer_policy(
    trusted_signers: Mapping[str, Any], signoff: VerifiedRegulatorySignoff
) -> Any:
    signer_id = _text(signoff.signer_id, "signer_id")
    if signer_id not in trusted_signers:
        raise ValueError(f"signer is not allowlisted: {signer_id}")
    policy = trusted_signers[signer_id]
    if isinstance(policy, Mapping):
        organizations = policy.get("organizations")
        if organizations and signoff.organization not in set(organizations):
            raise ValueError("signer is not allowlisted for this organization")
        roles = policy.get("roles")
        if roles and signoff.approver_role not in set(roles):
            raise ValueError("signer is not allowlisted for this approver role")
    return policy


class CommercialReleaseStore(SQLiteConnectionOwner):
    """SQLite store for signed evidence, rechecked before every release.

    The store owns no signing keys.  Operators provide immutable, allowlisted
    public keys at construction; unsigned data imported through the old API is
    retained for audit but is fail-closed.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        trusted_signers: Mapping[str, Any] | None = None,
    ):
        self.path = str(path)
        self.trusted_signers = dict(trusted_signers or {})
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS verified_regulatory_signoffs (
                release_spec_id TEXT NOT NULL,
                market_region TEXT NOT NULL,
                approver_role TEXT NOT NULL,
                organization TEXT NOT NULL,
                signer_id TEXT NOT NULL,
                approved_on TEXT NOT NULL,
                valid_until TEXT NOT NULL,
                report_ref TEXT NOT NULL,
                report_path TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                signature_b64 TEXT NOT NULL,
                PRIMARY KEY (release_spec_id, market_region, report_sha256, signer_id)
            );
            CREATE TABLE IF NOT EXISTS release_revocations (
                release_spec_id TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                revoked_on TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (release_spec_id, report_sha256)
            );
            CREATE TABLE IF NOT EXISTS legacy_regulatory_signoffs (
                formula_id TEXT NOT NULL,
                market_region TEXT NOT NULL,
                approver_role TEXT NOT NULL,
                organization TEXT NOT NULL,
                approved_on TEXT NOT NULL,
                valid_until TEXT NOT NULL,
                report_ref TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                migration_status TEXT NOT NULL DEFAULT 'legacy_unverified',
                PRIMARY KEY (formula_id, market_region, report_sha256)
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _validate_dates(approved_on: str, valid_until: str) -> tuple[date, date]:
        approved = date.fromisoformat(approved_on)
        expires = date.fromisoformat(valid_until)
        if expires < approved:
            raise ValueError("sign-off validity cannot end before approval")
        return approved, expires

    @staticmethod
    def _validate_id(release_spec_id: str) -> str:
        value = _text(release_spec_id, "release_spec_id")
        if not value.startswith("sha256:") or not _SHA256.fullmatch(value[7:].lower()):
            raise ValueError("release_spec_id must be a SHA-256 release scope ID")
        return value.lower()

    def _verify_signed_evidence(
        self, scope: ReleaseSpec, signoff: VerifiedRegulatorySignoff
    ) -> tuple[str, str]:
        reconstructed = ReleaseSpec.from_payload(scope.payload)
        if reconstructed.release_spec_id != scope.release_spec_id:
            raise ValueError("release scope ID does not match canonical payload")
        if self._validate_id(signoff.release_spec_id) != scope.release_spec_id:
            raise ValueError("sign-off is bound to a different release scope")
        expected_market = str(scope.payload["finished_product"]["market_region"]).upper()
        if _text(signoff.market_region, "market_region").upper() != expected_market:
            raise ValueError("sign-off market does not match the release scope")
        self._validate_dates(signoff.approved_on, signoff.valid_until)
        report_path = Path(_text(signoff.report_path, "report_path")).expanduser().resolve(strict=True)
        if not report_path.is_file():
            raise ValueError("report_path must reference a regular file")
        report_sha = sha256_file(report_path)
        claimed = _text(signoff.report_sha256, "report_sha256").lower()
        if not _SHA256.fullmatch(claimed) or claimed != report_sha:
            raise ValueError("report_sha256 does not match report bytes")
        policy = _signer_policy(self.trusted_signers, signoff)
        key = _public_key(policy)
        try:
            key.verify(_signature_bytes(signoff.signature), signing_payload(signoff))
        except Exception as error:  # noqa: BLE001 - do not expose backend details
            raise ValueError("Ed25519 signature verification failed") from error
        return str(report_path), report_sha

    def record_verified(self, scope: ReleaseSpec, signoff: VerifiedRegulatorySignoff) -> None:
        """Verify and persist real report bytes plus an allowlisted signature."""
        # A scope constructed from opaque hashes is unsuitable for an
        # operational release.  The builder attaches paths and verifies every
        # supplier document again here.
        scope.verify_bound_documents()
        report_path, report_sha = self._verify_signed_evidence(scope, signoff)
        self.connection.execute(
            """INSERT INTO verified_regulatory_signoffs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(release_spec_id, market_region, report_sha256, signer_id)
            DO UPDATE SET approved_on=excluded.approved_on, valid_until=excluded.valid_until,
                report_ref=excluded.report_ref, report_path=excluded.report_path,
                signature_b64=excluded.signature_b64""",
            (
                scope.release_spec_id,
                signoff.market_region.strip().upper(),
                _text(signoff.approver_role, "approver_role"),
                _text(signoff.organization, "organization"),
                _text(signoff.signer_id, "signer_id"),
                signoff.approved_on,
                signoff.valid_until,
                _text(signoff.report_ref, "report_ref"),
                report_path,
                report_sha,
                base64.b64encode(_signature_bytes(signoff.signature)).decode("ascii"),
            ),
        )
        self.connection.commit()

    def record(self, signoff: RegulatorySignoff) -> None:
        """Import legacy evidence as non-approving audit history.

        This deliberately has no path from input strings to a verified release.
        Call :meth:`record_verified` with a signed :class:`ReleaseSpec` instead.
        """
        if not signoff.formula_id.startswith("sha256:"):
            raise ValueError("formula_id must be a SHA-256 fingerprint")
        if not _SHA256.fullmatch(signoff.report_sha256.lower()):
            raise ValueError("report_sha256 must contain 64 hexadecimal characters")
        if not all(
            _text(value, name)
            for value, name in (
                (signoff.market_region, "market_region"),
                (signoff.approver_role, "approver_role"),
                (signoff.organization, "organization"),
                (signoff.report_ref, "report_ref"),
            )
        ):
            raise ValueError("legacy record is incomplete")
        self._validate_dates(signoff.approved_on, signoff.valid_until)
        self.connection.execute(
            "INSERT OR REPLACE INTO legacy_regulatory_signoffs VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'legacy_unverified')",
            (
                signoff.formula_id,
                signoff.market_region.upper(),
                signoff.approver_role,
                signoff.organization,
                signoff.approved_on,
                signoff.valid_until,
                signoff.report_ref,
                signoff.report_sha256.lower(),
            ),
        )
        self.connection.commit()

    def revoke(
        self,
        release_spec_id: str,
        report_sha256: str,
        revoked_on: date,
        reason: str,
    ) -> None:
        scope_id = self._validate_id(release_spec_id)
        report_sha = str(report_sha256).lower()
        if not _SHA256.fullmatch(report_sha):
            raise ValueError("report_sha256 must contain 64 hexadecimal characters")
        self.connection.execute(
            "INSERT OR REPLACE INTO release_revocations VALUES (?, ?, ?, ?)",
            (scope_id, report_sha, revoked_on.isoformat(), _text(reason, "reason")),
        )
        self.connection.commit()

    def _invalid_scope(self, reason: str) -> ReleaseEvidenceAssessment:
        return ReleaseEvidenceAssessment(
            False,
            "release_scope_unverifiable",
            0,
            ("verified canonical commercial release scope",),
            False,
            (reason,),
        )

    def assess_scope(
        self, scope: ReleaseSpec, as_of: date | None = None
    ) -> ReleaseEvidenceAssessment:
        """Re-verify scope documents, report bytes, signature, dates and revocation."""
        as_of = as_of or date.today()
        try:
            reconstructed = ReleaseSpec.from_payload(scope.payload)
            if reconstructed.release_spec_id != scope.release_spec_id:
                return self._invalid_scope("canonical scope ID mismatch")
            scope.verify_bound_documents()
        except (OSError, ValueError) as error:
            return self._invalid_scope(str(error))
        market = str(scope.payload["finished_product"]["market_region"]).upper()
        rows = self.connection.execute(
            """SELECT approver_role, organization, signer_id, approved_on, valid_until,
                       report_ref, report_path, report_sha256, signature_b64
                FROM verified_regulatory_signoffs
                WHERE release_spec_id = ? AND UPPER(market_region) = ?""",
            (scope.release_spec_id, market),
        ).fetchall()
        if not rows:
            return ReleaseEvidenceAssessment(
                False,
                "external_regulatory_signoff_missing",
                0,
                (f"verified external regulatory sign-off for {market}",),
                True,
            )

        valid = 0
        failures: list[str] = []
        revoked = 0
        for row in rows:
            signoff = VerifiedRegulatorySignoff(
                release_spec_id=scope.release_spec_id,
                market_region=market,
                approver_role=str(row[0]),
                organization=str(row[1]),
                signer_id=str(row[2]),
                approved_on=str(row[3]),
                valid_until=str(row[4]),
                report_ref=str(row[5]),
                report_path=str(row[6]),
                report_sha256=str(row[7]),
                signature=str(row[8]),
            )
            try:
                approved, expires = self._validate_dates(signoff.approved_on, signoff.valid_until)
                if approved > as_of:
                    failures.append(f"{signoff.report_ref}: approval is not yet effective")
                    continue
                if expires < as_of:
                    failures.append(f"{signoff.report_ref}: approval has expired")
                    continue
                is_revoked = self.connection.execute(
                    """SELECT revoked_on FROM release_revocations
                    WHERE release_spec_id = ? AND report_sha256 = ?""",
                    (scope.release_spec_id, signoff.report_sha256),
                ).fetchone()
                if is_revoked and date.fromisoformat(str(is_revoked[0])) <= as_of:
                    revoked += 1
                    failures.append(f"{signoff.report_ref}: approval has been revoked")
                    continue
                self._verify_signed_evidence(scope, signoff)
                valid += 1
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(f"{signoff.report_ref}: {error}")
        if valid:
            return ReleaseEvidenceAssessment(
                True,
                "verified_external_regulatory_signoff_valid",
                valid,
                (),
                True,
                tuple(failures),
            )
        status = "external_regulatory_signoff_revoked" if revoked == len(rows) else "external_regulatory_signoff_invalid"
        return ReleaseEvidenceAssessment(
            False,
            status,
            0,
            (f"current verified external regulatory sign-off for {market}",),
            True,
            tuple(failures),
        )

    def assess(
        self, formula_or_scope: str | ReleaseSpec, market_region: str | None = None,
        as_of: date | None = None,
    ) -> ReleaseEvidenceAssessment:
        """Assess a scope; raw formula fingerprints are intentionally fail-closed."""
        if isinstance(formula_or_scope, ReleaseSpec):
            return self.assess_scope(formula_or_scope, as_of)
        # A legacy string has neither supplier-lot scope nor files to re-hash.
        # It is never an approval, even if a legacy record exists in SQLite.
        return ReleaseEvidenceAssessment(
            False,
            "legacy_formula_fingerprint_not_approvable",
            0,
            ("verified canonical commercial release scope",),
            False,
            ("raw formula fingerprints and unsigned legacy records cannot approve release",),
        )

    def close(self) -> None:
        self.connection.close()
