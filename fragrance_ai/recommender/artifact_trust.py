"""Fail-closed trust verification for externally produced evidence artifacts.

This module is deliberately independent from the evidence stores.  A database
row is only an index to evidence; it is never evidence by itself.  Every use
of an artifact verifies the bytes again, verifies the Ed25519 signature against
an operator-provided allowlist, and checks the intended signer role, artifact
type, exact scope, validity window, and revocation state.

``cryptography`` is imported only when a verification is requested so the
research-only installation remains usable without commercial dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_SIGNATURE_SCHEMA = "perfumery-evidence-artifact-signature/v1"


def canonical_json(value: Any) -> bytes:
    """Return unambiguous canonical bytes for hashing and signing."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Hash real artifact bytes; paths, URLs and claimed hashes are not proof."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("evidence artifact path must be a regular file")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _parse_timestamp(value: Any, name: str) -> datetime:
    text = _required_text(value, name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _as_of_datetime(as_of: date | datetime | None) -> datetime:
    if as_of is None:
        return datetime.now(timezone.utc)
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            raise ValueError("as_of datetime must include a timezone")
        return as_of.astimezone(timezone.utc)
    return datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)


def _signature_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return base64.b64decode(_required_text(value, "signature"), validate=True)
    except Exception as error:  # noqa: BLE001 - untrusted external input
        raise ValueError("signature must be base64-encoded Ed25519 bytes") from error


def _public_key(value: Any):
    """Load an Ed25519 public key without a module-level cryptography import."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:  # pragma: no cover - optional commercial extra
        raise RuntimeError(
            "Ed25519 evidence verification requires the commercial cryptography dependency"
        ) from error

    candidate = value.get("public_key") if isinstance(value, Mapping) else value
    if isinstance(candidate, str):
        encoded = candidate.strip()
        if encoded.startswith("-----BEGIN"):
            key = serialization.load_pem_public_key(encoded.encode("ascii"))
            if not isinstance(key, Ed25519PublicKey):
                raise ValueError("trusted evidence signer key must be Ed25519")
            return key
        try:
            candidate = bytes.fromhex(encoded)
        except ValueError:
            try:
                candidate = base64.b64decode(encoded, validate=True)
            except Exception as error:  # noqa: BLE001
                raise ValueError("trusted signer key must be PEM, hex, or base64") from error
    if not isinstance(candidate, bytes) or len(candidate) != 32:
        raise ValueError("trusted Ed25519 public key must be 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(candidate)


def signing_payload(envelope: Mapping[str, Any]) -> bytes:
    """Return the exact envelope bytes which must be signed.

    The detached signature itself is intentionally excluded.  Artifact hashes
    are part of the payload, so an unchanged signature cannot be replayed for
    different bytes or a different release/study/formula scope.
    """
    unsigned = dict(envelope)
    unsigned.pop("signature", None)
    return canonical_json(unsigned)


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    artifact_type: str
    signer_id: str
    signer_role: str
    signer_key_sha256: str
    scope: dict[str, Any]
    issued_at: str
    expires_at: str
    artifact_hashes: dict[str, str]


class EvidenceTrustRoot:
    """Independent signer allowlist and revocation policy.

    The constructor accepts a mapping to make deployment configuration simple:

    ``{"signers": {"qa-lab": {"public_key": "...", "roles": [...],
    "artifact_types": [...], "scope_constraints": {...}}},
    "revoked_signers": {"old-key": "2026-01-01"},
    "revoked_artifacts": {"artifact-id": "2026-01-01"}}``.

    Empty or omitted policy is intentional: it trusts nobody and therefore
    fails closed.  ``scope_constraints`` is optional, but an exact caller
    supplied scope is always required at verification time.
    """

    def __init__(self, policy: Mapping[str, Any] | None = None):
        policy = dict(policy or {})
        raw_signers = (
            policy["signers"]
            if "signers" in policy
            else {
                key: value
                for key, value in policy.items()
                if key not in {"revoked_signers", "revoked_artifacts"}
            }
        )
        if not isinstance(raw_signers, Mapping):
            raise ValueError("evidence trust root signers must be a mapping")
        self.signers: dict[str, Any] = {str(key): value for key, value in raw_signers.items()}
        self.revoked_signers = self._normalise_revocations(policy.get("revoked_signers", {}))
        self.revoked_artifacts = self._normalise_revocations(policy.get("revoked_artifacts", {}))

    @staticmethod
    def _normalise_revocations(value: Any) -> dict[str, date]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("revocation policy must be a mapping of ID to ISO date")
        result: dict[str, date] = {}
        for identifier, revoked_on in value.items():
            result[_required_text(identifier, "revocation identifier")] = date.fromisoformat(
                _required_text(revoked_on, "revoked_on")
            )
        return result

    @classmethod
    def from_json_file(cls, path: str | Path) -> "EvidenceTrustRoot":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("evidence trust-root file must contain an object")
        return cls(payload)

    @staticmethod
    def _policy_values(policy: Mapping[str, Any], name: str) -> set[str]:
        values = policy.get(name)
        if values is None:
            return set()
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ValueError(f"signer policy {name} must be a list")
        return {_required_text(value, name) for value in values}

    @staticmethod
    def _scope_matches_constraints(scope: Mapping[str, Any], constraints: Any) -> bool:
        if constraints is None:
            return True
        if not isinstance(constraints, Mapping):
            raise ValueError("signer scope_constraints must be an object")
        for key, allowed in constraints.items():
            if key not in scope:
                return False
            values = allowed if isinstance(allowed, (list, tuple, set, frozenset)) else [allowed]
            if "*" not in values and scope[key] not in values:
                return False
        return True

    def verify(
        self,
        envelope: Mapping[str, Any],
        artifact_paths: Mapping[str, str | Path],
        *,
        expected_artifact_type: str,
        expected_scope: Mapping[str, Any],
        allowed_roles: set[str] | frozenset[str] | None = None,
        as_of: date | datetime | None = None,
        local_revocations: Mapping[str, date] | None = None,
    ) -> VerifiedArtifact:
        """Verify bytes, signature, allowlist policy, scope and validity.

        Raises ``ValueError`` (or ``RuntimeError`` if the optional verifier is
        unavailable).  Callers must treat every exception as non-evidence.
        """
        if not isinstance(envelope, Mapping):
            raise ValueError("signed evidence envelope must be an object")
        if envelope.get("schema") != ARTIFACT_SIGNATURE_SCHEMA:
            raise ValueError("unsupported evidence signature schema")
        artifact_type = _required_text(envelope.get("artifact_type"), "artifact_type")
        if artifact_type != _required_text(expected_artifact_type, "expected_artifact_type"):
            raise ValueError("evidence artifact type does not match required type")
        artifact_id = _required_text(envelope.get("artifact_id"), "artifact_id")
        signer_id = _required_text(envelope.get("signer_id"), "signer_id")
        signer_role = _required_text(envelope.get("signer_role"), "signer_role")
        scope = envelope.get("scope")
        if not isinstance(scope, Mapping) or not scope:
            raise ValueError("signed evidence scope must be a nonempty object")
        expected_scope = dict(expected_scope)
        for key, value in expected_scope.items():
            if scope.get(key) != value:
                raise ValueError(f"signed evidence scope mismatch for {key}")
        hashes = envelope.get("artifact_hashes")
        if not isinstance(hashes, Mapping) or set(hashes) != set(artifact_paths):
            raise ValueError("signed evidence artifact labels do not match supplied files")
        normalised_hashes: dict[str, str] = {}
        for label, claimed_hash in hashes.items():
            claimed = _required_text(claimed_hash, f"artifact_hashes.{label}").lower()
            if len(claimed) != 64 or any(char not in "0123456789abcdef" for char in claimed):
                raise ValueError("signed evidence artifact hash must be SHA-256 hexadecimal")
            actual = sha256_file(artifact_paths[label])
            if actual != claimed:
                raise ValueError(f"evidence artifact bytes changed: {label}")
            normalised_hashes[str(label)] = claimed

        issued = _parse_timestamp(envelope.get("issued_at"), "issued_at")
        expires = _parse_timestamp(envelope.get("expires_at"), "expires_at")
        moment = _as_of_datetime(as_of)
        if expires <= issued:
            raise ValueError("evidence expiration must be after issuance")
        if issued > moment:
            raise ValueError("evidence is not yet effective")
        if expires < moment:
            raise ValueError("evidence has expired")
        if signer_id not in self.signers:
            raise ValueError("evidence signer is not in the independent allowlist")
        if self.revoked_signers.get(signer_id) and self.revoked_signers[signer_id] <= moment.date():
            raise ValueError("evidence signer has been revoked")
        revocations = dict(self.revoked_artifacts)
        revocations.update(dict(local_revocations or {}))
        if revocations.get(artifact_id) and revocations[artifact_id] <= moment.date():
            raise ValueError("evidence artifact has been revoked")
        policy = self.signers[signer_id]
        if not isinstance(policy, Mapping):
            policy = {"public_key": policy}
        policy_roles = self._policy_values(policy, "roles")
        if not policy_roles or signer_role not in policy_roles:
            raise ValueError("signer is not allowlisted for the claimed evidence role")
        if allowed_roles is not None and signer_role not in set(allowed_roles):
            raise ValueError("signer role is not acceptable for this evidence class")
        policy_types = self._policy_values(policy, "artifact_types")
        if not policy_types or artifact_type not in policy_types:
            raise ValueError("signer is not allowlisted for this artifact type")
        if not self._scope_matches_constraints(scope, policy.get("scope_constraints")):
            raise ValueError("signer is not allowlisted for this evidence scope")
        key = _public_key(policy)
        try:
            key.verify(_signature_bytes(envelope.get("signature")), signing_payload(envelope))
        except Exception as error:  # noqa: BLE001 - do not leak backend detail
            raise ValueError("Ed25519 evidence signature verification failed") from error
        from cryptography.hazmat.primitives import serialization

        signer_key_sha256 = hashlib.sha256(
            key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        ).hexdigest()
        return VerifiedArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            signer_id=signer_id,
            signer_role=signer_role,
            signer_key_sha256=signer_key_sha256,
            scope=dict(scope),
            issued_at=issued.isoformat(),
            expires_at=expires.isoformat(),
            artifact_hashes=normalised_hashes,
        )
