"""Fail-closed champion/challenger governance for continual model improvement.

The controller intentionally separates three things which are often conflated:

* training may create any number of research challengers;
* a prospective, source/molecule/scaffold-cold evaluation may promote a shadow;
* production scoring changes only after an independent Ed25519 authorization.

Synthetic scores, simulator outputs and the model's own predictions can never
be used as labels.  Candidate bundles are immutable, hash-bound directories;
the mutable registry is atomic and every transition is written to a hash-chain
audit log.  The module has no network client and never downloads outcomes, so a
model process cannot quietly inspect a future test set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    EvidenceTrustRoot,
    canonical_json,
    sha256_file,
)
from .concentration_response import (
    ConcentrationResponseTrustDecision,
    FrozenConcentrationResponse,
)


CANDIDATE_SCHEMA = "perfumery-continuous-candidate/v1"
EVALUATION_SCHEMA = "perfumery-continuous-evaluation/v1"
DATASET_RECEIPT_SCHEMA = "perfumery-external-dataset-receipt/v1"
STATE_SCHEMA = "perfumery-continuous-registry/v1"
AUDIT_SCHEMA = "perfumery-continuous-audit/v1"
SUPPORTED_MODEL_FAMILY = "concentration_response"
PROSPECTIVE_DATASET_ACQUISITION_ARTIFACT_TYPE = "prospective_dataset_acquisition"
REQUIRED_ARTIFACTS = frozenset(
    {
        "runtime",
        "model_manifest",
        "training_data",
        "challenge_inputs",
        "acquisition_authorization",
        "evaluation_report",
        "dataset_receipt",
        "predictions",
        "prediction_seal",
        "outcomes",
        "timestamp_response",
    }
)
UNSAFE_SUFFIXES = frozenset({".joblib", ".pickle", ".pkl", ".pt", ".pth"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
BOOTSTRAP_SEED_DOMAIN = b"perfumery-ai-source-target-bootstrap-v1\x00"
TRAINING_COLUMNS = frozenset(
    {
        "row_id",
        "source_id",
        "target_id",
        "molecule_id",
        "scaffold_id",
        "dilution_fraction",
        "intensity",
        "label_origin",
        "evidence_class",
    }
)
CHALLENGE_COLUMNS = frozenset(
    {
        "row_id",
        "source_id",
        "target_id",
        "molecule_id",
        "scaffold_id",
        "dilution_fraction",
    }
)
OUTCOME_COLUMNS = frozenset({"row_id", "intensity"})


class ContinualImprovementError(RuntimeError):
    """A candidate or registry failed a fail-closed contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_finite_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContinualImprovementError(f"invalid JSON artifact {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ContinualImprovementError(f"JSON artifact must contain an object: {path.name}")
    return value


def _required_text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ContinualImprovementError(f"{name} is required")
    return result


def _sha256_hex(value: Any, name: str) -> str:
    result = _required_text(value, name).lower()
    if not SHA256_RE.fullmatch(result):
        raise ContinualImprovementError(f"{name} must be lowercase SHA-256 hexadecimal")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ContinualImprovementError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContinualImprovementError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ContinualImprovementError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ContinualImprovementError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ContinualImprovementError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ContinualImprovementError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ContinualImprovementError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ContinualImprovementError(f"{name} must be a nonnegative integer") from error
    if result < 0 or result != value:
        raise ContinualImprovementError(f"{name} must be a nonnegative integer")
    return result


def bootstrap_seed(prediction_sha256: str) -> int:
    digest = _sha256_hex(prediction_sha256, "prediction_sha256")
    return int.from_bytes(
        hashlib.sha256(BOOTSTRAP_SEED_DOMAIN + bytes.fromhex(digest)).digest()[:4],
        "big",
    )


def _strict_csv_rows(
    path: Path,
    columns: frozenset[str],
    name: str,
    *,
    maximum_rows: int,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != columns:
            raise ContinualImprovementError(
                f"{name} columns differ from the frozen schema"
            )
        rows = []
        for row in reader:
            if len(rows) >= maximum_rows:
                raise ContinualImprovementError(
                    f"{name} exceeds the policy row limit"
                )
            rows.append(dict(row))
    if not rows:
        raise ContinualImprovementError(f"{name} contains no rows")
    identifiers = [_required_text(row.get("row_id"), f"{name}.row_id") for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ContinualImprovementError(f"{name} row IDs are duplicated")
    return rows


def _parse_time(value: Any, name: str) -> datetime:
    text = _required_text(value, name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise ContinualImprovementError(f"{name} must be ISO-8601") from error
    if result.tzinfo is None:
        raise ContinualImprovementError(f"{name} must include a timezone")
    return result.astimezone(timezone.utc)


def _string_set(value: Any, name: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ContinualImprovementError(f"{name} must be a nonempty list")
    if any(not isinstance(item, str) for item in value):
        raise ContinualImprovementError(f"{name} values must be strings")
    result = {_required_text(item, name) for item in value}
    if len(result) != len(value):
        raise ContinualImprovementError(f"{name} must not contain duplicates")
    return result


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ContinuousImprovementPolicy:
    policy_id: str
    document_sha256: str
    allowed_evidence_class: str
    blocked_label_origins: frozenset[str]
    required_runtime_format: str
    required_runtime_engine: str
    minimum_external_sources: int
    minimum_evaluation_targets: int
    minimum_evaluation_rows: int
    maximum_evaluation_rows: int
    maximum_training_rows: int
    minimum_bootstrap_draws: int
    maximum_bootstrap_draws: int
    minimum_mae_gain_lower: float
    minimum_rank_gain_lower: float
    maximum_portable_parity_error: float
    maximum_primary_score_weight: float
    maximum_runtime_bytes: int
    maximum_evidence_artifact_bytes: int
    maximum_candidate_bundle_bytes: int
    lock_stale_after_seconds: int
    required_scientific_gates: tuple[str, ...]
    authorization_artifact_type: str
    allowed_signer_roles: frozenset[str]
    acquisition_artifact_type: str
    acquisition_signer_roles: frozenset[str]
    require_distinct_production_signer: bool

    @classmethod
    def load_builtin(cls) -> "ContinuousImprovementPolicy":
        ref = (
            resources.files("fragrance_ai")
            .joinpath("data")
            .joinpath("continuous_improvement_policy.json")
        )
        return cls.from_mapping(json.loads(ref.read_text(encoding="utf-8")))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ContinuousImprovementPolicy":
        return cls.from_mapping(_read_json(Path(path).expanduser().resolve(strict=True)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContinuousImprovementPolicy":
        if str(value.get("schema_version")) != "1.0":
            raise ContinualImprovementError("unsupported continual-improvement policy")
        production = value.get("production_contract")
        if not isinstance(production, Mapping):
            raise ContinualImprovementError("production_contract must be an object")
        if production.get("signed_production_authorization_required") is not True:
            raise ContinualImprovementError("production authorization must fail closed")
        if production.get("unsafe_serialization_allowed") is not False:
            raise ContinualImprovementError("unsafe model serialization cannot be enabled")
        if production.get("human_olfactory_90_percent_certified") is not False:
            raise ContinualImprovementError("policy cannot manufacture a human-90 claim")
        acquisition = value.get("external_evidence_contract")
        if not isinstance(acquisition, Mapping):
            raise ContinualImprovementError(
                "external_evidence_contract must be an object"
            )
        if acquisition.get("signed_acquisition_required") is not True:
            raise ContinualImprovementError("external acquisition must be signed")
        if (
            acquisition.get("production_signer_must_differ_from_acquisition_signer")
            is not True
        ):
            raise ContinualImprovementError(
                "production and acquisition signers must be separated"
            )
        gates = value.get("required_scientific_gates")
        if not isinstance(gates, list) or not gates:
            raise ContinualImprovementError("required_scientific_gates must be nonempty")
        policy = cls(
            policy_id=_required_text(value.get("policy_id"), "policy_id"),
            document_sha256=_hash_bytes(canonical_json(dict(value))),
            allowed_evidence_class=_required_text(
                value.get("allowed_evidence_class"), "allowed_evidence_class"
            ),
            blocked_label_origins=frozenset(
                _string_set(value.get("blocked_label_origins"), "blocked_label_origins")
            ),
            required_runtime_format=_required_text(
                value.get("required_runtime_format"), "required_runtime_format"
            ),
            required_runtime_engine=_required_text(
                value.get("required_runtime_engine"), "required_runtime_engine"
            ),
            minimum_external_sources=_positive_int(
                value.get("minimum_external_sources"), "minimum_external_sources"
            ),
            minimum_evaluation_targets=_positive_int(
                value.get("minimum_evaluation_targets"), "minimum_evaluation_targets"
            ),
            minimum_evaluation_rows=_positive_int(
                value.get("minimum_evaluation_rows"), "minimum_evaluation_rows"
            ),
            maximum_evaluation_rows=_positive_int(
                value.get("maximum_evaluation_rows"), "maximum_evaluation_rows"
            ),
            maximum_training_rows=_positive_int(
                value.get("maximum_training_rows"), "maximum_training_rows"
            ),
            minimum_bootstrap_draws=_positive_int(
                value.get("minimum_bootstrap_draws"), "minimum_bootstrap_draws"
            ),
            maximum_bootstrap_draws=_positive_int(
                value.get("maximum_bootstrap_draws"), "maximum_bootstrap_draws"
            ),
            minimum_mae_gain_lower=_finite_float(
                value.get("minimum_baseline_minus_candidate_mae_bootstrap_lower"),
                "minimum_baseline_minus_candidate_mae_bootstrap_lower",
            ),
            minimum_rank_gain_lower=_finite_float(
                value.get("minimum_candidate_minus_baseline_spearman_bootstrap_lower"),
                "minimum_candidate_minus_baseline_spearman_bootstrap_lower",
            ),
            maximum_portable_parity_error=_finite_float(
                value.get("maximum_portable_parity_absolute_error"),
                "maximum_portable_parity_absolute_error",
            ),
            maximum_primary_score_weight=_finite_float(
                value.get("maximum_primary_score_weight"),
                "maximum_primary_score_weight",
            ),
            maximum_runtime_bytes=_positive_int(
                value.get("maximum_runtime_bytes"), "maximum_runtime_bytes"
            ),
            maximum_evidence_artifact_bytes=_positive_int(
                value.get("maximum_evidence_artifact_bytes"),
                "maximum_evidence_artifact_bytes",
            ),
            maximum_candidate_bundle_bytes=_positive_int(
                value.get("maximum_candidate_bundle_bytes"),
                "maximum_candidate_bundle_bytes",
            ),
            lock_stale_after_seconds=_positive_int(
                value.get("lock_stale_after_seconds"), "lock_stale_after_seconds"
            ),
            required_scientific_gates=tuple(_required_text(item, "gate") for item in gates),
            authorization_artifact_type=_required_text(
                production.get("authorization_artifact_type"),
                "authorization_artifact_type",
            ),
            allowed_signer_roles=frozenset(
                _string_set(production.get("allowed_signer_roles"), "allowed_signer_roles")
            ),
            acquisition_artifact_type=_required_text(
                acquisition.get("authorization_artifact_type"),
                "external_evidence_contract.authorization_artifact_type",
            ),
            acquisition_signer_roles=frozenset(
                _string_set(
                    acquisition.get("allowed_signer_roles"),
                    "external_evidence_contract.allowed_signer_roles",
                )
            ),
            require_distinct_production_signer=True,
        )
        if policy.maximum_evaluation_rows < policy.minimum_evaluation_rows:
            raise ContinualImprovementError(
                "maximum_evaluation_rows is below the minimum"
            )
        if policy.maximum_bootstrap_draws < policy.minimum_bootstrap_draws:
            raise ContinualImprovementError(
                "maximum_bootstrap_draws is below the minimum"
            )
        return policy


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    manifest_sha256: str
    status: str
    scientific_gate_passed: bool
    shadow_promoted: bool
    production_promoted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    model_family: str = SUPPORTED_MODEL_FAMILY


class _ExclusiveFileLock:
    def __init__(self, path: Path, stale_after_seconds: int):
        self.path = path
        self.stale_after_seconds = stale_after_seconds
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "_ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age <= self.stale_after_seconds:
                    raise ContinualImprovementError("continual-improvement controller is locked")
                stale = self.path.with_name(self.path.name + f".stale-{uuid.uuid4().hex}")
                try:
                    os.replace(self.path, stale)
                    stale.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                continue
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "token": self.token,
                            "acquired_at": _utc_now(),
                        },
                        handle,
                        sort_keys=True,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                self.path.unlink(missing_ok=True)
                raise
            return self
        raise ContinualImprovementError("could not acquire continual-improvement lock")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            value = _read_json(self.path)
        except (ContinualImprovementError, FileNotFoundError):
            return
        if value.get("token") == self.token:
            self.path.unlink(missing_ok=True)


class ContinuousImprovementController:
    """Evaluate immutable candidate bundles and atomically maintain champions."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: ContinuousImprovementPolicy | None = None,
        trust_root: EvidenceTrustRoot | Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.inbox = self.root / "inbox"
        self.state_path = self.root / "registry.json"
        self.audit_path = self.root / "audit.jsonl"
        self.pending_path = self.root / ".registry.pending.json"
        self.lock_path = self.root / ".controller.lock"
        self.policy = policy or ContinuousImprovementPolicy.load_builtin()
        self.trust_root = (
            trust_root
            if isinstance(trust_root, EvidenceTrustRoot)
            else EvidenceTrustRoot(trust_root)
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _ExclusiveFileLock(self.lock_path, self.policy.lock_stale_after_seconds):
            yield

    def _initial_state(self) -> dict[str, Any]:
        bundled_manifest = (
            resources.files("fragrance_ai")
            .joinpath("data")
            .joinpath("concentration_response_manifest.json")
        )
        bundled_manifest_bytes = bundled_manifest.read_bytes()
        bundled_contract = json.loads(bundled_manifest_bytes.decode("utf-8"))
        bundled_runtime_bytes = (
            resources.files("fragrance_ai")
            .joinpath("data")
            .joinpath(str(bundled_contract["runtime_file"]))
            .read_bytes()
        )
        return {
            "schema": STATE_SCHEMA,
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.document_sha256,
            "sequence": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "champions": {
                SUPPORTED_MODEL_FAMILY: {
                    "production": {
                        "candidate_id": "bundled_concentration_response_v1",
                        "status": "bundled_diagnostic_only",
                        "approved_primary_score_weight": 0.0,
                        "authorization_verified": False,
                        "model_manifest_sha256": _hash_bytes(
                            bundled_manifest_bytes
                        ),
                        "runtime_sha256": _hash_bytes(bundled_runtime_bytes),
                    },
                    "shadow": None,
                }
            },
            "processed": {},
            "evaluation_ledger": {
                "datasets": {},
                "outcomes": {},
                "rows": {},
            },
            "last_audit_hash": "",
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        state = _read_json(self.state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise ContinualImprovementError("unsupported continual registry schema")
        if state.get("policy_id") != self.policy.policy_id:
            raise ContinualImprovementError("registry policy does not match active policy")
        if state.get("policy_sha256") != self.policy.document_sha256:
            raise ContinualImprovementError(
                "registry policy hash does not match active policy"
            )
        if not isinstance(state.get("processed"), dict):
            raise ContinualImprovementError("registry processed index is invalid")
        ledger = state.get("evaluation_ledger")
        if not isinstance(ledger, dict) or any(
            not isinstance(ledger.get(key), dict)
            for key in ("datasets", "outcomes", "rows")
        ):
            raise ContinualImprovementError("registry evaluation ledger is invalid")
        return state

    @staticmethod
    def _state_commit_hash(state: Mapping[str, Any]) -> str:
        core = dict(state)
        core.pop("last_audit_hash", None)
        return _hash_bytes(canonical_json(core))

    def _read_audit(self) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        events = []
        with self.audit_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line, parse_constant=_finite_json_constant)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ContinualImprovementError(
                        f"audit line {line_number} is invalid"
                    ) from error
                if not isinstance(event, dict):
                    raise ContinualImprovementError(f"audit line {line_number} is not an object")
                events.append(event)
        return events

    def verify_registry(self) -> dict[str, Any]:
        state = self._load_state()
        events = self._read_audit()
        previous = ""
        for index, event in enumerate(events, 1):
            if event.get("schema") != AUDIT_SCHEMA or event.get("previous_hash") != previous:
                raise ContinualImprovementError(f"audit chain broke at event {index}")
            unsigned = dict(event)
            claimed = _sha256_hex(unsigned.pop("event_hash", ""), "event_hash")
            actual = _hash_bytes(canonical_json(unsigned))
            if actual != claimed:
                raise ContinualImprovementError(f"audit event {index} hash mismatch")
            previous = claimed
        if events:
            if state.get("last_audit_hash") != previous:
                raise ContinualImprovementError("registry is not bound to the audit head")
            if events[-1].get("state_commit_sha256") != self._state_commit_hash(state):
                raise ContinualImprovementError("registry state differs from its audit commit")
        elif state.get("last_audit_hash"):
            raise ContinualImprovementError("registry names a missing audit head")
        self._verify_champion_artifacts(state)
        return {
            "valid": True,
            "event_count": len(events),
            "audit_head": previous,
            "state_commit_sha256": self._state_commit_hash(state),
        }

    def _verify_champion_artifacts(self, state: Mapping[str, Any]) -> None:
        family = state.get("champions", {}).get(SUPPORTED_MODEL_FAMILY)
        if not isinstance(family, Mapping):
            raise ContinualImprovementError("concentration-response champions are missing")
        for tier in ("production", "shadow"):
            record = family.get(tier)
            if record is None:
                continue
            if not isinstance(record, Mapping):
                raise ContinualImprovementError(f"{tier} champion record is invalid")
            if record.get("candidate_id") == "bundled_concentration_response_v1":
                initial = self._initial_state()["champions"][SUPPORTED_MODEL_FAMILY][
                    "production"
                ]
                if (
                    record.get("model_manifest_sha256")
                    != initial["model_manifest_sha256"]
                    or record.get("runtime_sha256") != initial["runtime_sha256"]
                    or float(record.get("approved_primary_score_weight", -1.0))
                    != 0.0
                ):
                    raise ContinualImprovementError(
                        "bundled concentration-response champion changed"
                    )
                continue
            manifest_path = Path(
                _required_text(record.get("manifest_path"), f"{tier}.manifest_path")
            ).resolve(strict=True)
            if sha256_file(manifest_path) != _sha256_hex(
                record.get("manifest_sha256"), f"{tier}.manifest_sha256"
            ):
                raise ContinualImprovementError(f"{tier} candidate manifest changed")
            candidate = _read_json(manifest_path)
            paths = self._validate_artifacts(manifest_path, candidate)
            for label, state_key in (
                ("model_manifest", "model_manifest_sha256"),
                ("runtime", "runtime_sha256"),
                ("evaluation_report", "evaluation_report_sha256"),
                ("dataset_receipt", "dataset_receipt_sha256"),
            ):
                if sha256_file(paths[label]) != _sha256_hex(
                    record.get(state_key), f"{tier}.{state_key}"
                ):
                    raise ContinualImprovementError(
                        f"{tier} state differs from candidate artifact: {label}"
                    )
            if record.get("authorization_verified") is True:
                authorization_path = Path(
                    _required_text(
                        record.get("authorization_path"),
                        f"{tier}.authorization_path",
                    )
                ).resolve(strict=True)
                if sha256_file(authorization_path) != _sha256_hex(
                    record.get("authorization_sha256"),
                    f"{tier}.authorization_sha256",
                ):
                    raise ContinualImprovementError(
                        f"{tier} production authorization changed"
                    )

    def _recover_pending(self) -> None:
        if not self.pending_path.exists():
            return
        pending = _read_json(self.pending_path)
        events = self._read_audit()
        if (
            events
            and pending.get("last_audit_hash") == events[-1].get("event_hash")
            and events[-1].get("state_commit_sha256") == self._state_commit_hash(pending)
        ):
            os.replace(self.pending_path, self.state_path)
        else:
            self.pending_path.unlink(missing_ok=True)

    def _commit(self, state: dict[str, Any], decisions: list[CandidateAssessment]) -> None:
        events = self._read_audit()
        previous = str(events[-1].get("event_hash", "")) if events else ""
        state["sequence"] = int(state.get("sequence", 0)) + 1
        state["updated_at"] = _utc_now()
        event = {
            "schema": AUDIT_SCHEMA,
            "event_id": uuid.uuid4().hex,
            "occurred_at": _utc_now(),
            "previous_hash": previous,
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.document_sha256,
            "registry_sequence": state["sequence"],
            "decisions": [asdict(item) for item in decisions],
            "state_commit_sha256": self._state_commit_hash(state),
        }
        event["event_hash"] = _hash_bytes(canonical_json(event))
        state["last_audit_hash"] = event["event_hash"]
        _atomic_write_json(self.pending_path, state)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("ab") as handle:
            handle.write(canonical_json(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.pending_path, self.state_path)

    @staticmethod
    def _artifact_path(bundle_root: Path, record: Mapping[str, Any], label: str) -> Path:
        relative_text = _required_text(record.get("path"), f"artifacts.{label}.path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ContinualImprovementError(f"artifact path escapes candidate bundle: {label}")
        try:
            resolved = (bundle_root / relative).resolve(strict=True)
        except OSError as error:
            raise ContinualImprovementError(f"missing candidate artifact: {label}") from error
        try:
            resolved.relative_to(bundle_root)
        except ValueError as error:
            raise ContinualImprovementError(f"artifact path escapes candidate bundle: {label}") from error
        if not resolved.is_file():
            raise ContinualImprovementError(f"candidate artifact is not a file: {label}")
        if resolved.suffix.lower() in UNSAFE_SUFFIXES:
            raise ContinualImprovementError(f"unsafe serialized artifact is forbidden: {label}")
        expected_hash = _sha256_hex(record.get("sha256"), f"artifacts.{label}.sha256")
        actual_hash = sha256_file(resolved)
        if actual_hash != expected_hash:
            raise ContinualImprovementError(f"candidate artifact hash mismatch: {label}")
        expected_bytes = _positive_int(record.get("bytes"), f"artifacts.{label}.bytes")
        if resolved.stat().st_size != expected_bytes:
            raise ContinualImprovementError(f"candidate artifact byte-size mismatch: {label}")
        return resolved

    def _validate_artifacts(
        self, manifest_path: Path, candidate: Mapping[str, Any]
    ) -> dict[str, Path]:
        raw = candidate.get("artifacts")
        if not isinstance(raw, Mapping) or set(raw) != REQUIRED_ARTIFACTS:
            raise ContinualImprovementError(
                "candidate artifacts must contain exactly: "
                + ", ".join(sorted(REQUIRED_ARTIFACTS))
            )
        bundle_root = manifest_path.parent.resolve(strict=True)
        paths: dict[str, Path] = {}
        seen: set[Path] = set()
        total_bytes = 0
        for label in sorted(REQUIRED_ARTIFACTS):
            record = raw[label]
            if not isinstance(record, Mapping):
                raise ContinualImprovementError(f"artifacts.{label} must be an object")
            path = self._artifact_path(bundle_root, record, label)
            if path in seen:
                raise ContinualImprovementError("candidate artifact labels must name distinct files")
            paths[label] = path
            seen.add(path)
            size = path.stat().st_size
            if size > self.policy.maximum_evidence_artifact_bytes:
                raise ContinualImprovementError(
                    f"candidate artifact exceeds policy size limit: {label}"
                )
            total_bytes += size
        if total_bytes > self.policy.maximum_candidate_bundle_bytes:
            raise ContinualImprovementError("candidate bundle exceeds policy size limit")
        if paths["runtime"].suffix.lower() != ".json":
            raise ContinualImprovementError("runtime must be portable JSON")
        if paths["runtime"].stat().st_size > self.policy.maximum_runtime_bytes:
            raise ContinualImprovementError("runtime exceeds policy size limit")
        return paths

    def _validate_runtime(self, paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = _read_json(paths["runtime"])
        manifest = _read_json(paths["model_manifest"])
        if runtime.get("runtime") != self.policy.required_runtime_engine:
            raise ContinualImprovementError("candidate runtime engine is unsupported")
        if runtime.get("format") != self.policy.required_runtime_format:
            raise ContinualImprovementError("candidate runtime format is unsupported")
        if runtime.get("allow_pickle") is not False:
            raise ContinualImprovementError("candidate runtime permits unsafe deserialization")
        if runtime.get("source_training_artifact_required_at_runtime") is not False:
            raise ContinualImprovementError("candidate requires a training artifact at runtime")
        if manifest.get("runtime_sha256") != sha256_file(paths["runtime"]):
            raise ContinualImprovementError("model manifest does not bind runtime bytes")
        if manifest.get("runtime_file") != paths["runtime"].name:
            raise ContinualImprovementError("model manifest runtime filename mismatch")
        release = manifest.get("release_gate")
        if not isinstance(release, Mapping) or release.get("passed") is not True:
            raise ContinualImprovementError("candidate model release gate is closed")
        checks = release.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
            raise ContinualImprovementError("candidate model release checks are incomplete")
        probe = FrozenConcentrationResponse(
            manifest_path=paths["model_manifest"], runtime_path=paths["runtime"]
        )
        midpoint = math.sqrt(
            float(runtime["dilution_range_fraction"][0])
            * float(runtime["dilution_range_fraction"][1])
        )
        value, in_domain = probe.intensity(midpoint)
        if not in_domain or not math.isfinite(value):
            raise ContinualImprovementError("portable runtime failed an executable probe")
        minimum, maximum = (
            float(value) for value in runtime["dilution_range_fraction"]
        )
        monotone_values = [
            probe.intensity(float(dilution))[0]
            for dilution in np.geomspace(minimum, maximum, 257)
        ]
        if np.any(np.diff(np.asarray(monotone_values, dtype=float)) < -1e-9):
            raise ContinualImprovementError(
                "candidate concentration response is not monotone"
            )
        return manifest, runtime

    def _recompute_evaluation(
        self,
        paths: Mapping[str, Path],
        report: Mapping[str, Any],
        baseline_model: FrozenConcentrationResponse,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Recompute every promotion statistic from sealed row-level bytes."""

        predictions_doc = _read_json(paths["predictions"])
        if predictions_doc.get("schema") != "perfumery-blind-challenge-predictions/v1":
            raise ContinualImprovementError("unsupported sealed prediction schema")
        raw_predictions = predictions_doc.get("predictions")
        if not isinstance(raw_predictions, list) or not raw_predictions:
            raise ContinualImprovementError("sealed prediction rows are missing")
        predictions: dict[str, Mapping[str, Any]] = {}
        for row in raw_predictions:
            if not isinstance(row, Mapping):
                raise ContinualImprovementError("sealed prediction row must be an object")
            row_id = _required_text(row.get("row_id"), "prediction.row_id")
            if row_id in predictions:
                raise ContinualImprovementError("sealed prediction row IDs are duplicated")
            predictions[row_id] = row
        seal = _read_json(paths["prediction_seal"])
        if seal.get("schema") != "perfumery-blind-challenge-local-seal/v1":
            raise ContinualImprovementError("unsupported prediction-seal schema")
        expected_seal = {
            "candidate_id": report.get("candidate_id"),
            "prediction_sha256": sha256_file(paths["predictions"]),
            "runtime_sha256": sha256_file(paths["runtime"]),
            "model_manifest_sha256": sha256_file(paths["model_manifest"]),
            "challenge_inputs_sha256": sha256_file(paths["challenge_inputs"]),
            "outcomes_present_or_read_by_this_process": False,
        }
        if any(seal.get(key) != value for key, value in expected_seal.items()):
            raise ContinualImprovementError("prediction seal does not bind frozen inputs")
        if predictions_doc.get("challenge_inputs_sha256") != sha256_file(
            paths["challenge_inputs"]
        ):
            raise ContinualImprovementError("predictions do not bind challenge inputs")
        if predictions_doc.get("runtime_sha256") != sha256_file(paths["runtime"]):
            raise ContinualImprovementError("predictions do not bind candidate runtime")
        if predictions_doc.get("model_manifest_sha256") != sha256_file(
            paths["model_manifest"]
        ):
            raise ContinualImprovementError("predictions do not bind model manifest")

        training_rows = _strict_csv_rows(
            paths["training_data"],
            TRAINING_COLUMNS,
            "training_data",
            maximum_rows=self.policy.maximum_training_rows,
        )
        training_sources: set[str] = set()
        training_molecules: set[str] = set()
        training_scaffolds: set[str] = set()
        for row in training_rows:
            training_sources.add(
                _required_text(row.get("source_id"), "training_data.source_id")
            )
            _required_text(row.get("target_id"), "training_data.target_id")
            training_molecules.add(
                _required_text(row.get("molecule_id"), "training_data.molecule_id")
            )
            training_scaffolds.add(
                _required_text(row.get("scaffold_id"), "training_data.scaffold_id")
            )
            dilution = _finite_float(
                row.get("dilution_fraction"), "training_data.dilution_fraction"
            )
            intensity = _finite_float(
                row.get("intensity"), "training_data.intensity"
            )
            if not 1e-8 <= dilution <= 1.0 or not 0.0 <= intensity <= 100.0:
                raise ContinualImprovementError("training numeric value is outside domain")
            if row.get("label_origin") != "external_human_measurement":
                raise ContinualImprovementError(
                    "training label is not an external human measurement"
                )
            if row.get("evidence_class") not in {
                "retrospective_external_human",
                "prospective_external_human",
            }:
                raise ContinualImprovementError("training evidence class is not external human")

        challenge_rows = _strict_csv_rows(
            paths["challenge_inputs"],
            CHALLENGE_COLUMNS,
            "challenge_inputs",
            maximum_rows=self.policy.maximum_evaluation_rows,
        )
        challenge = {row["row_id"].strip(): row for row in challenge_rows}
        if set(challenge) != set(predictions):
            raise ContinualImprovementError(
                "challenge-input and sealed-prediction row IDs differ"
            )
        for row_id, challenge_row in challenge.items():
            prediction_row = predictions[row_id]
            for field in (
                "source_id",
                "target_id",
                "molecule_id",
                "scaffold_id",
            ):
                if _required_text(
                    challenge_row.get(field), f"challenge_inputs.{field}"
                ) != _required_text(prediction_row.get(field), f"prediction.{field}"):
                    raise ContinualImprovementError(
                        f"challenge input differs from sealed prediction: {field}"
                    )
            challenge_dilution = _finite_float(
                challenge_row.get("dilution_fraction"),
                "challenge_inputs.dilution_fraction",
            )
            prediction_dilution = _finite_float(
                prediction_row.get("dilution_fraction"),
                "prediction.dilution_fraction",
            )
            if abs(challenge_dilution - prediction_dilution) > 1e-15:
                raise ContinualImprovementError(
                    "challenge dilution differs from sealed prediction"
                )
        outcome_rows = _strict_csv_rows(
            paths["outcomes"],
            OUTCOME_COLUMNS,
            "outcomes",
            maximum_rows=self.policy.maximum_evaluation_rows,
        )
        outcomes: dict[str, float] = {}
        for row in outcome_rows:
            row_id = _required_text(row.get("row_id"), "outcome.row_id")
            if row_id in outcomes:
                raise ContinualImprovementError("outcome row IDs are duplicated")
            intensity = _finite_float(row.get("intensity"), "outcome.intensity")
            if not 0.0 <= intensity <= 100.0:
                raise ContinualImprovementError("outcome intensity is outside 0..100")
            outcomes[row_id] = intensity
        if set(predictions) != set(outcomes):
            raise ContinualImprovementError("sealed prediction and outcome row IDs differ")
        ordered = sorted(outcomes)
        row_tokens = {
            _hash_bytes(
                canonical_json(
                    {
                        "row_id": row_id,
                        "source_id": predictions[row_id].get("source_id"),
                        "target_id": predictions[row_id].get("target_id"),
                        "molecule_id": predictions[row_id].get("molecule_id"),
                        "scaffold_id": predictions[row_id].get("scaffold_id"),
                    }
                )
            )
            for row_id in ordered
        }
        candidate_values = []
        baseline_values = []
        target_values = []
        sources = []
        row_target_ids = []
        target_ids: set[str] = set()
        molecule_ids: set[str] = set()
        scaffold_ids: set[str] = set()
        portable_errors = []
        baseline_errors = []
        all_in_domain = True
        portable = FrozenConcentrationResponse(
            manifest_path=paths["model_manifest"], runtime_path=paths["runtime"]
        )
        for row_id in ordered:
            row = predictions[row_id]
            candidate_prediction = _finite_float(
                row.get("candidate_prediction"), "candidate_prediction"
            )
            baseline_prediction = _finite_float(
                row.get("baseline_prediction"), "baseline_prediction"
            )
            if not 0.0 <= candidate_prediction <= 100.0:
                raise ContinualImprovementError("candidate prediction is outside 0..100")
            if not 0.0 <= baseline_prediction <= 100.0:
                raise ContinualImprovementError("baseline prediction is outside 0..100")
            dilution = _finite_float(row.get("dilution_fraction"), "dilution_fraction")
            if not 1e-8 <= dilution <= 1.0:
                raise ContinualImprovementError("prediction dilution is outside domain")
            runtime_prediction, in_domain = portable.intensity(dilution)
            baseline_runtime_prediction, baseline_in_domain = baseline_model.intensity(
                dilution
            )
            all_in_domain = all_in_domain and in_domain and baseline_in_domain
            portable_errors.append(abs(runtime_prediction - candidate_prediction))
            baseline_errors.append(
                abs(baseline_runtime_prediction - baseline_prediction)
            )
            candidate_values.append(candidate_prediction)
            baseline_values.append(baseline_runtime_prediction)
            target_values.append(outcomes[row_id])
            sources.append(_required_text(row.get("source_id"), "source_id"))
            target_id = _required_text(row.get("target_id"), "target_id")
            target_ids.add(target_id)
            row_target_ids.append(target_id)
            molecule_ids.add(_required_text(row.get("molecule_id"), "molecule_id"))
            scaffold_ids.add(_required_text(row.get("scaffold_id"), "scaffold_id"))
        candidate_array = np.asarray(candidate_values, dtype=float)
        baseline_array = np.asarray(baseline_values, dtype=float)
        target_array = np.asarray(target_values, dtype=float)
        sources_array = np.asarray(sources, dtype=object)
        row_targets_array = np.asarray(row_target_ids, dtype=object)
        candidate_mae = float(np.mean(np.abs(candidate_array - target_array)))
        baseline_mae = float(np.mean(np.abs(baseline_array - target_array)))
        candidate_rank = _spearman(candidate_array, target_array)
        baseline_rank = _spearman(baseline_array, target_array)

        bootstrap = report.get("bootstrap")
        if (
            not isinstance(bootstrap, Mapping)
            or bootstrap.get("method")
            != "source_then_target_cluster_percentile"
        ):
            raise ContinualImprovementError("unsupported bootstrap contract")
        draws = _positive_int(bootstrap.get("draws"), "bootstrap.draws")
        if draws > self.policy.maximum_bootstrap_draws:
            raise ContinualImprovementError("bootstrap draws exceed the policy maximum")
        seed = _nonnegative_int(bootstrap.get("seed"), "bootstrap.seed")
        expected_seed = bootstrap_seed(sha256_file(paths["predictions"]))
        if seed != expected_seed:
            raise ContinualImprovementError(
                "bootstrap seed is not derived from sealed prediction bytes"
            )
        if draws != _positive_int(
            report.get("counts", {}).get("bootstrap_draws"),
            "counts.bootstrap_draws",
        ):
            raise ContinualImprovementError("bootstrap draw counts differ")
        unique_sources = np.asarray(sorted(set(sources)), dtype=object)
        if len(unique_sources) < 2:
            raise ContinualImprovementError("bootstrap requires at least two sources")
        targets_by_source = {
            source: np.asarray(
                sorted(set(row_targets_array[sources_array == source].tolist())),
                dtype=object,
            )
            for source in unique_sources
        }
        source_target_indices = {
            (source, target_id): np.where(
                (sources_array == source) & (row_targets_array == target_id)
            )[0]
            for source in unique_sources
            for target_id in targets_by_source[source]
        }
        rng = np.random.RandomState(seed)
        mae_gain = np.empty(draws, dtype=float)
        rank_gain = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled = rng.choice(unique_sources, size=len(unique_sources), replace=True)
            selected_parts = []
            for source in sampled:
                source_targets = targets_by_source[source]
                sampled_targets = rng.choice(
                    source_targets, size=len(source_targets), replace=True
                )
                selected_parts.extend(
                    source_target_indices[(source, target_id)]
                    for target_id in sampled_targets
                )
            selected = np.concatenate(selected_parts)
            mae_gain[draw] = float(
                np.mean(np.abs(baseline_array[selected] - target_array[selected]))
                - np.mean(np.abs(candidate_array[selected] - target_array[selected]))
            )
            rank_gain[draw] = _spearman(
                candidate_array[selected], target_array[selected]
            ) - _spearman(baseline_array[selected], target_array[selected])
        mae_interval = np.quantile(mae_gain, [0.025, 0.975])
        rank_interval = np.quantile(rank_gain, [0.025, 0.975])
        return (
            {
                "candidate_mae": candidate_mae,
                "baseline_mae": baseline_mae,
                "candidate_spearman": candidate_rank,
                "baseline_spearman": baseline_rank,
                "portable_parity_max_abs_error": float(max(portable_errors, default=0.0)),
                "mae_gain_lower": float(mae_interval[0]),
                "mae_gain_upper": float(mae_interval[1]),
                "rank_gain_lower": float(rank_interval[0]),
                "rank_gain_upper": float(rank_interval[1]),
            },
            {
                "rows": len(ordered),
                "sources": set(sources),
                "targets": target_ids,
                "molecules": molecule_ids,
                "scaffolds": scaffold_ids,
                "all_in_domain": all_in_domain,
                "baseline_parity_max_abs_error": float(
                    max(baseline_errors, default=0.0)
                ),
                "baseline_manifest_sha256": _required_text(
                    predictions_doc.get("baseline_manifest_sha256"),
                    "baseline_manifest_sha256",
                ),
                "training_rows": len(training_rows),
                "training_sources": training_sources,
                "training_molecules": training_molecules,
                "training_scaffolds": training_scaffolds,
                "evaluation_row_tokens": row_tokens,
            },
        )

    def _verify_acquisition_authorization(
        self,
        *,
        candidate_id: str,
        receipt: Mapping[str, Any],
        paths: Mapping[str, Path],
    ):
        acquisition_envelope = _read_json(paths["acquisition_authorization"])
        acquisition_scope = {
            "candidate_id": candidate_id,
            "dataset_id": _required_text(receipt.get("dataset_id"), "dataset_id"),
            "evidence_class": receipt.get("evidence_class"),
            "label_origin": receipt.get("label_origin"),
            "source_ids": sorted(
                _string_set(receipt.get("source_ids"), "receipt.source_ids")
            ),
            "target_ids": sorted(
                _string_set(receipt.get("target_ids"), "receipt.target_ids")
            ),
            "row_count": _positive_int(receipt.get("row_count"), "receipt.row_count"),
            "prediction_sealed_at": receipt.get("prediction_sealed_at"),
            "outcome_first_read_at": receipt.get("outcome_first_read_at"),
            "timestamp_authority_verified": True,
        }
        verified = self.trust_root.verify(
            acquisition_envelope,
            {
                "dataset_receipt": paths["dataset_receipt"],
                "outcomes": paths["outcomes"],
                "predictions": paths["predictions"],
                "prediction_seal": paths["prediction_seal"],
                "timestamp_response": paths["timestamp_response"],
            },
            expected_artifact_type=self.policy.acquisition_artifact_type,
            expected_scope=acquisition_scope,
            allowed_roles=self.policy.acquisition_signer_roles,
        )
        acquired_after = _parse_time(
            receipt.get("outcome_first_read_at"), "outcome_first_read_at"
        )
        if _parse_time(verified.issued_at, "acquisition issued_at") < acquired_after:
            raise ContinualImprovementError(
                "acquisition authorization predates outcome acquisition"
            )
        return verified

    def _scientific_reasons(
        self,
        candidate: Mapping[str, Any],
        report: Mapping[str, Any],
        receipt: Mapping[str, Any],
        paths: Mapping[str, Path],
        expected_baseline_manifest_sha256: str,
        baseline_model: FrozenConcentrationResponse,
    ) -> tuple[list[str], dict[str, float], dict[str, Any]]:
        reasons: list[str] = []
        candidate_id = _required_text(candidate.get("candidate_id"), "candidate_id")
        if report.get("schema") != EVALUATION_SCHEMA:
            reasons.append("evaluation_schema_invalid")
        if receipt.get("schema") != DATASET_RECEIPT_SCHEMA:
            reasons.append("dataset_receipt_schema_invalid")
        if report.get("candidate_id") != candidate_id:
            reasons.append("evaluation_candidate_id_mismatch")
        if receipt.get("candidate_id") != candidate_id:
            reasons.append("receipt_candidate_id_mismatch")
        try:
            dataset_id = _required_text(receipt.get("dataset_id"), "dataset_id")
        except ContinualImprovementError:
            dataset_id = ""
            reasons.append("dataset_id_missing")
        evidence_class = str(candidate.get("evidence_class", "")).strip()
        if evidence_class != self.policy.allowed_evidence_class:
            reasons.append("not_prospective_external_human_evidence")
        if report.get("evidence_class") != evidence_class or receipt.get("evidence_class") != evidence_class:
            reasons.append("evidence_class_mismatch")
        label_origin = str(candidate.get("label_origin", "")).strip()
        if label_origin in self.policy.blocked_label_origins:
            reasons.append("synthetic_or_model_generated_labels_forbidden")
        if label_origin != "external_human_measurement":
            reasons.append("label_origin_not_external_human_measurement")
        if report.get("label_origin") != label_origin or receipt.get("label_origin") != label_origin:
            reasons.append("label_origin_mismatch")
        claim = candidate.get("claim_boundary")
        if not isinstance(claim, Mapping) or claim.get("human_olfactory_90_percent_certified") is not False:
            reasons.append("human_olfactory_90_claim_forbidden")
        receipt_bindings_ok = True
        if receipt.get("prediction_sha256") != sha256_file(paths["predictions"]):
            reasons.append("receipt_prediction_hash_mismatch")
            receipt_bindings_ok = False
        if receipt.get("prediction_seal_sha256") != sha256_file(paths["prediction_seal"]):
            reasons.append("receipt_prediction_seal_hash_mismatch")
            receipt_bindings_ok = False
        if receipt.get("outcome_sha256") != sha256_file(paths["outcomes"]):
            reasons.append("receipt_outcome_hash_mismatch")
            receipt_bindings_ok = False
        if receipt.get("timestamp_response_sha256") != sha256_file(paths["timestamp_response"]):
            reasons.append("receipt_timestamp_response_hash_mismatch")
            receipt_bindings_ok = False
        if receipt.get("timestamp_authority_verified") is not True:
            reasons.append("prediction_timestamp_not_independently_verified")
        try:
            sealed = _parse_time(receipt.get("prediction_sealed_at"), "prediction_sealed_at")
            opened = _parse_time(receipt.get("outcome_first_read_at"), "outcome_first_read_at")
            if not sealed < opened:
                reasons.append("prediction_not_sealed_before_outcome")
        except ContinualImprovementError:
            reasons.append("prediction_outcome_chronology_invalid")
        if int(receipt.get("synthetic_rows", -1)) != 0:
            reasons.append("synthetic_rows_present")
        if int(receipt.get("model_generated_label_rows", -1)) != 0:
            reasons.append("model_generated_label_rows_present")
        if receipt.get("evaluation_labels_available_during_training") is not False:
            reasons.append("evaluation_labels_were_available_during_training")

        acquisition_signer_id = ""
        try:
            verified_acquisition = self._verify_acquisition_authorization(
                candidate_id=candidate_id,
                receipt=receipt,
                paths=paths,
            )
            acquisition_signer_id = verified_acquisition.signer_id
        except Exception as error:  # noqa: BLE001 - independent evidence boundary
            reasons.append(
                (
                    "signed_acquisition_authorization_invalid:"
                    + type(error).__name__
                    + ":"
                    + str(error)
                )[:500]
            )

        recomputed_metrics: dict[str, float] = {}
        recomputed_facts: dict[str, Any] = {}
        try:
            recomputed_metrics, recomputed_facts = self._recompute_evaluation(
                paths, report, baseline_model
            )
        except Exception as error:  # noqa: BLE001 - sealed row data is untrusted
            reasons.append(
                ("evaluation_recomputation_failed:" + type(error).__name__ + ":" + str(error))[
                    :500
                ]
            )
        if (
            recomputed_facts
            and recomputed_facts.get("baseline_manifest_sha256")
            != expected_baseline_manifest_sha256
        ):
            reasons.append("candidate_baseline_is_not_current_champion")
        if (
            recomputed_facts
            and recomputed_facts.get("baseline_parity_max_abs_error", math.inf)
            > self.policy.maximum_portable_parity_error
        ):
            reasons.append("sealed_baseline_predictions_do_not_match_current_champion")

        gates = report.get("gates")
        if not isinstance(gates, Mapping):
            gates = {}
            reasons.append("scientific_gates_missing")
        for gate in self.policy.required_scientific_gates:
            if gates.get(gate) is not True:
                reasons.append(f"gate_failed:{gate}")

        counts = report.get("counts")
        if not isinstance(counts, Mapping):
            counts = {}
            reasons.append("evaluation_counts_missing")
        try:
            rows = _positive_int(counts.get("rows"), "counts.rows")
            targets = _positive_int(counts.get("targets"), "counts.targets")
            sources = _positive_int(counts.get("sources"), "counts.sources")
            molecules = _positive_int(counts.get("molecules"), "counts.molecules")
            scaffolds = _positive_int(counts.get("scaffolds"), "counts.scaffolds")
            bootstrap_draws = _positive_int(
                counts.get("bootstrap_draws"), "counts.bootstrap_draws"
            )
            if rows < self.policy.minimum_evaluation_rows:
                reasons.append("evaluation_rows_below_policy")
            if rows > self.policy.maximum_evaluation_rows:
                reasons.append("evaluation_rows_above_policy")
            if targets < self.policy.minimum_evaluation_targets:
                reasons.append("evaluation_targets_below_policy")
            if sources < self.policy.minimum_external_sources:
                reasons.append("external_sources_below_policy")
            if bootstrap_draws < self.policy.minimum_bootstrap_draws:
                reasons.append("bootstrap_draws_below_policy")
            if bootstrap_draws > self.policy.maximum_bootstrap_draws:
                reasons.append("bootstrap_draws_above_policy")
            if recomputed_facts and (
                rows != recomputed_facts["rows"]
                or targets != len(recomputed_facts["targets"])
                or sources != len(recomputed_facts["sources"])
                or molecules != len(recomputed_facts["molecules"])
                or scaffolds != len(recomputed_facts["scaffolds"])
            ):
                reasons.append("evaluation_counts_do_not_match_sealed_rows")
            if receipt.get("row_count") != rows:
                reasons.append("receipt_row_count_mismatch")
        except ContinualImprovementError:
            reasons.append("evaluation_counts_invalid")

        lineage = report.get("lineage")
        if not isinstance(lineage, Mapping):
            lineage = {}
            reasons.append("evaluation_lineage_missing")
        try:
            training_sources = _string_set(
                lineage.get("training_source_ids"), "training_source_ids"
            )
            evaluation_sources = _string_set(
                lineage.get("evaluation_source_ids"), "evaluation_source_ids"
            )
            training_molecules = _string_set(
                lineage.get("training_molecule_ids"), "training_molecule_ids"
            )
            evaluation_molecules = _string_set(
                lineage.get("evaluation_molecule_ids"), "evaluation_molecule_ids"
            )
            training_scaffolds = _string_set(
                lineage.get("training_scaffold_ids"), "training_scaffold_ids"
            )
            evaluation_scaffolds = _string_set(
                lineage.get("evaluation_scaffold_ids"), "evaluation_scaffold_ids"
            )
            if training_sources & evaluation_sources:
                reasons.append("training_evaluation_source_overlap")
            if training_molecules & evaluation_molecules:
                reasons.append("training_evaluation_molecule_overlap")
            if training_scaffolds & evaluation_scaffolds:
                reasons.append("training_evaluation_scaffold_overlap")
            receipt_sources = _string_set(receipt.get("source_ids"), "receipt.source_ids")
            receipt_targets = _string_set(receipt.get("target_ids"), "receipt.target_ids")
            if evaluation_sources != receipt_sources:
                reasons.append("receipt_evaluation_sources_mismatch")
            if recomputed_facts:
                if training_sources != recomputed_facts["training_sources"]:
                    reasons.append("training_sources_do_not_match_training_bytes")
                if training_molecules != recomputed_facts["training_molecules"]:
                    reasons.append("training_molecules_do_not_match_training_bytes")
                if training_scaffolds != recomputed_facts["training_scaffolds"]:
                    reasons.append("training_scaffolds_do_not_match_training_bytes")
                if evaluation_sources != recomputed_facts["sources"]:
                    reasons.append("evaluation_sources_do_not_match_sealed_rows")
                if evaluation_molecules != recomputed_facts["molecules"]:
                    reasons.append("evaluation_molecules_do_not_match_sealed_rows")
                if evaluation_scaffolds != recomputed_facts["scaffolds"]:
                    reasons.append("evaluation_scaffolds_do_not_match_sealed_rows")
                if receipt_targets != recomputed_facts["targets"]:
                    reasons.append("receipt_evaluation_targets_mismatch")
                if not recomputed_facts["all_in_domain"]:
                    reasons.append("challenge_outside_runtime_domain")
            model_document = _read_json(paths["model_manifest"])
            model_lineage = model_document.get("continual_training_lineage")
            if not isinstance(model_lineage, Mapping):
                reasons.append("model_training_lineage_missing")
            else:
                if training_sources != _string_set(
                    model_lineage.get("source_ids"), "model source_ids"
                ):
                    reasons.append("report_model_training_sources_mismatch")
                if training_molecules != _string_set(
                    model_lineage.get("molecule_ids"), "model molecule_ids"
                ):
                    reasons.append("report_model_training_molecules_mismatch")
                if training_scaffolds != _string_set(
                    model_lineage.get("scaffold_ids"), "model scaffold_ids"
                ):
                    reasons.append("report_model_training_scaffolds_mismatch")
                if recomputed_facts:
                    if _string_set(
                        model_lineage.get("source_ids"), "model source_ids"
                    ) != recomputed_facts["training_sources"]:
                        reasons.append("model_sources_do_not_match_training_bytes")
                    if _string_set(
                        model_lineage.get("molecule_ids"), "model molecule_ids"
                    ) != recomputed_facts["training_molecules"]:
                        reasons.append("model_molecules_do_not_match_training_bytes")
                    if _string_set(
                        model_lineage.get("scaffold_ids"), "model scaffold_ids"
                    ) != recomputed_facts["training_scaffolds"]:
                        reasons.append("model_scaffolds_do_not_match_training_bytes")
                    if model_document.get("training_records") != recomputed_facts[
                        "training_rows"
                    ]:
                        reasons.append("model_training_row_count_mismatch")
                if model_lineage.get("training_csv_sha256") != sha256_file(
                    paths["training_data"]
                ):
                    reasons.append("model_training_hash_mismatch")
                if model_lineage.get("label_origin") != "external_human_measurement":
                    reasons.append("model_training_label_origin_invalid")
                evidence_classes = _string_set(
                    model_lineage.get("evidence_classes"),
                    "model evidence_classes",
                )
                if not evidence_classes.issubset(
                    {
                        "retrospective_external_human",
                        "prospective_external_human",
                    }
                ):
                    reasons.append("model_training_evidence_class_invalid")
                if int(model_lineage.get("synthetic_rows", -1)) != 0:
                    reasons.append("model_training_synthetic_rows_present")
                if int(model_lineage.get("model_generated_label_rows", -1)) != 0:
                    reasons.append("model_training_generated_label_rows_present")
        except ContinualImprovementError:
            reasons.append("evaluation_lineage_invalid")

        metrics_value = report.get("metrics")
        metrics: dict[str, float] = dict(recomputed_metrics)
        try:
            if not isinstance(metrics_value, Mapping):
                raise ContinualImprovementError("metrics must be an object")
            for name in (
                "candidate_mae",
                "baseline_mae",
                "candidate_spearman",
                "baseline_spearman",
                "portable_parity_max_abs_error",
            ):
                reported = _finite_float(metrics_value.get(name), f"metrics.{name}")
                if name in recomputed_metrics and abs(reported - recomputed_metrics[name]) > 1e-10:
                    reasons.append(f"reported_metric_differs_from_sealed_rows:{name}")
                metrics.setdefault(name, reported)
            mae_interval = metrics_value.get("baseline_minus_candidate_mae_bootstrap_95")
            rank_interval = metrics_value.get(
                "candidate_minus_baseline_spearman_bootstrap_95"
            )
            if not isinstance(mae_interval, list) or len(mae_interval) != 2:
                raise ContinualImprovementError("MAE bootstrap interval is invalid")
            if not isinstance(rank_interval, list) or len(rank_interval) != 2:
                raise ContinualImprovementError("rank bootstrap interval is invalid")
            reported_intervals = {
                "mae_gain_lower": _finite_float(mae_interval[0], "mae interval lower"),
                "mae_gain_upper": _finite_float(mae_interval[1], "mae interval upper"),
                "rank_gain_lower": _finite_float(rank_interval[0], "rank interval lower"),
                "rank_gain_upper": _finite_float(rank_interval[1], "rank interval upper"),
            }
            for name, reported in reported_intervals.items():
                if name in recomputed_metrics and abs(reported - recomputed_metrics[name]) > 1e-10:
                    reasons.append(f"reported_metric_differs_from_sealed_rows:{name}")
                metrics.setdefault(name, reported)
            if not metrics["candidate_mae"] < metrics["baseline_mae"]:
                reasons.append("candidate_mae_did_not_improve")
            if metrics["mae_gain_lower"] <= self.policy.minimum_mae_gain_lower:
                reasons.append("bootstrap_mae_gain_not_positive")
            if metrics["rank_gain_lower"] < self.policy.minimum_rank_gain_lower:
                reasons.append("rank_noninferiority_not_established")
            if metrics["portable_parity_max_abs_error"] > self.policy.maximum_portable_parity_error:
                reasons.append("portable_runtime_parity_failed")
        except ContinualImprovementError:
            reasons.append("evaluation_metrics_invalid")
        evaluation_identity = (
            {
                "dataset_id": dataset_id,
                "outcome_sha256": sha256_file(paths["outcomes"]),
                "row_tokens": sorted(recomputed_facts["evaluation_row_tokens"]),
                "acquisition_signer_id": acquisition_signer_id,
                "acquisition_signer_key_sha256": (
                    verified_acquisition.signer_key_sha256
                ),
                "acquisition_issued_at": verified_acquisition.issued_at,
                "outcome_first_read_at": receipt.get("outcome_first_read_at"),
            }
            if (
                dataset_id
                and recomputed_facts
                and receipt_bindings_ok
                and acquisition_signer_id
            )
            else {}
        )
        return sorted(set(reasons)), metrics, evaluation_identity

    def _authorization_path(
        self, manifest_path: Path, candidate: Mapping[str, Any]
    ) -> Path | None:
        # Authorization is a detached, conventionally named envelope.  Keeping
        # it out of candidate.json avoids an impossible hash cycle: the signed
        # envelope binds candidate.json, but candidate.json must not bind the
        # signature that does not exist until after the candidate is frozen.
        if "production_authorization" in candidate:
            raise ContinualImprovementError(
                "production authorization must be detached as authorization.json"
            )
        path = manifest_path.parent / "authorization.json"
        if not path.exists():
            return None
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.parent != manifest_path.parent.resolve(strict=True):
            raise ContinualImprovementError("detached authorization path is invalid")
        return resolved

    @staticmethod
    def _submission_sha(manifest_path: Path) -> str:
        authorization = manifest_path.parent / "authorization.json"
        return _hash_bytes(
            canonical_json(
                {
                    "manifest_sha256": sha256_file(manifest_path),
                    "authorization_sha256": (
                        sha256_file(authorization) if authorization.is_file() else ""
                    ),
                }
            )
        )

    def _production_decision(
        self,
        *,
        candidate: Mapping[str, Any],
        manifest_path: Path,
        manifest_sha: str,
        paths: Mapping[str, Path],
        authorization_path: Path,
        acquisition_signer_id: str,
        acquisition_signer_key_sha256: str,
        acquisition_issued_at: str,
        outcome_first_read_at: str,
    ) -> ConcentrationResponseTrustDecision:
        requested_weight = _finite_float(
            candidate.get("requested_primary_score_weight"),
            "requested_primary_score_weight",
        )
        if not 0.0 < requested_weight <= self.policy.maximum_primary_score_weight:
            raise ContinualImprovementError("requested primary-score weight is out of policy")
        envelope = _read_json(authorization_path)
        if envelope.get("artifact_type") != self.policy.authorization_artifact_type:
            raise ContinualImprovementError("production authorization artifact type is invalid")
        decision = ConcentrationResponseTrustDecision.from_signed_authorization(
            self.trust_root,
            envelope,
            manifest_path=paths["model_manifest"],
            runtime_path=paths["runtime"],
            additional_artifact_paths={
                "candidate_manifest": manifest_path,
                "dataset_receipt": paths["dataset_receipt"],
                "evaluation_report": paths["evaluation_report"],
                "outcomes": paths["outcomes"],
                "predictions": paths["predictions"],
                "prediction_seal": paths["prediction_seal"],
                "timestamp_response": paths["timestamp_response"],
                "training_data": paths["training_data"],
                "challenge_inputs": paths["challenge_inputs"],
                "acquisition_authorization": paths[
                    "acquisition_authorization"
                ],
            },
            expected_scope_extra={
                "candidate_id": candidate["candidate_id"],
                "model_family": SUPPORTED_MODEL_FAMILY,
                "candidate_manifest_sha256": manifest_sha,
                "evaluation_report_sha256": sha256_file(paths["evaluation_report"]),
                "dataset_receipt_sha256": sha256_file(paths["dataset_receipt"]),
            },
            allowed_roles=self.policy.allowed_signer_roles,
        )
        if (
            self.policy.require_distinct_production_signer
            and decision.signer_id == acquisition_signer_id
        ):
            raise ContinualImprovementError(
                "production signer must differ from acquisition signer"
            )
        if (
            self.policy.require_distinct_production_signer
            and decision.signer_key_sha256 == acquisition_signer_key_sha256
        ):
            raise ContinualImprovementError(
                "production signer key must differ from acquisition signer key"
            )
        earliest_release = max(
            _parse_time(acquisition_issued_at, "acquisition_issued_at"),
            _parse_time(outcome_first_read_at, "outcome_first_read_at"),
        )
        if _parse_time(decision.issued_at, "production issued_at") < earliest_release:
            raise ContinualImprovementError(
                "production authorization predates acquired evaluation evidence"
            )
        return decision

    def _assess(
        self,
        manifest_path: Path,
        *,
        expected_baseline_manifest_sha256: str,
        baseline_model: FrozenConcentrationResponse,
    ) -> tuple[CandidateAssessment, dict[str, Any] | None, dict[str, Any]]:
        manifest_sha = sha256_file(manifest_path)
        candidate_id = manifest_path.parent.name
        try:
            resolved_manifest = manifest_path.resolve(strict=True)
            try:
                resolved_manifest.relative_to(self.inbox.resolve(strict=True))
            except ValueError as error:
                raise ContinualImprovementError(
                    "candidate manifest escapes controller inbox"
                ) from error
            candidate = _read_json(manifest_path)
            candidate_id = _required_text(candidate.get("candidate_id"), "candidate_id")
            if not CANDIDATE_ID_RE.fullmatch(candidate_id):
                raise ContinualImprovementError("candidate_id contains unsupported characters")
            if candidate_id != manifest_path.parent.name:
                raise ContinualImprovementError("candidate_id differs from bundle directory")
            if candidate.get("schema") != CANDIDATE_SCHEMA:
                raise ContinualImprovementError("unsupported candidate schema")
            if candidate.get("model_family") != SUPPORTED_MODEL_FAMILY:
                raise ContinualImprovementError("unsupported model family")
            _parse_time(candidate.get("created_at"), "created_at")
            paths = self._validate_artifacts(manifest_path, candidate)
            _model_manifest, _runtime = self._validate_runtime(paths)
            report = _read_json(paths["evaluation_report"])
            receipt = _read_json(paths["dataset_receipt"])
            reasons, metrics, evaluation_identity = self._scientific_reasons(
                candidate,
                report,
                receipt,
                paths,
                expected_baseline_manifest_sha256,
                baseline_model,
            )
            if reasons:
                return (
                    CandidateAssessment(
                        candidate_id=candidate_id,
                        manifest_sha256=manifest_sha,
                        status="rejected",
                        scientific_gate_passed=False,
                        shadow_promoted=False,
                        production_promoted=False,
                        reasons=tuple(reasons),
                        metrics=metrics,
                    ),
                    None,
                    evaluation_identity,
                )
            authorization_path = self._authorization_path(manifest_path, candidate)
            decision = None
            authorization_error = "signed_production_authorization_missing"
            if authorization_path is not None:
                try:
                    decision = self._production_decision(
                        candidate=candidate,
                        manifest_path=manifest_path,
                        manifest_sha=manifest_sha,
                        paths=paths,
                        authorization_path=authorization_path,
                        acquisition_signer_id=evaluation_identity[
                            "acquisition_signer_id"
                        ],
                        acquisition_signer_key_sha256=evaluation_identity[
                            "acquisition_signer_key_sha256"
                        ],
                        acquisition_issued_at=evaluation_identity[
                            "acquisition_issued_at"
                        ],
                        outcome_first_read_at=evaluation_identity[
                            "outcome_first_read_at"
                        ],
                    )
                    authorization_error = ""
                except Exception as error:  # noqa: BLE001 - untrusted boundary
                    authorization_error = (
                        "signed_production_authorization_invalid:"
                        + type(error).__name__
                        + ":"
                        + str(error)
                    )[:500]
            record = {
                "candidate_id": candidate_id,
                "manifest_path": str(manifest_path.resolve(strict=True)),
                "manifest_sha256": manifest_sha,
                "model_manifest_path": str(paths["model_manifest"]),
                "model_manifest_sha256": sha256_file(paths["model_manifest"]),
                "runtime_path": str(paths["runtime"]),
                "runtime_sha256": sha256_file(paths["runtime"]),
                "evaluation_report_path": str(paths["evaluation_report"]),
                "evaluation_report_sha256": sha256_file(paths["evaluation_report"]),
                "dataset_receipt_path": str(paths["dataset_receipt"]),
                "dataset_receipt_sha256": sha256_file(paths["dataset_receipt"]),
                "metrics": metrics,
                "promoted_at": _utc_now(),
                "approved_primary_score_weight": (
                    decision.approved_primary_score_weight if decision else 0.0
                ),
                "authorization_verified": decision is not None,
                "authorization_path": str(authorization_path) if authorization_path else "",
                "authorization_sha256": (
                    sha256_file(authorization_path) if authorization_path else ""
                ),
                "authorization_artifact_id": (
                    decision.authorization_artifact_id if decision else ""
                ),
                "authorization_signer_id": decision.signer_id if decision else "",
                "acquisition_signer_id": evaluation_identity[
                    "acquisition_signer_id"
                ],
                "acquisition_signer_key_sha256": evaluation_identity[
                    "acquisition_signer_key_sha256"
                ],
                "acquisition_issued_at": evaluation_identity[
                    "acquisition_issued_at"
                ],
                "acquisition_authorization_sha256": sha256_file(
                    paths["acquisition_authorization"]
                ),
            }
            production = decision is not None
            return (
                CandidateAssessment(
                    candidate_id=candidate_id,
                    manifest_sha256=manifest_sha,
                    status="production_promoted" if production else "shadow_promoted",
                    scientific_gate_passed=True,
                    shadow_promoted=True,
                    production_promoted=production,
                    reasons=() if production else (authorization_error,),
                    metrics=metrics,
                ),
                record,
                evaluation_identity,
            )
        except Exception as error:  # noqa: BLE001 - candidate input is untrusted
            return (
                CandidateAssessment(
                    candidate_id=candidate_id,
                    manifest_sha256=manifest_sha,
                    status="rejected",
                    scientific_gate_passed=False,
                    shadow_promoted=False,
                    production_promoted=False,
                    reasons=((f"invalid_candidate:{type(error).__name__}:{error}")[:500],),
                    metrics={},
                ),
                None,
                {},
            )

    def run_once(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._recover_pending()
            state = self._load_state()
            if self.state_path.exists() or self.audit_path.exists():
                self.verify_registry()
            processed = state["processed"]
            decisions: list[CandidateAssessment] = []
            for path in sorted(self.inbox.glob("*/candidate.json")):
                submission_sha = self._submission_sha(path)
                if submission_sha in processed:
                    continue
                manifest_sha = sha256_file(path)
                family = state["champions"][SUPPORTED_MODEL_FAMILY]
                shadow = family.get("shadow")
                production = family["production"]
                baseline = (
                    production
                    if isinstance(shadow, Mapping)
                    and shadow.get("manifest_sha256") == manifest_sha
                    else (shadow if isinstance(shadow, Mapping) else production)
                )
                expected_baseline = _sha256_hex(
                    baseline.get("model_manifest_sha256"),
                    "current champion model_manifest_sha256",
                )
                if baseline.get("candidate_id") == "bundled_concentration_response_v1":
                    baseline_model = FrozenConcentrationResponse()
                else:
                    baseline_manifest_path = Path(
                        _required_text(
                            baseline.get("model_manifest_path"),
                            "current champion model_manifest_path",
                        )
                    ).resolve(strict=True)
                    baseline_runtime_path = Path(
                        _required_text(
                            baseline.get("runtime_path"),
                            "current champion runtime_path",
                        )
                    ).resolve(strict=True)
                    if sha256_file(baseline_manifest_path) != expected_baseline:
                        raise ContinualImprovementError(
                            "current champion model manifest changed"
                        )
                    if sha256_file(baseline_runtime_path) != _sha256_hex(
                        baseline.get("runtime_sha256"),
                        "current champion runtime_sha256",
                    ):
                        raise ContinualImprovementError(
                            "current champion runtime changed"
                        )
                    baseline_model = FrozenConcentrationResponse(
                        manifest_path=baseline_manifest_path,
                        runtime_path=baseline_runtime_path,
                    )
                assessment, record, evaluation_identity = self._assess(
                    path,
                    expected_baseline_manifest_sha256=expected_baseline,
                    baseline_model=baseline_model,
                )
                ledger = state["evaluation_ledger"]
                conflicts: list[str] = []
                if evaluation_identity:
                    manifest_sha = assessment.manifest_sha256
                    dataset_owner = ledger["datasets"].get(
                        evaluation_identity["dataset_id"]
                    )
                    outcome_owner = ledger["outcomes"].get(
                        evaluation_identity["outcome_sha256"]
                    )
                    if dataset_owner not in {None, manifest_sha}:
                        conflicts.append("prospective_dataset_already_consumed")
                    if outcome_owner not in {None, manifest_sha}:
                        conflicts.append("prospective_outcome_already_consumed")
                    if any(
                        ledger["rows"].get(token) not in {None, manifest_sha}
                        for token in evaluation_identity["row_tokens"]
                    ):
                        conflicts.append("prospective_evaluation_rows_already_consumed")
                    if conflicts:
                        assessment = replace(
                            assessment,
                            status="rejected",
                            scientific_gate_passed=False,
                            shadow_promoted=False,
                            production_promoted=False,
                            reasons=tuple(
                                sorted(set((*assessment.reasons, *conflicts)))
                            ),
                        )
                        record = None
                    else:
                        ledger["datasets"][evaluation_identity["dataset_id"]] = (
                            manifest_sha
                        )
                        ledger["outcomes"][evaluation_identity["outcome_sha256"]] = (
                            manifest_sha
                        )
                        for token in evaluation_identity["row_tokens"]:
                            ledger["rows"][token] = manifest_sha
                decisions.append(assessment)
                processed[submission_sha] = {
                    **asdict(assessment),
                    "processed_at": _utc_now(),
                    "path": str(path.resolve(strict=True)),
                    "submission_sha256": submission_sha,
                }
                if record is not None and assessment.shadow_promoted:
                    state["champions"][SUPPORTED_MODEL_FAMILY]["shadow"] = record
                if record is not None and assessment.production_promoted:
                    state["champions"][SUPPORTED_MODEL_FAMILY]["production"] = record
            if decisions or not self.state_path.exists():
                self._commit(state, decisions)
            verification = self.verify_registry()
            return {
                "status": "ok",
                "processed_now": len(decisions),
                "decisions": [asdict(item) for item in decisions],
                "registry": state,
                "verification": verification,
            }

    def status(self) -> dict[str, Any]:
        with self._locked():
            self._recover_pending()
            state = self._load_state()
            verification = self.verify_registry() if self.state_path.exists() else {
                "valid": True,
                "event_count": 0,
                "audit_head": "",
                "state_commit_sha256": self._state_commit_hash(state),
            }
            return {"registry": state, "verification": verification}

    def build_authorization_request(
        self,
        *,
        candidate_id: str,
        signer_id: str,
        issued_at: str,
        expires_at: str,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Build, but never sign, the exact current-shadow promotion envelope."""

        with self._locked():
            self._recover_pending()
            self.verify_registry()
            state = self._load_state()
            shadow = state["champions"][SUPPORTED_MODEL_FAMILY].get("shadow")
            if not isinstance(shadow, Mapping) or shadow.get("candidate_id") != candidate_id:
                raise ContinualImprovementError(
                    "authorization requests are limited to the current scientific shadow"
                )
            manifest_path = Path(
                _required_text(shadow.get("manifest_path"), "shadow.manifest_path")
            ).resolve(strict=True)
            if sha256_file(manifest_path) != shadow.get("manifest_sha256"):
                raise ContinualImprovementError("current shadow manifest changed")
            candidate = _read_json(manifest_path)
            paths = self._validate_artifacts(manifest_path, candidate)
            model_manifest, _runtime = self._validate_runtime(paths)
            receipt = _read_json(paths["dataset_receipt"])
            acquisition = self._verify_acquisition_authorization(
                candidate_id=candidate_id,
                receipt=receipt,
                paths=paths,
            )
            if signer_id == acquisition.signer_id:
                raise ContinualImprovementError(
                    "production signer must differ from acquisition signer"
                )
            weight = _finite_float(
                candidate.get("requested_primary_score_weight"),
                "requested_primary_score_weight",
            )
            if not 0.0 < weight <= self.policy.maximum_primary_score_weight:
                raise ContinualImprovementError("requested primary-score weight is out of policy")
            issued = _parse_time(issued_at, "issued_at")
            expires = _parse_time(expires_at, "expires_at")
            if expires <= issued:
                raise ContinualImprovementError("authorization expiration must follow issuance")
            earliest_release = max(
                _parse_time(acquisition.issued_at, "acquisition issued_at"),
                _parse_time(
                    receipt.get("outcome_first_read_at"), "outcome_first_read_at"
                ),
            )
            if issued < earliest_release:
                raise ContinualImprovementError(
                    "production authorization cannot predate acquired evidence"
                )
            artifact_paths = {
                "manifest": paths["model_manifest"],
                "model": paths["runtime"],
                "candidate_manifest": manifest_path,
                "dataset_receipt": paths["dataset_receipt"],
                "evaluation_report": paths["evaluation_report"],
                "outcomes": paths["outcomes"],
                "predictions": paths["predictions"],
                "prediction_seal": paths["prediction_seal"],
                "timestamp_response": paths["timestamp_response"],
                "training_data": paths["training_data"],
                "challenge_inputs": paths["challenge_inputs"],
                "acquisition_authorization": paths[
                    "acquisition_authorization"
                ],
            }
            envelope = {
                "schema": ARTIFACT_SIGNATURE_SCHEMA,
                "artifact_id": artifact_id or f"{candidate_id}-production-release",
                "artifact_type": self.policy.authorization_artifact_type,
                "signer_id": _required_text(signer_id, "signer_id"),
                "signer_role": "model_release_approver",
                "scope": {
                    "model_sha256": sha256_file(paths["runtime"]),
                    "manifest_sha256": sha256_file(paths["model_manifest"]),
                    "algorithm": model_manifest.get("algorithm"),
                    "approved_primary_score_weight": weight,
                    "candidate_id": candidate_id,
                    "model_family": SUPPORTED_MODEL_FAMILY,
                    "candidate_manifest_sha256": sha256_file(manifest_path),
                    "evaluation_report_sha256": sha256_file(
                        paths["evaluation_report"]
                    ),
                    "dataset_receipt_sha256": sha256_file(paths["dataset_receipt"]),
                },
                "issued_at": issued.isoformat(),
                "expires_at": expires.isoformat(),
                "artifact_hashes": {
                    label: sha256_file(path)
                    for label, path in artifact_paths.items()
                },
            }
            return {
                "envelope": envelope,
                "signing_payload_sha256": _hash_bytes(canonical_json(envelope)),
                "output_path": str(manifest_path.parent / "authorization.json"),
                "instruction": (
                    "Sign canonical JSON of envelope with the allowlisted Ed25519 key, "
                    "base64-encode the signature, add it as envelope.signature, and write "
                    "the envelope to output_path. The controller never receives the private key."
                ),
            }


def load_production_concentration_response(
    state_path: str | Path,
    trust_root: EvidenceTrustRoot | Mapping[str, Any] | str | Path,
) -> FrozenConcentrationResponse:
    """Load one signed production champion from a verified registry snapshot.

    This function is deliberately explicit.  The API only calls it when both
    continual-registry and trust-root environment variables are configured.
    Missing, tampered or unsigned state fails closed rather than silently
    enabling a research model.
    """

    path = Path(state_path).expanduser().resolve(strict=True)
    root_dir = path.parent
    if isinstance(trust_root, (str, Path)):
        trusted = EvidenceTrustRoot.from_json_file(trust_root)
    elif isinstance(trust_root, EvidenceTrustRoot):
        trusted = trust_root
    else:
        trusted = EvidenceTrustRoot(trust_root)
    controller = ContinuousImprovementController(root_dir, trust_root=trusted)
    controller.state_path = path
    verification = controller.verify_registry()
    if not verification.get("valid"):
        raise ContinualImprovementError("continuous registry verification failed")
    state = _read_json(path)
    production = state.get("champions", {}).get(SUPPORTED_MODEL_FAMILY, {}).get("production")
    if not isinstance(production, Mapping) or production.get("authorization_verified") is not True:
        raise ContinualImprovementError("no signed production champion is available")
    required = {
        "manifest_path": "manifest_sha256",
        "model_manifest_path": "model_manifest_sha256",
        "runtime_path": "runtime_sha256",
        "evaluation_report_path": "evaluation_report_sha256",
        "dataset_receipt_path": "dataset_receipt_sha256",
        "authorization_path": "authorization_sha256",
    }
    verified_paths: dict[str, Path] = {}
    for path_key, hash_key in required.items():
        artifact = Path(_required_text(production.get(path_key), path_key)).resolve(strict=True)
        if sha256_file(artifact) != _sha256_hex(production.get(hash_key), hash_key):
            raise ContinualImprovementError(f"production artifact changed: {path_key}")
        verified_paths[path_key] = artifact
    candidate = _read_json(verified_paths["manifest_path"])
    candidate_artifacts = candidate.get("artifacts")
    if not isinstance(candidate_artifacts, Mapping) or set(candidate_artifacts) != REQUIRED_ARTIFACTS:
        raise ContinualImprovementError("production candidate artifact set changed")
    candidate_paths = {
        label: controller._artifact_path(
            verified_paths["manifest_path"].parent,
            candidate_artifacts[label],
            label,
        )
        for label in REQUIRED_ARTIFACTS
    }
    receipt = _read_json(candidate_paths["dataset_receipt"])
    acquisition = controller._verify_acquisition_authorization(
        candidate_id=_required_text(candidate.get("candidate_id"), "candidate_id"),
        receipt=receipt,
        paths=candidate_paths,
    )
    decision = controller._production_decision(
        candidate=candidate,
        manifest_path=verified_paths["manifest_path"],
        manifest_sha=sha256_file(verified_paths["manifest_path"]),
        paths=candidate_paths,
        authorization_path=verified_paths["authorization_path"],
        acquisition_signer_id=acquisition.signer_id,
        acquisition_signer_key_sha256=acquisition.signer_key_sha256,
        acquisition_issued_at=acquisition.issued_at,
        outcome_first_read_at=_required_text(
            receipt.get("outcome_first_read_at"), "outcome_first_read_at"
        ),
    )
    if abs(
        decision.approved_primary_score_weight
        - _finite_float(
            production.get("approved_primary_score_weight"),
            "approved_primary_score_weight",
        )
    ) > 1e-12:
        raise ContinualImprovementError("registry weight differs from signed authorization")
    return FrozenConcentrationResponse(
        decision,
        manifest_path=verified_paths["model_manifest_path"],
        runtime_path=verified_paths["runtime_path"],
    )
