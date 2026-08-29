"""Blind sensory-study collection with signed, re-verifiable human evidence.

Direct form entry and ordinary CSV import are retained solely for operational
audit history.  A caller claiming ``source=human`` is not proof that a human
study occurred.  Only an allowlisted, signed research-data artifact bound to
the exact study and formula set may contribute to sensory pass/fail evidence
or model calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import secrets
import sqlite3
import statistics
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .artifact_trust import EvidenceTrustRoot
from .sqlite_lifecycle import SQLiteConnectionOwner


SENSORY_ARTIFACT_TYPE = "sensory_study_data/v1"
SENSORY_SIGNER_ROLES = frozenset(
    {"sensory_principal_investigator", "sensory_laboratory", "independent_sensory_lab"}
)


@dataclass(frozen=True)
class SensoryEvidence:
    formula_id: str
    unique_panelists: int
    mean_similarity: float | None
    standard_deviation: float | None
    lower_confidence_bound_95: float | None
    target_similarity: float
    passed: bool
    status: str
    expert_panelists: int = 0
    evidence_source: str = "verified_external_human_study_only"
    verification_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationArtifact:
    slope: float
    intercept: float
    observations: int
    unique_formulas: int
    holdout_mae: float | None
    status: str
    trained_at: str
    artifact_version: str = "1.0"
    data_fingerprint: str = ""
    integrity_sha256: str = ""

    def _integrity_payload(self) -> dict:
        payload = asdict(self)
        payload.pop("integrity_sha256", None)
        return payload

    def sealed(self) -> "CalibrationArtifact":
        digest = hashlib.sha256(
            json.dumps(self._integrity_payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return replace(self, integrity_sha256=digest)

    def is_trusted(self) -> bool:
        if self.artifact_version != "1.0" or not self.integrity_sha256:
            return False
        expected = self.sealed().integrity_sha256
        statistical_gate = (
            self.status == "validated"
            and self.observations >= 30
            and self.unique_formulas >= 10
            and self.holdout_mae is not None
            and self.holdout_mae <= 10.0
            and bool(self.data_fingerprint)
        )
        return self.integrity_sha256 == expected and statistical_gate

    def predict(self, raw_score: float) -> float | None:
        if not self.is_trusted():
            return None
        return max(0.0, min(100.0, self.slope * raw_score + self.intercept))

    def save(self, path: str | Path) -> None:
        sealed = self.sealed()
        Path(path).write_text(json.dumps(asdict(sealed), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        artifact = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if artifact.status == "validated" and not artifact.is_trusted():
            raise ValueError("calibration artifact integrity/statistical gate failed")
        return artifact


class SensoryEvaluationStore(SQLiteConnectionOwner):
    """SQLite index for blind studies and independently verified study artifacts."""

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
            CREATE TABLE IF NOT EXISTS formulas (
                formula_id TEXT PRIMARY KEY,
                brief TEXT NOT NULL,
                predicted_similarity REAL NOT NULL,
                formula_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS studies (
                study_id TEXT PRIMARY KEY,
                brief TEXT NOT NULL,
                protocol_ref TEXT NOT NULL,
                created_at TEXT NOT NULL,
                migration_status TEXT NOT NULL DEFAULT 'legacy_unverified'
            );
            CREATE TABLE IF NOT EXISTS samples (
                study_id TEXT NOT NULL,
                formula_id TEXT NOT NULL,
                blind_code TEXT NOT NULL,
                PRIMARY KEY (study_id, blind_code),
                UNIQUE (study_id, formula_id),
                FOREIGN KEY (study_id) REFERENCES studies(study_id),
                FOREIGN KEY (formula_id) REFERENCES formulas(formula_id)
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                study_id TEXT NOT NULL,
                blind_code TEXT NOT NULL,
                panelist_id TEXT NOT NULL,
                expert INTEGER NOT NULL,
                similarity REAL NOT NULL,
                liking REAL NOT NULL,
                dimension_json TEXT NOT NULL,
                defects_json TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'legacy_unverified',
                evidence_ref TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (study_id, blind_code, panelist_id)
            );
            CREATE TABLE IF NOT EXISTS verified_studies (
                study_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL UNIQUE,
                data_path TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS verified_evaluations (
                study_id TEXT NOT NULL,
                blind_code TEXT NOT NULL,
                panelist_id TEXT NOT NULL,
                expert INTEGER NOT NULL,
                similarity REAL NOT NULL,
                liking REAL NOT NULL,
                dimension_json TEXT NOT NULL,
                defects_json TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                PRIMARY KEY (study_id, blind_code, panelist_id),
                FOREIGN KEY (study_id) REFERENCES verified_studies(study_id)
            );
            CREATE TABLE IF NOT EXISTS sensory_artifact_revocations (
                artifact_id TEXT PRIMARY KEY,
                revoked_on TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            """
        )
        study_columns = {str(row[1]) for row in self.connection.execute("PRAGMA table_info(studies)")}
        if "migration_status" not in study_columns:
            self.connection.execute(
                "ALTER TABLE studies ADD COLUMN migration_status TEXT NOT NULL DEFAULT 'legacy_unverified'"
            )
        evaluation_columns = {str(row[1]) for row in self.connection.execute("PRAGMA table_info(evaluations)")}
        if "source" not in evaluation_columns:
            self.connection.execute(
                "ALTER TABLE evaluations ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy_unverified'"
            )
        if "evidence_ref" not in evaluation_columns:
            self.connection.execute("ALTER TABLE evaluations ADD COLUMN evidence_ref TEXT NOT NULL DEFAULT ''")
        self.connection.commit()

    def register_formula(
        self,
        formula_id: str,
        brief: str,
        predicted_similarity: float,
        formula_payload: list[dict],
    ) -> None:
        if not formula_id.startswith("sha256:"):
            raise ValueError("formula_id must be a sha256 fingerprint")
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "INSERT OR REPLACE INTO formulas VALUES (?, ?, ?, ?, ?)",
            (
                formula_id,
                brief,
                float(predicted_similarity),
                json.dumps(formula_payload, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        self.connection.commit()

    def create_study(
        self,
        brief: str,
        formula_ids: Iterable[str],
        protocol_ref: str,
        seed: int | None = None,
    ) -> dict[str, str]:
        formula_ids = list(dict.fromkeys(formula_ids))
        if len(formula_ids) < 2:
            raise ValueError("a blind comparison requires at least two formulas")
        if not protocol_ref.strip():
            raise ValueError("protocol_ref is required")
        known = {
            row[0]
            for row in self.connection.execute(
                f"SELECT formula_id FROM formulas WHERE formula_id IN ({','.join('?' for _ in formula_ids)})",
                formula_ids,
            )
        }
        missing = sorted(set(formula_ids) - known)
        if missing:
            raise ValueError(f"unregistered formulas: {missing}")

        study_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "INSERT INTO studies (study_id, brief, protocol_ref, created_at, migration_status) VALUES (?, ?, ?, ?, 'legacy_unverified')",
            (study_id, brief, protocol_ref, now),
        )
        shuffled = formula_ids.copy()
        random.Random(seed).shuffle(shuffled) if seed is not None else secrets.SystemRandom().shuffle(shuffled)
        mapping: dict[str, str] = {}
        for index, formula_id in enumerate(shuffled):
            token = hashlib.sha256(f"{study_id}:{index}:{formula_id}".encode()).hexdigest()[:6].upper()
            code = f"S-{token}"
            mapping[code] = formula_id
            self.connection.execute("INSERT INTO samples VALUES (?, ?, ?)", (study_id, formula_id, code))
        self.connection.commit()
        return {"study_id": study_id, **mapping}

    @staticmethod
    def _validate_score(value: float, name: str) -> float:
        result = float(value)
        if not 0 <= result <= 100:
            raise ValueError(f"{name} must be between 0 and 100")
        return result

    def record_evaluation(
        self,
        study_id: str,
        blind_code: str,
        panelist_id: str,
        similarity: float,
        liking: float,
        dimension_scores: dict[str, float] | None = None,
        defects: list[str] | None = None,
        expert: bool = False,
        source: str = "human",
        evidence_ref: str | None = None,
        _commit: bool = True,
    ) -> None:
        """Record direct entry as ``legacy_unverified`` audit data.

        ``source`` remains accepted for API compatibility but a self-asserted
        human source is not treated as external human evidence.
        """
        if not panelist_id.strip():
            raise ValueError("a pseudonymous panelist_id is required")
        similarity = self._validate_score(similarity, "similarity")
        liking = self._validate_score(liking, "liking")
        exists = self.connection.execute(
            "SELECT 1 FROM samples WHERE study_id = ? AND blind_code = ?", (study_id, blind_code)
        ).fetchone()
        if not exists:
            raise ValueError("unknown study/blind code")
        if str(source).strip().casefold() not in {"human", "legacy_unverified"}:
            raise ValueError("direct evaluation may only claim human or legacy_unverified source")
        if not (evidence_ref or "").strip():
            study = self.connection.execute("SELECT protocol_ref FROM studies WHERE study_id = ?", (study_id,)).fetchone()
            evidence_ref = str(study[0]) if study else ""
        if not str(evidence_ref or "").strip():
            raise ValueError("evidence_ref or study protocol_ref is required")
        dimensions = dimension_scores or {}
        if any(not 0 <= float(value) <= 100 for value in dimensions.values()):
            raise ValueError("dimension scores must be between 0 and 100")
        self.connection.execute(
            """INSERT INTO evaluations
            (study_id, blind_code, panelist_id, expert, similarity, liking,
             dimension_json, defects_json, evaluated_at, source, evidence_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'legacy_unverified', ?)""",
            (
                study_id,
                blind_code,
                panelist_id,
                int(expert),
                similarity,
                liking,
                json.dumps(dimensions, sort_keys=True),
                json.dumps(defects or [], ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                str(evidence_ref),
            ),
        )
        if _commit:
            self.connection.commit()

    def import_human_csv(self, path: str | Path) -> int:
        """Import a historical CSV as audit data, never as a human-evidence pass."""
        source = Path(path)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"study_id", "blind_code", "panelist_id", "similarity", "liking", "source", "evidence_ref"}
        for row in rows:
            missing = required - set(row)
            if missing:
                raise ValueError(f"missing required sensory columns: {sorted(missing)}")
            if str(row["source"]).strip().casefold() != "human":
                raise ValueError("legacy human CSV must explicitly declare source=human")
            if not str(row["evidence_ref"]).strip():
                raise ValueError("evidence_ref is required for every legacy human row")
        with self.connection:
            for row in rows:
                self.record_evaluation(
                    row["study_id"], row["blind_code"], row["panelist_id"],
                    float(row["similarity"]), float(row["liking"]),
                    json.loads(row.get("dimension_json", "{}")),
                    json.loads(row.get("defects_json", "[]")),
                    str(row.get("expert", "false")).casefold() in {"1", "true", "yes"},
                    "human", row["evidence_ref"], False,
                )
        return len(rows)

    def _study_formula_ids(self, study_id: str) -> list[str]:
        values = [
            str(row[0]) for row in self.connection.execute(
                "SELECT formula_id FROM samples WHERE study_id = ? ORDER BY formula_id", (study_id,)
            )
        ]
        if not values:
            raise ValueError("study has no registered formula samples")
        return values

    @staticmethod
    def _read_verified_rows(path: str | Path, study_id: str) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"study_id", "blind_code", "panelist_id", "similarity", "liking"}
        if not rows:
            raise ValueError("signed sensory study artifact contains no rows")
        parsed: list[dict[str, Any]] = []
        for row in rows:
            missing = required - set(row)
            if missing:
                raise ValueError(f"signed sensory study is missing columns: {sorted(missing)}")
            if str(row["study_id"]).strip() != study_id:
                raise ValueError("signed sensory data contains a different study ID")
            panelist = str(row["panelist_id"]).strip()
            code = str(row["blind_code"]).strip()
            if not panelist or not code:
                raise ValueError("signed sensory data requires panelist_id and blind_code")
            dimensions = json.loads(row.get("dimension_json", "{}"))
            defects = json.loads(row.get("defects_json", "[]"))
            if not isinstance(dimensions, dict) or not isinstance(defects, list):
                raise ValueError("signed sensory data has invalid dimensions or defects JSON")
            if any(not 0 <= float(value) <= 100 for value in dimensions.values()):
                raise ValueError("signed sensory dimension scores must be between 0 and 100")
            parsed.append(
                {
                    "blind_code": code,
                    "panelist_id": panelist,
                    "similarity": SensoryEvaluationStore._validate_score(row["similarity"], "similarity"),
                    "liking": SensoryEvaluationStore._validate_score(row["liking"], "liking"),
                    "expert": str(row.get("expert", "false")).casefold() in {"1", "true", "yes"},
                    "dimension_json": json.dumps(dimensions, sort_keys=True),
                    "defects_json": json.dumps(defects, ensure_ascii=False),
                    "evaluated_at": str(row.get("evaluated_at", "")).strip() or datetime.now(timezone.utc).isoformat(),
                }
            )
        if len({(item["blind_code"], item["panelist_id"]) for item in parsed}) != len(parsed):
            raise ValueError("signed sensory data has duplicate sample/panelist evaluations")
        return parsed

    def import_verified(
        self,
        study_id: str,
        data_path: str | Path,
        signed_envelope: Mapping[str, Any],
        *,
        as_of: date | None = None,
    ) -> int:
        """Verify an entire signed human-study dataset and index its rows.

        The signature binds the CSV bytes, exact study ID, and the complete
        formula set.  This prevents reuse of a valid study for a different
        formula, substitution of a single CSV, or row-level self-attestation.
        """
        formula_ids = self._study_formula_ids(study_id)
        verified = self.trust_root.verify(
            signed_envelope,
            {"study_data": data_path},
            expected_artifact_type=SENSORY_ARTIFACT_TYPE,
            expected_scope={"study_id": study_id, "formula_ids": formula_ids},
            allowed_roles=SENSORY_SIGNER_ROLES,
            as_of=as_of,
            local_revocations=self._local_revocations(),
        )
        rows = self._read_verified_rows(data_path, study_id)
        known_codes = {
            str(row[0]) for row in self.connection.execute(
                "SELECT blind_code FROM samples WHERE study_id = ?", (study_id,)
            )
        }
        unknown = sorted({row["blind_code"] for row in rows} - known_codes)
        if unknown:
            raise ValueError(f"signed sensory data contains unknown blind codes: {unknown}")
        now = datetime.now(timezone.utc).isoformat()
        resolved = str(Path(data_path).expanduser().resolve(strict=True))
        with self.connection:
            self.connection.execute(
                """INSERT INTO verified_studies (study_id, artifact_id, data_path, envelope_json, imported_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(study_id) DO UPDATE SET artifact_id=excluded.artifact_id,
                       data_path=excluded.data_path, envelope_json=excluded.envelope_json,
                       imported_at=excluded.imported_at""",
                (study_id, verified.artifact_id, resolved,
                 json.dumps(dict(signed_envelope), ensure_ascii=False, sort_keys=True), now),
            )
            self.connection.execute("DELETE FROM verified_evaluations WHERE study_id = ?", (study_id,))
            for row in rows:
                self.connection.execute(
                    """INSERT INTO verified_evaluations
                    (study_id, blind_code, panelist_id, expert, similarity, liking,
                     dimension_json, defects_json, evaluated_at, artifact_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (study_id, row["blind_code"], row["panelist_id"], int(row["expert"]),
                     row["similarity"], row["liking"], row["dimension_json"], row["defects_json"],
                     row["evaluated_at"], verified.artifact_id),
                )
            self.connection.execute(
                "UPDATE studies SET migration_status = 'verified_external_study' WHERE study_id = ?",
                (study_id,),
            )
        return len(rows)

    def revoke(self, artifact_id: str, revoked_on: date, reason: str) -> None:
        if not str(artifact_id).strip() or not str(reason).strip():
            raise ValueError("artifact_id and revocation reason are required")
        self.connection.execute(
            """INSERT INTO sensory_artifact_revocations VALUES (?, ?, ?)
               ON CONFLICT(artifact_id) DO UPDATE SET revoked_on=excluded.revoked_on, reason=excluded.reason""",
            (str(artifact_id).strip(), revoked_on.isoformat(), str(reason).strip()),
        )
        self.connection.commit()

    def _local_revocations(self) -> dict[str, date]:
        return {
            str(identifier): date.fromisoformat(str(revoked_on))
            for identifier, revoked_on in self.connection.execute(
                "SELECT artifact_id, revoked_on FROM sensory_artifact_revocations"
            )
        }

    def _verified_studies(self, as_of: date | None = None) -> tuple[set[str], tuple[str, ...]]:
        valid: set[str] = set()
        failures: list[str] = []
        revocations = self._local_revocations()
        rows = self.connection.execute(
            "SELECT study_id, artifact_id, data_path, envelope_json FROM verified_studies"
        ).fetchall()
        for study_id, artifact_id, data_path, envelope_json in rows:
            study = str(study_id)
            try:
                self.trust_root.verify(
                    json.loads(str(envelope_json)),
                    {"study_data": str(data_path)},
                    expected_artifact_type=SENSORY_ARTIFACT_TYPE,
                    expected_scope={"study_id": study, "formula_ids": self._study_formula_ids(study)},
                    allowed_roles=SENSORY_SIGNER_ROLES,
                    as_of=as_of,
                    local_revocations=revocations,
                )
                valid.add(study)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{study}/{artifact_id}: {error}")
        return valid, tuple(failures)

    def _has_legacy_rows(self, formula_id: str) -> bool:
        return bool(self.connection.execute(
            """SELECT 1 FROM evaluations e JOIN samples s
               ON e.study_id = s.study_id AND e.blind_code = s.blind_code
               WHERE s.formula_id = ? LIMIT 1""", (formula_id,)
        ).fetchone())

    def formula_evidence(
        self,
        formula_id: str,
        target_similarity: float = 90.0,
        min_panelists: int = 12,
        min_experts: int = 3,
        as_of: date | None = None,
    ) -> SensoryEvidence:
        valid_studies, failures = self._verified_studies(as_of)
        if not valid_studies:
            legacy = self._has_legacy_rows(formula_id)
            return SensoryEvidence(
                formula_id=formula_id,
                unique_panelists=0,
                mean_similarity=None,
                standard_deviation=None,
                lower_confidence_bound_95=None,
                target_similarity=target_similarity,
                passed=False,
                status=(
                    "verified_evidence_invalid" if failures
                    else "legacy_unverified" if legacy else "not_tested"
                ),
                expert_panelists=0,
                evidence_source="legacy_unverified" if legacy else "no_verified_evidence",
                verification_failures=failures,
            )
        placeholders = ",".join("?" for _ in valid_studies)
        rows = self.connection.execute(
            f"""SELECT e.panelist_id, e.similarity, e.expert
                FROM verified_evaluations e
                JOIN samples s ON s.study_id = e.study_id AND s.blind_code = e.blind_code
                WHERE s.formula_id = ? AND e.study_id IN ({placeholders})""",
            (formula_id, *sorted(valid_studies)),
        ).fetchall()
        by_panelist: dict[str, list[float]] = {}
        expert_ids: set[str] = set()
        for panelist, score, expert in rows:
            panelist = str(panelist)
            by_panelist.setdefault(panelist, []).append(float(score))
            if int(expert):
                expert_ids.add(panelist)
        scores = [statistics.mean(values) for values in by_panelist.values()]
        if not scores:
            return SensoryEvidence(
                formula_id, 0, None, None, None, target_similarity, False,
                "not_tested", 0, "verified_external_human_study_only", failures,
            )
        average = statistics.mean(scores)
        deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lower_bound = average - 1.96 * deviation / math.sqrt(len(scores))
        enough = len(scores) >= min_panelists
        expert_enough = len(expert_ids) >= min_experts
        passed = enough and expert_enough and lower_bound >= target_similarity
        if passed:
            status = "verified_passed"
        elif not enough:
            status = "insufficient_verified_panel"
        elif not expert_enough:
            status = "insufficient_verified_experts"
        else:
            status = "below_target"
        return SensoryEvidence(
            formula_id, len(scores), round(average, 4), round(deviation, 4),
            round(lower_bound, 4), target_similarity, passed, status,
            len(expert_ids), "verified_external_human_study_only", failures,
        )

    def _verified_calibration_rows(self, as_of: date | None = None) -> tuple[list[tuple], tuple[str, ...]]:
        valid_studies, failures = self._verified_studies(as_of)
        if not valid_studies:
            return [], failures
        placeholders = ",".join("?" for _ in valid_studies)
        rows = self.connection.execute(
            f"""SELECT f.formula_id, f.predicted_similarity, e.similarity
                FROM formulas f JOIN samples s ON s.formula_id = f.formula_id
                JOIN verified_evaluations e ON e.study_id = s.study_id AND e.blind_code = s.blind_code
                WHERE e.study_id IN ({placeholders})
                ORDER BY f.formula_id, e.panelist_id""",
            tuple(sorted(valid_studies)),
        ).fetchall()
        return rows, failures

    def fit_calibrator(
        self,
        minimum_observations: int = 30,
        minimum_formulas: int = 10,
        maximum_holdout_mae: float = 10.0,
        as_of: date | None = None,
    ) -> CalibrationArtifact:
        """Fit only currently valid signed external-human study observations."""
        rows, _ = self._verified_calibration_rows(as_of)
        unique_formulas = len({row[0] for row in rows})
        now = datetime.now(timezone.utc).isoformat()
        data_fingerprint = hashlib.sha256(
            json.dumps(rows, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()
        if len(rows) < minimum_observations or unique_formulas < minimum_formulas:
            return CalibrationArtifact(
                1.0, 0.0, len(rows), unique_formulas, None,
                "insufficient_verified_data", now, data_fingerprint=data_fingerprint,
            ).sealed()
        train = [row for row in rows if int(hashlib.sha256(str(row[0]).encode()).hexdigest(), 16) % 5 != 0]
        holdout = [row for row in rows if row not in train]
        if not train or not holdout:
            return CalibrationArtifact(
                1.0, 0.0, len(rows), unique_formulas, None,
                "insufficient_verified_holdout", now, data_fingerprint=data_fingerprint,
            ).sealed()
        x = np.asarray([float(row[1]) for row in train], dtype=float)
        y = np.asarray([float(row[2]) for row in train], dtype=float)
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        slope = max(0.0, min(2.0, float(slope)))
        intercept = max(-100.0, min(100.0, float(intercept)))
        errors = [
            abs(max(0.0, min(100.0, slope * float(row[1]) + intercept)) - float(row[2]))
            for row in holdout
        ]
        holdout_mae = statistics.mean(errors)
        status = "validated" if holdout_mae <= maximum_holdout_mae else "failed_verified_holdout"
        return CalibrationArtifact(
            round(slope, 8), round(intercept, 8), len(rows), unique_formulas,
            round(holdout_mae, 4), status, now, data_fingerprint=data_fingerprint,
        ).sealed()

    def close(self) -> None:
        self.connection.close()
