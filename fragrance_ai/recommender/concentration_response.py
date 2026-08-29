"""Fail-closed adapter for the measured Ravia dilution/intensity response."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact_trust import EvidenceTrustRoot
from .models import Ingredient, RecipeLine, profile_vector


CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE = (
    "concentration_response_model_authorization"
)


@dataclass(frozen=True, init=False)
class ConcentrationResponseTrustDecision:
    """Externally verified permission for the concentration curve to score.

    The bundled manifest is deliberately *not* an authority to change the
    primary similarity score: it is shipped alongside the artifact it
    describes.  A deployment must construct this decision with
    :meth:`from_signed_authorization`, which verifies a detached Ed25519
    envelope against an operator-provided trust root and binds it to both the
    manifest and model bytes.  Supplying no decision is the safe default and
    retains the curve for diagnostics only.
    """

    model_sha256: str
    manifest_sha256: str
    approved_primary_score_weight: float
    authorization_artifact_id: str
    signer_id: str
    signer_key_sha256: str
    issued_at: str
    expires_at: str

    @classmethod
    def from_signed_authorization(
        cls,
        trust_root: EvidenceTrustRoot | Mapping[str, Any],
        envelope: Mapping[str, Any],
        *,
        manifest_path: str | Path | None = None,
        runtime_path: str | Path | None = None,
        additional_artifact_paths: Mapping[str, str | Path] | None = None,
        expected_scope_extra: Mapping[str, Any] | None = None,
        allowed_roles: set[str] | frozenset[str] | None = None,
        as_of: date | datetime | None = None,
    ) -> "ConcentrationResponseTrustDecision":
        """Verify an independent model-release authorization.

        The signed scope must bind the exact package manifest and model hashes,
        the approved algorithm, and a bounded primary-score weight.  The
        envelope itself is external deployment evidence; no bundled file can
        manufacture this decision.
        """
        root = (
            trust_root
            if isinstance(trust_root, EvidenceTrustRoot)
            else EvidenceTrustRoot(trust_root)
        )
        if runtime_path is not None and manifest_path is None:
            raise ValueError("runtime_path requires manifest_path")
        if manifest_path is None:
            manifest_ref = (
                resources.files("fragrance_ai")
                .joinpath("data")
                .joinpath(FrozenConcentrationResponse.MANIFEST)
            )
            manifest_bytes = manifest_ref.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            runtime_ref = (
                resources.files("fragrance_ai")
                .joinpath("data")
                .joinpath(str(manifest["runtime_file"]))
            )
            runtime_bytes = runtime_ref.read_bytes()
            external_paths = None
        else:
            manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
            manifest_bytes = manifest_file.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            runtime_file = (
                Path(runtime_path).expanduser().resolve(strict=True)
                if runtime_path is not None
                else manifest_file.with_name(str(manifest["runtime_file"]))
            )
            runtime_file = runtime_file.resolve(strict=True)
            if runtime_file.name != str(manifest.get("runtime_file", "")):
                raise ValueError("external runtime filename differs from model manifest")
            runtime_bytes = runtime_file.read_bytes()
            external_paths = {"manifest": manifest_file, "model": runtime_file}
        model_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if model_sha256 != str(manifest.get("runtime_sha256", "")).lower():
            raise ValueError("concentration runtime does not match its manifest")
        scope = envelope.get("scope")
        if not isinstance(scope, Mapping):
            raise ValueError("model authorization scope must be an object")
        try:
            approved_weight = float(scope["approved_primary_score_weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "model authorization must specify an approved weight"
            ) from error
        if (
            not 0.0
            < approved_weight
            <= FrozenConcentrationResponse.MAX_PRIMARY_SCORE_WEIGHT
        ):
            raise ValueError("approved concentration-response weight is out of bounds")
        expected_scope = {
            "model_sha256": model_sha256,
            "manifest_sha256": manifest_sha256,
            "algorithm": manifest.get("algorithm"),
            "approved_primary_score_weight": approved_weight,
        }
        for key, value in dict(expected_scope_extra or {}).items():
            if key in expected_scope and expected_scope[key] != value:
                raise ValueError(f"extra authorization scope conflicts with {key}")
            expected_scope[key] = value
        extra_paths = dict(additional_artifact_paths or {})
        if set(extra_paths) & {"manifest", "model"}:
            raise ValueError("additional artifact labels conflict with manifest/model")
        accepted_roles = allowed_roles or {"model_release_approver"}
        if external_paths is not None:
            verified = root.verify(
                envelope,
                {**external_paths, **extra_paths},
                expected_artifact_type=CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
                expected_scope=expected_scope,
                allowed_roles=accepted_roles,
                as_of=as_of,
            )
        else:
            with (
                resources.as_file(manifest_ref) as builtin_manifest_path,
                resources.as_file(runtime_ref) as builtin_runtime_path,
            ):
                verified = root.verify(
                    envelope,
                    {
                        "manifest": builtin_manifest_path,
                        "model": builtin_runtime_path,
                        **extra_paths,
                    },
                    expected_artifact_type=CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
                    expected_scope=expected_scope,
                    allowed_roles=accepted_roles,
                    as_of=as_of,
                )
        decision = object.__new__(cls)
        object.__setattr__(decision, "model_sha256", model_sha256)
        object.__setattr__(decision, "manifest_sha256", manifest_sha256)
        object.__setattr__(decision, "approved_primary_score_weight", approved_weight)
        object.__setattr__(decision, "authorization_artifact_id", verified.artifact_id)
        object.__setattr__(decision, "signer_id", verified.signer_id)
        object.__setattr__(
            decision, "signer_key_sha256", verified.signer_key_sha256
        )
        object.__setattr__(decision, "issued_at", verified.issued_at)
        object.__setattr__(decision, "expires_at", verified.expires_at)
        return decision


@dataclass(frozen=True)
class ConcentrationResponseResult:
    status: str
    profile: np.ndarray
    total_relative_intensity: float
    evidence_coverage_percent: float
    flags: tuple[str, ...]
    approved_primary_score_weight: float = 0.0


class FrozenConcentrationResponse:
    MANIFEST = "concentration_response_manifest.json"
    MAX_PRIMARY_SCORE_WEIGHT = 0.05

    def __init__(
        self,
        trust_decision: ConcentrationResponseTrustDecision | None = None,
        *,
        manifest_path: str | Path | None = None,
        runtime_path: str | Path | None = None,
    ) -> None:
        if runtime_path is not None and manifest_path is None:
            raise ValueError("runtime_path requires manifest_path")
        self._loaded = False
        self._load_error: str | None = None
        self._portable_parameters: dict[str, np.ndarray | float] | None = None
        self._minimum = 1e-4
        self._maximum = 1.0
        self._trust_decision = trust_decision
        self._model_sha256 = ""
        self._manifest_sha256 = ""
        self._approved_primary_score_weight = 0.0
        self._authorization_error = "independent_signed_artifact_authorization_missing"
        self._manifest_path = (
            Path(manifest_path).expanduser().resolve(strict=True)
            if manifest_path is not None
            else None
        )
        self._runtime_path = (
            Path(runtime_path).expanduser().resolve(strict=True)
            if runtime_path is not None
            else None
        )

    @staticmethod
    def _resource(name: str) -> bytes:
        return (
            resources.files("fragrance_ai").joinpath("data").joinpath(name).read_bytes()
        )

    def _manifest_bytes(self) -> bytes:
        if self._manifest_path is not None:
            return self._manifest_path.read_bytes()
        return self._resource(self.MANIFEST)

    def _runtime_bytes(self, name: str) -> bytes:
        if self._manifest_path is None:
            return self._resource(name)
        path = self._runtime_path or self._manifest_path.with_name(name)
        path = path.resolve(strict=True)
        if path.name != name:
            raise RuntimeError("external runtime filename differs from model manifest")
        return path.read_bytes()

    def _load(self) -> None:
        if self._loaded or self._load_error is not None:
            return
        try:
            manifest_bytes = self._manifest_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            release = manifest.get("release_gate", {})
            checks = release.get("checks", {})
            if (
                not release.get("passed")
                or not checks
                or not all(bool(value) for value in checks.values())
            ):
                raise RuntimeError("concentration response release gate failed")
            if manifest.get("algorithm") != "concentration_only_ridge":
                raise RuntimeError("unapproved molecule-specific concentration model")
            if float(manifest.get("structure_specific_weight", 1.0)) != 0.0:
                raise RuntimeError(
                    "structure-specific concentration weight must remain zero"
                )
            if manifest.get("schema_version") != "1.1":
                raise RuntimeError("unsupported concentration manifest schema")
            distribution = manifest.get("distribution_contract", {})
            if distribution != {
                "runtime_format": "json_numeric_arrays_only",
                "source_model_packaged": False,
                "source_model_required_at_runtime": False,
                "pickle_deserialization_allowed": False,
            }:
                raise RuntimeError("unsafe concentration distribution contract")
            runtime_bytes = self._runtime_bytes(str(manifest["runtime_file"]))
            self._model_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
            self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if self._model_sha256 != str(manifest["runtime_sha256"]).lower():
                raise RuntimeError("concentration runtime artifact hash mismatch")
            portable = json.loads(runtime_bytes.decode("utf-8"))
            if not isinstance(portable, Mapping):
                raise RuntimeError("portable concentration-response parameters missing")
            if (
                portable.get("schema_version") != "1.0"
                or portable.get("runtime") != "numpy_concentration_response_v1"
                or portable.get("format")
                != "standard_scaler_plus_ridge_coefficients_v1"
                or portable.get("feature_contract")
                != ["log10_dilution", "log10_dilution_squared"]
                or portable.get("allow_pickle") is not False
                or portable.get("source_training_artifact_required_at_runtime")
                is not False
                or portable.get("source_training_artifact_sha256")
                != manifest.get("source_model_sha256")
            ):
                raise RuntimeError("portable concentration-response contract changed")
            mean = np.asarray(portable.get("feature_mean", []), dtype=float)
            scale = np.asarray(portable.get("feature_scale", []), dtype=float)
            coefficients = np.asarray(portable.get("coefficients", []), dtype=float)
            intercept = float(portable.get("intercept"))
            if mean.shape != (2,) or scale.shape != (2,) or coefficients.shape != (2,):
                raise RuntimeError("portable concentration-response shape mismatch")
            if (
                not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale))
                or not np.all(np.isfinite(coefficients))
                or not math.isfinite(intercept)
                or np.any(scale <= 0.0)
            ):
                raise RuntimeError("portable concentration-response values are invalid")
            self._portable_parameters = {
                "mean": mean,
                "scale": scale,
                "coefficients": coefficients,
                "intercept": intercept,
            }
            runtime_range = tuple(
                float(value) for value in portable["dilution_range_fraction"]
            )
            manifest_range = tuple(
                float(value) for value in manifest["dilution_range_fraction"]
            )
            if runtime_range != manifest_range or len(runtime_range) != 2:
                raise RuntimeError("concentration dilution range contract mismatch")
            self._minimum, self._maximum = runtime_range
            if (
                not math.isfinite(self._minimum)
                or not math.isfinite(self._maximum)
                or not 0.0 < self._minimum < self._maximum <= 1.0
            ):
                raise RuntimeError("concentration dilution range is invalid")
            decision = self._trust_decision
            if decision is not None:
                if (
                    decision.model_sha256.lower() != self._model_sha256
                    or decision.manifest_sha256.lower() != self._manifest_sha256
                ):
                    self._authorization_error = (
                        "signed_authorization_artifact_hash_mismatch"
                    )
                elif not decision.authorization_artifact_id or not decision.signer_id:
                    self._authorization_error = "signed_authorization_identity_missing"
                elif not (
                    0.0
                    < decision.approved_primary_score_weight
                    <= self.MAX_PRIMARY_SCORE_WEIGHT
                ):
                    self._authorization_error = (
                        "signed_authorization_weight_out_of_bounds"
                    )
                else:
                    self._approved_primary_score_weight = (
                        decision.approved_primary_score_weight
                    )
                    self._authorization_error = ""
            self._loaded = True
        except Exception as error:  # Optional scientific dependency, fail closed.
            self._load_error = f"{type(error).__name__}:{error}"

    @property
    def approved_primary_score_weight(self) -> float:
        """Return a nonzero score weight only after external authorization."""
        self._load()
        return round(
            self._approved_primary_score_weight
            if self._authorization_is_current()
            else 0.0,
            4,
        )

    def _authorization_is_current(self) -> bool:
        decision = self._trust_decision
        if self._approved_primary_score_weight <= 0.0 or decision is None:
            return False
        try:
            expires = datetime.fromisoformat(decision.expires_at)
        except (AttributeError, TypeError, ValueError):
            return False
        if expires.tzinfo is None:
            return False
        return datetime.now(timezone.utc) <= expires.astimezone(timezone.utc)

    def _authorization_flags(self) -> tuple[str, ...]:
        if self._authorization_is_current():
            return (
                "independent_signed_artifact_authorization_verified",
                "concentration_response_authorized_for_primary_score",
            )
        expiration = (
            "signed_authorization_expired_or_invalid_time"
            if self._approved_primary_score_weight > 0.0
            else self._authorization_error
        )
        return (
            expiration
            or "independent_signed_artifact_authorization_missing",
            "concentration_response_diagnostic_only_weight_zero",
        )

    def intensity(self, dilution_fraction: float) -> tuple[float, bool]:
        self._load()
        if not self._loaded or self._portable_parameters is None:
            return 0.0, False
        in_domain = self._minimum <= dilution_fraction <= self._maximum
        concentration = float(np.clip(dilution_fraction, self._minimum, self._maximum))
        log_c = math.log10(concentration)
        features = np.asarray([log_c, log_c * log_c], dtype=float)
        mean = self._portable_parameters["mean"]
        scale = self._portable_parameters["scale"]
        coefficients = self._portable_parameters["coefficients"]
        intercept = float(self._portable_parameters["intercept"])
        prediction = float(((features - mean) / scale) @ coefficients + intercept)
        return float(np.clip(prediction, 0.0, 100.0)), in_domain

    def formula_profile(
        self,
        lines: list[RecipeLine],
        ingredients: dict[str, Ingredient],
    ) -> ConcentrationResponseResult:
        self._load()
        if not self._loaded:
            return ConcentrationResponseResult(
                status="unavailable",
                profile=np.zeros(19, dtype=float),
                total_relative_intensity=0.0,
                evidence_coverage_percent=0.0,
                flags=(
                    "concentration_response_not_applied",
                    self._load_error or "unknown",
                    *self._authorization_flags(),
                ),
            )
        weighted = np.zeros(19, dtype=float)
        total = 0.0
        covered_mass = 0.0
        formula_mass = 0.0
        for line in lines:
            ingredient = ingredients.get(line.ingredient_id)
            if ingredient is None:
                continue
            active_finished_percent = (
                max(0.0, line.finished_product_percent)
                * max(0.0, line.active_strength_percent)
                / 100.0
            )
            formula_mass += active_finished_percent
            intensity, in_domain = self.intensity(active_finished_percent / 100.0)
            if in_domain:
                covered_mass += active_finished_percent
            # The measured intensity already contains the concentration effect;
            # multiplying by concentration again would double count dilution.
            weighted += profile_vector(ingredient.profile) * intensity
            total += intensity
        if total <= 0:
            return ConcentrationResponseResult(
                status="outside_applicability",
                profile=np.zeros(19, dtype=float),
                total_relative_intensity=0.0,
                evidence_coverage_percent=0.0,
                flags=(
                    "no_concentration_response_in_domain",
                    *self._authorization_flags(),
                ),
            )
        coverage = covered_mass / max(1e-12, formula_mass) * 100.0
        approved_weight = self.approved_primary_score_weight
        flags = [
            "ravia_monomolecular_intensity_calibration",
            "molecule_specific_concentration_modulation_weight_zero",
            "not_a_mixture_similarity_label",
        ]
        if coverage < 99.999:
            flags.append("some_formula_concentrations_clamped_to_measured_domain")
        flags.extend(self._authorization_flags())
        return ConcentrationResponseResult(
            status=(
                "validation_gated"
                if coverage >= 70.0 and approved_weight > 0.0
                else (
                    "validation_gated_diagnostic_only"
                    if coverage >= 70.0
                    else "outside_applicability"
                )
            ),
            profile=weighted / total,
            total_relative_intensity=round(total, 6),
            evidence_coverage_percent=round(coverage, 4),
            flags=tuple(flags),
            approved_primary_score_weight=approved_weight,
        )


def concentration_response_from_environment() -> FrozenConcentrationResponse:
    """Load a signed continual champion when deployment explicitly enables it.

    Partial configuration is an operator error and fails startup.  With neither
    variable present, the immutable bundled diagnostic model remains active at
    zero primary-score weight.
    """

    state_path = os.environ.get("PERFUMERY_AI_CONTINUAL_STATE", "").strip()
    trust_root_path = os.environ.get(
        "PERFUMERY_AI_CONTINUAL_TRUST_ROOT", ""
    ).strip()
    if not state_path and not trust_root_path:
        return FrozenConcentrationResponse()
    if not state_path or not trust_root_path:
        raise RuntimeError(
            "PERFUMERY_AI_CONTINUAL_STATE and PERFUMERY_AI_CONTINUAL_TRUST_ROOT "
            "must be configured together"
        )
    # Local import prevents a module cycle: the continual controller itself
    # validates FrozenConcentrationResponse artifacts.
    from .continuous_improvement import load_production_concentration_response

    return load_production_concentration_response(state_path, trust_root_path)
