"""Human-behavior calibration with explicit scope and conformal abstention.

The bundled artifact is derived only from the pre-declared Bushdid calibration
partition.  It predicts odd-one-out discrimination probability for equal-
presence molecular mixtures; it does not turn that endpoint into perfume smell
similarity.  Formula requests outside the exact study scope abstain.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .models import RecipeLine


@dataclass(frozen=True)
class HumanCalibrationResult:
    status: str
    discrimination_probability: float | None
    lower_95: float | None
    upper_95: float | None
    applicability_percent: float
    artifact_id: str
    flags: tuple[str, ...]
    similarity_90_claim_authorized: bool = False


class HumanMixtureCalibration:
    """Apply a versioned human calibration artifact only inside its scope."""

    ARTIFACT = "human_mixture_calibration.json"
    EXPECTED_SOURCE_HASHES = {
        "blind_report_sha256": (
            "4f0c94b6f7d5896793fba3a8e269f558380a90bb6dd98a5bd40afe641e7c2849"
        ),
        "sealed_prediction_sha256": (
            "9e254e9c00c5835f54d1167834445e8e19b9c2d82dfb1f1b7c132776fb254b69"
        ),
        "stimulus_protocol_sha256": (
            "6e375a0ef6190b31b44d930f23c70dc193cc61997f83bcac73817c92b405d8be"
        ),
    }

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self._artifact_path = Path(artifact_path) if artifact_path else None
        self._loaded = False
        self._error = ""
        self._artifact: dict[str, Any] = {}
        self._artifact_id = ""

    def _read(self) -> bytes:
        if self._artifact_path is not None:
            return self._artifact_path.expanduser().resolve(strict=True).read_bytes()
        return (
            resources.files("fragrance_ai")
            .joinpath("data")
            .joinpath(self.ARTIFACT)
            .read_bytes()
        )

    def _load(self) -> None:
        if self._loaded or self._error:
            return
        try:
            payload_bytes = self._read()
            payload = json.loads(payload_bytes.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("human calibration artifact must be an object")
            if payload.get("schema_version") != "2.0":
                raise ValueError("unsupported human calibration schema")
            if payload.get("artifact_name") != (
                "bushdid_protocol_aware_human_calibration_v2"
            ):
                raise ValueError("unsupported human calibration artifact")
            if payload.get("endpoint") != "three_alternative_odd_one_out_correct_rate":
                raise ValueError("unsupported human calibration endpoint")
            calibration = payload.get("calibration")
            scope = payload.get("applicability_scope")
            validation = payload.get("historical_final_evaluation")
            feature_contract = payload.get("feature_contract")
            source = payload.get("source")
            if not isinstance(calibration, Mapping) or not isinstance(scope, Mapping):
                raise ValueError("human calibration sections are missing")
            if not isinstance(validation, Mapping):
                raise ValueError("historical final evaluation is missing")
            if not isinstance(feature_contract, Mapping):
                raise ValueError("human calibration feature contract is missing")
            if not isinstance(source, Mapping):
                raise ValueError("human calibration source contract is missing")
            if (
                scope.get("matrix_ids")
                != ["bushdid_2014_equal_presence_molecular_mixture"]
                or scope.get("product_concentration_percent_range")
                != [100.0, 100.0]
                or scope.get("components_per_mixture") != [10, 20, 30]
                or scope.get("study_protocol_id")
                != "bushdid_2014_supplemental_3afc_protocol"
                or scope.get("requires_registered_stimulus_table") is not True
                or scope.get("requires_exact_vial_dilution_design") is not True
                or scope.get("allowed_vial_dilutions") != [0.25, 0.5, 1.0]
                or scope.get("formula_projection_supported") is not False
            ):
                raise ValueError("human calibration applicability contract changed")
            if any(
                source.get(name) != expected
                for name, expected in self.EXPECTED_SOURCE_HASHES.items()
            ):
                raise ValueError("human calibration source hashes are not authorized")
            if validation.get("human_similarity_90_claim_authorized") is not False:
                raise ValueError(
                    "human calibration cannot authorize perfume similarity"
                )
            if (
                source.get("historical_final_labels_used_for_parameter_selection")
                is not False
                or source.get("fit_partition") != "predeclared_calibration_only"
                or source.get("development_timing")
                != "post_unblinding_protocol_model"
            ):
                raise ValueError("historical final labels entered model selection")
            if (
                validation.get("stimuli") != 208
                or validation.get("prospective_external_validation") is not False
            ):
                raise ValueError("human calibration validation contract changed")
            alpha = float(feature_contract.get("selected_alpha", math.nan))
            if not math.isfinite(alpha) or not -2.0 <= alpha <= 0.0:
                raise ValueError("human calibration protocol coefficient is invalid")
            if feature_contract.get("component_overlap_dissimilarity_range") != [
                0.0,
                1.0,
            ]:
                raise ValueError("human calibration dissimilarity range changed")
            if feature_contract.get("dilution_feature") != (
                "population_stddev(log10(wrong_vial_dilutions))"
            ):
                raise ValueError("human calibration dilution feature changed")
            x = np.asarray(calibration.get("x_protocol_score", []), dtype=float)
            y = np.asarray(calibration.get("predicted_correct_rate", []), dtype=float)
            if x.ndim != 1 or y.shape != x.shape or x.size < 2:
                raise ValueError("invalid isotonic calibration arrays")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("non-finite human calibration values")
            if np.any(np.diff(x) < 0) or np.any(np.diff(y) < -1e-12):
                raise ValueError("human calibration must be monotonic")
            if np.any((y < 0.0) | (y > 1.0)):
                raise ValueError("human calibration probabilities are out of range")
            q95 = float(calibration.get("cross_conformal_absolute_error_q95", -1.0))
            if not 0.0 <= q95 <= 1.0:
                raise ValueError("invalid cross-conformal error quantile")
            if (
                int(calibration.get("fold_count", 0)) != 4
                or int(calibration.get("fit_stimuli", 0)) != 52
                or int(calibration.get("crossfit_stimuli", 0)) != 52
            ):
                raise ValueError("human calibration cross-fit contract changed")
            self._artifact = payload
            self._artifact_id = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
            self._loaded = True
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            self._error = f"{type(error).__name__}:{error}"

    def unavailable(self, reason: str) -> HumanCalibrationResult:
        self._load()
        return HumanCalibrationResult(
            status=reason,
            discrimination_probability=None,
            lower_95=None,
            upper_95=None,
            applicability_percent=0.0,
            artifact_id=self._artifact_id,
            flags=(
                "human_endpoint_not_identifiable_for_this_request",
                self._error or reason,
            ),
        )

    def compare(
        self,
        candidate: list[RecipeLine],
        target: list[RecipeLine] | None,
        *,
        matrix_id: str,
        product_concentration_percent: float,
    ) -> HumanCalibrationResult:
        self._load()
        if not self._loaded:
            return self.unavailable("calibration_unavailable")
        if not target:
            return self.unavailable("abstained_no_evidenced_target")
        # RecipeLine does not carry the Bushdid stimulus-table row, per-molecule
        # stock dilution/solvent, or right/wrong vial dilution. Treating equal
        # percentages as that experiment would be an invalid domain expansion.
        return HumanCalibrationResult(
            "abstained_formula_endpoint_not_supported",
            None,
            None,
            None,
            0.0,
            self._artifact_id,
            (
                "actual_human_behavior_calibration_available_for_study_audit",
                "recipe_lines_cannot_reconstruct_bushdid_preparation_protocol",
                "endpoint_is_discrimination_probability_not_perfume_similarity",
                f"requested_matrix:{matrix_id}",
                f"requested_product_concentration:{product_concentration_percent}",
            ),
        )

    def predict_study_endpoint(
        self,
        component_overlap_dissimilarity: float,
        *,
        study_protocol_id: str,
        source_report_sha256: str,
        stimulus_protocol_sha256: str | None = None,
        components_per_mixture: int | None = None,
        right_dilution: float | None = None,
        wrong_dilutions: Sequence[float] | None = None,
    ) -> HumanCalibrationResult:
        """Predict only when auditing the exact registered study protocol."""

        self._load()
        if not self._loaded:
            return self.unavailable("calibration_unavailable")
        scope = self._artifact["applicability_scope"]
        flags = [
            "actual_human_behavior_calibration",
            "endpoint_is_discrimination_probability_not_perfume_similarity",
            "registered_study_protocol_required",
            "protocol_dilution_nuisance_adjusted",
            "retrospective_protocol_model_not_prospective_external_validation",
        ]
        if study_protocol_id != scope.get(
            "study_protocol_id"
        ) or source_report_sha256 != self._artifact.get("source", {}).get(
            "blind_report_sha256"
        ):
            return HumanCalibrationResult(
                "abstained_study_protocol_mismatch",
                None,
                None,
                None,
                0.0,
                self._artifact_id,
                tuple((*flags, "study_protocol_or_source_report_hash_mismatch")),
            )
        if stimulus_protocol_sha256 != self._artifact.get("source", {}).get(
            "stimulus_protocol_sha256"
        ):
            return HumanCalibrationResult(
                "abstained_stimulus_protocol_hash_mismatch",
                None,
                None,
                None,
                0.0,
                self._artifact_id,
                tuple((*flags, "registered_stimulus_protocol_hash_required")),
            )
        if (
            components_per_mixture is None
            or right_dilution is None
            or wrong_dilutions is None
        ):
            return HumanCalibrationResult(
                "abstained_required_stimulus_protocol_features_missing",
                None,
                None,
                None,
                0.0,
                self._artifact_id,
                tuple((*flags, "exact_vial_dilution_design_required")),
            )
        try:
            component_count_value = float(components_per_mixture)
            right = float(right_dilution)
            wrong = tuple(float(value) for value in wrong_dilutions)
        except (TypeError, ValueError) as error:
            raise ValueError("stimulus protocol features must be numeric") from error
        if (
            isinstance(components_per_mixture, bool)
            or not math.isfinite(component_count_value)
            or not component_count_value.is_integer()
        ):
            raise ValueError("components_per_mixture must be a finite integer")
        component_count = int(component_count_value)
        allowed_components = tuple(
            int(value) for value in scope.get("components_per_mixture", [])
        )
        if component_count not in allowed_components:
            return HumanCalibrationResult(
                "abstained_stimulus_protocol_outside_scope",
                None,
                None,
                None,
                0.0,
                self._artifact_id,
                tuple((*flags, "components_per_mixture_outside_registered_scope")),
            )
        if (
            len(wrong) != 2
            or not all(
                math.isfinite(value) and value > 0.0 for value in (right, *wrong)
            )
            or wrong[0] == wrong[1]
        ):
            raise ValueError("vial dilutions must be distinct positive finite values")
        allowed_dilutions = sorted(
            float(value) for value in scope.get("allowed_vial_dilutions", [])
        )
        if len(allowed_dilutions) != 3 or not np.allclose(
            sorted((right, *wrong)), allowed_dilutions, atol=1e-12, rtol=0.0
        ):
            return HumanCalibrationResult(
                "abstained_stimulus_protocol_outside_scope",
                None,
                None,
                None,
                0.0,
                self._artifact_id,
                tuple((*flags, "vial_dilution_design_outside_registered_scope")),
            )
        dissimilarity = float(component_overlap_dissimilarity)
        if not math.isfinite(dissimilarity) or not 0.0 <= dissimilarity <= 1.0:
            raise ValueError("component_overlap_dissimilarity must be in [0, 1]")
        calibration = self._artifact["calibration"]
        feature_contract = self._artifact["feature_contract"]
        alpha = float(feature_contract["selected_alpha"])
        dilution_spread = float(np.std(np.log10(np.asarray(wrong, dtype=float))))
        protocol_score = dissimilarity + alpha * dilution_spread
        x = np.asarray(calibration["x_protocol_score"], dtype=float)
        y = np.asarray(calibration["predicted_correct_rate"], dtype=float)
        prediction = float(np.interp(protocol_score, x, y))
        q95 = float(calibration["cross_conformal_absolute_error_q95"])
        lower = max(0.0, prediction - q95)
        upper = min(1.0, prediction + q95)
        claim_authorized = bool(
            self._artifact["historical_final_evaluation"].get(
                "human_similarity_90_claim_authorized", False
            )
        )
        if not claim_authorized:
            flags.append(
                "historical_validation_does_not_authorize_90_percent_similarity_claim"
            )
        return HumanCalibrationResult(
            status="calibrated_registered_study_endpoint_protocol_aware",
            discrimination_probability=round(prediction, 6),
            lower_95=round(lower, 6),
            upper_95=round(upper, 6),
            applicability_percent=100.0,
            artifact_id=self._artifact_id,
            flags=tuple(flags),
            similarity_90_claim_authorized=claim_authorized,
        )
