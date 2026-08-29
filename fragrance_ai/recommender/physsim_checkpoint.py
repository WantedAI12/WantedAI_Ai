"""Fail-closed runtime adapter for the frozen JCIM R2 checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import numpy as np

from .models import RecipeLine


@dataclass(frozen=True)
class FrozenR2Result:
    status: str
    similarity: float | None
    candidate_structure_coverage_percent: float
    target_structure_coverage_percent: float
    descriptor_domain_coverage_percent: float
    applicability_percent: float
    approved_primary_score_weight: float
    applied_primary_score_weight: float
    neutral_similarity_percent: float
    centered_score_adjustment: float
    checkpoint_sha256: str
    flags: tuple[str, ...]
    member_predictions_percent: tuple[float, ...]
    member_disagreement_percent: float
    prediction_interval_lower_percent: float
    prediction_interval_upper_percent: float
    ensemble_manifest_sha256: str


class FrozenR2PhysSim:
    """Loads model/data resources only when learned inference is requested."""

    CHECKPOINT = "physsim_r2_checkpoint.pt"
    MANIFEST = "physsim_r2_manifest.json"
    COMPONENTS = "r2_ingredient_components.npz"
    COMPONENT_MANIFEST = "r2_ingredient_components_manifest.json"
    ENSEMBLE_MANIFEST = "physsim_r2_ensemble_manifest.json"
    RUNTIME_WEIGHTS = "physsim_r2_runtime_weights.npz"
    RUNTIME_MANIFEST = "physsim_r2_runtime_manifest.json"
    MIN_APPLICABILITY = 70.0

    def __init__(self) -> None:
        self._loaded = False
        self._load_error: str | None = None
        self._models: list[object] = []
        self._ensemble_weights: list[float] = []
        self._normalizer_mean: np.ndarray | None = None
        self._normalizer_std: np.ndarray | None = None
        self._component_rows: dict[str, list[dict[str, object]]] = {}
        self._composition_coverage: dict[str, float] = {}
        self._approved_weight = 0.0
        self._neutral_similarity_percent = 50.0
        self._checkpoint_sha256 = ""
        self._member_checkpoint_sha256: tuple[str, ...] = ()
        self._ensemble_manifest_sha256 = ""
        self._maximum_member_disagreement = 0.0
        self._absolute_error_q95 = 0.0
        self._validation_contract_passed = False
        self._direct_formulation_capability_authorized = False

    @staticmethod
    def _resource_bytes(name: str) -> bytes:
        return (
            resources.files("fragrance_ai").joinpath("data").joinpath(name).read_bytes()
        )

    @staticmethod
    def _sha256(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def _load(self) -> None:
        if self._loaded or self._load_error is not None:
            return
        try:
            from .numpy_r2 import EXPECTED_STATE_SHAPES, NumpyR2Model

            descriptor_dim = 217
            max_molecules = 50

            manifest = json.loads(self._resource_bytes(self.MANIFEST).decode("utf-8"))
            ensemble_manifest_bytes = self._resource_bytes(self.ENSEMBLE_MANIFEST)
            ensemble_manifest = json.loads(ensemble_manifest_bytes.decode("utf-8"))
            self._ensemble_manifest_sha256 = self._sha256(ensemble_manifest_bytes)
            runtime_manifest_bytes = self._resource_bytes(self.RUNTIME_MANIFEST)
            runtime_manifest = json.loads(runtime_manifest_bytes.decode("utf-8"))
            if (
                runtime_manifest.get("schema_version") != "1.0"
                or runtime_manifest.get("runtime") != "numpy_only_r2_inference_v1"
            ):
                raise RuntimeError("unsupported portable R2 runtime contract")
            if runtime_manifest.get("distribution_contract") != {
                "source_serialized_checkpoints_packaged": False,
                "source_serialized_checkpoints_required_at_runtime": False,
                "portable_weights_allow_pickle": False,
            }:
                raise RuntimeError("unsafe portable R2 distribution contract")
            numeric_contract = runtime_manifest.get("numeric_contract", {})
            if numeric_contract != {
                "dtype": "float32_weights_float64_dynamics",
                "gelu": "exact_erf",
                "layer_norm_epsilon": 1e-5,
                "dropout": "disabled_inference",
                "torch_reference_required_at_build_time_only": True,
            }:
                raise RuntimeError("portable R2 numeric contract mismatch")
            equivalence = runtime_manifest.get("numeric_equivalence", {})
            if (
                not bool(equivalence.get("passed"))
                or int(equivalence.get("cases", 0)) < 2
                or float(equivalence.get("maximum_absolute_error", 1.0))
                > float(equivalence.get("tolerance", 0.0))
                or float(equivalence.get("tolerance", 0.0)) > 1e-5
            ):
                raise RuntimeError("portable R2 numeric equivalence gate failed")
            runtime_weights_bytes = self._resource_bytes(self.RUNTIME_WEIGHTS)
            if self._sha256(runtime_weights_bytes) != runtime_manifest.get(
                "artifact_sha256"
            ):
                raise RuntimeError("portable R2 runtime artifact hash mismatch")
            if runtime_manifest.get("ensemble_manifest_sha256") != (
                self._ensemble_manifest_sha256
            ):
                raise RuntimeError("portable R2 runtime is bound to another ensemble")
            component_bytes = self._resource_bytes(self.COMPONENTS)
            component_manifest = json.loads(
                self._resource_bytes(self.COMPONENT_MANIFEST).decode("utf-8")
            )
            if self._sha256(component_bytes) != component_manifest["artifact_sha256"]:
                raise RuntimeError(
                    "ingredient component sha256 does not match manifest"
                )
            contracts = {
                manifest.get("descriptor_contract_sha256"),
                component_manifest.get("descriptor_contract_sha256"),
                ensemble_manifest.get("descriptor_contract_sha256"),
                runtime_manifest.get("descriptor_contract_sha256"),
            }
            if len(contracts) != 1:
                raise RuntimeError(
                    "ensemble/checkpoint/component descriptor contracts differ"
                )
            ensemble_release = ensemble_manifest.get("release_gate", {})
            ensemble_checks = ensemble_release.get("checks", {})
            if (
                not bool(ensemble_release.get("passed"))
                or not ensemble_checks
                or not all(bool(value) for value in ensemble_checks.values())
            ):
                raise RuntimeError("ensemble validation release gate failed")
            members = ensemble_manifest.get("members", [])
            if len(members) < 2:
                raise RuntimeError("ensemble requires at least two members")
            weights = np.asarray([float(member["weight"]) for member in members])
            if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
                raise RuntimeError(
                    "ensemble member weights must be positive and sum to one"
                )
            models: list[NumpyR2Model] = []
            member_hashes = []
            runtime_members = runtime_manifest.get("members", [])
            state_keys = runtime_manifest.get("state_keys", [])
            if (
                len(runtime_members) != len(members)
                or set(state_keys) != set(EXPECTED_STATE_SHAPES)
                or len(state_keys) != len(EXPECTED_STATE_SHAPES)
            ):
                raise RuntimeError("portable R2 ensemble member contract mismatch")
            with np.load(
                io.BytesIO(runtime_weights_bytes), allow_pickle=False
            ) as runtime:
                mean = np.asarray(runtime["normalizer_mean"], dtype=np.float32)
                std = np.asarray(runtime["normalizer_std"], dtype=np.float32)
                if mean.shape != (descriptor_dim,) or std.shape != (descriptor_dim,):
                    raise RuntimeError("portable R2 normalizer shape mismatch")
                for index, (member, runtime_member) in enumerate(
                    zip(members, runtime_members)
                ):
                    checkpoint_sha = str(member["sha256"]).lower()
                    if (
                        len(checkpoint_sha) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in checkpoint_sha
                        )
                        or checkpoint_sha != str(runtime_member["sha256"]).lower()
                        or str(member["file"]) != str(runtime_member["file"])
                        or int(member["model_seed"])
                        != int(runtime_member["model_seed"])
                        or not np.isclose(
                            float(member["weight"]),
                            float(runtime_member["weight"]),
                            atol=1e-12,
                        )
                    ):
                        raise RuntimeError("portable R2 source member mismatch")
                    state = {
                        str(state_key): np.asarray(
                            runtime[f"member_{index}::{state_key}"], dtype=np.float32
                        )
                        for state_key in state_keys
                    }
                    models.append(NumpyR2Model(state))
                    member_hashes.append(checkpoint_sha)
            with np.load(io.BytesIO(component_bytes), allow_pickle=False) as data:
                if data["descriptors"].shape[1] != descriptor_dim:
                    raise RuntimeError("ingredient descriptor dimension mismatch")
                for index, ingredient_id in enumerate(data["ingredient_ids"]):
                    self._component_rows.setdefault(str(ingredient_id), []).append(
                        {
                            "smiles": str(data["smiles"][index]),
                            "fraction": float(data["material_fractions"][index]),
                            "descriptor": np.asarray(
                                data["descriptors"][index], dtype=np.float32
                            ),
                        }
                    )
            self._composition_coverage = {
                str(key): float(value)
                for key, value in component_manifest[
                    "ingredient_composition_coverage_percent"
                ].items()
            }
            release = manifest.get("release_gate", {})
            required_gate_checks = {
                "molecule_cold_improves_baseline",
                "scaffold_cold_improves_baseline",
                "strict_molecule_disjoint_improves_baseline",
                "strict_scaffold_disjoint_improves_baseline",
                "ravia_transfer_improves_baseline",
                "molecule_split_has_zero_component_leakage",
                "scaffold_split_has_zero_component_leakage",
                "strict_molecule_disjoint_has_zero_leakage",
                "strict_scaffold_disjoint_has_zero_leakage",
                "training_had_no_nonfinite_batches",
                "ravia_molecule_and_scaffold_disjoint_from_supervision_and_normalizer",
                "pu_safe_positive_only_descriptor_pretraining",
                "strict_scores_use_all_components_held_out_pairs",
                "r2_direct_formulation_capability_authorizes_primary_score",
            }
            release_checks = release.get("checks", {})
            validation_contract_passed = (
                bool(release.get("passed", False))
                and required_gate_checks.issubset(release_checks)
                and all(bool(release_checks[name]) for name in required_gate_checks)
            )
            capability_contract = manifest.get("capability_contract", {})
            direct_inputs = capability_contract.get("direct_formulation_inputs", {})
            required_direct_inputs = {
                "relative_ingredient_amounts_directly_encoded",
                "finished_product_concentration_directly_encoded",
                "time_or_headspace_trajectory_directly_encoded",
            }
            self._direct_formulation_capability_authorized = (
                capability_contract.get("version") == "direct-formulation-inputs-1.0"
                and required_direct_inputs.issubset(direct_inputs)
                and all(bool(direct_inputs[name]) for name in required_direct_inputs)
                and bool(capability_contract.get("authorized_for_primary_score_weight"))
            )
            self._validation_contract_passed = validation_contract_passed
            calibration = ensemble_manifest.get("calibration", {})
            self._approved_weight = (
                float(calibration.get("approved_primary_score_weight", 0.0))
                if validation_contract_passed
                and self._direct_formulation_capability_authorized
                else 0.0
            )
            self._neutral_similarity_percent = float(
                calibration.get("neutral_similarity_percent", 50.0)
            )
            if not 0.0 <= self._neutral_similarity_percent <= 100.0:
                raise RuntimeError("ensemble neutral similarity is outside [0, 100]")
            uncertainty = ensemble_manifest.get("uncertainty", {})
            self._maximum_member_disagreement = float(
                uncertainty.get("maximum_member_disagreement", 0.0)
            )
            self._absolute_error_q95 = float(uncertainty.get("absolute_error_q95", 0.0))
            if self._maximum_member_disagreement <= 0 or self._absolute_error_q95 <= 0:
                raise RuntimeError("ensemble uncertainty calibration is missing")
            self._checkpoint_sha256 = self._ensemble_manifest_sha256
            self._member_checkpoint_sha256 = tuple(member_hashes)
            self._models = models
            self._ensemble_weights = weights.tolist()
            self._normalizer_mean = mean
            self._normalizer_std = std
            self._max_molecules = max_molecules
            self._loaded = True
        except Exception as error:  # Fail closed on optional deps/artifact drift.
            self._load_error = f"{type(error).__name__}:{error}"

    def _empty_result(self, status: str, *flags: str) -> FrozenR2Result:
        return FrozenR2Result(
            status=status,
            similarity=None,
            candidate_structure_coverage_percent=0.0,
            target_structure_coverage_percent=0.0,
            descriptor_domain_coverage_percent=0.0,
            applicability_percent=0.0,
            approved_primary_score_weight=0.0,
            applied_primary_score_weight=0.0,
            neutral_similarity_percent=self._neutral_similarity_percent,
            centered_score_adjustment=0.0,
            checkpoint_sha256=self._checkpoint_sha256,
            flags=tuple(flags),
            member_predictions_percent=(),
            member_disagreement_percent=0.0,
            prediction_interval_lower_percent=0.0,
            prediction_interval_upper_percent=0.0,
            ensemble_manifest_sha256=self._ensemble_manifest_sha256,
        )

    def _mixture(self, lines: Sequence[RecipeLine]) -> tuple[list[np.ndarray], float]:
        total = sum(max(0.0, float(line.concentrate_percent)) for line in lines)
        if total <= 0:
            return [], 0.0
        coverage = (
            sum(
                max(0.0, float(line.concentrate_percent))
                * self._composition_coverage.get(line.ingredient_id, 0.0)
                / 100.0
                for line in lines
            )
            / total
        )
        # The manuscript architecture is presence-based, not concentration-
        # weighted. Concentration and headspace remain in the independent
        # deterministic branch. Scores below only select/limit the 50 most
        # composition-relevant unique molecular particles.
        merged: dict[str, tuple[float, np.ndarray]] = {}
        for line in lines:
            line_weight = max(0.0, float(line.concentrate_percent))
            for component in self._component_rows.get(line.ingredient_id, []):
                relevance = line_weight * float(component["fraction"])
                smiles = str(component["smiles"])
                previous = merged.get(smiles)
                if previous is None or relevance > previous[0]:
                    merged[smiles] = (
                        relevance,
                        np.asarray(component["descriptor"], dtype=np.float32),
                    )
        ranked = sorted(merged.values(), key=lambda value: -value[0])[
            : self._max_molecules
        ]
        return [descriptor for _, descriptor in ranked], coverage * 100.0

    def evaluate(
        self,
        candidate_lines: Sequence[RecipeLine],
        target_lines: Sequence[RecipeLine],
    ) -> FrozenR2Result:
        self._load()
        if not self._loaded:
            return self._empty_result(
                "unavailable",
                "frozen_r2_checkpoint_not_applied",
                self._load_error or "unknown_load_error",
            )
        assert self._normalizer_mean is not None
        assert self._normalizer_std is not None
        candidate, candidate_coverage = self._mixture(candidate_lines)
        target, target_coverage = self._mixture(target_lines)
        if not candidate or not target:
            return self._empty_result(
                "outside_applicability",
                "formula_has_no_registered_molecular_components",
            )
        all_descriptors = np.asarray([*candidate, *target], dtype=np.float32)
        raw_normalized = (
            all_descriptors.astype(np.float64)
            - self._normalizer_mean.astype(np.float64)
        ) / self._normalizer_std.astype(np.float64)
        finite = np.isfinite(raw_normalized)
        domain_coverage = float(
            100.0 * np.mean(finite & (np.abs(raw_normalized) <= 8.0))
        )
        normalized = np.clip(
            np.nan_to_num(raw_normalized, nan=0.0, posinf=100.0, neginf=-100.0),
            -100.0,
            100.0,
        ).astype(np.float32)
        applicability = (
            min(candidate_coverage, target_coverage) * 0.70 + domain_coverage * 0.30
        )
        shared_size = min(self._max_molecules, max(len(candidate), len(target)))

        def padded(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            array = np.zeros((shared_size, normalized.shape[1]), dtype=np.float32)
            mask = np.zeros(shared_size, dtype=np.float32)
            for index, descriptor in enumerate(values[:shared_size]):
                standardized = (
                    descriptor.astype(np.float64)
                    - self._normalizer_mean.astype(np.float64)
                ) / self._normalizer_std.astype(np.float64)
                array[index] = np.clip(
                    np.nan_to_num(standardized, nan=0.0, posinf=100.0, neginf=-100.0),
                    -100.0,
                    100.0,
                ).astype(np.float32)
                mask[index] = 1.0
            return array, mask

        first, first_mask = padded(candidate)
        second, second_mask = padded(target)
        assert self._models
        first_active = first[first_mask > 0]
        second_active = second[second_mask > 0]
        member_predictions = [
            float(model.predict(first_active, second_active)) for model in self._models
        ]
        similarity_fraction = float(
            np.dot(np.asarray(self._ensemble_weights), np.asarray(member_predictions))
        )
        similarity = similarity_fraction * 100.0
        member_disagreement = float(max(member_predictions) - min(member_predictions))
        uncertainty_ood = member_disagreement > self._maximum_member_disagreement
        applied_weight = (
            self._approved_weight
            if applicability >= self.MIN_APPLICABILITY and not uncertainty_ood
            else 0.0
        )
        centered_adjustment = applied_weight * (
            similarity - self._neutral_similarity_percent
        )
        flags = [
            "historical_mixture_similarity_checkpoint_not_text_to_odor_ground_truth",
            "learned_latent_properties_are_not_literal_physical_observables",
            "concentration_and_headspace_are_scored_by_the_independent_deterministic_branch",
            "two_seed_development_selected_ensemble",
            "portable_numpy_inference_without_pickle_or_torch_runtime",
            "split_conformal_interval_is_historical_mixture_label_uncertainty",
        ]
        if self._approved_weight <= 0:
            status = "checkpoint_loaded_weight_zero"
            flags.append("validation_release_gate_did_not_authorize_nonzero_weight")
            if not self._validation_contract_passed:
                flags.append("scientific_release_contract_missing_or_failed")
            if not self._direct_formulation_capability_authorized:
                flags.extend(
                    (
                        "direct_formulation_capability_manifest_missing_or_not_authorized",
                        "r2_presence_only_architecture_cannot_contribute_to_primary_score",
                    )
                )
        elif uncertainty_ood:
            status = "outside_uncertainty_domain"
            flags.append("ensemble_member_disagreement_exceeds_development_p95")
            flags.append("learned_score_not_ensembled_outside_uncertainty_domain")
        elif applied_weight <= 0:
            status = "outside_applicability"
            flags.append("learned_score_not_ensembled_outside_applicability")
        else:
            status = "validation_gated"
            flags.append("learned_r2_checkpoint_ensembled_with_headspace_branch")
        return FrozenR2Result(
            status=status,
            similarity=round(similarity, 4),
            candidate_structure_coverage_percent=round(candidate_coverage, 4),
            target_structure_coverage_percent=round(target_coverage, 4),
            descriptor_domain_coverage_percent=round(domain_coverage, 4),
            applicability_percent=round(applicability, 4),
            approved_primary_score_weight=round(self._approved_weight, 4),
            applied_primary_score_weight=round(applied_weight, 4),
            neutral_similarity_percent=round(self._neutral_similarity_percent, 4),
            centered_score_adjustment=round(centered_adjustment, 4),
            checkpoint_sha256=self._checkpoint_sha256,
            flags=tuple(flags),
            member_predictions_percent=tuple(
                round(value * 100.0, 4) for value in member_predictions
            ),
            member_disagreement_percent=round(member_disagreement * 100.0, 4),
            prediction_interval_lower_percent=round(
                max(0.0, (similarity_fraction - self._absolute_error_q95) * 100.0),
                4,
            ),
            prediction_interval_upper_percent=round(
                min(100.0, (similarity_fraction + self._absolute_error_q95) * 100.0),
                4,
            ),
            ensemble_manifest_sha256=self._ensemble_manifest_sha256,
        )
