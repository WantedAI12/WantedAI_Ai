"""Offline one-coefficient calibration with experiment-disjoint validation.

No dataset or fitted coefficient is bundled, promoted or deployed by this module.
Fitting one parameter conditional on fixed others avoids pretending a single
release curve identifies every partition/transfer coefficient simultaneously.
"""
from datetime import date
import hashlib
import json
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import least_squares

from ..platform.lotion_inputs import LotionInput, LotionSimulationRequest
from ..recommender.lotion import simulate_lotion


class ReleaseObservation(LotionInput):
    experiment_id: str = Field(min_length=1, max_length=200)
    split: Literal["calibration", "validation"]
    minutes: float = Field(gt=0, le=1440)
    observable: Literal["air_concentration_mg_m3", "remaining_mg_cm2", "skin_sink_mg_cm2"]
    value: float = Field(ge=0, le=1e12)
    standard_error: float = Field(gt=0, le=1e12)
    source_reference: str = Field(min_length=1, max_length=1000)
    source_kind: Literal["measured", "simulated"]
    source_date: date

    @model_validator(mode="after")
    def dated(self):
        if self.source_date > date.today():
            raise ValueError("release observation cannot be future-dated")
        return self


class LotionCalibrationRequest(LotionInput):
    simulation: LotionSimulationRequest
    observation_context_id: str = Field(min_length=64, max_length=64)
    ingredient_id: str = Field(min_length=1, max_length=300)
    parameter: Literal["lipid_water_partition", "air_water_partition", "gas_transfer_cm_min", "skin_permeability_cm_min"]
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    observations: list[ReleaseObservation] = Field(min_length=7, max_length=512)
    validation_nrmse_limit: float = Field(default=.05, gt=0, le=.05)

    @model_validator(mode="after")
    def identifiable_scope(self):
        simulation = self.simulation
        if simulation.transport_mode != "bidirectional_air":
            raise ValueError("calibration requires bidirectional air transport")
        if self.observation_context_id != simulation.parameter_context_id:
            raise ValueError("release observations do not match the simulation context")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("parameter bounds must be strictly ordered")
        material = next((m for m in simulation.materials if m.ingredient_id == self.ingredient_id), None)
        if material is None:
            raise ValueError("calibration ingredient is not in the simulation")
        initial = getattr(material, self.parameter)
        if not self.lower_bound < initial < self.upper_bound:
            raise ValueError("initial coefficient must lie strictly inside bounds")
        for bound in (self.lower_bound, self.upper_bound):
            material.__class__.model_validate({**material.model_dump(mode="json"), self.parameter: bound})
        train = [o for o in self.observations if o.split == "calibration"]
        valid = [o for o in self.observations if o.split == "validation"]
        if len(train) < 4 or len(valid) < 3:
            raise ValueError("need at least four calibration and three validation observations")
        if {o.experiment_id for o in train} & {o.experiment_id for o in valid}:
            raise ValueError("calibration and validation experiments must be disjoint")
        keys = [(o.experiment_id, o.minutes, o.observable) for o in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate release observation")
        if any(o.minutes not in simulation.times_minutes for o in self.observations):
            raise ValueError("observation times must be explicit simulation timepoints")
        if len({o.observable for o in self.observations}) != 1:
            raise ValueError("use one observable/unit per calibration run")
        if not any(o.value > 0 for o in train) or not any(o.value > 0 for o in valid):
            raise ValueError("nonzero observations required in each split")
        if len({o.minutes for o in train}) < 4:
            raise ValueError("calibration needs at least four distinct positive timepoints")
        times = simulation.times_minutes
        steps = sum(math.ceil((b-a)/simulation.integration_step_minutes) for a,b in zip(times,times[1:]))
        if (steps + len(times))*len(simulation.materials)*128 > 10_000_000:
            raise ValueError("offline calibration work budget exceeded")
        return self


def fit_lotion_transport(request: LotionCalibrationRequest, catalog):
    train = [o for o in request.observations if o.split == "calibration"]
    valid = [o for o in request.observations if o.split == "validation"]
    simulation = request.simulation
    index = next(i for i,m in enumerate(simulation.materials) if m.ingredient_id == request.ingredient_id)
    observable = train[0].observable
    initial = getattr(simulation.materials[index], request.parameter)
    calls = 0

    def predict(coefficient):
        nonlocal calls
        calls += 1
        if calls > 128:
            raise ValueError("calibration evaluation budget exceeded")
        materials = list(simulation.materials)
        materials[index] = materials[index].model_copy(update={request.parameter: float(coefficient)})
        candidate = simulation.model_copy(update={"materials": materials})
        result = simulate_lotion(candidate, catalog)
        values = {p["minutes"]: p["materials"][index][observable] for p in result["temporal_profile"]}
        return values, candidate

    def residual(log_value):
        values, _ = predict(float(np.exp(log_value[0])))
        # Only calibration values enter the objective; no validation-based tuning.
        return [(values[o.minutes]-o.value)/o.standard_error for o in train]

    baseline, _ = predict(initial)
    fitted = least_squares(residual, [np.log(initial)], bounds=([np.log(request.lower_bound)], [np.log(request.upper_bound)]),
                           max_nfev=60, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    coefficient = float(np.exp(fitted.x[0]))
    predicted, candidate = predict(coefficient)

    def error(values, observations):
        truth = np.array([o.value for o in observations])
        estimate = np.array([values[o.minutes] for o in observations])
        rmse = float(np.sqrt(np.mean((estimate-truth)**2)))
        return {"rmse": rmse, "nrmse_by_observed_rms": rmse / float(np.sqrt(np.mean(truth**2))),
                "observation_count": len(observations)}

    before = error(baseline, valid)
    after = error(predicted, valid)
    sensitivity = float(np.linalg.norm(fitted.jac))
    at_bound = bool(np.any(fitted.active_mask)) or min(
        fitted.x[0]-np.log(request.lower_bound), np.log(request.upper_bound)-fitted.x[0]) < 1e-6
    identifiable = bool(np.isfinite(sensitivity) and sensitivity > 1e-6)
    numerical_pass = bool(fitted.success and identifiable and not at_bound
        and after["nrmse_by_observed_rms"] <= request.validation_nrmse_limit
        and after["nrmse_by_observed_rms"] < before["nrmse_by_observed_rms"])
    all_measured = all(o.source_kind == "measured" for o in request.observations)
    data = request.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # A fitted value is estimated, even when the observations were measured.
    fitted_material = candidate.materials[index].model_copy(update={"source_kind": "estimated",
        "source_reference": "offline-lotion-calibration:" + digest, "source_date": date.today()})
    candidate.materials[index] = fitted_material
    return {"schema_version": "lotion-calibration-1", "calibration_id": digest,
        "status": "candidate_validation_pass" if numerical_pass else "candidate_validation_failed",
        "parameter": request.parameter, "ingredient_id": request.ingredient_id,
        "initial_value": initial, "fitted_value": coefficient,
        "calibration_error": error(predicted, train), "validation_before": before, "validation_after": after,
        "numerical_validation_passed": numerical_pass, "solver_converged": bool(fitted.success),
        "at_parameter_bound": at_bound, "local_sensitivity_norm": sensitivity,
        "conditional_single_parameter_identifiable": identifiable,
        "all_observations_caller_declared_measured": all_measured,
        "observation_sources": [{"experiment_id": o.experiment_id, "split": o.split,
            "source_reference": o.source_reference, "source_kind": o.source_kind, "source_date": str(o.source_date)}
            for o in request.observations],
        "source_verified": False, "automatic_promotion_allowed": False,
        "calibrated_simulation_candidate": candidate.model_dump(mode="json"),
        "transport_simulation_calls": calls,
        "scope": "one_coefficient_conditional_on_fixed_others_and_exact_context_not_human_similarity",
        "limitations": ["Experiment IDs and measured labels are caller assertions, not verified data independence.",
            "Local sensitivity does not establish joint/global parameter identifiability or cross-base generalization.",
            "A 0.05 concentration NRMSE is not 95 percent scent similarity."]}
