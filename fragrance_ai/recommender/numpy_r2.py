"""Dependency-light NumPy inference for the frozen R2 architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


N_STEPS = 16
TIME_HORIZON = 0.1
SOFT_CORE_DELTA = 0.5
LAYER_NORM_EPSILON = 1e-5

EXPECTED_STATE_SHAPES: dict[str, tuple[int, ...]] = {
    "log_attraction": (),
    "log_velocity_limit": (),
    "log_mass_decay": (),
    "log_charge_coupling": (),
    "log_lj_well_depth": (),
    "log_nonlinear_distance": (),
    "log_spin_coupling": (),
    "chemical_encoder.0.weight": (128, 217),
    "chemical_encoder.0.bias": (128,),
    "chemical_encoder.1.weight": (128,),
    "chemical_encoder.1.bias": (128,),
    "chemical_encoder.3.weight": (128, 128),
    "chemical_encoder.3.bias": (128,),
    "chemical_encoder.4.weight": (128,),
    "chemical_encoder.4.bias": (128,),
    "mass_mapper.weight": (1, 128),
    "mass_mapper.bias": (1,),
    "charge_mapper.weight": (1, 128),
    "charge_mapper.bias": (1,),
    "sigma_mapper.weight": (1, 128),
    "sigma_mapper.bias": (1,),
    "position_mapper.weight": (128, 128),
    "position_mapper.bias": (128,),
    "velocity_mapper.weight": (128, 128),
    "velocity_mapper.bias": (128,),
    "spin_mapper.weight": (128, 128),
    "spin_mapper.bias": (128,),
    "fingerprint_projection.0.weight": (128, 134),
    "fingerprint_projection.0.bias": (128,),
    "fingerprint_projection.1.weight": (128,),
    "fingerprint_projection.1.bias": (128,),
    "fingerprint_projection.4.weight": (128, 128),
    "fingerprint_projection.4.bias": (128,),
    "similarity_head.0.weight": (128, 257),
    "similarity_head.0.bias": (128,),
    "similarity_head.3.weight": (1, 128),
    "similarity_head.3.bias": (1,),
}


def _linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return values @ weight.T + bias


def _layer_norm(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + LAYER_NORM_EPSILON) * weight + bias


def _gelu(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1)
    error_function = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in flattened),
        dtype=np.float64,
        count=flattened.size,
    ).reshape(values.shape)
    return 0.5 * values * (1.0 + error_function)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


class NumpyR2Model:
    """Execute the frozen network without importing torch at runtime."""

    def __init__(self, state: Mapping[str, np.ndarray]) -> None:
        supplied = set(state)
        expected = set(EXPECTED_STATE_SHAPES)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"R2 state contract mismatch: missing={missing}, extra={extra}"
            )
        validated: dict[str, np.ndarray] = {}
        for key, shape in EXPECTED_STATE_SHAPES.items():
            value = np.asarray(state[key])
            if value.shape != shape:
                raise ValueError(
                    f"R2 state shape mismatch for {key}: {value.shape} != {shape}"
                )
            if value.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"R2 state dtype is not numeric for {key}")
            numeric = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"R2 state contains non-finite values for {key}")
            validated[key] = numeric
        self.state = validated

    def _chemical_encoder(self, descriptors: np.ndarray) -> np.ndarray:
        state = self.state
        values = _linear(
            descriptors,
            state["chemical_encoder.0.weight"],
            state["chemical_encoder.0.bias"],
        )
        values = _layer_norm(
            values,
            state["chemical_encoder.1.weight"],
            state["chemical_encoder.1.bias"],
        )
        values = _gelu(values)
        values = _linear(
            values,
            state["chemical_encoder.3.weight"],
            state["chemical_encoder.3.bias"],
        )
        values = _layer_norm(
            values,
            state["chemical_encoder.4.weight"],
            state["chemical_encoder.4.bias"],
        )
        return _gelu(values)

    def _map(self, name: str, atoms: np.ndarray) -> np.ndarray:
        return _linear(
            atoms,
            self.state[f"{name}.weight"],
            self.state[f"{name}.bias"],
        )

    def _mixture_fingerprint(self, descriptors: np.ndarray) -> np.ndarray:
        if (
            descriptors.ndim != 2
            or descriptors.shape[1] != 217
            or len(descriptors) == 0
        ):
            raise ValueError("R2 descriptors must have shape [N, 217] with N > 0")
        eps = 1e-8
        atoms = self._chemical_encoder(np.asarray(descriptors, dtype=np.float64))
        masses = _softplus(self._map("mass_mapper", atoms))
        charges = np.tanh(self._map("charge_mapper", atoms))
        sigmas = _softplus(self._map("sigma_mapper", atoms)) + 0.1
        positions = self._map("position_mapper", atoms)
        velocities = self._map("velocity_mapper", atoms)

        attraction = math.exp(float(self.state["log_attraction"]))
        velocity_limit = math.exp(float(self.state["log_velocity_limit"]))
        mass_decay = math.exp(float(self.state["log_mass_decay"]))
        charge_coupling = math.exp(float(self.state["log_charge_coupling"]))
        lj_well_depth = math.exp(float(self.state["log_lj_well_depth"]))
        time_step = TIME_HORIZON / N_STEPS
        trajectory = [positions.copy()]
        mass_history = [masses.copy()]
        off_diagonal = 1.0 - np.eye(len(descriptors), dtype=np.float64)

        for _ in range(N_STEPS):
            difference = positions[:, None, :] - positions[None, :, :]
            distance_squared = np.sum(difference**2, axis=-1, keepdims=True)
            soft_distance = np.sqrt(distance_squared + SOFT_CORE_DELTA**2)
            direction = difference / soft_distance
            mass_pair = masses[:, None, :] * masses[None, :, :]
            force = -attraction * mass_pair / soft_distance**2 * direction
            charge_pair = charges[:, None, :] * charges[None, :, :]
            force += charge_coupling * charge_pair / soft_distance**2 * direction
            sigma_pair = (sigmas[:, None, :] + sigmas[None, :, :]) / 2.0
            sigma_over_r_6 = (sigma_pair / soft_distance) ** 6
            force += (
                24.0
                * lj_well_depth
                / soft_distance
                * (2.0 * sigma_over_r_6**2 - sigma_over_r_6)
                * direction
            )
            acceleration = (force * off_diagonal[..., None]).sum(axis=1) / (
                masses + eps
            )
            speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
            speed_ratio = np.minimum(speed / (velocity_limit + eps), 0.999)
            gamma = 1.0 / np.sqrt(1.0 - speed_ratio**2 + eps)
            acceleration /= gamma + eps
            velocities = velocities + acceleration * time_step
            velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
            positions = positions + velocities * time_step
            positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
            masses = np.maximum(
                masses - mass_decay / (masses**2 + eps) * time_step,
                eps,
            )
            trajectory.append(positions.copy())
            mass_history.append(masses.copy())

        trajectory_array = np.stack(trajectory, axis=0)
        mass_array = np.stack(mass_history, axis=0)
        final_position = trajectory_array[-1]
        position_variance = trajectory_array.var(axis=0, ddof=1).sum(
            axis=-1, keepdims=True
        )
        displacement = trajectory_array[1:] - trajectory_array[:-1]
        speed_history = np.linalg.norm(displacement, axis=-1)
        maximum_speed = speed_history.max(axis=0)[:, None]
        speed_variance = speed_history.var(axis=0, ddof=1)[:, None]
        final_mass = mass_array[-1]
        mass_ratio = final_mass / (mass_array[0] + eps)
        molecule_features = np.concatenate(
            (
                final_position,
                position_variance,
                maximum_speed,
                speed_variance,
                final_mass,
                mass_ratio,
                np.abs(charges),
            ),
            axis=-1,
        )
        return molecule_features.mean(axis=0)

    def _project(self, fingerprint: np.ndarray) -> np.ndarray:
        state = self.state
        values = _linear(
            fingerprint,
            state["fingerprint_projection.0.weight"],
            state["fingerprint_projection.0.bias"],
        )
        values = _layer_norm(
            values,
            state["fingerprint_projection.1.weight"],
            state["fingerprint_projection.1.bias"],
        )
        values = _gelu(values)
        return _linear(
            values,
            state["fingerprint_projection.4.weight"],
            state["fingerprint_projection.4.bias"],
        )

    def predict(self, mixture_a: np.ndarray, mixture_b: np.ndarray) -> float:
        projected_a = self._project(self._mixture_fingerprint(mixture_a))
        projected_b = self._project(self._mixture_fingerprint(mixture_b))
        absolute_difference = np.abs(projected_a - projected_b)
        product = projected_a * projected_b
        denominator = np.linalg.norm(projected_a) * np.linalg.norm(projected_b)
        cosine = float(projected_a @ projected_b / max(denominator, 1e-8))
        features = np.concatenate((absolute_difference, product, [cosine]))
        hidden = _linear(
            features,
            self.state["similarity_head.0.weight"],
            self.state["similarity_head.0.bias"],
        )
        hidden = _gelu(hidden)
        logit = float(
            _linear(
                hidden,
                self.state["similarity_head.3.weight"],
                self.state["similarity_head.3.bias"],
            )[0]
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)
