"""Train-only, missingness-aware molecular geometry and convex kernel learning.

No chemical equilibrium or measured concentration is inferred from a molecular
descriptor or an ordinal label. The kernels encode statistical similarity, not
receptor affinity. Every sum/product below preserves positive semidefiniteness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

SCHEMA = "molecular-geometry-alignment/v68"
BASE_NAMES = (
    "chiral_morgan_tanimoto",
    "physical_whitened_rbf_0.25",
    "physical_whitened_rbf_1",
    "physical_whitened_rbf_4",
    "observed_native_hellinger",
    "observed_description_tanimoto",
)
NAMES = tuple(
    f"{name}/{condition}"
    for condition in ("shared", "same_ordinal")
    for name in BASE_NAMES
)


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}, with no QP dependency."""
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("finite nonempty simplex vector required")
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    positive = ordered - cumulative / np.arange(1, len(value) + 1) > 0
    index = np.flatnonzero(positive)[-1]
    return np.maximum(value - cumulative[index] / (index + 1), 0.0)


def fit_alignment(
    kernels: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float,
    prior: np.ndarray | None = None,
) -> dict:
    """Solve a strictly convex, centered Gram-matching QP on the simplex.

    min .5 ||sum_j w_j Kcj/||Kcj||F - YYc/||YYc||F||F^2
        + .5*regularization*||w-prior||2^2.
    A projected KKT residual, not an iteration count alone, authorizes the fit.
    """
    kernels, targets = np.asarray(kernels, float), np.asarray(targets, float)
    if (
        kernels.ndim != 3
        or kernels.shape[1] != kernels.shape[2]
        or kernels.shape[1] < 2
        or targets.ndim != 2
        or len(targets) != kernels.shape[1]
        or not targets.shape[1]
        or not np.isfinite(kernels).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError(
            "finite aligned training kernels and positive regularization required"
        )
    count = len(kernels)
    prior = np.full(count, 1 / count) if prior is None else np.asarray(prior, float)
    if (
        prior.shape != (count,)
        or not np.isfinite(prior).all()
        or np.any(prior < 0)
        or abs(prior.sum() - 1) > 1e-12
    ):
        raise ValueError("kernel prior must be a simplex vector")
    mean = kernels.mean(axis=1, keepdims=True)
    centered = (
        kernels - mean - mean.transpose(0, 2, 1) + mean.mean(axis=2, keepdims=True)
    )
    norms = np.linalg.norm(centered, axis=(1, 2))
    norms = np.where(norms > 1e-12, norms, 1.0)
    features = (centered / norms[:, None, None]).reshape(count, -1)
    response = targets - targets.mean(axis=0)
    target_kernel = response @ response.T
    target_norm = np.linalg.norm(target_kernel)
    target_kernel /= max(target_norm, 1e-12)
    hessian = features @ features.T + regularization * np.eye(count)
    rhs = features @ target_kernel.ravel() + regularization * prior
    step = 1 / float(np.linalg.eigvalsh(hessian)[-1])
    weights = prior.copy()
    residual = float("inf")
    for iteration in range(20000):
        gradient = hessian @ weights - rhs
        candidate = project_simplex(weights - step * gradient)
        weights = candidate
        residual = float(
            np.max(
                np.abs(weights - project_simplex(weights - (hessian @ weights - rhs)))
            )
        )
        if residual < 2e-10:
            break
    if residual >= 2e-10:
        raise ValueError("kernel alignment did not satisfy its KKT tolerance")
    # A small explicit structural prior preserves an input-space definition for
    # unannotated molecules. It does not add target values or raise recipe scores.
    weights = 0.99 * weights + 0.01 * prior
    return {
        "weights": weights.tolist(),
        "centered_norms": norms.tolist(),
        "diagnostics": {
            "solver": "simplex_projected_gradient",
            "projected_kkt_residual": residual,
            "iterations": iteration + 1,
            "regularization": float(regularization),
            "structural_prior_fraction": 0.01,
            "objective_before_prior_mix": float(
                0.5 * candidate @ hessian @ candidate - rhs @ candidate
            ),
            "training_rows": len(targets),
            "training_only": True,
        },
    }


def _features(value: np.ndarray) -> np.ndarray:
    from .atlas_profiles import _valid_features

    value = np.asarray(value, float)
    if (
        not _valid_features(value)
        or np.any(value[:, 1040:1059] < 0)
        or np.any((value[:, 1059] == 0) & np.any(value[:, 1040:1059] != 0, axis=1))
        or np.any((value[:, -3] == 0) & np.any(value[:, 1060:-3] != 0, axis=1))
    ):
        raise ValueError(
            "molecular geometry requires consistent observed-feature masks"
        )
    return value


def fit_geometry(features: np.ndarray, *, shrinkage: float = 0.1) -> dict:
    """Nondimensionalize and decorrelate molecular descriptors on train rows only.

    The positive shrinkage is numerical/statistical regularization, not a
    thermodynamic parameter. Correlated size descriptors no longer count as
    independent physical evidence. Missing native odor annotations stay missing.
    """
    x = _features(features)
    if len(x) < 2 or not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("two training rows and shrinkage in (0,1] required")
    physical = x[:, 1024:1040]
    mean, scale = physical.mean(axis=0), physical.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (physical - mean) / scale
    correlation = z.T @ z / len(z)
    metric = (1 - shrinkage) * correlation + shrinkage * np.eye(16)
    values, vectors = np.linalg.eigh(metric)
    whitener = (vectors / np.sqrt(values)) @ vectors.T
    radius = np.sqrt(np.mean((z @ whitener) ** 2, axis=1))
    return {
        "schema": SCHEMA,
        "names": list(NAMES),
        "feature_width": x.shape[1],
        "physical_mean": mean.tolist(),
        "physical_scale": scale.tolist(),
        "physical_whitener": whitener.tolist(),
        "covariance_shrinkage": float(shrinkage),
        "training_radius_max": float(radius.max()),
        "training_rows": len(x),
    }


def _readonly(value) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MolecularGeometry:
    """Validated immutable transform, compiled once for bounded-memory inference."""

    mean: np.ndarray
    scale: np.ndarray
    whitener: np.ndarray
    width: int
    radius_max: float

    @classmethod
    def from_dict(cls, value: dict) -> MolecularGeometry:
        mean = _readonly(value["physical_mean"])
        scale = _readonly(value["physical_scale"])
        whitener = _readonly(value["physical_whitener"])
        width, radius = value["feature_width"], value["training_radius_max"]
        if (
            value.get("schema") != SCHEMA
            or tuple(value.get("names", ())) != NAMES
            or type(width) is not int
            or width <= 1063
            or mean.shape != (16,)
            or scale.shape != (16,)
            or whitener.shape != (16, 16)
            or not all(np.isfinite(v).all() for v in (mean, scale, whitener))
            or np.any(scale <= 0)
            or not np.isfinite(radius)
            or radius < 0
            or not np.allclose(whitener, whitener.T, atol=1e-12, rtol=1e-12)
            or np.linalg.eigvalsh(whitener)[0] <= 0
        ):
            raise ValueError("invalid molecular geometry checkpoint")
        return cls(mean, scale, whitener, width, float(radius))

    def physical(self, x: np.ndarray) -> np.ndarray:
        return ((x[:, 1024:1040] - self.mean) / self.scale) @ self.whitener

    def blocks(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _features(a), _features(b)
        if a.shape[1] != self.width or b.shape[1] != self.width:
            raise ValueError("molecular geometry width mismatch")

        def tanimoto(left, right):
            dot = left @ right.T
            union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - dot
            return np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)

        def native(value):
            profiles = value[:, 1040:1059]
            totals = profiles.sum(axis=1, keepdims=True)
            return np.sqrt(
                np.divide(
                    profiles, totals, out=np.zeros_like(profiles), where=totals > 0
                )
            )

        # Direct squared differences avoid catastrophic dot-product cancellation
        # for two nearby molecules far from the training descriptor center.
        distance = cdist(self.physical(a), self.physical(b), metric="sqeuclidean") / 16
        base = [
            tanimoto(a[:, :1024], b[:, :1024]),
            *(np.exp(-bandwidth * distance) for bandwidth in (0.25, 1.0, 4.0)),
            native(a) @ native(b).T,
            tanimoto(a[:, 1060:-3], b[:, 1060:-3]),
        ]
        conditional = a[:, -2:] @ b[:, -2:].T
        return np.stack([*base, *(kernel * conditional for kernel in base)])

    def diagonal(self, x: np.ndarray) -> np.ndarray:
        base = np.stack(
            [
                (x[:, :1024].sum(axis=1) > 0).astype(float),
                np.ones(len(x)),
                np.ones(len(x)),
                np.ones(len(x)),
                (x[:, 1040:1059].sum(axis=1) > 0).astype(float),
                (x[:, 1060:-3].sum(axis=1) > 0).astype(float),
            ]
        )
        return np.concatenate((base, base), axis=0)


def kernel_prior() -> np.ndarray:
    families = np.array([0.25, 1 / 12, 1 / 12, 1 / 12, 0.25, 0.25])
    return np.r_[0.75 * families, 0.25 * families]


class ScientificKernel:
    """An explicitly normalized convex PSD kernel; not a fitted regressor."""

    def __init__(self, specification: dict):
        self.geometry = MolecularGeometry.from_dict(specification["geometry"])
        self.weights = _readonly(specification["alignment"]["weights"])
        self.norms = _readonly(specification["alignment"]["centered_norms"])
        if (
            self.weights.shape != (len(NAMES),)
            or self.norms.shape != self.weights.shape
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.norms).all()
            or np.any(self.weights < 0)
            or abs(self.weights.sum() - 1) > 1e-10
            or np.any(self.norms <= 0)
            or np.sum(self.weights[[1, 2, 3, 7, 8, 9]]) <= 0
        ):
            raise ValueError("invalid learned molecular kernel coefficients")
        self.coefficients = _readonly(self.weights / self.norms)

    def combine(self, blocks: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        left = self.coefficients @ self.geometry.diagonal(a)
        right = self.coefficients @ self.geometry.diagonal(b)
        kernel = np.einsum("j,jab->ab", self.coefficients, blocks)
        kernel /= np.sqrt(left[:, None] * right[None, :])
        if not np.isfinite(kernel).all():
            raise ValueError("nonfinite molecular kernel")
        return kernel

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.combine(self.geometry.blocks(a, b), a, b)

    def diagnostics(self, query: np.ndarray, cross_kernel: np.ndarray) -> dict:
        radius = np.sqrt(np.mean(self.geometry.physical(query) ** 2, axis=1))
        return {
            "maximum_training_similarity": cross_kernel.max(axis=1),
            "physical_radius": radius,
            "outside_training_radius": radius > self.geometry.radius_max + 1e-12,
            "native_annotation_present": query[:, 1059].astype(bool),
            "description_annotation_present": query[:, -3].astype(bool),
        }
