"""Portable learned autoregressive optimizer; outputs proposals, not approvals."""

import numpy as np

SCHEMA = "shared-formulation-core/v73"
FEATURES = 16
HIDDEN = 48


def project_simplex(values, mask):
    values = np.where(mask, values, -1e9)
    ordered = np.sort(values, axis=-1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=-1) - 1.0
    indices = np.arange(1, values.shape[-1] + 1)
    support = ordered - cumulative / indices > 0
    count = np.maximum(1, support.sum(-1))
    threshold = cumulative[np.arange(len(values)), count - 1] / count
    return np.maximum(values - threshold[:, None], 0.0) * mask


def residual_features(
    profiles, targets, responses, weights, previous, product, step, mask
):
    """Fixed-width physical-error features for arbitrary odor dimensions/counts.

    profiles B,H,N,D; targets B,H,T,D; responses B,T,N; weights B,N.
    The gradient is the exact derivative of mean squared normalized exposure
    profile error, not a descriptor subset or a relabelled public score.
    """
    from .exposure_normalization import normalized_exposure
    predicted, response_derivative = normalized_exposure(profiles,responses,weights)
    residual = predicted - targets
    centered = np.einsum("bhtd,bhnd->bhtn", residual, profiles)
    centered -= np.einsum("bhtd,bhtd->bht", residual, predicted)[..., None]
    gradient = (centered * response_derivative[:, None]).mean(
        axis=(1, 2)
    )
    scale = np.maximum(np.abs(gradient).max(-1, keepdims=True), 1e-12)
    root = np.sqrt((residual**2).sum(-1).mean(axis=(1, 2)) + 1e-12)
    variation = 0.5 * np.abs(residual).sum(-1).mean(axis=(1, 2))
    cosine = np.einsum("bhtd,bhtd->bht", predicted, targets) / np.maximum(
        np.linalg.norm(predicted, axis=-1) * np.linalg.norm(targets, axis=-1), 1e-12
    )
    affinity = np.einsum("bhnd,bhtd->bhtn", profiles, targets) / np.maximum(
        np.linalg.norm(profiles, axis=-1)[:, :, None]
        * np.linalg.norm(targets, axis=-1)[..., None],
        1e-12,
    )
    maximum = np.maximum(responses.max(axis=(1, 2)), 1e-12)
    result = np.zeros((*weights.shape, FEATURES), np.float32)
    result[:, :, 0], result[:, :, 1], result[:, :, 2] = (
        gradient / scale,
        weights,
        np.log(np.maximum(weights, 1e-6)) / 14.0,
    )
    result[:, :, 3] = previous
    result[:, :, 4] = responses.mean(1) / maximum[:, None]
    result[:, :, 5] = np.ptp(responses, axis=1) / maximum[:, None]
    result[:, :, 6], result[:, :, 7], result[:, :, 8] = (
        root[:, None],
        variation[:, None],
        cosine.min(axis=(1, 2))[:, None],
    )
    result[:, :, 9] = min(1.0, step / 8.0)
    result[:, :, 10], result[:, :, 11] = (
        (product == 0)[:, None],
        (product == 1)[:, None],
    )
    result[:, :, 12], result[:, :, 13] = affinity.mean(axis=(1, 2)), np.minimum(scale,1e30)
    result[:, :, 14], result[:, :, 15] = (
        np.linalg.norm(targets, axis=-1).mean(axis=(1, 2))[:, None],
        mask,
    )
    return result, predicted


def recurrent_step(arrays, embedded, features, state, weights, previous, mask):
    from .formulation_core import sigmoid

    prefix = "autoregressive."
    inputs = np.concatenate((embedded, features), axis=-1)
    gi = inputs @ arrays[prefix + "cell.weight_ih"].T + arrays[prefix + "cell.bias_ih"]
    gh = state @ arrays[prefix + "cell.weight_hh"].T + arrays[prefix + "cell.bias_hh"]
    ir, iz, inn = np.split(gi, 3, axis=-1)
    hr, hz, hn = np.split(gh, 3, axis=-1)
    reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
    candidate = np.tanh(inn + reset * hn)
    new_state = np.where(
        mask[..., None], (1 - update) * candidate + update * state, state
    )
    controls = (
        new_state @ arrays[prefix + "controls.weight"].T
        + arrays[prefix + "controls.bias"]
    )
    rate, momentum = 2.0 * sigmoid(controls[..., 0]), 0.85 * sigmoid(controls[..., 1])
    proposed = project_simplex(
        weights - rate * features[:, :, 0] * features[:, :, 6] + momentum * previous,
        mask,
    )
    return proposed.astype(np.float32), new_state.astype(np.float32)


def refine_arrays(
    arrays,
    latents,
    profiles,
    targets,
    responses,
    weights,
    product,
    *,
    steps=8,
    mask=None,
):
    weights = np.asarray(weights, np.float32)
    mask = np.ones_like(weights, dtype=bool) if mask is None else np.asarray(mask, bool)
    if not isinstance(steps, int) or not 1 <= steps <= 16 or np.any(mask.sum(-1) == 0):
        raise ValueError(
            "bounded autoregressive rollout and nonempty candidate mask required"
        )
    state = np.zeros((*weights.shape, HIDDEN), np.float32)
    previous = np.zeros_like(weights)
    embedded = np.tanh(
        latents @ arrays["autoregressive.chemistry.weight"].T
        + arrays["autoregressive.chemistry.bias"]
    )
    initial = weights.copy()
    for step in range(steps):
        features, _ = residual_features(
            profiles, targets, responses, weights, previous, product, step, mask
        )
        proposed, state = recurrent_step(
            arrays, embedded, features, state, weights, previous, mask
        )
        previous, weights = proposed - weights, proposed
    return weights, {
        "steps": steps,
        "autoregressive_state_reused": True,
        "mass_l1_change": np.abs(weights - initial).sum(-1).tolist(),
        "scope": "learned_physical_error_correction_proposal_not_approved_formula",
    }
