"""Constraint-aware, cardinality-stable learned refinement (CPU reference)."""

import numpy as np

SCHEMA = "shared-formulation-core/v74"
FEATURES = 24


def capped_simplex(values, lower, upper):
    original_dtype=np.asarray(values).dtype
    epsilon=max(np.finfo(original_dtype).eps if original_dtype.kind=='f' else 0.,np.finfo(np.float32).eps)
    values, lower, upper = map(
        lambda x: np.asarray(x, np.float64), (values, lower, upper)
    )
    if (
        values.ndim != 2
        or lower.shape != values.shape
        or upper.shape != values.shape
        or any(not np.isfinite(x).all() for x in (values, lower, upper))
        or np.any(lower < 0)
        or np.any(lower > upper)
        or np.any(lower.sum(-1) > 1 + 1e-9)
        or np.any(upper.sum(-1) < 1 - 1e-9)
    ):
        raise ValueError("feasible finite mass bounds required")
    left = (values - upper).min(-1, keepdims=True)
    right = (values - lower).max(-1, keepdims=True)
    for _ in range(48):
        middle = (left + right) * 0.5
        mass = np.clip(values - middle, lower, upper).sum(-1, keepdims=True)
        left = np.where(mass > 1, middle, left)
        right = np.where(mass > 1, right, middle)
    from .mass_projection import preserve_feasible_support
    projected=np.clip(values - (left + right) * 0.5, lower, upper)
    return preserve_feasible_support(values,projected,lower,upper,epsilon)


def cheapest_feasible(lower, upper, prices):
    order = np.argsort(prices, axis=-1, kind="stable")
    capacity = np.take_along_axis(upper - lower, order, -1)
    available = (1 - lower.sum(-1))[:, None]
    preceding = np.cumsum(capacity, -1) - capacity
    additions = np.minimum(capacity, np.maximum(available - preceding, 0.0))
    result = lower.copy()
    np.put_along_axis(
        result, order, np.take_along_axis(lower, order, -1) + additions, -1
    )
    return result


def project_constraints(values, lower, upper, prices, budget, *, cheapest=None):
    projected = capped_simplex(values, lower, upper)
    cheapest = (
        cheapest_feasible(np.asarray(lower, float), np.asarray(upper, float), prices)
        if cheapest is None
        else cheapest
    )
    minimum = (cheapest * prices).sum(-1)
    if np.any(minimum > budget + 1e-7):
        raise ValueError("price budget has no feasible mass composition")
    cost = (projected * prices).sum(-1)
    fraction = np.where(
        cost > budget + 1e-7,
        np.clip((cost - budget) / np.maximum(cost - minimum, 1e-12), 0, 1),
        0.0,
    )
    return (1 - fraction[:, None]) * projected + fraction[:, None] * cheapest


def features(
    profiles,
    targets,
    responses,
    weights,
    previous,
    product,
    step,
    mask,
    lower,
    upper,
    prices,
    budget,
):
    from .autoregressive_neural import residual_features

    legacy, predicted = residual_features(
        profiles, targets, responses, weights, previous, product, step, mask
    )
    residual = predicted - targets
    norm = np.maximum(np.linalg.norm(predicted, axis=-1), 1e-12)
    goal_norm = np.maximum(np.linalg.norm(targets, axis=-1), 1e-12)
    cosine = np.einsum("bhtd,bhtd->bht", predicted, targets) / (norm * goal_norm)
    tv = 0.5 * np.abs(residual).sum(-1)
    losses = np.stack((tv, 1 - cosine), -1)
    logits = 64.0 * losses
    logits -= logits.max(axis=(1, 2, 3), keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=(1, 2, 3), keepdims=True)
    partial = attention[..., 0, None] * 0.5 * np.sign(residual) + attention[
        ..., 1, None
    ] * (
        cosine[..., None] * predicted / norm[..., None] ** 2
        - targets / (norm * goal_norm)[..., None]
    )
    from .exposure_normalization import normalized_exposure
    _, response_derivative = normalized_exposure(profiles,responses,weights)
    gradient = (
        (
            np.einsum("bhtd,bhnd->bhtn", partial, profiles)
            - np.einsum("bhtd,bhtd->bht", partial, predicted)[..., None]
        )
        * response_derivative[:, None]
    )
    gradient = gradient.sum(axis=(1, 2)) * mask
    scale = np.maximum(np.abs(gradient).max(-1, keepdims=True), 1e-12)
    result = np.zeros((*weights.shape, FEATURES), np.float32)
    result[:, :, :16] = legacy
    result[:, :, 16] = gradient / scale
    result[:, :, 17] = losses.max(axis=(1, 2, 3))[:, None]
    result[:, :, 18], result[:, :, 19] = upper, lower
    result[:, :, 20] = prices / budget[:, None]
    result[:, :, 21] = np.log1p(mask.sum(-1))[:, None] / np.log(4097.0)
    result[:, :, 22] = (upper - weights) / np.maximum(upper - lower, 1e-8)
    result[:, :, 23] = ((weights * prices).sum(-1) / budget)[:, None]
    return result, losses.max(axis=(1, 2, 3)), predicted


def rollout(
    arrays,
    latents,
    profiles,
    targets,
    responses,
    weights,
    product,
    lower,
    upper,
    prices,
    budget,
    *,
    steps=8,
):
    from .formulation_core import sigmoid

    weights = np.asarray(weights, np.float64)
    lower, upper, prices = map(
        lambda x: np.asarray(x, np.float64), (lower, upper, prices)
    )
    budget = np.asarray(budget, np.float64)
    if not isinstance(steps, int) or not 1 <= steps <= 16 or np.any(budget <= 0):
        raise ValueError("bounded rollout and positive price budget required")
    mask = upper > lower
    cheapest = cheapest_feasible(lower, upper, prices)
    weights = project_constraints(
        weights, lower, upper, prices, budget, cheapest=cheapest
    )
    first = weights.copy()
    hidden = arrays['autoregressive.cell.weight_hh'].shape[1]
    if hidden not in (48,96,128):
        raise ValueError('unsupported recurrent width')
    state = np.zeros((*weights.shape, hidden), np.float32)
    previous = np.zeros_like(weights)
    embedded = np.tanh(
        latents @ arrays["autoregressive.chemistry.weight"].T
        + arrays["autoregressive.chemistry.bias"]
    )
    accepted = np.zeros(len(weights), int)
    prefix = "autoregressive."
    for step in range(steps):
        f, before, _ = features(
            profiles,
            targets,
            responses,
            weights,
            previous,
            product,
            step,
            mask,
            lower,
            upper,
            prices,
            budget,
        )
        inputs = np.concatenate((embedded, f), -1)
        gi = (
            inputs @ arrays[prefix + "cell.weight_ih"].T
            + arrays[prefix + "cell.bias_ih"]
        )
        gh = (
            state @ arrays[prefix + "cell.weight_hh"].T
            + arrays[prefix + "cell.bias_hh"]
        )
        ir, iz, inn = np.split(gi, 3, -1)
        hr, hz, hn = np.split(gh, 3, -1)
        reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
        state = np.where(
            mask[..., None],
            (1 - update) * np.tanh(inn + reset * hn) + update * state,
            state,
        ).astype(np.float32)
        control = (
            state @ arrays[prefix + "controls.weight"].T
            + arrays[prefix + "controls.bias"]
        )
        rate, momentum = 2 * sigmoid(control[..., 0]), 0.85 * sigmoid(control[..., 1])
        radius = 0.025 + 0.475 * sigmoid(
            (control[..., 2] * mask).sum(-1) / np.maximum(mask.sum(-1), 1)
        )
        raw = weights - rate * f[:, :, 16] * f[:, :, 17] + momentum * previous
        proposed = project_constraints(
            raw, lower, upper, prices, budget, cheapest=cheapest
        )
        distance = np.abs(proposed - weights).sum(-1)
        fraction = np.minimum(1, radius / np.maximum(distance, 1e-12))
        proposed = weights + fraction[:, None] * (proposed - weights)
        _, after, _ = features(
            profiles,
            targets,
            responses,
            proposed,
            proposed - weights,
            product,
            step,
            mask,
            lower,
            upper,
            prices,
            budget,
        )
        # Same safeguarded accept/reject operation is unrolled in training.
        good = after <= before + 1e-8
        chosen = np.where(good[:, None], proposed, weights)
        accepted += good & (np.abs(chosen - weights).sum(-1) > 1e-9)
        previous, weights = chosen - weights, chosen
    return weights, {
        "steps": steps,
        "autoregressive_state_reused": True,
        "feasible_proxy_steps": accepted.tolist(),
        "caps_and_price_in_each_neural_step": True,
        "complete_profile_worst_case_objective": True,
        "mass_l1_change": np.abs(weights - first).sum(-1).tolist(),
        "scope": "constraint_aware_neural_proposal_original_final_verifier_still_required",
    }
