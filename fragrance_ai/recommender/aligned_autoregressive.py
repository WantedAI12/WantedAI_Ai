"""V75 product-aligned recurrent optimizer with accepted-state line search.

This optimizes supplied transport profiles, not a substitute for the final
nonlinear perfume verifier or new human sensory evidence.
"""

import numpy as np

from .constrained_autoregressive import project_constraints, cheapest_feasible
from .exposure_normalization import normalized_exposure

SCHEMA = "shared-formulation-core/v75"


def objective(
    profiles, targets, responses, weights, product, *, time_weights=None, avoided=None, physics=None, gradient_geometry=None
):
    if physics is not None:
        return physics(profiles,targets,responses,weights,product,time_weights=time_weights,avoided=avoided)
    predicted, derivative = normalized_exposure(profiles, responses, weights)
    q = np.asarray(targets, float)
    residual = predicted - q
    norm = np.linalg.norm(predicted, axis=-1)
    goal_norm = np.linalg.norm(q, axis=-1)
    cosine = (predicted * q).sum(-1) / (norm * goal_norm)
    avoid = np.zeros_like(q) if avoided is None else np.broadcast_to(avoided, q.shape)
    losses = np.stack(
        (0.5 * np.abs(residual).sum(-1), 1 - cosine, (predicted * avoid).sum(-1)), -1
    )
    kind = losses.argmax(-1)
    point = np.take_along_axis(losses, kind[..., None], -1)[..., 0]
    partial = np.where(
        (kind == 0)[..., None],
        0.5 * np.sign(residual),
        np.where(
            (kind == 1)[..., None],
            cosine[..., None] * predicted / norm[..., None] ** 2
            - q / (norm * goal_norm)[..., None],
            avoid,
        ),
    )
    batch, heads, times = point.shape
    coeff = np.zeros_like(point)
    value = np.zeros(batch)
    temporal = (
        np.zeros((batch, times))
        if time_weights is None
        else np.array(time_weights, float, copy=True)
    )
    if time_weights is None and times > 1:
        temporal[:, 1:] = 1 / (times - 1)
    if (
        temporal.shape != (batch, times)
        or np.any(temporal < 0)
        or not np.isfinite(temporal).all()
    ):
        raise ValueError("nonnegative product time weights required")
    if times > 1:
        temporal[:, 0] = 0
        sums = temporal.sum(-1)
        if np.any((product == 0) & (sums <= 0)):
            raise ValueError("perfume needs positive temporal weights")
        temporal /= np.where(sums > 0, sums, 1)[:, None]
    for b in range(batch):
        if product[b] == 0 and times > 1:
            temporal_loss = point[b] @ temporal[b]
            head_loss = np.maximum(point[b, :, 0], temporal_loss)
            h = int(head_loss.argmax())
            value[b] = head_loss[h]
            if point[b, h, 0] >= temporal_loss[h]:
                coeff[b, h, 0] = 1
            else:
                coeff[b, h] = temporal[b]
        else:
            h, t = np.unravel_index(point[b].argmax(), point[b].shape)
            value[b] = point[b, h, t]
            coeff[b, h, t] = 1
    partial *= coeff[..., None]
    gradient = (
        np.einsum("bhtd,bhnd->bhtn", partial, profiles)
        - np.einsum("bhtd,bhtd->bht", partial, predicted)[..., None]
    ) * derivative[:, None]
    return value, gradient.sum(axis=(1, 2)), predicted


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
    gradient_geometry=None,
    **objective_kwargs,
):
    from .constrained_autoregressive import features as prior_features

    f, _, _ = prior_features(
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
    loss, gradient, predicted = objective(
        profiles, targets, responses, weights, product, **objective_kwargs
    )
    gradient *= mask
    if gradient_geometry is not None:
        from .simplex_geometry import tangent_gradient
        gradient=tangent_gradient(gradient,weights,mask,product,gradient_geometry)
    scale = np.maximum(np.abs(gradient).max(-1, keepdims=True), 1e-30)
    f[:, :, 16] = gradient / scale
    f[:, :, 17] = loss[:, None]
    # Large physical derivatives are not directly fed into sigmoid gates.
    f[:, :, 13] = np.log1p(np.maximum(f[:, :, 13], 0)) / 16.0
    return f, loss, predicted


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
    time_weights=None,
    avoided=None,
    physics=None,
    gradient_geometry=None,
):
    from .formulation_core import sigmoid

    weights, lower, upper, prices = [
        np.asarray(v, float) for v in (weights, lower, upper, prices)
    ]
    budget = np.asarray(budget, float)
    if (
        not isinstance(steps, int)
        or not 1 <= steps <= 16
        or budget.shape != (len(weights),)
        or np.any(budget <= 0)
        or not np.isfinite(budget).all()
        or prices.shape != weights.shape
        or np.any(prices < 0)
        or not np.isfinite(prices).all()
    ):
        raise ValueError("finite constrained recurrent configuration required")
    cheapest = cheapest_feasible(lower, upper, prices)
    weights = project_constraints(
        weights, lower, upper, prices, budget, cheapest=cheapest
    )
    first = weights.copy()
    mask = upper > lower
    hidden = arrays['autoregressive.cell.weight_hh'].shape[1]
    if hidden not in (48,96,128):
        raise ValueError('unsupported recurrent width')
    state = np.zeros((*weights.shape, hidden), np.float32)
    previous = np.zeros_like(weights)
    embedded = np.tanh(
        latents @ arrays["autoregressive.chemistry.weight"].T
        + arrays["autoregressive.chemistry.bias"]
    )
    options = {"time_weights": time_weights, "avoided": avoided, "physics": physics, "gradient_geometry":gradient_geometry}
    accepted = np.zeros(len(weights), int)
    shrunk = np.zeros(len(weights), int)
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
            **options,
        )
        gi = (
            np.concatenate((embedded, f), -1)
            @ arrays["autoregressive.cell.weight_ih"].T
            + arrays["autoregressive.cell.bias_ih"]
        )
        gh = (
            state @ arrays["autoregressive.cell.weight_hh"].T
            + arrays["autoregressive.cell.bias_hh"]
        )
        ir, iz, inn = np.split(gi, 3, -1)
        hr, hz, hn = np.split(gh, 3, -1)
        update, reset = sigmoid(iz + hz), sigmoid(ir + hr)
        state = np.where(
            mask[..., None],
            (1 - update) * np.tanh(inn + reset * hn) + update * state,
            state,
        ).astype(np.float32)
        control = (
            state @ arrays["autoregressive.controls.weight"].T
            + arrays["autoregressive.controls.bias"]
        )
        rate, momentum = 2 * sigmoid(control[..., 0]), 0.85 * sigmoid(control[..., 1])
        radius = 0.005 + 0.495 * sigmoid(
            (control[..., 2] * mask).sum(-1) / np.maximum(mask.sum(-1), 1)
        )
        proposed = project_constraints(
            weights - rate * f[:, :, 16] * f[:, :, 17] + momentum * previous,
            lower,
            upper,
            prices,
            budget,
            cheapest=cheapest,
        )
        delta = proposed - weights
        delta *= np.minimum(1, radius / np.maximum(np.abs(delta).sum(-1), 1e-30))[
            :, None
        ]
        best, best_loss, best_fraction = (
            weights.copy(),
            before.copy(),
            np.zeros(len(weights)),
        )
        # Evaluate all widths against the SAME accepted anchor. The decision
        # is reproduced during training, including retained-state loss.
        for fraction in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.001953125,
        ):
            candidate = weights + fraction * delta
            if physics is None:
                loss, _, _ = objective(profiles, targets, responses, candidate, product, **options)
            else:
                loss,_,_=physics(profiles,targets,responses,candidate,product,time_weights=time_weights,
                                  avoided=avoided,compute_gradient=False)
            good = loss < best_loss - 1e-10
            best = np.where(good[:, None], candidate, best)
            best_loss = np.where(good, loss, best_loss)
            best_fraction = np.where(good, fraction, best_fraction)
        accepted += best_fraction > 0
        shrunk += (best_fraction > 0) & (best_fraction < 1)
        previous, weights = best - weights, best
    return weights, {
        "steps": steps,
        "autoregressive_state_reused": True,
        "feasible_proxy_steps": accepted.tolist(),
        "backtracked_steps": shrunk.tolist(),
        "caps_and_price_in_each_neural_step": True,
        "objective": "perfume_nominal_and_time_weighted_or_lotion_worst_case_including_avoidance",
        "mass_l1_change": np.abs(weights - first).sum(-1).tolist(),
        "scope": "product_aligned_transport_proposal_final_nonlinear_and_evidence_verification_required",
        "nonlinear_shared_physics_in_rollout": physics is not None,
        "gradient_geometry":gradient_geometry or 'legacy_ambient_coordinates',
    }
