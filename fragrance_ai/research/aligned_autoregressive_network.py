"""Differentiable accepted-state V75 training, matched to NumPy rollout."""

import torch

from .constrained_autoregressive_network import (
    ConstrainedAutoregressiveCell,
    project_constraints,
    cheapest_feasible,
)
from fragrance_ai.recommender.exposure_normalization import torch_normalized_exposure


def objective(
    profiles, targets, responses, weights, product, *, time_weights=None, avoided=None, physics=None, gradient_geometry=None
):
    if physics is not None:
        return physics(profiles,targets,responses,weights,product,time_weights=time_weights,avoided=avoided)
    profiles, targets, responses, weights = [
        v.double() for v in (profiles, targets, responses, weights)
    ]
    predicted, derivative = torch_normalized_exposure(profiles, responses, weights)
    residual = predicted - targets
    norm = predicted.norm(dim=-1)
    goal_norm = targets.norm(dim=-1)
    cosine = (predicted * targets).sum(-1) / (norm * goal_norm)
    avoid = torch.zeros_like(targets) if avoided is None else avoided.expand_as(targets)
    losses = torch.stack(
        (0.5 * residual.abs().sum(-1), 1 - cosine, (predicted * avoid).sum(-1)), -1
    )
    point, kind = losses.max(-1)
    partial = torch.where(
        (kind == 0)[..., None],
        0.5 * residual.sign(),
        torch.where(
            (kind == 1)[..., None],
            cosine[..., None] * predicted / norm[..., None] ** 2
            - targets / (norm * goal_norm)[..., None],
            avoid,
        ),
    )
    batch, heads, times = point.shape
    if time_weights is None:
        temporal = torch.zeros(
            (batch, times), device=weights.device, dtype=weights.dtype
        )
        if times > 1:
            temporal[:, 1:] = 1 / (times - 1)
    else:
        temporal = time_weights.double().clone()
        temporal[:, 0] = 0
        temporal = temporal / temporal.sum(-1, keepdim=True).clamp_min(1e-30)
    flat = point.flatten(1)
    worst, index = flat.max(-1)
    coefficients = (
        torch.nn.functional.one_hot(index, heads * times)
        .reshape(batch, heads, times)
        .to(weights.dtype)
    )
    values = worst
    if times > 1:
        temporal_loss = (point * temporal[:, None]).sum(-1)
        head_loss = torch.maximum(point[:, :, 0], temporal_loss)
        perfume_value, head = head_loss.max(-1)
        time_zero = torch.zeros(times, device=weights.device, dtype=weights.dtype)
        time_zero[0] = 1
        coeff = torch.nn.functional.one_hot(head, heads).to(weights.dtype)[
            ..., None
        ] * torch.where(
            (point[:, :, 0] >= temporal_loss)[..., None],
            time_zero[None, None],
            temporal[:, None],
        )
        coefficients = torch.where((product == 0)[:, None, None], coeff, coefficients)
        values = torch.where(product == 0, perfume_value, values)
    partial = partial * coefficients[..., None]
    gradient = (
        torch.einsum("bhtd,bhnd->bhtn", partial, profiles)
        - torch.einsum("bhtd,bhtd->bht", partial, predicted)[..., None]
    ) * derivative[:, None]
    return values, gradient.sum(dim=(1, 2)), predicted


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
    **options,
):
    from .constrained_autoregressive_network import features as prior_features

    old, _, _ = prior_features(
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
        profiles, targets, responses, weights, product, **options
    )
    gradient = gradient * mask
    if gradient_geometry is not None:
        from fragrance_ai.recommender.simplex_geometry import torch_tangent_gradient
        gradient=torch_tangent_gradient(gradient,weights,mask,product,gradient_geometry)
    scale = gradient.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    result = torch.cat(
        (
            old[:, :, :13],
            old[:, :, 13:14].clamp_min(0).log1p() / 16.0,
            old[:, :, 14:16],
            (gradient / scale)[..., None],
            loss[:, None, None].expand(-1, weights.shape[1], 1),
            old[:, :, 18:],
        ),
        -1,
    )
    return result.to(weights.dtype), loss, predicted


class AlignedAutoregressiveCell(ConstrainedAutoregressiveCell):
    def forward(
        self,
        latents,
        profiles,
        targets,
        responses,
        initial,
        product,
        lower,
        upper,
        prices,
        budget,
        steps=8,
        time_weights=None,
        avoided=None,
        physics=None,
        gradient_geometry=None,
    ):
        mask = upper > lower
        cheapest = cheapest_feasible(lower, upper, prices)
        weights = project_constraints(initial, lower, upper, prices, budget, cheapest)
        embedded = torch.tanh(self.chemistry(latents))
        previous = torch.zeros_like(weights)
        state = torch.zeros((*weights.shape, self.cell.hidden_size), device=weights.device)
        traces = []
        options = {"time_weights": time_weights, "avoided": avoided, "physics": physics, "gradient_geometry":gradient_geometry}
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
            if physics is not None:
                # First-order learned-optimizer training: environment gradient
                # is an observed feedback feature; GRU state still uses BPTT.
                f=f.detach()
            updated = self.cell(
                torch.cat((embedded, f), -1).reshape(-1, 88), state.reshape(-1, self.cell.hidden_size)
            ).reshape_as(state)
            state = torch.where(mask[..., None], updated, state)
            control = self.controls(state)
            rate, momentum = (
                2 * control[..., 0].sigmoid(),
                0.85 * control[..., 1].sigmoid(),
            )
            radius = (
                0.005
                + 0.495
                * (
                    (control[..., 2] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                ).sigmoid()
            )
            proposed = project_constraints(
                weights - rate * f[:, :, 16] * f[:, :, 17] + momentum * previous,
                lower,
                upper,
                prices,
                budget,
                cheapest,
            )
            delta = proposed - weights
            delta = (
                delta
                * (radius / delta.abs().sum(-1).clamp_min(1e-30)).clamp_max(1)[:, None]
            )
            best, best_loss = weights, before
            best_fraction=torch.zeros(len(weights),device=weights.device)
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
                    with torch.no_grad():
                        loss,_,_=physics(profiles,targets,responses,candidate,product,time_weights=time_weights,
                                          avoided=avoided,compute_gradient=False)
                good = (loss < best_loss - 1e-10).detach()
                best = torch.where(good[:, None], candidate, best)
                best_loss = torch.where(good, loss, best_loss)
                best_fraction=torch.where(good,torch.full_like(best_fraction,fraction),best_fraction)
            if physics is not None:
                best=weights+best_fraction[:,None]*delta
                best_loss,_,_=physics(profiles,targets,responses,best,product,time_weights=time_weights,
                                      avoided=avoided,compute_gradient=False)
            # Train on the state actually retained, not a rejected proposal.
            traces.append(best_loss)
            previous, weights = best - weights, best
        return weights, torch.stack(traces, -1)
