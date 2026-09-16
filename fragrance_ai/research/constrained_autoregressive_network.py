"""Differentiable bounded/price-feasible GRU optimizer, matched to CPU inference."""

import torch
from torch import nn


class _CappedSimplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, lower, upper):
        if values.is_cuda:
            # These are small constrained projections, not dense neural work.
            # Use the very same float64 NumPy operator as CPU inference. This
            # avoids 48 GPU reduction/synchronization rounds per recurrent step.
            from fragrance_ai.recommender.constrained_autoregressive import capped_simplex
            result = torch.as_tensor(capped_simplex(values.detach().cpu().numpy(),
                lower.detach().cpu().numpy(),upper.detach().cpu().numpy()),
                device=values.device,dtype=values.dtype)
            ctx.save_for_backward((result>lower)&(result<upper))
            return result
        v, low, u = values.double(), lower.double(), upper.double()
        left = (v - u).amin(-1, keepdim=True)
        right = (v - low).amax(-1, keepdim=True)
        for _ in range(48):
            middle = (left + right) * 0.5
            mass = torch.clamp(v - middle, min=low, max=u).sum(-1, keepdim=True)
            left = torch.where(mass > 1, middle, left)
            right = torch.where(mass > 1, right, middle)
        from fragrance_ai.recommender.mass_projection import torch_preserve_feasible_support
        projected = torch.clamp(v - (left + right) * 0.5, min=low, max=u)
        result=torch_preserve_feasible_support(v,projected,low,u,max(torch.finfo(values.dtype).eps,torch.finfo(torch.float32).eps)).to(values.dtype)
        # Genuine trace doses remain differentiable. A fixed 1e-7 margin
        # incorrectly froze them although they were strictly inside bounds.
        free = (result > lower) & (result < upper)
        ctx.save_for_backward(free)
        return result

    @staticmethod
    def backward(ctx, gradient):
        (free,) = ctx.saved_tensors
        average = (gradient * free).sum(-1, keepdim=True) / free.sum(
            -1, keepdim=True
        ).clamp_min(1)
        return (gradient - average) * free, None, None


def cheapest_feasible(lower, upper, prices):
    order = prices.argsort(dim=-1, stable=True)
    capacity = (upper - lower).gather(-1, order)
    preceding = capacity.cumsum(-1) - capacity
    additions = torch.minimum(
        capacity, torch.relu(1 - lower.sum(-1, keepdim=True) - preceding)
    )
    return lower.scatter_add(-1, order, additions)


def project_constraints(values, lower, upper, prices, budget, cheapest):
    projected = _CappedSimplex.apply(values, lower, upper)
    cost = (projected * prices).sum(-1)
    minimum = (cheapest * prices).sum(-1)
    fraction = torch.where(
        cost > budget + 1e-7,
        ((cost - budget) / (cost - minimum).clamp_min(1e-12)).clamp(0, 1),
        torch.zeros_like(cost),
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
    from .autoregressive_network import residual_features

    legacy, predicted = residual_features(
        profiles, targets, responses, weights, previous, product, step, mask
    )
    residual = predicted - targets
    norm = predicted.norm(dim=-1).clamp_min(1e-12)
    goal_norm = targets.norm(dim=-1).clamp_min(1e-12)
    cosine = torch.einsum("bhtd,bhtd->bht", predicted, targets) / (norm * goal_norm)
    tv = 0.5 * residual.abs().sum(-1)
    losses = torch.stack((tv, 1 - cosine), -1)
    attention = torch.softmax((64 * losses).flatten(1), -1).reshape_as(losses)
    partial = attention[..., 0, None] * 0.5 * residual.sign() + attention[
        ..., 1, None
    ] * (
        cosine[..., None] * predicted / norm[..., None] ** 2
        - targets / (norm * goal_norm)[..., None]
    )
    from fragrance_ai.recommender.exposure_normalization import torch_normalized_exposure
    _,response_derivative=torch_normalized_exposure(profiles,responses,weights)
    gradient = (
        (
            torch.einsum("bhtd,bhnd->bhtn", partial, profiles)
            - torch.einsum("bhtd,bhtd->bht", partial, predicted)[..., None]
        )
        * response_derivative[:, None]
    )
    gradient = gradient.sum(dim=(1, 2)) * mask
    scale = gradient.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    extra = torch.stack(
        (
            gradient / scale,
            losses.amax(dim=(1, 2, 3))[:, None].expand_as(weights),
            upper,
            lower,
            prices / budget[:, None],
            torch.log1p(mask.sum(-1).float())[:, None].expand_as(weights)
            / 8.3180102775,
            (upper - weights) / (upper - lower).clamp_min(1e-8),
            ((weights * prices).sum(-1) / budget)[:, None].expand_as(weights),
        ),
        -1,
    )
    return torch.cat((legacy, extra), -1).to(weights.dtype), losses.amax(dim=(1, 2, 3)), predicted


class ConstrainedAutoregressiveCell(nn.Module):
    def __init__(self, hidden_size=48):
        super().__init__()
        if isinstance(hidden_size,bool) or hidden_size not in (48,96,128):
            raise ValueError('supported recurrent widths are 48, 96 and 128')
        self.chemistry = nn.Linear(256, 64)
        self.cell = nn.GRUCell(88, hidden_size)
        self.controls = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.controls.weight)
        with torch.no_grad():
            self.controls.bias.copy_(torch.tensor([-2.5, -2.0, -1.5]))

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
    ):
        mask = upper > lower
        cheapest = cheapest_feasible(lower, upper, prices)
        weights = project_constraints(initial, lower, upper, prices, budget, cheapest)
        embedded = torch.tanh(self.chemistry(latents))
        previous = torch.zeros_like(weights)
        state = torch.zeros((*weights.shape, self.cell.hidden_size), device=weights.device)
        traces = []
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
            next_state = self.cell(
                torch.cat((embedded, f), -1).reshape(-1, 88), state.reshape(-1, self.cell.hidden_size)
            ).reshape_as(state)
            state = torch.where(mask[..., None], next_state, state)
            control = self.controls(state)
            rate, momentum = (
                2 * control[..., 0].sigmoid(),
                0.85 * control[..., 1].sigmoid(),
            )
            radius = (
                0.025
                + 0.475
                * (
                    (control[..., 2] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                ).sigmoid()
            )
            raw = weights - rate * f[:, :, 16] * f[:, :, 17] + momentum * previous
            proposed = project_constraints(raw, lower, upper, prices, budget, cheapest)
            distance = (proposed - weights).abs().sum(-1)
            fraction = (radius / distance.clamp_min(1e-12)).clamp_max(1)
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
            good = (after <= before + 1e-8).detach()
            chosen = torch.where(good[:, None], proposed, weights)
            # Preserve learning signals from a rejected overshoot, while the
            # next recurrent state consumes the actually retained masses.
            traces.append(after + torch.relu(after - before))
            previous, weights = chosen - weights, chosen
        return weights, torch.stack(traces, -1)
