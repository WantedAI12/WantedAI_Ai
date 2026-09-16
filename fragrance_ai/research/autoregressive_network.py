"""Trainable stateful dose decoder sharing the core molecular representation."""

import torch
from torch import nn


def project_simplex(values, mask):
    values = values.masked_fill(~mask, -1e9)
    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1) - 1.0
    indices = torch.arange(1, values.shape[-1] + 1, device=values.device)
    count = (ordered - cumulative / indices > 0).sum(-1).clamp_min(1)
    threshold = cumulative.gather(1, (count - 1)[:, None]) / count[:, None]
    return torch.relu(values - threshold) * mask


def residual_features(
    profiles, targets, responses, weights, previous, product, step, mask
):
    from fragrance_ai.recommender.exposure_normalization import torch_normalized_exposure
    predicted,response_derivative=torch_normalized_exposure(profiles,responses,weights)
    residual = predicted - targets
    centered = torch.einsum("bhtd,bhnd->bhtn", residual, profiles)
    centered = centered - torch.einsum("bhtd,bhtd->bht", residual, predicted)[..., None]
    gradient = (centered * response_derivative[:, None]).mean(
        dim=(1, 2)
    )
    scale = gradient.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    root = ((residual**2).sum(-1).mean(dim=(1, 2)) + 1e-12).sqrt()
    variation = 0.5 * residual.abs().sum(-1).mean(dim=(1, 2))
    cosine = torch.einsum("bhtd,bhtd->bht", predicted, targets) / (
        predicted.norm(dim=-1) * targets.norm(dim=-1)
    ).clamp_min(1e-12)
    affinity = torch.einsum("bhnd,bhtd->bhtn", profiles, targets) / (
        profiles.norm(dim=-1)[:, :, None] * targets.norm(dim=-1)[..., None]
    ).clamp_min(1e-12)
    maximum = responses.amax(dim=(1, 2)).clamp_min(1e-12)

    def broadcast(value):
        return value[:, None].expand_as(weights)

    features = torch.stack(
        (
            gradient / scale,
            weights,
            weights.clamp_min(1e-6).log() / 14.0,
            previous,
            responses.mean(1) / maximum[:, None],
            (responses.amax(1) - responses.amin(1)) / maximum[:, None],
            broadcast(root),
            broadcast(variation),
            broadcast(cosine.amin(dim=(1, 2))),
            torch.full_like(weights, min(1.0, step / 8.0)),
            broadcast((product == 0).float()),
            broadcast((product == 1).float()),
            affinity.mean(dim=(1, 2)),
            scale.clamp_max(1e30).expand_as(weights),
            broadcast(targets.norm(dim=-1).mean(dim=(1, 2))),
            mask.float(),
        ),
        -1,
    )
    return features.to(weights.dtype), predicted


class AutoregressiveDoseCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.chemistry = nn.Linear(256, 64)
        self.cell = nn.GRUCell(80, 48)
        self.controls = nn.Linear(48, 2)
        nn.init.zeros_(self.controls.weight)
        with torch.no_grad():
            self.controls.bias.copy_(torch.tensor([-2.5, -2.0]))

    def step(self, embedded, features, state, weights, previous, mask):
        shape = state.shape
        inputs = torch.cat((embedded, features), -1).reshape(-1, 80)
        updated = self.cell(inputs, state.reshape(-1, 48)).reshape(shape)
        updated = torch.where(mask[..., None], updated, state)
        controls = self.controls(updated)
        rate, momentum = (
            2.0 * controls[..., 0].sigmoid(),
            0.85 * controls[..., 1].sigmoid(),
        )
        proposed = project_simplex(
            weights
            - rate * features[:, :, 0] * features[:, :, 6]
            + momentum * previous,
            mask,
        )
        return proposed, updated

    def forward(
        self, latents, profiles, targets, responses, initial, product, mask, steps=8
    ):
        embedded = torch.tanh(self.chemistry(latents))
        weights = initial
        previous = torch.zeros_like(weights)
        state = torch.zeros((*weights.shape, 48), device=weights.device)
        losses = []
        for step in range(steps):
            features, _ = residual_features(
                profiles, targets, responses, weights, previous, product, step, mask
            )
            proposed, state = self.step(
                embedded, features, state, weights, previous, mask
            )
            previous, weights = proposed - weights, proposed
            _, predicted = residual_features(
                profiles, targets, responses, weights, previous, product, step, mask
            )
            losses.append(((predicted - targets) ** 2).sum(-1).mean(dim=(1, 2)))
        return weights, torch.stack(losses, -1)
