"""Evaluate all nonlinear line-search widths in one dense kernel operation.

Exact same states and score definitions; this batches work, not candidate
pruning, smaller Monte Carlo counts, or a different acceptance threshold.
"""

import numpy as np


def numpy_values(model, profiles, targets, weights, time_weights, avoided):
    w, p, q = (
        np.asarray(weights, float),
        np.asarray(profiles, float),
        np.asarray(targets, float),
    )
    denominator = model.base_moles + w @ model.total_moles
    activity = (
        w[:, None, None, :]
        * model.coefficients[None]
        / denominator[:, None, None, None]
    )
    powered = activity**0.55
    response = powered / (1 + powered) * model.transport
    suppressed = response / (
        1 + model.suppression[None, :, None, None] * (response @ model.interaction.T)
    )
    raw = np.einsum("ckti,hid->ckthd", suppressed, p, optimize=True)
    scaled = np.divide(
        raw,
        raw.sum(-1, keepdims=True),
        out=np.zeros_like(raw),
        where=raw.sum(-1, keepdims=True) > 0,
    )
    nominal_raw = np.einsum("ci,hid->chd", w * model.gain, p, optimize=True)
    nominal = np.divide(
        nominal_raw,
        nominal_raw.sum(-1, keepdims=True),
        out=np.zeros_like(nominal_raw),
        where=nominal_raw.sum(-1, keepdims=True) > 0,
    )
    predicted = np.concatenate(
        (nominal[:, :, None], scaled.mean(1).transpose(0, 2, 1, 3)), 2
    )
    cosine = (predicted * q).sum(-1) / np.maximum(
        np.linalg.norm(predicted, axis=-1) * np.linalg.norm(q, axis=-1), 1e-30
    )
    mask = np.zeros_like(q) if avoided is None else np.asarray(avoided)
    points = np.maximum(
        np.maximum(0.5 * np.abs(predicted - q).sum(-1), 1 - cosine),
        (predicted * mask).sum(-1),
    )
    tw = np.array(time_weights, float, copy=True)
    tw[0] = 0.0
    tw /= tw.sum()
    return np.maximum(points[:, :, 0], points @ tw).max(-1)


def torch_values(physics, profiles, targets, weights, time_weights, avoided):
    import torch

    m = physics.values
    denominator = physics.base + weights @ m["total_moles"]
    activity = (
        weights[:, None, None, :]
        * m["coefficients"][None]
        / denominator[:, None, None, None]
    )
    powered = activity.clamp_min(1e-30) ** 0.55
    response = torch.where(
        activity > 0,
        powered / (1 + powered) * m["transport"],
        torch.zeros_like(powered),
    )
    suppressed = response / (
        1 + m["suppression"][None, :, None, None] * (response @ m["interaction"].T)
    )
    raw = torch.einsum("ckti,hid->ckthd", suppressed, profiles)
    scaled = raw / raw.sum(-1, keepdim=True).clamp_min(1e-30)
    nr = torch.einsum("ci,hid->chd", weights * m["gain"], profiles)
    nominal = nr / nr.sum(-1, keepdim=True).clamp_min(1e-30)
    predicted = torch.cat((nominal[:, :, None], scaled.mean(1).permute(0, 2, 1, 3)), 2)
    cosine = (predicted * targets).sum(-1) / (
        predicted.norm(dim=-1) * targets.norm(dim=-1)
    ).clamp_min(1e-30)
    mask = torch.zeros_like(targets) if avoided is None else avoided
    points = torch.maximum(
        torch.maximum(0.5 * (predicted - targets).abs().sum(-1), 1 - cosine),
        (predicted * mask).sum(-1),
    )
    tw = time_weights.clone()
    tw[0] = 0.0
    tw = tw / tw.sum()
    return torch.maximum(points[:, :, 0], points @ tw).amax(-1)
