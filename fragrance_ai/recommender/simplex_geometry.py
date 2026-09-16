"""Intrinsic diagonal-metric gradient on the mass-conservation tangent space.

Solve min_d g^T d + 0.5 d^T D^-1 d subject to sum(d)=0.
This removes the arbitrary additive gradient constant before feature scaling.
Positive diagonal Hill scaling can regularize the one-sided zero-dose slope;
it does not alter predictions, the loss, the bounds, or acceptance thresholds.
"""

import numpy as np

GEOMETRIES = ("simplex_tangent", "hill_simplex_tangent")


def tangent_gradient(gradient, weights, mask, product, geometry):
    if geometry not in GEOMETRIES:
        raise ValueError("unknown constrained gradient geometry")
    g, w = np.asarray(gradient, float), np.asarray(weights, float)
    mask = np.asarray(mask, bool)
    if (
        g.shape != w.shape
        or mask.shape != g.shape
        or not np.isfinite(g).all()
        or np.any(w < 0)
    ):
        raise ValueError("finite matched gradient and nonnegative mass required")
    metric = mask.astype(float)
    if geometry == "hill_simplex_tangent":
        metric *= np.where((np.asarray(product) == 0)[:, None], (w + 1e-8) ** 0.45, 1.0)
    # Subtract a free-coordinate pivot first: a truly constant gradient then
    # remains EXACTLY zero, not roundoff that normalization amplifies to one.
    pivot = np.take_along_axis(g, mask.argmax(-1)[:, None], axis=-1)
    centered = g - pivot
    center = (metric * centered).sum(-1, keepdims=True) / np.maximum(
        metric.sum(-1, keepdims=True), 1e-30
    )
    return metric * (centered - center)


def torch_tangent_gradient(gradient, weights, mask, product, geometry):
    import torch

    if geometry not in GEOMETRIES:
        raise ValueError("unknown constrained gradient geometry")
    metric = mask.to(gradient.dtype)
    if geometry == "hill_simplex_tangent":
        metric = metric * torch.where(
            (product == 0)[:, None], (weights + 1e-8) ** 0.45, torch.ones_like(weights)
        )
    pivot = gradient.gather(-1, mask.to(torch.int64).argmax(-1)[:, None])
    centered = gradient - pivot
    center = (metric * centered).sum(-1, keepdim=True) / metric.sum(
        -1, keepdim=True
    ).clamp_min(1e-30)
    return metric * (centered - center)
