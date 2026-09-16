"""Roundoff-only mass repair without injecting absent ingredients.

Already feasible floating-point compositions keep their support. The residual
is assigned inside an existing positive coordinate's bound, never smeared over
thousands of zeros. No positive dose is pruned, even at trace concentration.
The shared tolerance follows the float32 neural calculation precision even in
the float64 CPU oracle, preventing different feasible faces across runtimes.
"""

import numpy as np


def repair_mass_residual(values, lower, upper):
    result = np.array(values, dtype=float, copy=True)
    for _ in range(2):
        residual = 1 - result.sum(-1)
        space = np.where(residual[:, None] >= 0, upper - result, result - lower)
        eligible = (result > 0) & (space > 0)
        index = np.where(eligible, space, -np.inf).argmax(-1)
        rows = np.arange(len(result))
        delta = np.minimum(
            np.abs(residual), np.maximum(space[rows, index], 0.0)
        ) * np.sign(residual)
        result[rows, index] += delta
    return result


def preserve_feasible_support(original, projected, lower, upper, epsilon):
    near = np.all((original >= lower) & (original <= upper), axis=-1) & (
        np.abs(original.sum(-1) - 1) <= 16 * epsilon
    )
    selected = np.where(near[:, None], original, projected)
    return repair_mass_residual(selected, lower, upper)


def torch_preserve_feasible_support(original, projected, lower, upper, epsilon):
    import torch

    near = ((original >= lower) & (original <= upper)).all(-1) & (
        (original.sum(-1) - 1).abs() <= 16 * epsilon
    )
    result = torch.where(near[:, None], original, projected)
    for _ in range(2):
        residual = 1 - result.sum(-1)
        space = torch.where(residual[:, None] >= 0, upper - result, result - lower)
        eligible = (result > 0) & (space > 0)
        index = space.masked_fill(~eligible, -torch.inf).argmax(-1, keepdim=True)
        delta = (
            torch.minimum(residual.abs()[:, None], space.gather(-1, index).clamp_min(0))
            * residual.sign()[:, None]
        )
        result = result.scatter_add(-1, index, delta)
    return result
