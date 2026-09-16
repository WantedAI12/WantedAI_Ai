"""Canonical solver states, with all original physical constraints rechecked."""

import numpy as np


def canonical_solver_weights(values, lower, upper, validator, *, tolerance=1e-10):
    x, lo, hi = (np.asarray(v, float) for v in (values, lower, upper))
    if (
        x.ndim != 1
        or x.shape != lo.shape
        or x.shape != hi.shape
        or not len(x)
        or not np.isfinite([tolerance]).all()
        or not 0 < tolerance <= 1e-8
        or not all(np.isfinite(v).all() for v in (x, lo, hi))
        or np.any(lo < 0)
        or np.any(lo > hi)
    ):
        return None
    if np.any(x < lo - tolerance) or np.any(x > hi + tolerance):
        return None
    canonical = np.clip(x, lo, hi)
    # Never renormalize across a budget/equality face without verifying it.
    return canonical if validator(canonical) else None
