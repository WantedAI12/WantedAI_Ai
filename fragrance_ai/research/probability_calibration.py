"""Positive-slope held-out logit calibration; not a recipe-score adjustment."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_logit_calibration(logits, labels):
    x, y = np.asarray(logits, float), np.asarray(labels, float)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isin(y, [0.0, 1.0]).all()
        or len(set(y)) != 2
    ):
        raise ValueError("finite held-out logits and both observed classes required")

    def loss(v):
        z = v[0] * x + v[1]
        return float(
            np.mean(np.logaddexp(0, z) - y * z) + 1e-4 * ((v[0] - 1) ** 2 + v[1] ** 2)
        )

    def jac(v):
        residual = expit(v[0] * x + v[1]) - y
        return np.array(
            [np.mean(residual * x) + 2e-4 * (v[0] - 1), residual.mean() + 2e-4 * v[1]]
        )

    result = minimize(
        loss,
        [1.0, 0.0],
        jac=jac,
        bounds=[(0.01, 20.0), (-10.0, 10.0)],
        method="L-BFGS-B",
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("probability calibration did not converge")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "rows": len(y),
        "method": "positive_slope_regularized_logit_calibration",
        "fit_split": "validation",
        "test_labels_used": False,
    }
