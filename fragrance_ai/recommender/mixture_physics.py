"""Shared mass/response primitives for search, training and forward evaluation.

Masses are grams per 100 g finished product. The implicit bulk matrix is
ethanol. An unspecified stock carrier uses that same declared reference;
explicit unknown carriers are rejected, not silently assigned ethanol.
"""

import numpy as np

ETHANOL_MOLECULAR_WEIGHT = 46.06844
CARRIER_MW = {
    "ethanol": 46.06844,
    "alcohol": 46.06844,
    "water": 18.01528,
    "dpg": 134.174,
    "dipropylene glycol": 134.174,
    "tec": 276.283,
    "triethyl citrate": 276.283,
}


def carrier_moles_per_gram(ingredients, strengths=None):
    ingredients = list(ingredients)
    strengths = np.asarray(
        [i.active_strength_percent for i in ingredients]
        if strengths is None
        else strengths,
        float,
    )
    if (
        strengths.shape != (len(ingredients),)
        or not np.isfinite(strengths).all()
        or np.any((strengths < 0) | (strengths > 100))
    ):
        raise ValueError("stock strengths must be in [0, 100] percent")
    result = np.zeros(len(ingredients))
    for j, (item, s) in enumerate(zip(ingredients, strengths)):
        if s == 100:
            continue
        carrier = (item.carrier or "ethanol").casefold().strip()
        if carrier not in CARRIER_MW:
            raise ValueError(
                "unknown stock carrier; explicit molecular composition required: "
                + carrier
            )
        result[j] = (1.0 - s / 100.0) / CARRIER_MW[carrier]
    return result


def hill_response(activity):
    activity = np.asarray(activity, float)
    if not np.isfinite(activity).all() or np.any(activity < 0):
        raise ValueError("finite nonnegative odor activity required")
    powered = activity**0.55
    return powered / (1.0 + powered)


def interaction_matrix(vectors, molecular_weights, logp, has_properties):
    vectors = np.asarray(vectors, float)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("nonempty material profile matrix required")
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    profile = np.clip(normalized @ normalized.T, 0.0, 1.0)
    mw, lp, known = (
        np.asarray(molecular_weights),
        np.asarray(logp),
        np.asarray(has_properties, bool),
    )
    chemical = np.exp(-np.abs(mw[:, None] - mw[None, :]) / 300.0)
    polarity = np.exp(-np.abs(lp[:, None] - lp[None, :]) / 3.0)
    chemical *= np.where(np.isfinite(polarity), polarity, 0.5)
    chemical = np.where(known[:, None] & known[None, :], chemical, 0.5)
    result = np.clip(0.7 * profile + 0.3 * chemical, 0.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result
