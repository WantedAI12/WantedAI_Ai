"""Positive, mass-conserving propagation with augmented exposure moments.

An exponential of the interval-mean generator is composed in physical time.
Step doubling controls noncommuting drying/return/reaction generators. The
moments integrate the same positive propagator; they are not inferred from
plotting samples, divided by ventilation, or filled by a neural correction.
"""

from __future__ import annotations

import numpy as np

from .lotion_transport import bidirectional_step
from .unified_transport import _domain_mesh, interval_features, validate_raw

VERSION = "adaptive-positive-augmented-exposure/v68"


def constant_moment_kernel(
    evaporation, uptake, reaction, return_rate, ventilation, duration
):
    """Rows: initial film/air. Columns: five masses, integral film, integral air."""
    rates = np.broadcast_arrays(
        *[
            np.asarray(value, float)
            for value in (evaporation, uptake, reaction, return_rate, ventilation)
        ]
    )
    if (
        not np.isfinite(duration)
        or duration <= 0
        or any(not np.isfinite(value).all() or np.any(value < 0) for value in rates)
    ):
        raise ValueError("finite nonnegative rates and positive duration required")
    e, u, h, r, v = rates
    sink = u + h
    share = np.divide(u, sink, out=np.zeros_like(sink), where=sink > 0)
    rows = []
    for origin in (0, 1):
        result = bidirectional_step(
            np.full_like(e, 1.0 - origin),
            np.full_like(e, float(origin)),
            e,
            sink,
            r,
            v,
            duration,
            return_integrals=True,
        )
        film, air, loss, vent, film_integral, air_integral = result
        rows.append(
            np.stack(
                (
                    film,
                    air,
                    loss * share,
                    vent,
                    loss * (1 - share),
                    film_integral,
                    air_integral,
                ),
                axis=-1,
            )
        )
    kernel = np.stack(rows, axis=-2)
    if (
        not np.isfinite(kernel).all()
        or np.min(kernel, initial=0.0) < -1e-12
        or np.max(np.abs(kernel[..., :5].sum(axis=-1) - 1), initial=0.0) > 1e-8
    ):
        raise ValueError(
            "augmented transition violates finite positive mass conservation"
        )
    # Only a checked roundoff envelope is repaired, never an unstable solution.
    kernel = np.maximum(kernel, 0.0)
    return kernel


def compose_moments(first, second):
    """Exact semigroup composition for masses and additive occupation moments."""
    result = np.einsum("nbi,nij->nbj", first[:, :, :2], second)
    result[:, :, 2:] += first[:, :, 2:]
    return result


def _interval_kernel(rates, fractions, left, right):
    duration = right - left
    raw = validate_raw(interval_features(rates, fractions, left, right))
    e, u, h, r, v, decay, water, lipid = raw.T
    fixed = 1.0 - water
    inverse_capacity = np.divide(
        np.log1p(fixed * np.expm1(decay)),
        fixed * decay,
        out=np.ones_like(decay),
        where=decay > 1e-10,
    )
    active = 1.0 - lipid * inverse_capacity
    if np.min(active, initial=0.0) < -1e-12:
        raise ValueError("negative reactive phase capacity")
    # Dimensionless raw rates correspond to a unit interval. Restore minutes
    # for the moments without changing the exponential transition itself.
    kernel = constant_moment_kernel(
        e * inverse_capacity,
        u * inverse_capacity,
        h * np.maximum(active, 0.0),
        r,
        v,
        1.0,
    )
    kernel[:, :, 5:] *= duration
    return kernel


def exposure_trajectory(
    rates,
    fractions,
    times,
    *,
    duration,
    initial,
    work_budget=None,
    rtol=1e-7,
    atol=1e-10,
):
    """Return states, final state, cumulative film/air integrals and diagnostics.

    The accepted mesh is independent of display times. A displayed partial
    interval is integrated from its left mesh state and cannot modify the mesh.
    Error estimates are numerical local defects, not empirical physical bounds.
    """
    rates, fractions, times = (
        np.asarray(value, float) for value in (rates, fractions, times)
    )
    state = np.asarray(initial, float).copy()
    if (
        rates.ndim != 2
        or not len(rates)
        or rates.shape[1] != 6
        or fractions.shape != (len(rates), 2)
        or state.shape != (len(rates), 6)
        or times.ndim != 1
        or not len(times)
        or not np.isfinite(duration)
        or duration <= 0
        or not all(
            np.isfinite(value).all() for value in (rates, fractions, state, times)
        )
        or np.any(rates < 0)
        or np.any(fractions < 0)
        or np.any(state < 0)
        or np.any(fractions.sum(axis=1) > 1)
        or np.any(fractions[:, 0] > 0.999)
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or times[-1] > duration
        or not np.isfinite([rtol, atol]).all()
        or not 0 < rtol < 1
        or not 0 < atol < 1
    ):
        raise ValueError("invalid augmented exposure trajectory")
    budget = (
        {"remaining_material_transitions": 2_000_000}
        if work_budget is None
        else work_budget
    )
    if (
        type(budget.get("remaining_material_transitions")) is not int
        or not 0 <= budget["remaining_material_transitions"] <= 2_000_000
    ):
        raise ValueError("invalid augmented exposure work budget")
    work_start = budget["remaining_material_transitions"]
    accepted_cells, maximum_defect = 0, 0.0

    def kernel(left, right):
        if budget["remaining_material_transitions"] < len(rates):
            raise ValueError("augmented exposure work budget exceeded")
        budget["remaining_material_transitions"] -= len(rates)
        return _interval_kernel(rates, fractions, left, right)

    constant = bool(np.all((rates[:, 5] == 0) | (fractions[:, 0] == 0)))

    def integrate(left, right, depth=0):
        nonlocal accepted_cells, maximum_defect
        coarse = kernel(left, right)
        if constant:
            accepted_cells += 1
            return coarse
        middle = left + (right - left) / 2
        first, second = kernel(left, middle), kernel(middle, right)
        fine = compose_moments(first, second)
        difference = np.abs(fine - coarse)
        difference[:, :, 5:] /= right - left
        defect = float(difference.max()) / 3
        # Occupation moments/dt and transition masses are dimensionless and
        # bounded by one, permitting a common absolute local defect test.
        if defect <= atol * (right - left) / duration + rtol:
            maximum_defect = max(maximum_defect, defect)
            accepted_cells += 1
            return fine
        if depth >= 20 or middle in (left, right):
            raise ValueError("augmented exposure failed to resolve requested tolerance")
        return compose_moments(
            integrate(left, middle, depth + 1), integrate(middle, right, depth + 1)
        )

    def advance(current, exposure, operator):
        flow = np.einsum("ni,nij->nj", current[:, :2], operator)
        updated = current.copy()
        updated[:, :2] = flow[:, :2]
        updated[:, 2:5] += flow[:, 2:5]
        return updated, exposure + flow[:, 5:]

    edges = _domain_mesh(rates, fractions, np.linspace(0.0, 1.0, 33) ** 2 * duration)
    exposure = np.zeros((len(rates), 2))
    states, moments, index = [], [], 0
    if times[0] == 0:
        states.append(state.copy())
        moments.append(exposure.copy())
        index = 1
    for left, right in zip(edges[:-1], edges[1:]):
        operation = integrate(left, right)
        end_state, end_exposure = advance(state, exposure, operation)
        while index < len(times) and times[index] <= right:
            stop = times[index]
            if stop == right:
                sample_state, sample_exposure = end_state, end_exposure
            else:
                sample_state, sample_exposure = advance(
                    state, exposure, integrate(left, stop)
                )
            states.append(sample_state.copy())
            moments.append(sample_exposure.copy())
            index += 1
        state, exposure = end_state, end_exposure
    return (
        np.stack(states),
        state,
        np.stack(moments),
        exposure,
        {
            "method": VERSION,
            "local_dimensionless_defect_max": maximum_defect,
            "rtol": rtol,
            "atol": atol,
            "accepted_intervals_including_display": accepted_cells,
            "material_transition_evaluations": work_start
            - budget["remaining_material_transitions"],
            "learned_transport_proposal_applied": False,
            "error_scope": "step_doubling_local_estimate_not_measured_product_error",
        },
    )
