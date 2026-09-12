"""Finite-dose O/W film, evaporation, uptake and ventilated-headspace model.

Rapid lipid/water equilibrium; each constant-volume substep is integrated
analytically. Midpoint water volume makes the varying-film solution approximate.
All transport coefficients must be supplied for the actual application context.
This is an uncalibrated research model, not a human perception or safety model.
"""
import hashlib
import json
import math

import numpy as np

from .models import SCENT_DIMENSIONS
from .lotion_transport import bidirectional_step
from .lotion_evaluation import evaluation_contract


def simulate_lotion(request, catalog, *, perception_guidance=None):
    """Transport plus optional configured research-only learned profile output."""
    from .lotion_perception import attach_lotion_perception, resolve_provider
    provider = resolve_provider(perception_guidance)
    from .lotion_surrogate import attach_release_prediction
    from .lotion_perception import make_lotion_shape_predictor
    predictor = make_lotion_shape_predictor(provider) if provider is not None else None
    result = attach_release_prediction(_simulate_lotion_transport(request, catalog), request, catalog,
                                       provider=provider, shape_predictor=predictor)
    return attach_lotion_perception(result, catalog, provider, _shape_predictor=predictor)


EXPOSURE_INTEGRATION_VERSION = "canonical_midpoint_analytic_air_integral_v65"


def _simulate_lotion_transport(request, catalog, *, exposure_windows=()):
    """Integrate a positive compartment system on an output-independent mesh.

    Each cell freezes the *physical* rates at its midpoint. Both observations
    and exposure-window boundaries use that same analytic semigroup. Windows
    accumulate local integrals directly, avoiding subtraction of almost equal
    lifetime exhaust totals in a late, low-concentration interval.
    """
    windows = tuple((float(a), float(b)) for a, b in exposure_windows)
    if len(windows) > 3 or any(not (math.isfinite(a) and math.isfinite(b)
            and 0 <= a < b <= request.times_minutes[-1]) for a, b in windows):
        raise ValueError("at most three finite exposure windows within the simulated horizon required")
    known = {row.ingredient_id: row for row in catalog.ingredients}
    ids = [row.ingredient_id for row in request.materials]
    missing = set(ids) - set(known)
    if missing:
        raise ValueError("unknown lotion ingredient IDs: " + ", ".join(sorted(missing)[:10]))
    if any(known[key].blocked for key in ids):
        raise ValueError("blocked ingredients cannot enter lotion simulation")
    # Diluted supplier materials need a carrier mass balance that this model
    # does not implement; silently treating their supplied mass as pure is wrong.
    if any(abs(known[key].active_strength_percent - 100) > 1e-8 for key in ids):
        raise ValueError("expand diluted ingredients into active and carrier components first")
    context = request.application_context
    base = {row.name.strip().casefold(): row for row in context.base_components}
    dose = context.application_mass_mg_cm2
    fragrance_fraction = context.fragrance_concentration_percent / 100.
    base_dose = dose * (1 - fragrance_fraction)
    aqueous_fixed = water = lipid = 0.
    for row in request.phase_components:
        component = base[row.name.casefold()]
        if row.portions is not None:
            for part in row.portions:
                volume = base_dose * component.mass_percent / 100. * part.mass_percent / 100. / (1000. * part.density_g_ml)
                if part.compartment == "lipid":
                    lipid += volume
                elif part.compartment == "water":
                    water += volume
                else:
                    aqueous_fixed += volume
            continue
        volume = base_dose * component.mass_percent / 100. / (1000. * row.density_g_ml)
        if row.phase == "lipid":
            lipid += volume
        elif component.role == "water":
            water += volume
        else:
            aqueous_fixed += volume
    materials = request.materials
    partition = np.array([row.lipid_water_partition for row in materials])
    air_partition = np.array([row.air_water_partition for row in materials])
    gas = np.array([row.gas_transfer_cm_min for row in materials]) * air_partition
    skin = np.array([row.skin_permeability_cm_min for row in materials])
    initial = dose * fragrance_fraction * np.array([row.concentrate_percent / 100. for row in materials])
    remaining = initial * np.array([row.initial_parent_fraction for row in materials])
    degraded, headspace = initial - remaining, np.zeros(len(ids))
    hydrolysis = np.array([row.aqueous_hydrolysis_per_min for row in materials])
    absorbed, exhausted = np.zeros(len(ids)), np.zeros(len(ids))
    profiles = np.array([known[key].vector() for key in ids])
    thresholds = np.array([row.odor_threshold_mg_m3 or np.nan for row in materials])
    full_thresholds = bool(np.all(np.isfinite(thresholds))) and request.profile_weighting != "air_mass"
    ventilation = request.air_exchange_per_min
    bidirectional = request.transport_mode == "bidirectional_air"
    return_rate = np.array([row.gas_transfer_cm_min for row in materials]) / request.headspace_height_cm
    if not bidirectional:
        return_rate = np.zeros_like(return_rate)
    output, max_error, max_backpressure = [], 0., 0.
    integrated_air = np.zeros(len(ids))
    integral_correction = np.zeros(len(ids))
    window_integrals = np.zeros((len(windows), len(ids)))
    window_corrections = np.zeros_like(window_integrals)

    def volumes(time):
        retained = request.retained_water_fraction
        aqueous = aqueous_fixed + water * (retained + (1 - retained) * math.exp(-request.water_loss_per_min * time))
        return aqueous, aqueous + partition * lipid

    if np.any(volumes(request.times_minutes[-1])[1] < 1e-18):
        raise ValueError("dry-film partition capacity is below the numerical model range")

    def snapshot(time, state, cumulative_air):
        nonlocal max_error, max_backpressure
        remaining, headspace, absorbed, exhausted, degraded = state
        aqueous, capacity = volumes(time)
        concentration = headspace / request.headspace_height_cm * 1e6  # mg/cm3 -> mg/m3
        surface_equilibrium = air_partition * remaining / capacity
        ratio = np.divide(headspace / request.headspace_height_cm, surface_equilibrium,
                          out=np.zeros_like(remaining), where=surface_equilibrium > 1e-20)
        if np.any((surface_equilibrium <= 1e-20) & (headspace > initial * 1e-12)):
            ratio = np.where((surface_equilibrium <= 1e-20) & (headspace > initial * 1e-12), 1e12, ratio)
        max_backpressure = max(max_backpressure, float(np.max(ratio)))
        error = float(np.max(np.abs(initial - remaining - headspace - absorbed - exhausted - degraded)))
        max_error = max(max_error, error)
        weights = concentration / thresholds if full_thresholds else concentration
        total = float(np.sum(weights))
        profile = weights @ profiles / total if total > 0 else None
        return {"minutes": time, "aqueous_volume_ml_cm2": aqueous, "lipid_volume_ml_cm2": lipid,
            "total_air_concentration_mg_m3": float(concentration.sum()),
            "scent_profile": dict(zip(SCENT_DIMENSIONS, profile.tolist())) if profile is not None else None,
            "profile_basis": "linear_odor_activity_proxy" if full_thresholds else "air_mass_weighted_catalog_proxy",
            "total_odor_activity_proxy": total if full_thresholds else None,
            "materials": [{"ingredient_id": key, "initial_mg_cm2": float(initial[i]),
                "remaining_mg_cm2": float(remaining[i]), "headspace_mg_cm2": float(headspace[i]),
                "skin_sink_mg_cm2": float(absorbed[i]), "ventilated_mg_cm2": float(exhausted[i]),
                "degraded_parent_equivalent_mg_cm2": float(degraded[i]),
                "air_concentration_mg_m3": float(concentration[i]),
                "cumulative_air_exposure_mg_min_m3": float(cumulative_air[i] * 1e6 / request.headspace_height_cm),
                "odor_activity_proxy": float(concentration[i] / thresholds[i]) if np.isfinite(thresholds[i]) else None}
                for i, key in enumerate(ids)]}

    def advance(state, dt, rates):
        if dt == 0:
            return state, np.zeros(len(ids))
        film, air, skin_mass, exhaust, destroyed = state
        evap, uptake, reaction = rates
        total_sink = uptake + reaction
        film, air, _, exhaust_delta, film_integral, air_integral = bidirectional_step(
            film, air, evap, total_sink, return_rate, ventilation, dt, return_integrals=True)
        return (film, air, skin_mass + uptake * film_integral,
                exhaust + exhaust_delta, destroyed + reaction * film_integral), air_integral

    def accumulate(total, correction, increment):
        # Compensated summation preserves small local contributions. No clipping
        # or renormalization is used to force positivity or conservation.
        adjusted = increment - correction
        result = total + adjusted
        correction[:] = (result - total) - adjusted
        total[:] = result

    state = (remaining, headspace, absorbed, exhausted, degraded)
    output.append(snapshot(0., state, integrated_air))
    cursor = 1
    step_size, horizon = request.integration_step_minutes, request.times_minutes[-1]
    steps = math.ceil(horizon / step_size)
    for step in range(steps):
        start, end = step * step_size, min((step + 1) * step_size, horizon)
        # For a partial last cell, use the same midpoint as a longer run. This
        # also makes common-time predictions independent of the chosen horizon.
        aqueous, capacity = volumes(start + .5 * step_size)
        rates = (gas / capacity, skin / capacity, hydrolysis * aqueous / capacity)
        final_state, increment = advance(state, end - start, rates)
        for index, (a, b) in enumerate(windows):
            left, right = max(start, a), min(end, b)
            if right <= left:
                continue
            if left == start and right == end:
                local_integral = increment
            else:
                left_state, _ = advance(state, left - start, rates)
                _, local_integral = advance(left_state, right - left, rates)
            accumulate(window_integrals[index], window_corrections[index], local_integral)
        while cursor < len(request.times_minutes) and request.times_minutes[cursor] <= end:
            time = request.times_minutes[cursor]
            if time == end:
                observed, partial = final_state, increment
            else:
                observed, partial = advance(state, time - start, rates)
            output.append(snapshot(time, observed, integrated_air + partial))
            cursor += 1
        accumulate(integrated_air, integral_correction, increment)
        state = final_state
        # Check conservation at every integration step, not just display times.
        max_error = max(max_error, float(np.max(np.abs(initial - sum(state)))))
        if not bidirectional:
            surface = air_partition * state[0] / capacity
            ratio = np.divide(state[1] / request.headspace_height_cm, surface,
                              out=np.zeros_like(surface), where=surface > 1e-20)
            ratio[(surface <= 1e-20) & (state[1] > initial * 1e-12)] = 1e12
            max_backpressure = max(max_backpressure, float(np.max(ratio)))
    exposure = []
    for (a, b), integral in zip(windows, window_integrals):
        concentration_integral = integral * 1e6 / request.headspace_height_cm
        mean = concentration_integral / (b - a)
        exposure.append({"window_start_minutes": a, "minutes": b,
            "aggregation": EXPOSURE_INTEGRATION_VERSION,
            "total_mean_air_concentration_mg_m3": float(mean.sum()),
            "materials": [{"ingredient_id": key,
                "air_exposure_mg_min_m3": float(concentration_integral[i]),
                "mean_air_concentration_mg_m3": float(mean[i]),
                "mean_odor_activity_proxy": float(mean[i] / thresholds[i]) if np.isfinite(thresholds[i]) else None}
                for i, key in enumerate(ids)]})
    if (not np.isfinite(state).all() or np.any(np.asarray(state) < 0)
            or not np.isfinite(integrated_air).all() or np.any(integrated_air < 0)
            or not np.isfinite(window_integrals).all() or np.any(window_integrals < 0)):
        raise ValueError("lotion transport failed finite nonnegative state/integral checks")
    # A forward-model defect must not become a scored candidate. This is a
    # numerical conservation guard, not a confidence interval or safety limit.
    if np.any(np.abs(initial-sum(state)) > 1e-8 * np.maximum(initial, np.finfo(float).tiny)):
        raise ValueError("lotion transport failed per-material conservation check")
    peak = max(point["total_air_concentration_mg_m3"] for point in output)
    for point in output:
        point["relative_to_sampled_peak_air_mass_percent"] = point["total_air_concentration_mg_m3"] / peak * 100 if peak else None
    valid_sink = max_backpressure <= .1
    identity = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"schema_version": "body-lotion-simulation-1", "request_id": identity,
        "product_model": {**evaluation_contract(), "aggregation": "not_scored_simulation_only"},
        "status": "research_simulation" if bidirectional or valid_sink else "outside_open_sink_assumption",
        "model": "finite_dose_rapid_partition_bidirectional_v2" if bidirectional else "finite_dose_rapid_partition_open_sink_v1", "temporal_profile": output,
        "exposure_windows": exposure,
        "parameter_context_id": request.parameter_context_id, "parameter_source_verified": False,
        "caller_declared_parameter_sources": [{"ingredient_id": row.ingredient_id, "kind": row.source_kind,
            "reference": row.source_reference, "date": str(row.source_date)} for row in materials],
        "water_loss_source_reference": request.water_loss_source_reference,
        "base_phase_assignments": [row.model_dump(mode="json") for row in request.phase_components],
        "base_phase_assignment_source_verified": False,
        "diagnostics": {"mass_balance_max_abs_error_mg_cm2": max_error,
            "maximum_air_to_surface_equilibrium_ratio": max_backpressure, "open_sink_assumption_met": valid_sink,
            "open_sink_assumption_required": not bidirectional,
            "ratio_sampling": "output_timepoints" if bidirectional else "integration_steps_and_output_timepoints",
            "integration_step_minutes": request.integration_step_minutes,
            "integration_version": EXPOSURE_INTEGRATION_VERSION,
            "integration_mesh": "fixed_step_origin_zero_independent_of_output_times",
            "exposure_integral_method": "analytic_local_air_integrals_compensated_sum",
            "coefficient_time_approximation": "second_order_midpoint_frozen_rates_not_exact_variable_coefficient_solution",
            "integration_steps": steps,
            "odor_threshold_coverage_percent": 100 * sum(row.odor_threshold_mg_m3 is not None for row in materials) / len(materials)},
        "human_similarity_percent": None, "longevity_hours": None, "manufacturing_approved": False,
        "calibrated_lotion_generator_enabled": False,
        "limitations": ["User-supplied context-specific transport coefficients are not source-verified or fitted here.",
            "Rapid lipid/water equilibrium and dilute additive phase volumes; no micellar kinetics or crystallization. Optional aqueous parent hydrolysis is included only with supplied kinetic provenance; products and their odors are not modeled.",
            "Gas back-transfer from a well-mixed finite air reservoir is included." if bidirectional else "Gas transfer assumes an open sink; concentrations exceeding 10% of surface equilibrium flag this approximation as invalid.",
            "Skin is a one-way sink, not a reversible skin reservoir or systemic exposure model.",
            "Linear catalog/odor-activity profiles do not model perceptual masking, synergy or human similarity.",
            "No temperature or humidity extrapolation: coefficients must correspond to the exact supplied context.",
            "Relative peak is based on returned sampling times, not a continuous peak or human detection duration.",
            "This fixed-composition simulation does not optimize or approve a lotion formula."]}
