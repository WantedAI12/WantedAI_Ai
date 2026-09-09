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


def _simulate_lotion_transport(request, catalog):
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
    output, max_error, max_backpressure = [], 0., 0.

    def volumes(time):
        retained = request.retained_water_fraction
        aqueous = aqueous_fixed + water * (retained + (1 - retained) * math.exp(-request.water_loss_per_min * time))
        return aqueous, aqueous + partition * lipid

    if np.any(volumes(request.times_minutes[-1])[1] < 1e-18):
        raise ValueError("dry-film partition capacity is below the numerical model range")

    def snapshot(time):
        nonlocal max_error, max_backpressure
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
                "odor_activity_proxy": float(concentration[i] / thresholds[i]) if np.isfinite(thresholds[i]) else None}
                for i, key in enumerate(ids)]}

    output.append(snapshot(0.))
    for start, end in zip(request.times_minutes, request.times_minutes[1:]):
        steps = math.ceil((end - start) / request.integration_step_minutes)
        dt = (end - start) / steps
        for step in range(steps):
            midpoint = start + (step + .5) * dt
            aqueous, capacity = volumes(midpoint)
            evap, uptake = gas / capacity, skin / capacity
            reaction = hydrolysis * aqueous / capacity
            total_sink = uptake + reaction
            if bidirectional:
                remaining, headspace, skin_delta, exhaust_delta = bidirectional_step(
                    remaining, headspace, evap, total_sink, return_rate, ventilation, dt)
                share = np.divide(uptake, total_sink, out=np.zeros_like(uptake), where=total_sink > 0)
                absorbed += skin_delta * share
                degraded += skin_delta * (1-share)
                exhausted += exhaust_delta
                continue
            loss = evap + total_sink
            lost = remaining * (-np.expm1(-loss * dt))
            emitted = lost * evap / loss
            absorbed += lost * uptake / loss
            degraded += lost * reaction / loss
            # Stable exact convolution, including equal transfer/ventilation rates.
            gap = np.abs(loss - ventilation)
            kernel = np.exp(-np.minimum(loss, ventilation) * dt) * np.divide(
                -np.expm1(-gap * dt), gap, out=np.full_like(gap, dt), where=gap > 0)
            new_air = headspace * math.exp(-ventilation * dt) + evap * remaining * kernel
            exhausted += headspace + emitted - new_air
            remaining -= lost
            headspace = new_air
            # Check the open-sink assumption throughout, not just display times.
            surface = air_partition * remaining / capacity
            ratio = np.divide(headspace / request.headspace_height_cm, surface,
                              out=np.zeros_like(surface), where=surface > 1e-20)
            ratio[(surface <= 1e-20) & (headspace > initial * 1e-12)] = 1e12
            max_backpressure = max(max_backpressure, float(np.max(ratio)))
        output.append(snapshot(end))
    peak = max(point["total_air_concentration_mg_m3"] for point in output)
    for point in output:
        point["relative_to_sampled_peak_air_mass_percent"] = point["total_air_concentration_mg_m3"] / peak * 100 if peak else None
    valid_sink = max_backpressure <= .1
    identity = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"schema_version": "body-lotion-simulation-1", "request_id": identity,
        "product_model": {**evaluation_contract(), "aggregation": "not_scored_simulation_only"},
        "status": "research_simulation" if bidirectional or valid_sink else "outside_open_sink_assumption",
        "model": "finite_dose_rapid_partition_bidirectional_v2" if bidirectional else "finite_dose_rapid_partition_open_sink_v1", "temporal_profile": output,
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
