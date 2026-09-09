"""Explicit physical submodels; no clinical skin or shelf-life certification."""
import math


def permeability_estimates(molecular_weight, logp):
    if not math.isfinite(molecular_weight) or molecular_weight <= 0 or not math.isfinite(logp):
        raise ValueError("finite molecular weight and logP required")
    if not 18 <= molecular_weight <= 1000 or not -3 <= logp <= 6:
        return {"status": "outside_selected_qspr_domain", "cm_min": None}
    potts = 10 ** (-2.72 + .71 * logp - .0061 * molecular_weight) / 60
    epa = 10 ** (-2.80 + .66 * logp - .0056 * molecular_weight) / 60
    return {"status": "aqueous_qspr_estimate", "cm_min": potts, "epa_cm_min": epa,
        "model_disagreement_range_cm_min": [min(potts, epa), max(potts, epa)],
        "sources": ["https://pubmed.ncbi.nlm.nih.gov/1608900/",
                    "https://www.epa.gov/sites/default/files/2015-09/documents/part_e_final_revision_10-03-07.pdf"],
        "domain_kind": "selected_engineering_range_not_validated_lotion_domain",
        "skin_compatibility_verified": False}


def aqueous_hydrolysis_rate(parameters, ph):
    if not parameters.minimum_ph <= ph <= parameters.maximum_ph:
        raise ValueError("pH outside coefficient domain")
    # Ideal dilute aqueous activities, pKw=14 at the explicitly fixed 25 C.
    return (parameters.neutral_per_min + parameters.acid_m_inv_min * 10**(-ph)
            + parameters.base_m_inv_min * 10**(ph-14))


def storage_retention(parameters, conditions, aqueous_fraction):
    if not 0 <= aqueous_fraction <= 1:
        raise ValueError("aqueous partition fraction must be in [0,1]")
    aqueous_rate = aqueous_hydrolysis_rate(parameters, conditions.ph)
    rate = aqueous_rate * aqueous_fraction
    return {"parent_fraction": math.exp(-rate * conditions.days * 1440),
            "aqueous_rate_per_min": aqueous_rate,
            "effective_rate_per_min": rate,
            "half_life_days": math.log(2) / rate / 1440 if rate > 0 else None,
            "zero_rate_in_supplied_model": rate == 0,
            "source_reference": parameters.source_reference, "source_kind": parameters.source_kind,
            "source_verified": False,
            "scope": "fixed_pH_25C_rapid_partition_parent_hydrolysis_not_full_emulsion_stability"}


def skin_exposure_report(result, exposure, catalog):
    if not result.get("simulation"):
        return {"status": "no_formula", "skin_compatibility_verified": False}
    scenarios = [result["simulation"]] + result.get("scenario_simulations", [])
    endpoints = [s["temporal_profile"][-1] for s in scenarios]
    uptake = [sum(m["skin_sink_mg_cm2"] for m in point["materials"]) for point in endpoints]
    known = {i.ingredient_id: i for i in catalog.ingredients}
    lines = result.get("recipe") or result.get("closest_candidate") or []
    flags = [{"ingredient_id": line["ingredient_id"], "catalog_allergen_flags": list(known[line["ingredient_id"]].eu_allergens),
              "structural_alerts": list(known[line["ingredient_id"]].registry_structural_alerts),
              "finished_product_percent": line["finished_product_percent"]}
             for line in lines if known[line["ingredient_id"]].eu_allergens or known[line["ingredient_id"]].registry_structural_alerts]
    maximum = max(uptake)
    return {"status": "modeled_exposure_only_skin_compatibility_unresolved", "skin_type": exposure.skin_type,
            "modeled_uptake_mg_cm2_by_scenario": uptake,
            "modeled_daily_sink_dose_mg_kg": maximum * exposure.area_cm2 * exposure.applications_per_day / exposure.body_mass_kg,
            "area_cm2": exposure.area_cm2, "body_mass_kg": exposure.body_mass_kg,
            "applications_per_day": exposure.applications_per_day,
            "maximum_modeled_uptake_mg_cm2": exposure.maximum_modeled_uptake_mg_cm2,
            "caller_uptake_constraint_met": maximum <= exposure.maximum_modeled_uptake_mg_cm2 * (1+1e-8)
                if exposure.maximum_modeled_uptake_mg_cm2 is not None else None,
            "flagged_materials": flags, "unflagged_does_not_mean_nonallergenic": True,
            "skin_compatibility_verified": False,
            "limitations": ["One-way skin sink is not measured systemic absorption.",
                "No validated dry/oily/sensitive-skin multiplier or sensitisation potency is inferred.",
                "Uptake limit is caller-supplied, not a clinical safety threshold."]}
