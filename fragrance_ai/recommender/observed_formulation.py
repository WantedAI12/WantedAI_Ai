"""Measured outcome and pair-annotation readouts from the ONE shared core."""

import numpy as np


def aqueous_context(percentages):
    x = np.asarray(percentages, float)
    if (
        x.ndim != 2
        or x.shape[1] != 18
        or not np.isfinite(x).all()
        or np.any(x < 0)
        or np.any(x.sum(-1) > 100 + 1e-8)
    ):
        raise ValueError("18 finite as-supplied percentages, sum <= 100, required")
    context = np.zeros((len(x), 64), np.float32)
    context[:, 2] = 1.0
    context[:, 11] = -1.0  # separate observed aqueous outcome task from droplet task
    context[:, 33:51] = x / 30.0
    context[:, 51] = (100 - x.sum(-1)) / 100.0
    return context


def predict_liquid(core, ingredient_percent, protocol):
    from .formulation_core import sigmoid, SYSTEM_VERSION

    if core.version != SYSTEM_VERSION:
        raise ValueError("observed outcome head requires trained V76 checkpoint")
    specification = core.manifest["aqueous"]
    names = specification["ingredient_names"]
    if protocol != specification["reference_protocol"] or set(ingredient_percent) - set(
        names
    ):
        raise ValueError(
            "exact observed protocol and supported ingredient identities required"
        )
    x = np.array([[ingredient_percent.get(name, 0.0) for name in names]], float)
    context = aqueous_context(x)
    result = core.forward(
        np.zeros((1, 1, core.feature_width)), np.zeros((1, 1)), context
    )["aqueous"][0]
    values = result * np.asarray(specification["outcome_scale"]) + np.asarray(
        specification["outcome_mean"]
    )
    within = bool(
        np.all((x[0] >= specification["raw_min"]) & (x[0] <= specification["raw_max"]))
    )
    combination = "|".join(np.array(names)[x[0] > 0])
    calibration = specification.get(
        "stability_calibration", {"slope": 1.0, "intercept": 0.0}
    )
    return {
        "schema_version": "observed-formulation-outcomes/v76",
        "checkpoint_sha256": core.sha256,
        "source_doi": specification["source_doi"],
        "reference_protocol": protocol,
        "stability_probability": float(
            sigmoid(calibration["slope"] * result[0] + calibration["intercept"])
        ),
        "stability_calibration": calibration,
        "estimated_turbidity_ntu": float(np.expm1(np.clip(values[1], 0, 30))),
        "estimated_viscosity_mpa_s": np.exp(np.clip(values[2:], -20, 30)).tolist(),
        "shear_rates_s_inverse": specification["shear_rates_s_inverse"],
        "within_training_coordinate_box": within,
        "prediction_status": "reference_domain_estimate"
        if within
        else "outside_observed_component_ranges",
        "out_of_range_probability_calibration_claimed": False,
        "ingredient_combination_seen_in_training": combination
        in specification["training_ingredient_combinations"],
        "scope": "short_term_rinse_off_reference_protocol_not_generic_lotion_or_skin_safety",
        "manufacturing_approved": False,
        "human_scent_similarity_percent": None,
    }


def predict_pair(core, first, second):
    from .formulation_core import sigmoid, SYSTEM_VERSION

    if core.version != SYSTEM_VERSION:
        raise ValueError("pair annotation head requires trained V76 checkpoint")
    x = core.features([first, second])
    context = np.zeros((1, 64), np.float32)
    context[0, 3] = 1.0
    context[0, 12] = -1.0
    result = core.forward(x[None], np.ones((1, 2)), context)["blend"][0]
    labels = {}
    metadata = {}
    for label, score in zip(core.manifest["blend_labels"], sigmoid(result).tolist()):
        if not label.strip():
            metadata["missing_note_field"] = score
        elif label == "No odor group found for these":
            metadata["unclassified_odor_group"] = score
        else:
            labels[label] = score
    return {
        "schema_version": "pair-annotation-prediction/v76",
        "checkpoint_sha256": core.sha256,
        "labels": labels,
        "non_odor_annotation_metadata_scores": metadata,
        "odor_annotation_label_count": len(labels),
        "annotation_missingness_is_not_odorlessness": True,
        "global_unseen_molecule_validation_claimed": False,
        "composition_ratio_known": False,
        "set_pooling_is_physical_50_50_recipe": False,
        "scope": "published_pair_annotation_likelihood_not_calibrated_mixture_intensity_or_similarity",
    }
