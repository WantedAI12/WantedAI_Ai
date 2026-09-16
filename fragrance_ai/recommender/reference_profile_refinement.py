"""Use the shared model's full component profiles as additional perfume seeds.

Reference targets are frozen from source observations, never fitted to the
candidate catalogue. Legacy requested targets and final scores are unchanged.
"""

import numpy as np

from .autoregressive_refinement import neural_mass_seeds
from .aligned_autoregressive import SCHEMA
from .science import TIMEPOINTS_MINUTES


def reference_profile_seeds(
    items, brief, transport, original, minimums, guidance, *, bank=None
):
    core = getattr(getattr(guidance, "provider", None), "core", None)
    if core is None or getattr(core, "version", None) not in (SCHEMA,'shared-formulation-core/v76'):
        return [], {"status": "aligned_shared_model_not_selected"}
    if bank is None:
        from .lotion_reference_objective import load_configured_reference_bank

        bank = load_configured_reference_bank()
    if bank is None or bank.parent_sha256 != core.sha256:
        return [], {"status": "matching_source_reference_unavailable"}
    rows = [
        {
            "phase": "overall",
            "target_profile": brief.target_profile,
            "avoided": brief.avoided_dimensions,
        }
    ]
    for minutes, (target, _, avoided) in zip(
        TIMEPOINTS_MINUTES, transport.engine.targets_by_time(brief)
    ):
        from .models import SCENT_DIMENSIONS

        rows.append(
            {
                "phase": transport.engine._phase_for_time(minutes),
                "target_profile": dict(zip(SCENT_DIMENSIONS, target)),
                "avoided": avoided,
            }
        )
    targets, unsupported = bank.targets(brief, rows)
    from .odor_space import target_coverage
    coverage = target_coverage(targets, unsupported)
    if (not coverage['searchable'] or
            (getattr(bank, 'odor_space', None) is None and unsupported)):
        return [], {
            "status": "unresolved_source_reference_intent",
            "unsupported": unsupported,
        }
    guidance.shapes.prefetch(items)
    profiles = [guidance.shapes.shape(item) for item in items]
    covered = [i for i, p in enumerate(profiles) if p is not None]
    missing = [item.ingredient_id for item, p in zip(items, profiles) if p is None]
    if any(original.get(key, 0) > 0 or minimums.get(key, 0) > 0 for key in missing):
        return [], {
            "status": "incumbent_or_required_reference_profile_missing",
            "missing": missing,
        }
    if not covered:
        return [], {"status": "no_quantitative_component_profiles"}
    selected = [items[i] for i in covered]
    p = np.stack([profiles[i] for i in covered], axis=1)
    q = np.stack([t["profiles"] for t in targets], axis=1)
    avoid = np.array(
        [[float(name in t["avoided"]) for name in bank.endpoints] for t in targets]
    )
    r = np.vstack((transport.gain, transport.coefficient.T))[:, covered]
    r /= r.max(-1, keepdims=True)
    w = np.array([original.get(item.ingredient_id, 0) / 100 for item in selected])
    learned = core.autoregressive_proposal(
        selected,
        p,
        q,
        r,
        w,
        product="perfume",
        minimum_fractions=np.array(
            [minimums.get(item.ingredient_id, 0) / 100 for item in selected]
        ),
        maximum_fractions=np.array(
            [min(1.0, item.as_supplied_cap_percent() / 100) for item in selected]
        ),
        prices=np.array([item.price_per_kg for item in selected]),
        price_budget=brief.constraints.max_formula_cost_per_kg,
        time_weights=np.r_[0.0, transport.engine.time_weights(brief)],
        avoided=np.broadcast_to(avoid[None], q.shape),
        **({'physics_properties':transport.properties,'concentration_percent':brief.constraints.product_concentration_percent}
           if core.version=='shared-formulation-core/v76' else {}),
    )
    weights, report = learned
    seeds, projection = neural_mass_seeds(
        selected, weights, original, minimums, brief.constraints.max_ingredients
    )
    names = {
        "trusted_neural_feedback": "reference_neural_feedback",
        "trained_autoregressive_decoder": "reference_autoregressive_decoder",
    }
    return [(names[name], value) for name, value in seeds], {
        **report,
        **projection,
        "status": "source_anchored_full_component_profile_seeds",
        "source_reference_sha256": bank.sha256,
        "descriptor_count": len(bank.endpoints),
        "reference_heads": p.shape[0],
        "known_component_profiles": len(covered),
        "missing_component_profiles": missing,
        "target_fit_to_candidate_catalogue": False,
        "legacy_requested_target_and_score_unchanged": True,
        "source_scope": "component_ordinal_reference_not_measured_perfume_target",
        "source_overlap_with_existing_reference_tests": True,
    }
