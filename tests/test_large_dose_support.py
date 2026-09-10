"""Large-support derivatives and incremental growth, without a recipe-size cutoff."""
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fragrance_ai.recommender import dose_refinement as dose
from fragrance_ai import NaturalLanguagePerfumeryAI
from fragrance_ai.recommender import service
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.adaptive_pyramid import prepare_adaptive_policy
from fragrance_ai.recommender.models import RecipeConstraints
from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer
from tests.test_dose_refinement import materials, setup_case


def dense_jacobians(model, weights):
    denominator = max(1e-12, model.base_moles + weights @ model.moles)
    activity = weights * model.coefficients / denominator
    current = dose.TemporalMixtureSimulator._odor_response(np.maximum(activity, 1e-12)) * model.transport
    slope = np.where(activity > 1e-12, .55 * current * (1 - current / model.transport), 0.)
    current_jac = slope[..., None] * (np.diag(1 / np.maximum(weights, 1e-12)) - model.moles[None, :] / denominator)
    suppression = 1 + model.suppression[:, None, None] * (current @ model.interaction.T)
    suppressed_jac = (current_jac / suppression[..., None]
        - (current / suppression ** 2)[..., None] * model.suppression[:, None, None, None]
        * (model.interaction @ current_jac))
    raw = (current / suppression) @ model.vectors
    total = np.maximum(raw.sum(axis=2), 1e-12)
    temporal = raw / total[..., None]
    raw_jac = model.vectors.T @ suppressed_jac
    temporal_jac = (raw_jac - temporal[..., None] * raw_jac.sum(axis=2)[:, :, None, :]) / total[..., None, None]
    return temporal_jac.mean(axis=0), suppressed_jac.sum(axis=2).mean(axis=0)


@pytest.mark.parametrize("count,concentration", [(33, 15.), (64, 100.), (128, 7.5)])
def test_large_contracted_jacobian_matches_dense_and_finite_difference(count, concentration):
    base = materials()
    items = [replace(base[i % 3], ingredient_id=f"large-{i}") for i in range(count)]
    model = dose.DoseModel(items, {}, concentration)
    # Also exercise the raw profile-mass derivative for an unmodeled row.
    model.vectors[-1] = 0
    weights = np.random.default_rng(52).dirichlet(np.ones(count))
    state = model.evaluate(weights)
    temporal, signal = dense_jacobians(model, weights)
    np.testing.assert_allclose(state.temporal_jac, temporal, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(state.signal_jac, signal, rtol=1e-10, atol=1e-10)
    for index in (0, count - 1):
        delta = np.zeros(count)
        delta[index] = 1e-7
        plus, minus = model.evaluate(weights + delta), model.evaluate(weights - delta)
        np.testing.assert_allclose(state.temporal_jac[..., index], (plus.temporal - minus.temporal) / 2e-7, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("limit,expected", [(40, 40), (50, 50), (50000, 56)])
def test_existing_large_recipe_is_refined_and_can_grow_without_dropping_materials(monkeypatch, limit, expected):
    items, brief, _, _ = setup_case(constraints=RecipeConstraints(max_ingredients=limit))
    pool = [replace(items[i % 3], ingredient_id=f"material-{i}") for i in range(80)]
    lines, *_ = ConstrainedFormulaOptimizer().variant_from_weights(pool[:40], brief, np.full(40, 2.5))
    policy = prepare_adaptive_policy(brief, lines, {item.ingredient_id: item for item in pool})
    selected_ids = []

    def refine(selected, *args, **kwargs):
        selected_ids.append({item.ingredient_id for item in selected})
        return {"weights_percent": {item.ingredient_id: 100 / len(selected) for item in selected}}

    monkeypatch.setattr(dose, "optimize_dose_support", refine)
    monkeypatch.setattr(dose, "actual_dose_replacement_seeds", lambda *args: [])
    proposals = list(dose._bounded_dose_refinement_proposals(pool, brief, {}, lines, policy, [], {}))
    assert proposals
    assert len(selected_ids[0]) == 40
    assert max(map(len, selected_ids)) == expected
    assert all(selected_ids[0].issubset(ids) for ids in selected_ids)
    assert all(row["full_pool_considered"] == 80 for row in proposals)


def test_real_service_enters_refinement_for_45_materials_without_lowering_final_gate(monkeypatch):
    base = materials()
    pool = [replace(base[i % 3], ingredient_id=f"large-{i}", name=f"large {i}",
                    max_concentrate_percent=3.) for i in range(45)]
    original = service.dose_refinement_proposals
    counts = []

    def traced(*args, **kwargs):
        counts.append(len(args[3]))
        yield from original(*args, **kwargs)

    monkeypatch.setattr(service, "dose_refinement_proposals", traced)
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(pool, {}), require_full_profile_match=True) as ai:
        result = ai.create_recipe("citrus floral woody", RecipeConstraints(max_ingredients=100,
            target_similarity=95, simulation_draws=64, physics_search_population=1), as_of=date(2026, 9, 5))
    assert 45 in counts
    assert result.brief.constraints.max_ingredients == 100
    assert result.calculated_profile_similarity < 95
    assert not result.full_profile_target_met and not result.recipe
