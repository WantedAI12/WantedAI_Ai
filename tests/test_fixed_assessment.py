from datetime import date
import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.fixed_assessment import assess_fixed_formula
from tests.test_dose_refinement import materials


def test_fixed_weights_are_recomputed_not_optimized_or_approved():
    weights = {"top": 25., "heart": 40., "base": 35.}
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials(), {}), require_full_profile_match=True) as ai:
        result = assess_fixed_formula(ai, "citrus floral woody", RecipeConstraints(), weights, as_of=date(2026, 9, 5))
    assert {line.ingredient_id: line.concentrate_percent for line in result.closest_candidate} == weights
    assert result.calculated_profile_similarity is not None
    assert len(result.temporal_profile) == 5 and result.scientific_monte_carlo_draws == 200
    assert result.temporal_similarity_score > 0 and result.minimum_temporal_similarity > 0
    assert result.score_contract["assessment_kind"] == "fixed_formula_scientific_reassessment"
    assert not result.score_contract["full_generator_approval"] and not result.recipe
    assert result.safety.internal_gate_passed


@pytest.mark.parametrize("weights,brief", [
    ({"top": 25., "heart": 40., "base": 34.}, "citrus floral woody"),
    ({"unknown": 100.}, "woody"),
    ({"top": float("nan"), "base": 100.}, "woody"),
    ({"top": 25., "heart": 40., "base": 35.}, "citrus floral woody base 60%"),
])
def test_invalid_fixed_inputs_are_rejected_without_normalization(weights, brief):
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials(), {}), require_full_profile_match=True) as ai:
        with pytest.raises(ValueError):
            assess_fixed_formula(ai, brief, RecipeConstraints(), weights, as_of=date(2026, 9, 5))


def test_excluded_material_is_not_reintroduced_by_manual_weights():
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials(), {}), require_full_profile_match=True) as ai:
        with pytest.raises(ValueError, match="ineligible"):
            assess_fixed_formula(ai, "woody", RecipeConstraints(explicit_bans={"top"}),
                                 {"top": 25., "heart": 40., "base": 35.}, as_of=date(2026, 9, 5))


@pytest.mark.parametrize('weights', [
    {'top': 25.00001, 'heart': 39.99999, 'base': 35.},
    {'top': 1e-7, 'heart': 40., 'base': 59.9999999},
])
def test_generated_precision_is_preserved_and_target_is_not_silently_raised(weights):
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials(), {}),
            minimum_profile_target=90., require_full_profile_match=True) as ai:
        result = assess_fixed_formula(ai, 'citrus floral woody',
            RecipeConstraints(target_similarity=90.), weights, as_of=date(2026, 9, 5))
    assert {line.ingredient_id: line.concentrate_percent for line in result.closest_candidate} == weights
    assert result.brief.constraints.target_similarity == 90.
