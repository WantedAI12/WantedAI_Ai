from datetime import date
import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.intent_controls import apply_intent_controls
from tests.test_dose_refinement import materials


def test_partial_phase_edit_preserves_other_phases_and_constraints():
    parser = NaturalLanguageBriefParser(IngredientCatalog(materials(), {}))
    base = parser.parse("opening citrus, drydown woody", RecipeConstraints(max_formula_cost_per_kg=99))
    result = apply_intent_controls(base, {"phase_target_profiles": {"heart": {"floral": 2}}, "intensity_level": 4})
    assert result.phase_target_profiles["opening"] == base.phase_target_profiles["opening"]
    assert result.phase_target_profiles["drydown"] == base.phase_target_profiles["drydown"]
    assert result.phase_target_profiles["heart"]["floral"] == 1
    assert "heart" not in base.phase_target_profiles
    assert result.absolute_intensity_target == .75 and result.constraints.max_formula_cost_per_kg == 99


@pytest.mark.parametrize("controls", [{"intensity_level": True}, {"intensity_level": 6},
    {"phase_target_profiles": {"unknown": {"woody": 1}}}, {"phase_target_profiles": {"heart": {"woody": 0}}},
    {"phase_target_profiles": {"opening": {"gourmand": 1}}}])
def test_invalid_or_avoidance_conflicting_controls_are_rejected(controls):
    parser = NaturalLanguageBriefParser(IngredientCatalog(materials(), {}))
    base = parser.parse("opening no sweetness, drydown woody", RecipeConstraints())
    with pytest.raises(ValueError):
        apply_intent_controls(base, controls)


def test_real_engine_uses_phase_controls_in_simulation():
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials(), {}), require_full_profile_match=True) as ai:
        result = ai.create_recipe("citrus floral woody", RecipeConstraints(max_ingredients=12, simulation_draws=64),
            as_of=date(2026, 9, 5), intent_controls={"phase_target_profiles": {"heart": {"floral": 1}}, "intensity_level": 4})
    assert result.brief.absolute_intensity_target == .75
    assert result.brief.phase_target_profiles["heart"]["floral"] == 1
    assert any(point["phase"] == "heart" and point["target_profile"]["floral"] == 1 for point in result.temporal_profile)
