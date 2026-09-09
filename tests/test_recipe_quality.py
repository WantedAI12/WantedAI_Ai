"""User-intent and concentration-response quality regression cases."""

from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import (
    HistoricalReferenceCorpus,
    IngredientCatalog,
)
from fragrance_ai.recommender.models import RecipeLine
from fragrance_ai.recommender.physsim import ConcentrationAwarePhysSim
from fragrance_ai.recommender.science import (
    ATMOSPHERIC_PRESSURE_PA,
    ScientificPropertyStore,
    TemporalMixtureSimulator,
)
from tests.test_inference_efficiency import _properties


CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "recipe_quality_cases_20260905.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES["parser_cases"], ids=lambda case: case["id"])
def test_request_intent_contract(case):
    brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(
        case["brief"]
    )
    assert set(case.get("desired", [])).issubset(brief.desired_dimensions)
    assert set(case.get("avoided", [])).issubset(brief.avoided_dimensions)
    assert set(case.get("excluded", [])).issubset(brief.excluded_ingredients)
    assert not set(case.get("not_avoided", [])) & set(brief.avoided_dimensions)
    if "max_ingredients" in case:
        assert brief.constraints.max_ingredients == case["max_ingredients"]
    if "weaker_than" in case:
        weak, strong = case["weaker_than"]
        assert 0 < brief.target_profile[weak] < brief.target_profile[strong]


@pytest.mark.parametrize("case", CASES["recipe_cases"], ids=lambda case: case["id"])
def test_recipe_obeys_material_exclusions_and_count(case, tmp_path):
    corpus = HistoricalReferenceCorpus(tmp_path / "unpackaged-reference.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        result = ai.create_recipe(
            case["brief"],
            RecipeConstraints(
                target_similarity=50, simulation_draws=64, physics_search_population=7
            ),
            as_of=date(2026, 9, 5),
        )
    assert result.recipe
    assert len(result.recipe) <= case.get(
        "max_lines", result.brief.constraints.max_ingredients
    )
    assert not set(case.get("forbidden", [])) & {
        line.name for line in result.closest_candidate
    }
    assert sum(line.concentrate_percent for line in result.recipe) == pytest.approx(
        100, abs=0.001
    )


def test_temporal_contributions_use_current_concentration():
    catalog = IngredientCatalog.load_builtin()
    ingredients = {item.ingredient_id: item for item in catalog.ingredients}
    engine = TemporalMixtureSimulator()
    lines = [
        RecipeLine(
            ingredient_id=identifier,
            name=ingredients[identifier].name,
            pyramid=ingredients[identifier].pyramid,
            concentrate_percent=percent,
            finished_product_percent=percent * 0.15,
            volume_ml_for_batch=None,
            price_per_kg=10,
            availability=1,
            risk_tier=1,
            reason="test",
        )
        for identifier, percent in [("dihydromyrcenol", 60), ("vanillin", 40)]
    ]
    with ScientificPropertyStore() as store:
        store.upsert(_properties("dihydromyrcenol", pressure=20))
        store.upsert(
            replace(_properties("vanillin", pressure=0.01), odor_threshold_ppm=0.0001)
        )
        prepared = engine._prepare(lines, ingredients, store)
        interaction = engine._interaction_matrix(prepared)
        curves = engine._build_ingredient_temporal_profiles(prepared, interaction)
        for time_index, minutes in enumerate([0, 15, 60, 240, 480]):
            responses = []
            log_weights = []
            for item in prepared:
                gas = (
                    item.mole_fraction
                    * item.activity_coefficient
                    * item.vapor_pressure_pa
                    / ATMOSPHERIC_PRESSURE_PA
                    * 1e6
                )
                remaining = 0.5 ** (
                    minutes
                    / engine._half_life_minutes(
                        item.ingredient, item.properties, item.vapor_pressure_pa
                    )
                )
                oav = gas * remaining / item.odor_threshold_ppm
                transport = engine._air_to_receptor_transport(item.properties)
                responses.append(oav**0.55 / (1 + oav**0.55) * transport)
                log_weights.append(
                    np.log1p(oav * transport) * max(0.05, item.ingredient.odor_impact)
                )
            responses = np.asarray(responses)
            suppressed = responses / (1 + 0.2 * (interaction @ responses))
            expected = 100 * suppressed / suppressed.sum()
            assert [
                curve.points[time_index].odor_contribution_percent for curve in curves
            ] == pytest.approx(expected, abs=1e-6)
            field = ConcentrationAwarePhysSim()._particle_field(
                lines, ingredients, store, minutes
            )
            assert field.pooling_weights == pytest.approx(
                np.asarray(log_weights) / sum(log_weights), abs=1e-12
            )


def test_upper_bound_never_increases_and_request_object_is_unchanged():
    constraints = RecipeConstraints(max_ingredients=6)
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    brief = parser.parse("clean woody, maximum 20 ingredients", constraints)
    assert brief.constraints.max_ingredients == 6
    assert constraints.max_ingredients == 6


@pytest.mark.parametrize("brief, retained", [
    ("rose and vanilla-free clean woody scent", "rose"),
    ("vanillin and musk-free woody fragrance", "Vanillin"),
])
def test_hyphenated_free_only_excludes_its_own_name(brief, retained):
    parsed = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(brief)
    if retained == "rose":
        assert "rose" in parsed.desired_dimensions
        assert "rose" not in parsed.avoided_dimensions
        assert "gourmand" in parsed.avoided_dimensions
    else:
        assert "Vanillin" in parsed.requested_ingredients
        assert "Vanillin" not in parsed.excluded_ingredients
        assert "musky" in parsed.avoided_dimensions


def test_monte_carlo_keeps_response_saturated_while_oav_is_high():
    catalog = IngredientCatalog.load_builtin()
    ingredient = catalog.lookup("Dihydromyrcenol")
    line = RecipeLine(
        ingredient.ingredient_id,
        ingredient.name,
        ingredient.pyramid,
        100,
        15,
        None,
        10,
        1,
        1,
        "response saturation regression",
    )
    brief = NaturalLanguageBriefParser(catalog).parse("fresh citrus")
    with ScientificPropertyStore() as store:
        store.upsert(
            replace(
                _properties(ingredient.ingredient_id, pressure=20),
                odor_threshold_ppm=1e-9,
            )
        )
        result = TemporalMixtureSimulator().evaluate(
            [line],
            {ingredient.ingredient_id: ingredient},
            brief,
            store,
            draws=64,
            seed=17,
        )
    residue = result.ingredient_temporal_profiles[0].points[-1]
    assert residue.application_surface_remaining_fraction_percent < 1
    assert result.temporal_points[-1].relative_to_opening_intensity_percent > 99
