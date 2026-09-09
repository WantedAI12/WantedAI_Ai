"""Numerical/policy counterexamples plus actual adaptive integration tests."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from fragrance_ai.recommender.adaptive_pyramid import (
    apply_explicit_pyramid, blended_proposals, check_adaptive_response, explicit_pyramid, prepare_adaptive_policy,
)
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog, HistoricalReferenceCorpus
from fragrance_ai import NaturalLanguagePerfumeryAI
from fragrance_ai.recommender.global_profile_search import optimize_full_pool
from fragrance_ai.recommender.models import Ingredient, RecipeConstraints
from fragrance_ai.recommender.profile_match import assess_recipe_profiles


def parse(text="woody"):
    return NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(text)


def material(name, group, profile=None, cap=100., price=10., impact=1.):
    return Ingredient(name, name, (), None, group, profile or {"woody": 1.}, price, 1., "common", 0, impact, cap, True)


def test_explicit_full_and_partial_ratios_are_not_rescaled():
    full = parse("citrus top 60%, heart 25%, base 15%")
    assert full.pyramid_ratios == {"top": 60, "heart": 25, "base": 15}
    assert full.desired_dimensions == ["citrus"]
    assert full.phase_target_profiles == {}
    assert full.temporal_emphasis == parse("citrus").temporal_emphasis
    korean = parse("시트러스 탑/미들/베이스 60/25/15")
    assert korean.pyramid_ratios == full.pyramid_ratios
    partial = parse("woody base 60%")
    assert partial.pyramid_ratios["base"] == 60
    assert sum(partial.pyramid_ratios.values()) == pytest.approx(100)
    assert partial.pyramid_ratios["top"] / partial.pyramid_ratios["heart"] == pytest.approx(25 / 40)
    assert apply_explicit_pyramid({"top": 25., "heart": 40., "base": 35.}, {}) == {"top": 25., "heart": 40., "base": 35.}


def test_percentages_do_not_erase_attached_phase_odor_requests():
    before = parse("top notes citrus, heart notes rose, base notes woody")
    after = parse("top notes 60% citrus, heart notes 25% rose, base notes 15% woody")
    assert after.pyramid_ratios == {"top": 60, "heart": 25, "base": 15}
    assert after.phase_target_profiles == before.phase_target_profiles
    assert after.phase_avoided_dimensions == before.phase_avoided_dimensions
    assert after.temporal_emphasis == before.temporal_emphasis
    negative = parse("top notes 60% citrus, heart notes 25% no sweetness, base notes 15% woody")
    assert negative.phase_avoided_dimensions["heart"] == ["gourmand"]


@pytest.mark.parametrize("text", ["top 60% heart 50%", "top 30% heart 30% base 30%", "top -1%", "top 101%", "top NaN%", "base inf%", "top 10% top 20%"])
def test_invalid_explicit_ratios_fail_closed(text):
    with pytest.raises(ValueError):
        explicit_pyramid(text)


def test_adaptive_lp_keeps_mass_caps_cost_and_group_bands():
    pool = [material("a", "top"), material("b", "heart"), material("c", "base")]
    result = optimize_full_pool(pool, parse(), pyramid_bounds={g: (10, 80) for g in ("top", "heart", "base")})
    assert result.weights_percent
    assert sum(result.weights_percent.values()) == pytest.approx(100)
    assert all(10 - 1e-6 <= value <= 80 + 1e-6 for value in result.weights_percent.values())
    expensive = [replace(item, price_per_kg=200) for item in pool]
    request = replace(parse(), constraints=RecipeConstraints(max_formula_cost_per_kg=100))
    assert not optimize_full_pool(expensive, request, pyramid_bounds={g: (10, 80) for g in ("top", "heart", "base")}).weights_percent


def test_adaptive_partial_lock_and_support_capacity_are_resolved():
    pool = [material("a", "top", cap=80), material("b", "heart", cap=10), material("c", "base", cap=10)]
    request = replace(parse(), constraints=RecipeConstraints(max_ingredients=3))
    result = optimize_full_pool(pool, request, pyramid_bounds={g: (10, 80) for g in ("top", "heart", "base")})
    assert result.weights_percent == pytest.approx({"a": 80, "b": 10, "c": 10})
    locked = optimize_full_pool([replace(item, max_concentrate_percent=100) for item in pool], request,
                               pyramid_bounds={"top": (60, 60), "heart": (10, 80), "base": (10, 80)})
    assert locked.weights_percent["a"] == pytest.approx(60)


def test_intensity_range_uses_structural_not_headspace_gain():
    pool = [material("a", "top", impact=2.5), material("b", "heart", impact=2.5), material("c", "base", impact=2.5)]
    result = optimize_full_pool(pool, parse(), factors={item.ingredient_id: .01 for item in pool},
                                pyramid_bounds={g: (10, 80) for g in ("top", "heart", "base")}, intensity_range=(0., .5))
    assert not result.weights_percent


def response(brief, profiles, signals, weights):
    points = [{"minutes": i * 480, "phase": "opening" if i == 0 else "drydown", "scent_profile": profile,
               "target_profile": {"woody": 1.}} for i, profile in enumerate(profiles)]
    assessment = assess_recipe_profiles(brief, brief.target_profile, points, weights)
    twin = SimpleNamespace(temporal_points=[SimpleNamespace(total_relative_intensity=s,
                           relative_to_opening_intensity_percent=100 * s / signals[0]) for s in signals])
    return assessment, twin


def test_higher_normalized_score_cannot_hide_vanishing_odor():
    brief = parse()
    policy = prepare_adaptive_policy(brief, [], {})
    before, old_twin = response(brief, [{"woody": .9, "floral": .1}] * 2, [1, 1], [.5, .5])
    after, new_twin = response(brief, [{"woody": 1.}] * 2, [1, 1e-6], [.5, .5])
    assert after["score"] > before["score"]
    violations = check_adaptive_response(brief, before, after, old_twin, new_twin, [], {}, policy)
    assert "temporal_odor_signal_lost" in violations
    tiny_before, tiny_old = response(brief, [{"woody": .9, "floral": .1}] * 2, [2e-6, 2e-6], [.5, .5])
    tiny_after, tiny_new = response(brief, [{"woody": 1.}] * 2, [2e-6, 1e-6], [.5, .5])
    assert "temporal_odor_signal_lost" in check_adaptive_response(brief, tiny_before, tiny_after, tiny_old, tiny_new, [], {}, policy)


def test_phase_regression_is_not_hidden_by_a_better_time_average():
    brief = replace(parse(), phase_target_profiles={"opening": {"woody": 1}, "drydown": {"woody": 1}})
    policy = prepare_adaptive_policy(brief, [], {})
    before, old_twin = response(brief, [{"woody": .9, "floral": .1}] * 2, [1, 1], [.1, .9])
    after, new_twin = response(brief, [{"woody": .2, "floral": .8}, {"woody": 1.}], [1, 1], [.1, .9])
    assert after["score"] > before["score"]
    violations = check_adaptive_response(brief, before, after, old_twin, new_twin, [], {}, policy)
    assert "explicit_phase_profile_regressed" in violations
    assert "worst_time_profile_regressed" in violations


def test_long_lasting_protects_absolute_and_relative_signal():
    brief = parse("long lasting woody")
    policy = prepare_adaptive_policy(brief, [], {})
    before, old_twin = response(brief, [{"woody": .9, "floral": .1}] * 2, [1, 1], [.5, .5])
    after, new_twin = response(brief, [{"woody": 1.}] * 2, [.95, .95], [.5, .5])
    assert "requested_persistence_regressed" in check_adaptive_response(brief, before, after, old_twin, new_twin, [], {}, policy)


def test_long_lasting_checks_final_time_even_when_its_scent_weight_is_zero():
    brief = parse("opening fresh, heart woody long lasting")
    policy = prepare_adaptive_policy(brief, [], {})
    before, old_twin = response(brief, [{"woody": .9, "floral": .1}] * 2, [1, 1], [1., 0.])
    after, new_twin = response(brief, [{"woody": 1.}] * 2, [1, 1e-6], [1., 0.])
    assert after["score"] >= before["score"]
    assert "requested_persistence_regressed" in check_adaptive_response(brief, before, after, old_twin, new_twin, [], {}, policy)
    after, new_twin = response(brief, [{"woody": 1.}] * 2, [2., 1.], [1., 0.])
    assert "requested_persistence_regressed" in check_adaptive_response(brief, before, after, old_twin, new_twin, [], {}, policy)


def test_blending_keeps_mass_without_trimming_a_count_violation():
    baseline = [SimpleNamespace(ingredient_id="a", concentrate_percent=60), SimpleNamespace(ingredient_id="b", concentrate_percent=40)]
    proposed = {"a": 20., "c": 80.}
    mixes = list(blended_proposals(baseline, proposed, 3, {"a", "b", "c"}))
    assert [value for value, _ in mixes] == [.25, .5, .75, 1.]
    assert all(sum(weights.values()) == pytest.approx(100) for _, weights in mixes)
    assert list(blended_proposals(baseline, proposed, 2, {"a", "b", "c"})) == [(1., proposed)]


def test_intensity_improvement_cannot_hide_diffusion_regression():
    brief = parse("woody low projection intensity 30%")
    ingredients = {"old": material("old", "base", impact=.25), "new": material("new", "heart", impact=.75)}
    old_lines = [SimpleNamespace(ingredient_id="old", concentrate_percent=100., pyramid="base")]
    new_lines = [SimpleNamespace(ingredient_id="new", concentrate_percent=100., pyramid="heart")]
    policy = prepare_adaptive_policy(brief, old_lines, ingredients)
    before, old_twin = response(brief, [{"woody": .9, "floral": .1}] * 2, [1, 1], [.5, .5])
    after, new_twin = response(brief, [{"woody": 1.}] * 2, [1, 1], [.5, .5])
    assert .7 * abs(.3 - .3) + .3 * abs(.55 - .25) < .7 * abs(.1 - .3) + .3 * abs(.2 - .25)
    assert "diffusion_request_regressed" in check_adaptive_response(brief, before, after, old_twin, new_twin, new_lines, ingredients, policy)


@pytest.mark.parametrize("text", ["clean fresh citrus woody", "fresh green aquatic", "woody base 60%"])
def test_actual_adaptive_generation_retains_v8_score_and_explicit_requirements(tmp_path, text):
    results = []
    for enabled in (False, True):
        with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db"), enable_adaptive_pyramid=enabled) as ai:
            results.append(ai.create_recipe(text, as_of=date(2026, 9, 5)))
    before, after = results
    info = after.full_profile_assessment["search"]["full_pool_search"]["adaptive_pyramid"]
    assert before.brief.target_profile == after.brief.target_profile
    assert before.brief.phase_target_profiles == after.brief.phase_target_profiles
    assert after.calculated_profile_similarity + 1e-8 >= before.calculated_profile_similarity
    assert info["baseline_v8_score"] == pytest.approx(before.calculated_profile_similarity)
    assert after.brief.constraints == before.brief.constraints
    assert not after.human_similarity_90_claim_authorized
    for group, value in explicit_pyramid(text)[0].items():
        assert after.brief.pyramid_ratios[group] == pytest.approx(value, abs=.001)
    if info["candidate_changed"]:
        assert after.calculated_profile_similarity > before.calculated_profile_similarity
        for group, value in after.brief.pyramid_ratios.items():
            assert value == pytest.approx(sum(line.concentrate_percent for line in after.closest_candidate if line.pyramid == group), abs=.001)


def test_qualified_and_reference_requests_do_not_auto_change_pyramid(tmp_path):
    with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("clean citrus", RecipeConstraints(reference_target_id="unavailable"), as_of=date(2026, 9, 5))
    assert result.recipe == []
    assert result.calculated_profile_similarity is None
    with pytest.raises(ValueError, match="boolean"):
        NaturalLanguagePerfumeryAI(enable_adaptive_pyramid="yes")


def test_explicit_zero_group_needs_no_material_from_that_group(tmp_path):
    source = IngredientCatalog.load_builtin()
    catalog = IngredientCatalog([item for item in source.ingredients if item.pyramid != "top"], source.metadata)
    with NaturalLanguagePerfumeryAI(catalog=catalog, corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("woody top 0% heart 50% base 50%", as_of=date(2026, 9, 5))
    assert result.brief.pyramid_ratios == {"top": 0, "heart": 50, "base": 50}
    assert result.closest_candidate
    assert all(line.pyramid != "top" for line in result.closest_candidate)
    assert sum(line.concentrate_percent for line in result.closest_candidate) == pytest.approx(100, abs=.001)
