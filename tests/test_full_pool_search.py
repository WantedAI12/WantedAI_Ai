"""Full-pool optimization and feasibility counterexamples; no sensory claims."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus
from fragrance_ai import NaturalLanguagePerfumeryAI
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.global_profile_search import optimize_full_pool, profile_upper_bound
from fragrance_ai.recommender.models import Ingredient, RecipeConstraints, SCENT_DIMENSIONS
from fragrance_ai.recommender.profile_match import compare_profiles


def material(identifier, group="heart", *, profile=None, cap=100., price=10.):
    return Ingredient(identifier, identifier, (), None, group, profile or {"woody": 1.}, price, 1., "common", 0, 1., cap, True)


def brief(constraints=None):
    value = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse("woody", constraints)
    return replace(value, pyramid_ratios={"top": 25., "heart": 40., "base": 35.})


def test_fractional_lp_conserves_percent_units_and_pyramid():
    pool = [material("a", "top"), material("b", "heart"), material("c", "base")]
    result = optimize_full_pool(pool, brief())
    assert result.status == "relaxed_profile_optimum"
    assert result.relaxed_overlap_score == pytest.approx(100)
    assert result.weights_percent == pytest.approx({"a": 25, "b": 40, "c": 35})


def test_price_constraint_is_not_accidentally_scaled_by_100():
    pool = [material("a", "top", price=200), material("b", "heart", price=200), material("c", "base", price=200)]
    result = optimize_full_pool(pool, brief(RecipeConstraints(max_formula_cost_per_kg=100)))
    assert not result.weights_percent
    assert result.status == "infeasible_constraints"


def test_full_pool_selection_can_choose_material_omitted_by_an_initial_subset():
    pool = [material("a", "top"), material("b", "heart", profile={"floral": 1}), material("c", "base"), material("better", "heart")]
    result = optimize_full_pool(pool, brief())
    assert result.weights_percent.get("b", 0) == 0
    assert result.weights_percent["better"] == pytest.approx(40)


def test_material_count_can_make_a_100_point_relaxation_infeasible():
    pool = [material(f"{group}-{i}", group, cap=10) for group, count in [("top", 3), ("heart", 4), ("base", 4)] for i in range(count)]
    result = optimize_full_pool(pool, brief(RecipeConstraints(max_ingredients=6)))
    assert not result.weights_percent
    assert result.relaxed_overlap_score == pytest.approx(100)
    assert result.status == "cardinality_support_search_exhausted"


def test_capacity_efficient_alternative_is_not_mistaken_for_impossibility():
    pool = [material(f"{group}-{i}", group, cap=10) for group, count in [("top", 3), ("heart", 4), ("base", 4)] for i in range(count)]
    pool += [material(f"fallback-{group}", group, profile={"woody": .9, "floral": .1}) for group in ("top", "heart", "base")]
    result = optimize_full_pool(pool, brief(RecipeConstraints(max_ingredients=6)))
    assert result.weights_percent and len(result.weights_percent) <= 6
    assert result.relaxed_overlap_score >= 90 - 1e-7
    assert sum(result.weights_percent.values()) == pytest.approx(100)
    assert result.restricted_support


def test_required_materials_and_active_strength_are_preserved():
    pool = [material("a", "top"), material("b", "heart", profile={"floral": 1}), material("c", "base"), material("better", "heart")]
    required = optimize_full_pool(pool, brief(), minimum_percent={"b": 20})
    assert required.weights_percent["b"] >= 20 - 1e-7
    assert not optimize_full_pool(pool, brief(), minimum_percent={"missing": 10}).weights_percent
    dilute = [replace(item, active_strength_percent=10) for item in pool]
    dilution = optimize_full_pool(dilute, brief(), minimum_percent={"b": 20})
    assert dilution.relaxed_overlap_score == pytest.approx(required.relaxed_overlap_score)


def test_conservative_bound_includes_legacy_small_axis_removal():
    rng = np.random.default_rng(71)
    pool = [material(str(i), profile=dict(zip(SCENT_DIMENSIONS, rng.random(19) ** 4))) for i in range(12)]
    for _ in range(50):
        target = dict(zip(SCENT_DIMENSIONS, (rng.random(19) > .7).astype(float)))
        if not any(target.values()):
            continue
        weights = rng.random(len(pool))
        profile = weights @ np.asarray([item.vector() for item in pool])
        profile /= profile.sum()
        rendered = {d: round(float(v), 6) for d, v in zip(SCENT_DIMENSIONS, profile) if v >= .001}
        result = compare_profiles(target, rendered)
        bound = profile_upper_bound(pool, target)
        assert result.score <= bound["upper_score"] + 1e-8
        assert not bound["actual_human_accuracy_bound"]
        assert not bound["upper_bound_is_achievable_score"]


def test_current_woody_catalog_cannot_claim_universal_90():
    catalog = IngredientCatalog.load_builtin()
    ready = [item for item in catalog.ingredients if item.formulation_ready and not item.blocked]
    bound = profile_upper_bound(ready, {"woody": 1})
    assert bound["upper_score"] < 90
    assert bound["unrounded_profile_upper_score"] < bound["upper_score"]


def test_empty_target_or_pool_has_no_invented_success():
    assert profile_upper_bound([], {"woody": 1})["upper_score"] is None
    assert not optimize_full_pool([], brief()).weights_percent
    assert not optimize_full_pool([material("a")], brief(), target={}).weights_percent


@pytest.mark.parametrize("text", ["clean fresh citrus woody", "floral fruity woody", "soft clean musk"])
def test_actual_whole_pool_search_preserves_v7_and_all_recipe_constraints(tmp_path, text):
    results = []
    for enabled in (False, True):
        with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db"), enable_full_pool_search=enabled) as ai:
            results.append(ai.create_recipe(text, as_of=date(2026, 9, 5)))
    before, after = results
    pool = after.full_profile_assessment["search"]["full_pool_search"]
    assert after.calculated_profile_similarity + 1e-8 >= before.calculated_profile_similarity
    assert pool["baseline_v7_score"] == pytest.approx(before.calculated_profile_similarity)
    assert pool["selected_score"] == pytest.approx(after.calculated_profile_similarity)
    assert pool["candidate_count"] == 29
    assert after.safety.internal_gate_passed
    assert after.brief.constraints.target_similarity == 90
    assert not after.human_similarity_90_claim_authorized
    assert len(after.recipe) <= after.brief.constraints.max_ingredients
    assert sum(line.concentrate_percent for line in after.recipe) == pytest.approx(100, abs=.001)
    assert after.estimated_concentrate_cost_per_kg <= after.brief.constraints.max_formula_cost_per_kg
    if text == "clean fresh citrus woody":
        assert after.calculated_profile_similarity > before.calculated_profile_similarity + 1


def test_unsupported_complete_phase_and_explicit_control_are_not_hardcoded_success(tmp_path):
    with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("opening no sweetness, drydown woody musk", as_of=date(2026, 9, 5))
    assert result.calculated_profile_similarity is None
    assert not result.full_profile_target_met
    assert result.full_profile_assessment["search"]["full_pool_search"]["status"] == "undefined_complete_target"
    with pytest.raises(ValueError, match="boolean"):
        NaturalLanguagePerfumeryAI(enable_full_pool_search="true")


def test_large_pool_uses_sparse_dual_simplex_without_dense_presolve(monkeypatch):
    from fragrance_ai.recommender import global_profile_search
    original = global_profile_search.linprog
    seen = []

    def solver(*args, **kwargs):
        seen.append((kwargs["method"], kwargs["options"]["presolve"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(global_profile_search, "linprog", solver)
    pool = [material(f"{group}-{i}", group) for group in ("top", "heart", "base") for i in range(350)]
    result = optimize_full_pool(pool, brief())
    assert result.weights_percent
    assert result.relaxed_overlap_score == pytest.approx(100)
    assert seen[0] == ("highs-ds", False)


def test_already_met_target_does_not_repeat_lp_search(tmp_path, monkeypatch):
    from fragrance_ai.recommender import service
    # Synthetic identical vectors test control flow, not real recipe quality.
    original = IngredientCatalog.load_builtin()
    catalog = IngredientCatalog([replace(item, profile={"woody": 1.}) for item in original.ingredients], original.metadata)

    def unexpected(*args, **kwargs):
        pytest.fail("no LP should run after the requested full-profile target is already met")

    monkeypatch.setattr(service, "optimize_full_pool", unexpected)
    with NaturalLanguagePerfumeryAI(catalog=catalog, corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("woody", as_of=date(2026, 9, 5))
    assert result.calculated_profile_similarity == pytest.approx(100)
    assert result.full_profile_assessment["search"]["full_pool_search"]["status"] == "requested_target_already_met"
    assert not result.human_similarity_90_claim_authorized


@pytest.mark.parametrize("text,expected", [
    ("프레시 향", {"fresh"}), ("프레쉬 향", {"fresh"}), ("fresh scent", {"fresh"}),
    ("클린 향", {"clean"}), ("clean scent", {"clean"}), ("얼씨 향", {"earthy"}), ("어시 향", {"earthy"}),
    ("레더리 향", {"leathery"}), ("leathery scent", {"leathery"}), ("파우더리 향", {"powdery"}),
    ("프루티 향", {"fruity"}), ("프룻티 향", {"fruity"}),
    ("프레시 클린 향", {"fresh", "clean"}), ("fresh clean scent", {"fresh", "clean"}),
])
def test_literal_request_words_cannot_invent_other_odor_axes(text, expected):
    parsed = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(text)
    assert set(parsed.desired_dimensions) == expected
    assert set(key for key, value in parsed.target_profile.items() if value > 0) == expected


def test_semantic_residual_preserves_negation_and_real_metaphors():
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    negative = parser.parse("스모키하지 않은 프레시 향")
    assert negative.desired_dimensions == ["fresh"]
    assert negative.avoided_dimensions == ["smoky"]
    assert "clean" in parser.parse("햇빛에 말린 흰 셔츠").desired_dimensions
    assert {"woody", "smoky"}.issubset(parser.parse("charred timber").desired_dimensions)


def test_all_bilingual_single_and_pair_requests_preserve_declared_families():
    from scripts.evaluate_request_space import request_cases
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    count = 0
    paired_profiles = {}
    for case in request_cases():
        if case["group"] == "phase_or_regression":
            continue
        expected = {SCENT_DIMENSIONS[int(i)] for i in case["id"].split("-")[1:]}
        allowed = expected | ({"floral"} if expected & {"rose", "white_floral"} else set())
        parsed = parser.parse(case["brief"])
        assert expected.issubset(parsed.desired_dimensions), case["brief"]
        assert set(parsed.desired_dimensions).issubset(allowed), case["brief"]
        key = case["id"].split("-", 1)[1]
        if key in paired_profiles:
            assert parsed.target_profile == pytest.approx(paired_profiles[key]), case["brief"]
        paired_profiles[key] = parsed.target_profile
        count += 1
    assert count == 380
