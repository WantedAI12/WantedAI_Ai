"""Cheap orchestration checks; performance evidence is the real recipe probe."""
from types import SimpleNamespace

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.models import MAX_FORMULA_INGREDIENTS
from fragrance_ai.recommender import service


@pytest.mark.parametrize("limit,scores,bound,expected,selected", [
    (50000, [96], 100, [12], 96),
    (50000, [94, 92], 94, [12, 24], 94),
    (50000, [93, 96], 100, [12, 24], 96),
    (50000, [94, 92, 96], 100, [12, 24, 50000], 96),
    (50000, [94, 92, 93], 100, [12, 24, 50000], 94),
    (24, [94, 96], 100, [12, 24], 96),
    (18, [94, 92], 100, [12, 18], 94),
    (12, [94], 100, [12], 94),
    (6, [94], 100, [6], 94),
])
def test_support_portfolio_preserves_target_and_explicit_cap(monkeypatch, limit, scores, bound, expected, selected):
    ai = object.__new__(NaturalLanguagePerfumeryAI)
    ai.perception_guidance = None
    ai.catalog = IngredientCatalog.load_builtin()
    ai.parser = NaturalLanguageBriefParser(ai.catalog)
    ai.require_full_profile_match = True
    ai.minimum_profile_target = 95
    ai.temporal_simulator = SimpleNamespace(time_weights=lambda brief: {})
    calls = []

    def generate(text, constraints, as_of, **kwargs):
        calls.append(constraints.max_ingredients)
        brief = ai.parser.parse(text, constraints)
        return SimpleNamespace(brief=brief, recipe=[object()], achieved_profile=scores[len(calls)-1],
            temporal_profile={}, score_contract={},
            _full_profile_search={"full_pool_search": {"bound": {"upper_score": bound}}})

    monkeypatch.setattr(ai, "_create_recipe_impl", generate)
    monkeypatch.setattr(service, "assess_recipe_profiles", lambda brief, score, temporal, weights:
        {"score": score, "target_met": score >= brief.constraints.target_similarity})
    monkeypatch.setattr(service, "attach_profile_assessment", lambda result, assessment, strict: result)
    # This test isolates support orchestration; expression-head behavior is
    # covered by its real-data tests, not the synthetic scalar score below.
    monkeypatch.setattr('fragrance_ai.recommender.odor_expression.recipe_expression', lambda *args: {})
    request = RecipeConstraints(max_ingredients=limit, target_similarity=95)
    result = ai.create_recipe("green scent", request)
    assert calls == expected == result.score_contract["search_support_budgets"]
    assert [row["score"] for row in result.score_contract["search_support_results"]] == scores
    assert result.achieved_profile == selected
    assert result.brief.constraints.max_ingredients == request.max_ingredients == limit
    assert result.brief.constraints.target_similarity == request.target_similarity == 95


def test_default_is_automatic_and_natural_language_can_still_limit_it():
    assert RecipeConstraints().max_ingredients == MAX_FORMULA_INGREDIENTS
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    brief = parser.parse("green scent, 최대 6개 원료", RecipeConstraints())
    assert brief.constraints.max_ingredients == 6
