from copy import deepcopy
from datetime import date

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.regulatory_status import regulatory_summary


def payload():
    return {"brief": {"constraints": {"target_region": "EU", "product_category": "eau_de_parfum",
            "product_concentration_percent": 15}}, "recipe": [{"ingredient_id": "x"}], "closest_candidate": [],
            "safety": {"internal_gate_passed": True, "violations": [], "missing_documents": [],
                "ifra_screen": {"compliant": True, "details": [], "coverage": {"embedded_amendment": "50",
                    "is_complete_ifra_rule_pack": False, "uncovered_ingredients": ["x"]}}}}


def test_all_five_tabs_report_missing_evidence_without_fabricated_compliance():
    item = payload()
    before = deepcopy(item)
    result = regulatory_summary(item)
    assert [tab["id"] for tab in result["tabs"]] == ["IFRA", "EU_REACH", "K_REACH", "FDA", "REGULATORY_STATUS"]
    assert result["tabs"][0]["status"] == "partial_screen_only"
    assert [tab["status"] for tab in result["tabs"][1:4]] == ["not_assessed"] * 3
    assert result["status"] == "review_required"
    assert not result["tabs"][-1]["commercial_release_authorized_by_this_summary"]
    result["tabs"][0]["coverage"]["uncovered_ingredients"].append("new")
    assert item == before


@pytest.mark.parametrize("subject", ["returned_recipe", "closest_candidate_only", "no_formula"])
def test_unapproved_candidates_and_missing_formula_are_not_presented_as_approved(subject):
    item = payload()
    if subject == "closest_candidate_only":
        item["closest_candidate"] = item.pop("recipe")
    elif subject == "no_formula":
        item["recipe"] = []
    result = regulatory_summary(item)
    assert result["subject"] == subject
    assert result["status"] == ("not_assessed" if subject == "no_formula" else "review_required")
    if subject == "no_formula":
        assert result["tabs"][0]["status"] == "not_assessed"


def test_ifra_violations_propagate_but_do_not_fabricate_reach_or_fda_findings():
    item = payload()
    item["safety"]["ifra_screen"].update(compliant=False, details=[{"ingredient": "x", "reason": "above_limit"}])
    result = regulatory_summary(item)
    assert result["status"] == result["tabs"][0]["status"] == "blocked"
    assert result["tabs"][0]["findings"] == [{"ingredient": "x", "reason": "above_limit"}]
    assert all(row["findings"] == [] for row in result["tabs"][1:4])


def test_actual_recipe_serialization_connects_screen_without_changing_score():
    with NaturalLanguagePerfumeryAI(require_full_profile_match=True) as ai:
        result = ai.create_recipe("clean scent", RecipeConstraints(max_ingredients=12,
            simulation_draws=64, physics_search_population=1, target_similarity=95), as_of=date(2026, 9, 5))
    before = result.calculated_profile_similarity
    data = result.to_dict()
    assert data["calculated_profile_similarity"] == before
    assert data["regulatory"]["tabs"][0]["coverage"] == data["safety"]["ifra_screen"].get("coverage", {})
    assert len(data["regulatory"]["tabs"]) == 5
    assert data["regulatory"]["live_regulatory_lookup_performed"] is False
