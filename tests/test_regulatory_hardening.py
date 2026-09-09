from __future__ import annotations

import math

import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.models import RecipeConstraints, ScentBrief
from fragrance_ai.recommender.safety import CandidateSafetyScreen, FormulaSafetyGate
from fragrance_ai.recommender.supplier import SupplierRegistry
from fragrance_ai.rules.ifra_rules import IFRAComplianceChecker, ProductCategory


def _concentrations(recipe: dict) -> dict[str, float]:
    return {
        item["name"]: float(item["concentration"])
        for item in recipe["ingredients"]
    }


@pytest.mark.parametrize(
    "bergamot,oakmoss,musk,lavender",
    [
        (99.0, 0.0, 0.0, 1.0),
        (10.0, 0.5, 2.0, 87.5),
        (2.0, 0.1, 0.0, 97.9),
        (50.0, 20.0, 5.0, 25.0),
    ],
)
def test_ifra_limit_application_never_reintroduces_a_clipped_limit(
    bergamot: float, oakmoss: float, musk: float, lavender: float
):
    checker = IFRAComplianceChecker()
    original = {
        "ingredients": [
            {"name": "Bergamot Oil", "concentration": bergamot},
            {"name": "Oakmoss Absolute", "concentration": oakmoss},
            {"name": "Musk Xylene", "concentration": musk},
            {"name": "Lavender", "concentration": lavender},
        ]
    }
    applied = checker.apply_ifra_limits(original, ProductCategory.EAU_DE_PARFUM)
    result = checker.check_ifra_violations(applied, ProductCategory.EAU_DE_PARFUM)
    values = _concentrations(applied)

    assert result["compliant"]
    assert applied["embedded_limits_compliant"]
    assert applied["formula_complete"]
    assert not applied["ifra_compliant"]
    assert applied["ifra_application_status"] == "embedded_subset_pass_not_ifra_certified"
    assert math.isclose(
        applied["result_total_concentration"],
        applied["original_total_concentration"],
        abs_tol=1e-8,
    )
    assert values.get("Bergamot Oil", 0.0) <= 2.0 + 1e-9
    assert values.get("Oakmoss Absolute", 0.0) <= 0.1 + 1e-9
    assert "Musk Xylene" not in values
    assert "Lavender" in applied["ifra_coverage"]["uncovered_ingredients"]
    # Caller input must remain immutable.
    assert original["ingredients"][0]["concentration"] == bergamot


def test_ifra_limit_application_reports_incomplete_formula_when_nothing_can_absorb_deficit():
    checker = IFRAComplianceChecker()
    applied = checker.apply_ifra_limits(
        {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 99.0},
                {"name": "Oakmoss Absolute", "concentration": 1.0},
            ]
        },
        ProductCategory.EAU_DE_PARFUM,
    )
    result = checker.check_ifra_violations(applied, ProductCategory.EAU_DE_PARFUM)
    assert result["compliant"]
    assert not applied["formula_complete"]
    assert not applied["ifra_compliant"]
    assert applied["unallocated_concentration"] > 0.0
    assert applied["ifra_application_status"] == "incomplete_or_embedded_limit_failure"


def test_ifra_subset_is_explicitly_partial_and_unknown_is_not_an_unrestricted_claim():
    result = IFRAComplianceChecker().check_ifra_violations(
        {"ingredients": [{"name": "Unlisted Material", "concentration": 10.0}]},
        ProductCategory.EAU_DE_PARFUM,
    )
    coverage = result["coverage"]
    assert result["compliant"]  # No local violation is not full IFRA approval.
    assert coverage["is_complete_ifra_rule_pack"] is False
    assert coverage["embedded_amendment"] == 50
    assert coverage["uncovered_ingredients"] == ["Unlisted Material"]
    assert "partial" in coverage["coverage_status"]


def test_qualified_and_commercial_candidate_screens_still_require_supplier_evidence():
    catalog = IngredientCatalog.load_builtin()
    registry = SupplierRegistry()
    screen = CandidateSafetyScreen()
    for level in ("qualified", "commercial"):
        brief = ScentBrief(
            original_text="test",
            target_profile={},
            desired_dimensions=[],
            avoided_dimensions=[],
            requested_ingredients=[],
            excluded_ingredients=[],
            intensity="balanced",
            pyramid_ratios={"top": 0.3, "heart": 0.4, "base": 0.3},
            constraints=RecipeConstraints(validation_level=level),
        )
        accepted, rejected = screen.screen(catalog, brief, supplier_registry=registry)
        assert accepted == []
        assert rejected["missing_supplier_offer"] > 0

    report = FormulaSafetyGate(registry).evaluate(
        [], {}, RecipeConstraints(validation_level="commercial")
    )
    assert report.status == "blocked"
    assert not report.manufacturing_ready
