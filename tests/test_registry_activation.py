from datetime import date
from pathlib import Path

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry
from fragrance_ai.recommender.registry_activation import (
    REGISTRY_CONDITIONAL_CAP_PERCENT,
    REGISTRY_CONDITIONAL_DATA_SOURCE,
    activate_registry_conditionals,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REGISTRY_SHA256 = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"


def _activated_catalog():
    return activate_registry_conditionals(
        IngredientCatalog.load_builtin(),
        REGISTRY,
        expected_sha256=REGISTRY_SHA256,
    )


def test_full_registry_is_connected_with_strict_conditional_trace_boundary():
    catalog, report = _activated_catalog()

    assert report.reference_molecules_connected == 29_240
    assert report.structurally_blocked == 8_455
    assert report.evidence_pending == 20_785
    assert report.strict_conditional_rows == 850
    assert report.conditional_trace_candidates_active == 637
    assert report.blocked_known_policy == 4
    assert report.blocked_unsupported_descriptor == 209
    assert report.activation_mode == "prototype_conditional_trace_only"

    conditionals = [
        item
        for item in catalog.ingredients
        if item.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
    ]
    assert len(conditionals) == 637
    assert len(catalog.ingredients) == 684
    assert len(catalog.formulation_candidates()) == 671
    assert all(item.risk_tier == 2 for item in conditionals)
    assert all(
        item.max_concentrate_percent == REGISTRY_CONDITIONAL_CAP_PERCENT
        for item in conditionals
    )
    assert not any(item.name.casefold() == "methyl eugenol" for item in conditionals)
    assert catalog.stats()["industrial_registry_connected_total"] == 29_240
    assert catalog.stats()["industrial_registry_conditional_trace_active"] == 637


def test_registry_conditionals_require_explicit_risk_tier_two():
    catalog, _ = _activated_catalog()
    with NaturalLanguagePerfumeryAI(catalog=catalog) as ai:
        tier_one = RecipeConstraints(
            max_risk_tier=1,
            max_ingredient_price_per_kg=300,
            max_formula_cost_per_kg=500,
            min_availability=0.5,
            max_ingredients=20,
        )
        brief_one = ai.parser.parse("balanced fresh floral woody fragrance", tier_one)
        accepted_one, _ = ai.screen.screen(
            ai.catalog,
            brief_one,
            supplier_registry=ai.supplier_registry,
            as_of=date.today(),
        )
        assert len(accepted_one) == 29
        assert not any(
            item.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
            for item in accepted_one
        )

        tier_two_default = RecipeConstraints(
            max_risk_tier=2,
            max_ingredient_price_per_kg=300,
            max_formula_cost_per_kg=500,
            min_availability=0.5,
            max_ingredients=20,
        )
        default_brief = ai.parser.parse(
            "balanced fresh floral woody fragrance", tier_two_default
        )
        default_candidates, default_rejected = ai.screen.screen(
            ai.catalog,
            default_brief,
            supplier_registry=ai.supplier_registry,
            as_of=date.today(),
        )
        assert len(default_candidates) == 34
        assert default_rejected["registry_conditional_not_requested"] == 637

        tier_two = RecipeConstraints(
            max_risk_tier=2,
            enable_registry_trace_candidates=True,
            max_ingredient_price_per_kg=300,
            max_formula_cost_per_kg=500,
            min_availability=0.5,
            max_ingredients=20,
        )
        brief_two = ai.parser.parse("balanced fresh floral woody fragrance", tier_two)
        accepted_two, _ = ai.screen.screen(
            ai.catalog,
            brief_two,
            supplier_registry=ai.supplier_registry,
            as_of=date.today(),
        )
        assert len(accepted_two) == 671
        assert sum(
            item.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
            for item in accepted_two
        ) == 637

        qualified = RecipeConstraints(
            max_risk_tier=2,
            enable_registry_trace_candidates=True,
            max_ingredient_price_per_kg=300,
            max_formula_cost_per_kg=500,
            min_availability=0.5,
            max_ingredients=20,
            validation_level="qualified",
        )
        qualified_brief = ai.parser.parse(
            "balanced fresh floral woody fragrance", qualified
        )
        qualified_candidates, qualified_rejected = ai.screen.screen(
            ai.catalog,
            qualified_brief,
            supplier_registry=ai.supplier_registry,
            as_of=date.today(),
        )
        assert not any(
            item.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
            for item in qualified_candidates
        )
        assert qualified_rejected["registry_conditional_prototype_only"] == 637


def test_registry_formula_is_returned_only_as_experimental_candidate():
    catalog, _ = _activated_catalog()
    constraints = RecipeConstraints(
        max_risk_tier=2,
        enable_registry_trace_candidates=True,
        max_ingredient_price_per_kg=180,
        min_availability=0.75,
        max_ingredients=12,
        target_similarity=50,
        minimum_realism_score=50,
        simulation_draws=64,
        physics_search_population=2,
    )
    with NaturalLanguagePerfumeryAI(catalog=catalog) as ai:
        result = ai.create_recipe("smoky leathery woody dry fragrance", constraints)

    registry_lines = [
        line
        for line in result.closest_candidate
        if line.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
    ]
    assert result.status == "experimental_registry_candidate"
    assert result.recipe
    assert registry_lines
    assert all(
        line.active_material_percent <= REGISTRY_CONDITIONAL_CAP_PERCENT + 1e-9
        for line in registry_lines
    )
    assert any("조건부 산업 레지스트리 trace 후보" in item for item in result.safety.warnings)
    assert result.safety.manufacturing_ready is False


def test_registry_activation_rejects_wrong_hash_and_invalid_limit():
    with pytest.raises(ValueError, match="hash mismatch"):
        activate_registry_conditionals(
            IngredientCatalog.load_builtin(),
            REGISTRY,
            expected_sha256="0" * 64,
        )
    with IndustrialIngredientRegistry(REGISTRY) as registry:
        with pytest.raises(ValueError, match="between 1 and 5000"):
            registry.conditional_runtime_candidates(limit=0)
