from __future__ import annotations

import hashlib
from datetime import date

from fragrance_ai.recommender.manufacturing import ManufacturingPlanner
from fragrance_ai.recommender.manufacturing_profiles import (
    ManufacturingProfileRegistry,
    PackagingProfile,
    ProductBaseProfile,
    TechnicalEvidence,
)
from fragrance_ai.recommender.models import Ingredient, RecipeConstraints, RecipeLine


AS_OF = date(2026, 7, 28)


def _ingredient() -> Ingredient:
    return Ingredient(
        ingredient_id="mat-1",
        name="Verified material",
        aliases=(),
        cas_number="00-00-0",
        pyramid="heart",
        profile={"floral": 1.0},
        price_per_kg=10.0,
        availability=1.0,
        rarity="common",
        risk_tier=0,
        odor_impact=1.0,
        max_concentrate_percent=100.0,
        formulation_ready=True,
        density_g_ml=0.95,
        solubility=("ethanol",),
        oxidation_risk="low",
        discoloration_risk="low",
        shelf_life_months=24,
    )


def _line() -> RecipeLine:
    return RecipeLine(
        ingredient_id="mat-1",
        name="Verified material",
        pyramid="heart",
        concentrate_percent=100.0,
        finished_product_percent=15.0,
        volume_ml_for_batch=None,
        price_per_kg=10.0,
        availability=1.0,
        risk_tier=0,
        reason="test",
        density_g_ml=0.95,
    )


def _registry(tmp_path) -> ManufacturingProfileRegistry:
    evidence_path = tmp_path / "technical-document.pdf"
    evidence_path.write_bytes(b"immutable-test-document")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence = TechnicalEvidence(
        path=str(evidence_path),
        sha256=digest,
        issued_on=date(2026, 1, 1),
        expires_on=date(2027, 1, 1),
        evidence_type="technical_data",
    )
    return ManufacturingProfileRegistry(
        base_profiles=(
            ProductBaseProfile(
                base_id="base-1",
                version="2026.1",
                compatible_product_categories=("eau_de_parfum",),
                solvent_system=("ethanol",),
                density_g_ml=0.86,
                flash_point_c=16.0,
                maximum_process_temperature_c=10.0,
                evidence=(evidence,),
            ),
        ),
        packaging_profiles=(
            PackagingProfile(
                packaging_id="pack-1",
                version="2026.1",
                compatible_product_categories=("eau_de_parfum",),
                compatible_solvent_systems=("ethanol",),
                evidence=(evidence,),
            ),
        ),
    )


def _constraints() -> RecipeConstraints:
    return RecipeConstraints(
        commercial_product_base_id="base-1",
        commercial_packaging_id="pack-1",
    )


def test_mass_arithmetic_does_not_imply_lab_readiness():
    ingredient = _ingredient()
    plan = ManufacturingPlanner().build(
        [_line()],
        {ingredient.ingredient_id: ingredient},
        RecipeConstraints(),
        as_of=AS_OF,
    )
    assert plan.total_as_supplied_material_mass_g > 0
    assert not plan.ready_for_lab_trial
    assert not plan.ready_for_manufacture
    assert "product_base_id_missing" in plan.readiness_blockers
    assert "packaging_id_missing" in plan.readiness_blockers


def test_verified_profiles_separate_lab_and_manufacturing_readiness(tmp_path):
    ingredient = _ingredient()
    planner = ManufacturingPlanner(_registry(tmp_path))
    lab_plan = planner.build(
        [_line()],
        {ingredient.ingredient_id: ingredient},
        _constraints(),
        as_of=AS_OF,
        stability_passed=False,
    )
    assert lab_plan.ready_for_lab_trial
    assert not lab_plan.ready_for_manufacture
    assert lab_plan.readiness_blockers == ["finished_product_stability_not_passed"]

    released_plan = planner.build(
        [_line()],
        {ingredient.ingredient_id: ingredient},
        _constraints(),
        as_of=AS_OF,
        stability_passed=True,
    )
    assert released_plan.ready_for_lab_trial
    assert released_plan.ready_for_manufacture
    assert released_plan.readiness_blockers == []


def test_changed_evidence_bytes_fail_closed(tmp_path):
    ingredient = _ingredient()
    registry = _registry(tmp_path)
    document = tmp_path / "technical-document.pdf"
    document.write_bytes(b"tampered")
    readiness = registry.assess(
        lines=[_line()],
        ingredients_by_id={ingredient.ingredient_id: ingredient},
        constraints=_constraints(),
        as_of=AS_OF,
        stability_passed=True,
    )
    assert not readiness.ready_for_lab_trial
    assert not readiness.ready_for_manufacture
    assert "technical_data:document_hash_mismatch" in readiness.blockers
