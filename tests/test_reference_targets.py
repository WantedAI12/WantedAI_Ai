from __future__ import annotations

import hashlib
from datetime import date

import pytest

from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.models import RecipeConstraints
from fragrance_ai.recommender.physsim import ConcentrationAwarePhysSim
from fragrance_ai.recommender.reference_targets import (
    ReferenceComponent,
    ReferenceEvidence,
    ReferenceTarget,
    ReferenceTargetStore,
)
from fragrance_ai.recommender.science import ScientificPropertyStore


AS_OF = date(2026, 7, 28)


def _target_store(
    tmp_path,
    ingredient_id: str,
    *,
    component_percent: float = 100.0,
    product_concentration_percent: float = 15.0,
):
    source = tmp_path / "quantitative-composition.csv"
    source.write_text("ingredient_id,percent\nmaterial,100\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = ReferenceTargetStore(
        (
            ReferenceTarget(
                target_id="reference-1",
                version="lot-2026-07",
                composition_basis="quantitative_gc_ms",
                product_category="eau_de_parfum",
                product_concentration_percent=product_concentration_percent,
                matrix_id="ethanol-base-1",
                components=(ReferenceComponent(ingredient_id, component_percent),),
                evidence=(
                    ReferenceEvidence(
                        evidence_type="composition",
                        path=str(source),
                        sha256=digest,
                        issued_on=date(2026, 7, 1),
                    ),
                ),
            ),
        )
    )
    return store, source


@pytest.mark.parametrize(
    ("component_percent", "product_concentration_percent", "message"),
    [
        (float("nan"), 15.0, "finite and positive"),
        (100.0, float("nan"), "product concentration is invalid"),
    ],
)
def test_reference_target_rejects_nonfinite_quantities(
    tmp_path,
    component_percent,
    product_concentration_percent,
    message,
):
    ingredient = IngredientCatalog.load_builtin().ingredients[0]
    store, _ = _target_store(
        tmp_path,
        ingredient.ingredient_id,
        component_percent=component_percent,
        product_concentration_percent=product_concentration_percent,
    )
    constraints = RecipeConstraints(
        reference_target_id="reference-1",
        commercial_product_base_id="ethanol-base-1",
    )
    with pytest.raises(ValueError, match=message):
        store.resolve(
            "reference-1",
            ingredients={ingredient.ingredient_id: ingredient},
            constraints=constraints,
            as_of=AS_OF,
        )


def test_reference_target_requires_current_bytes_and_exact_matrix(tmp_path):
    catalog = IngredientCatalog.load_builtin()
    ingredient = catalog.ingredients[0]
    store, source = _target_store(tmp_path, ingredient.ingredient_id)
    constraints = RecipeConstraints(
        reference_target_id="reference-1",
        commercial_product_base_id="ethanol-base-1",
    )
    resolved = store.resolve(
        "reference-1",
        ingredients={ingredient.ingredient_id: ingredient},
        constraints=constraints,
        as_of=AS_OF,
    )
    assert resolved.composition_basis == "quantitative_gc_ms"
    assert resolved.lines[0].finished_product_percent == 15.0

    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence bytes changed"):
        store.resolve(
            "reference-1",
            ingredients={ingredient.ingredient_id: ingredient},
            constraints=constraints,
            as_of=AS_OF,
        )

    wrong_matrix = RecipeConstraints(
        reference_target_id="reference-1",
        commercial_product_base_id="water-base",
    )
    with pytest.raises(ValueError, match="matrix does not match"):
        store.resolve(
            "reference-1",
            ingredients={ingredient.ingredient_id: ingredient},
            constraints=wrong_matrix,
            as_of=AS_OF,
        )


def test_physsim_uses_evidenced_formula_instead_of_self_generated_target(tmp_path):
    catalog = IngredientCatalog.load_builtin()
    ingredient = catalog.ingredients[0]
    store, _ = _target_store(tmp_path, ingredient.ingredient_id)
    constraints = RecipeConstraints(
        reference_target_id="reference-1",
        commercial_product_base_id="ethanol-base-1",
    )
    ingredients = {item.ingredient_id: item for item in catalog.ingredients}
    resolved = store.resolve(
        "reference-1",
        ingredients=ingredients,
        constraints=constraints,
        as_of=AS_OF,
    )
    brief = NaturalLanguageBriefParser(catalog).parse(
        "clean citrus fragrance",
        constraints,
    )
    result = ConcentrationAwarePhysSim().evaluate(
        list(resolved.lines),
        ingredients,
        brief,
        ScientificPropertyStore.load_builtin(),
        reference_target_lines=list(resolved.lines),
    )
    assert "evidenced_reference_composition_target" in result.flags
    assert "self_generated_text_target_prototype" not in result.flags
    assert result.target_ingredient_ids == (ingredient.ingredient_id,)
