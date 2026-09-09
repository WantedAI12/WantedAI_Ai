"""Regression checks for per-request batching and reusable candidate selection."""

from dataclasses import replace
from datetime import date
from unittest.mock import patch

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.catalog import (
    HistoricalReferenceCorpus,
    IngredientCatalog,
    IngredientMention,
    find_text_spans,
    normalize_name,
)
from fragrance_ai.recommender.science import MolecularProperties, ScientificPropertyStore


def _properties(ingredient_id="dihydromyrcenol", pressure=1.0):
    return MolecularProperties(
        ingredient_id, None, 156.0, 2.0, 20.0, 0, 1, 3, 120.0,
        pressure, 220.0, 0.1, "TEST", "2026-09-05",
    )


def test_bulk_properties_are_bounded_fresh_and_independent(tmp_path):
    path = tmp_path / "properties.db"
    with ScientificPropertyStore(path) as store:
        quoted_id = "material'quoted"
        store.upsert(_properties(quoted_id))
        ids = [quoted_id, *(f"missing-{i}" for i in range(1200)), quoted_id]
        statements = []
        store.connection.set_trace_callback(statements.append)
        snapshot = store.get_many(ids)
        queries = [statement for statement in statements if statement.startswith("SELECT")]
        assert len(queries) == 3
        assert snapshot == {quoted_id: store.get(quoted_id)}
        assert snapshot.get("missing-1") is None
        statements.clear()
        assert store.get_many([]) == {}
        assert statements == []
        with ScientificPropertyStore(path) as writer:
            writer.upsert(_properties(quoted_id, pressure=2.0))
            writer.upsert(_properties("missing-1", pressure=3.0))
        updated = store.get_many(ids)
        assert updated[quoted_id].vapor_pressure_pa_25c == 2.0
        assert updated["missing-1"].vapor_pressure_pa_25c == 3.0
        assert snapshot[quoted_id].vapor_pressure_pa_25c == 1.0
        assert snapshot.get("missing-1") is None


@pytest.mark.parametrize("text", [
    "clean musk, no rose", "장미와 라벤더 향은 빼고", "pear peach PEA",
    "BERGAMOT and white   musk", "", "notaningredient",
])
def test_precomputed_aliases_preserve_original_spans(text):
    catalog = IngredientCatalog.load_builtin()
    raw = []
    normalized_text = normalize_name(text)
    for ingredient in catalog.ingredients:
        for alias in sorted(ingredient.all_names(), key=len, reverse=True):
            normalized_alias = normalize_name(alias)
            if len(normalized_alias) < 2 or normalized_alias not in normalized_text:
                continue
            raw.extend(
                IngredientMention(ingredient, alias, start, end)
                for start, end in find_text_spans(text, alias)
            )
    raw.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias))
    expected = []
    for mention in raw:
        if not any(mention.start < old.end and old.start < mention.end for old in expected):
            expected.append(mention)
    assert catalog.mentioned_ingredient_spans(text) == expected


def test_request_selects_once_without_stale_cross_request_results(tmp_path):
    corpus = HistoricalReferenceCorpus(tmp_path / "unpackaged-reference.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        constraints = RecipeConstraints(simulation_draws=64, physics_search_population=7)
        with (
            patch.object(ai.optimizer, "_select_candidates", wraps=ai.optimizer._select_candidates) as selection,
            patch.object(ai.scientific_store, "get_many", wraps=ai.scientific_store.get_many) as bulk,
            patch.object(ai.scientific_store, "get", side_effect=AssertionError("per-material SQL")),
        ):
            first = ai.create_recipe("clean fresh citrus woody", constraints, as_of=date(2026, 9, 5))
            second = ai.create_recipe("rose powdery floral", constraints, as_of=date(2026, 9, 5))
            again = ai.create_recipe("clean fresh citrus woody", constraints, as_of=date(2026, 9, 5))
        assert selection.call_count == 3
        assert bulk.call_count == 3
        assert first.to_dict() == again.to_dict()
        assert first.formula_id != second.formula_id
        assert first.recipe and second.recipe
        assert first.candidate_variants_evaluated > 1


def test_preselected_optimizer_matches_original_selection_for_each_objective(tmp_path):
    corpus = HistoricalReferenceCorpus(tmp_path / "unpackaged-reference.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        brief = ai.parser.parse("clean fresh citrus woody", RecipeConstraints())
        candidates, _ = ai.screen.screen(ai.catalog, brief, supplier_registry=ai.supplier_registry)
        chosen = ai.optimizer._select_candidates(candidates, brief)
        for factors in (None, {item.ingredient_id: 0.8 + i / 10 for i, item in enumerate(candidates)}):
            expected = ai.optimizer.optimize(candidates, brief, factors)
            actual = ai.optimizer._optimize_selected(chosen, brief, factors)
            assert actual == expected
        changed = replace(brief, constraints=replace(brief.constraints, max_ingredients=20))
        assert ai.optimizer.optimize(candidates, changed) == ai.optimizer._optimize_selected(
            ai.optimizer._select_candidates(candidates, changed), changed
        )
