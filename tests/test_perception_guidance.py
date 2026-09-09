"""Synthetic contract tests; real frozen-model recipe runs are reported separately."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.models import Ingredient, SCENT_DIMENSIONS, ScentBrief
from fragrance_ai.recommender.perception_guidance import (
    LOG_DOSES, PerceptionGuidance, PerceptionSearchSession, attach_guidance,
)
from scripts.benchmark_human_mixture_profiles import ENDPOINTS


def material(identifier="a", *, strength=100.0, cap=100.0, profile=None):
    return Ingredient(identifier, identifier, (), None, "top", profile or {"citrus": 2.0},
                      10.0, 1.0, "standard", 0, 1.0, cap, True, active_strength_percent=strength)


def brief(target=None, *, desired=None, avoided=None):
    target = target or {"citrus": 1.0}
    return ScentBrief("unit-test intent", target, desired or list(target), avoided or [], [], [], "medium",
                      {"top": 100.0}, RecipeConstraints(product_concentration_percent=20.0))


def provider(structures=None):
    value = SimpleNamespace(structures=structures or {}, bank={}, by_structure={}, endpoints=ENDPOINTS,
                            solvent="pg", model={}, weight=0.25)
    value.begin = lambda request: PerceptionSearchSession(value, request)
    return value


def test_explicit_research_optin_solvent_and_hash_are_required(tmp_path):
    with pytest.raises(ValueError, match="opt-in"):
        PerceptionGuidance(tmp_path / "absent", tmp_path / "absent", solvent="pg")
    with pytest.raises(ValueError, match="solvent"):
        PerceptionGuidance(tmp_path / "absent", tmp_path / "absent", solvent="other", experimental=True)
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        PerceptionGuidance(wrong, wrong, solvent="pg", experimental=True)


def test_unsupported_active_ingredient_is_not_a_zero_vector(monkeypatch):
    session = PerceptionSearchSession(provider({"a": ("CCO", None)}), brief())
    monkeypatch.setattr(session, "_predict", lambda *args: pytest.fail("unsupported formula should abstain before prediction"))
    assert session.evaluate([50, 50], [material("a"), material("missing")], exact=True) is None
    assert "missing" in session.missing_ids


def test_pure_unknown_graph_and_cas_conflict_do_not_become_identity_matches():
    session = PerceptionSearchSession(provider({"a": ("CCO", "64-17-5"), "mixture": ("CCO.CC", None)}), brief())
    assert not session.supports(material("a"))
    assert session.supports(replace(material("a"), cas_number="64-17-5"))
    assert not session.supports(material("mixture"))


def test_final_active_dose_applied_once_and_not_weighted_twice(monkeypatch):
    session = PerceptionSearchSession(provider({name: ("CCO", None) for name in ("a", "b")}), brief())
    received = {}

    def predict(item, doses):
        received[item.ingredient_id] = doses.copy()
        result = np.zeros((len(doses), len(ENDPOINTS)))
        result[:, ENDPOINTS.index("Citrus" if item.ingredient_id == "a" else "Woody")] = 2 if item.ingredient_id == "a" else 4
        return result, [{"basis": "synthetic_test_profile"} for _ in doses]

    monkeypatch.setattr(session, "_predict", predict)
    result = session.evaluate([25, 75], [material("a", strength=50), material("b")], exact=True)
    assert received["a"][1] == pytest.approx(.025)
    assert received["b"][1] == pytest.approx(.15)
    assert result["predicted_rata_profile"]["Citrus"] == 1
    assert result["predicted_rata_profile"]["Woody"] == 2


def test_zero_mass_unsupported_line_does_not_block_real_formula(monkeypatch):
    session = PerceptionSearchSession(provider({"a": ("CCO", None)}), brief())
    monkeypatch.setattr(session, "_predict", lambda ingredient, doses: (np.ones((len(doses), len(ENDPOINTS))), [{} for _ in doses]))
    assert session.evaluate([100, 0], [material("a"), material("missing")], exact=True) is not None


@pytest.mark.parametrize("weights", [[-1, 101], [25, 25], [float("nan"), 100]])
def test_invalid_formula_fraction_is_rejected(weights):
    session = PerceptionSearchSession(provider(), brief())
    with pytest.raises(ValueError, match="formula"):
        session.evaluate(weights, [material("a"), material("b")])


def test_all_unsupported_intent_does_not_get_a_match_score():
    session = PerceptionSearchSession(provider(), brief({"musky": 1.0}))
    assert not session.enabled
    assert session.evaluate([100], [material()]) is None
    assert session.unsupported_intent == ["musky"]


def test_partial_intent_and_phase_limits_remain_explicit():
    request = brief({"citrus": .5, "musky": .5})
    request.phase_target_profiles = {"opening": {"citrus": 1}, "drydown": {"musky": 1}}
    session = PerceptionSearchSession(provider(), request)
    report = session.report(None, None, changed=False, variants=0)
    assert report["target_mass_covered_percent"] == 50
    assert report["unsupported_intent_dimensions"] == ["musky"]
    assert not report["time_evolution_validated"]
    assert not report["actual_human_accuracy_90_authorized"]


def test_unmapped_fishy_mass_cannot_create_perfect_woody_score():
    session = PerceptionSearchSession(provider(), brief({"woody": 1.0}))
    rata = np.zeros((1, len(ENDPOINTS)))
    rata[0, ENDPOINTS.index("Woody")] = 1
    rata[0, ENDPOINTS.index("Fishy")] = 99
    assert session._scores(rata @ session.projection)[0] <= 1.0000001


def test_grid_projection_matches_exact_linear_log_dose_predictor(monkeypatch):
    session = PerceptionSearchSession(provider({"a": ("CCO", None)}), brief())

    def predict(item, doses):
        result = np.zeros((len(doses), len(ENDPOINTS)))
        result[:, ENDPOINTS.index("Citrus")] = 10 + np.log10(doses)
        result[:, ENDPOINTS.index("Woody")] = 1
        return result, [{} for _ in doses]

    monkeypatch.setattr(session, "_predict", predict)
    approximate = session.evaluate([100], [material()])
    exact = session.evaluate([100], [material()], exact=True)
    assert approximate["score"] == pytest.approx(exact["score"], abs=1e-12)
    assert session.tables["a"].shape == (len(LOG_DOSES), len(session.dimensions) + 2)


def test_unknown_graph_uses_raw_native_feature_scale(monkeypatch):
    from fragrance_ai.research import conditional_profiles
    seen = {}

    def fake_features(smiles, native):
        seen.update(native)
        return {"canonical_smiles": "test-graph", "native": native}

    monkeypatch.setattr(conditional_profiles, "molecule_features", fake_features)
    session = PerceptionSearchSession(provider({"a": ("CCO", None)}), brief())
    ingredient = material(profile={"citrus": 2.0, "fresh": 1.5})
    cid, _ = session._prepare(ingredient)
    assert cid is None
    assert sum(seen["profile"]) == 3.5
    assert seen["profile"][SCENT_DIMENSIONS.index("citrus")] == 2


def test_coordinate_refinement_never_calls_objective_with_negative_weights():
    with NaturalLanguagePerfumeryAI() as ai:
        request = brief()
        request.constraints.surrogate_objective_weight = .5
        ingredients = [material("a", cap=.25), material("b", cap=60), material("c", cap=60)]
        seen = []

        def objective(weights, selected):
            seen.append(float(weights.min()))
            assert np.all(weights >= 0), "a depleted donor must be rechecked inside the receiver loop"
            return 100 - 20 * weights[0]

        result = ai.optimizer._optimize_selected(ingredients, request, formula_objective=objective)
        assert seen
        assert min(seen) >= 0
        assert sum(line.concentrate_percent for line in result[0]) == pytest.approx(100, abs=.001)


def test_default_generation_unchanged_and_unsupported_guidance_falls_back():
    constraints = RecipeConstraints(simulation_draws=64, physics_search_population=2)
    with NaturalLanguagePerfumeryAI() as ai:
        baseline = ai.create_recipe("floral fruity woody", constraints, as_of=date(2026, 9, 5))
    with NaturalLanguagePerfumeryAI(perception_guidance=provider()) as ai:
        guided = ai.create_recipe("floral fruity woody", constraints, as_of=date(2026, 9, 5))
    assert baseline.recipe and guided.recipe
    assert baseline.formula_id == guided.formula_id
    assert "perception_guidance" not in baseline.to_dict()
    assert guided.to_dict()["perception_guidance"]["status"].startswith("abstained")
    assert guided.actual_olfactory_similarity_score is None
    assert not guided.human_similarity_90_claim_authorized


def test_experimental_guidance_cannot_enter_commercial_generation():
    with NaturalLanguagePerfumeryAI(perception_guidance=provider()) as ai:
        with pytest.raises(ValueError, match="prototype"):
            ai.create_recipe("floral fruity woody", RecipeConstraints(validation_level="commercial"))


def test_additive_result_extension_preserves_existing_fields():
    with NaturalLanguagePerfumeryAI() as ai:
        result = ai.create_recipe("floral fruity woody", RecipeConstraints(simulation_draws=64, physics_search_population=2))
    base = result.to_dict()
    extended = attach_guidance(result, {"actual_human_accuracy_90_authorized": False}).to_dict()
    assert extended.pop("perception_guidance") == {"actual_human_accuracy_90_authorized": False}
    assert extended == base


def test_request_caches_and_unsupported_reference_context_are_isolated():
    value = provider()
    first, second = value.begin(brief()), value.begin(brief({"woody": 1.0}))
    first.missing_ids.add("first-request-only")
    first.tables["a"] = np.zeros((49, 13))
    assert not second.missing_ids
    assert not second.tables
    reference = brief()
    reference.constraints.reference_target_id = "verified-reference"
    assert not value.begin(reference).enabled
