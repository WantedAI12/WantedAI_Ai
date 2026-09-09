"""Scalar physics oracle, analytic gradients and fail-closed dose constraints."""

from dataclasses import replace
from datetime import date
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI
from fragrance_ai.recommender.adaptive_pyramid import prepare_adaptive_policy
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog, HistoricalReferenceCorpus
from fragrance_ai.recommender.dose_refinement import DoseModel, actual_dose_replacement_seeds, agreement_components, dose_refinement_proposals, full_inferred_note_policy, optimize_dose_support, project_seed
from fragrance_ai.recommender.models import Ingredient, profile_vector
from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer
from fragrance_ai.recommender.science import ATMOSPHERIC_PRESSURE_PA, TIMEPOINTS_MINUTES, TemporalMixtureSimulator


def materials(profile=None):
    return [Ingredient(group, group, (), None, group, profile or {dimension: 1.}, 10., 1., "common", 0, 1. + index / 2, 100., True)
            for index, (group, dimension) in enumerate(zip(("top", "heart", "base"), ("citrus", "floral", "woody")))]


def setup_case(text="citrus floral woody", constraints=None, profile=None):
    items = materials(profile)
    brief = NaturalLanguageBriefParser(IngredientCatalog(items, {})).parse(text, constraints)
    variant = ConstrainedFormulaOptimizer().variant_from_weights(items, brief, np.array([25., 40., 35.]))
    policy = prepare_adaptive_policy(brief, variant[0], {item.ingredient_id: item for item in items})
    return items, brief, variant[0], policy


def scalar_arrays(simulator, prepared, interaction, targets, draws, rng):
    """Original draw/time-loop equations, independent of the new batch code."""
    result = np.zeros((draws, len(TIMEPOINTS_MINUTES), 19))
    signals = np.zeros(result.shape[:2])
    scores = np.zeros_like(signals)
    vectors = np.asarray([item.ingredient.vector() for item in prepared])
    transport = np.asarray([simulator._air_to_receptor_transport(item.properties) for item in prepared])
    for draw in range(draws):
        vapor = np.asarray([item.vapor_pressure_pa * math.exp(rng.normal(0., item.vapor_log_sigma)) for item in prepared])
        threshold = np.asarray([item.odor_threshold_ppm * math.exp(rng.normal(0., item.threshold_log_sigma)) for item in prepared])
        activity = np.asarray([item.activity_coefficient * math.exp(rng.normal(0., .18 if item.properties else .50)) for item in prepared])
        gas = np.asarray([item.mole_fraction for item in prepared]) * activity
        gas *= vapor / ATMOSPHERIC_PRESSURE_PA * 1e6
        oav = np.maximum(1e-12, gas / np.maximum(threshold, 1e-12))
        lives = np.asarray([simulator._half_life_minutes(item.ingredient, item.properties, value) for item, value in zip(prepared, vapor)])
        strength = float(np.clip(rng.normal(.2, .05), .08, .4))
        for t, minutes in enumerate(TIMEPOINTS_MINUTES):
            current = simulator._odor_response(oav * np.power(.5, minutes / lives)) * transport
            suppressed = current / (1 + strength * (interaction @ current))
            mixture = suppressed @ vectors
            if mixture.sum() > 0:
                mixture /= mixture.sum()
            result[draw, t] = mixture
            signals[draw, t] = suppressed.sum()
            target, desired, avoided = targets[t]
            scores[draw, t] = simulator.temporal_target_similarity(target, mixture, desired, avoided)
    return scores, signals, result


@pytest.mark.parametrize("draws", [64, 200, 257])
def test_batched_physics_preserves_scalar_draw_order_and_values(draws):
    items, brief, lines, _ = setup_case("opening no sweetness, drydown woody")
    engine = TemporalMixtureSimulator()
    prepared = engine._prepare(lines, {item.ingredient_id: item for item in items}, {})
    interaction = engine._interaction_matrix(prepared)
    targets = engine.targets_by_time(brief)
    old_rng, new_rng = np.random.default_rng(915), np.random.default_rng(915)
    expected = scalar_arrays(engine, prepared, interaction, targets, draws, old_rng)
    actual = engine._sampled_temporal_arrays(prepared, interaction, targets, draws, new_rng)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.parametrize("target,desired,avoided", [
    ({"woody": 1.}, ["woody"], []), ({"citrus": .5, "floral": .5}, [], ["musky"]),
    ({}, [], ["gourmand"]), ({}, [], []),
])
def test_batched_legacy_similarity_matches_scalar(target, desired, avoided):
    matrix = np.random.default_rng(33).dirichlet(np.ones(19), 30)
    matrix[0] = 0.
    vector = profile_vector(target)
    expected = [TemporalMixtureSimulator.temporal_target_similarity(vector, row, desired, avoided) for row in matrix]
    np.testing.assert_allclose(TemporalMixtureSimulator._batch_temporal_similarity(vector, matrix, desired, avoided), expected, atol=1e-12)


@pytest.mark.parametrize("concentration", [7.5, 15., 100.])
def test_actual_dose_analytic_jacobians_match_finite_differences(concentration):
    model = DoseModel(materials(), {}, concentration)
    weights = np.array([.29, .41, .30])
    state = model.evaluate(weights)
    for value_name, jac_name in [("nominal", "nominal_jac"), ("temporal", "temporal_jac"), ("signal", "signal_jac")]:
        numerical = np.stack([(getattr(model.evaluate(weights + row * 1e-6), value_name) - getattr(model.evaluate(weights - row * 1e-6), value_name)) / 2e-6
                              for row in np.eye(3)], axis=-1)
        np.testing.assert_allclose(numerical, getattr(state, jac_name), atol=1e-7, rtol=1e-5)


def test_proposal_samples_do_not_depend_on_support_order():
    items = materials()
    weights = np.array([.2, .3, .5])
    a = DoseModel(items, {}, 15).evaluate(weights)
    b = DoseModel(items[::-1], {}, 15).evaluate(weights[::-1])
    np.testing.assert_allclose(a.temporal, b.temporal, atol=1e-12)
    np.testing.assert_allclose(a.signal, b.signal, atol=1e-12)
    np.testing.assert_allclose(a.temporal_jac, b.temporal_jac[..., ::-1], atol=1e-12)


def test_complete_component_gradients_and_undefined_target():
    model = DoseModel(materials(), {}, 15)
    x = np.array([.29, .41, .30])
    goal = {"citrus": .37, "floral": .2, "woody": .43}
    current = model.evaluate(x)
    _, gradient = agreement_components(goal, current.nominal, current.nominal_jac, ["floral"])
    def values(point):
        state = model.evaluate(point)
        return agreement_components(goal, state.nominal, state.nominal_jac, ["floral"])[0]
    numerical = np.stack([(values(x + row * 1e-6) - values(x - row * 1e-6)) / 2e-6 for row in np.eye(3)], axis=-1)
    np.testing.assert_allclose(numerical, gradient, atol=1e-7)
    with pytest.raises(ValueError, match="positive target"):
        agreement_components({}, current.nominal, current.nominal_jac)


def test_projection_preserves_partial_percentages_caps_cost_and_required_materials():
    items, brief, _, policy = setup_case("woody base 60%")
    seed = {"top": 25., "heart": 40., "base": 35.}
    answer = project_seed(items, seed, brief, policy, {"top": 15.})
    weights, lower, caps, matrix, rhs = answer
    assert weights.sum() == pytest.approx(1)
    assert weights[2] == pytest.approx(.60)
    assert weights[0] >= .15 - 1e-8
    assert np.all(weights >= lower - 1e-8) and np.all(weights <= caps + 1e-8)
    assert np.all(matrix @ weights <= rhs + 1e-8)
    assert project_seed(items, seed, brief, policy, {"top": 50.}) is None
    cheap = [replace(item, price_per_kg=.6) for item in items]
    assert project_seed(cheap, seed, replace(brief, constraints=replace(brief.constraints, max_formula_cost_per_kg=.5)), policy, {}) is None


@pytest.mark.parametrize("bad", [np.array([.2, .3, .4, .9]), np.array([np.nan, .3, .7, .9]), np.array([-.1, .4, .7, .9])])
def test_invalid_solver_output_is_rejected_without_approval(monkeypatch, bad):
    items, brief, _, policy = setup_case(profile={"woody": 1.})
    baseline = DoseModel(items, {}, 15).evaluate(np.array([.25, .40, .35]))
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.minimize", lambda *a, **k: SimpleNamespace(x=bad, success=True, nit=1))
    assert optimize_dose_support(items, brief, {}, {"top": 25., "heart": 40., "base": 35.}, baseline, policy, {}) is None


def test_unexpected_solver_warnings_are_not_hidden(monkeypatch):
    items, brief, _, policy = setup_case(profile={"woody": 1.})
    baseline = DoseModel(items, {}, 15).evaluate(np.array([.25, .40, .35]))
    def warn(*args, **kwargs):
        warnings.warn("unexpected numerical failure", RuntimeWarning)
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.minimize", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="unexpected numerical failure"):
            optimize_dose_support(items, brief, {}, {"top": 25., "heart": 40., "base": 35.}, baseline, policy, {})


def test_actual_dose_pool_replacements_keep_mass_groups_cost_and_required_ids():
    items, brief, _, _ = setup_case("woody")
    original = {"top": 25., "heart": 40., "base": 35.}
    better = replace(items[0], ingredient_id="woody_top", name="woody top", profile={"woody": 1.})
    pool = [*items, better]
    seeds = actual_dose_replacement_seeds(pool, brief, {}, original, {})
    assert seeds
    assert seeds[0] == {"heart": 40., "base": 35., "woody_top": 25.}
    assert actual_dose_replacement_seeds(pool, brief, {}, original, {"top": 25.}) == []
    expensive = replace(better, price_per_kg=11.)
    request = replace(brief, constraints=replace(brief.constraints, max_formula_cost_per_kg=10.))
    assert actual_dose_replacement_seeds([*items, expensive], request, {}, original, {}) == []
    low_cap = replace(better, max_concentrate_percent=24.)
    assert actual_dose_replacement_seeds([*items, low_cap], brief, {}, original, {}) == []


def test_exclusion_only_phase_remains_unscorable_and_is_identified(tmp_path):
    with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "not-present.db")) as ai:
        result = ai.create_recipe("opening no sweetness, drydown woody musk", as_of=date(2026, 9, 5))
    assert result.calculated_profile_similarity is None
    info = result.full_profile_assessment["search"]["full_pool_search"]["dose_refinement"]
    assert info["missing_positive_phase_targets"] == ["opening"]
    assert not result.full_profile_target_met and not result.human_similarity_90_claim_authorized
    with pytest.raises(ValueError, match="boolean"):
        NaturalLanguagePerfumeryAI(enable_dose_refinement="yes")


def test_large_requested_maximum_does_not_disable_small_support_refinement(monkeypatch):
    from fragrance_ai.recommender.dose_refinement import _bounded_dose_refinement_proposals
    items, brief, lines, policy = setup_case()
    brief = replace(brief, constraints=replace(brief.constraints, max_ingredients=500))
    pool = [*items, *(replace(items[i % 3], ingredient_id=f"extra-{i}", name=f"extra {i}") for i in range(40))]
    supports = []
    def refine(selected, *args, **kwargs):
        supports.append(len(selected))
        return {"weights_percent": {selected[0].ingredient_id: 100.}}
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.optimize_dose_support", refine)
    proposals = list(_bounded_dose_refinement_proposals(pool, brief, {}, lines, policy, [], {}))
    assert proposals and supports[0] == len(lines)
    assert all(proposal["full_pool_considered"] == len(pool) for proposal in proposals)
    assert max(supports) <= 32


def test_coordinated_substitutions_preserve_mass_and_reject_duplicate_donors():
    from fragrance_ai.recommender.dose_refinement import coordinated_replacement_seeds
    original = {"a": 60., "b": 40.}
    replacements = [{"c": 60., "b": 40.}, {"a": 60., "d": 40.}, {"e": 60., "b": 40.}]
    proposals = list(coordinated_replacement_seeds(original, replacements))
    assert proposals == [{"c": 60., "d": 40.}, {"d": 40., "e": 60.}]
    assert all(sum(seed.values()) == 100 and min(seed.values()) > 0 for seed in proposals)


@pytest.mark.parametrize("margin", [-.01, 1.01, float("nan")])
def test_invalid_proposal_margin_is_not_accepted(margin):
    items, brief, lines, policy = setup_case()
    with pytest.raises(ValueError, match="proposal signal margin"):
        optimize_dose_support(items, brief, {}, {}, None, policy, {}, proposal_signal_margin=margin)


def test_open_note_policy_changes_only_inferred_bounds():
    _, _, _, policy = setup_case("woody base 60%")
    result = full_inferred_note_policy(policy)
    assert result["bounds"] == {"top": (0., 100.), "heart": (0., 100.), "base": (60., 60.)}
    assert policy["bounds"]["top"] == (10., 80.)
    assert {key: value for key, value in result.items() if key != "bounds"} == {key: value for key, value in policy.items() if key != "bounds"}


def test_verified_bounded_pass_can_exit_without_any_new_fallback(monkeypatch):
    items, brief, lines, policy = setup_case()
    def bounded(*args):
        yield {"marker": "original proposal"}
    def forbidden(*args, **kwargs):
        pytest.fail("the unconsumed fallback must not run")
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement._bounded_dose_refinement_proposals", bounded)
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.profile_upper_bound", forbidden)
    proposals = dose_refinement_proposals(items, brief, {}, lines, policy, [], {}, current_lines=forbidden)
    assert next(proposals) == {"marker": "original proposal"}
    proposals.close()


@pytest.mark.parametrize("changing,expected_rounds", [(False, 1), (True, 2)])
def test_open_rounds_stop_without_progress_and_have_a_fixed_limit(monkeypatch, changing, expected_rounds):
    items, brief, lines, policy = setup_case()
    def bounded(*args):
        yield {"weights_percent": {"top": 25., "heart": 40., "base": 35.}}
    count = 0
    def feedback():
        nonlocal count
        count += 1
        return [replace(line, concentrate_percent=line.concentrate_percent + (count * .0001 if changing else 0)) for line in lines]
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement._bounded_dose_refinement_proposals", bounded)
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.profile_upper_bound", lambda *args: {"upper_score": 100.})
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.optimize_full_pool", lambda *args, **kwargs: SimpleNamespace(weights_percent={}))
    proposals = list(dose_refinement_proposals(items, brief, {}, lines, policy, [], {}, current_lines=feedback))
    assert "allocation_mode" not in proposals[0]
    assert [p["open_round"] for p in proposals[1:]] == list(range(1, expected_rounds + 1))
    assert all(p["allocation_mode"] == "inferred_full_range" for p in proposals[1:])


def test_fully_explicit_notes_cannot_enter_open_range_fallback(monkeypatch):
    items, brief, lines, policy = setup_case("woody top 60% heart 25% base 15%")
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement._bounded_dose_refinement_proposals", lambda *args: iter(()))
    def forbidden(*args, **kwargs):
        pytest.fail("explicit percentages must stay locked")
    monkeypatch.setattr("fragrance_ai.recommender.dose_refinement.profile_upper_bound", forbidden)
    assert list(dose_refinement_proposals(items, brief, {}, lines, policy, [], {})) == []
