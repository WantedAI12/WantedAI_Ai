"""Mathematical counterexamples; not human sensory validation."""

from dataclasses import replace
from datetime import date
import json

import numpy as np
import pytest

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus, IngredientCatalog
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.profile_match import (
    PROFILE_MATCH_KIND,
    assess_recipe_profiles,
    attach_profile_assessment,
    compare_profiles,
    full_profile_similarity,
)


def test_unrequested_flower_mass_cannot_receive_perfect_woody_score():
    result = compare_profiles({"woody": 1}, {"woody": .345, "floral": .655})
    assert result.score == pytest.approx(34.5)
    assert result.overlap_score == pytest.approx(34.5)
    assert result.total_variation == pytest.approx(.655)
    assert result.excess == {"floral": .655}
    assert result.deficit == {"woody": .655}
    assert not result.actual_human_similarity_measured


def test_identical_full_profile_matches_and_is_scale_invariant():
    result = compare_profiles({"woody": 2, "floral": 1}, {"woody": 20, "floral": 10})
    assert result.score == pytest.approx(100)
    assert result.total_variation == 0
    extreme = compare_profiles({"woody": 1e308, "floral": 1e308}, {"woody": 1, "floral": 1})
    assert extreme.score == pytest.approx(100)


def test_fast_optimizer_score_equals_full_report():
    rng = np.random.default_rng(7)
    for _ in range(30):
        a, b = rng.random(19), rng.random(19)
        assert full_profile_similarity(a, b, ["woody"], ["floral"]) == pytest.approx(
            compare_profiles(a, b, avoided=["floral"]).score, abs=1e-12
        )


def test_disjoint_and_empty_profiles_do_not_match():
    assert compare_profiles({"woody": 1}, {"floral": 1}).score == 0
    empty = compare_profiles({"woody": 0}, {"woody": 0})
    assert empty.score is None
    assert empty.status == "undefined_empty_profile"
    assert full_profile_similarity(np.zeros(19), np.zeros(19)) == 0


def test_extra_odor_monotonically_decreases_single_target_agreement():
    values = [compare_profiles({"woody": 1}, {"woody": 1 - p, "floral": p}).score for p in (0, .1, .3, .5, .9)]
    assert values == sorted(values, reverse=True)
    assert values[1] == pytest.approx(90)


def test_avoidance_cannot_increase_score_even_with_conflicting_target():
    a, b = {"woody": .5, "floral": .5}, {"woody": .5, "floral": .5}
    assert compare_profiles(a, b, avoided=["floral"]).score == 50


@pytest.mark.parametrize("bad", [{"unknown": 1}, {"woody": -1}, {"woody": float("nan")}, {"woody": float("inf")}, [1, 2]])
def test_invalid_profiles_are_rejected(bad):
    with pytest.raises(ValueError):
        compare_profiles({"woody": 1}, bad)


def test_caller_requested_dimensions_do_not_hide_other_notes():
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    a, b = np.zeros(19), np.zeros(19)
    a[SCENT_DIMENSIONS.index("woody")] = 1
    b[SCENT_DIMENSIONS.index("woody")] = .1
    b[SCENT_DIMENSIONS.index("floral")] = .9
    assert full_profile_similarity(a, b, ["woody"], []) == pytest.approx(10)


def parse(text):
    return NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(text)


def test_time_profile_uses_each_phase_target_and_does_not_mutate_weights():
    brief = parse("opening citrus, drydown woody without citrus")
    points = [
        {"phase": "opening", "target_profile": {"citrus": 1}, "scent_profile": {"citrus": .8, "woody": .2}},
        {"phase": "drydown", "target_profile": {"woody": 1}, "scent_profile": {"woody": 1}},
    ]
    weights = np.array([2., 2.])
    report = assess_recipe_profiles(brief, brief.target_profile, points, weights)
    assert weights.tolist() == [2., 2.]
    assert report["score"] == pytest.approx(90)
    assert report["temporal_minimum_score"] == pytest.approx(80)
    assert report["target_met"]
    assert report["temporal"][0]["target_profile"]["citrus"] == 1
    assert report["temporal"][1]["target_profile"]["woody"] == 1
    assert not report["actual_human_similarity_measured"]
    assert report["uncertainty_interval"] is None
    assert assess_recipe_profiles(brief, brief.target_profile, points, [1e308, 1e308])["score"] == pytest.approx(90)


def test_negative_only_phase_is_not_silently_filled_or_dropped():
    brief = parse("opening no sweetness, drydown woody musk")
    points = [
        {"phase": "opening", "target_profile": {}, "scent_profile": {"woody": 1}},
        {"phase": "drydown", "target_profile": {"woody": 1}, "scent_profile": {"woody": 1}},
    ]
    report = assess_recipe_profiles(brief, brief.target_profile, points, [.5, .5])
    assert report["temporal"][0]["score"] is None
    assert report["temporal_mean_score"] is None
    assert report["score"] is None
    assert not report["target_met"]
    # A genuinely zero-weight phase does not affect the declared mean.
    zero_weight = assess_recipe_profiles(brief, brief.target_profile, points, [0, 1])
    assert zero_weight["score"] == pytest.approx(100)


@pytest.mark.parametrize("weights", [[0], [-1], [float("nan")], [float("inf")], [1, 2]])
def test_invalid_temporal_weights_fail_closed(weights):
    brief = parse("woody")
    with pytest.raises(ValueError, match="weights"):
        assess_recipe_profiles(brief, brief.target_profile, [{"scent_profile": {"woody": 1}}], weights)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    corpus = HistoricalReferenceCorpus(tmp_path_factory.mktemp("full-profile") / "absent-reference.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        yield ai.create_recipe("clean fresh citrus woody", as_of=date(2026, 9, 5))


def test_actual_generation_exposes_honest_additive_contract(generated):
    payload = generated.to_dict()
    assert generated.recipe and generated.safety.internal_gate_passed
    assert generated.similarity_score == generated.legacy_preference_score
    assert generated.similarity_score >= 90
    assert 0 < generated.calculated_profile_similarity < 90
    assert not generated.full_profile_target_met
    report = payload["full_profile_assessment"]
    search = report["search"]
    assert search["additional_variants"] > 0
    assert search["selected_score"] >= search["baseline_score"]
    assert report["score"] == pytest.approx(search["selected_score"], abs=.001)
    assert payload["score_contract"]["primary_profile_score_field"] == "calculated_profile_similarity"
    assert not payload["score_contract"]["strict_full_profile_gate"]
    assert not payload["score_contract"]["actual_human_90_proven_by_this_score"]
    assert payload["actual_olfactory_similarity_score"] is None
    assert not payload["human_similarity_90_claim_authorized"]
    json.dumps(payload, allow_nan=False)


def test_strict_gate_rejects_under_target_preserves_candidate_and_original(generated):
    original = generated.to_dict()
    strict = attach_profile_assessment(generated, generated.full_profile_assessment, strict=True)
    assert strict.similarity_score == generated.calculated_profile_similarity
    assert strict.legacy_preference_score == generated.similarity_score
    assert strict.similarity_kind == PROFILE_MATCH_KIND
    assert strict.status == "no_safe_match" and strict.recipe == []
    assert strict.closest_candidate == generated.closest_candidate
    assert strict.closest_candidate
    assert strict.brief.constraints.target_similarity == 90
    assert not strict.simulation_only_approved
    assert not strict.manufacturing_plan.ready_for_lab_trial
    assert not strict.manufacturing_plan.ready_for_manufacture
    assert strict.manufacturing_plan.readiness_blockers
    assert generated.to_dict() == original


def test_gate_can_pass_a_complete_profile_but_cannot_promote_a_blocked_result(generated):
    # Structural gate test only: not a measured or generated 100-point recipe.
    assessment = assess_recipe_profiles(generated.brief, generated.brief.target_profile, [], [])
    assert assessment["score"] == pytest.approx(100)
    strict = attach_profile_assessment(generated, assessment, strict=True)
    assert strict.recipe == generated.recipe and strict.full_profile_target_met
    blocked = replace(generated, recipe=[], status="no_safe_match", message="original safety block")
    still_blocked = attach_profile_assessment(blocked, assessment, strict=True)
    assert still_blocked.recipe == [] and still_blocked.status == "no_safe_match"
    assert still_blocked.message == "original safety block"
    failed = attach_profile_assessment(blocked, generated.full_profile_assessment, strict=True)
    assert "original safety block" in failed.message


def test_real_strict_service_and_reference_request_cannot_fake_target_pass(tmp_path, generated):
    with NaturalLanguagePerfumeryAI(
        corpus=HistoricalReferenceCorpus(tmp_path / "absent-reference.db"),
        require_full_profile_match=True,
    ) as ai:
        strict = ai.create_recipe("clean fresh citrus woody", as_of=date(2026, 9, 5))
        reference = ai.create_recipe(
            "clean fresh citrus woody", RecipeConstraints(reference_target_id="unknown-reference"),
            as_of=date(2026, 9, 5),
        )
    assert strict.formula_id == generated.formula_id
    assert strict.recipe == [] and strict.closest_candidate
    assert strict.similarity_score == pytest.approx(generated.calculated_profile_similarity)
    assert strict.to_dict()["score_contract"]["strict_full_profile_gate"]
    assert reference.calculated_profile_similarity is None
    assert not reference.full_profile_target_met and reference.recipe == []
    assert not reference.physsim_comparison_authorized


def test_invalid_strict_option_is_rejected():
    with pytest.raises(ValueError, match="boolean"):
        NaturalLanguagePerfumeryAI(require_full_profile_match="false")


def test_final_draw_count_and_report_use_the_same_profile(tmp_path):
    with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        result = ai.create_recipe("green aromatic woody", as_of=date(2026, 9, 5))
    search = result.full_profile_assessment["search"]
    assert search["screening_draws"] == 64
    assert search["final_comparison_draws"] == result.scientific_monte_carlo_draws == 200
    assert search["selected_score"] == pytest.approx(result.calculated_profile_similarity, abs=1e-9)
    assert search["selected_score"] + 1e-8 >= search["baseline_score"]


def test_final_recheck_restores_baseline_if_screening_gain_disappears(tmp_path, monkeypatch):
    # Inject a counterexample into final-draw predictions, not into public data.
    with NaturalLanguagePerfumeryAI(corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        original = ai.temporal_simulator.evaluate
        final_calls = []

        def evaluate(*args, **kwargs):
            twin = original(*args, **kwargs)
            if kwargs["draws"] == 200:
                final_calls.append(args[0])
                if len(final_calls) == 1:
                    return replace(twin, temporal_points=tuple(
                        replace(point, scent_profile={"earthy": 1.0}) for point in twin.temporal_points
                    ))
            return twin

        monkeypatch.setattr(ai.temporal_simulator, "evaluate", evaluate)
        result = ai.create_recipe("clean fresh citrus woody", as_of=date(2026, 9, 5))
    search = result.full_profile_assessment["search"]
    from fragrance_ai.recommender.quality import formula_fingerprint
    pool = search["full_pool_search"]
    # Preserve the original two-candidate V7 recheck invariant, and separately
    # account for V8's full-pool candidates evaluated after that recheck.
    assert len(final_calls) == 2 + sum("full_profile_score" in row for row in pool["attempts"])
    assert search["screening_selected_score"] > search["screening_baseline_score"]
    assert pool["baseline_v7_formula_id"] == formula_fingerprint(final_calls[1])
    assert pool["baseline_v7_score"] == search["baseline_score"]
    assert search["selected_score"] + 1e-8 >= search["baseline_score"]
    assert result.calculated_profile_similarity == pytest.approx(search["selected_score"])
    assert pool["selected_formula_id"] == formula_fingerprint(result.closest_candidate)
    if not pool["candidate_changed"]:
        assert not search["candidate_changed"]
        assert result.closest_candidate == final_calls[1]


def test_authenticated_http_response_keeps_new_score_fields_and_strict_gate(tmp_path):
    from fragrance_ai.api import TokenAuthorizer, create_app
    from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog
    from tests._http_client import TestClient

    authorizer = TokenAuthorizer.from_plaintext({"local-test-token": ("test-operator", "formulator")})
    audit = AppendOnlyAuditLog(tmp_path / "audit.db")
    app = create_app(
        ai_factory=lambda: NaturalLanguagePerfumeryAI(
            corpus=HistoricalReferenceCorpus(tmp_path / "absent.db"), require_full_profile_match=True,
        ),
        authorizer=authorizer, audit_log=audit,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/recipes", json={"brief": "clean fresh citrus woody"},
                headers={"Authorization": "Bearer local-test-token"},
            )
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["similarity_kind"] == PROFILE_MATCH_KIND
        assert result["score_contract"]["strict_full_profile_gate"]
        assert result["calculated_profile_similarity"] == result["full_profile_assessment"]["score"]
        assert result["legacy_preference_score"] >= 90
        assert result["recipe"] == [] and not result["full_profile_target_met"]
        assert result["actual_olfactory_similarity_score"] is None
        assert not result["human_similarity_90_claim_authorized"]
    finally:
        audit.close()


def test_manual_formula_edit_invalidates_previous_full_profile_scores(tmp_path, generated):
    from fragrance_ai.platform.store import SqliteWorkspaceStore
    from fragrance_ai.platform.workspace import FormulaWorkspaceService

    store = SqliteWorkspaceStore(tmp_path / "manual.db")
    try:
        service = FormulaWorkspaceService(store=store, ai_factory=lambda: None)
        original = generated.to_dict()
        lines = [{"ingredient_id": line.ingredient_id, "concentrate_percent": line.concentrate_percent}
                 for line in generated.recipe]
        receiver = next(line for line in lines
                        if line["concentrate_percent"] + .01 < service._ingredients[line["ingredient_id"]].as_supplied_cap_percent())
        donor = next(line for line in lines if line is not receiver and line["concentrate_percent"] > .02)
        receiver["concentrate_percent"] += .01
        donor["concentrate_percent"] -= .01
        manual = service._manual_payload(original, lines)
        assert manual["formula_id"] != original["formula_id"]
        assert manual["status"] == "draft_manual_edit"
        assert manual["calculated_profile_similarity"] is None
        assert manual["legacy_preference_score"] is None
        assert not manual["full_profile_target_met"]
        assert manual["full_profile_assessment"]["score"] is None
        assert not manual["score_contract"]["assessment_valid"]
        assert generated.to_dict() == original
    finally:
        store.close()


def test_actual_queued_generation_respects_configured_strict_gate(tmp_path):
    from fragrance_ai.platform.store import SqliteWorkspaceStore
    from fragrance_ai.platform.worker import run_worker

    store = SqliteWorkspaceStore(tmp_path / "queue.db")
    try:
        project = store.create_project(tenant_id="tenant-a", name="Test", description="", actor_id="chemist")
        job = store.enqueue_job(
            tenant_id="tenant-a", kind="recipe.generate", actor_id="chemist",
            payload={"project_id": project.project_id, "brief": "clean fresh citrus woody", "constraints": {}},
        )
        count = run_worker(
            store=store, worker_id="strict-test-worker", once=True,
            ai_factory=lambda: NaturalLanguagePerfumeryAI(
                corpus=HistoricalReferenceCorpus(tmp_path / "absent.db"), require_full_profile_match=True,
            ),
        )
        result = store.get_job(tenant_id="tenant-a", job_id=job.job_id)
        assert count == 1 and result.status == "succeeded"
        payload = result.result["result"]
        assert result.result["workspace_formula"] is None
        assert payload["recipe"] == []
        assert payload["closest_candidate"]
        assert not payload["full_profile_target_met"]
        assert payload["score_contract"]["strict_full_profile_gate"]
        assert store.list_formulas(tenant_id="tenant-a", project_id=project.project_id) == []
    finally:
        store.close()
