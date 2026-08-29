from datetime import date

import pytest

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import (
    NaturalLanguageBriefParser,
    apply_relative_revision_profile,
)
from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus, IngredientCatalog
from fragrance_ai.recommender.evaluation import evaluate_benchmark


AS_OF = date(2026, 7, 11)


def test_korean_natural_language_parser():
    catalog = IngredientCatalog.load_builtin()
    brief = NaturalLanguageBriefParser(catalog).parse(
        "깨끗하고 시원한 시트러스에 은은한 우디 향, 달지 않게"
    )

    assert {"clean", "fresh", "citrus", "woody"}.issubset(brief.desired_dimensions)
    assert "gourmand" in brief.avoided_dimensions
    assert brief.intensity == "low"

    accord = NaturalLanguageBriefParser(catalog).parse("투명한 앰버 우드 머스크")
    assert {"amber", "woody", "musky"}.issubset(accord.desired_dimensions)


def test_relative_revision_preserves_context_and_changes_requested_dimensions():
    base = {"citrus": 0.45, "woody": 0.30, "gourmand": 0.15, "fresh": 0.10}
    revised, adjustments = apply_relative_revision_profile(
        base,
        "우디함을 조금 높이고 시트러스는 유지하되 단맛은 낮춰줘",
    )

    assert adjustments["woody"] > 1.0
    assert adjustments["gourmand"] < 1.0
    assert "citrus" not in adjustments
    assert revised["woody"] > base["woody"]
    assert revised["gourmand"] < base["gourmand"]
    assert sum(revised.values()) == pytest.approx(1.0)


def test_safe_affordable_recipe_exceeds_model_threshold():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스에 은은한 우디 향, 달지 않게",
        as_of=AS_OF,
    )

    assert result.status == "prototype_ready"
    assert result.similarity_score >= 90.0
    assert result.recipe
    assert abs(sum(line.concentrate_percent for line in result.recipe) - 100.0) < 0.01
    assert result.estimated_concentrate_cost_per_kg <= 180.0
    assert all(line.price_per_kg <= 300.0 for line in result.recipe)
    assert all(line.availability >= 0.75 for line in result.recipe)
    assert all(line.risk_tier <= 1 for line in result.recipe)
    assert result.realism_score >= 65.0
    assert result.realism_kind == "engineering_plausibility_not_sensory_accuracy"
    assert result.confidence == "heuristic_only"
    assert isinstance(result.historical_reference_matches, list)
    if result.historical_reference_matches:
        assert result.historical_reference_matches[0]["reference_only"] is True
    assert (
        result.reference_molecular_composition_status == "not_available_note_names_only"
    )
    assert "not verified" in result.reference_molecular_composition_claim_boundary
    assert result.physsim_learned_r2_neutral_similarity_percent > 0.0
    assert result.physsim_learned_r2_member_predictions == []
    assert result.physsim_learned_r2_similarity_score is None
    assert result.physsim_status == "target_unavailable"
    assert result.physsim_comparison_authorized is False
    assert result.reference_target_comparison_kind == "no_evidenced_target"
    assert result.physsim_learned_r2_member_disagreement_percent >= 0.0
    assert (
        result.physsim_learned_r2_prediction_interval_upper_percent
        >= result.physsim_learned_r2_prediction_interval_lower_percent
    )
    assert result.physsim_learned_r2_ensemble_manifest_sha256 == ""
    assert result.concentration_response_status == "validation_gated_diagnostic_only"
    assert result.concentration_response_similarity_score is None
    assert result.concentration_response_applied_weight == 0.0
    assert "independent_signed_artifact_authorization_missing" in result.physsim_flags
    assert (
        "concentration_response_independent_authorization_missing_weight_zero"
        not in result.physsim_flags
    )
    assert result.candidate_variants_evaluated >= 3
    assert result.physics_guided_search is True
    assert result.physics_search_objective > 0.0
    assert result.simulation_status == "diagnostic_only"
    assert result.olfactory_validation_status == "abstained_no_evidenced_target"
    assert result.human_similarity_90_claim_authorized is False


def test_green_aquatic_forest_brief_exceeds_threshold():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "비 온 뒤 숲처럼 맑고 그린하며 아쿠아틱한 향",
        as_of=AS_OF,
    )
    assert result.status == "prototype_ready"
    assert result.similarity_score >= 90.0


def test_rare_and_blocked_materials_are_never_used():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "천연 장미와 자스민, 오우드 같은 고급스러운 향이지만 희귀하거나 비싼 재료는 빼고",
        as_of=AS_OF,
    )
    prohibited = {
        "Rose Absolute",
        "Jasmine Absolute",
        "Oud Oil",
        "Lilial",
        "Lyral",
        "Musk Xylene",
        "Oakmoss Absolute",
        "Natural Civet",
        "Natural Castoreum",
    }
    assert not prohibited.intersection(line.name for line in result.closest_candidate)
    assert result.rejected_candidate_counts["blocked_or_prohibited"] >= 10


def test_gourmand_musk_recipe_uses_no_rare_animal_material():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "포근하고 깨끗한 바닐라 머스크에 살짝 달콤한 향",
        as_of=AS_OF,
    )
    names = {line.name for line in result.recipe}
    assert result.status == "prototype_ready"
    assert result.similarity_score >= 90.0
    assert "Natural Civet" not in names
    assert "Natural Ambergris" not in names


def test_threshold_is_fail_closed():
    constraints = RecipeConstraints(target_similarity=99.99)
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        constraints,
        as_of=AS_OF,
    )
    assert result.status == "no_safe_match"
    assert result.recipe == []
    assert result.closest_candidate


def test_stale_regulatory_policy_blocks_recipe():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        as_of=date(2026, 12, 1),
    )
    assert result.status == "no_safe_match"
    assert result.recipe == []
    assert result.safety.status == "blocked"
    assert any("검토 기한" in item for item in result.safety.violations)


def test_catalog_and_reference_corpus_are_reported(tmp_path):
    corpus = HistoricalReferenceCorpus(tmp_path / "excluded-reference-corpus.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        stats = {**ai.catalog.stats(), **ai.corpus.stats()}
    assert stats["ifra_transparency_2025_reference_count"] == 3691
    assert stats["formulation_ready"] >= 25
    assert stats["reference_perfumes"] == 0
    assert stats["reference_note_rows"] == 0


def test_bundled_semantic_gate_benchmark():
    benchmark = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "brief_benchmark.json"
    )
    report = evaluate_benchmark(benchmark)
    assert report["semantic_gate_classification_accuracy"] == 100.0
    assert report["parser_dimension_recall"] == 100.0
    assert report["minimum_ready_similarity"] >= 90.0
    assert report["minimum_ready_realism"] >= 65.0
    assert report["blocked_material_leaks"] == []


def test_developer_holdout_semantic_and_safety_invariants():
    benchmark = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "holdout_benchmark.json"
    )
    report = evaluate_benchmark(benchmark)
    assert report["semantic_gate_classification_accuracy"] == 100.0
    assert report["parser_dimension_recall"] == 100.0
    assert report["parser_avoidance_recall"] == 100.0
    assert report["full_case_pass_rate"] == 100.0
    assert report["minimum_ready_similarity"] >= 90.0
    assert report["minimum_ready_realism"] >= 65.0
    assert report["blocked_material_leaks"] == []
    assert report["invariant_failures"] == []
