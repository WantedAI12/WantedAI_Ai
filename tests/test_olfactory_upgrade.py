from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.human_calibration import HumanMixtureCalibration
from fragrance_ai.recommender.models import RecipeConstraints, RecipeLine
from fragrance_ai.recommender.numpy_r2 import NumpyR2Model
from fragrance_ai.recommender.physsim_checkpoint import FrozenR2PhysSim
from fragrance_ai.recommender.service import _olfactory_validation_status
from fragrance_ai.recommender.simulation import SimulatedSensoryEngine


ROOT = Path(__file__).resolve().parents[1]


def _line(identifier: str, percent: float) -> RecipeLine:
    return RecipeLine(
        ingredient_id=identifier,
        name=identifier,
        pyramid="heart",
        concentrate_percent=percent,
        finished_product_percent=percent,
        volume_ml_for_batch=None,
        price_per_kg=1.0,
        availability=1.0,
        risk_tier=0,
        reason="test",
    )


def test_actual_human_calibrator_is_study_bound_and_rejects_recipe_projection():
    calibration = HumanMixtureCalibration()
    artifact = json.loads(
        (ROOT / "fragrance_ai" / "data" / "human_mixture_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    result = calibration.predict_study_endpoint(
        0.5,
        study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
        source_report_sha256=artifact["source"]["blind_report_sha256"],
        stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
        components_per_mixture=20,
        right_dilution=0.5,
        wrong_dilutions=(0.25, 1.0),
    )
    assert result.status == "calibrated_registered_study_endpoint_protocol_aware"
    assert result.discrimination_probability is not None
    assert result.lower_95 <= result.discrimination_probability <= result.upper_95
    assert result.applicability_percent == 100.0
    assert not result.similarity_90_claim_authorized

    candidate = [_line(f"molecule-{index}", 10.0) for index in range(10)]
    target = [_line(f"molecule-{index}", 10.0) for index in range(5)] + [
        _line(f"target-{index}", 10.0) for index in range(5)
    ]
    formula_result = calibration.compare(
        candidate,
        target,
        matrix_id="bushdid_2014_equal_presence_molecular_mixture",
        product_concentration_percent=100.0,
    )
    assert formula_result.status == "abstained_formula_endpoint_not_supported"
    assert formula_result.discrimination_probability is None

    mismatch = calibration.predict_study_endpoint(
        0.5,
        study_protocol_id="unregistered",
        source_report_sha256=artifact["source"]["blind_report_sha256"],
    )
    assert mismatch.status == "abstained_study_protocol_mismatch"

    incomplete = calibration.predict_study_endpoint(
        0.5,
        study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
        source_report_sha256=artifact["source"]["blind_report_sha256"],
        stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
    )
    assert incomplete.status == "abstained_required_stimulus_protocol_features_missing"
    assert incomplete.discrimination_probability is None

    with pytest.raises(ValueError, match="finite integer"):
        calibration.predict_study_endpoint(
            0.5,
            study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
            source_report_sha256=artifact["source"]["blind_report_sha256"],
            stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
            components_per_mixture=20.5,
            right_dilution=0.5,
            wrong_dilutions=(0.25, 1.0),
        )

    dilution_mismatch = calibration.predict_study_endpoint(
        0.5,
        study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
        source_report_sha256=artifact["source"]["blind_report_sha256"],
        stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
        components_per_mixture=20,
        right_dilution=0.50001,
        wrong_dilutions=(0.25, 1.0),
    )
    assert dilution_mismatch.status == "abstained_stimulus_protocol_outside_scope"

    with pytest.raises(ValueError, match="finite integer"):
        calibration.predict_study_endpoint(
            0.5,
            study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
            source_report_sha256=artifact["source"]["blind_report_sha256"],
            stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
            components_per_mixture=True,
            right_dilution=0.5,
            wrong_dilutions=(0.25, 1.0),
        )


def test_human_calibration_rejects_an_unrecognized_source_artifact(tmp_path):
    artifact_path = ROOT / "fragrance_ai" / "data" / "human_mixture_calibration.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["source"]["blind_report_sha256"] = "a" * 64
    tampered = tmp_path / "tampered_human_calibration.json"
    tampered.write_text(json.dumps(artifact), encoding="utf-8")

    calibration = HumanMixtureCalibration(tampered)
    result = calibration.predict_study_endpoint(
        0.5,
        study_protocol_id="bushdid_2014_supplemental_3afc_protocol",
        source_report_sha256="a" * 64,
        stimulus_protocol_sha256=artifact["source"]["stimulus_protocol_sha256"],
        components_per_mixture=20,
        right_dilution=0.5,
        wrong_dilutions=(0.25, 1.0),
    )
    assert result.status == "calibration_unavailable"
    assert result.discrimination_probability is None
    assert any("source hashes are not authorized" in flag for flag in result.flags)


def test_human_panel_status_never_renames_a_lower_target_as_90_percent():
    evidence = SimpleNamespace(
        passed=True,
        lower_confidence_bound_95=86.0,
        status="verified_passed",
        unique_panelists=12,
    )
    assert (
        _olfactory_validation_status(evidence, "not_tested", 80.0)
        == "human_validated_requested_target"
    )
    evidence.lower_confidence_bound_95 = 92.0
    assert (
        _olfactory_validation_status(evidence, "not_tested", 90.0)
        == "human_validated_90"
    )


def test_protocol_aware_human_model_improves_rank_but_does_not_authorize_90_claim():
    audit = json.loads(
        (ROOT / "benchmarks" / "bushdid_human_protocol_calibration_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["protocol_aware_rank_spearman"] > audit["component_overlap_spearman"]
    assert audit["protocol_aware_rank_spearman"] == pytest.approx(0.6460129905022272)
    assert audit["paired_stimulus_bootstrap_95_interval"][0] > 0.0
    assert audit["prospective_external_validation"] is False
    assert audit["human_similarity_90_claim_authorized"] is False


def test_text_only_request_abstains_and_never_builds_a_self_target():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        as_of=date(2026, 7, 28),
    )
    assert result.status == "prototype_ready"
    assert result.physsim_status == "target_unavailable"
    assert result.physsim_similarity_score == 0.0
    assert "self_generated_target_scoring_disabled" in result.physsim_flags
    assert "self_generated_text_target_prototype" not in result.physsim_flags
    assert result.perceptual_prediction_status == "abstained_no_evidenced_target"


def test_evidenced_nonhuman_gate_requires_reference_physsim_to_reach_target():
    with NaturalLanguagePerfumeryAI() as ai:
        result = ai.create_recipe(
            "깨끗하고 시원한 시트러스 우디 향",
            RecipeConstraints(target_similarity=80.0),
            as_of=date(2026, 7, 28),
        )
        ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
        scientific = SimpleNamespace(
            temporal_similarity_mean=96.0,
            minimum_temporal_similarity=94.0,
            model_applicability_percent=100.0,
            temporal_similarity_p05=95.0,
            temporal_similarity_p95=97.0,
            model_domain_passed=True,
            flags=(),
        )
        low_reference = SimpleNamespace(
            similarity=25.0,
            comparison_authorized=True,
            model_applicability_percent=100.0,
            learned_r2_similarity=None,
            learned_r2_applied_weight=0.0,
            learned_r2_centered_score_adjustment=0.0,
            flags=(),
        )
        simulation = SimulatedSensoryEngine().evaluate(
            result.closest_candidate,
            ingredients,
            result.brief,
            ai.corpus,
            target=80.0,
            draws=64,
            scientific_twin=scientific,
            physsim=low_reference,
            target_evidenced=True,
        )
    assert simulation.p05 >= 80.0
    assert simulation.status == "evidenced_nonhuman_fail"
    assert "evidenced_reference_physsim_below_target" in simulation.flags


def test_parser_preserves_absolute_intensity_diffusion_and_texture():
    with NaturalLanguagePerfumeryAI() as ai:
        low = ai.parser.parse("살결 가까이 은은하고 투명한 클린 머스크")
        high = ai.parser.parse("확산력 강하고 진한 농밀 우디 앰버")
    assert low.absolute_intensity_target < high.absolute_intensity_target
    assert low.diffusion_target < high.diffusion_target
    assert low.texture_profile["transparent"] == 1.0
    assert high.texture_profile["dense"] == 1.0
    assert low.constraints.product_concentration_percent == 15.0
    assert high.constraints.product_concentration_percent == 15.0


def test_parser_request_overrides_do_not_mutate_reused_constraints():
    constraints = RecipeConstraints(product_concentration_percent=15.0)
    with NaturalLanguagePerfumeryAI() as ai:
        parsed = ai.parser.parse("시트러스 향, 향료 농도 22%", constraints)
    assert parsed.constraints.product_concentration_percent == 22.0
    assert constraints.product_concentration_percent == 15.0


def test_parser_negated_ingredient_does_not_mutate_reused_ban_set():
    constraints = RecipeConstraints(explicit_bans={"oakmoss"})
    with NaturalLanguagePerfumeryAI() as ai:
        parsed = ai.parser.parse(
            "clean citrus woody fragrance without vanillin", constraints
        )
    assert constraints.explicit_bans == {"oakmoss"}
    assert parsed.constraints.explicit_bans is not constraints.explicit_bans
    assert parsed.constraints.explicit_bans.issuperset(constraints.explicit_bans)
    assert len(parsed.constraints.explicit_bans) > len(constraints.explicit_bans)


@pytest.mark.parametrize(
    "constraints",
    [
        RecipeConstraints(simulation_draws=29),
        RecipeConstraints(simulation_draws=63),
        RecipeConstraints(surrogate_objective_weight=0.51),
        RecipeConstraints(minimum_dimension_material_strength=0.0),
        RecipeConstraints(physics_search_population=8),
        RecipeConstraints(require_catalog_dimension_support=1),
        RecipeConstraints(max_ingredients=3.5),
    ],
)
def test_invalid_scientific_controls_fail_before_formula_search(constraints):
    with NaturalLanguagePerfumeryAI() as ai:
        with pytest.raises(ValueError):
            ai.create_recipe(
                "깨끗하고 시원한 시트러스 우디 향",
                constraints,
                as_of=date(2026, 7, 28),
            )


def test_unsupported_safe_catalog_dimension_fails_closed():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "강한 스모키 레더 향",
        as_of=date(2026, 7, 28),
    )
    assert result.status == "no_safe_match"
    assert "smoky" in result.message or "leathery" in result.message


def test_frozen_r2_uses_portable_numpy_runtime():
    adapter = FrozenR2PhysSim()
    adapter._load()
    assert adapter._loaded is True
    assert adapter._models
    assert all(model.__class__.__name__ == "NumpyR2Model" for model in adapter._models)


def test_portable_r2_rejects_incomplete_state_contract():
    with pytest.raises(ValueError, match="state contract mismatch"):
        NumpyR2Model({"log_attraction": np.array(0.0)})


def test_catalog_capability_report_exposes_rank_and_missing_axes():
    catalog = IngredientCatalog.load_builtin()
    safe = [
        item
        for item in catalog.ingredients
        if item.formulation_ready
        and not item.blocked
        and item.risk_tier <= 1
        and item.price_per_kg <= 300.0
        and item.availability >= 0.75
        and item.rarity != "rare"
    ]
    report = catalog.capability_report(safe)
    assert report["profile_rank"] < report["profile_dimension_count"]
    assert {"smoky", "leathery"}.issubset(report["unsupported_dimensions"])
