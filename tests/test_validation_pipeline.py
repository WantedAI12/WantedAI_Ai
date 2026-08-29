from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date

import pytest

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.manufacturing import REQUIRED_STABILITY_TESTS
from fragrance_ai.recommender.odor_profiles import OdorProfileStore
from fragrance_ai.recommender.quality import QualityEvidenceStore
from fragrance_ai.recommender.sensory import SensoryEvaluationStore
from fragrance_ai.recommender.simulation import SimulatedSensoryEngine
from fragrance_ai.recommender.sensory import CalibrationArtifact
from fragrance_ai.recommender.science import (
    MolecularProperties,
    ScientificPropertyStore,
    TemporalMixtureSimulator,
)
from fragrance_ai.recommender.physsim import ConcentrationAwarePhysSim
from fragrance_ai.recommender.release import CommercialReleaseStore, RegulatorySignoff
from fragrance_ai.recommender.data_hub import NonHumanDataHub
from fragrance_ai.recommender.epa_comptox import EPACompToxStore
from fragrance_ai.recommender.supplier import SupplierMaterial, SupplierRegistry


AS_OF = date(2026, 7, 11)


def test_nonhuman_hub_connects_lineage_and_quarantines_synthetic_data():
    hub = NonHumanDataHub()
    stats = hub.stats()
    assert stats["nonhuman_data_sources"] >= 60
    assert stats["nonhuman_material_observations"] >= 6000
    assert stats["synthetic_training_assets_quarantined"] >= 20
    assert stats["human_validation_assets_excluded"] >= 1
    assert stats["epa_open_data_sources"] == 5
    assert stats["epa_source_rows_connected"] >= 28_000
    assert stats["epa_catalog_ingredients_matched"] >= 38
    assert hub.material_evidence("dihydromyrcenol")
    allowed_rows = hub.additional_reference_formulas()
    assert allowed_rows
    assert all(
        source_id != "ai_perfume_synthetic_recipes" for source_id, *_ in allowed_rows
    )


def test_epa_comptox_bulk_extract_is_catalog_filtered_and_fully_fingerprinted():
    store = EPACompToxStore()
    stats = store.stats()
    assert stats["epa_source_files"] == 5
    assert stats["epa_catalog_ingredients_matched"] == 38
    assert stats["epa_cpdat_rows"] >= 25_000
    assert stats["epa_toxref_studies"] >= 10
    assert stats["epa_toxref_pods"] >= 25
    assert stats["epa_toxval_rows"] >= 2_500
    summary = store.material_summary("vanillin")
    assert summary["dtxsid"]
    assert summary["cpdat_rows"] > 0
    assert summary["toxval_rows"] > 0
    metadata = dict(store.connection.execute("SELECT key, value FROM metadata"))
    assert metadata["human_sensory_data_included"] == "0"
    assert metadata["toxval_workbooks_scanned"] == "446"
    assert metadata["toxval_workbooks_failed"] == "0"
    assert metadata["toxval_csv_files_scanned"] == "2"
    assert metadata["toxval_csv_files_failed"] == "0"
    assert metadata["toxval_xls_files_scanned"] == "1"
    assert metadata["toxval_xls_files_unparsed"] == "0"
    sources = list(
        store.connection.execute(
            "SELECT sha256, materialization_status FROM source_files"
        )
    )
    assert all(len(checksum) == 64 for checksum, _status in sources)
    assert all("pending" not in status for _checksum, status in sources)
    store.close()


def test_nonhuman_hub_contains_no_human_validation_paths():
    hub = NonHumanDataHub()
    paths = [
        str(row[0]).casefold()
        for row in hub.connection.execute("SELECT local_path FROM data_sources")
    ]
    forbidden = ("human", "sensory", "expert", "feedback", "validation", "panel")
    assert not [path for path in paths if any(term in path for term in forbidden)]


def test_recipe_automatically_uses_nonhuman_reference_hub():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    assert result.catalog_stats["hub_reference_formulas_used"] > 0
    assert result.catalog_stats["nonhuman_data_sources"] >= 60
    assert result.catalog_stats["synthetic_training_assets_quarantined"] >= 20
    assert not result.safety.regulatory_data_complete


def qualified_offer(ingredient_id: str = "dihydromyrcenol") -> SupplierMaterial:
    return SupplierMaterial(
        ingredient_id=ingredient_id,
        supplier="Verified Supplier",
        sku="SKU-1",
        cas_number="18479-58-8",
        price_per_kg=25.0,
        currency="USD",
        moq_kg=1.0,
        in_stock=True,
        lead_time_days=7,
        density_g_ml=0.85,
        active_strength_percent=100.0,
        carrier=None,
        regions=("EU",),
        ifra_amendment="51",
        ifra_certificate_valid_until="2027-07-11",
        sds_valid_until="2027-07-11",
        coa_available=True,
        allergen_statement_valid_until="2027-07-11",
        allergen_fractions={"Linalool": 0.0},
        lot_number="LOT-1",
    )


def test_supplier_offer_qualification_and_overlay():
    catalog = IngredientCatalog.load_builtin()
    ingredient = catalog.lookup("Dihydromyrcenol")
    assert ingredient is not None
    registry = SupplierRegistry([qualified_offer()])
    assessment = registry.best_assessment(ingredient, RecipeConstraints(), AS_OF)
    assert assessment.qualified
    assert assessment.offer is not None
    overlaid = registry.overlay(ingredient, assessment.offer)
    assert overlaid.price_per_kg == 25.0
    assert overlaid.density_g_ml == 0.85
    assert overlaid.data_source.startswith("supplier:")


def test_observed_odor_profiles_overlay_only_after_replicates():
    store = OdorProfileStore()
    for panelist in ("P-01", "P-02", "P-03"):
        store.record(
            "dihydromyrcenol",
            panelist,
            "LOT-1",
            10.0,
            {"fresh": 0.95, "clean": 0.9, "citrus": 0.4},
            "LAB-ODOR-001",
            AS_OF,
        )
    catalog = IngredientCatalog.load_builtin()
    overlays = store.overlays()
    assert overlays["dihydromyrcenol"].panelist_count == 3
    observed_catalog = store.apply_to_catalog(catalog)
    observed = observed_catalog.lookup("Dihydromyrcenol")
    assert observed is not None
    assert observed.data_source.startswith("odor-observed:")
    assert observed.profile["fresh"] > observed.profile.get("woody", 0.0)
    assert store.stats(catalog)["odor_profile_coverage_percent"] > 0


def test_duplicate_odor_observation_is_rejected():
    store = OdorProfileStore()
    arguments = (
        "dihydromyrcenol",
        "P-01",
        "LOT-1",
        10.0,
        {"fresh": 0.9, "clean": 0.8},
        "LAB-1",
        AS_OF,
    )
    store.record(*arguments)
    with pytest.raises(sqlite3.IntegrityError):
        store.record(*arguments)


def test_expired_supplier_documents_fail_closed():
    offer = qualified_offer()
    expired = SupplierMaterial(**{**offer.__dict__, "sds_valid_until": "2026-01-01"})
    registry = SupplierRegistry([expired])
    ingredient = IngredientCatalog.load_builtin().lookup("Dihydromyrcenol")
    assessment = registry.best_assessment(ingredient, RecipeConstraints(), AS_OF)
    assert not assessment.qualified
    assert "current SDS" in assessment.missing_documents


def test_qualified_request_without_real_supplier_documents_is_blocked():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(validation_level="qualified"),
        as_of=AS_OF,
    )
    assert result.status == "no_safe_match"
    assert result.recipe == []
    assert result.rejected_candidate_counts["missing_supplier_offer"] > 0


def test_commercial_request_without_external_evidence_is_fail_closed():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디, 달지 않게",
        RecipeConstraints(
            validation_level="commercial",
            require_simulation_pass=False,
        ),
        as_of=AS_OF,
    )
    assert result.status == "no_safe_match"
    assert result.recipe == []
    assert not result.safety.manufacturing_ready
    assert result.actual_olfactory_similarity_score is None
    assert result.actual_olfactory_lower_bound_95 is None


def test_manufacturing_plan_is_mass_balanced_and_does_not_invent_density():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스에 은은한 우디 향, 달지 않게",
        RecipeConstraints(
            finished_batch_mass_g=100.0, product_concentration_percent=15.0
        ),
        as_of=AS_OF,
    )
    plan = result.manufacturing_plan
    assert plan is not None
    assert plan.fragrance_concentrate_mass_g == 15.0
    assert abs(plan.total_as_supplied_material_mass_g - 15.0) < 0.001
    assert plan.product_base_mass_g == 85.0
    assert plan.stability_status == "not_tested"
    assert all(line.estimated_volume_ml is None for line in plan.lines)
    assert plan.warnings


def test_simulated_diagnostic_does_not_pollute_human_evidence_or_approve():
    store = SensoryEvaluationStore()
    result = NaturalLanguagePerfumeryAI(sensory_store=store).create_recipe(
        "깨끗하고 시원한 시트러스에 은은한 우디 향, 달지 않게",
        constraints=RecipeConstraints(target_similarity=80.0),
        as_of=AS_OF,
    )
    assert result.status == "prototype_ready"
    assert result.simulated_similarity_score >= 90.0
    assert result.simulation_status == "diagnostic_only"
    assert result.simulation_confidence == "unvalidated_text_target_diagnostic"
    assert not result.simulation_only_approved
    assert result.olfactory_validation_status == "abstained_no_evidenced_target"
    assert result.temporal_similarity_p05 >= 90.0
    assert result.model_applicability_percent < 55.0
    assert not result.scientific_model_domain_passed
    assert "synthetic_draws_are_not_human_panel_records" in result.simulation_flags
    # Simulation is intentionally not a human evaluation row.
    evidence = store.formula_evidence(result.formula_id)
    assert evidence.status == "not_tested"


def test_simulation_is_seed_reproducible_and_has_minimum_draws():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스에 은은한 우디 향, 달지 않게",
        as_of=AS_OF,
    )
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    engine = SimulatedSensoryEngine()
    first = engine.evaluate(
        result.closest_candidate,
        ingredients,
        result.brief,
        ai.corpus,
        draws=100,
        seed=123,
    )
    second = engine.evaluate(
        result.closest_candidate,
        ingredients,
        result.brief,
        ai.corpus,
        draws=100,
        seed=123,
    )
    assert first == second
    with pytest.raises(ValueError):
        engine.evaluate(
            result.closest_candidate,
            ingredients,
            result.brief,
            ai.corpus,
            draws=10,
        )
    assert first.status == "diagnostic_only"


def test_unsealed_calibration_cannot_inflate_simulation():
    artifact = CalibrationArtifact(
        slope=0.0,
        intercept=100.0,
        observations=999,
        unique_formulas=99,
        holdout_mae=0.0,
        status="validated",
        trained_at="2026-07-13T00:00:00+00:00",
        data_fingerprint="fake",
    )
    assert not artifact.is_trusted()
    ai = NaturalLanguagePerfumeryAI(calibration=artifact)
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    assert result.simulation_confidence in {
        "physics_informed_nonhuman_proxy",
        "insufficient_nonhuman_evidence",
        "unvalidated_text_target_diagnostic",
    }
    assert result.simulation_confidence != "calibrated_human_proxy"
    assert result.sensory_similarity_score is None


def test_parser_handles_korean_negation_emphasis_and_numeric_constraints():
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    brief = parser.parse(
        "아주 시원한 시트러스 중심, 스모키하지 않게, 원료 kg당 최대 120, 100그램 배치, 향료 농도 12%"
    )
    assert {"fresh", "citrus"}.issubset(brief.desired_dimensions)
    assert "smoky" in brief.avoided_dimensions
    assert brief.constraints.max_ingredient_price_per_kg == 120.0
    assert brief.constraints.finished_batch_mass_g == 100.0
    assert brief.constraints.product_concentration_percent == 12.0


def test_direct_blind_sensory_rows_remain_unverified_audit_data():
    store = SensoryEvaluationStore()
    formula_a = "sha256:" + "a" * 64
    formula_b = "sha256:" + "b" * 64
    for formula_id, predicted in ((formula_a, 94.0), (formula_b, 80.0)):
        store.register_formula(formula_id, "clean citrus", predicted, [])
    mapping = store.create_study(
        "clean citrus", [formula_a, formula_b], "PROTO-001", seed=42
    )
    study_id = mapping.pop("study_id")
    code_a = next(
        code for code, formula_id in mapping.items() if formula_id == formula_a
    )
    for index in range(12):
        store.record_evaluation(
            study_id,
            code_a,
            f"P-{index:02d}",
            95.0,
            80.0,
            expert=index < 3,
        )
    evidence = store.formula_evidence(formula_a, target_similarity=90, min_panelists=12)
    assert not evidence.passed
    assert evidence.status == "legacy_unverified"
    assert evidence.unique_panelists == 0
    assert evidence.lower_confidence_bound_95 is None
    with pytest.raises(ValueError):
        # Invalid score must be rejected before it can pollute the evidence store.
        store.record_evaluation(study_id, code_a, "bad", 101.0, 50.0)


def test_direct_rows_cannot_satisfy_the_expert_gate():
    store = SensoryEvaluationStore()
    formula_a = "sha256:" + "1" * 64
    formula_b = "sha256:" + "2" * 64
    store.register_formula(formula_a, "brief", 95, [])
    store.register_formula(formula_b, "brief", 80, [])
    mapping = store.create_study(
        "brief", [formula_a, formula_b], "PROTO-EXPERT", seed=3
    )
    study_id = mapping.pop("study_id")
    code = next(code for code, value in mapping.items() if value == formula_a)
    for index in range(12):
        store.record_evaluation(study_id, code, f"N-{index}", 95, 80, expert=False)
    evidence = store.formula_evidence(formula_a)
    assert not evidence.passed
    assert evidence.status == "legacy_unverified"
    assert evidence.expert_panelists == 0


def test_legacy_sensory_rows_are_not_promoted_to_human_evidence(tmp_path):
    db_path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """CREATE TABLE evaluations (
            study_id TEXT NOT NULL, blind_code TEXT NOT NULL, panelist_id TEXT NOT NULL,
            expert INTEGER NOT NULL, similarity REAL NOT NULL, liking REAL NOT NULL,
            dimension_json TEXT NOT NULL, defects_json TEXT NOT NULL, evaluated_at TEXT NOT NULL,
            PRIMARY KEY (study_id, blind_code, panelist_id))"""
        )
    store = SensoryEvaluationStore(db_path)
    formula_a = "sha256:" + "3" * 64
    formula_b = "sha256:" + "4" * 64
    store.register_formula(formula_a, "brief", 95, [])
    store.register_formula(formula_b, "brief", 80, [])
    mapping = store.create_study("brief", [formula_a, formula_b], "LEGACY", seed=2)
    study_id = mapping.pop("study_id")
    code = next(code for code, value in mapping.items() if value == formula_a)
    store.connection.execute(
        """INSERT INTO evaluations
        (study_id, blind_code, panelist_id, expert, similarity, liking,
         dimension_json, defects_json, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (study_id, code, "P-OLD", 1, 99, 80, "{}", "[]", "2026-01-01"),
    )
    store.connection.commit()
    evidence = store.formula_evidence(formula_a, min_panelists=1, min_experts=1)
    assert evidence.status == "legacy_unverified"


def test_temporal_scientific_twin_requires_complete_properties():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    store = ScientificPropertyStore()
    for index, line in enumerate(result.closest_candidate):
        store.upsert(
            MolecularProperties(
                ingredient_id=line.ingredient_id,
                cas_number=None,
                molecular_weight=150.0 + index,
                xlogp=2.0,
                tpsa=20.0,
                hbond_donors=0,
                hbond_acceptors=1,
                rotatable_bonds=3,
                complexity=120.0,
                vapor_pressure_pa_25c=1.0 + index,
                boiling_point_c=220.0,
                odor_threshold_ppm=0.1,
                source_ref=f"TEST-{index}",
                verified_on="2026-07-13",
            )
        )
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    twin = TemporalMixtureSimulator().evaluate(
        result.closest_candidate, ingredients, result.brief, store
    )
    assert twin.scientific_data_coverage_percent == 100.0
    assert twin.molecular_descriptor_coverage_percent == 100.0
    assert len(twin.temporal_points) == 5
    assert twin.status == "measured_property_twin"
    assert twin.vapor_pressure_coverage_percent == 100.0
    assert twin.odor_threshold_coverage_percent == 100.0
    assert twin.model_applicability_percent == 100.0
    assert twin.monte_carlo_draws == 256


def test_scientific_twin_fails_closed_outside_nonhuman_applicability():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    empty_store = ScientificPropertyStore()
    strict = RecipeConstraints(
        require_simulation_pass=False,
        simulation_min_applicability_percent=80.0,
    )
    strict_brief = ai.parser.parse("깨끗하고 시원한 시트러스 우디 향", strict)
    twin = TemporalMixtureSimulator().evaluate(
        result.closest_candidate,
        ingredients,
        strict_brief,
        empty_store,
        draws=64,
        seed=7,
    )
    assert not twin.simulation_only_approved
    assert twin.model_applicability_percent < 80.0
    assert "outside_configured_model_applicability" in twin.flags


def test_headspace_twin_is_seed_reproducible_with_uncertainty_bounds():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    first = ai.temporal_simulator.evaluate(
        result.closest_candidate,
        ingredients,
        result.brief,
        ai.scientific_store,
        draws=64,
        seed=99,
    )
    second = ai.temporal_simulator.evaluate(
        result.closest_candidate,
        ingredients,
        result.brief,
        ai.scientific_store,
        draws=64,
        seed=99,
    )
    assert first == second
    assert first.temporal_similarity_p05 <= first.temporal_similarity_mean
    assert first.temporal_similarity_mean <= first.temporal_similarity_p95
    assert all(
        point.similarity_p05 <= point.similarity <= point.similarity_p95
        for point in first.temporal_points
    )


def test_concentration_aware_physsim_is_permutation_invariant_and_weight_sensitive():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    lines = result.closest_candidate
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    engine = ConcentrationAwarePhysSim()
    identical = engine.compare(
        lines, list(reversed(lines)), ingredients, ai.scientific_store
    )
    assert identical.similarity == 100.0

    changed = list(lines)
    donor, receiver = changed[0], changed[1]
    delta = min(5.0, donor.concentrate_percent / 2.0)
    changed[0] = replace(
        donor,
        concentrate_percent=donor.concentrate_percent - delta,
        finished_product_percent=(
            donor.finished_product_percent
            - delta * result.brief.constraints.product_concentration_percent / 100.0
        ),
    )
    changed[1] = replace(
        receiver,
        concentrate_percent=receiver.concentrate_percent + delta,
        finished_product_percent=(
            receiver.finished_product_percent
            + delta * result.brief.constraints.product_concentration_percent / 100.0
        ),
    )
    concentration_shift = engine.compare(
        lines, changed, ingredients, ai.scientific_store
    )
    assert 0.0 <= concentration_shift.similarity < identical.similarity


def test_physsim_soft_core_is_finite_with_duplicate_particles_and_missing_properties():
    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        "깨끗한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False),
        as_of=AS_OF,
    )
    line = result.closest_candidate[0]
    duplicate = [
        line,
        replace(line, concentrate_percent=line.concentrate_percent / 2.0),
    ]
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    comparison = ConcentrationAwarePhysSim().compare(
        duplicate, duplicate, ingredients, ScientificPropertyStore()
    )
    assert comparison.similarity == 100.0
    assert comparison.status == "mostly_inferred"
    assert comparison.model_applicability_percent == 20.0


def test_physsim_is_exposed_as_a_nonhuman_recipe_ranking_component():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향, 달지 않게",
        as_of=AS_OF,
    )
    assert result.physsim_model_version.startswith(
        "concentration-headspace-physsim-core-1.1"
    )
    assert result.physsim_status == "target_unavailable"
    assert not result.physsim_comparison_authorized
    assert result.physsim_similarity_score == 0.0
    assert result.physsim_temporal_profile == []
    assert result.simulation_components["physsim_applied"] == 0.0
    assert "self_generated_target_scoring_disabled" in result.physsim_flags
    assert result.physsim_learned_r2_status == "target_unavailable"
    assert result.actual_olfactory_similarity_score is None


def test_physsim_target_prototype_respects_explicit_material_bans():
    result = NaturalLanguagePerfumeryAI().create_recipe(
        "깨끗하고 시원한 시트러스 우디 향",
        RecipeConstraints(require_simulation_pass=False, explicit_bans={"vanillin"}),
        as_of=AS_OF,
    )
    assert "vanillin" not in result.physsim_target_ingredient_ids


def test_scientific_transport_uses_molecular_descriptors():
    neutral = MolecularProperties(
        "x",
        None,
        150.0,
        2.0,
        0.0,
        0,
        0,
        0,
        1.0,
        None,
        None,
        None,
        "TEST",
        "2026-07-13",
    )
    heavy_polar = MolecularProperties(
        "x",
        None,
        450.0,
        7.0,
        180.0,
        0,
        0,
        0,
        1.0,
        None,
        None,
        None,
        "TEST",
        "2026-07-13",
    )
    simulator = TemporalMixtureSimulator()
    assert simulator._air_to_receptor_transport(
        heavy_polar
    ) < simulator._air_to_receptor_transport(neutral)


def test_scientific_csv_rejects_unknown_ids_before_mutation(tmp_path):
    csv_path = tmp_path / "properties.csv"
    csv_path.write_text(
        "ingredient_id,cas_number,molecular_weight,xlogp,tpsa,hbond_donors,hbond_acceptors,"
        "rotatable_bonds,complexity,vapor_pressure_pa_25c,boiling_point_c,odor_threshold_ppm,source_ref,verified_on\n"
        "unknown,,150,2,20,0,1,2,100,1,200,0.1,TEST,2026-07-13\n",
        encoding="utf-8",
    )
    store = ScientificPropertyStore()
    with pytest.raises(ValueError, match="unknown ingredient"):
        store.import_csv(csv_path, allowed_ingredient_ids={"known"})
    assert store.stats()["scientific_property_records"] == 0


@pytest.mark.parametrize(
    "invalid_row",
    [
        "known,,nan,2,20,0,1,2,100,1,200,0.1,TEST,2026-07-13",
        "known,,150,2,20,0,1,2,100,1,200,nan,TEST,2026-07-13",
    ],
)
def test_scientific_csv_rejects_nonfinite_values_before_mutation(
    tmp_path, invalid_row
):
    csv_path = tmp_path / "nonfinite-properties.csv"
    csv_path.write_text(
        "ingredient_id,cas_number,molecular_weight,xlogp,tpsa,hbond_donors,hbond_acceptors,"
        "rotatable_bonds,complexity,vapor_pressure_pa_25c,boiling_point_c,odor_threshold_ppm,source_ref,verified_on\n"
        + invalid_row
        + "\n",
        encoding="utf-8",
    )
    store = ScientificPropertyStore()
    with pytest.raises(ValueError, match="finite"):
        store.import_csv(csv_path, allowed_ingredient_ids={"known"})
    assert store.stats()["scientific_property_records"] == 0


def test_legacy_release_signoff_cannot_approve_any_formula():
    store = CommercialReleaseStore()
    formula_id = "sha256:" + "a" * 64
    store.record(
        RegulatorySignoff(
            formula_id,
            "EU",
            "Responsible Person",
            "Independent Regulatory Lab",
            "2026-07-13",
            "2027-07-13",
            "EU-SAFETY-REPORT-001",
            "b" * 64,
        )
    )
    assert not store.assess(formula_id, "EU", date(2026, 7, 13)).passed
    assert (
        store.assess(formula_id, "EU", date(2026, 7, 13)).status
        == "legacy_formula_fingerprint_not_approvable"
    )
    assert not store.assess("sha256:" + "c" * 64, "EU", date(2026, 7, 13)).passed


def test_human_sensory_csv_bridge_rejects_synthetic_rows(tmp_path):
    store = SensoryEvaluationStore()
    formula_a = "sha256:" + "d" * 64
    formula_b = "sha256:" + "e" * 64
    store.register_formula(formula_a, "brief", 95, [])
    store.register_formula(formula_b, "brief", 80, [])
    mapping = store.create_study("brief", [formula_a, formula_b], "P-001", seed=1)
    study_id = mapping.pop("study_id")
    code = next(iter(mapping))
    path = tmp_path / "human.csv"
    path.write_text(
        "study_id,blind_code,panelist_id,expert,similarity,liking,dimension_json,defects_json,source,evidence_ref\n"
        f'{study_id},{code},P-01,false,95,80,"{{}}","[]",human,REPORT-001\n',
        encoding="utf-8",
    )
    assert store.import_human_csv(path) == 1
    legacy = store.formula_evidence(mapping[code])
    assert legacy.unique_panelists == 0
    assert legacy.status == "legacy_unverified"
    assert not legacy.passed
    synthetic = tmp_path / "synthetic.csv"
    synthetic.write_text(
        path.read_text(encoding="utf-8").replace(",human,", ",simulation,"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        store.import_human_csv(synthetic)


def test_unsigned_quality_rows_remain_legacy_even_when_every_test_is_present():
    store = QualityEvidenceStore()
    formula_id = "sha256:" + "c" * 64
    assert not store.assess(formula_id).passed
    for test_name in REQUIRED_STABILITY_TESTS:
        store.record(
            formula_id,
            test_name,
            "pass",
            AS_OF,
            "STAB-PROTOCOL-1",
            f"REPORT-{test_name}",
        )
    assessment = store.assess(formula_id)
    assert not assessment.passed
    assert assessment.status == "legacy_unverified"


def test_sensory_calibrator_ignores_unsigned_direct_rows():
    store = SensoryEvaluationStore()
    formula_ids: list[str] = []
    holdout_seen = False
    candidate = 0
    while len(formula_ids) < 6:
        formula_id = "sha256:" + hashlib.sha256(str(candidate).encode()).hexdigest()
        is_holdout = int(hashlib.sha256(formula_id.encode()).hexdigest(), 16) % 5 == 0
        if is_holdout or len(formula_ids) < 5:
            formula_ids.append(formula_id)
            holdout_seen = holdout_seen or is_holdout
        candidate += 1
        if len(formula_ids) == 6 and not holdout_seen:
            formula_ids.pop()
    for index, formula_id in enumerate(formula_ids):
        predicted = 70.0 + index * 4.0
        store.register_formula(formula_id, "calibration", predicted, [])
    mapping = store.create_study("calibration", formula_ids, "CAL-001", seed=1)
    study_id = mapping.pop("study_id")
    for code, formula_id in mapping.items():
        predicted = 70.0 + formula_ids.index(formula_id) * 4.0
        for panelist in range(6):
            store.record_evaluation(study_id, code, f"P-{panelist}", predicted, 75.0)
    artifact = store.fit_calibrator(minimum_observations=30, minimum_formulas=6)
    assert artifact.status == "insufficient_verified_data"
    assert artifact.observations == 0
    assert artifact.holdout_mae is None
