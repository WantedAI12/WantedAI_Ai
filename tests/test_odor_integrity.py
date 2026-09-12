"""Regression tests for invented odor data, stale approval and computed properties."""

from dataclasses import asdict, replace
from contextlib import closing
from datetime import date
import gzip
import hashlib
import json
import sqlite3

import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.platform.models import FormulaVersionRecord
from fragrance_ai.platform.store import SqliteWorkspaceStore
from fragrance_ai.platform.workspace import FormulaWorkspaceService
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog, HistoricalReferenceCorpus
from fragrance_ai.recommender.models import RecipeLine
from fragrance_ai.recommender.odor_integrity import (
    ODOR_INTEGRITY_VERSION, POSITIVE_ODOR_STATUS, REGISTRY_ODOR_SOURCE,
    assess_odor_assertions, legacy_registry_line_ids, quarantine_legacy_payload,
    registry_odor_rejection,
)
from fragrance_ai.recommender.registry_activation import (
    RegistryActivationReport, _calculated_structure_properties, _project_profile,
    load_runtime_catalog, write_runtime_catalog,
)
from fragrance_ai.recommender.safety import CandidateSafetyScreen, FormulaSafetyGate
from fragrance_ai.recommender.science import ScientificPropertyStore, TemporalMixtureSimulator


def _positive():
    smiles = "OCCc1ccccc1"
    props, version = _calculated_structure_properties(smiles)
    assertions = ("goodscents:rose",)
    status, profile = assess_odor_assertions(assertions)
    ref = json.dumps({"source_tag": "goodscents", "source_file_sha256": "a" * 64,
        "source_record_id": "test-source-row", "source_field": "Descriptors", "semantic_role": "odor",
        "descriptor": "rose", "conditions_status": "not_reported_in_source", "normalization_version": ODOR_INTEGRITY_VERSION})
    return replace(IngredientCatalog.load_builtin().ingredients[0],
        ingredient_id="registry_" + hashlib.sha256(smiles.encode()).hexdigest()[:24],
        name="Test phenethyl alcohol", aliases=(), cas_number="60-12-8", pyramid="heart",
        profile=dict(profile), risk_tier=2, formulation_ready=True, blocked=False, odor_impact=1.,
        max_concentrate_percent=100., data_source=REGISTRY_ODOR_SOURCE,
        odor_integrity_version=ODOR_INTEGRITY_VERSION, odor_evidence_status=status,
        odor_assertions=assertions, odor_evidence_refs=(ref,), odor_registry_sha256="b" * 64,
        structure_smiles=smiles, structure_properties=props, structure_properties_version=version)


def _legacy():
    return replace(_positive(), data_source="industrial-registry-public-descriptor-conditional-v2",
                   odor_integrity_version="", odor_assertions=(), odor_evidence_refs=())


@pytest.mark.parametrize("terms", [(), ("odorless",), ("white", "powder", "bland"), ("mystery",)])
def test_names_smiles_and_hashes_never_supply_missing_odor(terms):
    assert _project_profile(terms, "rose powdery musk OCCc1ccccc1") == {}


@pytest.mark.parametrize("assertions,status", [
    (("goodscents:odorless",), "reported_odorless"),
    (("goodscents:rose", "leffingwell:odorless"), "conflicting_odor_reports"),
    (("flavordb:rose", "flavordb:sweet"), "no_positive_odor_evidence"),
    (("goodscents:white", "goodscents:powder"), "no_positive_odor_evidence"),
    (("not-an-assertion",), "invalid_odor_assertions"),
])
def test_explicit_nonodor_evidence_cannot_be_projected(assertions, status):
    assert assess_odor_assertions(assertions) == (status, ())


def test_positive_descriptors_and_observations_keep_lineage():
    material = _positive()
    assert material.odor_evidence_status == POSITIVE_ODOR_STATUS
    assert registry_odor_rejection(material) is None
    observed = replace(material, data_source="odor-observed:test", profile={"floral": 1.})
    assert registry_odor_rejection(observed) is None
    assert registry_odor_rejection(replace(observed, odor_integrity_version=""))


@pytest.mark.parametrize("changes", [
    {"odor_evidence_refs": ()}, {"odor_registry_sha256": "wrong"},
    {"ingredient_id": "registry_wrong"}, {"registry_structural_alerts": ("review",)},
    {"profile": {"powdery": 1.}}, {"profile": {"rose": float("nan")}},
    {"structure_properties": {"molecular_weight": 122.16}},
    {"structure_properties_version": "measured"}, {"odor_evidence_status": "reported_odorless"},
])
def test_mutated_identity_profile_or_structure_is_rejected(changes):
    assert registry_odor_rejection(replace(_positive(), **changes)) is not None


def test_flavor_column_cannot_be_relabelled_as_an_odor_assertion():
    material = _positive()
    ref = json.loads(material.odor_evidence_refs[0])
    ref.update(source_tag="flavordb_odor", source_field="Flavor Percepts")
    invalid = replace(material, odor_assertions=("flavordb_odor:rose",), odor_evidence_refs=(json.dumps(ref),))
    assert registry_odor_rejection(invalid) == "registry_odor_lineage_missing"
    ref["source_field"] = "Odor Percepts"
    assert registry_odor_rejection(replace(invalid, odor_evidence_refs=(json.dumps(ref),))) is None


def test_fake_signed_source_does_not_exempt_registry_identity():
    relabelled = replace(_legacy(), data_source="signed-industrial-promotion:fake")
    assert registry_odor_rejection(relabelled) == "legacy_registry_odor_profile_unverified"
    payload = {"recipe": [{"ingredient_id": relabelled.ingredient_id, "data_source": relabelled.data_source}]}
    assert legacy_registry_line_ids(payload) == [relabelled.ingredient_id]
    assert quarantine_legacy_payload(payload)["recipe"] == []


@pytest.mark.parametrize("smiles,reason", [("O=[Si]=O", "nonorganic_structure"),
    ("CCO.[Na+]", "invalid_or_multicomponent_structure"), ("C[N+](C)(C)C", "charged_structure_requires_review")])
def test_no_fragment_selection_or_inorganic_descriptor_surrogate(smiles, reason):
    assert _calculated_structure_properties(smiles) == ({}, reason)


def test_old_runtime_snapshot_quarantined_even_with_safety_disabled(tmp_path):
    core = IngredientCatalog.load_builtin().ingredients[0]
    legacy = _legacy()
    report = RegistryActivationReport("b" * 64, 1, 0, 1, 1, 1, 2, 0, 0, 0)
    old = {"schema": "perfumery-runtime-catalog/v1", "wheel_sha256": "c" * 64, "registry_sha256": "b" * 64,
        "ingredients": [asdict(core), asdict(legacy)], "metadata": {}, "activation_report": asdict(report), "registry_stats": {}}
    path = tmp_path / "legacy.gz"
    path.write_bytes(gzip.compress(json.dumps(old).encode(), mtime=0))
    catalog, restored, _ = load_runtime_catalog(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_wheel_sha256="c" * 64, expected_registry_sha256="b" * 64)
    assert len(catalog.ingredients) == 2
    assert catalog.ingredients[1].profile == {}
    assert restored.active_odorant_candidates == 1
    assert restored.conditional_trace_candidates_active == 0
    assert restored.connected_catalog_rows == 2
    limits = RecipeConstraints(max_risk_tier=2, enable_registry_trace_candidates=True, experimental_disable_safety=True)
    brief = NaturalLanguageBriefParser(catalog).parse("rose scent", limits)
    accepted, rejected = CandidateSafetyScreen().screen(catalog, brief)
    assert [item.ingredient_id for item in accepted] == [core.ingredient_id]
    assert rejected["legacy_registry_odor_profile_unverified"] == 1


def test_v2_writer_refuses_unbound_active_profiles(tmp_path):
    invalid = replace(_positive(), profile={"powdery": 1.})
    report = RegistryActivationReport("b" * 64, 1, 0, 1, 1, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="unverified active odorant"):
        write_runtime_catalog(tmp_path / "bad.gz", IngredientCatalog([invalid]), report, {}, wheel_sha256="c" * 64)
    assert not (tmp_path / "bad.gz").exists()


def test_final_gate_and_direct_sdk_cannot_bypass_integrity(tmp_path):
    legacy = _legacy()
    line = RecipeLine(legacy.ingredient_id, legacy.name, "heart", 100, 15, None, 100, .9, 2, "test")
    limits = RecipeConstraints(max_risk_tier=2, enable_registry_trace_candidates=True, experimental_disable_safety=True)
    safety = FormulaSafetyGate().evaluate([line], {legacy.ingredient_id: legacy}, limits, as_of=date(2026, 9, 5))
    assert not safety.internal_gate_passed
    assert any("odor-integrity" in value for value in safety.violations)
    core = IngredientCatalog.load_builtin().ingredients[0]
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog([core, legacy]),
            corpus=HistoricalReferenceCorpus(tmp_path / "absent.db")) as ai:
        assert ai.catalog.ingredients[1].blocked
        assert ai.catalog.ingredients[1].profile == {}
        assert not legacy.blocked  # caller's original dataclass was not mutated


def _old_payload():
    line = {"ingredient_id": _legacy().ingredient_id, "concentrate_percent": 100., "data_source": _legacy().data_source}
    return {"status": "experimental_registry_candidate", "recipe": [line], "closest_candidate": [line],
        "brief": {"original_text": "powdery scent", "constraints": {"max_risk_tier": 2}},
        "achieved_profile": {"powdery": 1.}, "calculated_profile_similarity": 99., "full_profile_target_met": True,
        "simulation_p05": 98., "human_similarity_90_claim_authorized": True, "similarity_score": 99.,
        "safety": {"internal_gate_passed": True, "manufacturing_ready": False},
        "score_contract": {}, "manufacturing_plan": {"ready_for_lab_trial": True}}


def test_historical_api_view_invalidates_scores_without_rewriting_evidence():
    payload = _old_payload()
    original = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(original.encode()).hexdigest()
    record = FormulaVersionRecord("t", "p", "f", "v", 1, None, "generated", "", "a", "2026-09-06", payload, digest)
    view = record.to_dict()
    assert json.dumps(record.payload, sort_keys=True) == original
    assert view["content_sha256"] == digest
    assert view["content_sha256_scope"] == "immutable_stored_historical_payload"
    repaired = view["payload"]
    assert repaired["recipe"] == repaired["closest_candidate"] == []
    assert repaired["calculated_profile_similarity"] is None
    assert repaired["simulation_p05"] is None
    assert not repaired["full_profile_target_met"]
    assert not repaired["human_similarity_90_claim_authorized"]
    assert not repaired["safety"]["internal_gate_passed"]
    assert not repaired["manufacturing_plan"]["ready_for_lab_trial"]
    assert repaired["historical_odor_snapshot"] == payload
    assert record.to_dict(include_payload=False)["odor_data_integrity"] == "legacy_registry_profile_quarantined"
    assert quarantine_legacy_payload(repaired) == repaired
    current = {"recipe": [{"ingredient_id": _positive().ingredient_id, "data_source": REGISTRY_ODOR_SOURCE,
                            "odor_integrity_version": ODOR_INTEGRITY_VERSION}], "similarity_score": 91.}
    assert not legacy_registry_line_ids(current)
    assert quarantine_legacy_payload(current) == current


def test_persisted_versions_and_jobs_cannot_restore_old_odor_approval(tmp_path, request):
    store = SqliteWorkspaceStore(tmp_path / "workspace.db")
    request.addfinalizer(store.close)
    project = store.create_project(tenant_id="t", name="test", description="", actor_id="a")
    payload = _old_payload()
    formula = store.create_formula(tenant_id="t", project_id=project.project_id, name="old", kind="formula",
        payload=payload, actor_id="a", change_note="old")
    kwargs = dict(tenant_id="t", project_id=project.project_id, formula_id=formula.formula_id)
    service = FormulaWorkspaceService(store=store, ai_factory=lambda: pytest.fail("stale revision must not run inference"),
                                     catalog=IngredientCatalog([_legacy()]))
    assert service.catalog_payload()["ingredients"] == []
    with pytest.raises(ValueError, match="require regeneration"):
        service.revise_formula(**kwargs, actor_id="a", base_version_id=formula.latest_version.version_id, instruction="more powdery")
    with pytest.raises(ValueError, match="not formulation-ready"):
        service._manual_payload(payload, [{"ingredient_id": _legacy().ingredient_id, "concentrate_percent": 100}])
    comparison = service.compare_versions(**kwargs, left_version_id=formula.latest_version.version_id,
                                          right_version_id=formula.latest_version.version_id)
    assert comparison["metric_changes"] == comparison["profile_delta"] == {}
    assert comparison["odor_metric_comparison_status"] == "invalidated_legacy_odor_data"
    assert store.get_formula(**kwargs).latest_version.payload == payload
    job = store.enqueue_job(tenant_id="t", kind="recipe.generate", payload={}, actor_id="a")
    store.claim_job(worker_id="w", lease_seconds=30)
    store.complete_job(job_id=job.job_id, worker_id="w", result={"formula": {"latest_version": asdict(formula.latest_version)}, "result": payload})
    saved = store.get_job(tenant_id="t", job_id=job.job_id)
    assert saved.result["result"] == payload
    assert not saved.to_dict()["result"]["result"]["recipe"]
    version_view = saved.to_dict()["result"]["formula"]["latest_version"]
    assert not version_view["payload"]["recipe"]
    assert version_view["payload_view_transformed"]


@pytest.fixture
def no_physical_evidence_index(monkeypatch):
    # Isolate the structure-only branch. The shipped V56 index now has a
    # genuine exact-identity measurement for this fixture's molecule.
    from fragrance_ai.recommender import physical_evidence
    monkeypatch.setattr(physical_evidence, "_read_index", lambda *args: {"by_structure": {}, "by_cas": {}})


def test_calculated_structure_overlay_has_no_queries_or_fake_measurements(no_physical_evidence_index):
    material = _positive()
    with ScientificPropertyStore.load_builtin() as store:
        statements = []
        store.connection.set_trace_callback(statements.append)
        overlay = store.with_catalog_structures([material], {})
        assert statements == []
        store.connection.set_trace_callback(None)
    prop = overlay[material.ingredient_id]
    assert prop.molecular_weight == pytest.approx(122.167)
    assert prop.vapor_pressure_pa_25c is prop.boiling_point_c is prop.odor_threshold_ppm is None
    assert "calculated-not-measured" in prop.source_ref
    assert prop.verified_on == "structure_snapshot_not_physical_measurement"
    measured = replace(prop, vapor_pressure_pa_25c=1., odor_threshold_ppm=.01, source_ref="unit-test-measurement")
    original = {material.ingredient_id: measured}
    assert ScientificPropertyStore.with_catalog_structures([material], original)[material.ingredient_id] is measured
    assert ScientificPropertyStore.with_catalog_structures([_legacy()], {}) == {}
    assert original == {material.ingredient_id: measured}


def test_structure_coverage_does_not_become_measured_physics_coverage(no_physical_evidence_index):
    material = _positive()
    brief = NaturalLanguageBriefParser(IngredientCatalog([material])).parse("rose scent")
    line = RecipeLine(material.ingredient_id, material.name, "heart", 100, 15, None, 100, .9, 2, "test")
    props = ScientificPropertyStore.with_catalog_structures([material], {})
    result = TemporalMixtureSimulator().evaluate([line], {material.ingredient_id: material}, brief, props, draws=64)
    assert result.calculated_structure_coverage_percent == 100
    assert result.molecular_descriptor_coverage_percent == 100
    assert result.scientific_data_coverage_percent == result.vapor_pressure_coverage_percent == result.odor_threshold_coverage_percent == 0
    assert not result.model_domain_passed
    assert "calculated_molecular_descriptors_not_measured_vapor_pressure_or_odor_threshold" in result.flags


def test_structure_overlay_keeps_real_identity_joined_evidence_separate():
    material = _positive()
    original = {}
    prop = ScientificPropertyStore.with_catalog_structures([material], original)[material.ingredient_id]
    assert prop.vapor_pressure_pa_25c > 0
    assert prop.boiling_point_c > -273.15
    assert prop.odor_threshold_ppm > 0
    assert "calculated-not-measured" in prop.source_ref
    assert ";identity-joined-evidence:" in prop.source_ref
    assert prop.verified_on == "structure_snapshot_not_physical_measurement"
    assert original == {}


def test_source_builder_reads_odor_column_only_and_checks_source_hashes(tmp_path):
    from scripts.build_odor_integrity_catalog import read_assertions
    source = tmp_path / "sources"
    source.mkdir()
    behavior = source / "behavior.csv"
    behavior.write_text("Stimulus,Odor Percepts,Flavor Percepts,Odor Modifiers\n1,rose,sweet,dilute\n2,,rose,\n", encoding="utf-8")
    mapping = source / "cas_to_cid.json"
    mapping.write_text("{}", encoding="utf-8")
    aroma = source / "aroma.csv"
    aroma.write_text("Stimulus,Raw Descriptors,Filtered Descriptors\n3,rose;odorless,rose\n", encoding="utf-8")
    db = tmp_path / "registry.db"
    with closing(sqlite3.connect(db)) as con, con:
        con.executescript("CREATE TABLE source_files(source_id,file_kind,path,sha256,redistribution_allowed); CREATE TABLE ingredient_sources(registry_id,source_id,source_cid);")
        con.executemany("INSERT INTO source_files VALUES(?,?,?,?,?)", [
            ("flavordb", "behavior", str(behavior), hashlib.sha256(behavior.read_bytes()).hexdigest(), 0),
            ("aromadb", "behavior", str(aroma), hashlib.sha256(aroma.read_bytes()).hexdigest(), 0),
            ("goodscents", "cas_to_cid", str(mapping), hashlib.sha256(mapping.read_bytes()).hexdigest(), 0)])
        con.executemany("INSERT INTO ingredient_sources VALUES(?,?,?)", [("one", "flavordb", "1"), ("two", "flavordb", "2"), ("three", "aromadb", "3")])
    assertions, refs, provenance = read_assertions(db, source)
    assert assertions == {"one": ("flavordb_odor:rose",), "three": ("aromadb:odorless", "aromadb:rose")}
    assert assess_odor_assertions(assertions["three"])[0] == "conflicting_odor_reports"
    ref = json.loads(refs["one"][0])
    assert ref["source_field"] == "Odor Percepts"
    assert ref["source_record_id"] == "1"
    assert ref["source_conditions_or_modifiers"] == {"Odor Modifiers": "dilute"}
    assert not provenance["flavor_fields_used_as_odor"]
    behavior.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_assertions(db, source)
