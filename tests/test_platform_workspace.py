from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import pytest

from fragrance_ai.platform.store import SqliteWorkspaceStore
from fragrance_ai.platform.worker import process_one, run_worker
from fragrance_ai.platform.workspace import (
    FormulaWorkspaceService,
    constraints_from_payload,
)
from fragrance_ai.recommender.brief_parser import apply_relative_revision_profile


def _payload(percent_a: float = 60.0, percent_b: float = 40.0):
    return {
        "status": "prototype_ready",
        "formula_id": "sha256:" + "a" * 64,
        "brief": {
            "original_text": "clean citrus woody",
            "target_profile": {"citrus": 0.6, "woody": 0.4},
            "constraints": {
                "product_concentration_percent": 15.0,
                "finished_batch_mass_g": 50.0,
                "max_risk_tier": 1,
                "explicit_bans": [],
            },
        },
        "recipe": [
            {
                "ingredient_id": "bergamot_fcf",
                "name": "Bergamot FCF",
                "concentrate_percent": percent_a,
            },
            {
                "ingredient_id": "cedarwood_virginia",
                "name": "Cedarwood Virginia",
                "concentrate_percent": percent_b,
            },
        ],
        "achieved_profile": {"citrus": 0.6, "woody": 0.4},
        "similarity_score": 95.0,
        "simulated_similarity_score": 93.0,
        "simulation_p05": 91.0,
        "realism_score": 75.0,
        "estimated_concentrate_cost_per_kg": 80.0,
        "safety": {
            "internal_gate_passed": True,
            "status": "prototype_partial_screen",
            "regulatory_data_complete": False,
            "manufacturing_ready": False,
            "warnings": [],
        },
    }


def test_sqlite_store_is_tenant_scoped_and_versions_are_optimistic(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "workspace.db")
    first = store.create_project(
        tenant_id="tenant-a", name="A", description="", actor_id="chemist-a"
    )
    second = store.create_project(
        tenant_id="tenant-b", name="B", description="", actor_id="chemist-b"
    )
    formula = store.create_formula(
        tenant_id="tenant-a",
        project_id=first.project_id,
        name="Fresh woods",
        kind="formula",
        payload=_payload(),
        actor_id="chemist-a",
        change_note="initial",
    )
    assert (
        store.get_formula(
            tenant_id="tenant-b",
            project_id=second.project_id,
            formula_id=formula.formula_id,
        )
        is None
    )
    version = store.append_formula_version(
        tenant_id="tenant-a",
        project_id=first.project_id,
        formula_id=formula.formula_id,
        expected_parent_version_id=formula.latest_version.version_id,
        change_kind="manual_edit",
        change_note="change",
        payload=_payload(55, 45),
        actor_id="chemist-a",
    )
    assert version.version_number == 2
    with pytest.raises(RuntimeError, match="version conflict"):
        store.append_formula_version(
            tenant_id="tenant-a",
            project_id=first.project_id,
            formula_id=formula.formula_id,
            expected_parent_version_id=formula.latest_version.version_id,
            change_kind="manual_edit",
            change_note="stale",
            payload=_payload(50, 50),
            actor_id="chemist-a",
        )


def test_queue_claims_each_job_once_across_concurrent_workers(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "queue.db")
    for index in range(50):
        store.enqueue_job(
            tenant_id=f"tenant-{index % 2}",
            kind="recipe.generate",
            payload={"index": index},
            actor_id=f"actor-{index % 2}",
        )

    def claim(worker: int):
        claimed = []
        while True:
            job = store.claim_job(worker_id=f"worker-{worker}", lease_seconds=30)
            if job is None:
                return claimed
            claimed.append(job.job_id)
            store.complete_job(
                job_id=job.job_id,
                worker_id=f"worker-{worker}",
                result={"ok": True},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(claim, range(8)))
    claimed = [job_id for batch in batches for job_id in batch]
    assert len(claimed) == 50
    assert len(set(claimed)) == 50


def test_worker_renews_lease_during_slow_inference(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "heartbeat.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )

    @dataclass
    class SlowResult:
        def to_dict(self):
            return _payload()

    class SlowAI:
        def create_recipe(self, brief, constraints):
            time.sleep(0.08)
            return SlowResult()

    job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={
            "project_id": project.project_id,
            "brief": "slow clean woods",
            "constraints": {},
            "name": "Slow formula",
        },
        actor_id="chemist",
    )
    renewals = 0
    original = store.renew_job_lease

    def counted_renewal(**kwargs):
        nonlocal renewals
        renewals += 1
        return original(**kwargs)

    store.renew_job_lease = counted_renewal
    workspace = FormulaWorkspaceService(store=store, ai_factory=SlowAI)
    assert process_one(
        store=store,
        workspace=workspace,
        worker_id="worker-heartbeat",
        lease_seconds=10,
        heartbeat_interval_seconds=0.01,
    )
    assert renewals >= 2
    assert store.get_job(tenant_id="tenant-a", job_id=job.job_id).status == "succeeded"


def test_natural_language_revision_uses_relative_profile_context(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "revision.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    formula = store.create_formula(
        tenant_id="tenant-a",
        project_id=project.project_id,
        name="Formula",
        kind="formula",
        payload=_payload(),
        actor_id="chemist",
        change_note="initial",
    )
    captured = {}

    @dataclass
    class FakeResult:
        payload: dict

        def to_dict(self):
            return self.payload

    class ContextualAI:
        def create_recipe_with_target_profile(self, brief, constraints, target_profile):
            captured.update(
                brief=brief,
                constraints=constraints,
                target_profile=target_profile,
            )
            payload = _payload()
            payload["brief"]["original_text"] = brief
            payload["brief"]["target_profile"] = target_profile
            payload["achieved_profile"] = target_profile
            return FakeResult(payload)

    service = FormulaWorkspaceService(store=store, ai_factory=ContextualAI)
    result = service.revise_formula(
        tenant_id="tenant-a",
        actor_id="chemist",
        project_id=project.project_id,
        formula_id=formula.formula_id,
        base_version_id=formula.latest_version.version_id,
        instruction="우디함을 조금 높이고 단맛은 줄여줘",
    )

    assert captured["target_profile"]["woody"] > 0.4
    assert result["workspace_version"]["version_number"] == 2
    context = result["result"]["revision_context"]
    assert context["mode"] == "relative_profile_edit"
    assert context["adjustment_multipliers"]["woody"] > 1.0
    assert context["adjustment_multipliers"]["gourmand"] < 1.0


def test_manual_edit_invalidates_prior_approvals_and_compares_versions(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "edit.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    initial_payload = _payload(70, 30)
    initial_payload.update(
        {
            "scientific_model_domain_passed": True,
            "scientific_uncertainty_kind": "stale_prior_propagation",
            "physsim_comparison_target_status": "evidenced_quantitative_composition",
            "physsim_comparison_authorized": True,
            "perceptual_prediction_status": "calibrated_registered_study_endpoint",
            "human_discrimination_probability": 0.95,
            "human_discrimination_lower_95": 0.91,
            "human_discrimination_upper_95": 0.98,
            "human_calibration_applicability_percent": 100.0,
            "human_calibration_artifact_id": "sha256:" + "b" * 64,
            "human_calibration_flags": ["stale"],
            "human_similarity_90_claim_authorized": True,
            "sensory_similarity_score": 95.0,
            "sensory_panel_size": 12,
            "sensory_validation_status": "verified_passed",
            "release_spec_id": "sha256:" + "c" * 64,
            "historical_support_score": 88.0,
            "historical_reference_matches": [{"name": "stale"}],
            "reference_target_comparison_kind": (
                "evidenced_composition_physsim_target"
            ),
        }
    )
    initial_payload["safety"].update(
        {
            "audit_id": "stale-audit",
            "evidence_coverage_percent": 100.0,
            "internal_evidence_complete": True,
            "eu_label_declarations": ["stale-label"],
        }
    )
    formula = store.create_formula(
        tenant_id="tenant-a",
        project_id=project.project_id,
        name="Formula",
        kind="formula",
        payload=initial_payload,
        actor_id="chemist",
        change_note="initial",
    )
    service = FormulaWorkspaceService(store=store, ai_factory=lambda: None)
    # Respect the catalog caps by editing a diverse set of formulation-ready materials.
    candidates = [
        item
        for item in service.catalog.ingredients
        if item.formulation_ready
        and not item.blocked
        and item.as_supplied_cap_percent() >= 10
    ][:10]
    assert len(candidates) >= 10
    edited = service.manual_edit(
        tenant_id="tenant-a",
        actor_id="chemist",
        project_id=project.project_id,
        formula_id=formula.formula_id,
        base_version_id=formula.latest_version.version_id,
        lines=[
            {"ingredient_id": item.ingredient_id, "concentrate_percent": 10.0}
            for item in candidates
        ],
        change_note="rebalance visually",
    )
    payload = edited["payload"]
    assert payload["status"] == "draft_manual_edit"
    assert payload["simulation_status"] == "not_run_after_manual_edit"
    assert payload["realism_score"] == 0.0
    assert payload["temporal_similarity_score"] == 0.0
    assert payload["physsim_similarity_score"] == 0.0
    assert payload["concentration_response_similarity_score"] is None
    assert payload["actual_olfactory_similarity_score"] is None
    assert payload["scientific_model_domain_passed"] is False
    assert payload["scientific_uncertainty_kind"] == "invalidated_by_formula_change"
    assert payload["physsim_comparison_authorized"] is False
    assert payload["perceptual_prediction_status"] == "manual_draft_unvalidated"
    assert payload["human_discrimination_probability"] is None
    assert payload["human_calibration_applicability_percent"] == 0.0
    assert payload["human_similarity_90_claim_authorized"] is False
    assert payload["sensory_similarity_score"] is None
    assert payload["sensory_panel_size"] == 0
    assert payload["sensory_validation_status"] == "invalidated_by_formula_change"
    assert payload["release_spec_id"] == ""
    assert payload["historical_support_score"] == 0.0
    assert payload["historical_reference_matches"] == []
    assert (
        payload["reference_target_comparison_kind"]
        == "not_recomputed_after_manual_edit"
    )
    assert payload["safety"]["internal_gate_passed"] is False
    assert payload["safety"]["audit_id"] == ""
    assert payload["safety"]["evidence_coverage_percent"] == 0.0
    assert payload["safety"]["eu_label_declarations"] == []
    comparison = service.compare_versions(
        tenant_id="tenant-a",
        project_id=project.project_id,
        formula_id=formula.formula_id,
        left_version_id=formula.latest_version.version_id,
        right_version_id=edited["version_id"],
    )
    assert comparison["ingredient_changes"]
    assert "estimated_concentrate_cost_per_kg" in comparison["metric_changes"]


def test_manual_edit_persists_an_exactly_mass_balanced_formula(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "balanced-edit.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    formula = store.create_formula(
        tenant_id="tenant-a",
        project_id=project.project_id,
        name="Formula",
        kind="formula",
        payload=_payload(),
        actor_id="chemist",
        change_note="initial",
    )
    service = FormulaWorkspaceService(store=store, ai_factory=lambda: None)
    edited = service.manual_edit(
        tenant_id="tenant-a",
        actor_id="chemist",
        project_id=project.project_id,
        formula_id=formula.formula_id,
        base_version_id=formula.latest_version.version_id,
        lines=[
            {"ingredient_id": "iso_e_super", "concentrate_percent": 30.0},
            {"ingredient_id": "hedione", "concentrate_percent": 30.0},
            {"ingredient_id": "dihydromyrcenol", "concentrate_percent": 20.0},
            {"ingredient_id": "linalyl_acetate", "concentrate_percent": 19.99},
        ],
        change_note="normalize rounding drift",
    )
    total = sum(line["concentrate_percent"] for line in edited["payload"]["recipe"])
    assert total == pytest.approx(100.0, abs=1e-9)


def test_worker_persists_generated_formula_and_job_result(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "worker.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )

    @dataclass
    class FakeResult:
        def to_dict(self):
            return _payload()

    class FakeAI:
        def create_recipe(self, brief, constraints):
            assert brief == "clean woods"
            return FakeResult()

    job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={
            "project_id": project.project_id,
            "brief": "clean woods",
            "constraints": {"require_simulation_pass": False},
            "name": "Queued formula",
        },
        actor_id="chemist",
    )
    workspace = FormulaWorkspaceService(store=store, ai_factory=FakeAI)
    assert process_one(store=store, workspace=workspace, worker_id="worker-1")
    completed = store.get_job(tenant_id="tenant-a", job_id=job.job_id)
    assert completed.status == "succeeded"
    assert completed.result["workspace_formula"]["name"] == "Queued formula"


def test_worker_owns_one_engine_for_its_process_lifetime(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "worker-engine.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={
            "project_id": project.project_id,
            "brief": "clean woods",
            "constraints": {},
            "name": "Engine lifetime",
        },
        actor_id="chemist",
    )
    created = 0
    closed = 0

    @dataclass
    class FakeResult:
        def to_dict(self):
            return _payload()

    class ReusableAI:
        def __init__(self):
            nonlocal created
            created += 1

        def create_recipe(self, brief, constraints):
            return FakeResult()

        def close(self):
            nonlocal closed
            closed += 1

    assert (
        run_worker(
            store=store,
            ai_factory=ReusableAI,
            worker_id="worker-reuse",
            once=True,
        )
        == 1
    )
    assert created == 1
    assert closed == 1


@pytest.mark.parametrize(
    "constraints, message",
    [
        ({"simulation_draws": 63}, "simulation_draws"),
        ({"max_ingredients": 2}, "max_ingredients"),
        ({"target_similarity": float("nan")}, "finite"),
        ({"allow_rare": 1}, "boolean"),
        ({"require_evidenced_olfactory_target": 1}, "boolean"),
        ({"minimum_dimension_material_strength": 0}, "minimum_dimension"),
        ({"surrogate_objective_weight": 0.51}, "surrogate_objective_weight"),
    ],
)
def test_workspace_constraints_fail_closed_on_invalid_values(constraints, message):
    with pytest.raises(ValueError, match=message):
        constraints_from_payload(constraints)


def test_relative_decrease_is_not_treated_as_dimension_removal():
    base = {"woody": 0.6, "citrus": 0.4}
    decreased, adjustments = apply_relative_revision_profile(base, "reduce woody")
    removed, remove_adjustments = apply_relative_revision_profile(base, "remove woody")
    assert adjustments["woody"] == 0.5
    assert decreased["woody"] > 0
    assert remove_adjustments["woody"] == 0.0
    assert removed.get("woody", 0.0) == 0.0


def test_job_effect_is_idempotent_for_formula_creation_and_revision(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "idempotency.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    create_job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={"project_id": project.project_id},
        actor_id="chemist",
    )
    first = store.create_formula(
        tenant_id="tenant-a",
        project_id=project.project_id,
        name="Idempotent formula",
        kind="formula",
        payload=_payload(),
        actor_id="chemist",
        change_note="generated",
        source_job_id=create_job.job_id,
    )
    replayed = store.create_formula(
        tenant_id="tenant-a",
        project_id=project.project_id,
        name="Ignored replay name",
        kind="formula",
        payload=_payload(),
        actor_id="chemist",
        change_note="generated replay",
        source_job_id=create_job.job_id,
    )
    assert replayed.formula_id == first.formula_id
    assert (
        len(store.list_formulas(tenant_id="tenant-a", project_id=project.project_id))
        == 1
    )

    revision_job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="formula.revise",
        payload={"project_id": project.project_id},
        actor_id="chemist",
    )
    revised = store.append_formula_version(
        tenant_id="tenant-a",
        project_id=project.project_id,
        formula_id=first.formula_id,
        expected_parent_version_id=first.latest_version.version_id,
        change_kind="natural_language_revision",
        change_note="increase citrus",
        payload=_payload(70, 30),
        actor_id="chemist",
        source_job_id=revision_job.job_id,
    )
    replayed_revision = store.append_formula_version(
        tenant_id="tenant-a",
        project_id=project.project_id,
        formula_id=first.formula_id,
        expected_parent_version_id=first.latest_version.version_id,
        change_kind="natural_language_revision",
        change_note="replayed",
        payload=_payload(75, 25),
        actor_id="chemist",
        source_job_id=revision_job.job_id,
    )
    assert replayed_revision.version_id == revised.version_id
    assert (
        len(
            store.list_formula_versions(
                tenant_id="tenant-a",
                project_id=project.project_id,
                formula_id=first.formula_id,
            )
        )
        == 2
    )


def test_invalid_heartbeat_does_not_claim_a_job(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "invalid-heartbeat.db")
    job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={"project_id": "prj_unused"},
        actor_id="chemist",
    )
    workspace = FormulaWorkspaceService(store=store, ai_factory=lambda: None)
    with pytest.raises(ValueError, match="heartbeat interval"):
        process_one(
            store=store,
            workspace=workspace,
            worker_id="worker-invalid",
            lease_seconds=10,
            heartbeat_interval_seconds=10,
        )
    untouched = store.get_job(tenant_id="tenant-a", job_id=job.job_id)
    assert untouched.status == "queued"
    assert untouched.attempts == 0


def test_post_commit_audit_failure_does_not_flip_successful_job(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "post-commit-audit.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={
            "project_id": project.project_id,
            "brief": "clean woods",
            "constraints": {"require_simulation_pass": False},
            "name": "Audited formula",
        },
        actor_id="chemist",
    )

    class FailOnCompletionAudit:
        calls = 0

        def append(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise OSError("audit backend unavailable")

    @dataclass
    class Result:
        def to_dict(self):
            return _payload()

    class GenerateAI:
        def create_recipe(self, brief, constraints):
            return Result()

    workspace = FormulaWorkspaceService(store=store, ai_factory=GenerateAI)
    assert process_one(
        store=store,
        workspace=workspace,
        worker_id="worker-audit",
        audit_log=FailOnCompletionAudit(),
    )
    completed = store.get_job(tenant_id="tenant-a", job_id=job.job_id)
    assert completed.status == "succeeded"
    assert (
        len(store.list_formulas(tenant_id="tenant-a", project_id=project.project_id))
        == 1
    )


def test_worker_requeues_before_inference_when_start_audit_is_unavailable(tmp_path):
    store = SqliteWorkspaceStore(tmp_path / "start-audit.db")
    project = store.create_project(
        tenant_id="tenant-a", name="Project", description="", actor_id="chemist"
    )
    job = store.enqueue_job(
        tenant_id="tenant-a",
        kind="recipe.generate",
        payload={
            "project_id": project.project_id,
            "brief": "clean woods",
            "constraints": {},
        },
        actor_id="chemist",
    )

    class UnavailableAudit:
        def append(self, **kwargs):
            raise OSError("audit backend unavailable")

    workspace = FormulaWorkspaceService(store=store, ai_factory=lambda: None)
    assert process_one(
        store=store,
        workspace=workspace,
        worker_id="worker-audit-down",
        audit_log=UnavailableAudit(),
    )
    requeued = store.get_job(tenant_id="tenant-a", job_id=job.job_id)
    assert requeued.status == "queued"
    assert requeued.error_code == "AuditUnavailable"
    assert (
        store.list_formulas(tenant_id="tenant-a", project_id=project.project_id) == []
    )
