from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from fragrance_ai.api import (
    Principal,
    SlidingWindowRateLimiter,
    TokenAuthorizer,
    create_app,
)
from fragrance_ai.platform.audit import _canonical_occurred_at
from fragrance_ai.platform.observability import ServiceMetrics
from fragrance_ai.platform.store import SqliteWorkspaceStore, workspace_store_from_env
from fragrance_ai.platform.worker import process_one
from fragrance_ai.platform.workspace import FormulaWorkspaceService
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog


@dataclass
class FakeResult:
    def to_dict(self):
        return {
            "status": "prototype_ready",
            "formula_id": "sha256:" + "a" * 64,
            "brief": {
                "original_text": "clean citrus woods",
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
                    "concentrate_percent": 50.0,
                },
                {
                    "ingredient_id": "cedarwood_virginia",
                    "name": "Cedarwood Virginia",
                    "concentrate_percent": 50.0,
                },
            ],
            "achieved_profile": {"citrus": 0.6, "woody": 0.4},
            "similarity_score": 95.0,
            "simulated_similarity_score": 93.0,
            "simulation_p05": 91.0,
            "realism_score": 75.0,
            "estimated_concentrate_cost_per_kg": 80.0,
            "olfactory_validation_status": "simulation_only",
            "actual_olfactory_similarity_score": None,
            "actual_olfactory_lower_bound_95": None,
            "safety": {
                "internal_gate_passed": True,
                "status": "prototype_partial_screen",
                "regulatory_data_complete": False,
                "manufacturing_ready": False,
                "warnings": [],
            },
        }


class FakeAI:
    def create_recipe(self, brief, constraints):
        return FakeResult()


@pytest.fixture
def platform(tmp_path):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    store = SqliteWorkspaceStore(tmp_path / "workspace.db")
    audit = AppendOnlyAuditLog(tmp_path / "audit.db", signing_key=b"a" * 32)
    authorizer = TokenAuthorizer.from_plaintext(
        {
            "formulator-a": ("chemist-a", "formulator", "tenant-a"),
            "viewer-b": ("viewer-b", "viewer", "tenant-b"),
            "auditor-a": ("auditor-a", "auditor", "tenant-a"),
        }
    )
    app = create_app(
        ai_factory=FakeAI,
        authorizer=authorizer,
        audit_log=audit,
        workspace_store=store,
        rate_limiter=SlidingWindowRateLimiter(500, 60),
        enable_ui=False,
    )
    with TestClient(app) as client:
        yield client, store, audit
    audit.close()


def _headers(token: str, tenant: str):
    return {"Authorization": f"Bearer {token}", "X-Tenant-ID": tenant}


def test_project_queue_worker_formula_and_tenant_isolation(platform):
    client, store, _ = platform
    created = client.post(
        "/v1/projects",
        headers=_headers("formulator-a", "tenant-a"),
        json={"name": "Summer launch", "description": "visual workbench"},
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]
    queued = client.post(
        "/v1/jobs/recipes",
        headers=_headers("formulator-a", "tenant-a"),
        json={
            "project_id": project_id,
            "brief": "clean citrus woods",
            "name": "Prototype A",
            "constraints": {"require_simulation_pass": False},
        },
    )
    assert queued.status_code == 202
    job_id = queued.json()["job_id"]
    listed = client.get(
        "/v1/jobs", headers=_headers("formulator-a", "tenant-a")
    )
    assert listed.status_code == 200
    assert [item["job_id"] for item in listed.json()["items"]] == [job_id]
    hidden_jobs = client.get(
        "/v1/jobs", headers=_headers("viewer-b", "tenant-b")
    )
    assert hidden_jobs.status_code == 200
    assert hidden_jobs.json()["items"] == []
    workspace = FormulaWorkspaceService(store=store, ai_factory=FakeAI)
    assert process_one(store=store, workspace=workspace, worker_id="worker-a")
    completed = client.get(
        f"/v1/jobs/{job_id}", headers=_headers("formulator-a", "tenant-a")
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "succeeded"
    formula_id = completed.json()["result"]["workspace_formula"]["formula_id"]
    formula = client.get(
        f"/v1/projects/{project_id}/formulas/{formula_id}",
        headers=_headers("formulator-a", "tenant-a"),
    )
    assert formula.status_code == 200
    # Tenant B cannot distinguish a foreign resource from a nonexistent one.
    hidden = client.get(
        f"/v1/projects/{project_id}/formulas/{formula_id}",
        headers=_headers("viewer-b", "tenant-b"),
    )
    assert hidden.status_code == 404


def test_rbac_metrics_request_id_and_security_headers(platform):
    client, _, _ = platform
    denied = client.post(
        "/v1/projects",
        headers=_headers("viewer-b", "tenant-b"),
        json={"name": "Denied"},
    )
    assert denied.status_code == 403
    assert denied.headers["x-content-type-options"] == "nosniff"
    assert denied.headers["x-request-id"].startswith("req_")
    metrics_denied = client.get(
        "/metrics", headers=_headers("formulator-a", "tenant-a")
    )
    assert metrics_denied.status_code == 403
    metrics = client.get(
        "/metrics",
        headers={**_headers("auditor-a", "tenant-a"), "X-Request-ID": "probe-123"},
    )
    assert metrics.status_code == 200
    assert metrics.headers["x-request-id"] == "probe-123"
    assert "perfumery_api_requests_total" in metrics.text
    assert client.get("/health/ready").json()["workspace_backend"] == "sqlite"


def test_production_store_never_downgrades_to_sqlite(monkeypatch):
    monkeypatch.setenv("PERFUMERY_AI_ENV", "production")
    monkeypatch.delenv("PERFUMERY_AI_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="production requires a PostgreSQL"):
        workspace_store_from_env()


def test_ui_assets_are_served_and_redirected(tmp_path):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    store = SqliteWorkspaceStore(tmp_path / "workspace.db")
    audit = AppendOnlyAuditLog(tmp_path / "audit.db", signing_key=b"a" * 32)
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"viewer": ("viewer", "viewer", "tenant-a")}
        ),
        audit_log=audit,
        workspace_store=store,
        enable_ui=True,
    )
    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/ui/"
        page = client.get("/ui/")
        script = client.get("/ui/app.js")
        style = client.get("/ui/app.css")
    assert page.status_code == script.status_code == style.status_code == 200
    assert "Perfumery AI Workbench" in page.text
    assert "application/javascript" in script.headers["content-type"]
    assert "text/css" in style.headers["content-type"]
    assert "setInterval" not in script.text
    assert "setTimeout" in script.text
    assert 'api("/v1/jobs?limit=100")' in script.text
    assert "state.pollers.clear()" in script.text
    audit.close()


def test_chunked_request_body_limit_is_enforced_without_content_length(tmp_path):
    audit = AppendOnlyAuditLog(tmp_path / "audit.db", signing_key=b"a" * 32)
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"formulator": ("chemist", "formulator", "tenant-a")}
        ),
        audit_log=audit,
        enable_ui=False,
        max_request_bytes=4096,
    )
    chunks = iter(
        [
            {"type": "http.request", "body": b"x" * 3000, "more_body": True},
            {"type": "http.request", "body": b"y" * 2000, "more_body": False},
        ]
    )
    sent: list[dict] = []

    async def receive():
        return next(chunks, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/v1/recipes",
        "raw_path": b"/v1/recipes",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer formulator"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 413
    audit.close()


def test_metrics_histograms_use_constant_memory_and_cumulative_buckets():
    metrics = ServiceMetrics()
    for _ in range(10_000):
        metrics.request_started()
        metrics.request_finished(
            method="GET",
            route="/health",
            status_code=200,
            duration_seconds=0.02,
        )
    rendered = metrics.render_prometheus()
    assert 'le="0.025"} 10000' in rendered
    assert 'le="+Inf"} 10000' in rendered
    state = metrics._request_durations[("GET", "/health")]
    assert state.sample_count == 10_000
    assert len(state.bucket_counts) < 20


def test_postgres_audit_timestamp_canonicalization_is_hash_stable():
    assert _canonical_occurred_at("2026-08-03T12:34:56Z") == (
        "2026-08-03T12:34:56+00:00"
    )
    assert _canonical_occurred_at("2026-08-03T21:34:56+09:00") == (
        "2026-08-03T12:34:56+00:00"
    )
    with pytest.raises(ValueError, match="timezone"):
        _canonical_occurred_at("2026-08-03T12:34:56")


def test_job_routes_require_formula_permission_and_catalog_uses_catalog_permission(
    tmp_path,
):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    principals = {
        "full": Principal("chemist", "formulator", "full", "tenant-a"),
        "job-only": Principal(
            "chemist",
            "formulator",
            "job-only",
            "tenant-a",
            frozenset({"job:create"}),
        ),
        "catalog-only": Principal(
            "viewer",
            "viewer",
            "catalog-only",
            "tenant-a",
            frozenset({"catalog:read"}),
        ),
    }

    class NarrowAuthorizer:
        def authenticate(self, token):
            return principals.get(token)

        @staticmethod
        def permits(principal, permission):
            allowed = principal.permissions
            return allowed is None or permission in allowed

    store = SqliteWorkspaceStore(tmp_path / "permissions.db")
    audit = AppendOnlyAuditLog(tmp_path / "permissions-audit.db", signing_key=b"a" * 32)
    app = create_app(
        ai_factory=FakeAI,
        authorizer=NarrowAuthorizer(),
        audit_log=audit,
        workspace_store=store,
        enable_ui=False,
    )
    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            headers=_headers("full", "tenant-a"),
            json={"name": "Scoped project"},
        ).json()
        denied_create = client.post(
            "/v1/jobs/recipes",
            headers=_headers("job-only", "tenant-a"),
            json={"project_id": project["project_id"], "brief": "clean woods"},
        )
        denied_revision = client.post(
            "/v1/jobs/revisions",
            headers=_headers("job-only", "tenant-a"),
            json={
                "project_id": project["project_id"],
                "formula_id": "frm_missing",
                "base_version_id": "ver_missing",
                "instruction": "more citrus",
            },
        )
        catalog = client.get(
            "/v1/catalog", headers=_headers("catalog-only", "tenant-a")
        )
    assert denied_create.status_code == 403
    assert denied_revision.status_code == 403
    assert catalog.status_code == 200
    audit.close()


def test_invalid_resource_identifiers_are_validation_errors(platform):
    client, _, _ = platform
    response = client.get(
        "/v1/projects/not%20an%20identifier",
        headers=_headers("formulator-a", "tenant-a"),
    )
    assert response.status_code == 422


def test_mutation_fails_closed_before_storage_when_audit_intent_is_unavailable(
    tmp_path,
):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    class UnavailableAudit:
        def append(self, **kwargs):
            raise OSError("audit unavailable")

        def verify(self):
            return {"valid": False}

    store = SqliteWorkspaceStore(tmp_path / "fail-closed.db")
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"formulator": ("chemist", "formulator", "tenant-a")}
        ),
        audit_log=UnavailableAudit(),
        workspace_store=store,
        enable_ui=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/projects",
            headers=_headers("formulator", "tenant-a"),
            json={"name": "Must not persist"},
        )
    assert response.status_code == 503
    assert store.list_projects(tenant_id="tenant-a") == []


def test_committed_mutation_remains_successful_when_completion_audit_is_unavailable(
    tmp_path,
):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    class FailSecondAudit:
        calls = 0

        def append(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise OSError("completion audit unavailable")

        def verify(self):
            return {"valid": True}

    audit = FailSecondAudit()
    store = SqliteWorkspaceStore(tmp_path / "completion-audit.db")
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"formulator": ("chemist", "formulator", "tenant-a")}
        ),
        audit_log=audit,
        workspace_store=store,
        enable_ui=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/projects",
            headers=_headers("formulator", "tenant-a"),
            json={"name": "Committed project"},
        )
    assert response.status_code == 201
    assert audit.calls == 2
    assert len(store.list_projects(tenant_id="tenant-a")) == 1
