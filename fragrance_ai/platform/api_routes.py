"""FastAPI routes for tenant projects, versions, accords, and durable jobs."""

import logging
import uuid
from typing import Annotated, Any, Callable

from .observability import ServiceMetrics
from .store import WorkspaceStore
from .workspace import FormulaWorkspaceService, constraints_from_payload


LOGGER = logging.getLogger("perfumery_ai.platform.api_routes")
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$"


def install_workspace_routes(
    *,
    app,
    store: WorkspaceStore,
    ai_factory: Callable[[], Any],
    require,
    audit_log,
    metrics: ServiceMetrics,
) -> None:
    try:
        from fastapi import Depends, HTTPException, Path, Query
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("commercial API dependencies are missing") from error

    workspace = FormulaWorkspaceService(store=store, ai_factory=ai_factory)
    IdentifierPath = Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN),
    ]

    class ProjectCreate(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(min_length=1, max_length=160)
        description: str = Field(default="", max_length=4000)

    class GenerateJobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        project_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        brief: str = Field(min_length=1, max_length=4000)
        name: str = Field(default="Generated formula", min_length=1, max_length=160)
        constraints: dict[str, Any] = Field(default_factory=dict)

    class AccordJobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        project_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        brief: str = Field(min_length=1, max_length=2000)
        name: str = Field(default="Generated accord", min_length=1, max_length=160)
        constraints: dict[str, Any] = Field(default_factory=dict)

    class RevisionJobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        project_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        formula_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        base_version_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        instruction: str = Field(min_length=1, max_length=2000)
        constraints: dict[str, Any] = Field(default_factory=dict)

    class ManualLine(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ingredient_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        concentrate_percent: float = Field(gt=0, le=100)

    class ManualEditRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        base_version_id: str = Field(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        )
        change_note: str = Field(default="Visual formula edit", max_length=2000)
        lines: list[ManualLine] = Field(min_length=1, max_length=30)

    def not_found(detail: str = "resource not found") -> HTTPException:
        return HTTPException(status_code=404, detail=detail)

    def append_audit(
        current,
        event_type: str,
        scope_id: str,
        payload: dict[str, Any],
    ) -> None:
        audit_log.append(
            actor_id=current.actor_id,
            actor_role=current.role,
            event_type=event_type,
            scope_id=scope_id,
            payload={"tenant_id": current.tenant_id, **payload},
        )

    def begin_mutation_audit(
        current,
        *,
        action: str,
        scope_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Write a durable intent before state mutation; audit outages fail closed."""

        operation_id = str(uuid.uuid4())
        try:
            append_audit(
                current,
                f"{action}.requested",
                scope_id,
                {"operation_id": operation_id, **payload},
            )
        except Exception as error:
            LOGGER.error(
                "audit_intent_failed",
                extra={
                    "action": action,
                    "operation_id": operation_id,
                    "tenant_id": current.tenant_id,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="audit service unavailable; mutation was not applied",
            ) from error
        return operation_id

    def complete_mutation_audit(
        current,
        *,
        event_type: str,
        scope_id: str,
        operation_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Do not turn a committed mutation into an apparent API failure."""

        try:
            append_audit(
                current,
                event_type,
                scope_id,
                {"operation_id": operation_id, **payload},
            )
        except Exception:  # pragma: no cover - backend outage path
            LOGGER.critical(
                "audit_completion_failed",
                extra={
                    "event_type": event_type,
                    "operation_id": operation_id,
                    "tenant_id": current.tenant_id,
                },
                exc_info=True,
            )

    @app.get("/v1/system/capabilities")
    def capabilities(current=Depends(require("formula:read"))) -> dict[str, Any]:
        return {
            "release": "1.4.0",
            "workspace_backend": store.backend,
            "horizontal_scaling": store.horizontally_scalable,
            "features": {
                "projects": True,
                "formula_versions": True,
                "manual_visual_edit": True,
                "natural_language_revision": True,
                "accord_generation": True,
                "durable_jobs": True,
                "prometheus_metrics": True,
                "authenticated_audit_chain": True,
            },
            "tenant_id": current.tenant_id,
        }

    @app.get("/v1/catalog")
    def catalog(current=Depends(require("catalog:read"))) -> dict[str, Any]:
        return workspace.catalog_payload()

    @app.post("/v1/projects", status_code=201)
    def create_project(
        request: ProjectCreate,
        current=Depends(require("project:create")),
    ) -> dict[str, Any]:
        operation_id = begin_mutation_audit(
            current,
            action="project.create",
            scope_id=f"tenant:{current.tenant_id}",
            payload={},
        )
        try:
            project = store.create_project(
                tenant_id=current.tenant_id,
                name=request.name,
                description=request.description,
                actor_id=current.actor_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        complete_mutation_audit(
            current,
            event_type="project.created",
            scope_id=f"tenant:{current.tenant_id}:project:{project.project_id}",
            operation_id=operation_id,
            payload={"project_id": project.project_id},
        )
        return project.to_dict()

    @app.get("/v1/projects")
    def list_projects(
        limit: int = Query(default=100, ge=1, le=500),
        current=Depends(require("project:read")),
    ) -> dict[str, Any]:
        projects = store.list_projects(tenant_id=current.tenant_id, limit=limit)
        return {"items": [project.to_dict() for project in projects]}

    @app.get("/v1/projects/{project_id}")
    def get_project(
        project_id: IdentifierPath,
        current=Depends(require("project:read")),
    ) -> dict[str, Any]:
        project = store.get_project(tenant_id=current.tenant_id, project_id=project_id)
        if project is None:
            raise not_found("project not found")
        return project.to_dict()

    @app.get("/v1/projects/{project_id}/formulas")
    def list_formulas(
        project_id: IdentifierPath,
        limit: int = Query(default=100, ge=1, le=500),
        current=Depends(require("formula:read")),
    ) -> dict[str, Any]:
        if store.get_project(tenant_id=current.tenant_id, project_id=project_id) is None:
            raise not_found("project not found")
        formulas = store.list_formulas(
            tenant_id=current.tenant_id,
            project_id=project_id,
            limit=limit,
        )
        return {
            "items": [formula.to_dict(include_payload=False) for formula in formulas]
        }

    @app.get("/v1/projects/{project_id}/formulas/{formula_id}")
    def get_formula(
        project_id: IdentifierPath,
        formula_id: IdentifierPath,
        current=Depends(require("formula:read")),
    ) -> dict[str, Any]:
        formula = store.get_formula(
            tenant_id=current.tenant_id,
            project_id=project_id,
            formula_id=formula_id,
        )
        if formula is None:
            raise not_found("formula not found")
        return formula.to_dict()

    @app.get("/v1/projects/{project_id}/formulas/{formula_id}/versions")
    def list_versions(
        project_id: IdentifierPath,
        formula_id: IdentifierPath,
        limit: int = Query(default=100, ge=1, le=500),
        current=Depends(require("formula:read")),
    ) -> dict[str, Any]:
        if store.get_formula(
            tenant_id=current.tenant_id,
            project_id=project_id,
            formula_id=formula_id,
        ) is None:
            raise not_found("formula not found")
        versions = store.list_formula_versions(
            tenant_id=current.tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            limit=limit,
        )
        return {
            "items": [version.to_dict(include_payload=False) for version in versions]
        }

    @app.get("/v1/projects/{project_id}/formulas/{formula_id}/versions/{version_id}")
    def get_version(
        project_id: IdentifierPath,
        formula_id: IdentifierPath,
        version_id: IdentifierPath,
        current=Depends(require("formula:read")),
    ) -> dict[str, Any]:
        version = store.get_formula_version(
            tenant_id=current.tenant_id,
            project_id=project_id,
            formula_id=formula_id,
            version_id=version_id,
        )
        if version is None:
            raise not_found("formula version not found")
        return version.to_dict()

    @app.post(
        "/v1/projects/{project_id}/formulas/{formula_id}/versions",
        status_code=201,
    )
    def manual_edit(
        project_id: IdentifierPath,
        formula_id: IdentifierPath,
        request: ManualEditRequest,
        current=Depends(require("formula:edit")),
    ) -> dict[str, Any]:
        operation_id = begin_mutation_audit(
            current,
            action="formula.version_create",
            scope_id=f"tenant:{current.tenant_id}:formula:{formula_id}",
            payload={
                "project_id": project_id,
                "formula_id": formula_id,
                "base_version_id": request.base_version_id,
                "change_kind": "manual_edit",
            },
        )
        try:
            version = workspace.manual_edit(
                tenant_id=current.tenant_id,
                actor_id=current.actor_id,
                project_id=project_id,
                formula_id=formula_id,
                base_version_id=request.base_version_id,
                lines=[line.model_dump() for line in request.lines],
                change_note=request.change_note,
            )
        except KeyError as error:
            raise not_found("formula version not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail="formula version conflict") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        complete_mutation_audit(
            current,
            event_type="formula.version_created",
            scope_id=f"tenant:{current.tenant_id}:formula:{formula_id}",
            operation_id=operation_id,
            payload={
                "project_id": project_id,
                "formula_id": formula_id,
                "version_id": version["version_id"],
                "change_kind": "manual_edit",
                "content_sha256": version["content_sha256"],
            },
        )
        return version

    @app.get("/v1/projects/{project_id}/formulas/{formula_id}/compare")
    def compare_versions(
        project_id: IdentifierPath,
        formula_id: IdentifierPath,
        left: str = Query(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        ),
        right: str = Query(
            min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN
        ),
        current=Depends(require("formula:read")),
    ) -> dict[str, Any]:
        try:
            return workspace.compare_versions(
                tenant_id=current.tenant_id,
                project_id=project_id,
                formula_id=formula_id,
                left_version_id=left,
                right_version_id=right,
            )
        except KeyError as error:
            raise not_found("formula version not found") from error

    def enqueue(current, kind: str, payload: dict[str, Any]):
        project_id = str(payload.get("project_id", ""))
        if store.get_project(
            tenant_id=current.tenant_id, project_id=project_id
        ) is None:
            raise not_found("project not found")
        try:
            constraints_from_payload(payload.get("constraints"))
            operation_id = begin_mutation_audit(
                current,
                action="job.queue",
                scope_id=f"tenant:{current.tenant_id}:project:{project_id}",
                payload={"kind": kind, "project_id": project_id},
            )
            job = store.enqueue_job(
                tenant_id=current.tenant_id,
                kind=kind,
                payload=payload,
                actor_id=current.actor_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        complete_mutation_audit(
            current,
            event_type="job.queued",
            scope_id=f"tenant:{current.tenant_id}:job:{job.job_id}",
            operation_id=operation_id,
            payload={"job_id": job.job_id, "kind": kind, "project_id": project_id},
        )
        metrics.observe_job(kind=kind, outcome="queued")
        return job.to_dict()

    @app.post("/v1/jobs/recipes", status_code=202)
    def enqueue_recipe(
        request: GenerateJobRequest,
        current=Depends(require("job:create", "formula:create")),
    ) -> dict[str, Any]:
        return enqueue(current, "recipe.generate", request.model_dump())

    @app.post("/v1/jobs/accords", status_code=202)
    def enqueue_accord(
        request: AccordJobRequest,
        current=Depends(require("job:create", "formula:create")),
    ) -> dict[str, Any]:
        return enqueue(current, "accord.generate", request.model_dump())

    @app.post("/v1/jobs/revisions", status_code=202)
    def enqueue_revision(
        request: RevisionJobRequest,
        current=Depends(require("job:create", "formula:edit")),
    ) -> dict[str, Any]:
        if store.get_formula_version(
            tenant_id=current.tenant_id,
            project_id=request.project_id,
            formula_id=request.formula_id,
            version_id=request.base_version_id,
        ) is None:
            raise not_found("formula version not found")
        return enqueue(current, "formula.revise", request.model_dump())

    @app.get("/v1/jobs")
    def list_jobs(
        limit: int = Query(default=100, ge=1, le=500),
        current=Depends(require("job:read")),
    ) -> dict[str, Any]:
        jobs = store.list_jobs(tenant_id=current.tenant_id, limit=limit)
        return {"items": [job.to_dict() for job in jobs]}

    @app.get("/v1/jobs/{job_id}")
    def get_job(
        job_id: IdentifierPath,
        current=Depends(require("job:read")),
    ) -> dict[str, Any]:
        job = store.get_job(tenant_id=current.tenant_id, job_id=job_id)
        if job is None:
            raise not_found("job not found")
        return job.to_dict()
