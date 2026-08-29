"""Lease-based worker for horizontally distributed inference jobs."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable

from ..recommender.service import NaturalLanguagePerfumeryAI
from .audit import audit_log_from_env
from .observability import ServiceMetrics, configure_json_logging
from .store import WorkspaceStore, workspace_store_from_env
from .workspace import FormulaWorkspaceService


LOGGER = logging.getLogger("perfumery_ai.worker")


def process_one(
    *,
    store: WorkspaceStore,
    workspace: FormulaWorkspaceService,
    worker_id: str,
    lease_seconds: int = 300,
    audit_log=None,
    metrics: ServiceMetrics | None = None,
    heartbeat_interval_seconds: float | None = None,
) -> bool:
    heartbeat_interval = (
        max(1.0, min(30.0, lease_seconds / 3.0))
        if heartbeat_interval_seconds is None
        else float(heartbeat_interval_seconds)
    )
    if not 0.01 <= heartbeat_interval < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than the lease")
    job = store.claim_job(worker_id=worker_id, lease_seconds=lease_seconds)
    if job is None:
        return False

    def append_worker_audit(event_type: str, payload: dict[str, Any]) -> None:
        if audit_log is None:
            return
        audit_log.append(
            actor_id=worker_id,
            actor_role="worker",
            event_type=event_type,
            scope_id=f"tenant:{job.tenant_id}:job:{job.job_id}",
            payload={
                "tenant_id": job.tenant_id,
                "job_id": job.job_id,
                "kind": job.kind,
                **payload,
            },
        )

    try:
        append_worker_audit(
            "job.started",
            {"attempt": job.attempts, "lease_owner": worker_id},
        )
    except Exception:
        try:
            store.fail_job(
                job_id=job.job_id,
                worker_id=worker_id,
                error_code="AuditUnavailable",
            )
        except Exception:
            LOGGER.exception(
                "job_requeue_failed_after_audit_outage",
                extra={"job_id": job.job_id, "kind": job.kind},
            )
        if metrics is not None:
            metrics.observe_job(kind=job.kind, outcome="failed")
        LOGGER.exception(
            "job_start_audit_failed",
            extra={"job_id": job.job_id, "kind": job.kind},
        )
        return True

    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()

    def maintain_lease() -> None:
        while not heartbeat_stop.wait(heartbeat_interval):
            try:
                if not store.renew_job_lease(
                    job_id=job.job_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                ):
                    lease_lost.set()
                    return
            except Exception:
                lease_lost.set()
                LOGGER.exception(
                    "job_lease_heartbeat_failed",
                    extra={"job_id": job.job_id, "kind": job.kind},
                )
                return

    def renew_before_persist() -> None:
        """Fence every durable formula effect with a fresh owned lease."""

        if lease_lost.is_set():
            raise RuntimeError("job lease was lost before persistence")
        try:
            renewed = store.renew_job_lease(
                job_id=job.job_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except Exception as error:
            lease_lost.set()
            raise RuntimeError("job lease renewal failed before persistence") from error
        if not renewed:
            lease_lost.set()
            raise RuntimeError("job lease was lost before persistence")

    heartbeat = threading.Thread(
        target=maintain_lease,
        name=f"lease-{job.job_id}",
        daemon=True,
    )
    heartbeat.start()
    try:
        try:
            result = workspace.process_job(job, before_persist=renew_before_persist)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(1.0, heartbeat_interval * 2.0))
        if lease_lost.is_set():
            raise RuntimeError("job lease was lost during processing")
        store.complete_job(job_id=job.job_id, worker_id=worker_id, result=result)
        workspace_formula = result.get("workspace_formula") or {}
        workspace_version = result.get("workspace_version") or {}
        try:
            append_worker_audit(
                "job.succeeded",
                {
                    "formula_id": workspace_formula.get("formula_id")
                    or job.payload.get("formula_id"),
                    "version_id": (
                        workspace_formula.get("latest_version", {}).get("version_id")
                        or workspace_version.get("version_id")
                    ),
                },
            )
        except Exception:
            LOGGER.critical(
                "job_success_audit_failed_after_commit",
                extra={"job_id": job.job_id, "kind": job.kind},
                exc_info=True,
            )
        if metrics is not None:
            metrics.observe_job(kind=job.kind, outcome="succeeded")
        LOGGER.info("job_succeeded", extra={"job_id": job.job_id, "kind": job.kind})
    except Exception as error:
        heartbeat_stop.set()
        heartbeat.join(timeout=max(1.0, heartbeat_interval * 2.0))
        code = f"{type(error).__name__}"[:128]
        failure_recorded = True
        try:
            store.fail_job(
                job_id=job.job_id,
                worker_id=worker_id,
                error_code=code,
            )
        except Exception:
            failure_recorded = False
            LOGGER.exception(
                "job_failure_state_not_recorded",
                extra={"job_id": job.job_id, "kind": job.kind},
            )
        try:
            append_worker_audit(
                "job.failed" if failure_recorded else "job.state_unknown",
                {"error_code": code},
            )
        except Exception:
            LOGGER.critical(
                "job_failure_audit_failed",
                extra={"job_id": job.job_id, "kind": job.kind},
                exc_info=True,
            )
        if metrics is not None:
            metrics.observe_job(kind=job.kind, outcome="failed")
        LOGGER.exception("job_failed", extra={"job_id": job.job_id, "kind": job.kind})
    return True


def run_worker(
    *,
    store: WorkspaceStore,
    ai_factory: Callable[[], Any],
    worker_id: str,
    poll_seconds: float = 1.0,
    lease_seconds: int = 300,
    once: bool = False,
    audit_log=None,
    metrics: ServiceMetrics | None = None,
) -> int:
    # A worker is sequential, so one thread-owned engine can safely serve its
    # whole process lifetime instead of reopening model and SQLite assets for
    # every queued job.
    ai = ai_factory()
    workspace = FormulaWorkspaceService(
        store=store,
        ai_factory=ai_factory,
        ai_instance=ai,
    )
    try:
        processed = 0
        while True:
            found = process_one(
                store=store,
                workspace=workspace,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                audit_log=audit_log,
                metrics=metrics,
            )
            if found:
                processed += 1
            if once:
                return processed
            if not found:
                time.sleep(poll_seconds)
    finally:
        close = getattr(ai, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0.05 <= args.poll_seconds <= 60:
        raise SystemExit("--poll-seconds must be between 0.05 and 60")
    if not 10 <= args.lease_seconds <= 3600:
        raise SystemExit("--lease-seconds must be between 10 and 3600")
    configure_json_logging(os.environ.get("PERFUMERY_AI_LOG_LEVEL", "INFO"))
    store = workspace_store_from_env()
    audit_log = audit_log_from_env()
    metrics = ServiceMetrics()
    try:
        try:
            run_worker(
                store=store,
                ai_factory=NaturalLanguagePerfumeryAI,
                worker_id=args.worker_id,
                poll_seconds=args.poll_seconds,
                lease_seconds=args.lease_seconds,
                once=args.once,
                audit_log=audit_log,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            LOGGER.info("worker_shutdown_requested", extra={"worker_id": args.worker_id})
    finally:
        audit_log.close()
        store.close()


if __name__ == "__main__":
    main()
