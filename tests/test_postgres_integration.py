from __future__ import annotations

import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from fragrance_ai.platform.postgres import (
    PostgresFixedWindowRateLimiter,
    PostgresWorkspaceStore,
)


DATABASE_URL = os.environ.get("PERFUMERY_AI_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PERFUMERY_AI_TEST_POSTGRES_URL is not configured",
)


def _payload(marker: str) -> dict:
    return {
        "status": "prototype_ready",
        "formula_id": "sha256:" + marker * 64,
        "recipe": [
            {
                "ingredient_id": "bergamot_fcf",
                "name": "Bergamot FCF",
                "concentrate_percent": 100.0,
            }
        ],
    }


def test_postgres_concurrency_idempotency_tenant_queue_and_rate_limit():
    store = PostgresWorkspaceStore(DATABASE_URL, min_size=1, max_size=12)
    suffix = uuid.uuid4().hex[:12]
    tenant = f"tenant-{suffix}"
    other_tenant = f"other-{suffix}"
    actor = f"actor-{suffix}"
    try:
        project = store.create_project(
            tenant_id=tenant,
            name="PostgreSQL integration",
            description="",
            actor_id=actor,
        )
        other_project = store.create_project(
            tenant_id=other_tenant,
            name="Other tenant",
            description="",
            actor_id=actor,
        )
        create_job = store.enqueue_job(
            tenant_id=tenant,
            kind="recipe.generate",
            payload={"project_id": project.project_id},
            actor_id=actor,
        )
        formula = store.create_formula(
            tenant_id=tenant,
            project_id=project.project_id,
            name="Formula",
            kind="formula",
            payload=_payload("a"),
            actor_id=actor,
            change_note="initial",
            source_job_id=create_job.job_id,
        )
        replay = store.create_formula(
            tenant_id=tenant,
            project_id=project.project_id,
            name="Replay",
            kind="formula",
            payload=_payload("b"),
            actor_id=actor,
            change_note="replay",
            source_job_id=create_job.job_id,
        )
        assert replay.formula_id == formula.formula_id
        assert store.get_formula(
            tenant_id=other_tenant,
            project_id=other_project.project_id,
            formula_id=formula.formula_id,
        ) is None

        revision_jobs = [
            store.enqueue_job(
                tenant_id=tenant,
                kind="formula.revise",
                payload={"project_id": project.project_id},
                actor_id=actor,
            )
            for _ in range(2)
        ]

        def append(index: int):
            return store.append_formula_version(
                tenant_id=tenant,
                project_id=project.project_id,
                formula_id=formula.formula_id,
                expected_parent_version_id=formula.latest_version.version_id,
                change_kind="natural_language_revision",
                change_note=f"concurrent-{index}",
                payload=_payload(str(index + 2)),
                actor_id=actor,
                source_job_id=revision_jobs[index].job_id,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(append, index) for index in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except RuntimeError as error:
                assert "version conflict" in str(error)
        assert len(outcomes) == 1
        versions = store.list_formula_versions(
            tenant_id=tenant,
            project_id=project.project_id,
            formula_id=formula.formula_id,
        )
        assert [version.version_number for version in versions] == [2, 1]

        # Drain setup jobs before measuring the global SKIP LOCKED queue.
        while claimed := store.claim_job(
            worker_id=f"drain-{suffix}", lease_seconds=30
        ):
            store.complete_job(
                job_id=claimed.job_id,
                worker_id=f"drain-{suffix}",
                result={"drained": True},
            )
        for index in range(24):
            store.enqueue_job(
                tenant_id=tenant,
                kind="recipe.generate",
                payload={"index": index},
                actor_id=actor,
            )

        def claim_all(index: int) -> list[str]:
            worker = f"queue-{suffix}-{index}"
            claimed_ids = []
            while job := store.claim_job(worker_id=worker, lease_seconds=30):
                claimed_ids.append(job.job_id)
                store.complete_job(
                    job_id=job.job_id,
                    worker_id=worker,
                    result={"ok": True},
                )
            return claimed_ids

        with ThreadPoolExecutor(max_workers=8) as pool:
            batches = list(pool.map(claim_all, range(8)))
        claimed_ids = [job_id for batch in batches for job_id in batch]
        assert len(claimed_ids) == 24
        assert len(set(claimed_ids)) == 24
        tenant_jobs = store.list_jobs(tenant_id=tenant, limit=100)
        assert set(claimed_ids).issubset({job.job_id for job in tenant_jobs})
        assert store.list_jobs(tenant_id=other_tenant, limit=100) == []

        limiter = PostgresFixedWindowRateLimiter(store, requests=2, window_seconds=60)
        identity = f"integration-{suffix}"
        stale_identity = f"stale-{suffix}"
        assert limiter.allow(stale_identity, now=1_700_000_000)
        assert limiter.allow(identity, now=1_800_000_000)
        assert limiter.allow(identity, now=1_800_000_001)
        assert not limiter.allow(identity, now=1_800_000_002)
        stale_hash = hashlib.sha256(stale_identity.encode("utf-8")).hexdigest()
        with store.pool.connection() as connection:
            stale_count = connection.execute(
                """
                SELECT COUNT(*) AS count FROM perfumery_rate_limit_buckets
                WHERE identity_hash=%s
                """,
                (stale_hash,),
            ).fetchone()["count"]
        assert stale_count == 0
    finally:
        store.close()
