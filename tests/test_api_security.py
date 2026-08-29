import asyncio
import threading
from dataclasses import dataclass

import pytest

from fragrance_ai.api import SlidingWindowRateLimiter, TokenAuthorizer, create_app
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog


def test_token_authorizer_roles_and_rate_limit():
    authorizer = TokenAuthorizer.from_plaintext(
        {"secret-formulator": ("operator-1", "formulator")}
    )
    principal = authorizer.authenticate("secret-formulator")
    assert principal is not None
    assert authorizer.permits(principal, "recipe:create")
    assert not authorizer.permits(principal, "audit:verify")
    assert authorizer.authenticate("wrong") is None
    limiter = SlidingWindowRateLimiter(requests=2, window_seconds=10)
    assert limiter.allow("operator", now=0)
    assert limiter.allow("operator", now=1)
    assert not limiter.allow("operator", now=2)
    assert limiter.allow("operator", now=11)


def test_authenticated_api_uses_roles_and_appends_audit_events(tmp_path):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    @dataclass
    class FakeResult:
        def to_dict(self):
            return {
                "status": "prototype_ready",
                "formula_id": "sha256:" + "a" * 64,
                "olfactory_validation_status": "simulation_only",
                "actual_olfactory_similarity_score": None,
            }

    class FakeAI:
        def create_recipe(self, brief, constraints):
            assert brief == "clean citrus"
            return FakeResult()

    audit = AppendOnlyAuditLog(tmp_path / "audit.db")
    authorizer = TokenAuthorizer.from_plaintext(
        {
            "formulator-token": ("operator-1", "formulator"),
            "auditor-token": ("auditor-1", "auditor"),
        }
    )
    app = create_app(
        ai_factory=FakeAI,
        authorizer=authorizer,
        audit_log=audit,
        rate_limiter=SlidingWindowRateLimiter(100, 60),
    )
    client = TestClient(app)
    client.__enter__()
    assert client.post("/v1/recipes", json={"brief": "clean citrus"}).status_code == 401
    denied = client.get(
        "/v1/audit/verify",
        headers={"Authorization": "Bearer formulator-token"},
    )
    assert denied.status_code == 403
    response = client.post(
        "/v1/recipes",
        json={"brief": "clean citrus"},
        headers={"Authorization": "Bearer formulator-token"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["actual_olfactory_similarity_score"] is None
    verified = client.get(
        "/v1/audit/verify",
        headers={"Authorization": "Bearer auditor-token"},
    )
    assert verified.status_code == 200
    assert verified.json()["passed"]
    assert audit.verify()["events"] == 3
    client.__exit__(None, None, None)
    audit.close()


def test_twenty_parallel_requests_have_no_cross_thread_audit_failure(tmp_path):
    pytest.importorskip("fastapi")
    import httpx

    @dataclass
    class FakeResult:
        marker: str

        def to_dict(self):
            return {
                "status": "prototype_ready",
                "formula_id": "sha256:" + self.marker * 64,
                "olfactory_validation_status": "simulation_only",
            }

    class FakeAI:
        def create_recipe(self, brief, constraints):
            return FakeResult("a" if brief.startswith("A") else "b")

    audit = AppendOnlyAuditLog(tmp_path / "parallel-audit.db")
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"parallel-token": ("operator-1", "formulator")}
        ),
        audit_log=audit,
        max_concurrent_inference=4,
        rate_limiter=SlidingWindowRateLimiter(requests=40, window_seconds=60),
    )

    async def exercise():
        app.state.warm_inference_runtime()
        transport = httpx.ASGITransport(app=app)
        capacity = asyncio.Semaphore(4)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test.local",
            ) as client:
                async def invoke(index: int):
                    async with capacity:
                        return await client.post(
                            "/v1/recipes",
                            headers={"Authorization": "Bearer parallel-token"},
                            json={"brief": f"{'A' if index % 2 == 0 else 'B'}-{index}"},
                        )

                return await asyncio.gather(*(invoke(index) for index in range(20)))
        finally:
            app.state.close_inference_runtime()

    responses = asyncio.run(exercise())
    assert all(response.status_code == 200 for response in responses)
    assert len({response.json()["request_id"] for response in responses}) == 20
    assert audit.verify()["events"] == 40
    audit.close()


def test_async_recipe_keeps_audit_io_off_the_event_loop(tmp_path):
    pytest.importorskip("fastapi")
    import httpx

    @dataclass
    class FakeResult:
        def to_dict(self):
            return {
                "status": "prototype_ready",
                "formula_id": "sha256:" + "d" * 64,
                "olfactory_validation_status": "simulation_only",
            }

    class FakeAI:
        def create_recipe(self, brief, constraints):
            return FakeResult()

    append_threads: list[int] = []

    class RecordingAudit(AppendOnlyAuditLog):
        def append(self, **fields):
            append_threads.append(threading.get_ident())
            return super().append(**fields)

    audit = RecordingAudit(tmp_path / "offloaded-audit.db")
    app = create_app(
        ai_factory=FakeAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"offload-token": ("operator-1", "formulator")}
        ),
        audit_log=audit,
    )

    async def exercise():
        loop_thread = threading.get_ident()
        app.state.warm_inference_runtime()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test.local",
            ) as client:
                response = await client.post(
                    "/v1/recipes",
                    headers={"Authorization": "Bearer offload-token"},
                    json={"brief": "clean citrus"},
                )
            return loop_thread, response
        finally:
            app.state.close_inference_runtime()

    loop_thread, response = asyncio.run(exercise())
    assert response.status_code == 200
    assert len(append_threads) == 2
    assert all(thread_id != loop_thread for thread_id in append_threads)
    audit.close()


def test_inference_engines_are_bounded_and_closed_on_owner_threads(tmp_path):
    pytest.importorskip("fastapi")
    import httpx

    @dataclass
    class FakeResult:
        def to_dict(self):
            return {
                "status": "prototype_ready",
                "formula_id": "sha256:" + "c" * 64,
                "olfactory_validation_status": "simulation_only",
            }

    owners: list[int] = []
    inference_threads: list[int] = []
    closed_threads: list[int] = []
    rendezvous = threading.Barrier(2)

    class ThreadBoundAI:
        def __init__(self):
            self.owner = threading.get_ident()
            owners.append(self.owner)

        def create_recipe(self, brief, constraints):
            assert threading.get_ident() == self.owner
            inference_threads.append(threading.get_ident())
            rendezvous.wait(timeout=2)
            return FakeResult()

        def close(self):
            assert threading.get_ident() == self.owner
            closed_threads.append(threading.get_ident())

    audit = AppendOnlyAuditLog(tmp_path / "thread-owned-audit.db")
    app = create_app(
        ai_factory=ThreadBoundAI,
        authorizer=TokenAuthorizer.from_plaintext(
            {"thread-token": ("operator-1", "formulator")}
        ),
        audit_log=audit,
        max_concurrent_inference=2,
        rate_limiter=SlidingWindowRateLimiter(requests=10, window_seconds=60),
    )

    async def exercise():
        app.state.warm_inference_runtime()
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test.local",
            ) as client:
                return await asyncio.gather(
                    *(
                        client.post(
                            "/v1/recipes",
                            headers={"Authorization": "Bearer thread-token"},
                            json={"brief": f"request-{index}"},
                        )
                        for index in range(2)
                    )
                )
        finally:
            app.state.close_inference_runtime()

    responses = asyncio.run(exercise())
    assert [response.status_code for response in responses] == [200, 200]
    assert len(owners) == app.state.inference_engine_capacity == 2
    assert len(set(owners)) == 2
    assert sorted(inference_threads) == sorted(owners)
    assert sorted(closed_threads) == sorted(owners)
    assert audit.verify()["events"] == 4
    audit.close()
