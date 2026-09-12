import socket
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import httpx
from pydantic import BaseModel
import pytest
import uvicorn

from deploy.formula_stream import FormulaStreams, STAGES, register_formula_stream
from tests.test_audit_reporting_api import parse_sse


class Formula(BaseModel):
    brief: str


def app(generate, rate_limit=lambda: None):
    value = FastAPI()
    register_formula_stream(value, Formula, generate, rate_limit, lambda: None)
    return value


def test_monotone_stages_exact_result_and_success_terminal():
    result = {"recipe": [], "closest_candidate": [{"name": "연구 후보"}], "full_profile_target_met": False}

    def generate(request, response, *, progress_callback, rate_limited):
        assert rate_limited and request.brief == "different request"
        for stage in ("INGREDIENT_SCREENING", "SAFETY_CHECK", "RATIO_OPTIMIZATION",
                      "INGREDIENT_SCREENING", "RATIO_OPTIMIZATION", "TEMPORAL_PROFILE"):
            progress_callback(stage)
        response.headers["X-Perfumery-Cache"] = "miss"
        return result

    with TestClient(app(generate)) as client:
        response = client.post("/v1/formulas/stream", json={"brief": "different request"})
    assert response.status_code == 200
    rows = parse_sse(response.text)
    progress = [row["data"] for row in rows if row["event"] == "progress"]
    assert [row["stage"] for row in progress] == list(STAGES)
    assert [row["percent"] for row in progress] == [0, 15, 55, 80, 95, 100]
    assert [row["data"] for row in rows if row["event"] == "result"] == [result]
    assert rows[-1]["event"] == "done" and rows[-1]["data"] == {"status": "completed", "cache_status": "miss"}


@pytest.mark.parametrize("error,code,retryable", [
    (ValueError("private secret"), "INFERENCE_FAILED", False),
    (RuntimeError("private secret"), "INFERENCE_FAILED", False),
    (TimeoutError("private secret"), "INFERENCE_TIMEOUT", True),
    (HTTPException(status_code=503, detail="private secret"), "INFERENCE_BUSY", True),
    (HTTPException(status_code=422, detail="private secret"), "INVALID_REQUEST_OR_RUNTIME", False),
])
def test_midstream_errors_are_explicit_sanitized_and_have_no_result(error, code, retryable):
    def generate(*args, **kwargs):
        kwargs["progress_callback"]("INGREDIENT_SCREENING")
        raise error
    with TestClient(app(generate)) as client:
        response = client.post("/v1/formulas/stream", json={"brief": "request"})
    rows = parse_sse(response.text)
    assert "private secret" not in response.text
    assert not any(row["event"] == "result" for row in rows)
    assert rows[-2]["event"] == "error"
    assert rows[-2]["data"]["code"] == code and rows[-2]["data"]["retryable"] is retryable
    assert rows[-1]["data"]["status"] == "failed"


def test_cache_fast_path_does_not_invent_engine_stages():
    def generate(request, response, **kwargs):
        response.headers["X-Perfumery-Cache"] = "hit"
        return {"recipe": []}
    with TestClient(app(generate)) as client:
        rows = parse_sse(client.post("/v1/formulas/stream", json={"brief": "request"}).text)
    assert [row["data"]["stage"] for row in rows if row["event"] == "progress"] == ["RECEIVED", "DONE"]
    assert rows[-1]["data"]["cache_status"] == "hit"


def test_resume_schema_and_rate_failures_are_http_errors_before_stream():
    def forbidden(*a, **kw):
        pytest.fail("must not run inference")
    with TestClient(app(forbidden)) as client:
        assert client.post("/v1/formulas/stream", json={}).status_code == 422
        assert client.post("/v1/formulas/stream", json={"brief": "a"}, headers={"Last-Event-ID": "old:1"}).status_code == 409
    def limited():
        raise HTTPException(status_code=429, detail="limited")
    with TestClient(app(forbidden, limited)) as client:
        assert client.post("/v1/formulas/stream", json={"brief": "a"}).status_code == 429


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_http_first_chunk_heartbeat_and_disconnect_with_bounded_work(disconnect):
    """Real Uvicorn/TCP transport; a controlled worker proves no response buffering."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def generate(request, response, **kwargs):
        started.set()
        try:
            assert release.wait(8)
            return {"recipe": [], "message": "finished"}
        finally:
            finished.set()

    web = app(generate)
    service = web.state.formula_streams
    service.heartbeat_seconds = .05
    # Keep the configured admission bound at four and test saturation directly.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(web, log_level="error", lifespan="on", ws="none", http="h11"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    try:
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            with client.stream("POST", "/v1/formulas/stream", json={"brief": "test"}) as response:
                assert response.status_code == 200
                lines = response.iter_lines()
                first = []
                for line in lines:
                    first.append(line)
                    if line == "":
                        break
                assert '"stage":"RECEIVED"' in "\n".join(first)
                assert started.wait(1) and not finished.is_set()
                assert next(lines) == ": keep-alive"
                if not disconnect:
                    release.set()
                    remaining = "\n".join(lines)
                    assert "event: result" in remaining and "event: done" in remaining
            # On disconnect the CPU work is still accounted for until it ends.
            if disconnect:
                assert not finished.is_set()
                release.set()
            assert finished.wait(2)
            assert client.get("/openapi.json").status_code == 200
    finally:
        release.set()
        server.should_exit = True
        thread.join(5)
        sock.close()
    assert not thread.is_alive()


def test_disconnect_does_not_release_admission_until_cpu_work_finishes():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    service = FormulaStreams(max_streams=1)
    def generate(*args, **kwargs):
        entered.set()
        release.wait(5)
        finished.set()
        return {"recipe": []}
    try:
        response = service.open(generate, Formula(brief="one"))
        assert entered.wait(1)
        response.session.disconnect()
        with pytest.raises(HTTPException) as error:
            service.open(generate, Formula(brief="two"))
        assert error.value.status_code == 503
        release.set()
        assert finished.wait(1)
        deadline = time.monotonic() + 2
        while not response.session._released and time.monotonic() < deadline:
            time.sleep(.01)
        assert response.session._released
    finally:
        release.set()
        service.close()


def test_stream_timeout_is_explicit_and_does_not_mark_work_successful():
    release = threading.Event()
    def generate(*args, **kwargs):
        release.wait(5)
        return {"recipe": []}
    web = app(generate)
    web.state.formula_streams.timeout_seconds = .05
    try:
        with TestClient(web) as client:
            response = client.post("/v1/formulas/stream", json={"brief": "test"})
        rows = parse_sse(response.text)
        assert rows[-2]["data"]["code"] == "STREAM_TIMEOUT"
        assert rows[-1]["data"]["status"] == "failed"
        assert not any(row["event"] == "result" for row in rows)
    finally:
        release.set()


def test_real_engine_progress_observer_preserves_uncached_numerical_result(monkeypatch):
    from datetime import date
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    monkeypatch.setenv("PERFUMERY_AI_LOCAL_PROFILE", "disabled")
    stages = []
    constraints = RecipeConstraints(physics_search_population=1)
    with NaturalLanguagePerfumeryAI() as ai:
        ordinary = ai.create_recipe("clean fresh citrus woody musk", constraints, as_of=date(2026, 7, 11)).to_dict()
        observed = ai.create_recipe("clean fresh citrus woody musk", constraints, as_of=date(2026, 7, 11),
                                    progress_callback=stages.append).to_dict()
    assert stages == ["INGREDIENT_SCREENING", "SAFETY_CHECK", "RATIO_OPTIMIZATION", "TEMPORAL_PROFILE"]
    assert observed == ordinary
