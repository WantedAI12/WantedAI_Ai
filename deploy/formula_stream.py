"""Bounded SSE transport around the existing synchronous formula engine."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from queue import Empty, Queue
import threading
import time
from uuid import uuid4

import anyio
from fastapi import HTTPException, Request, Response
from starlette.responses import StreamingResponse


STAGES = {
    "RECEIVED": (0, "요청 접수 완료"),
    "INGREDIENT_SCREENING": (15, "원료 후보 필터링 중"),
    "SAFETY_CHECK": (55, "원료 안전·가격·공급 조건 대조 완료; 최종 조향식 승인은 별도"),
    "RATIO_OPTIMIZATION": (80, "배합비 최적화 중"),
    "TEMPORAL_PROFILE": (95, "시간별 향 프로필 최종 평가 중"),
    "DONE": (100, "계산 결과 전송 완료"),
}


class StreamSession:
    def __init__(self, release):
        self.stream_id = uuid4().hex
        self.queue = Queue(maxsize=16)  # At most six stages + result/error + done.
        self.closed = threading.Event()
        self._finished = False
        self._released = False
        self._lock = threading.Lock()
        self._release = release
        self._percent = -1

    def publish(self, event, data):
        if not self.closed.is_set():
            # Encode before publishing; malformed results become explicit errors.
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            self.queue.put_nowait((event, encoded))

    def progress(self, stage):
        percent, message = STAGES[stage]
        # Wider-support retries must not rewind the frontend's progress bar.
        if percent > self._percent:
            self._percent = percent
            self.publish("progress", {"stage": stage, "percent": percent, "message": message})

    def _release_when_idle(self):
        if self.closed.is_set() and self._finished and not self._released:
            self._released = True
            self._release()

    def disconnect(self):
        with self._lock:
            self.closed.set()
            self._release_when_idle()

    def finish(self):
        with self._lock:
            self._finished = True
            self._release_when_idle()


class FormulaStreamResponse(StreamingResponse):
    def __init__(self, session, *, heartbeat_seconds, timeout_seconds):
        self.session = session
        super().__init__(self.events(heartbeat_seconds, timeout_seconds), media_type="text/event-stream",
                         headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no",
                                  "X-Perfumery-Stream-ID": session.stream_id})

    async def events(self, heartbeat_seconds, timeout_seconds):
        start = last_sent = time.monotonic()
        sequence = 0

        def frame(event, encoded):
            nonlocal sequence
            sequence += 1
            return f"id: {self.session.stream_id}:{sequence}\nevent: {event}\ndata: {encoded}\n\n".encode("utf-8")

        while not self.session.closed.is_set():
            now = time.monotonic()
            if now - start >= timeout_seconds:
                yield frame("error", json.dumps({"code": "STREAM_TIMEOUT", "message": "스트림 제한 시간을 초과했습니다.",
                                                "retryable": True, "http_status": 504}))
                yield frame("done", '{"status":"failed"}')
                return
            try:
                event, encoded = self.session.queue.get_nowait()
            except Empty:
                if now - last_sent >= heartbeat_seconds:
                    yield b": keep-alive\n\n"
                    last_sent = now
                await anyio.sleep(min(0.05, heartbeat_seconds))
                continue
            yield frame(event, encoded)
            last_sent = now
            if event == "done":
                return
            await anyio.sleep(0)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # CPU work is not force-killed. The slot stays reserved until it ends.
            self.session.disconnect()


def stream_error(error):
    if isinstance(error, HTTPException):
        status = error.status_code
        if status == 503:
            return {"code": "INFERENCE_BUSY", "message": "추론 처리 용량이 부족합니다.", "retryable": True, "http_status": status}
        if status == 429:
            return {"code": "RATE_LIMITED", "message": "요청 한도를 초과했습니다.", "retryable": True, "http_status": status}
        return {"code": "INVALID_REQUEST_OR_RUNTIME", "message": "요청 조건 또는 런타임 설정을 확인하세요.", "retryable": False, "http_status": status}
    if isinstance(error, TimeoutError):
        return {"code": "INFERENCE_TIMEOUT", "message": "추론 제한 시간을 초과했습니다.", "retryable": True, "http_status": 504}
    # Exception text may contain paths, credentials or private model metadata.
    return {"code": "INFERENCE_FAILED", "message": "조향 계산 중 오류가 발생했습니다.", "retryable": False, "http_status": 500}


class FormulaStreams:
    def __init__(self, *, max_streams=4, heartbeat_seconds=10., timeout_seconds=285.):
        self._slots = threading.BoundedSemaphore(max_streams)
        self._pool = ThreadPoolExecutor(max_workers=max_streams, thread_name_prefix="formula-sse")
        self.heartbeat_seconds, self.timeout_seconds = heartbeat_seconds, timeout_seconds

    def open(self, generate, request):
        if not self._slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="formula stream capacity exhausted", headers={"Retry-After": "5"})
        session = StreamSession(self._slots.release)

        def work():
            try:
                session.progress("RECEIVED")
                response = Response()
                result = generate(request, response, rate_limited=True, progress_callback=session.progress)
                session.publish("result", result)  # Exactly the existing FormulaGenerationResponse.
                session.progress("DONE")
                session.publish("done", {"status": "completed", "cache_status": response.headers.get("X-Perfumery-Cache")})
            except Exception as error:
                session.publish("error", stream_error(error))
                session.publish("done", {"status": "failed"})
            finally:
                session.finish()

        try:
            self._pool.submit(work)
        except RuntimeError as error:
            session.disconnect()
            session.finish()
            raise HTTPException(status_code=503, detail="formula stream service is shutting down") from error
        return FormulaStreamResponse(session, heartbeat_seconds=self.heartbeat_seconds, timeout_seconds=self.timeout_seconds)

    def close(self):
        self._pool.shutdown(wait=False)  # Existing bounded workers finish safely.


def register_formula_stream(app, formula_type, generate, rate_limit, runtime_guard):
    streams = FormulaStreams()
    app.state.formula_streams = streams
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application) as state:
                yield state
        finally:
            streams.close()

    app.router.lifespan_context = lifespan

    @app.post("/v1/formulas/stream", tags=["Formula streaming"], response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    def formula_stream(request: formula_type, http_request: Request):
        if http_request.headers.get("last-event-id"):
            raise HTTPException(status_code=409, detail="restart from the original POST body; Last-Event-ID resume is not supported")
        rate_limit()
        try:
            runtime_guard()
        except ValueError as error:
            raise HTTPException(status_code=422, detail="runtime snapshot unavailable; reload the API") from error
        return streams.open(generate, request)
