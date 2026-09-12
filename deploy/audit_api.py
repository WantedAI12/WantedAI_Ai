"""Stateless backend-history reports and SSE; never runs recipe inference."""

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

import anyio
from fastapi import HTTPException, Request
from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, JsonValue,
    field_validator, model_validator,
)
from starlette.responses import JSONResponse, StreamingResponse


MAX_HISTORY_BYTES = 1_048_576
AUDIT_PATHS = frozenset({
    "/v1/audit-logs/stream", "/v1/reports/audit", "/v1/reports/audit/stream",
})
JSON_INPUT_PATHS = AUDIT_PATHS | {"/v1/formulas/stream"}
SNAPSHOT_INPUT_PATHS = frozenset({"/v2/formulas/compare", "/v2/briefs/revise"})
JSON_INPUT_PATHS |= SNAPSHOT_INPUT_PATHS | {
    "/v2/briefs/prepare", "/v2/briefs/clarify", "/v2/formulas/evaluate", "/v2/formulas/reassess",
    "/v2/formulas/assess-evidence", "/v2/formulas/change-impact",
}
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Category = Literal["candidate", "formula", "data"]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Actor(StrictInput):
    actor_id: Identifier
    display_name: str = Field(min_length=1, max_length=160)
    kind: Literal["system", "user", "service"]


class Change(StrictInput):
    field: str = Field(min_length=1, max_length=160)
    label: str | None = Field(default=None, max_length=160)
    before: JsonValue
    after: JsonValue


class VersionEvidence(StrictInput):
    model_version: str | None = Field(default=None, max_length=200)
    data_version: str | None = Field(default=None, max_length=200)
    model_sha256: Digest | None = None
    data_sha256: Digest | None = None


class AuditEntry(VersionEvidence):
    event_id: Identifier
    occurred_at: AwareDatetime
    category: Category
    event_type: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_.-]*$")
    title: str = Field(min_length=1, max_length=300)
    actor: Actor
    reason: str | None = Field(default=None, max_length=2000)
    changes: list[Change] = Field(default_factory=list, max_length=100)
    reference_ids: list[Identifier] = Field(default_factory=list, max_length=30)
    version_id: Identifier | None = None

    @field_validator("occurred_at")
    @classmethod
    def utc_time(cls, value):
        return value.astimezone(timezone.utc)


class VersionEntry(VersionEvidence):
    version_id: Identifier
    version_label: str = Field(min_length=1, max_length=160)
    parent_version_id: Identifier | None = None
    created_at: AwareDatetime
    actor: Actor
    change_reason: str | None = Field(default=None, max_length=2000)
    changes: list[Change] = Field(default_factory=list, max_length=100)

    @field_validator("created_at")
    @classmethod
    def utc_time(cls, value):
        return value.astimezone(timezone.utc)


class AuditHistoryRequest(StrictInput):
    schema_version: Literal["audit-history-request/v1"] = "audit-history-request/v1"
    formula_id: Identifier
    formula_name: str = Field(min_length=1, max_length=160)
    events: list[AuditEntry] = Field(max_length=500)
    versions: list[VersionEntry] = Field(default_factory=list, max_length=200)
    category_filter: Literal["all", "candidate", "formula", "data"] = "all"
    # This is a backend assertion, not independently verified completeness.
    history_complete: bool = False
    snapshot_id: Identifier | None = None
    selected_version_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=200)
    period_start: AwareDatetime | None = None
    period_end: AwareDatetime | None = None

    @field_validator("period_start", "period_end")
    @classmethod
    def utc_period(cls, value):
        return value.astimezone(timezone.utc) if value is not None else None

    @model_validator(mode="after")
    def validate_history(self):
        if self.period_start is not None and self.period_end is not None and self.period_start > self.period_end:
            raise ValueError("period_start must not exceed period_end")
        if self.selected_version_ids is not None:
            if len(set(self.selected_version_ids)) != len(self.selected_version_ids):
                raise ValueError("selected_version_ids must be unique")
            self.selected_version_ids = sorted(self.selected_version_ids)
        if len({row.event_id for row in self.events}) != len(self.events):
            raise ValueError("event_id must be unique within this formula history")
        parents = {row.version_id: row.parent_version_id for row in self.versions}
        if len(parents) != len(self.versions):
            raise ValueError("version_id must be unique within this formula history")
        for start in parents:
            seen, current = set(), start
            while current in parents:
                if current in seen:
                    raise ValueError("version history contains a parent cycle")
                seen.add(current)
                current = parents[current]
        # JSON values nested in changes must also reject NaN/Infinity.
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


class AuditBodyLimit:
    """Bound these JSON inputs before FastAPI parses them, including chunked bodies."""

    def __init__(self, app, max_bytes=MAX_HISTORY_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in JSON_INPUT_PATHS or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        max_bytes = self.max_bytes * (8 if scope.get("path") in SNAPSHOT_INPUT_PATHS else 1)
        lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        if lengths:
            try:
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError
                declared = int(lengths[0])
            except ValueError:
                return await JSONResponse({"detail": "invalid Content-Length"}, status_code=400)(scope, receive, send)
            if declared > max_bytes:
                return await JSONResponse({"detail": "JSON body exceeds the endpoint size limit"}, status_code=413)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > max_bytes:
                return await JSONResponse({"detail": "JSON body exceeds the endpoint size limit"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body, delivered = b"".join(chunks), False

        def invalid_number(value):
            raise ValueError("non-finite JSON number")

        def finite_float(value):
            number = float(value)
            return number if math.isfinite(number) else invalid_number(value)

        try:
            # Reject non-finite numbers before FastAPI's error renderer tries
            # to echo their invalid JSON input, which would itself raise 500.
            json.loads(body, parse_constant=invalid_number, parse_float=finite_float)
        except (ValueError, RecursionError):
            return await JSONResponse({"detail": "invalid JSON or non-finite number"}, status_code=422)(scope, receive, send)

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def build_report(request: AuditHistoryRequest) -> dict:
    """Normalize supplied facts; do not infer approvals, actors, or missing changes."""
    events = sorted(request.events, key=lambda row: (row.occurred_at, row.event_id))
    versions = sorted(request.versions, key=lambda row: (row.created_at, row.version_id))
    def in_period(moment):
        return ((request.period_start is None or moment >= request.period_start)
                and (request.period_end is None or moment <= request.period_end))

    version_filter = set(request.selected_version_ids) if request.selected_version_ids is not None else None
    def event_in_scope(row):
        return (request.category_filter in ("all", row.category) and in_period(row.occurred_at)
                and (version_filter is None or row.version_id in version_filter))

    selected = [row for row in events if event_in_scope(row)]
    selected_versions = [row for row in versions if in_period(row.created_at)
                         and (version_filter is None or row.version_id in version_filter)]
    canonical = request.model_dump(mode="json")
    canonical["events"] = [row.model_dump(mode="json") for row in events]
    canonical["versions"] = [row.model_dump(mode="json") for row in versions]
    report_id = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode("utf-8")).hexdigest()
    counts = Counter(row.category for row in events)
    warnings = []
    if not request.history_complete:
        warnings.append("backend_did_not_assert_complete_history")
    if request.category_filter != "all":
        warnings.append("audit_log_is_category_filtered")
    if request.period_start is not None or request.period_end is not None:
        warnings.append("audit_report_is_period_filtered")
    if version_filter is not None:
        warnings.append("audit_report_is_version_filtered")
        if any(row.version_id is None for row in events):
            warnings.append("events_without_version_id_excluded_from_version_filter")
        if version_filter - {row.version_id for row in versions}:
            warnings.append("some_selected_version_records_not_supplied")
    if not events:
        warnings.append("no_audit_events_supplied")
    if any(not row.model_version and not row.model_sha256 for row in events):
        warnings.append("some_events_have_no_model_identifier")
    if any(not row.data_version and not row.data_sha256 for row in events):
        warnings.append("some_events_have_no_data_identifier")
    identifiers = {row.version_id for row in versions}
    if any(row.parent_version_id and row.parent_version_id not in identifiers for row in versions):
        warnings.append("some_parent_versions_not_supplied")
    displayed_ids = {row.version_id for row in selected_versions}
    if any(row.parent_version_id in identifiers - displayed_ids for row in selected_versions):
        warnings.append("some_parent_versions_outside_selected_scope")
    times = [row.occurred_at for row in events] + [row.created_at for row in versions]
    return {
        "schema_version": "audit-report/v1", "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formula": {"formula_id": request.formula_id, "formula_name": request.formula_name},
        "summary": {
            "supplied_event_count": len(events), "displayed_event_count": len(selected),
            "event_counts_by_category": {key: counts[key] for key in ("candidate", "formula", "data")},
            "version_count": len(selected_versions), "supplied_version_count": len(versions),
            "excluded_event_count": len(events) - len(selected),
            "excluded_version_count": len(versions) - len(selected_versions),
            "category_filter": request.category_filter,
            "backend_snapshot_id": request.snapshot_id,
            "filters": {"selected_version_ids": request.selected_version_ids,
                        "period_start": request.period_start.isoformat() if request.period_start else None,
                        "period_end": request.period_end.isoformat() if request.period_end else None,
                        "period_boundaries": "inclusive",
                        "version_filter_uses": "explicit_event_version_id_not_reference_ids",
                        "version_history_period_basis": "created_at",
                        "category_filter_applies_to": "audit_events_only"},
            "history_complete_asserted_by_backend": request.history_complete,
            "period_start": min(times).isoformat() if times else None,
            "period_end": max(times).isoformat() if times else None,
        },
        "audit_log": [row.model_dump(mode="json") for row in selected],
        "version_history": [row.model_dump(mode="json") for row in selected_versions],
        "provenance": {
            "source": "backend_supplied_history", "verification": "not_independently_verified",
            "missing_events_reconstructed": False, "approval_inferred": False,
            "recipe_inference_performed": False, "llm_calls": 0,
        },
        "warnings": warnings,
    }


def encode_event(stream_id, sequence, event, data):
    value = {"schema_version": "audit-sse/v1", "stream_id": stream_id,
             "sequence": sequence, "data": data}
    return (f"id: {stream_id}:{sequence}\nevent: {event}\ndata: "
            + json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n\n").encode("utf-8")


async def report_events(report, stream_id, kind):
    """Finite snapshot stream, no retained session and no background inference."""
    sequence = 0
    metadata = {key: report[key] for key in ("schema_version", "report_id", "generated_at", "formula")}
    yield encode_event(stream_id, sequence, f"{kind}.started", metadata)
    sequence += 1
    await anyio.sleep(0)
    if kind == "audit":
        for event, key in (("audit.entry", "audit_log"), ("version.entry", "version_history")):
            for row in report[key]:
                yield encode_event(stream_id, sequence, event, row)
                sequence += 1
                await anyio.sleep(0)
        yield encode_event(stream_id, sequence, "audit.summary",
                           {key: report[key] for key in ("summary", "provenance", "warnings")})
        sequence += 1
    else:
        for key in ("summary", "audit_log", "version_history", "provenance", "warnings"):
            yield encode_event(stream_id, sequence, "report.section", {"section": key, "value": report[key]})
            sequence += 1
            await anyio.sleep(0)
    yield encode_event(stream_id, sequence, "done", {
        "report_id": report["report_id"], "status": "completed",
        "events_before_done": sequence,
        "download_filename": f"audit-report-{report['report_id'][:16]}.json",
    })


def register_audit_routes(app, rate_limit):
    app.add_middleware(AuditBodyLimit)

    def prepare(request, http_request, *, streaming=False):
        if streaming and http_request.headers.get("last-event-id"):
            raise HTTPException(status_code=409, detail="snapshot streams do not support Last-Event-ID; resend the history and deduplicate by event_id")
        rate_limit()
        return build_report(request)

    def stream(report, kind):
        stream_id = uuid4().hex
        return StreamingResponse(report_events(report, stream_id, kind), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store, no-transform",
                                          "X-Accel-Buffering": "no", "X-Audit-Stream-ID": stream_id})

    @app.post("/v1/audit-logs/stream", tags=["Audit reports"], response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    def audit_stream(request: AuditHistoryRequest, http_request: Request):
        return stream(prepare(request, http_request, streaming=True), "audit")

    @app.post("/v1/reports/audit", tags=["Audit reports"])
    def audit_report(request: AuditHistoryRequest, http_request: Request):
        report = prepare(request, http_request)
        return JSONResponse(report, headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="audit-report-{report["report_id"][:16]}.json"',
        })

    @app.post("/v1/reports/audit/stream", tags=["Audit reports"], response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    def audit_report_stream(request: AuditHistoryRequest, http_request: Request):
        return stream(prepare(request, http_request, streaming=True), "report")
