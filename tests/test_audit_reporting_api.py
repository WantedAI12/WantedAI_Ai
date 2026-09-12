import copy
import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from deploy.audit_api import register_audit_routes


def history():
    actor = {"actor_id": "system", "display_name": "system", "kind": "system"}
    return {
        "formula_id": "formula-01", "formula_name": "FORMULA 01",
        "events": [
            {"event_id": "event-2", "occurred_at": "2026-08-15T09:14:00+09:00",
             "category": "candidate", "event_type": "candidate.selected", "title": "실험 후보 선택",
             "actor": {"actor_id": "user-1", "display_name": "담당자", "kind": "user"},
             "reason": "후보 비교 후 선택", "changes": [{"field": "selected", "before": False, "after": True}]},
            {"event_id": "event-1", "occurred_at": "2026-08-14T10:02:00+09:00",
             "category": "formula", "event_type": "safety.evaluated", "title": "안전 조건 평가 결과 기록",
             "actor": actor, "model_version": "model-test-1", "data_version": "data-test-1",
             "changes": [{"field": "safety.status", "label": "평가 상태", "before": None, "after": "pending"}]},
        ],
        "versions": [{"version_id": "v1", "version_label": "V1", "created_at": "2026-08-14T09:00:00+09:00",
                      "actor": actor, "change_reason": "최초 기록"}],
    }


def parse_sse(text):
    result = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line and not line.startswith(":"))
        if "data" in fields:
            fields["data"] = json.loads(fields["data"])
            result.append(fields)
    return result


def app(rate_limit=lambda: None):
    value = FastAPI()
    register_audit_routes(value, rate_limit)
    return value


def test_report_is_chronological_json_attachment_without_invented_facts():
    with TestClient(app()) as client:
        response = client.post("/v1/reports/audit", json=history())
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith('attachment; filename="audit-report-')
    result = response.json()
    assert [row["event_id"] for row in result["audit_log"]] == ["event-1", "event-2"]
    assert result["audit_log"][0]["occurred_at"] == "2026-08-14T01:02:00Z"
    assert result["audit_log"][1]["model_version"] is None
    assert result["audit_log"][0]["changes"][0]["before"] is None
    assert result["summary"]["version_count"] == 1
    assert result["provenance"]["approval_inferred"] is False
    assert result["provenance"]["llm_calls"] == 0
    assert "recipe" not in result
    assert "backend_did_not_assert_complete_history" in result["warnings"]


def test_filter_counts_and_semantic_report_identity_ignore_input_order():
    body = history()
    with TestClient(app()) as client:
        first = client.post("/v1/reports/audit", json=body).json()
        body["events"].reverse()
        assert client.post("/v1/reports/audit", json=body).json()["report_id"] == first["report_id"]
        body["category_filter"] = "candidate"
        filtered = client.post("/v1/reports/audit", json=body).json()
    assert filtered["summary"]["supplied_event_count"] == 2
    assert filtered["summary"]["displayed_event_count"] == 1
    assert filtered["summary"]["event_counts_by_category"] == {"candidate": 1, "formula": 1, "data": 0}
    assert filtered["report_id"] != first["report_id"]
    assert "audit_log_is_category_filtered" in filtered["warnings"]


def test_audit_stream_has_framed_json_and_separate_original_event_ids():
    body = history()
    body["events"][0]["reason"] = "설명\nevent: injected\ndata: forged"
    with TestClient(app()) as client:
        response = client.post("/v1/audit-logs/stream", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    rows = parse_sse(response.text)
    assert [row["event"] for row in rows] == ["audit.started", "audit.entry", "audit.entry", "version.entry", "audit.summary", "done"]
    assert [row["data"]["sequence"] for row in rows] == list(range(len(rows)))
    assert len({row["id"] for row in rows}) == len(rows)
    assert rows[2]["data"]["data"]["reason"] == body["events"][0]["reason"]
    assert rows[-1]["data"]["data"]["status"] == "completed"


def test_report_sections_reconstruct_the_json_report():
    with TestClient(app()) as client:
        expected = client.post("/v1/reports/audit", json=history()).json()
        rows = parse_sse(client.post("/v1/reports/audit/stream", json=history()).text)
    rebuilt = dict(rows[0]["data"]["data"])
    for row in rows[1:-1]:
        section = row["data"]["data"]
        rebuilt[section["section"]] = section["value"]
    rebuilt.pop("generated_at")
    expected.pop("generated_at")
    assert rebuilt == expected


@pytest.mark.parametrize("change", ["duplicate_event", "duplicate_version", "cycle", "naive_time", "extra_recipe", "bad_hash", "nan"])
def test_invalid_history_fails_before_starting_stream(change):
    body = history()
    if change == "duplicate_event":
        body["events"].append(body["events"][0])
    elif change == "duplicate_version":
        body["versions"].append(body["versions"][0])
    elif change == "cycle":
        body["versions"][0]["parent_version_id"] = "v1"
    elif change == "naive_time":
        body["events"][0]["occurred_at"] = "2026-08-15T09:14:00"
    elif change == "extra_recipe":
        body["recipe"] = []
    elif change == "bad_hash":
        body["events"][0]["model_sha256"] = "unverified"
    else:
        body["events"][0]["changes"][0]["after"] = float("nan")
    with TestClient(app()) as client:
        response = client.post("/v1/audit-logs/stream", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


def test_empty_history_and_unavailable_parent_remain_explicit():
    body = history()
    body["events"] = []
    body["versions"][0]["parent_version_id"] = "unavailable"
    with TestClient(app()) as client:
        result = client.post("/v1/reports/audit", json=body).json()
    assert result["audit_log"] == []
    assert "no_audit_events_supplied" in result["warnings"]
    assert "some_parent_versions_not_supplied" in result["warnings"]


def test_history_limit_checks_content_length_and_actual_chunked_size():
    with TestClient(app()) as client:
        assert client.post("/v1/reports/audit", content=b"{}", headers={"Content-Length": "1048577"}).status_code == 413
        assert client.post("/v1/reports/audit", content=iter([b"x" * 600_000] * 2)).status_code == 413
        assert client.post("/v1/reports/audit", content=b"{}", headers={"Content-Length": "-1"}).status_code == 400


@pytest.mark.parametrize("path", ["/v1/audit-logs/stream", "/v1/reports/audit/stream"])
def test_unsupported_resume_is_rejected_without_running_report(path):
    with TestClient(app()) as client:
        response = client.post(path, json=history(), headers={"Last-Event-ID": "old:3"})
    assert response.status_code == 409


def test_rate_limit_is_applied_and_inputs_not_mutated():
    def limited():
        raise HTTPException(status_code=429, detail="limited")
    body = history()
    original = copy.deepcopy(body)
    with TestClient(app(limited)) as client:
        assert client.post("/v1/reports/audit", json=body).status_code == 429
    assert body == original
