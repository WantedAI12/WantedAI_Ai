"""PDF contract regressions. Synthetic API fixtures, not sensory/approval evidence."""

import copy
from datetime import date

from fastapi.testclient import TestClient
import pytest

from fragrance_ai.platform.rd_evidence import content_id
from tests.test_rd_contract import data as data  # Re-export the shared pytest fixture.
from tests.test_rd_contract import client_for, reviewed_request, POLICY
from tests.test_audit_reporting_api import app as audit_app, history, parse_sse


def evaluate(client, request=None):
    request = request or reviewed_request()
    prepared = client.post("/v2/briefs/prepare", json=request)
    assert prepared.status_code == 200, prepared.text
    response = client.post(
        "/v2/formulas/evaluate",
        json={**request, "confirmed_review_id": prepared.json()["review_id"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def stored(value, version="backend-v1"):
    return {
        "evaluation": value,
        "candidate_id": (value["candidates"] or value["diagnostic_candidates"])[0][
            "candidate_id"
        ],
        "backend_version_id": version,
    }


def rehash(value):
    value["result_id"] = content_id(
        {k: v for k, v in value.items() if k != "result_id"}
    )
    return value


def second_fixture(value):
    """A second deterministic-engine fixture under the same input contract."""
    output = copy.deepcopy(value)
    candidate = (output["candidates"] or output["diagnostic_candidates"])[0]
    candidate["formula_id"] += "-second-fixture"
    candidate["result"]["formula_id"] = candidate["formula_id"]
    candidate["target_match_score"] = 97.0
    candidate["candidate_id"] = content_id(
        {
            "review_id": output["review_id"],
            "formula_id": candidate["formula_id"],
            "assessment": candidate["evidence_assessment"],
        }
    )
    return rehash(output)


def test_partial_rd_answers_never_confirm_remaining_defaults(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        original = {
            "request": {"formula": {"brief": "green woody scent"}},
            "evidence_policy": POLICY,
        }
        first = client.post("/v2/briefs/prepare", json=original).json()
        assert len(first["missing_fields"]) == 4
        assert all(
            {"id", "field", "reason_code", "input_type", "required"} <= row.keys()
            for row in first["questions"]
        )
        response = client.post(
            "/v2/briefs/clarify",
            json={
                **original,
                "prepared_result_id": first["result_id"],
                "answers": {"request.formula.product_category": "eau_de_parfum"},
            },
        )
        assert response.status_code == 200, response.text
        value = response.json()
        assert (
            len(value["prepared"]["missing_fields"]) == 3
            and value["prepared"]["review_id"] is None
        )
        assert set(value["request"]["request"]["formula"]) == {
            "brief",
            "product_category",
        }
        response = client.post(
            "/v2/briefs/clarify",
            json={
                **value["request"],
                "prepared_result_id": value["prepared"]["result_id"],
                "answers": {
                    "request.formula.target_region": "EU",
                    "request.formula.product_concentration_percent": 15.0,
                    "request.formula.max_formula_cost_per_kg": 180.0,
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["prepared"]["status"] == "ready"
        assert response.json()["new_inference_count"] == 0 and not calls


@pytest.mark.parametrize(
    "answers",
    [
        {"request.formula.product_concentration_percent": True},
        {"request.formula.product_concentration_percent": 31.0},
        {"request.formula.target_region": "ZZ"},
        {"request.formula.product_category": "body_lotion"},
        {"request.formula.experimental_disable_safety": True},
        {},
    ],
)
def test_rd_answers_are_bounded_and_cannot_change_safety(data, answers):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        request = {
            "request": {"formula": {"brief": "green woody scent"}},
            "evidence_policy": POLICY,
        }
        result = client.post("/v2/briefs/prepare", json=request).json()
        response = client.post(
            "/v2/briefs/clarify",
            json={
                **request,
                "prepared_result_id": result["result_id"],
                "answers": answers,
            },
        )
        assert response.status_code == 422 and not calls


def test_rd_answer_staleness_and_actual_conflict_resolution(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        request = reviewed_request()
        request["request"]["formula"]["brief"] = "woody scent concentration 3%"
        value = client.post("/v2/briefs/prepare", json=request).json()
        question = next(
            q
            for q in value["questions"]
            if q["id"].endswith("product_concentration_percent")
        )
        assert question["reason_code"] == "text_and_structured_value_conflict"
        answers = {"request.formula.product_concentration_percent": 3.0}
        changed = copy.deepcopy(request)
        changed["evidence_policy"]["maximum_lead_time_days"] = 4
        assert (
            client.post(
                "/v2/briefs/clarify",
                json={
                    **changed,
                    "prepared_result_id": value["result_id"],
                    "answers": answers,
                },
            ).status_code
            == 409
        )
        response = client.post(
            "/v2/briefs/clarify",
            json={
                **request,
                "prepared_result_id": value["result_id"],
                "answers": answers,
            },
        )
        assert (
            response.status_code == 200
            and response.json()["prepared"]["status"] == "ready"
        )
        assert (
            response.json()["prepared"]["prepared"]["scenarios"][0][
                "product_concentration_percent"
            ]
            == 3.0
        )
        assert not calls


def test_rd_legacy_budget_answers_use_the_same_new_endpoint(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        request = reviewed_request()
        request["request"]["budget"] = {"amount": 4500.0, "per_volume_ml": 100.0}
        value = client.post("/v2/briefs/prepare", json=request).json()
        answers = {
            "budget.scope": "fragrance_only",
            "budget.product_density_g_ml": 1.0,
            "budget.krw_per_catalog_price_unit": 1300.0,
            "budget.price_basis_reference": "fixture FX rate",
            "budget.price_as_of": date.today().isoformat(),
        }
        result = client.post(
            "/v2/briefs/clarify",
            json={
                **request,
                "prepared_result_id": value["result_id"],
                "answers": answers,
            },
        )
        assert result.status_code == 200, result.text
        assert result.json()["prepared"]["status"] == "ready" and not calls


def test_saved_comparison_survives_a_new_app_and_never_uses_candidate_cache(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        first = evaluate(client)
    other, other_calls = client_for(catalog, factory())
    with other:
        second = second_fixture(first)
        result = other.post(
            "/v2/formulas/compare",
            json={"candidates": [stored(first), stored(second, "backend-v2")]},
        )
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["pairs"][0]["score_difference_points"] == 1.0
        assert body["pairs"][0]["composition"]["changes"] == []
        assert body["new_inference_count"] == 0 and not other_calls and len(calls) == 1
        assert not body["provenance"]["source_authenticity_verified"]
        assert (
            not body["automatic_ranking_performed"]
            and not body["recommendation_decision_performed"]
        )


@pytest.mark.parametrize(
    "change,field",
    [("concentration", "scenario"), ("runtime", "runtime"), ("goal", "intent")],
)
def test_different_snapshot_bases_do_not_get_a_score_improvement(data, change, field):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        first = evaluate(client)
        second = second_fixture(first)
        if change == "concentration":
            second["input_snapshot"]["scenario"]["product_concentration_percent"] = 3.0
            second["candidates"][0]["product_concentration_percent"] = 3.0
        if change == "runtime":
            second["contract"]["runtime"]["scientific_model_version"] = (
                "another-model-fixture"
            )
        if change == "goal":
            second["input_snapshot"]["intent"]["target_profile"] = {"citrus": 1.0}
        result = client.post(
            "/v2/formulas/compare",
            json={"candidates": [stored(first), stored(rehash(second), "v2")]},
        ).json()
        pair = result["pairs"][0]
        assert (
            field in pair["different_basis_fields"]
            and pair["score_difference_points"] is None
        )
        assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "gates",
        "unknown_status",
        "duplicate_lines",
        "bad_total",
        "missing_context",
        "wrong_id",
    ],
)
def test_invalid_saved_results_are_not_treated_as_valid_candidates(data, change):
    catalog, factory, *_ = data
    client, _ = client_for(catalog, factory())
    with client:
        first = evaluate(client)
        second = second_fixture(first)
        row = second["candidates"][0]
        if change == "hash":
            row["target_match_score"] = 100.0
        if change == "gates":
            row["rd_gates"]["registered_evidence"] = False
        if change == "unknown_status":
            row["status"] = "approved"
        if change == "duplicate_lines":
            row["result"]["recipe"] *= 2
        if change == "bad_total":
            row["result"]["recipe"][0]["concentrate_percent"] = 25.0
        if change == "missing_context":
            second.pop("input_snapshot")
        if change == "wrong_id":
            row["candidate_id"] = "0" * 64
        if change != "hash":
            rehash(second)
        response = client.post(
            "/v2/formulas/compare",
            json={"candidates": [stored(first), stored(second, "v2")]},
        )
        assert response.status_code == 422, response.text


def test_saved_revision_requires_new_confirmation_and_keeps_parent_lineage(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        first = evaluate(client)
        original = copy.deepcopy(first)
        response = client.post(
            "/v2/briefs/revise",
            json={"source": stored(first), "instruction": "less woody"},
        )
        assert response.status_code == 200, response.text
        revision = response.json()
        assert revision["prepared"]["status"] == "ready"
        assert len(calls) == 1 and first == original
        assert revision["prepared"]["review_id"] != first["review_id"]
        request = revision["request"]
        assert request["revision"]["parent_backend_version_id"] == "backend-v1"
        assert (
            client.post(
                "/v2/formulas/evaluate",
                json={**request, "confirmed_review_id": first["review_id"]},
            ).status_code
            == 409
        )
        response = client.post(
            "/v2/formulas/evaluate",
            json={**request, "confirmed_review_id": revision["prepared"]["review_id"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["revision"]["parent_result_id"] == first["result_id"]
        assert (
            response.json()["candidates"][0]["revision_composition_diff"][
                "input_normalized"
            ]
            is False
        )
        assert len(calls) == 2 and first == original


def test_capabilities_explain_product_routes_units_and_storage(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    with client:
        capabilities = client.get("/v1/ai/capabilities").json()["integration_contract"]
        routes = capabilities["operations"]
        assert (
            routes["compare_saved"]["registered"]
            and routes["clarify"]["runtime_available"]
        )
        assert not routes["conditioned_product_prediction"]["runtime_available"]
        assert (
            capabilities["product_routing"]["body_lotion"][
                "legacy_formula_request_supported"
            ]
            is False
        )
        assert (
            capabilities["product_routing"]["body_wash"][
                "rinse_deposition_in_legacy_formula_model"
            ]
            is False
        )
        assert (
            capabilities["units"]["concentrate_percent"]
            != capabilities["units"]["finished_product_percent"]
        )
        assert (
            not capabilities["workflow"]["comparison_requires_active_cache"]
            and not calls
        )
        schema = client.get("/openapi.json").json()
        for key in ("clarify", "compare_saved", "revise_saved_intent"):
            assert routes[key]["path"] in schema["paths"]


def test_audit_scope_is_inclusive_and_stream_matches_filtered_json():
    body = history()
    body["snapshot_id"] = "backend-snapshot-17"
    body["events"][0]["version_id"] = "v2"
    body["events"][1]["version_id"] = "v1"
    body["selected_version_ids"] = ["v2"]
    body["period_start"] = body["events"][0]["occurred_at"]
    body["period_end"] = body["events"][0]["occurred_at"]
    with TestClient(audit_app()) as client:
        response = client.post("/v1/reports/audit", json=body)
        assert response.status_code == 200
        report = response.json()
        assert [r["event_id"] for r in report["audit_log"]] == ["event-2"]
        assert report["version_history"] == []
        assert report["summary"]["excluded_event_count"] == 1
        assert report["summary"]["backend_snapshot_id"] == "backend-snapshot-17"
        rows = parse_sse(client.post("/v1/reports/audit/stream", json=body).text)
        rebuilt = dict(rows[0]["data"]["data"])
        for row in rows[1:-1]:
            section = row["data"]["data"]
            rebuilt[section["section"]] = section["value"]
        assert rebuilt["report_id"] == report["report_id"]
        assert (
            rebuilt["audit_log"] == report["audit_log"]
            and rebuilt["summary"] == report["summary"]
        )
        assert rows[-1]["data"]["data"]["status"] == "completed"


def test_audit_filter_does_not_guess_event_version_from_unrelated_references():
    body = history()
    body["events"][0]["reference_ids"] = ["v1"]
    body["selected_version_ids"] = ["v1"]
    with TestClient(audit_app()) as client:
        report = client.post("/v1/reports/audit", json=body).json()
    assert not report["audit_log"]
    assert (
        "events_without_version_id_excluded_from_version_filter" in report["warnings"]
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"selected_version_ids": ["v1", "v1"]},
        {"selected_version_ids": []},
        {"period_start": "2026-09-02T00:00:00Z", "period_end": "2026-09-01T00:00:00Z"},
        {"period_start": "2026-09-01T00:00:00"},
    ],
)
def test_invalid_audit_scope_is_rejected_before_stream(patch):
    with TestClient(audit_app()) as client:
        assert (
            client.post(
                "/v1/audit-logs/stream", json={**history(), **patch}
            ).status_code
            == 422
        )
