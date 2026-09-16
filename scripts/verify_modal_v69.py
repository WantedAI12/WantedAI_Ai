"""Verify the actual authenticated V69 service; never log verification secrets."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.runtime_release_v69 import CORE_SHA256, RELEASE_ID, WHEEL_SHA256  # noqa: E402

BASE_URL = "https://junseong2im--perfumery-ai-core-web.modal.run"


def existing_proxy_fingerprint():
    listed = subprocess.run(
        [sys.executable, "-m", "modal", "workspace", "proxy-tokens", "list", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    if listed.returncode:
        raise RuntimeError("cannot verify existing credential inventory")
    identifiers = sorted(set(re.findall(r"wk-[A-Za-z0-9]+", listed.stdout)))
    return hashlib.sha256(json.dumps(identifiers).encode()).hexdigest()


def recipe_signature(value):
    rows = value.get("recipe") or value.get("closest_candidate") or []
    if not rows:
        raise ValueError("no recipe or candidate returned")
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def verify_stream(lines, expected):
    event, data, progress, results = None, [], [], []
    for line in [*lines, ""]:
        if not line:
            if data:
                value = json.loads("\n".join(data))
                if event == "error":
                    raise ValueError("explicit streaming error")
                if event == "progress":
                    percent = value["percent"]
                    if (
                        isinstance(percent, bool)
                        or not isinstance(percent, (int, float))
                        or not 0 <= percent <= 100
                    ):
                        raise ValueError("invalid stream percentage")
                    progress.append(percent)
                elif event == "result":
                    results.append(value)
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if progress != sorted(progress) or not progress or progress[-1] != 100:
        raise ValueError("nonmonotone or incomplete progress")
    if results != [expected]:
        raise ValueError("SSE result differs from the JSON contract")
    return {"progress": progress, "exact_result_match": True}


def audit_frames(text):
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines()
                      if ": " in line and not line.startswith(":"))
        if "data" in fields:
            fields["data"] = json.loads(fields["data"])
            frames.append(fields)
    if not frames or any(row.get("event") == "error" for row in frames):
        raise ValueError("audit stream missing or failed")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-verification", type=Path, required=True)
    parser.add_argument("--product-fixtures", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(
        (args.local_verification / "verification.json").read_text(encoding="utf-8")
    )
    if local["checkpoint_sha256"] != CORE_SHA256:
        raise ValueError("local comparison checkpoint differs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "release_id": RELEASE_ID,
        "base_url": BASE_URL,
        "scope": "remote_software_integration_not_human_similarity",
        "existing_backend_credentials_modified": False,
        "temporary_token_revoked": False,
        "checks": {},
        "seconds": {},
        "passed": False,
    }
    token_id, stage = None, "verify_existing_credential_inventory"
    before_credentials = existing_proxy_fingerprint()

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    try:
        stage = "create_temporary_verification_credential"
        created = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "workspace",
                "proxy-tokens",
                "create",
                "--json",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        identifier = re.search(r"wk-[A-Za-z0-9]+", created.stdout)
        secret = re.search(r"ws-[A-Za-z0-9]+", created.stdout)
        token_id = identifier.group() if identifier else None
        if created.returncode or not token_id or not secret:
            raise RuntimeError(
                "verification credential unavailable; raw output suppressed"
            )
        headers = {"Modal-Key": token_id, "Modal-Secret": secret.group()}
        with httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(310, connect=30),
            follow_redirects=False,
        ) as client:
            stage = "unauthenticated_health"
            response = client.get("/health")
            assert response.status_code in (401, 403)
            report["checks"]["unauthenticated_status"] = response.status_code

            def request(label, method, path, body=None):
                nonlocal stage
                stage = label
                start = time.perf_counter()
                response = client.request(method, path, headers=headers, json=body)
                report["seconds"][label] = round(time.perf_counter() - start, 4)
                report["checks"][label + "_http_status"] = response.status_code
                if response.status_code != 200:
                    raise ValueError("HTTP contract failed; response body suppressed")
                value = response.json()
                save(label + ".json", value)
                print(
                    json.dumps(
                        {
                            "phase": label,
                            "http_status": 200,
                            "seconds": report["seconds"][label],
                        }
                    ),
                    flush=True,
                )
                return value, response

            health, _ = request("health", "GET", "/health")
            assert (
                health["wheel_sha256"] == WHEEL_SHA256
                and health["gpu_required"] is False
            )
            capabilities, _ = request("capabilities", "GET", "/v1/ai/capabilities")
            core = capabilities["shared_formulation_model"]
            assert (
                core["checkpoint_sha256"] == CORE_SHA256
                and core["single_shared_checkpoint"] is True
            )
            assert (
                capabilities["odor_expression"]["prediction_model"]["model_sha256"]
                == CORE_SHA256
            )
            assert (
                capabilities["unified_product_model"]["shared_odor_backbone_sha256"]
                == CORE_SHA256
            )
            assert capabilities["stock_mixture_model"]["loaded"] is False
            report["checks"]["shared_checkpoint_sha256"] = CORE_SHA256
            catalog, _ = request("catalog", "GET", "/v1/catalog")
            assert catalog["connected_catalog_rows"] == local["catalog_rows"]
            report["checks"]["catalog_rows"] = catalog["connected_catalog_rows"]
            formula_body, formula_value = None, None
            for index, item in enumerate(local["api"]):
                value, _ = request(
                    f"reference_{index}", "POST", item["path"], item["request"]
                )
                if item["path"].endswith("/plan"):
                    assert value["learned_process"]["checkpoint_sha256"] == CORE_SHA256
                    assert value["learned_process"]["status"] == "source_consistent"
                elif item["path"].endswith("/emulsion-prediction"):
                    assert value["checkpoint_sha256"] == CORE_SHA256
                    assert value["fragrance_release_coefficients_inferred"] is False
                else:
                    score = value.get(
                        "calculated_profile_similarity", value.get("score")
                    )
                    assert score is not None and score >= item["score"] - 0.1
                    report["checks"][f"reference_{index}_score"] = score
                    if item["path"] == "/v1/formulas":
                        formula_body, formula_value = item["request"], value
                        assert (
                            value["perception_guidance"]["component_model_sha256"]
                            == CORE_SHA256
                        )
            assert formula_value is not None
            repeat, response = request(
                "formula_repeat", "POST", "/v1/formulas", formula_body
            )
            assert (
                repeat == formula_value
                and response.headers.get("X-Perfumery-Cache") == "hit"
            )
            report["checks"]["formula_cache_exact"] = True
            other, _ = request(
                "different_brief",
                "POST",
                "/v1/formulas",
                {
                    **formula_body,
                    "brief": "드라이한 시더우드와 샌달우드 향, 바닐라는 제외",
                },
            )
            assert recipe_signature(other) != recipe_signature(formula_value)
            report["checks"]["different_briefs_different_candidates"] = True
            for product in ("perfume", "body_lotion", "body_wash"):
                fixture = json.loads(
                    (args.product_fixtures / (product + ".json")).read_text(
                        encoding="utf-8"
                    )
                )
                assert (
                    fixture["response"]["model"]["shared_odor_backbone_sha256"]
                    == CORE_SHA256
                )
                value, _ = request(
                    product,
                    "POST",
                    "/v1/applications/unified/predict",
                    fixture["request"],
                )
                assert value["model"]["shared_odor_backbone_sha256"] == CORE_SHA256
                assert value["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
            stage = "formula_sse"
            with client.stream(
                "POST", "/v1/formulas/stream", headers=headers, json=formula_body
            ) as response:
                assert response.status_code == 200 and response.headers[
                    "content-type"
                ].startswith("text/event-stream")
                report["checks"]["formula_sse"] = verify_stream(
                    list(response.iter_lines()), formula_value
                )
            # These are explicitly supplied synthetic history rows; the AI
            # must not invent approvals or register them as actual evidence.
            history = {
                "formula_id": "v69-deployment-test", "formula_name": "TEST",
                "events": [{"event_id": "test-event", "occurred_at": "2026-09-13T00:00:00Z",
                            "category": "candidate", "event_type": "test.supplied",
                            "title": "Deployment integration fixture",
                            "actor": {"actor_id": "test", "display_name": "test", "kind": "system"}}],
                "versions": [],
            }
            audit, response = request("audit_report", "POST", "/v1/reports/audit", history)
            assert len(audit["audit_log"]) == 1 and audit["provenance"]["approval_inferred"] is False
            assert audit["provenance"]["llm_calls"] == 0
            assert response.headers["content-disposition"].startswith("attachment;")
            for label, path in (("audit_sse", "/v1/audit-logs/stream"),
                                ("audit_report_sse", "/v1/reports/audit/stream")):
                stage = label
                response = client.post(path, headers=headers, json=history)
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                rows = audit_frames(response.text)
                assert rows[-1]["event"] == "done"
                assert rows[-1]["data"]["data"]["status"] == "completed"
                if label == "audit_report_sse":
                    rebuilt = dict(rows[0]["data"]["data"])
                    for row in rows[1:-1]:
                        section = row["data"]["data"]
                        rebuilt[section["section"]] = section["value"]
                    assert {k: v for k, v in rebuilt.items() if k != "generated_at"} == {
                        k: v for k, v in audit.items() if k != "generated_at"}
                report["checks"][label] = True
            stock, _ = request(
                "legacy_stock_assay",
                "POST",
                "/v1/formulations/stock-mixture/predict",
                {
                    "basis": "relative_volume",
                    "components": [
                        {
                            "ingredient_id": name,
                            "stock_dilution": 0.01,
                            "solvent": "90% ethanol",
                            "relative_volume": 1.0,
                        }
                        for name in ("linalyl_acetate", "phenethyl_alcohol")
                    ],
                },
            )
            assert stock["application_domain"] == "stock_aliquot_assay"
            message = {"message": "머스크 없이 장미 향 바디로션으로 만들어줘"}
            assistant, _ = request("assistant", "POST", "/v1/ai/assistant", message)
            from scripts.verify_modal_v63 import verify_assistant_result

            verify_assistant_result(assistant)
            repeated, response = request(
                "assistant_repeat", "POST", "/v1/ai/assistant", message
            )
            assert repeated == assistant
            assert response.headers.get("X-Perfumery-LLM-Calls") == "0"
            assert response.headers.get("X-Perfumery-Language-Cache") == "hit"
            report["checks"]["language_cache_no_repeat_inference"] = True
            report["passed"] = True
    except Exception as error:
        report.update(passed=False, error_stage=stage, error_type=type(error).__name__)
    finally:
        if token_id:
            deleted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "workspace",
                    "proxy-tokens",
                    "delete",
                    token_id,
                    "-y",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
            )
            report["temporary_token_revoked"] = deleted.returncode == 0
        try:
            report["checks"]["existing_proxy_token_inventory_unchanged"] = (
                existing_proxy_fingerprint() == before_credentials
            )
        except Exception:
            report["checks"]["existing_proxy_token_inventory_unchanged"] = False
        save("report.json", report)
        print(json.dumps(report, ensure_ascii=True), flush=True)
    if (
        not report["passed"]
        or not report["temporary_token_revoked"]
        or not report["checks"]["existing_proxy_token_inventory_unchanged"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
