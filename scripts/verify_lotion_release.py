"""Bounded authenticated smoke check; never prints or persists proxy credentials."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"https://[a-zA-Z0-9-]+\.modal\.run", args.base_url):
        parser.error("an HTTPS Modal app origin without a path is required")
    if args.output.exists():
        parser.error("use a new evidence file")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from deploy.modal_app import WHEEL_SHA256, REGISTRY_SHA256
    from tests.test_lotion_reference_v23 import split_data

    report = {"base_url": args.base_url, "checks": {}, "request_seconds": {},
              "verification_scope": "deployed_API_software_not_lotion_accuracy", "temporary_token_revoked": False}
    token_id = None
    try:
        created = subprocess.run([sys.executable, "-m", "modal", "workspace", "proxy-tokens", "create", "--json"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
        raw = created.stdout
        identifier = re.search(r"wk-[A-Za-z0-9]+", raw)
        secret = re.search(r"ws-[A-Za-z0-9]+", raw)
        token_id = identifier.group() if identifier else None
        if created.returncode or token_id is None or secret is None:
            raise RuntimeError("temporary proxy credential creation failed; credential output suppressed")
        headers = {"Modal-Key": token_id, "Modal-Secret": secret.group()}
        with httpx.Client(base_url=args.base_url, timeout=55., follow_redirects=False) as client:
            unauthenticated = client.get("/v1/applications/body-lotion/references/LC-CCT-OW-01")
            assert unauthenticated.status_code in (401, 403), "proxy auth not enforced"
            report["checks"]["unauthenticated_status"] = unauthenticated.status_code

            def request(label, method, path, **kwargs):
                started = time.perf_counter()
                response = client.request(method, path, headers=headers, **kwargs)
                report["request_seconds"][label] = round(time.perf_counter() - started, 4)
                assert response.status_code == 200, f"{label} returned HTTP {response.status_code}"
                return response.json()

            health = request("health", "GET", "/health")
            assert health["wheel_sha256"] == WHEEL_SHA256 and health["registry_sha256"] == REGISTRY_SHA256
            report["checks"]["health"] = health
            catalog = request("catalog", "GET", "/v1/catalog")
            assert catalog["formulation_ready"] == 3758
            report["checks"]["active_candidates"] = catalog["formulation_ready"]
            reference = request("reference", "GET", "/v1/applications/body-lotion/references/LC-CCT-OW-01")
            assert reference["reference_id"] == "LC-CCT-OW-01" and not reference["ready_to_optimize"]
            assert sum(row["mass_percent"] for row in reference["application_context"]["base_components"]) == 100
            report["checks"]["reference_status"] = reference["status"]
            features = request("capabilities", "GET", "/v1/ai/capabilities")["features"]
            assert features["body_lotion_split_base_components"] and not features["body_lotion_matrix_model"]
            # Controlled numerical fixture, intentionally NOT this reference lotion.
            fixture = split_data()
            fixture["transport_mode"] = "bidirectional_air"
            simulation = request("synthetic_transport", "POST", "/v1/applications/body-lotion/simulate", json=fixture)
            assert simulation["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
            assert simulation["human_similarity_percent"] is None
            assert simulation["base_phase_assignments"][1]["source_kind"] == "simulated"
            report["checks"]["synthetic_transport_mass_error"] = simulation["diagnostics"]["mass_balance_max_abs_error_mg_cm2"]
            legacy = request("legacy_formula", "POST", "/v1/formulas", json={"brief": "clean woody scent"})
            assert legacy.get("recipe") or legacy.get("closest_candidate")
            report["checks"]["legacy_formula_status"] = legacy["status"]
            report["checks"]["legacy_formula_recipe_count"] = len(legacy.get("recipe") or [])
            report["checks"]["legacy_formula_closest_count"] = len(legacy.get("closest_candidate") or [])
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        # Exception type only: no headers, secrets, raw responses or subprocess output.
        report["error_type"] = type(error).__name__
        raise
    finally:
        if token_id:
            deleted = subprocess.run([sys.executable, "-m", "modal", "workspace", "proxy-tokens", "delete", token_id, "-y"],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
            report["temporary_token_revoked"] = deleted.returncode == 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["temporary_token_revoked"]:
        raise RuntimeError("temporary verification credential could not be revoked")


if __name__ == "__main__":
    main()
