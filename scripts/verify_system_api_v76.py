"""Exercise real local V76 readouts and generation, without deploying or approval."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def main():
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observed-data", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        PERFUMERY_AI_ENV="research", PERFUMERY_AI_LOCAL_PROFILE=str(a.profile.resolve())
    )
    from deploy.system_runtime_v76 import create_local_app
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.formulation_process import CHECKS
    from fragrance_ai.platform.lotion_reference import lotion_reference
    from fastapi.testclient import TestClient

    core = configured_formulation_core()
    checks = []

    def record(name, response, seconds):
        value = response.json()
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8")
        (a.output / (name + ".json.gz")).write_bytes(gzip.compress(raw, mtime=0))
        checks.append(
            {
                "name": name,
                "http": response.status_code,
                "seconds": seconds,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        (a.output / "progress.json").write_text(
            json.dumps(checks, indent=2), encoding="utf8"
        )
        return value

    with TestClient(create_local_app(), raise_server_exceptions=False) as client:
        start = time.perf_counter()
        r = client.get("/v1/ai/capabilities")
        cap = record("capabilities", r, time.perf_counter() - start)
        assert r.status_code == 200
        assert cap["shared_formulation_model"]["checkpoint_sha256"] == core.sha256
        assert cap["features"]["learned_pair_annotations"]
        assert (
            cap["physical_property_evidence"]["index_sha256"]
            == core.manifest["scientific_evidence_index_sha256"]
        )
        payload = {
            "product_type": "body_lotion",
            "process": {
                "method": "cold",
                "batch_mass_g": 400,
                "mixer_kind": "high_shear",
                "measured_ph": 5.5,
            },
            "completed_step_ids": [
                "brief",
                "material_review",
                "weigh",
                "phase_b",
                "phase_a",
            ],
        }
        start = time.perf_counter()
        r = client.post("/v1/formulation-workflows/plan", json=payload)
        missing = record("process_without_matrix", r, time.perf_counter() - start)
        assert r.status_code == 200 and missing["selected_method"] is None
        assert missing["learned_process"]["status"] == "unsupported_reference_process"
        payload["application_context"] = lotion_reference()["application_context"]
        payload["application_context"]["fragrance_concentration_percent"] = 0.5
        start = time.perf_counter()
        r = client.post("/v1/formulation-workflows/plan", json=payload)
        plan = record("process_graph", r, time.perf_counter() - start)
        assert (
            r.status_code == 200
            and plan["learned_process"]["status"] == "source_consistent"
        )
        assert plan["learned_process"]["source_checks"][CHECKS[0]] is False
        with np.load(a.observed_data / "liquid.npz", allow_pickle=False) as data:
            row = data["x"][np.flatnonzero(data["split"] == 2)[0]]
        payload = {
            "reference_protocol": core.manifest["aqueous"]["reference_protocol"],
            "ingredient_percent": dict(
                zip(core.manifest["aqueous"]["ingredient_names"], row.tolist())
            ),
        }
        start = time.perf_counter()
        r = client.post("/v1/formulation-workflows/observed-outcomes", json=payload)
        liquid = record("observed_outcomes", r, time.perf_counter() - start)
        assert r.status_code == 200 and 0 <= liquid["stability_probability"] <= 1
        assert (
            len(liquid["estimated_viscosity_mpa_s"]) == 16
            and np.isfinite(liquid["estimated_viscosity_mpa_s"]).all()
        )
        assert liquid["manufacturing_approved"] is False
        start = time.perf_counter()
        r = client.post(
            "/v1/formulations/pair-annotations",
            json={"first_smiles": "CCO", "second_smiles": "CC(=O)O"},
        )
        pair = record("pair_annotations", r, time.perf_counter() - start)
        assert r.status_code == 200 and len(pair["labels"]) == 107
        assert (
            "" not in pair["labels"]
            and "No odor group found for these" not in pair["labels"]
        )
        assert pair["composition_ratio_known"] is False
        r = client.post(
            "/v1/formulation-workflows/observed-outcomes",
            json={"reference_protocol": "x", "ingredient_percent": None},
        )
        record("invalid_percent_type", r, 0.0)
        assert r.status_code == 422
        for name, path, payload in [
            (
                "lotion_regression",
                "/v1/applications/body-lotion/design",
                {
                    "brief": "플로럴, 화이트플로럴 향",
                    "max_risk_tier": 2,
                    "target_similarity": 95.0,
                    "registry_pool": "conditional_research",
                },
            ),
            (
                "perfume_generation",
                "/v1/formulas",
                {
                    "brief": "깨끗하고 시원한 시트러스 우디, 코코넛은 제외",
                    "max_risk_tier": 2,
                    "target_similarity": 95.0,
                    "enable_registry_trace_candidates": True,
                    "require_full_profile_match": True,
                },
            ),
        ]:
            time.sleep(3.2)
            start = time.perf_counter()
            r = client.post(path, json=payload)
            value = record(name, r, time.perf_counter() - start)
            assert r.status_code == 200, (name, r.status_code, value)
            score = value.get("calculated_profile_similarity", value.get("score"))
            passed = value.get(
                "full_profile_target_met", value.get("profile_target_met", False)
            )
            assert not passed or score is not None and score >= 95 - 1e-7
            checks[-1].update(
                score=score, target_met=passed, status=value.get("status")
            )
    report = {
        "core_sha256": core.sha256,
        "profile_sha256": hashlib.sha256(a.profile.read_bytes()).hexdigest(),
        "checks": checks,
        "complete": True,
        "scope": "real_local_contract_and_regression_examples_not_full400",
        "deployed": False,
    }
    (a.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
