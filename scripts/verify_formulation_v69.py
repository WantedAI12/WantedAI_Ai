"""Exercise the actual shared checkpoint and local API without old model loads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402 - numerical thread settings must be applied first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = str(args.profile.resolve())
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.formulation_views import shared_views
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import load_verified_catalog_bundle
    from fragrance_ai.recommender.emulsion_science import FIELDS
    from fragrance_ai.platform.lotion_reference import lotion_reference
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app

    core = configured_formulation_core()
    perfume = configured_perception("perfume")
    lotion = configured_perception("body_lotion")
    assert perfume.core is lotion.core is core
    assert all(v.core is core for v in shared_views(core))
    source = dict(np.load(args.data / "molecules.npz", allow_pickle=False))
    data_meta = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    assert (
        hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
        == core.manifest["data_sha256"]
    )
    for name, sha in data_meta["files"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == sha
    indexes = np.flatnonzero(source["split"] == 2)
    prediction = []
    timings = []
    for offset in range(0, len(indexes), 128):
        ids = indexes[offset : offset + 128]
        start = time.perf_counter()
        output = core.forward(source["x"][ids, None], np.ones((len(ids), 1)))
        prediction.append(
            np.maximum(
                0.0,
                np.expm1(output["quantitative"] * core.arrays["quantitative_scale"]),
            )
        )
        timings.append(time.perf_counter() - start)
    prediction = np.concatenate(prediction)
    target = source["high"][indexes]
    cosine = (prediction * target).sum(1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1), 1e-12
    )
    strata = {
        "all": np.ones(len(indexes), bool),
        "native_profile_present": source["x"][indexes, 1059] > 0,
        "odor_annotations_present": np.any(source["x"][indexes, 1061:-2] > 0, axis=1),
    }
    fidelity = {
        k: {
            "rows": int(v.sum()),
            "median_cosine": float(np.median(cosine[v])) if v.any() else None,
            "log_rmse": float(
                np.sqrt(np.mean((np.log1p(prediction[v]) - np.log1p(target[v])) ** 2))
            )
            if v.any()
            else None,
        }
        for k, v in strata.items()
    }
    report = {
        "scope": "actual_local_cpu_and_asgi_not_deployment_or_new_human_validation",
        "checkpoint_sha256": core.sha256,
        "parameter_count": core.manifest["parameter_count"],
        "shared_object_identity_verified": True,
        "old_core_forward_calls": 0,
        "teacher_fidelity_strata": fidelity,
        "cpu_batch128_median_seconds": float(np.median(timings)),
        "api": [],
        "human_similarity_percent": None,
    }
    profile = local_profile()
    bundle = load_verified_catalog_bundle(*profile["catalog"])
    report["catalog_rows"] = len(bundle["catalog"].ingredients)
    report["active_materials"] = sum(
        i.formulation_ready and not i.blocked for i in bundle["catalog"].ingredients
    )

    def fail(*a, **k):
        raise AssertionError("legacy model construction called by shared recipe API")

    with (
        patch("fragrance_ai.recommender.local_runtime._atlas", fail),
        patch("fragrance_ai.recommender.perception_runtime._load", fail),
        patch("fragrance_ai.recommender.fine_odor_model._load", fail),
        patch("fragrance_ai.recommender.unified_transport._load", fail),
        patch(
            "fragrance_ai.recommender.stock_mixture.StockMixturePredictor.__init__",
            fail,
        ),
    ):
        app = create_app(enable_language=False)
        with TestClient(app) as client:
            cap = client.get("/v1/ai/capabilities")
            assert cap.status_code == 200, cap.text
            assert (
                cap.json()["shared_formulation_model"]["checkpoint_sha256"]
                == core.sha256
            )
            assert cap.json()["stock_mixture_model"]["loaded"] is False
            (args.output / "capabilities.json").write_text(
                json.dumps(cap.json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            requests = [
                (
                    "/v1/formulation-workflows/plan",
                    {
                        "product_type": "perfume",
                        "completed_step_ids": ["brief", "material_review", "weigh"],
                    },
                ),
                (
                    "/v1/formulation-workflows/plan",
                    {
                        "product_type": "body_lotion",
                        "application_context": lotion_reference()[
                            "application_context"
                        ],
                        "process": {
                            "batch_mass_g": 400,
                            "mixer_kind": "high_shear",
                            "measured_ph": 5.5,
                        },
                        "completed_step_ids": ["brief", "material_review", "weigh"],
                    },
                ),
            ]
            raw = np.load(args.data / "emulsion.npz")["raw"][0]
            requests.append(
                (
                    "/v1/formulation-workflows/emulsion-prediction",
                    {
                        "reference_domain": "rodgers_2025_stirred_tank_24h_silicone_emulsion",
                        "conditions": [dict(zip(FIELDS, raw.tolist()))],
                    },
                )
            )
            previous = json.loads(
                (
                    ROOT / ".benchmarks/main_backbone_v66/api-01/api-report.json"
                ).read_text(encoding="utf-8")
            )
            requests.extend((v["path"], v["request"]) for v in previous["results"])
            for index, (path, body) in enumerate(requests):
                start = time.perf_counter()
                response = client.post(path, json=body)
                elapsed = time.perf_counter() - start
                assert response.status_code == 200, response.text
                value = response.json()
                (args.output / f"response-{index}.json").write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if path.endswith("/plan"):
                    assert value["learned_process"]["checkpoint_sha256"] == core.sha256
                    assert value["learned_process"]["status"] == "source_consistent"
                    assert not value["manufacturing_approved"]
                if path.endswith("/emulsion-prediction"):
                    assert value["checkpoint_sha256"] == core.sha256
                    assert not value["results"][0]["lotion_product_validation"]
                if path == "/v1/formulas" and value.get("perception_guidance"):
                    assert (
                        value["perception_guidance"]["component_model_sha256"]
                        == core.sha256
                    )
                report["api"].append(
                    {
                        "path": path,
                        "request": body,
                        "http_status": 200,
                        "seconds": elapsed,
                        "status": value.get("status"),
                        "score": value.get(
                            "calculated_profile_similarity", value.get("score")
                        ),
                    }
                )
                print(json.dumps(report["api"][-1], ensure_ascii=True), flush=True)
            assert app.state.local_stock_mixture_predictor._model is None
    (args.output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
