"""Measure phase requests and local API cache behavior on an isolated wheel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import ipaddress
import json
from pathlib import Path
import statistics
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_HASH = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    installed = output / "installed"
    with zipfile.ZipFile(args.wheel) as archive:
        if any(
            not (installed / name).resolve().is_relative_to(installed)
            for name in archive.namelist()
        ):
            raise ValueError("wheel contains an escaping path")
        archive.extractall(installed)
    sys.path.insert(0, str(installed))
    sys.path.append(str(ROOT))
    outbound = []

    def guard(event, arguments):
        if event not in {"socket.connect", "socket.getaddrinfo"}:
            return
        address = arguments[1] if event == "socket.connect" else arguments[0]
        host = address[0] if isinstance(address, tuple) else address
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            outbound.append(event)
            raise RuntimeError("external network disabled in local benchmark")

    sys.addaudithook(guard)
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.registry_activation import (
        activate_registry_conditionals,
    )

    report = {
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "scope": "local software/phase-intent benchmark; no Modal deployment or human sensory measurement",
    }
    started = time.perf_counter()
    catalog, _ = activate_registry_conditionals(
        IngredientCatalog.load_builtin(),
        ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db",
        expected_sha256=REGISTRY_HASH,
    )
    report["raw_catalog_activation_seconds"] = time.perf_counter() - started
    report["phase_cases"] = []
    for text in [
        "opening citrus, drydown woody musk",
        "opening woody musk, drydown citrus",
        "오프닝은 시트러스, 잔향은 우디 머스크",
    ]:
        started = time.perf_counter()
        with NaturalLanguagePerfumeryAI(catalog=catalog) as ai:
            result = ai.create_recipe(
                text,
                RecipeConstraints(
                    max_ingredient_price_per_kg=180,
                    max_formula_cost_per_kg=160,
                    simulation_draws=64,
                    physics_search_population=7,
                    target_similarity=50,
                ),
                as_of=date(2026, 9, 5),
            )
        report["phase_cases"].append(
            {
                "brief": text,
                "seconds": time.perf_counter() - started,
                "status": result.status,
                "formula_id": result.formula_id,
                "phase_targets": getattr(result.brief, "phase_target_profiles", {}),
                "swaps_evaluated": getattr(result, "ingredient_swaps_evaluated", 0),
                "ingredient_sets_evaluated": getattr(
                    result, "ingredient_sets_evaluated", 1
                ),
                "physics_objective": result.physics_search_objective,
            }
        )
    if args.api:
        from deploy.modal_app import create_web_app, WHEEL_SHA256
        from fastapi.testclient import TestClient

        if WHEEL_SHA256 != report["wheel_sha256"]:
            raise ValueError("API deployment config does not match measured wheel")
        started = time.perf_counter()
        app = create_web_app(
            str(ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db")
        )
        api = {"startup_seconds": time.perf_counter() - started}
        with TestClient(app) as client:
            for label, brief, expanded in [
                ("standard", "clean fresh citrus woody musk", False),
                ("expanded", "smoky leathery woody dry fragrance", True),
            ]:
                durations = []
                for run in range(3):
                    body = {
                        "brief": brief + " " * run,
                        "enable_registry_trace_candidates": expanded,
                        "experimental_disable_safety": expanded,
                        "max_risk_tier": 2 if expanded else 1,
                        "target_similarity": 50 if expanded else 90,
                    }
                    started = time.perf_counter()
                    response = client.post("/v1/formulas", json=body)
                    durations.append(time.perf_counter() - started)
                    if (
                        response.status_code != 200
                        or not response.json()["recipe"]
                        or response.headers["X-Perfumery-Cache"] != "miss"
                    ):
                        raise RuntimeError(f"uncached {label} request failed")
                hits = []
                for _ in range(3):
                    started = time.perf_counter()
                    hit = client.post("/v1/formulas", json=body)
                    hits.append(time.perf_counter() - started)
                    if (
                        hit.status_code != 200
                        or hit.headers["X-Perfumery-Cache"] != "hit"
                        or hit.json() != response.json()
                    ):
                        raise RuntimeError("cache response changed")
                api[label] = {
                    "uncached_seconds": durations,
                    "uncached_median_seconds": statistics.median(durations),
                    "cached_seconds": hits,
                    "cached_median_seconds": statistics.median(hits),
                }
            before = app.state.formula_cache.stats()["computations"]
            body = {"brief": "fresh green aquatic rain in a forest"}
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(
                    pool.map(lambda _: client.post("/v1/formulas", json=body), range(4))
                )
            api["concurrent_statuses"] = [
                response.status_code for response in responses
            ]
            api["concurrent_cache_states"] = [
                response.headers.get("X-Perfumery-Cache") for response in responses
            ]
            api["concurrent_computations"] = (
                app.state.formula_cache.stats()["computations"] - before
            )
            if (
                api["concurrent_computations"] != 1
                or api["concurrent_statuses"] != [200] * 4
            ):
                raise RuntimeError("same-key concurrency did not coalesce")
            api["cache_stats"] = app.state.formula_cache.stats()
            api["phase_default_90"] = []
            for text in [
                "opening clean fresh citrus, drydown woody musk",
                "opening woody musk, drydown clean fresh citrus",
                "오프닝은 깨끗하고 시원한 시트러스, 잔향은 우디 머스크",
            ]:
                response = client.post("/v1/formulas", json={"brief": text})
                payload = response.json()
                if response.status_code != 200 or not payload.get("recipe") or payload["similarity_score"] < 90:
                    raise RuntimeError("default-threshold phase request failed")
                api["phase_default_90"].append({
                    "brief": text, "status": payload["status"], "formula_id": payload["formula_id"],
                    "lines": len(payload["recipe"]), "semantic_score": payload["similarity_score"],
                })
        report["api"] = api
    report["outbound_network_attempts"] = outbound
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "wheel_sha256": report["wheel_sha256"],
                "phase_formula_ids": [
                    case["formula_id"] for case in report["phase_cases"]
                ],
                "api": report.get("api"),
                "outbound_network_attempts": outbound,
                "report": str(output / "report.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
