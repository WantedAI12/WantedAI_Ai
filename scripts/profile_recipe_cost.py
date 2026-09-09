"""Measure an installed deployment wheel locally without outbound requests.

Runs the same core constraints and per-request service lifetime as Modal. Wall
time is measured without SQL/profile instrumentation; a separate invocation
records SQL counts and CPU call stacks. No deployment or credential is needed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import cProfile
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import pstats
import re
import sqlite3
import statistics
import sys
import time
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SHA256 = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"
SQL_LITERALS = re.compile(r"'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error("--runs must be in [1, 10]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    installed = output / "installed"
    with zipfile.ZipFile(args.wheel) as archive:
        for member in archive.namelist():
            if not (installed / member).resolve().is_relative_to(installed):
                raise ValueError("wheel contains an escaping path")
        archive.extractall(installed)
    sys.path.insert(0, str(installed))

    network_attempts: Counter[str] = Counter()

    def guard(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            network_attempts[event] += 1
            raise RuntimeError("outbound network is disabled during profiling")

    sys.addaudithook(guard)
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.registry_activation import activate_registry_conditionals
    import numpy as np

    started = time.perf_counter()
    catalog, activation = activate_registry_conditionals(
        IngredientCatalog.load_builtin(),
        ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db",
        expected_sha256=REGISTRY_SHA256,
    )
    activation_seconds = time.perf_counter() - started
    cases = [
        ("standard", "clean fresh citrus woody musk", False),
        ("expanded", "smoky leathery woody dry fragrance", True),
    ]
    evidence = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "registry_sha256": activation.registry_sha256,
        "registry_activation_seconds": activation_seconds,
        "scope": "local deployment-wheel core; excludes Modal network and cold container startup",
        "cases": {},
    }

    for case_name, brief, expanded in cases:
        def invoke() -> tuple[dict, dict]:
            constraints = RecipeConstraints(
                max_risk_tier=2 if expanded else 1,
                max_ingredient_price_per_kg=180.0,
                max_formula_cost_per_kg=160.0,
                min_availability=0.75,
                target_similarity=50.0 if expanded else 90.0,
                product_concentration_percent=15.0,
                max_ingredients=12,
                allow_rare=False,
                enable_registry_trace_candidates=expanded,
                experimental_disable_safety=expanded,
                target_region="EU",
                product_category="eau_de_parfum",
                simulation_draws=64,
                physics_search_population=7,
                minimum_realism_score=50.0,
            )
            start = time.perf_counter()
            with NaturalLanguagePerfumeryAI(catalog=catalog) as ai:
                initialized = time.perf_counter()
                result = ai.create_recipe(brief, constraints, as_of=date(2026, 9, 5))
                calculated = time.perf_counter()
                payload = result.to_dict()
            stop = time.perf_counter()
            return payload, {
                "init_seconds": initialized - start,
                "calculation_seconds": calculated - initialized,
                "total_seconds": stop - start,
            }

        timings = []
        hashes = []
        for run in range(args.runs):
            payload, duration = invoke()
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
            timings.append(duration)
            if run == 0:
                (output / f"{case_name}_response.json").write_text(
                    canonical + "\n", encoding="utf-8"
                )
            print(json.dumps({"case": case_name, "run": run, **duration}), flush=True)

        queries: Counter[tuple[str, str]] = Counter()
        opens: Counter[str] = Counter()
        connections = []
        original_connect = sqlite3.connect

        def connect(database, *positional, **keywords):
            connection = original_connect(database, *positional, **keywords)
            database_name = str(database).split("?")[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            opens[database_name] += 1

            def trace(statement: str) -> None:
                signature = " ".join(SQL_LITERALS.sub("?", statement).split())
                queries[(database_name, signature)] += 1

            connection.set_trace_callback(trace)
            connections.append(connection)
            return connection

        profiler = cProfile.Profile()
        with patch.object(sqlite3, "connect", connect):
            profiler.enable()
            measured_payload, _ = invoke()
            profiler.disable()
        measured_hash = hashlib.sha256(
            json.dumps(measured_payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        profiler.dump_stats(str(output / f"{case_name}.prof"))
        stats = pstats.Stats(profiler)
        functions = [
            {
                "function": f"{Path(file).name}:{line}:{function}",
                "calls": calls,
                "own_seconds": own,
                "cumulative_seconds": cumulative,
            }
            for (file, line, function), (_, calls, own, cumulative, _) in stats.stats.items()
            if "fragrance_ai" in file
        ]
        row = {
            "timings": timings,
            "median_seconds": statistics.median(item["total_seconds"] for item in timings),
            "response_sha256": hashes[0],
            "responses_identical": len(set(hashes + [measured_hash])) == 1,
            "status": measured_payload["status"],
            "recipe_lines": len(measured_payload["recipe"]),
            "variants_evaluated": measured_payload["candidate_variants_evaluated"],
            "connections": dict(opens),
            "sql_total": sum(queries.values()),
            "sql_top": [
                {"database": database, "sql": sql, "count": count}
                for (database, sql), count in queries.most_common(12)
            ],
            "cpu_top": sorted(functions, key=lambda item: item["cumulative_seconds"], reverse=True)[:20],
        }
        evidence["cases"][case_name] = row
        print(json.dumps({"case": case_name, "sql_total": row["sql_total"], "median_seconds": row["median_seconds"], "responses_identical": row["responses_identical"]}), flush=True)
    evidence["outbound_network_attempts"] = dict(network_attempts)
    (output / "report.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
