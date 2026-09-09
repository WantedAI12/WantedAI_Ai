"""Build using the selected wheel; optionally preserve a hash-bound evidence catalog."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path)
    parser.add_argument("--source-runtime-sha256")
    parser.add_argument("--source-wheel-sha256")
    args = parser.parse_args()
    source_fields = (args.source_runtime, args.source_runtime_sha256, args.source_wheel_sha256)
    if any(source_fields) and not all(source_fields):
        parser.error("source runtime, runtime hash and original wheel hash must be supplied together")
    if hashlib.sha256(args.registry.read_bytes()).hexdigest() != args.registry_sha256:
        raise ValueError("registry artifact hash mismatch")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.wheel) as wheel:
        if any(
            not (workspace / name).resolve().is_relative_to(workspace)
            for name in wheel.namelist()
        ):
            raise ValueError("wheel contains escaping path")
        wheel.extractall(workspace)
    sys.path.insert(0, str(workspace))
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry
    from fragrance_ai.recommender.registry_activation import (
        activate_registry_conditionals,
        write_runtime_catalog,
        load_runtime_catalog,
    )

    started = time.perf_counter()
    if args.source_runtime:
        # Revalidate source evidence under the NEW wheel's loader. Do not rebuild
        # from a less complete DB and silently discard separately connected data.
        catalog, report, source_stats = load_runtime_catalog(
            args.source_runtime, expected_sha256=args.source_runtime_sha256,
            expected_wheel_sha256=args.source_wheel_sha256,
            expected_registry_sha256=args.registry_sha256)
    else:
        catalog, report = activate_registry_conditionals(
            IngredientCatalog.load_builtin(),
            args.registry,
            expected_sha256=args.registry_sha256,
        )
    with IndustrialIngredientRegistry(args.registry) as registry:
        stats = registry.stats()
    if args.source_runtime:
        # Older evidence builders intentionally stored only a subset of stats.
        # Validate every stored value and preserve that response shape verbatim.
        if any(key not in stats or stats[key] != value for key, value in source_stats.items()):
            raise ValueError("source runtime registry statistics changed")
        stats = source_stats
    wheel_hash = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    digest = write_runtime_catalog(
        args.output, catalog, report, stats, wheel_sha256=wheel_hash
    )
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    restored, restored_report, restored_stats = load_runtime_catalog(
        args.output,
        expected_sha256=digest,
        expected_wheel_sha256=wheel_hash,
        expected_registry_sha256=args.registry_sha256,
    )
    load_seconds = time.perf_counter() - started
    if (
        restored.ingredients != catalog.ingredients
        or restored.metadata != catalog.metadata
        or restored_report != report
        or restored_stats != stats
    ):
        raise ValueError("runtime catalog round trip changed content")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "wheel_sha256": wheel_hash,
                "ingredients": len(catalog.ingredients),
                "build_seconds": build_seconds,
                "load_seconds": load_seconds,
                "round_trip_identical": True,
                "source_runtime_sha256": args.source_runtime_sha256,
                "source_wheel_sha256": args.source_wheel_sha256,
                "active_odorant_candidates": report.active_odorant_candidates,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
