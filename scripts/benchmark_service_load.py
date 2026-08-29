"""Measure real inference and authenticated ASGI service load.

The benchmark deliberately reports two different paths:

* ``real_inference`` exercises the shipped natural-language recipe engine.
* ``service_control_plane`` uses a constant-time deterministic engine so the
  cost of authentication, rate limiting, middleware, serialization, and the
  authenticated audit chain can be measured independently.

Results are machine-specific engineering evidence, not an external product
comparison or a production SLA.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_UNDER_TEST = Path(
    os.environ.get("PERFUMERY_BENCHMARK_PACKAGE_ROOT", str(ROOT))
).resolve()
sys.path.insert(0, str(PACKAGE_UNDER_TEST))

from fragrance_ai import __version__  # noqa: E402
from fragrance_ai.api import (  # noqa: E402
    SlidingWindowRateLimiter,
    TokenAuthorizer,
    create_app,
)
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog  # noqa: E402
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmarks" / "software_load_v1_4.json"
AUTH_HEADERS = {
    "Authorization": "Bearer benchmark-token",
    "X-Tenant-ID": "benchmark-tenant",
}
BRIEFS = (
    "clean fresh citrus woods, restrained sweetness",
    "transparent green tea, bergamot peel and dry cedar",
    "soft iris musk with airy mineral freshness",
    "bright grapefruit, aromatic herbs and vetiver",
    "creamy sandalwood with clean linen and subtle florals",
    "cool marine air, citrus zest and pale woods",
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(latencies_seconds: list[float]) -> dict[str, float]:
    if not latencies_seconds:
        return {}
    milliseconds = [value * 1000.0 for value in latencies_seconds]
    return {
        "minimum_ms": round(min(milliseconds), 3),
        "mean_ms": round(statistics.fmean(milliseconds), 3),
        "p50_ms": round(percentile(milliseconds, 0.50), 3),
        "p95_ms": round(percentile(milliseconds, 0.95), 3),
        "p99_ms": round(percentile(milliseconds, 0.99), 3),
        "maximum_ms": round(max(milliseconds), 3),
    }


@dataclass(frozen=True)
class _ControlResult:
    marker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "prototype_ready",
            "formula_id": "sha256:" + self.marker * 64,
            "olfactory_validation_status": "abstained_no_evidenced_target",
            "actual_olfactory_similarity_score": None,
            "actual_olfactory_lower_bound_95": None,
        }


class _ControlAI:
    def create_recipe(self, brief, constraints):
        marker = "a" if len(brief) % 2 == 0 else "b"
        return _ControlResult(marker)


async def _exercise_app(
    *,
    app,
    requests: int,
    concurrency: int,
    simulation_draws: int,
) -> dict[str, Any]:
    import httpx

    transport = httpx.ASGITransport(app=app)
    limiter = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: dict[str, int] = {}

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://benchmark.local",
        timeout=180.0,
    ) as client:
        await client.get("/health/live")

        async def invoke(index: int) -> None:
            async with limiter:
                started = time.perf_counter()
                response = await client.post(
                    "/v1/recipes",
                    headers=AUTH_HEADERS,
                    json={
                        "brief": BRIEFS[index % len(BRIEFS)],
                        "constraints": {
                            "simulation_draws": simulation_draws,
                            "require_simulation_pass": False,
                        },
                    },
                )
                latencies.append(time.perf_counter() - started)
                key = str(response.status_code)
                statuses[key] = statuses.get(key, 0) + 1

        started = time.perf_counter()
        await asyncio.gather(*(invoke(index) for index in range(requests)))
        wall_seconds = time.perf_counter() - started

    successes = statuses.get("200", 0)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "successes": successes,
        "errors": requests - successes,
        "error_rate": round((requests - successes) / requests, 6),
        "status_counts": statuses,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(requests / wall_seconds, 4),
        "latency": latency_summary(latencies),
    }


def run_path(
    *,
    ai_factory,
    requests: int,
    concurrency: int,
    simulation_draws: int,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = AppendOnlyAuditLog(
        audit_path, signing_key=b"benchmark-audit-key-material-32b"
    )
    app = create_app(
        ai_factory=ai_factory,
        authorizer=TokenAuthorizer.from_plaintext(
            {
                "benchmark-token": (
                    "benchmark-operator",
                    "formulator",
                    "benchmark-tenant",
                )
            }
        ),
        audit_log=audit,
        max_concurrent_inference=concurrency,
        rate_limiter=SlidingWindowRateLimiter(
            requests=max(100, requests * 2),
            window_seconds=300,
        ),
        enable_ui=False,
    )
    try:
        app.state.warm_inference_runtime()
        result = asyncio.run(
            _exercise_app(
                app=app,
                requests=requests,
                concurrency=concurrency,
                simulation_draws=simulation_draws,
            )
        )
        verification = audit.verify()
    finally:
        app.state.close_inference_runtime()
        audit.close()
    return result, verification


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_source_mismatches(
    wheel: Path,
    root: Path | None = None,
) -> list[str]:
    """Prove that the in-process package under load is the supplied wheel."""

    source_root = (root or PACKAGE_UNDER_TEST).resolve()
    mismatches: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.startswith("fragrance_ai/"):
                continue
            source = source_root / Path(member.filename)
            if not source.is_file() or source.read_bytes() != archive.read(member):
                mismatches.append(member.filename)
    return sorted(mismatches)


def run_from_extracted_wheel(wheel: Path) -> int:
    """Re-exec this benchmark with imports rooted in the exact wheel bytes."""

    with tempfile.TemporaryDirectory(prefix="perfumery-load-wheel-") as temporary:
        package_root = Path(temporary).resolve()
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                target = (package_root / member.filename).resolve()
                if not target.is_relative_to(package_root):
                    raise ValueError("wheel contains a path outside its package root")
            archive.extractall(package_root)
        environment = dict(os.environ)
        environment["PERFUMERY_BENCHMARK_PACKAGE_ROOT"] = str(package_root)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment,
            text=True,
        )
        return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-requests", type=positive_int, default=6)
    parser.add_argument("--real-concurrency", type=positive_int, default=2)
    parser.add_argument("--control-requests", type=positive_int, default=200)
    parser.add_argument("--control-concurrency", type=positive_int, default=24)
    parser.add_argument("--simulation-draws", type=positive_int, default=64)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.real_concurrency > args.real_requests:
        parser.error("--real-concurrency cannot exceed --real-requests")
    if args.control_concurrency > args.control_requests:
        parser.error("--control-concurrency cannot exceed --control-requests")
    if args.simulation_draws < 64:
        parser.error("--simulation-draws must be at least 64")
    wheel = args.wheel.resolve() if args.wheel is not None else None
    if wheel is not None and not wheel.is_file():
        parser.error(f"--wheel does not exist: {wheel}")
    if wheel is not None and "PERFUMERY_BENCHMARK_PACKAGE_ROOT" not in os.environ:
        raise SystemExit(run_from_extracted_wheel(wheel))
    wheel_mismatches = wheel_source_mismatches(wheel) if wheel is not None else []

    with tempfile.TemporaryDirectory(prefix="perfumery-load-") as temporary:
        temp = Path(temporary)
        control, control_audit = run_path(
            ai_factory=_ControlAI,
            requests=args.control_requests,
            concurrency=args.control_concurrency,
            simulation_draws=args.simulation_draws,
            audit_path=temp / "control-audit.db",
        )
        real = None
        real_audit = None
        if not args.skip_real:
            real, real_audit = run_path(
                ai_factory=NaturalLanguagePerfumeryAI,
                requests=args.real_requests,
                concurrency=args.real_concurrency,
                simulation_draws=args.simulation_draws,
                audit_path=temp / "real-audit.db",
            )

    audit_checks = [control_audit, *([real_audit] if real_audit else [])]
    paths = [control, *([real] if real else [])]
    loaded_from_wheel = wheel is not None and PACKAGE_UNDER_TEST != ROOT.resolve()
    wheel_binding_passed = wheel is None or (loaded_from_wheel and not wheel_mismatches)
    passed = (
        wheel_binding_passed
        and all(path["errors"] == 0 for path in paths)
        and all(check["passed"] for check in audit_checks)
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": "perfumery-ai-core",
        "package_version": __version__,
        "wheel": str(wheel) if wheel is not None else None,
        "wheel_sha256": sha256_file(wheel) if wheel is not None else None,
        "wheel_source_mismatches": wheel_mismatches,
        "package_root": str(PACKAGE_UNDER_TEST),
        "package_loaded_from_wheel": loaded_from_wheel,
        "wheel_binding_passed": wheel_binding_passed,
        "passed": passed,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "processor": platform.processor() or "not_reported",
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "transport": "in_process_asgi",
            "simulation_draws": args.simulation_draws,
            "engine_instances_preinitialized": {
                "real": args.real_concurrency if real is not None else 0,
                "control": args.control_concurrency,
            },
            "real_engine_warmup_requests": 0,
            "control_engine": "constant_time_deterministic_stub",
        },
        "real_inference": real,
        "service_control_plane": control,
        "audit_chain": {
            "control": control_audit,
            "real": real_audit,
        },
        "claim_boundary": (
            "Machine-specific in-process ASGI measurement. It excludes network, "
            "reverse proxy, external OIDC/JWKS, PostgreSQL, and multi-host effects; "
            "it is not a competitor comparison or production SLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
