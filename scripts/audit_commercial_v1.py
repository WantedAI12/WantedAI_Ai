"""Build a factual v1.4 software-release readiness report.

Passing this audit means that the packaged R&D software, its fail-closed
boundaries, and its supply-chain evidence were exercised.  It never converts
proxy scores into measured human olfactory accuracy and never approves a
formula for manufacture or sale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai import __version__  # noqa: E402
from fragrance_ai.recommender import (  # noqa: E402
    NaturalLanguagePerfumeryAI,
    RecipeConstraints,
)
from fragrance_ai.rules.ifra_rules import ProductCategory, check_compliance  # noqa: E402


RELEASE_MINIMUM_TESTS = 220


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one dist-info METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    for line in metadata.splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise ValueError("wheel metadata has no Version field")


def wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def junit_summary(paths: list[Path]) -> dict[str, int]:
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for path in paths:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            for name in totals:
                totals[name] += int(suite.attrib.get(name, 0))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--full-inference-report", type=Path, required=True)
    parser.add_argument("--service-load-report", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--release-policy", type=Path, required=True)
    parser.add_argument(
        "--junit-xml",
        type=Path,
        action="append",
        required=True,
        help="One or more non-overlapping pytest JUnit reports",
    )
    parser.add_argument("--minimum-tests", type=int, default=RELEASE_MINIMUM_TESTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "commercial_v1_4_readiness.json",
    )
    args = parser.parse_args()

    wheel = args.wheel.resolve(strict=True)
    full_inference_path = args.full_inference_report.resolve(strict=True)
    service_load_path = args.service_load_report.resolve(strict=True)
    sbom_path = args.sbom.resolve(strict=True)
    policy_path = args.release_policy.resolve(strict=True)
    junit_paths = [path.resolve(strict=True) for path in args.junit_xml]
    if args.minimum_tests < RELEASE_MINIMUM_TESTS:
        raise SystemExit(
            f"--minimum-tests cannot be below the release floor of {RELEASE_MINIMUM_TESTS}"
        )
    if len(junit_paths) != len(set(junit_paths)) or len(
        {sha256(path) for path in junit_paths}
    ) != len(junit_paths):
        raise SystemExit("JUnit reports must be unique, non-overlapping artifacts")

    full_inference = read_json(full_inference_path)
    service_load = read_json(service_load_path)
    sbom = read_json(sbom_path)
    policy = read_json(policy_path)
    tests = junit_summary(junit_paths)
    packaged_version = wheel_version(wheel)
    packaged_members = wheel_members(wheel)

    with NaturalLanguagePerfumeryAI() as ai:
        prototype = ai.create_recipe(
            "깨끗하고 시원한 시트러스 우디, 달지 않게",
            RecipeConstraints(require_simulation_pass=False),
            as_of=date.today(),
        )
        fail_closed: dict[str, dict] = {}
        for level in ("qualified", "commercial"):
            result = ai.create_recipe(
                "깨끗하고 시원한 시트러스 우디, 달지 않게",
                RecipeConstraints(
                    validation_level=level,
                    require_simulation_pass=False,
                ),
                as_of=date.today(),
            )
            fail_closed[level] = {
                "status": result.status,
                "recipe_emitted": bool(result.recipe),
                "manufacturing_ready": result.safety.manufacturing_ready,
                "actual_olfactory_similarity_score": (
                    result.actual_olfactory_similarity_score
                ),
                "actual_olfactory_lower_bound_95": (
                    result.actual_olfactory_lower_bound_95
                ),
                "passed": (
                    result.status == "no_safe_match"
                    and not result.recipe
                    and not result.safety.manufacturing_ready
                    and result.actual_olfactory_similarity_score is None
                    and result.actual_olfactory_lower_bound_95 is None
                ),
            }

    unknown_ifra = check_compliance(
        {"ingredients": [{"name": "Unlisted Material", "concentration": 1.0}]},
        ProductCategory.EAU_DE_PARFUM,
        product_concentration=15.0,
    )
    metadata_properties = {
        item.get("name"): item.get("value")
        for item in sbom.get("metadata", {}).get("properties", [])
        if isinstance(item, dict)
    }
    checks = {
        "package_version_1_4_0": (
            __version__ == "1.4.0" and packaged_version == "1.4.0"
        ),
        "pytest_complete": (
            tests["tests"] >= args.minimum_tests
            and tests["failures"] == 0
            and tests["errors"] == 0
            and tests["skipped"] == 0
        ),
        "cyclonedx_1_6_sbom": (
            sbom.get("bomFormat") == "CycloneDX"
            and sbom.get("specVersion") == "1.6"
            and metadata_properties.get("perfumery-ai:reproducible-build-claim")
            == "false"
        ),
        "release_policy_passed": (
            bool(policy.get("passed")) and policy.get("wheel_sha256") == sha256(wheel)
        ),
        "sbom_wheel_hash_bound": (
            metadata_properties.get("perfumery-ai:wheel-sha256") == sha256(wheel)
        ),
        "isolated_full_inference_passed": (
            bool(full_inference.get("passed"))
            and full_inference.get("expected_package_version") == packaged_version
            and full_inference.get("wheel_sha256") == sha256(wheel)
            and full_inference.get("details", {}).get("unsafe_serialized_assets") == []
        ),
        "service_load_baseline_passed": (
            bool(service_load.get("passed"))
            and service_load.get("package_version") == "1.4.0"
            and service_load.get("wheel_sha256") == sha256(wheel)
            and service_load.get("package_loaded_from_wheel") is True
            and service_load.get("wheel_binding_passed") is True
            and service_load.get("wheel_source_mismatches") == []
            and service_load.get("real_inference", {}).get("errors") == 0
            and service_load.get("service_control_plane", {}).get("errors") == 0
        ),
        "platform_runtime_and_ui_packaged": {
            "fragrance_ai/platform/postgres.py",
            "fragrance_ai/platform/worker.py",
            "fragrance_ai/ui/index.html",
            "fragrance_ai/ui/app.css",
            "fragrance_ai/ui/app.js",
        }.issubset(packaged_members),
        "portable_runtime_only_no_pickle_model_assets": (
            {
                "fragrance_ai/data/concentration_response_runtime.json",
                "fragrance_ai/data/physsim_r2_runtime_weights.npz",
                "fragrance_ai/data/physsim_r2_runtime_manifest.json",
            }.issubset(packaged_members)
            and not any(
                Path(member).suffix.lower()
                in {".joblib", ".pickle", ".pkl", ".pt", ".pth"}
                for member in packaged_members
            )
        ),
        "prototype_actual_olfactory_stays_null": (
            prototype.actual_olfactory_similarity_score is None
            and prototype.actual_olfactory_lower_bound_95 is None
            and prototype.olfactory_validation_status == "abstained_no_evidenced_target"
            and not prototype.physsim_comparison_authorized
        ),
        "legacy_r2_primary_weight_zero": (
            prototype.physsim_learned_r2_applied_weight == 0.0
        ),
        "unsigned_concentration_model_weight_zero": (
            prototype.concentration_response_applied_weight == 0.0
        ),
        "unknown_ifra_coverage_fails_closed": (
            not bool(unknown_ifra["overall_compliant"])
            and not bool(unknown_ifra["ifra_rule_pack_complete"])
            and not bool(unknown_ifra["commercial_release_eligible"])
        ),
        "qualified_without_external_evidence_fails_closed": fail_closed["qualified"][
            "passed"
        ],
        "commercial_without_external_evidence_fails_closed": fail_closed["commercial"][
            "passed"
        ],
    }
    software_release_passed = all(checks.values())

    report = {
        "schema_version": "1.4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": "perfumery-ai-core",
        "package_version": __version__,
        "status": (
            "r_and_d_software_release_gates_passed"
            if software_release_passed
            else "r_and_d_software_release_gate_failed"
        ),
        "r_and_d_software_release_passed": software_release_passed,
        "commercial_formula_release_approved": False,
        "manufacturing_release_approved": False,
        "human_olfactory_90_percent_certified": False,
        "actual_olfactory_similarity_score": None,
        "actual_olfactory_lower_bound_95": None,
        "checks": checks,
        "pytest": {
            **tests,
            "junit_reports": [str(path) for path in junit_paths],
        },
        "runtime": {
            "prototype_status": prototype.status,
            "proxy_score": prototype.simulated_similarity_score,
            "proxy_p05": prototype.simulation_p05,
            "proxy_score_claim_boundary": (
                "computational structural/headspace/temporal proxy; not a "
                "percentage of human olfactory similarity"
            ),
            "legacy_r2_status": prototype.physsim_learned_r2_status,
            "legacy_r2_applied_weight": (prototype.physsim_learned_r2_applied_weight),
            "concentration_response_status": (prototype.concentration_response_status),
            "concentration_response_applied_weight": (
                prototype.concentration_response_applied_weight
            ),
            "fail_closed_external_evidence": fail_closed,
        },
        "wheel": {
            "path": str(wheel),
            "version": packaged_version,
            "sha256": sha256(wheel),
            "bytes": wheel.stat().st_size,
        },
        "supply_chain_evidence": {
            "sbom": str(sbom_path),
            "release_policy": str(policy_path),
            "isolated_full_inference": str(full_inference_path),
            "service_load_baseline": str(service_load_path),
            "dependency_lock": str((ROOT / "requirements-ci.lock").resolve()),
            "dependency_lock_sha256": sha256(ROOT / "requirements-ci.lock"),
            "runtime_dependency_lock": str(
                (ROOT / "requirements-runtime.lock").resolve()
            ),
            "runtime_dependency_lock_sha256": sha256(
                ROOT / "requirements-runtime.lock"
            ),
        },
        "claim_boundary": (
            "This report validates software controls and fail-closed behavior "
            "only. It does not measure, infer, certify, or claim 90% real-world "
            "human olfactory similarity, regulatory compliance, or readiness "
            "to manufacture a fragrance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not software_release_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
