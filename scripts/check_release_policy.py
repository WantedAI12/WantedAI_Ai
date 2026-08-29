"""Fail-closed release-policy validation for a built wheel.

The check is intentionally stricter than a normal packaging lint: an asset
whose licence/provenance is unknown must not be distributed, every declared
asset must match its recorded digest, and the CI workflow must perform an
isolated full-inference probe.  A successful result is an automated supply
chain gate only; it is not a safety, regulatory, or sensory approval.
"""

from __future__ import annotations

import argparse
import json
import tomllib
import zipfile
from pathlib import Path
from typing import Any

try:  # `python scripts/check_release_policy.py` and `import scripts...`
    from .build_sbom import (
        DATA_PREFIX,
        ROOT,
        asset_distribution_allowed,
        asset_license_status,
        package_files,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script invocation path
    from build_sbom import (  # type: ignore[no-redef]
        DATA_PREFIX,
        ROOT,
        asset_distribution_allowed,
        asset_license_status,
        package_files,
        sha256_file,
    )


REQUIRED_WORKFLOW_TOKENS = (
    "pytest",
    "python -m ruff",
    "python -m compileall",
    "node --check",
    "python -m build",
    "docker build",
    "--no-isolation",
    "--require-hashes",
    "requirements-ci.lock",
    "requirements-runtime.lock",
    "check_release_policy.py",
    "full_inference_probe.py",
    "--lock",
    "benchmark_service_load.py",
    "audit_commercial_v1.py",
    "--service-load-report",
    "--junit-xml",
    "build_sbom.py",
    "attest-build-provenance@v2",
    "id-token: write",
    "attestations: write",
    "upload-artifact",
    "download-artifact",
    "timeout-minutes",
)


def dependency_lock_policy(root: Path) -> dict[str, Any]:
    path = root / "requirements-ci.lock"
    if not path.is_file():
        return {"passed": False, "path": str(path), "reason": "hash_lock_missing"}
    contents = path.read_text(encoding="utf-8")
    hash_count = contents.count("--hash=sha256:")
    required = (
        "numpy==",
        "pytest==",
        "ruff==",
        "cryptography==",
        "pyjwt[crypto]==",
        "fastapi==",
        "httpx==",
        "psycopg[binary,pool]==",
        "build==",
        "setuptools==",
        "wheel==",
    )
    missing = [item for item in required if item not in contents]
    passed = (
        hash_count >= len(required) and not missing and "not pinned" not in contents
    )
    return {
        "passed": passed,
        "path": str(path),
        "sha256": sha256_file(path),
        "hash_count": hash_count,
        "missing_required_pins": missing,
        "scope": "protected CI dependency resolution; not a cross-platform bit-for-bit claim",
    }


def runtime_dependency_lock_policy(root: Path) -> dict[str, Any]:
    path = root / "requirements-runtime.lock"
    if not path.is_file():
        return {
            "passed": False,
            "path": str(path),
            "reason": "runtime_hash_lock_missing",
        }
    contents = path.read_text(encoding="utf-8")
    required = (
        "numpy==",
        "cryptography==",
        "pyjwt[crypto]==",
        "fastapi==",
        "psycopg[binary,pool]==",
        "uvicorn==",
    )
    missing = [item for item in required if item not in contents]
    hash_count = contents.count("--hash=sha256:")
    return {
        "passed": hash_count >= len(required) and not missing,
        "path": str(path),
        "sha256": sha256_file(path),
        "hash_count": hash_count,
        "missing_required_pins": missing,
        "scope": "commercial container runtime dependency resolution",
    }


def read_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def project_license_consistent(root: Path) -> tuple[bool, str]:
    pyproject = read_pyproject(root)
    declared_value = pyproject.get("project", {}).get("license", "")
    declared = str(declared_value).strip()
    license_text = (root / "LICENSE").read_text(encoding="utf-8").lower()
    proprietary = (
        "proprietary" in license_text and "all rights reserved" in license_text
    )
    mit = "mit license" in license_text
    if proprietary:
        # PEP 621 permits either an SPDX-style proprietary reference or an
        # explicit pointer to the checked LICENSE file.  A generic or MIT
        # string is not an acceptable substitute for a restrictive license.
        points_to_license = (
            isinstance(declared_value, dict) and declared_value.get("file") == "LICENSE"
        )
        return (
            "proprietary" in declared.lower() or points_to_license,
            "proprietary_license_pair",
        )
    if mit:
        return (declared.lower() == "mit", "mit_license_pair")
    return False, "unrecognised_license_text"


def read_data_manifest(root: Path) -> dict[str, Any]:
    path = root / "fragrance_ai" / "data" / "data_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing data manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_integrity(root: Path) -> dict[str, Any]:
    try:
        manifest = read_data_manifest(root)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {
            "passed": False,
            "error": str(error),
            "failures": ["manifest_unreadable"],
        }
    failures: list[str] = []
    checked: list[str] = []
    for name, record in sorted(manifest.get("assets", {}).items()):
        path = root / "fragrance_ai" / "data" / name
        checked.append(name)
        if not path.is_file():
            failures.append(f"missing:{name}")
            continue
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            failures.append(f"missing_or_invalid_sha256:{name}")
        elif sha256_file(path) != expected_hash:
            failures.append(f"sha256_mismatch:{name}")
        if not isinstance(expected_bytes, int):
            failures.append(f"missing_or_invalid_bytes:{name}")
        elif path.stat().st_size != expected_bytes:
            failures.append(f"byte_count_mismatch:{name}")

    required_model_assets = (
        "r2_ingredient_components.npz",
        "physsim_r2_manifest.json",
        "physsim_r2_ensemble_manifest.json",
        "physsim_r2_runtime_manifest.json",
        "physsim_r2_runtime_weights.npz",
        "concentration_response_manifest.json",
        "concentration_response_runtime.json",
        "continuous_improvement_policy.json",
        "human_mixture_calibration.json",
    )
    for name in required_model_assets:
        if name not in manifest.get("assets", {}):
            failures.append(f"model_asset_not_manifested:{name}")
    # Verify each model's own sibling-manifest reference as well as the
    # project-level manifest.  This is integrity checking, not provenance
    # authentication: an independently trusted signature/release envelope is
    # still required before a regulated model release.
    data_root = root / "fragrance_ai" / "data"
    embedded_contracts = (
        ("physsim_r2_manifest.json", "checkpoint_file", "checkpoint_sha256"),
        ("concentration_response_manifest.json", "runtime_file", "runtime_sha256"),
        ("r2_ingredient_components_manifest.json", "artifact_file", "artifact_sha256"),
        ("physsim_r2_runtime_manifest.json", "artifact_file", "artifact_sha256"),
    )
    for manifest_name, file_key, hash_key in embedded_contracts:
        try:
            embedded = json.loads(
                (data_root / manifest_name).read_text(encoding="utf-8")
            )
            referenced_name = embedded[file_key]
            expected_hash = embedded[hash_key]
            referenced = data_root / referenced_name
            if not referenced.is_file() or sha256_file(referenced) != expected_hash:
                failures.append(f"embedded_model_manifest_mismatch:{manifest_name}")
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            failures.append(f"embedded_model_manifest_invalid:{manifest_name}")
    try:
        ensemble = json.loads(
            (data_root / "physsim_r2_ensemble_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        members = ensemble["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("members missing")
        for member in members:
            referenced = data_root / member["file"]
            if not referenced.is_file() or sha256_file(referenced) != member["sha256"]:
                failures.append(
                    "embedded_model_manifest_mismatch:physsim_r2_ensemble_manifest.json"
                )
                break
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        failures.append(
            "embedded_model_manifest_invalid:physsim_r2_ensemble_manifest.json"
        )
    return {"passed": not failures, "checked": checked, "failures": failures}


def package_asset_policy(root: Path, wheel: Path | None) -> dict[str, Any]:
    try:
        manifest = read_data_manifest(root)
        assets = manifest.get("assets", {})
        files = package_files(root, wheel)
    except (FileNotFoundError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        return {
            "passed": False,
            "error": str(error),
            "violations": ["package_inventory_failed"],
        }

    packaged_data_names = {
        Path(record["path"]).name
        for record in files
        if record["path"].startswith(DATA_PREFIX)
    }
    unknown = [
        name
        for name, record in sorted(assets.items())
        if not asset_distribution_allowed(record) and name in packaged_data_names
    ]
    unmanifested = sorted(
        name
        for name in packaged_data_names
        if name != "data_manifest.json" and name not in assets
    )
    unsafe_serialized_assets = sorted(
        record["path"]
        for record in files
        if Path(record["path"]).suffix.lower()
        in {".joblib", ".pickle", ".pkl", ".pt", ".pth"}
    )
    return {
        "passed": not unknown and not unmanifested and not unsafe_serialized_assets,
        "wheel_checked": str(wheel.resolve()) if wheel else None,
        "packaged_data_assets": sorted(packaged_data_names),
        "unknown_or_forbidden_assets": unknown,
        "unmanifested_assets": unmanifested,
        "unsafe_serialized_assets": unsafe_serialized_assets,
        "asset_license_statuses": {
            name: asset_license_status(record)
            for name, record in sorted(assets.items())
        },
    }


def workflow_policy(root: Path, workflow: Path | None = None) -> dict[str, Any]:
    path = workflow or root / ".github" / "workflows" / "release.yml"
    if not path.is_file():
        return {
            "passed": False,
            "path": str(path),
            "missing": list(REQUIRED_WORKFLOW_TOKENS),
        }
    contents = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_WORKFLOW_TOKENS if token not in contents]
    return {"passed": not missing, "path": str(path), "missing": missing}


def required_test_policy(root: Path) -> dict[str, Any]:
    path = root / "tests" / "test_release_pipeline.py"
    return {"passed": path.is_file(), "path": str(path)}


def sbom_policy(sbom: Path | None, wheel: Path | None) -> dict[str, Any]:
    if sbom is None:
        return {"passed": False, "reason": "sbom_required"}
    if not sbom.is_file():
        return {"passed": False, "reason": "sbom_missing"}
    try:
        document = json.loads(sbom.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"passed": False, "reason": "sbom_not_json"}
    format_ok = (
        document.get("bomFormat") == "CycloneDX"
        and document.get("specVersion") == "1.6"
        and document.get("$schema")
        == "https://cyclonedx.org/schema/bom-1.6.schema.json"
    )
    metadata_properties = {
        item.get("name"): item.get("value")
        for item in document.get("metadata", {}).get("properties", [])
        if isinstance(item, dict)
    }
    wheel_ok = wheel is None or metadata_properties.get("perfumery-ai:wheel") == str(
        wheel.resolve()
    )
    wheel_hash_ok = wheel is None or metadata_properties.get(
        "perfumery-ai:wheel-sha256"
    ) == sha256_file(wheel)
    files_ok = any(
        isinstance(component, dict)
        and component.get("type") == "file"
        and component.get("hashes")
        for component in document.get("components", [])
    )
    component_refs = [
        component.get("bom-ref")
        for component in document.get("components", [])
        if isinstance(component, dict)
    ]
    root_ref = document.get("metadata", {}).get("component", {}).get("bom-ref")
    dependency_rows = document.get("dependencies", [])
    dependency_ref_integrity = (
        bool(root_ref)
        and len(component_refs) == len(set(component_refs))
        and all(isinstance(ref, str) and ref for ref in component_refs)
        and all(
            isinstance(row, dict)
            and row.get("ref") in {root_ref, *component_refs}
            and all(item in component_refs for item in row.get("dependsOn", []))
            for row in dependency_rows
        )
    )
    return {
        "passed": (
            format_ok
            and wheel_ok
            and wheel_hash_ok
            and files_ok
            and dependency_ref_integrity
        ),
        "format_ok": format_ok,
        "wheel_ok": wheel_ok,
        "wheel_hash_ok": wheel_hash_ok,
        "files_ok": files_ok,
        "dependency_ref_integrity": dependency_ref_integrity,
    }


def evaluate_release_policy(
    root: Path = ROOT,
    wheel: Path | None = None,
    sbom: Path | None = None,
    workflow: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    wheel = wheel.resolve() if wheel is not None else None
    checks = {
        "project_license_consistency": dict(
            zip(("passed", "mode"), project_license_consistent(root))
        ),
        "hash_pinned_ci_dependencies": dependency_lock_policy(root),
        "hash_pinned_runtime_dependencies": runtime_dependency_lock_policy(root),
        "manifest_and_model_integrity": manifest_integrity(root),
        "packaged_asset_license_policy": package_asset_policy(root, wheel),
        "workflow_contract": workflow_policy(root, workflow),
        "required_tests_present": required_test_policy(root),
        "sbom_contract": sbom_policy(sbom, wheel),
        "wheel_exists": {
            "passed": wheel is not None and wheel.is_file(),
            "path": str(wheel) if wheel else None,
        },
    }
    passed = all(bool(check.get("passed")) for check in checks.values())
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed_closed",
        "passed": passed,
        "wheel_sha256": sha256_file(wheel)
        if wheel is not None and wheel.is_file()
        else None,
        "release_scope": "automated software supply-chain checks only",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_release_policy(args.root, args.wheel, args.sbom, args.workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
