"""Create a deterministic CycloneDX 1.6 software bill of materials.

The document inventories the built wheel byte-for-byte, records declared and
resolved dependency state, and carries the project's data-distribution policy
as namespaced CycloneDX properties.  A hash-pinned dependency lock improves
provenance but is deliberately not represented as a cross-platform,
bit-for-bit reproducible-build claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import tomllib
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PREFIX = "fragrance_ai/"
DATA_PREFIX = "fragrance_ai/data/"
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    if not match:
        raise ValueError(f"cannot parse dependency name: {requirement!r}")
    return match.group(1)


def read_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def declared_requirements(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject.get("project", {})
    requirements = list(project.get("dependencies", []))
    for group, values in sorted(project.get("optional-dependencies", {}).items()):
        requirements.extend(f"{value}  # optional:{group}" for value in values)
    return sorted(requirements, key=lambda item: (normalise_name(requirement_name(item)), item))


def distribution_record(requirement: str) -> dict[str, Any]:
    name = requirement_name(requirement)
    try:
        metadata = importlib.metadata.metadata(name)
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "requirement": requirement,
            "name": name,
            "installed": False,
            "version": None,
            "license": None,
            "license_status": "not_installed",
        }

    license_expression = metadata.get("License-Expression") or metadata.get("License")
    classifiers = metadata.get_all("Classifier") or []
    classifier_licenses = [entry for entry in classifiers if entry.startswith("License ::")]
    declared_license = license_expression or ("; ".join(classifier_licenses) if classifier_licenses else None)
    return {
        "requirement": requirement,
        "name": name,
        "installed": True,
        "version": version,
        "license": declared_license,
        "license_status": "declared" if declared_license else "not_declared",
    }


def asset_license_status(asset: dict[str, Any] | None) -> str:
    if asset is None:
        return "missing_manifest_entry"
    provenance = str(asset.get("provenance", "")).lower()
    if not provenance:
        return "missing_provenance"
    unknown_markers = (
        "license were not established",
        "license not established",
        "unverified historical",
        "unknown license",
    )
    if any(marker in provenance for marker in unknown_markers):
        return "unknown_or_unverified"
    return "declared_provenance"


def asset_distribution_allowed(asset: dict[str, Any] | None) -> bool:
    return asset_license_status(asset) == "declared_provenance"


def _source_files(root: Path) -> Iterable[tuple[str, bytes]]:
    package_root = root / "fragrance_ai"
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path.relative_to(root).as_posix(), path.read_bytes()


def _wheel_files(wheel: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(wheel) as archive:
        for info in sorted(archive.infolist(), key=lambda member: member.filename):
            # A wheel's dist-info metadata and embedded license files are also
            # part of the distributable artifact, so do not limit this to the
            # importable package directory.
            if info.is_dir():
                continue
            yield info.filename, archive.read(info)


def package_files(root: Path, wheel: Path | None = None) -> list[dict[str, Any]]:
    manifest_path = root / "fragrance_ai" / "data" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    assets = manifest.get("assets", {})
    file_source = _wheel_files(wheel) if wheel is not None else _source_files(root)
    files: list[dict[str, Any]] = []
    for relative, contents in file_source:
        record: dict[str, Any] = {
            "path": relative,
            "sha256": sha256_bytes(contents),
            "bytes": len(contents),
        }
        if relative.startswith(DATA_PREFIX):
            asset = assets.get(Path(relative).name)
            record["asset_license_status"] = asset_license_status(asset)
            record["distribution_allowed"] = asset_distribution_allowed(asset)
            if asset:
                record["prohibited_claim"] = asset.get("prohibited_claim")
        files.append(record)
    return files


def dependency_lock_status(root: Path) -> dict[str, Any]:
    """Report facts about locking without claiming a lock that does not exist."""
    candidates = sorted(root.glob("*lock*")) + sorted(root.glob("requirements*.txt"))
    hashed = False
    examined: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        examined.append(candidate.name)
        try:
            hashed = hashed or "--hash=" in candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return {
        "files_examined": examined,
        "hash_pinned_resolution_present": hashed,
        "reproducible_build_claim": False,
        "statement": (
            "Dependency hashes are not a complete cross-platform build proof; this SBOM "
            "makes no reproducible-build claim."
            if hashed
            else "No hash-pinned dependency resolution was found; this SBOM is provenance, not a reproducible-build claim."
        ),
    }


def build_sbom(root: Path = ROOT, wheel: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    pyproject = read_pyproject(root)
    project = pyproject["project"]
    files = package_files(root, wheel)
    dependencies = [distribution_record(entry) for entry in declared_requirements(pyproject)]
    project_name = str(project["name"])
    project_version = str(project["version"])
    root_ref = f"pkg:pypi/{normalise_name(project_name)}@{project_version}"
    lock_status = dependency_lock_status(root)

    components: list[dict[str, Any]] = []
    dependency_refs: list[str] = []
    grouped_dependencies: dict[str, list[dict[str, Any]]] = {}
    for dependency in dependencies:
        grouped_dependencies.setdefault(
            normalise_name(str(dependency["name"])), []
        ).append(dependency)
    for normalized_name, records in sorted(grouped_dependencies.items()):
        installed_records = [record for record in records if record["installed"]]
        dependency = installed_records[0] if installed_records else records[0]
        name = str(dependency["name"])
        version = str(dependency["version"] or "not-resolved")
        ref = f"dependency:{normalise_name(name)}:{version}"
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": ref,
            "name": name,
            "version": version,
            "properties": [
                {
                    "name": "perfumery-ai:declared-requirements",
                    "value": json.dumps(
                        sorted({str(record["requirement"]) for record in records}),
                        separators=(",", ":"),
                    ),
                },
                {"name": "perfumery-ai:installed", "value": str(bool(dependency["installed"])).lower()},
                {"name": "perfumery-ai:license-status", "value": str(dependency["license_status"])},
            ],
        }
        if dependency["installed"]:
            component["purl"] = (
                f"pkg:pypi/{normalise_name(name)}@{dependency['version']}"
            )
        if dependency["license"]:
            component["licenses"] = [
                {"license": {"name": str(dependency["license"])}}
            ]
        components.append(component)
        dependency_refs.append(ref)

    file_refs: list[str] = []
    for record in files:
        path = str(record["path"])
        ref = f"file:{path}"
        properties = [
            {"name": "perfumery-ai:bytes", "value": str(record["bytes"])},
        ]
        if "asset_license_status" in record:
            properties.extend(
                [
                    {
                        "name": "perfumery-ai:asset-license-status",
                        "value": str(record["asset_license_status"]),
                    },
                    {
                        "name": "perfumery-ai:distribution-allowed",
                        "value": str(bool(record["distribution_allowed"])).lower(),
                    },
                ]
            )
        if record.get("prohibited_claim"):
            properties.append(
                {
                    "name": "perfumery-ai:prohibited-claim",
                    "value": str(record["prohibited_claim"]),
                }
            )
        components.append(
            {
                "type": "file",
                "bom-ref": ref,
                "name": path,
                "hashes": [{"alg": "SHA-256", "content": str(record["sha256"])}],
                "properties": properties,
            }
        )
        file_refs.append(ref)

    serial = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{project_name}:{project_version}:{sha256_file(root / 'pyproject.toml')}",
    )
    declared_license = project.get("license")
    if isinstance(declared_license, dict) and declared_license.get("file"):
        component_licenses = [
            {
                "license": {
                    "name": (
                        "Proprietary license; see "
                        + str(declared_license["file"])
                    )
                }
            }
        ]
    else:
        component_licenses = [
            {"expression": str(declared_license or "NOASSERTION")}
        ]
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": project_name,
                "version": project_version,
                "purl": root_ref,
                "licenses": component_licenses,
            },
            "properties": [
                {
                    "name": "perfumery-ai:source-root",
                    "value": str(root),
                },
                {
                    "name": "perfumery-ai:wheel",
                    "value": str(wheel.resolve()) if wheel is not None else "",
                },
                {
                    "name": "perfumery-ai:wheel-sha256",
                    "value": sha256_file(wheel) if wheel is not None else "",
                },
                {
                    "name": "perfumery-ai:hash-pinned-resolution-present",
                    "value": str(bool(lock_status["hash_pinned_resolution_present"])).lower(),
                },
                {
                    "name": "perfumery-ai:reproducible-build-claim",
                    "value": "false",
                },
                {
                    "name": "perfumery-ai:lock-files-examined",
                    "value": json.dumps(lock_status["files_examined"], separators=(",", ":")),
                },
                {
                    "name": "perfumery-ai:reproducibility-statement",
                    "value": str(lock_status["statement"]),
                },
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": dependency_refs + file_refs},
            *({"ref": ref, "dependsOn": []} for ref in dependency_refs + file_refs),
        ],
    }


def write_deterministic_json(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--wheel", type=Path, help="Optional wheel whose package contents are inventoried")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.wheel is not None and not args.wheel.is_file():
        raise SystemExit(f"wheel does not exist: {args.wheel}")
    document = build_sbom(args.root, args.wheel)
    write_deterministic_json(document, args.output)
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
