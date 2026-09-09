from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.build_sbom import build_sbom, sha256_file, write_deterministic_json
from scripts.check_release_policy import (
    evaluate_release_policy,
    package_asset_policy,
    sbom_policy,
)


def _write_fixture_root(root: Path, *, unknown_asset: bool) -> Path:
    (root / "fragrance_ai" / "data").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\nversion = '0'\nlicense = 'LicenseRef-Proprietary'\ndependencies = []\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(
        "Fixture - Proprietary License\nAll rights reserved.\n", encoding="utf-8"
    )
    asset = root / "fragrance_ai" / "data" / "asset.json"
    asset.write_text("{}\n", encoding="utf-8")
    provenance = (
        "Original upstream source and license were not established."
        if unknown_asset
        else "Project-curated fixture."
    )
    manifest = {
        "assets": {
            "asset.json": {
                "sha256": sha256_file(asset),
                "bytes": asset.stat().st_size,
                "provenance": provenance,
            },
            "r2_ingredient_components.npz": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "physsim_r2_manifest.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "physsim_r2_ensemble_manifest.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "physsim_r2_runtime_manifest.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "physsim_r2_runtime_weights.npz": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "concentration_response_runtime.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "concentration_response_manifest.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
            "human_mixture_calibration.json": {
                "sha256": "0" * 64,
                "bytes": 0,
                "provenance": "Project-curated fixture.",
            },
        }
    }
    (root / "fragrance_ai" / "data" / "data_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (root / "tests" / "test_release_pipeline.py").write_text(
        "# required test marker\n", encoding="utf-8"
    )
    (root / ".github" / "workflows" / "release.yml").write_text(
        "\n".join(
            [
                "pytest",
                "python -m ruff",
                "python -m compileall",
                "node --check",
                "python -m build",
                "docker build",
                "check_release_policy.py",
                "full_inference_probe.py",
                "build_sbom.py",
                "upload-artifact",
                "download-artifact",
                "timeout-minutes",
                "requirements-runtime.lock",
            ]
        ),
        encoding="utf-8",
    )
    wheel = root / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fragrance_ai/data/asset.json", "{}\n")
        archive.writestr(
            "fragrance_ai/data/data_manifest.json", json.dumps(manifest, sort_keys=True)
        )
    return wheel


def test_sbom_output_is_byte_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    wheel = _write_fixture_root(root, unknown_asset=False)
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    write_deterministic_json(build_sbom(root, wheel), first)
    write_deterministic_json(build_sbom(root, wheel), second)
    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    properties = {
        item["name"]: item["value"] for item in document["metadata"]["properties"]
    }
    assert properties["perfumery-ai:reproducible-build-claim"] == "false"
    assert properties["perfumery-ai:wheel-sha256"] == sha256_file(wheel)
    refs = [component["bom-ref"] for component in document["components"]]
    assert len(refs) == len(set(refs))


def test_sbom_is_cryptographically_bound_to_the_exact_wheel(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    wheel = _write_fixture_root(root, unknown_asset=False)
    sbom = tmp_path / "sbom.json"
    write_deterministic_json(build_sbom(root, wheel), sbom)
    assert sbom_policy(sbom, wheel)["passed"]

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tampered-after-sbom.txt", "changed")
    policy = sbom_policy(sbom, wheel)
    assert not policy["passed"]
    assert not policy["wheel_hash_ok"]


def test_unknown_license_asset_is_rejected_when_packaged(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    wheel = _write_fixture_root(root, unknown_asset=True)
    policy = package_asset_policy(root, wheel)
    assert not policy["passed"]
    assert policy["unknown_or_forbidden_assets"] == ["asset.json"]


def test_unsafe_serialized_model_is_rejected_when_packaged(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    wheel = _write_fixture_root(root, unknown_asset=False)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("fragrance_ai/data/legacy_model.pt", b"serialized")
    policy = package_asset_policy(root, wheel)
    assert not policy["passed"]
    assert policy["unsafe_serialized_assets"] == ["fragrance_ai/data/legacy_model.pt"]


def test_release_policy_fails_closed_when_required_model_assets_are_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    wheel = _write_fixture_root(root, unknown_asset=False)
    sbom = tmp_path / "sbom.json"
    write_deterministic_json(build_sbom(root, wheel), sbom)
    report = evaluate_release_policy(root, wheel, sbom)
    assert report["status"] == "failed_closed"
    assert not report["checks"]["manifest_and_model_integrity"]["passed"]


def test_container_and_ci_contracts_limit_runtime_contents_and_pr_permissions() -> None:
    root = Path(__file__).resolve().parent.parent
    dockerfile = (root / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    probe = (root / "scripts" / "full_inference_probe.py").read_text(encoding="utf-8")

    assert "COPY . /app" not in dockerfile
    assert "--no-index --find-links=/wheels" in dockerfile
    assert dockerfile.count("python:3.11-slim@sha256:") == 2
    assert '"8000:8000"' in compose
    assert "postgres:17-alpine@sha256:" in compose
    assert "prom/prometheus:v3.5.0@sha256:" in compose
    assert "fragrance_ai/data/reference_fragrances.db" in dockerignore
    assert "fragrance_ai/data/*.joblib" in dockerignore
    assert "fragrance_ai/data/*.pt" in dockerignore
    assert "attest-wheel:" in workflow
    build_job = workflow.split("attest-wheel:", maxsplit=1)[0]
    assert "id-token: write" not in build_job
    assert "attestations: write" not in build_job
    assert 'client.get("/ui/")' in probe
    assert '"wheel_sha256"' in probe
    assert '"--wheelhouse"' in probe
    assert '"--no-index"' in probe
    assert "--minimum-tests 220" in workflow
