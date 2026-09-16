"""Pinned private V69 inference closure; no raw training corpus or executables."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "v69-20260913"
WHEEL_REL = (
    "dist/shared-formulation-v69/build-04/perfumery_ai_core-1.4.0-py3-none-any.whl"
)
WHEEL_SHA256 = "039bb684df56908263473e85f7b7e69e26f72e2717c3d4812bc151b6c0aedc07"
PROFILE_SHA256 = "503cb708860c874f5c5696eb58816157253232e4e6bc79646b089bff08156a21"
CORE_SHA256 = "3186e467446abc71f249c052c8248c17c97d0e337976abf3666628af47dd997f"
ROLES = (
    "catalog",
    "perfume",
    "body_lotion",
    "atlas",
    "stock_mixture",
    "formulation_core",
    "lotion_target_reference",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _member(root, name):
    if (
        not isinstance(name, str)
        or "\\" in name
        or ":" in name
        or PurePosixPath(name).is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise ValueError("invalid bundle member path")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaping runtime member")
    return path


def collect(root=ROOT):
    root = Path(root).resolve()
    selected = root / "perfumery.local.json"
    if sha(selected) != PROFILE_SHA256:
        raise ValueError("selected V69 profile changed; prepare a new release")
    source = json.loads(selected.read_text(encoding="utf-8"))
    if (
        source.get("scope") != "local_research"
        or source.get("formulation_core", {}).get("sha256") != CORE_SHA256
    ):
        raise ValueError("explicit shared V69 research profile required")
    # Keep the exact historical stock-assay parents, but do not upload unused
    # V58/V60/V61/V62/V68 inference weights or the V62 raw correction bank.
    profile = {
        k: deepcopy(source[k]) for k in ("schema", "scope", "lotion_reference", *ROLES)
    }
    profile["language"] = None  # supplied by the existing private Linux worker
    files = {}

    def add(path, digest):
        path = Path(path).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or not _digest(digest)
            or sha(path) != digest
        ):
            raise ValueError("runtime dependency path or hash mismatch")
        name = path.relative_to(root).as_posix()
        if not name.endswith((".json", ".json.gz", ".npz", ".db", ".whl")):
            raise ValueError("non-inference file cannot enter the runtime bundle")
        if name in files and files[name] != digest:
            raise ValueError("conflicting artifact bindings")
        files[name] = digest
        return path

    add(root / WHEEL_REL, WHEEL_SHA256)
    for role in ROLES:
        item = profile[role]
        path = add(root / item["path"], item["sha256"])
        document = json.loads(path.read_text(encoding="utf-8"))
        if role == "catalog":
            binding = document["runtime_catalog"]
            if binding["wheel_sha256"] != WHEEL_SHA256:
                raise ValueError("catalog does not bind the release wheel")
            add(path.parent / binding["path"], binding["sha256"])
        elif role in ("perfume", "body_lotion"):
            for key in ("base_model", "component_model", "registry"):
                binding = document[key]
                add(path.parent / binding["path"], binding["sha256"])
        elif role == "formulation_core":
            if (
                document.get("schema") != "shared-formulation-core/v69"
                or document.get("accepted_for_local_inference") is not True
            ):
                raise ValueError("unaccepted shared neural checkpoint")
            binding = document["weights"]
            add(path.parent / binding["path"], binding["sha256"])
    return profile, files


def prepare(output, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new private bundle directory")
    profile, files = collect(root)
    output.mkdir(parents=True)
    for name, digest in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if sha(target) != digest:
            raise ValueError("copied artifact differs")
    profile_path = output / "perfumery.local.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files["perfumery.local.json"] = sha(profile_path)
    manifest = {
        "release_id": RELEASE_ID,
        "scope": "authenticated_noncommercial_research_service",
        "source_profile_sha256": PROFILE_SHA256,
        "profile_sha256": files["perfumery.local.json"],
        "wheel_sha256": WHEEL_SHA256,
        "shared_checkpoint_sha256": CORE_SHA256,
        "files": files,
        "public_data_redistribution_authorized": False,
        "raw_training_corpus_included": False,
        "legacy_weights_scope": "explicit_stock_aliquot_compatibility_only",
    }
    manifest_path = output / "bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = verify(output, sha(manifest_path))
    print(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "files": len(files),
                "bytes": sum((output / name).stat().st_size for name in files),
                "bundle_sha256": sha(manifest_path),
            }
        )
    )
    return result


def verify(root, expected=None):
    root = Path(root).resolve()
    path = root / "bundle.json"
    if expected is not None and (not _digest(expected) or sha(path) != expected):
        raise ValueError("runtime bundle hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("release_id") != RELEASE_ID
        or manifest.get("wheel_sha256") != WHEEL_SHA256
        or manifest.get("shared_checkpoint_sha256") != CORE_SHA256
        or manifest.get("scope") != "authenticated_noncommercial_research_service"
        or manifest.get("public_data_redistribution_authorized") is not False
        or manifest.get("raw_training_corpus_included") is not False
    ):
        raise ValueError("wrong runtime release or distribution scope")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty runtime inventory")
    if files.get(WHEEL_REL) != WHEEL_SHA256 or files.get(
        "perfumery.local.json"
    ) != manifest.get("profile_sha256"):
        raise ValueError("missing release wheel or profile binding")
    for name, digest in files.items():
        if not _digest(digest) or sha(_member(root, name)) != digest:
            raise ValueError("runtime bundle file changed")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"bundle.json"}:
        raise ValueError("unlisted files in private runtime bundle")
    profile = json.loads((root / "perfumery.local.json").read_text(encoding="utf-8"))
    if profile.get("language") is not None or profile.get("scope") != "local_research":
        raise ValueError("unexpected executable or environment selection")
    for role in ROLES:
        item = profile[role]
        if files.get(item["path"]) != item["sha256"]:
            raise ValueError("profile role missing from runtime inventory")
    if profile["formulation_core"]["sha256"] != CORE_SHA256:
        raise ValueError("profile selects a different shared checkpoint")
    return manifest


def check_installed(root):
    root = Path(root).resolve()
    manifest = verify(root)
    if os.environ.get("PERFUMERY_AI_ENV") != "research":
        raise ValueError("explicit research environment required")
    selected = Path(os.environ.get("PERFUMERY_AI_LOCAL_PROFILE", "")).resolve()
    if selected != root / "perfumery.local.json":
        raise ValueError("installed preflight must use the bundled profile")
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.formulation_views import shared_views
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from deploy.shared_runtime_v69 import create_release_app
    from fastapi.testclient import TestClient

    core = configured_formulation_core()
    perfume, lotion = configured_perception(), configured_perception("body_lotion")
    assert core.sha256 == CORE_SHA256 and perfume.core is lotion.core is core
    assert all(view.core is core for view in shared_views(core))
    assert shared_views(core)[1].predict(["CCO"]).shape == (1, 450)
    catalog, _ = load_configured_catalog()
    app = create_release_app()
    assert app.state.local_stock_mixture_predictor._model is None
    with TestClient(app) as client:
        assert client.get("/health").json()["wheel_sha256"] == WHEEL_SHA256
        capabilities = client.get("/v1/ai/capabilities").json()
        assert (
            capabilities["shared_formulation_model"]["checkpoint_sha256"] == CORE_SHA256
        )
        response = client.post(
            "/v1/formulation-workflows/plan", json={"product_type": "perfume"}
        )
        assert response.status_code == 200 and response.json()["learned_process"]
        # Run the same fixed regression brief on Linux before promoting the
        # image. This is a solver-parity check, not a human-accuracy estimate.
        response = client.post(
            "/v1/applications/body-lotion/design",
            json={"brief": "woody scent", "registry_pool": "conditional_research",
                  "max_risk_tier": 2},
        )
        assert response.status_code == 200
        lotion_result = response.json()
        print(json.dumps({"linux_lotion_regression": {
            key: lotion_result.get(key) for key in
            ("status", "score", "profile_target_met", "search_incomplete", "material_column_search")
        }}), flush=True)
        assert lotion_result["profile_target_met"] and lotion_result["score"] >= 95.
    import resource

    print(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "installed_model_preflight": "passed",
                "shared_checkpoint_sha256": core.sha256,
                "catalog_rows": len(catalog.ingredients),
                "stock_assay_lazy": app.state.local_stock_mixture_predictor._model
                is None,
                "maximum_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-installed", type=Path)
    args = parser.parse_args()
    if args.output:
        prepare(args.output)
    elif args.check_installed:
        check_installed(args.check_installed)
    else:
        parser.error("choose --output or --check-installed")
