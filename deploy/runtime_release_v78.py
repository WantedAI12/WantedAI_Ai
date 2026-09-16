"""Minimal, hash-closed V78 research service artifacts; no deployment side effect."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from deploy.runtime_release_v69 import ROOT, sha, _digest, _member

RELEASE_ID = "v78-source-resolution-20260915"
CORE_SHA256 = "158b9f82ff9fb784eeec114ef1f7bbfc4d613badeb98f46e0f60435d5af38d2f"
ROLES = (
    "catalog",
    "perfume",
    "body_lotion",
    "atlas",
    "stock_mixture",
    "formulation_core",
    "lotion_target_reference",
    "public_evidence",
    "component_reference_observations",
    "physical_evidence",
    "odor_space",
)


def closure(root, source, wheel_path, wheel_sha):
    root = Path(root).resolve()
    if (
        source.get("schema") != "perfumery-local-runtime/v1"
        or source.get("scope") != "local_research"
        or source.get("formulation_core", {}).get("sha256") != CORE_SHA256
    ):
        raise ValueError(
            "exact accepted V76 neural parent and research profile required"
        )
    profile = {
        k: deepcopy(source[k]) for k in ("schema", "scope", "lotion_reference", *ROLES)
    }
    profile["language"] = None
    files = {}

    def add(path, digest, public=False):
        path = Path(path).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or not _digest(digest)
            or sha(path) != digest
        ):
            raise ValueError("dependency path/hash mismatch")
        name = path.relative_to(root).as_posix()
        allowed = (".json", ".json.gz", ".npz", ".db", ".whl") + (
            (".html", ".txt", ".pdf", ".xlsx") if public else ()
        )
        if not name.endswith(allowed):
            raise ValueError("unexpected private runtime file type")
        if name in files and files[name] != digest:
            raise ValueError("conflicting member digest")
        files[name] = digest
        return path

    add(root / wheel_path, wheel_sha)
    for role in ROLES:
        item = profile[role]
        path = add(root / item["path"], item["sha256"])
        value = json.loads(path.read_text(encoding="utf8"))
        if role == "catalog":
            binding = value["runtime_catalog"]
            if binding["wheel_sha256"] != wheel_sha:
                raise ValueError("catalog/wheel identity mismatch")
            add(path.parent / binding["path"], binding["sha256"])
        elif role in ("perfume", "body_lotion"):
            for key in ("base_model", "component_model", "registry"):
                binding = value[key]
                add(path.parent / binding["path"], binding["sha256"])
        elif role == "formulation_core":
            if (
                value.get("schema") != "shared-formulation-core/v76"
                or value.get("accepted_for_local_inference") is not True
            ):
                raise ValueError("unaccepted neural artifact")
            add(path.parent / value["weights"]["path"], value["weights"]["sha256"])
        elif role == "lotion_target_reference":
            from fragrance_ai.recommender.lotion_reference_objective import (
                ObservedReferenceBank,
            )

            bank = ObservedReferenceBank(path, item["sha256"])
            if bank.parent_sha256 != CORE_SHA256 or bank.resolution is None:
                raise ValueError("V78 typed reference resolution required")
            if (
                bank.odor_space.sha256 != profile["odor_space"]["sha256"]
                or bank.odor_space.path.resolve()
                != (root / profile["odor_space"]["path"]).resolve()
            ):
                raise ValueError("reference and language source spaces diverge")
    from fragrance_ai.platform.public_evidence import PublicEvidenceStore

    public = profile["public_evidence"]
    store = PublicEvidenceStore(root / public["path"], public["sha256"])
    for path, digest in store.members:
        add(path, digest, True)
    return profile, files


def prepare(preparation, output, root=ROOT, *, release_id=RELEASE_ID):
    root, output = Path(root).resolve(), Path(output).resolve()
    meta = json.loads(Path(preparation).read_text(encoding="utf8"))
    selected = Path(meta["profile"]).resolve()
    wheel = Path(meta["wheel"]).resolve()
    if (
        not selected.is_relative_to(root)
        or sha(selected) != meta["profile_sha256"]
        or not wheel.is_relative_to(root)
    ):
        raise ValueError("source-bound preparation required")
    profile, files = closure(
        root,
        json.loads(selected.read_text(encoding="utf8")),
        wheel.relative_to(root).as_posix(),
        meta["wheel_sha256"],
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if sha(target) != digest:
            raise ValueError("copy changed artifact")
    p = output / "perfumery.local.json"
    p.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    files[p.name] = sha(p)
    manifest = {
        "release_id": release_id,
        "scope": "authenticated_noncommercial_research_service",
        "shared_checkpoint_sha256": CORE_SHA256,
        "profile_sha256": sha(p),
        "wheel_path": wheel.relative_to(root).as_posix(),
        "wheel_sha256": meta["wheel_sha256"],
        "files": files,
        "raw_acquisition_archives_included": False,
        "public_data_redistribution_authorized": False,
        "typed_model_estimates_not_observations": True,
        "all_3595_measured_profiles_completed": False,
    }
    target = output / "bundle.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    verify(output, sha(target), release_id=release_id)
    report = {
        "bundle": str(output),
        "bundle_sha256": sha(target),
        "files": len(files),
        "bytes": sum((output / n).stat().st_size for n in files),
        "wheel_sha256": meta["wheel_sha256"],
        "wheel_path": manifest["wheel_path"],
        "deployed": False,
    }
    print(json.dumps(report))
    return report


def verify(root, expected, *, release_id=RELEASE_ID):
    root = Path(root).resolve()
    p = root / "bundle.json"
    if not _digest(expected) or sha(p) != expected:
        raise ValueError("trusted bundle hash required")
    value = json.loads(p.read_text(encoding="utf8"))
    if (
        value.get("release_id") != release_id
        or value.get("shared_checkpoint_sha256") != CORE_SHA256
        or value.get("scope") != "authenticated_noncommercial_research_service"
        or value.get("raw_acquisition_archives_included") is not False
        or value.get("typed_model_estimates_not_observations") is not True
        or value.get("public_data_redistribution_authorized") is not False
        or value.get("all_3595_measured_profiles_completed") is not False
    ):
        raise ValueError("wrong release or evidence scope")
    files = value["files"]
    for name, digest in files.items():
        if sha(_member(root, name)) != digest:
            raise ValueError("artifact changed")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"bundle.json"}:
        raise ValueError("unlisted runtime artifact")
    profile = json.loads((root / "perfumery.local.json").read_text(encoding="utf8"))
    if (
        files.get("perfumery.local.json") != value["profile_sha256"]
        or profile.get("language") is not None
    ):
        raise ValueError("unbound runtime profile")
    _, required = closure(root, profile, value["wheel_path"], value["wheel_sha256"])
    if required != {k: v for k, v in files.items() if k != "perfumery.local.json"}:
        raise ValueError("incomplete runtime closure")
    return value


def registry_path(root, manifest):
    from deploy.runtime_release_v70 import registry_path as resolve

    return resolve(root, manifest)


def check_installed(root, expected, *, release_id=RELEASE_ID, data_version='v78'):
    root = Path(root).resolve()
    manifest = verify(root, expected, release_id=release_id)
    if (
        os.environ.get("PERFUMERY_AI_ENV") != "research"
        or Path(os.environ.get("PERFUMERY_AI_LOCAL_PROFILE", "")).resolve()
        != root / "perfumery.local.json"
    ):
        raise ValueError("bundled research profile required")
    from deploy.system_runtime_v76 import create_local_app
    from fastapi.testclient import TestClient
    from fragrance_ai.recommender.formulation_core import configured_formulation_core

    with TestClient(create_local_app(registry_path=registry_path(root, manifest))) as c:
        h = c.get("/health")
        cap = c.get("/v1/ai/capabilities")
        assert h.status_code == cap.status_code == 200
        assert h.json()["wheel_sha256"] == manifest["wheel_sha256"]
        assert (
            cap.json()["odor_expression"]["hierarchical_space"]["data_version"] == data_version
        )
        assert configured_formulation_core().sha256 == CORE_SHA256
    verify(root, expected, release_id=release_id)
    result = {
        "installed_bundle_valid": True,
        "release_id": release_id,
        "wheel_sha256": manifest["wheel_sha256"],
        "checkpoint_sha256": CORE_SHA256,
        "auth_required_at_modal_edge": True,
    }
    print(json.dumps(result))
    return result


def assert_deployment_ready(root, expected):
    """Local preparation is allowed; actual deployment is held by user choice."""
    verify(root, expected)
    root = Path(root)
    profile = json.loads((root / "perfumery.local.json").read_text(encoding="utf8"))
    space = json.loads(
        (root / profile["odor_space"]["path"]).read_text(encoding="utf8")
    )
    reference = json.loads(
        (root / profile["lotion_target_reference"]["path"]).read_text(encoding="utf8")
    )
    from fragrance_ai.recommender.odor_resolution import deployment_readiness

    status = deployment_readiness(space, reference["profiles"])
    if not status["ready"]:
        raise ValueError(
            "DEPLOYMENT_HELD: all 134 required references must be present; missing="
            + str(len(status["missing_reference_ids"]))
        )
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--verify", type=Path)
    p.add_argument("--check-installed", type=Path)
    p.add_argument("--sha256")
    a = p.parse_args()
    if a.check_installed:
        check_installed(a.check_installed, a.sha256)
    elif a.verify:
        verify(a.verify, a.sha256)
    elif a.preparation and a.output:
        prepare(a.preparation, a.output)
    else:
        p.error("select prepare, verify, or installed check")
