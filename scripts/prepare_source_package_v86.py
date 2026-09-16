"""Freeze current source while preserving every original material/model/target."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-output", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=ROOT)
    a = p.parse_args()
    if a.output.exists() or a.profile_output.exists() or a.profile_output.resolve().parent != ROOT:
        raise ValueError("new package directory and root-local profile required")
    profile = json.loads(a.parent_profile.read_text(encoding="utf8"))
    manifest_path = ROOT / profile["catalog"]["path"]
    if sha(manifest_path) != profile["catalog"]["sha256"]:
        raise ValueError("parent catalogue checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    binding = manifest["runtime_catalog"]
    catalog_path = manifest_path.parent / binding["path"]
    if sha(catalog_path) != binding["sha256"]:
        raise ValueError("parent material data drift")
    raw = json.loads(gzip.decompress(catalog_path.read_bytes()))
    original = deepcopy(raw)
    a.output.mkdir(parents=True)
    build = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(a.output.resolve() / "wheel")],
                           cwd=a.source_root.resolve(), capture_output=True, text=True, encoding="utf8")
    (a.output / "build.log").write_text(build.stdout + build.stderr, encoding="utf8")
    if build.returncode:
        raise ValueError("wheel build failed; see build.log")
    wheel = next((a.output / "wheel").glob("*.whl"))
    installed = a.output / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(not (installed / n).resolve().is_relative_to(installed.resolve()) for n in names):
            raise ValueError("unsafe wheel entry")
        if any(n.endswith((".pt", ".pkl", ".pickle", ".joblib", ".pth")) for n in names):
            raise ValueError("unsafe serialized runtime asset")
        manifest["runtime_source_sha256"] = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names if n.startswith("fragrance_ai/") and n.endswith(".py")}
        manifest["runtime_data_sha256"] = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names if n.startswith("fragrance_ai/data/") and Path(n).suffix in (".db", ".json", ".npz")}
        archive.extractall(installed)
    shutil.copytree(a.source_root.resolve() / "deploy", installed / "deploy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheel_hash = sha(wheel)
    raw["wheel_sha256"] = wheel_hash
    if {k: v for k, v in raw.items() if k != "wheel_sha256"} != {k: v for k, v in original.items() if k != "wheel_sha256"}:
        raise ValueError("material data cannot change in a source-only package")
    target = a.output / "catalog"
    target.mkdir()
    catalog_file = target / "runtime_catalog_v3.json.gz"
    catalog_file.write_bytes(gzip.compress(json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    binding.update(path=catalog_file.name, sha256=sha(catalog_file), wheel_sha256=wheel_hash)
    manifest.update(status="local_candidate_not_deployed", source_only_rebind=True)
    new_manifest = target / "catalog_manifest.json"
    write(new_manifest, manifest)
    profile["catalog"] = {"path": new_manifest.resolve().relative_to(ROOT).as_posix(), "sha256": sha(new_manifest)}
    write(a.profile_output, profile)
    record = {"wheel": str(wheel.resolve()), "wheel_sha256": wheel_hash,
              "installed": str(installed.resolve()), "profile": str(a.profile_output.resolve()), "profile_sha256": sha(a.profile_output),
              "catalog_sha256": sha(new_manifest), "material_rows": len(raw["ingredients"]),
              "model_sha256": profile["formulation_core"]["sha256"],
              "target_reference_sha256": profile["lotion_target_reference"]["sha256"],
              "data_unchanged": True, "deployed": False,
              "deploy_sources": {p.relative_to(installed).as_posix(): sha(p) for p in (installed / "deploy").rglob("*.py")}}
    write(a.output / "preparation.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "deploy_sources"}), flush=True)


if __name__ == "__main__":
    main()
