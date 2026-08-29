"""Install a wheel into a disposable venv and execute an end-to-end inference.

Unlike import-only package probes, this exercises the packaged catalog,
semantic parser, safety screen and simulation path.  It intentionally makes
no sensory-accuracy or commercial-release claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import venv
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    for line in metadata.splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise ValueError("wheel metadata has no Version field")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Optional complete local wheelhouse; enables a network-free locked install.",
    )
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    environment = args.venv.resolve()
    lock = args.lock.resolve()
    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    if not wheel.is_file():
        raise SystemExit(f"missing wheel: {wheel}")
    if environment.exists():
        raise SystemExit(
            f"refusing to reuse inference probe environment: {environment}"
        )
    if not lock.is_file():
        raise SystemExit(f"missing hash lock: {lock}")
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise SystemExit(f"missing wheelhouse directory: {wheelhouse}")
    expected_version = wheel_version(wheel)

    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    locked_install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--require-hashes",
    ]
    if wheelhouse is not None:
        locked_install_command.extend(["--no-index", f"--find-links={wheelhouse}"])
    locked_install_command.extend(["-r", str(lock)])
    locked_install = subprocess.run(
        locked_install_command,
        capture_output=True,
        text=True,
        timeout=300,
    )
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    probe_code = r"""
import json
import tempfile
from pathlib import Path
import fragrance_ai
from fastapi.testclient import TestClient
from fragrance_ai.api import TokenAuthorizer, create_app
from fragrance_ai.platform.postgres import PostgresWorkspaceStore
from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog

with NaturalLanguagePerfumeryAI() as ai:
    result = ai.create_recipe(
        "clean cool citrus woody scent without gourmand sweetness",
        RecipeConstraints(require_simulation_pass=False),
    )
package_root = Path(fragrance_ai.__file__).resolve().parent
ui_assets = [package_root / "ui" / name for name in ("index.html", "app.css", "app.js")]
unsafe_serialized_assets = sorted(
    path.name
    for path in (package_root / "data").iterdir()
    if path.suffix.lower() in {".joblib", ".pickle", ".pkl", ".pt", ".pth"}
)
with tempfile.TemporaryDirectory(prefix="perfumery-wheel-api-") as temporary:
    with AppendOnlyAuditLog(Path(temporary) / "audit.db", signing_key=b"wheel-probe-audit-key-material-32") as audit:
        app = create_app(
            ai_factory=NaturalLanguagePerfumeryAI,
            authorizer=TokenAuthorizer.from_plaintext(
                {"probe-token": ("probe-user", "admin", "probe-tenant")}
            ),
            audit_log=audit,
        )
        with TestClient(app) as client:
            health = client.get("/health/live")
            ui = client.get("/ui/")
print(json.dumps({
    "package_version": fragrance_ai.__version__,
    "package_root": str(package_root),
    "status": result.status,
    "recipe_count": len(result.recipe),
    "simulation_status": result.simulation_status,
    "safety_status": result.safety.status,
    "actual_olfactory_similarity_score": result.actual_olfactory_similarity_score,
    "actual_olfactory_lower_bound_95": result.actual_olfactory_lower_bound_95,
    "missing_ui_assets": [str(path) for path in ui_assets if not path.is_file()],
    "api_health_status": health.status_code,
    "api_health_payload": health.json(),
    "ui_status": ui.status_code,
    "ui_contains_formula_editor": 'id="formulaTable"' in ui.text,
    "postgres_backend_imported": PostgresWorkspaceStore.__name__ == "PostgresWorkspaceStore",
    "unsafe_serialized_assets": unsafe_serialized_assets,
}))
"""
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=environment,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    try:
        details = json.loads(probe.stdout.strip()) if probe.returncode == 0 else {}
    except json.JSONDecodeError:
        details = {}
    passed = (
        locked_install.returncode == 0
        and install.returncode == 0
        and probe.returncode == 0
        and details.get("package_version") == expected_version
        and str(environment).lower() in str(details.get("package_root", "")).lower()
        and details.get("status")
        in {
            "prototype_ready",
            "no_safe_match",
        }
        and "simulation_status" in details
        and "safety_status" in details
        and details.get("actual_olfactory_similarity_score") is None
        and details.get("actual_olfactory_lower_bound_95") is None
        and not details.get("missing_ui_assets", ["probe_failed"])
        and details.get("api_health_status") == 200
        and details.get("api_health_payload", {}).get("version") == expected_version
        and details.get("ui_status") == 200
        and details.get("ui_contains_formula_editor") is True
        and details.get("postgres_backend_imported") is True
        and not details.get("unsafe_serialized_assets", ["probe_failed"])
    )
    report = {
        "schema_version": "1.4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "claim_boundary": "Package execution probe only; it does not validate sensory accuracy, safety, or release approval.",
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "expected_package_version": expected_version,
        "lock": str(lock),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "wheelhouse": str(wheelhouse) if wheelhouse is not None else None,
        "offline_install": wheelhouse is not None,
        "venv": str(environment),
        "details": details,
        "install_returncode": install.returncode,
        "locked_install_returncode": locked_install.returncode,
        "locked_install_stderr_tail": locked_install.stderr[-2000:],
        "probe_returncode": probe.returncode,
        "install_stderr_tail": install.stderr[-2000:],
        "probe_stderr_tail": probe.stderr[-2000:],
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
