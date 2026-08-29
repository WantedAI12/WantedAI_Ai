"""Install a wheel into a fresh venv and verify its packaged runtime contract.

The default check is deliberately offline-light. The production R2 and
concentration paths use portable NumPy inference and do not need Torch, RDKit,
scikit-learn, or joblib. ``--with-physsim`` remains a compatibility alias;
``--verify-model-inference`` executes both frozen model paths.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument("--venv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--with-physsim",
        action="store_true",
        help="install the compatibility physsim extra and require frozen-model inference",
    )
    parser.add_argument(
        "--verify-model-inference",
        action="store_true",
        help="require portable concentration-response and frozen R2 inference",
    )
    parser.add_argument(
        "--install-timeout",
        type=int,
        default=300,
        help="pip install timeout in seconds; relevant when --with-physsim downloads extras",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="optional local dependency wheelhouse; forces an offline --no-index install",
    )
    args = parser.parse_args()

    wheel = Path(args.wheel).resolve()
    environment = Path(args.venv).resolve()
    output = Path(args.output).resolve()
    if not wheel.is_file():
        raise SystemExit(f"missing wheel: {wheel}")
    if environment.exists():
        raise SystemExit(f"refusing to reuse verification venv: {environment}")
    if args.install_timeout <= 0:
        raise SystemExit("--install-timeout must be positive")
    expected_version = wheel_version(wheel)
    with zipfile.ZipFile(wheel) as archive:
        unsafe_serialized_members = sorted(
            name
            for name in archive.namelist()
            if Path(name).suffix.lower()
            in {".joblib", ".pickle", ".pkl", ".pt", ".pth"}
        )
    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise SystemExit(f"missing wheelhouse: {wheelhouse}")

    require_model_inference = args.with_physsim or args.verify_model_inference
    install_spec = str(wheel)
    if args.with_physsim:
        # PEP 508 direct-reference form retains the requested optional extra.
        install_spec = f"perfumery-ai-core[physsim] @ {wheel.as_uri()}"

    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if wheelhouse is not None:
        install_command.extend(("--no-index", "--find-links", str(wheelhouse)))
    install_command.append(install_spec)
    install = subprocess.run(
        install_command,
        capture_output=True,
        text=True,
        timeout=args.install_timeout,
    )

    probe_code = r"""
import json
import os
import sys
from pathlib import Path

import fragrance_ai
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog, HistoricalReferenceCorpus

data = Path(fragrance_ai.__file__).resolve().parent / "data"
required = [
    "safe_ingredient_catalog.json",
    "odor_descriptor_projections.json",
    "data_manifest.json",
    "r2_ingredient_components.npz",
    "physsim_r2_ensemble_manifest.json",
    "physsim_r2_runtime_manifest.json",
    "physsim_r2_runtime_weights.npz",
    "concentration_response_manifest.json",
    "concentration_response_runtime.json",
    "continuous_improvement_policy.json",
    "human_mixture_calibration.json",
]
brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(
    "cool ocean breeze with dry wood, avoid gourmand"
)
corpus = HistoricalReferenceCorpus()
service_result = None
try:
    from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI
    with NaturalLanguagePerfumeryAI() as service:
        service_result = service.create_recipe(
            "clean cool citrus woody, avoid gourmand"
        )
except Exception as error:
    service_result = {"error": f"{type(error).__name__}:{error}"}

model_probe = {"requested": bool(os.environ.get("PERFUMERY_VERIFY_MODEL"))}
if model_probe["requested"]:
    try:
        from fragrance_ai.recommender.concentration_response import FrozenConcentrationResponse
        from fragrance_ai.recommender.models import RecipeLine
        from fragrance_ai.recommender.physsim_checkpoint import FrozenR2PhysSim

        line = RecipeLine(
            ingredient_id="dihydromyrcenol", name="Dihydromyrcenol", pyramid="top",
            concentrate_percent=10.0, finished_product_percent=1.5,
            volume_ml_for_batch=None, price_per_kg=18.0, availability=0.99,
            risk_tier=1, reason="wheel inference probe",
        )
        concentration = FrozenConcentrationResponse()
        low, low_in_domain = concentration.intensity(0.0001)
        high, high_in_domain = concentration.intensity(0.1)
        r2_adapter = FrozenR2PhysSim()
        r2 = r2_adapter.evaluate([line], [line])
        model_probe.update({
            "concentration_low": low,
            "concentration_high": high,
            "concentration_in_domain": low_in_domain and high_in_domain,
            "r2_status": r2.status,
            "r2_similarity": r2.similarity,
            "r2_member_predictions": list(r2.member_predictions_percent),
            "runtime_model_classes": [
                item.__class__.__name__ for item in r2_adapter._models
            ],
            "forbidden_runtime_modules_loaded": [
                name
                for name in ("torch", "sklearn", "joblib")
                if name in sys.modules
            ],
            "passed": bool(
                low_in_domain and high_in_domain and high > low
                # A release gate may deliberately set the ensemble weight to
                # zero. That is fail-closed scoring, not failed inference: the
                # wheel contract here proves that both frozen members executed.
                and r2.status not in {"unavailable", "outside_applicability"}
                and r2.similarity is not None
                and len(r2.member_predictions_percent) == 2
                and all(
                    name == "NumpyR2Model"
                    for name in [
                        item.__class__.__name__ for item in r2_adapter._models
                    ]
                )
                and not any(
                    name in sys.modules for name in ("torch", "sklearn", "joblib")
                )
            ),
        })
    except Exception as error:
        model_probe.update({"passed": False, "error": f"{type(error).__name__}:{error}"})

print(json.dumps({
    "package_version": fragrance_ai.__version__,
    "package_root": str(Path(fragrance_ai.__file__).resolve().parent),
    "missing_assets": [name for name in required if not (data / name).is_file()],
    "desired_dimensions": brief.desired_dimensions,
    "avoided_dimensions": brief.avoided_dimensions,
    "semantic_backend": brief.semantic_backend,
    "ontology_version": brief.ontology_version,
    "reference_database_packaged": (data / "reference_fragrances.db").is_file(),
    "reference_corpus_total_perfumes": corpus.total_perfumes,
    "reference_corpus_fallback_safe": (
        not (data / "reference_fragrances.db").is_file() and corpus.total_perfumes == 0
    ),
    "service_fallback_status": getattr(service_result, "status", None),
    "service_fallback_error": service_result.get("error") if isinstance(service_result, dict) else None,
    "model_inference": model_probe,
    "unsafe_serialized_assets_packaged": sorted(
        item.name
        for item in data.iterdir()
        if item.suffix.lower() in {".joblib", ".pickle", ".pkl", ".pt", ".pth"}
    ),
}))
"""
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=environment,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PERFUMERY_VERIFY_MODEL": "1" if require_model_inference else "",
        },
    )
    try:
        details = json.loads(probe.stdout.strip()) if probe.returncode == 0 else {}
    except json.JSONDecodeError:
        details = {}
    passed = (
        install.returncode == 0
        and probe.returncode == 0
        and details.get("package_version") == expected_version
        and str(environment).lower() in str(details.get("package_root", "")).lower()
        and not details.get("missing_assets", ["probe_failed"])
        and details.get("ontology_version") == "scent-ontology-2.0.0"
        and "aquatic" in details.get("desired_dimensions", [])
        and "woody" in details.get("desired_dimensions", [])
        and "gourmand" in details.get("avoided_dimensions", [])
        and details.get("reference_database_packaged") is False
        and details.get("reference_corpus_fallback_safe") is True
        and not details.get("service_fallback_error")
        and not details.get("unsafe_serialized_assets_packaged", ["probe_failed"])
        and not unsafe_serialized_members
    )
    if require_model_inference:
        passed = passed and details.get("model_inference", {}).get("passed") is True
    result = {
        "schema_version": "1.4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "venv": str(environment),
        "package_version": details.get("package_version"),
        "expected_package_version": expected_version,
        "probe": details,
        "model_inference_required": require_model_inference,
        "install_spec": install_spec,
        "wheelhouse": str(wheelhouse) if wheelhouse is not None else None,
        "offline_install": wheelhouse is not None,
        "install_returncode": install.returncode,
        "probe_returncode": probe.returncode,
        "install_stderr_tail": install.stderr[-2000:],
        "probe_stderr_tail": probe.stderr[-2000:],
        "unsafe_serialized_wheel_members": unsafe_serialized_members,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
