"""Fail-closed audit of packaged data lineage, contracts, and database counts."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.numpy_r2 import EXPECTED_STATE_SHAPES  # noqa: E402
from fragrance_ai.recommender.continuous_improvement import (  # noqa: E402
    ContinuousImprovementController,
)
from fragrance_ai.research.prospective_formula_study import (  # noqa: E402
    verify_study_seal,
)

DATA = ROOT / "fragrance_ai" / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    verified: dict[str, dict[str, object]] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for name, expected in manifest["assets"].items():
        path = DATA / name
        actual_hash = sha256(path) if path.exists() else "missing"
        actual_bytes = path.stat().st_size if path.exists() else 0
        require(actual_hash == expected["sha256"], f"{name}: sha256 mismatch")
        if "bytes" in expected:
            require(actual_bytes == expected["bytes"], f"{name}: byte-size mismatch")
        verified[name] = {"sha256": actual_hash, "bytes": actual_bytes}

    with sqlite3.connect(
        (DATA / "reference_fragrances.db").resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        reference_counts = {
            "perfumes": scalar(connection, "SELECT COUNT(*) FROM perfumes"),
            "perfume_notes": scalar(connection, "SELECT COUNT(*) FROM perfume_notes"),
            "ingredients": scalar(connection, "SELECT COUNT(*) FROM ingredients"),
        }
    require(
        reference_counts
        == {"perfumes": 10000, "perfume_notes": 103419, "ingredients": 242},
        f"reference database counts changed: {reference_counts}",
    )

    with sqlite3.connect(
        (DATA / "scientific_properties.db").resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        scientific_counts = {
            "rows": scalar(connection, "SELECT COUNT(*) FROM molecular_properties"),
            "threshold_rows": scalar(
                connection,
                "SELECT COUNT(*) FROM molecular_properties WHERE odor_threshold_ppm IS NOT NULL",
            ),
            "composition_centroid_rows": scalar(
                connection,
                "SELECT COUNT(*) FROM molecular_properties WHERE source_ref LIKE 'composition-derived UVCB descriptor centroid;%'",
            ),
        }
    require(
        scientific_counts
        == {"rows": 33, "threshold_rows": 5, "composition_centroid_rows": 4},
        f"scientific database contract changed: {scientific_counts}",
    )

    with sqlite3.connect(
        (DATA / "nonhuman_data_hub.db").resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        hub_counts = {
            "sources": scalar(connection, "SELECT COUNT(*) FROM data_sources"),
            "observations": scalar(
                connection, "SELECT COUNT(*) FROM material_observations"
            ),
            "reference_formulas": scalar(
                connection, "SELECT COUNT(*) FROM formula_references"
            ),
            "reference_notes": scalar(connection, "SELECT COUNT(*) FROM formula_notes"),
            "threshold_observations": scalar(
                connection,
                "SELECT COUNT(*) FROM material_observations WHERE property_name = 'odor_threshold_ppm'",
            ),
            "composition_observations": scalar(
                connection,
                "SELECT COUNT(*) FROM material_observations WHERE evidence_class LIKE 'composition%'",
            ),
        }
        non_reference = scalar(
            connection,
            "SELECT COUNT(*) FROM formula_references WHERE reference_only != 1",
        )
    require(
        hub_counts
        == {
            "sources": 75,
            "observations": 7612,
            "reference_formulas": 98,
            "reference_notes": 605,
            "threshold_observations": 0,
            "composition_observations": 28,
        },
        f"non-human hub contract changed: {hub_counts}",
    )
    require(
        non_reference == 0,
        "non-human formula reference escaped reference-only quarantine",
    )

    with sqlite3.connect(
        (DATA / "epa_comptox_extract.db").resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        epa_counts = {
            "source_files": scalar(connection, "SELECT COUNT(*) FROM source_files"),
            "chemicals": scalar(connection, "SELECT COUNT(*) FROM chemicals"),
        }
    require(
        epa_counts == {"source_files": 5, "chemicals": 38},
        f"EPA extract contract changed: {epa_counts}",
    )

    headspace_paths = {
        "database": ROOT / "benchmarks" / "headspace_sensory_hub_v1.db",
        "report": ROOT / "benchmarks" / "headspace_sensory_hub_v1.json",
        "builder": ROOT / "scripts" / "build_headspace_sensory_hub_v1.py",
        "calibration": ROOT
        / "benchmarks"
        / "concentration_headspace_calibration_v1.json",
        "calibration_script": ROOT
        / "scripts"
        / "calibrate_concentration_headspace_v1.py",
        "dream_report": ROOT / "benchmarks" / "dream_headspace_retrospective_v3.json",
        "dream_script": ROOT / "scripts" / "benchmark_dream_headspace_v3.py",
        "dream_accuracy_report": ROOT / "benchmarks" / "dream_accuracy_search_v4.json",
        "dream_accuracy_script": ROOT / "scripts" / "experiment_dream_accuracy_v4.py",
        "dream_set_encoder_report": ROOT
        / "benchmarks"
        / "dream_set_encoder_search_v5.json",
        "dream_set_encoder_script": ROOT
        / "scripts"
        / "experiment_dream_set_encoder_v5.py",
        "vp_report": ROOT / "benchmarks" / "opera_vapor_pressure_imputer_v1.json",
        "vp_runtime": ROOT / "benchmarks" / "opera_vapor_pressure_runtime_v1.json",
        "vp_script": ROOT / "scripts" / "train_opera_vapor_pressure_imputer_v1.py",
    }
    headspace_counts: dict[str, int] = {}
    require(
        all(path.is_file() for path in headspace_paths.values()),
        "headspace/concentration research evidence is incomplete",
    )
    if all(path.is_file() for path in headspace_paths.values()):
        headspace_report = json.loads(
            headspace_paths["report"].read_text(encoding="utf-8")
        )
        concentration_report = json.loads(
            headspace_paths["calibration"].read_text(encoding="utf-8")
        )
        dream_headspace_report = json.loads(
            headspace_paths["dream_report"].read_text(encoding="utf-8")
        )
        dream_accuracy_report = json.loads(
            headspace_paths["dream_accuracy_report"].read_text(encoding="utf-8")
        )
        dream_set_encoder_report = json.loads(
            headspace_paths["dream_set_encoder_report"].read_text(encoding="utf-8")
        )
        vp_report = json.loads(headspace_paths["vp_report"].read_text(encoding="utf-8"))
        vp_runtime = json.loads(
            headspace_paths["vp_runtime"].read_text(encoding="utf-8")
        )
        headspace_counts = {
            str(name): int(value)
            for name, value in headspace_report.get("counts", {}).items()
        }
        require(
            headspace_report.get("schema") == "headspace-sensory-hub/v1"
            and headspace_report.get("database", {}).get("sha256")
            == sha256(headspace_paths["database"])
            and headspace_report.get("software", {}).get("script_sha256")
            == sha256(headspace_paths["builder"]),
            "headspace sensory hub report binding failed",
        )
        require(
            headspace_counts
            == {
                "source_files": 29,
                "molecules": 1642,
                "molecule_source_links": 2449,
                "physchem_observations": 27283,
                "molecules_with_physchem": 869,
                "stimuli": 2689,
                "stimulus_components": 21708,
                "stimulus_dilutions": 1473,
                "sensory_observations": 109688,
            },
            f"headspace sensory hub counts changed: {headspace_counts}",
        )
        require(
            headspace_report.get("source", {}).get("pyrfume_commit")
            == "8054ea98ed675005ec10e67359902f500e4911b0"
            and headspace_report.get("source", {}).get("opera_license") == "CC0"
            and headspace_report.get("database", {}).get("packaged_in_wheel") is False
            and headspace_report.get("claim_boundary", {}).get(
                "human_olfactory_90_percent_certified"
            )
            is False,
            "headspace sensory hub provenance or claim boundary changed",
        )
        with sqlite3.connect(
            headspace_paths["database"].resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            require(
                connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                and not connection.execute("PRAGMA foreign_key_check").fetchall(),
                "headspace sensory hub SQLite integrity failed",
            )
            threshold_units = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT unit FROM sensory_observations "
                    "WHERE dataset='abraham_2012'"
                )
            }
            require(
                threshold_units == {"log10_inverse_ppmv"},
                f"Abraham threshold units changed: {threshold_units}",
            )
        require(
            concentration_report.get("schema")
            == "concentration-headspace-calibration/v1"
            and concentration_report.get("source", {}).get("hub_sha256")
            == sha256(headspace_paths["database"])
            and concentration_report.get("source", {}).get("hub_report_sha256")
            == sha256(headspace_paths["report"])
            and concentration_report.get("implementation", {}).get("script_sha256")
            == sha256(headspace_paths["calibration_script"]),
            "concentration-headspace calibration binding failed",
        )
        require(
            concentration_report.get("gates", {})
            .get("molecule_holdout_transfer", {})
            .get("passed")
            is True
            and concentration_report.get("split", {}).get("molecule_overlap") == 0
            and concentration_report.get("gates", {})
            .get("production", {})
            .get("passed")
            is False
            and float(concentration_report.get("runtime_primary_score_weight", -1.0))
            == 0.0,
            "concentration-headspace gate or production boundary changed",
        )
        require(
            dream_headspace_report.get("schema") == "dream-headspace-retrospective/v3"
            and dream_headspace_report.get("source", {}).get("headspace_hub_sha256")
            == sha256(headspace_paths["database"])
            and dream_headspace_report.get("source", {}).get(
                "concentration_calibration_sha256"
            )
            == sha256(headspace_paths["calibration"])
            and dream_headspace_report.get("implementation", {}).get("script_sha256")
            == sha256(headspace_paths["dream_script"]),
            "DREAM headspace diagnostic binding failed",
        )
        require(
            dream_headspace_report.get("status")
            == "headspace_candidate_rejected_not_point_pareto"
            and dream_headspace_report.get("selection", {}).get(
                "point_pareto_candidates"
            )
            == 0
            and dream_headspace_report.get("gates", {})
            .get("production", {})
            .get("runtime_primary_score_weight")
            == 0.0
            and dream_headspace_report.get("claim_boundary", {}).get(
                "human_olfactory_90_percent_certified"
            )
            is False,
            "DREAM headspace rejection or claim boundary changed",
        )
        require(
            dream_accuracy_report.get("schema")
            == "dream-accuracy-outcome-aware-search/v4"
            and dream_accuracy_report.get("implementation", {}).get("script_sha256")
            == sha256(headspace_paths["dream_accuracy_script"])
            and dream_accuracy_report.get("implementation", {}).get("candidate_count")
            == 4960,
            "DREAM accuracy v4 report binding failed",
        )
        require(
            dream_accuracy_report.get("point_pareto_candidates") == 19
            and dream_accuracy_report.get("gates", {})
            .get("point_pareto", {})
            .get("passed")
            is True
            and dream_accuracy_report.get("gates", {})
            .get("statistical_improvement", {})
            .get("passed")
            is False
            and dream_accuracy_report.get("gates", {})
            .get("human_ceiling_90_percent", {})
            .get("passed")
            is False
            and dream_accuracy_report.get("gates", {})
            .get("production", {})
            .get("runtime_primary_score_weight")
            == 0.0
            and dream_accuracy_report.get("timing", {}).get(
                "eligible_for_model_selection_or_promotion"
            )
            is False,
            "DREAM accuracy v4 fail-closed boundary changed",
        )
        require(
            dream_set_encoder_report.get("schema")
            == "dream-attention-set-outcome-aware-search/v5"
            and dream_set_encoder_report.get("implementation", {}).get("script_sha256")
            == sha256(headspace_paths["dream_set_encoder_script"])
            and dream_set_encoder_report.get("source", {}).get("v4_report_sha256")
            == sha256(headspace_paths["dream_accuracy_report"]),
            "DREAM set encoder v5 report binding failed",
        )
        require(
            dream_set_encoder_report.get("status")
            == "attention_set_encoder_rejected_below_v4"
            and dream_set_encoder_report.get("gates", {})
            .get("point_pareto_above_v4", {})
            .get("passed")
            is False
            and dream_set_encoder_report.get("gates", {})
            .get("production", {})
            .get("runtime_primary_score_weight")
            == 0.0
            and dream_set_encoder_report.get("claim_boundary", {}).get(
                "human_olfactory_90_percent_certified"
            )
            is False,
            "DREAM set encoder v5 rejection boundary changed",
        )
        require(
            vp_report.get("schema") == "opera-vp-imputer-retrospective/v1"
            and vp_report.get("source", {}).get("hub_sha256")
            == sha256(headspace_paths["database"])
            and vp_report.get("software", {}).get("script_sha256")
            == sha256(headspace_paths["vp_script"])
            and vp_report.get("runtime", {}).get("sha256")
            == sha256(headspace_paths["vp_runtime"]),
            "OPERA vapor-pressure imputer binding failed",
        )
        require(
            vp_report.get("dataset", {}).get("strict_scaffold_test_molecules") == 93
            and vp_report.get("dataset", {}).get("strict_scaffold_overlap") == 0
            and vp_report.get("gates", {}).get("research_imputer", {}).get("passed")
            is False
            and vp_report.get("gates", {}).get("production", {}).get("passed") is False
            and vp_runtime.get("schema") == "opera-vp-portable-ridge/v1"
            and float(vp_runtime.get("runtime_primary_score_weight", -1.0)) == 0.0,
            "OPERA vapor-pressure imputer fail-closed boundary changed",
        )

    natural = json.loads(
        (DATA / "natural_material_compositions.json").read_text(encoding="utf-8")
    )
    threshold = json.loads(
        (DATA / "odor_threshold_registry.json").read_text(encoding="utf-8")
    )
    require(
        len(natural.get("materials", [])) == 4,
        "natural composition material count is not 4",
    )
    require(
        threshold.get("matched_record_count") == 12,
        "threshold exact-structure match count is not 12",
    )
    require(
        threshold.get("catalog_match_count") == 3,
        "threshold catalog match count is not 3",
    )
    require(
        threshold.get("natural_component_match_count") == 9,
        "threshold natural-component match count is not 9",
    )

    component_manifest = json.loads(
        (DATA / "r2_ingredient_components_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest = json.loads(
        (DATA / "physsim_r2_manifest.json").read_text(encoding="utf-8")
    )
    component_bytes = (DATA / component_manifest["artifact_file"]).read_bytes()
    require(
        hashlib.sha256(component_bytes).hexdigest()
        == component_manifest["artifact_sha256"],
        "R2 component artifact hash mismatch",
    )
    require(
        sha256(DATA / checkpoint_manifest["checkpoint_file"])
        == checkpoint_manifest["checkpoint_sha256"],
        "R2 checkpoint hash mismatch",
    )
    require(
        component_manifest["descriptor_contract_sha256"]
        == checkpoint_manifest["descriptor_contract_sha256"],
        "R2 descriptor contracts differ",
    )
    require(
        component_manifest["formulation_ready_ingredient_count"]
        == component_manifest["covered_ingredient_count"]
        == 34,
        "R2 formulation-ready ingredient coverage is incomplete",
    )
    release = checkpoint_manifest.get("release_gate", {})
    gate_passed = bool(release.get("passed", False))
    require(
        gate_passed == all(bool(value) for value in release.get("checks", {}).values()),
        "R2 release-gate aggregate does not match its checks",
    )
    expected_weight = 0.10 if gate_passed else 0.0
    require(
        abs(float(release.get("approved_primary_score_weight", -1.0)) - expected_weight)
        < 1e-12,
        "R2 release weight is inconsistent with the validation gate",
    )
    calibration = checkpoint_manifest.get("ensemble_calibration", {})
    require(
        calibration.get("method") == "centered_residual_on_primary_score",
        "R2 ensemble calibration is missing",
    )
    require(
        0.0 < float(calibration.get("neutral_similarity_percent", 0.0)) < 100.0,
        "R2 neutral similarity is invalid",
    )

    ensemble_manifest = json.loads(
        (DATA / "physsim_r2_ensemble_manifest.json").read_text(encoding="utf-8")
    )
    ensemble_release = ensemble_manifest.get("release_gate", {})
    require(bool(ensemble_release.get("passed")), "R2 ensemble release gate is closed")
    require(
        bool(ensemble_release.get("checks"))
        and all(bool(value) for value in ensemble_release.get("checks", {}).values()),
        "R2 ensemble release checks are incomplete",
    )
    require(
        ensemble_manifest.get("descriptor_contract_sha256")
        == checkpoint_manifest.get("descriptor_contract_sha256")
        == component_manifest.get("descriptor_contract_sha256"),
        "R2 ensemble descriptor contract differs",
    )
    member_weight = 0.0
    for member in ensemble_manifest.get("members", []):
        member_path = DATA / member["file"]
        require(
            member_path.is_file(), f"missing R2 ensemble member: {member_path.name}"
        )
        require(
            sha256(member_path) == member["sha256"],
            f"R2 ensemble member hash mismatch: {member_path.name}",
        )
        member_weight += float(member["weight"])
    require(
        len(ensemble_manifest.get("members", [])) == 2,
        "R2 ensemble must contain two members",
    )
    require(abs(member_weight - 1.0) < 1e-12, "R2 ensemble weights do not sum to one")
    uncertainty = ensemble_manifest.get("uncertainty", {})
    require(
        float(uncertainty.get("maximum_member_disagreement", 0.0)) > 0,
        "R2 OOD disagreement threshold is missing",
    )
    require(
        float(uncertainty.get("absolute_error_q95", 0.0)) > 0,
        "R2 conformal interval is missing",
    )

    runtime_manifest_path = DATA / "physsim_r2_runtime_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    runtime_path = DATA / runtime_manifest["artifact_file"]
    require(
        sha256(runtime_path) == runtime_manifest.get("artifact_sha256"),
        "portable R2 runtime artifact hash mismatch",
    )
    require(
        runtime_manifest.get("ensemble_manifest_sha256")
        == sha256(DATA / "physsim_r2_ensemble_manifest.json"),
        "portable R2 runtime is not bound to the ensemble manifest",
    )
    require(
        runtime_manifest.get("descriptor_contract_sha256")
        == ensemble_manifest.get("descriptor_contract_sha256"),
        "portable R2 descriptor contract differs",
    )
    require(
        runtime_manifest.get("runtime") == "numpy_only_r2_inference_v1",
        "portable R2 runtime contract changed",
    )
    equivalence = runtime_manifest.get("numeric_equivalence", {})
    require(bool(equivalence.get("passed")), "portable R2 numeric equivalence failed")
    require(
        int(equivalence.get("cases", 0)) >= 2
        and float(equivalence.get("maximum_absolute_error", 1.0))
        <= float(equivalence.get("tolerance", 0.0))
        <= 1e-5,
        "portable R2 numeric equivalence tolerance is invalid",
    )
    runtime_bytes = runtime_path.read_bytes()
    runtime_members = runtime_manifest.get("members", [])
    require(
        len(runtime_members) == len(ensemble_manifest.get("members", [])),
        "portable R2 member count differs",
    )
    with np.load(io.BytesIO(runtime_bytes), allow_pickle=False) as runtime_values:
        require(
            runtime_values["normalizer_mean"].shape == (217,)
            and runtime_values["normalizer_std"].shape == (217,),
            "portable R2 normalizer shape changed",
        )
        for index in range(len(runtime_members)):
            for state_key, expected_shape in EXPECTED_STATE_SHAPES.items():
                key = f"member_{index}::{state_key}"
                require(key in runtime_values, f"portable R2 state is missing: {key}")
                if key in runtime_values:
                    require(
                        runtime_values[key].shape == expected_shape,
                        f"portable R2 state shape changed: {key}",
                    )
                    require(
                        bool(np.isfinite(runtime_values[key]).all()),
                        f"portable R2 state is non-finite: {key}",
                    )

    human_calibration = json.loads(
        (DATA / "human_mixture_calibration.json").read_text(encoding="utf-8")
    )
    human_report = ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.json"
    require(human_report.is_file(), "human blind benchmark source report is missing")
    if human_report.is_file():
        require(
            human_calibration.get("source", {}).get("blind_report_sha256")
            == sha256(human_report),
            "human calibration source report hash mismatch",
        )
    require(
        human_calibration.get("endpoint")
        == "three_alternative_odd_one_out_correct_rate",
        "human calibration endpoint changed",
    )
    require(
        human_calibration.get("source", {}).get(
            "historical_final_labels_used_for_parameter_selection"
        )
        is False,
        "human final labels leaked into calibration parameter selection",
    )
    require(
        human_calibration.get("historical_final_evaluation", {}).get(
            "human_similarity_90_claim_authorized"
        )
        is False,
        "human calibration incorrectly authorizes a 90 percent claim",
    )
    human_scope = human_calibration.get("applicability_scope", {})
    require(
        human_scope.get("matrix_ids")
        == ["bushdid_2014_equal_presence_molecular_mixture"]
        and human_scope.get("product_concentration_percent_range") == [100.0, 100.0]
        and human_scope.get("components_per_mixture") == [10, 20, 30]
        and human_scope.get("study_protocol_id")
        == "bushdid_2014_supplemental_3afc_protocol"
        and human_scope.get("requires_registered_stimulus_table") is True
        and human_scope.get("requires_exact_vial_dilution_design") is True
        and human_scope.get("allowed_vial_dilutions") == [0.25, 0.5, 1.0]
        and human_scope.get("formula_projection_supported") is False,
        "human calibration applicability scope widened",
    )
    human_curve = human_calibration.get("calibration", {})
    human_x = np.asarray(human_curve.get("x_protocol_score", []), dtype=float)
    human_y = np.asarray(human_curve.get("predicted_correct_rate", []), dtype=float)
    require(
        human_x.size >= 2
        and human_x.shape == human_y.shape
        and bool(np.isfinite(human_x).all())
        and bool(np.isfinite(human_y).all())
        and bool((np.diff(human_x) >= 0).all())
        and bool((np.diff(human_y) >= -1e-12).all()),
        "human calibration curve is invalid",
    )
    require(
        human_curve.get("fold_count") == 4
        and human_curve.get("crossfit_stimuli") == 52
        and 0.0
        <= float(human_curve.get("cross_conformal_absolute_error_q95", -1.0))
        <= 1.0,
        "human cross-conformal contract is invalid",
    )

    bierling_paths = {
        "predictions": ROOT / "benchmarks" / "bierling_2025_blind_predictions_v1.json",
        "seal": ROOT / "benchmarks" / "bierling_2025_blind_prediction_seal_v1.json",
        "receipt": ROOT
        / "benchmarks"
        / "bierling_2025_blind_outcome_acquisition_v1.json",
        "report": ROOT / "benchmarks" / "bierling_2025_human_blind_benchmark_v1.json",
        "timestamp": ROOT
        / "benchmarks"
        / "bierling_2025_blind_timestamp_v1"
        / "seal.tsr",
        "parent_script": ROOT
        / "scripts"
        / "blind_bierling_human_olfaction_benchmark.py",
        "adjudicator": ROOT
        / "scripts"
        / "adjudicate_bierling_human_olfaction_parser.py",
    }
    require(
        all(path.is_file() for path in bierling_paths.values()),
        "Bierling public-human blind evidence is incomplete",
    )
    if all(path.is_file() for path in bierling_paths.values()):
        bierling_predictions = json.loads(
            bierling_paths["predictions"].read_text(encoding="utf-8")
        )
        bierling_seal = json.loads(bierling_paths["seal"].read_text(encoding="utf-8"))
        bierling_receipt = json.loads(
            bierling_paths["receipt"].read_text(encoding="utf-8")
        )
        bierling_report = json.loads(
            bierling_paths["report"].read_text(encoding="utf-8")
        )
        bierling_rows_hash = hashlib.sha256(
            json.dumps(
                bierling_predictions.get("predictions", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        require(
            bierling_seal.get("prediction_file_sha256")
            == sha256(bierling_paths["predictions"])
            and bierling_seal.get("prediction_rows_sha256") == bierling_rows_hash
            and bierling_seal.get("benchmark_script_sha256")
            == sha256(bierling_paths["parent_script"]),
            "Bierling prediction seal integrity failed",
        )
        require(
            bierling_report.get("blind_integrity", {}).get("prediction_sha256")
            == sha256(bierling_paths["predictions"])
            and bierling_report.get("blind_integrity", {}).get("seal_sha256")
            == sha256(bierling_paths["seal"])
            and bierling_report.get("blind_integrity", {}).get(
                "acquisition_receipt_sha256"
            )
            == sha256(bierling_paths["receipt"])
            and bierling_report.get("blind_integrity", {})
            .get("timestamp", {})
            .get("response_sha256")
            == sha256(bierling_paths["timestamp"])
            and bierling_report.get("blind_integrity", {}).get("target_outcome_sha256")
            == bierling_receipt.get("outcome_sha256"),
            "Bierling blind report evidence binding failed",
        )
        primary_bierling = bierling_report.get("results", {}).get(
            "primary_target_exact_label_excluded", {}
        )
        baseline_bierling = bierling_report.get("results", {}).get(
            "fixed_rdkit_baseline", {}
        )
        adjudication = bierling_report.get("parser_adjudication", {})
        require(
            len(bierling_predictions.get("predictions", [])) == 74
            and bierling_report.get("dataset", {}).get("population", {}).get("odors")
            == 73
            and bierling_report.get("dataset", {})
            .get("population", {})
            .get("participants")
            == 1119
            and bierling_report.get("improvement_gate", {}).get("passed") is False
            and bierling_report.get("measured_odor_improvement_gate", {}).get("passed")
            is True
            and adjudication.get("developed_after_target_file_opened") is True
            and adjudication.get("changes", {}).get("unscored_zero_row_target")
            == "4Isoprop"
            and adjudication.get("adjudicator_script_sha256")
            == sha256(bierling_paths["adjudicator"]),
            "Bierling parser/availability adjudication contract changed",
        )
        require(
            abs(
                float(primary_bierling.get("macro_endpoint_spearman", 0.0))
                - 0.34675334654216905
            )
            < 1e-12
            and abs(
                float(baseline_bierling.get("macro_endpoint_spearman", 0.0))
                - 0.24241133497812523
            )
            < 1e-12
            and bierling_report.get("two_way_bootstrap", {}).get(
                "primary_minus_baseline_95_interval", [0.0]
            )[0]
            > 0.0
            and bierling_report.get("human_olfactory_90_percent_certified") is False,
            "Bierling public-human result or claim boundary changed",
        )

    intensity_paths = {
        "predictions": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_blind_predictions_v1.json",
        "seal": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_blind_prediction_seal_v1.json",
        "receipt": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_blind_outcome_acquisition_v1.json",
        "report": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_blind_benchmark_v1.json",
        "calibration": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_crossfit_calibration_v1.json",
        "calibration_v2": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_calibration_v2.json",
        "timestamp": ROOT
        / "benchmarks"
        / "bierling_2025_intensity_blind_timestamp_v1"
        / "seal.tsr",
        "parent_script": ROOT
        / "scripts"
        / "blind_bierling_intensity_pilot_benchmark.py",
        "adjudicator": ROOT
        / "scripts"
        / "adjudicate_bierling_intensity_pilot_parser.py",
        "calibration_v2_script": ROOT
        / "scripts"
        / "build_bierling_intensity_calibration_v2.py",
    }
    require(
        all(path.is_file() for path in intensity_paths.values()),
        "Bierling intensity blind evidence is incomplete",
    )
    if all(path.is_file() for path in intensity_paths.values()):
        intensity_predictions = json.loads(
            intensity_paths["predictions"].read_text(encoding="utf-8")
        )
        intensity_seal = json.loads(intensity_paths["seal"].read_text(encoding="utf-8"))
        intensity_receipt = json.loads(
            intensity_paths["receipt"].read_text(encoding="utf-8")
        )
        intensity_report = json.loads(
            intensity_paths["report"].read_text(encoding="utf-8")
        )
        intensity_calibration = json.loads(
            intensity_paths["calibration"].read_text(encoding="utf-8")
        )
        intensity_calibration_v2 = json.loads(
            intensity_paths["calibration_v2"].read_text(encoding="utf-8")
        )
        intensity_rows_hash = hashlib.sha256(
            json.dumps(
                intensity_predictions.get("predictions", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        require(
            intensity_seal.get("prediction_sha256")
            == sha256(intensity_paths["predictions"])
            and intensity_seal.get("prediction_rows_sha256") == intensity_rows_hash
            and intensity_seal.get("script_sha256")
            == sha256(intensity_paths["parent_script"]),
            "Bierling intensity prediction seal integrity failed",
        )
        require(
            intensity_report.get("blind_integrity", {}).get("prediction_sha256")
            == sha256(intensity_paths["predictions"])
            and intensity_report.get("blind_integrity", {}).get("seal_sha256")
            == sha256(intensity_paths["seal"])
            and intensity_report.get("blind_integrity", {}).get("receipt_sha256")
            == sha256(intensity_paths["receipt"])
            and intensity_report.get("blind_integrity", {}).get("pilot_sha256")
            == intensity_receipt.get("pilot_sha256")
            and intensity_report.get("blind_integrity", {})
            .get("timestamp", {})
            .get("response_sha256")
            == sha256(intensity_paths["timestamp"]),
            "Bierling intensity report binding failed",
        )
        anchored_intensity = intensity_report.get("results", {}).get(
            "condition_transfer_anchored_curve", {}
        )
        ravia_intensity = intensity_report.get("results", {}).get(
            "frozen_ravia_global_curve", {}
        )
        require(
            intensity_report.get("population", {}).get("participants") == 100
            and intensity_report.get("population", {}).get("ratings") == 964
            and intensity_report.get("population", {}).get("molecules") == 73
            and intensity_report.get("population", {}).get("conditions") == 75
            and abs(
                float(anchored_intensity.get("row_spearman", 0.0)) - 0.5541519097654776
            )
            < 1e-12
            and abs(
                float(ravia_intensity.get("row_spearman", 0.0)) + 0.1184298164315976
            )
            < 1e-12
            and intensity_report.get("condition_transfer_improvement_gate", {}).get(
                "passed"
            )
            is False
            and intensity_report.get("strict_external_gate", {}).get("passed") is False
            and intensity_report.get("parser_adjudication", {}).get(
                "adjudicator_script_sha256"
            )
            == sha256(intensity_paths["adjudicator"]),
            "Bierling blind intensity result contract changed",
        )
        require(
            intensity_calibration.get("source_binding", {}).get(
                "blind_predictions_sha256"
            )
            == sha256(intensity_paths["predictions"])
            and intensity_calibration.get("source_binding", {}).get(
                "blind_report_sha256"
            )
            == sha256(intensity_paths["report"])
            and intensity_calibration.get("release_gate", {}).get("passed") is False
            and abs(
                float(
                    intensity_calibration.get("nested_crossfit", {}).get(
                        "spearman", 0.0
                    )
                )
                - 0.5360419983047057
            )
            < 1e-12
            and abs(
                float(intensity_calibration.get("nested_crossfit", {}).get("mae", 0.0))
                - 10.685287854006807
            )
            < 1e-12
            and intensity_calibration.get("portable_diagnostic_calibrator", {}).get(
                "runtime_primary_score_weight"
            )
            == 0.0
            and intensity_calibration.get("bootstrap", {}).get(
                "calibrated_minus_ravia_spearman_95_interval", [0.0]
            )[0]
            > 0.0
            and intensity_calibration.get("bootstrap", {}).get(
                "ravia_minus_calibrated_mae_95_interval", [0.0]
            )[0]
            < 0.0
            and intensity_calibration.get("human_olfactory_90_percent_certified")
            is False,
            "Bierling retrospective intensity calibration contract changed",
        )
        require(
            intensity_calibration_v2.get("source_binding", {}).get(
                "blind_predictions_sha256"
            )
            == sha256(intensity_paths["predictions"])
            and intensity_calibration_v2.get("source_binding", {}).get(
                "v1_calibration_sha256"
            )
            == sha256(intensity_paths["calibration"])
            and intensity_calibration_v2.get("implementation", {}).get("script_sha256")
            == sha256(intensity_paths["calibration_v2_script"])
            and intensity_calibration_v2.get("large_improvement_gate", {}).get("passed")
            is True
            and abs(
                float(
                    intensity_calibration_v2.get("repeated_nested_crossfit", {}).get(
                        "mae", 0.0
                    )
                )
                - 8.937125435542422
            )
            < 1e-12
            and abs(
                float(
                    intensity_calibration_v2.get("repeated_nested_crossfit", {}).get(
                        "spearman", 0.0
                    )
                )
                - 0.612721042117401
            )
            < 1e-12
            and intensity_calibration_v2.get("bootstrap", {}).get(
                "ravia_minus_calibrated_mae_95_interval", [0.0]
            )[0]
            > 0.0
            and intensity_calibration_v2.get("final_model", {}).get(
                "runtime_primary_score_weight"
            )
            == 0.0
            and intensity_calibration_v2.get("human_olfactory_90_percent_certified")
            is False,
            "Bierling intensity calibration v2 contract changed",
        )

    ma_paths = {
        "predictions": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_predictions_v1.json",
        "seal": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_prediction_seal_v1.json",
        "receipt": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_outcome_acquisition_v1.json",
        "report": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_benchmark_v1.json",
        "calibration_v2": ROOT / "benchmarks" / "ma_2021_mixture_calibration_v2.json",
        "adjudication": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_adjudication_v1.json",
        "timestamp": ROOT
        / "benchmarks"
        / "ma_2021_binary_mixture_blind_timestamp_v1"
        / "seal.tsr",
        "parent_script": ROOT / "scripts" / "blind_ma_2021_binary_mixture_benchmark.py",
        "calibration_v2_script": ROOT
        / "scripts"
        / "build_ma_2021_mixture_calibration_v2.py",
        "adjudicator": ROOT / "scripts" / "adjudicate_ma_2021_blind_scope.py",
    }
    require(
        all(path.is_file() for path in ma_paths.values()),
        "Ma 2021 binary-mixture blind evidence is incomplete",
    )
    if all(path.is_file() for path in ma_paths.values()):
        ma_predictions = json.loads(ma_paths["predictions"].read_text(encoding="utf-8"))
        ma_seal = json.loads(ma_paths["seal"].read_text(encoding="utf-8"))
        ma_receipt = json.loads(ma_paths["receipt"].read_text(encoding="utf-8"))
        ma_report = json.loads(ma_paths["report"].read_text(encoding="utf-8"))
        ma_calibration_v2 = json.loads(
            ma_paths["calibration_v2"].read_text(encoding="utf-8")
        )
        ma_adjudication = json.loads(
            ma_paths["adjudication"].read_text(encoding="utf-8")
        )
        ma_rows_hash = hashlib.sha256(
            json.dumps(
                ma_predictions.get("predictions", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        require(
            ma_seal.get("prediction_file_sha256") == sha256(ma_paths["predictions"])
            and ma_seal.get("prediction_rows_sha256") == ma_rows_hash
            and ma_seal.get("benchmark_script_sha256")
            == sha256(ma_paths["parent_script"])
            and ma_predictions.get("blind_contract", {}).get("prediction_rows_sha256")
            == ma_rows_hash
            and len(ma_predictions.get("target_odorants", [])) == 72
            and len(ma_predictions.get("predictions", [])) == 2556,
            "Ma 2021 all-pair prediction seal integrity failed",
        )
        require(
            ma_report.get("source_binding", {}).get("prediction_sha256")
            == sha256(ma_paths["predictions"])
            and ma_report.get("source_binding", {}).get("seal_sha256")
            == sha256(ma_paths["seal"])
            and ma_report.get("source_binding", {}).get("receipt_sha256")
            == sha256(ma_paths["receipt"])
            and ma_report.get("source_binding", {}).get("outcome_sha256")
            == ma_receipt.get("outcome", {}).get("sha256")
            and ma_report.get("blind_integrity", {})
            .get("timestamp", {})
            .get("response_sha256")
            == sha256(ma_paths["timestamp"])
            and ma_report.get("blind_integrity", {}).get(
                "all_2556_pair_predictions_preceded_outcome"
            )
            is True,
            "Ma 2021 blind report binding failed",
        )
        ma_primary = ma_report.get("distinct_mixture_results", {}).get(
            "interaction::ravia_weber_fechner_pool", {}
        )
        ma_baseline = ma_report.get("distinct_mixture_results", {}).get(
            "interaction::strongest_component", {}
        )
        require(
            ma_report.get("population", {}).get("participants") == 59
            and ma_report.get("population", {}).get("individual_rows") == 6531
            and ma_report.get("population", {}).get("trial_rows") == 222
            and ma_report.get("population", {}).get("distinct_mixtures") == 198
            and abs(float(ma_primary.get("mae", 0.0)) - 0.269321128109436) < 1e-12
            and abs(float(ma_baseline.get("mae", 0.0)) - 0.2652004803538995) < 1e-12
            and ma_report.get("mixture_operator_integration_gate", {}).get("passed")
            is False
            and ma_report.get("human_olfactory_90_percent_certified") is False
            and ma_report.get("complex_perfume_recipe_validated") is False,
            "Ma 2021 blind binary-mixture result contract changed",
        )
        require(
            ma_adjudication.get("source_binding", {})
            .get("predictions", {})
            .get("sha256")
            == sha256(ma_paths["predictions"])
            and ma_adjudication.get("source_binding", {})
            .get("blind_report", {})
            .get("sha256")
            == sha256(ma_paths["report"])
            and ma_adjudication.get("implementation", {}).get("script_sha256")
            == sha256(ma_paths["adjudicator"])
            and ma_adjudication.get("adjudication", {}).get(
                "row_level_outcome_workbook_opened_before_seal"
            )
            is False
            and ma_adjudication.get("adjudication", {}).get(
                "fully_outcome_naive_prospective_blind"
            )
            is False
            and ma_adjudication.get("adjudication", {}).get("metric_values_changed")
            is False
            and ma_adjudication.get("adjudication", {}).get("gate_values_changed")
            is False
            and ma_adjudication.get("authoritative_gate_status", {}).get(
                "runtime_integration_authorized"
            )
            is False,
            "Ma 2021 publication-summary-aware scope adjudication changed",
        )
        ma_nested = ma_calibration_v2.get("repeated_nested_pair_disjoint", {})
        ma_cold = ma_calibration_v2.get("all_components_cold", {})
        require(
            ma_calibration_v2.get("source_binding", {}).get("blind_predictions_sha256")
            == sha256(ma_paths["predictions"])
            and ma_calibration_v2.get("source_binding", {}).get("blind_report_sha256")
            == sha256(ma_paths["report"])
            and ma_calibration_v2.get("implementation", {}).get("script_sha256")
            == sha256(ma_paths["calibration_v2_script"])
            and abs(float(ma_nested.get("mae", 0.0)) - 0.21766237544772413) < 1e-12
            and abs(float(ma_nested.get("spearman", 0.0)) - 0.7980197820838711) < 1e-12
            and ma_calibration_v2.get("retrospective_improvement_gate", {}).get(
                "passed"
            )
            is False
            and ma_calibration_v2.get("seen_component_new_pair_gate", {}).get("passed")
            is True
            and ma_calibration_v2.get("strict_all_components_cold_gate", {}).get(
                "passed"
            )
            is False
            and ma_calibration_v2.get("bootstrap", {}).get(
                "strongest_minus_calibrated_mae_95_interval", [0.0]
            )[0]
            > 0.0
            and abs(float(ma_cold.get("model", {}).get("mae", 0.0)) - 0.274029683926782)
            < 1e-12
            and abs(
                float(ma_cold.get("strongest_component", {}).get("mae", 0.0))
                - 0.26422415463705
            )
            < 1e-12
            and ma_cold.get("selection_nested_within_component_cold_training") is True
            and all(
                row.get("component_leakage_count") == 0
                for row in ma_cold.get("folds", [])
            )
            and all(
                row.get("selection_used_held_out_outcomes") is False
                for row in ma_cold.get("folds", [])
            )
            and ma_calibration_v2.get("final_model", {})
            .get("parameters", {})
            .get("portable_parity_maximum_absolute_error")
            == 0.0
            and ma_calibration_v2.get("final_model", {}).get(
                "runtime_primary_score_weight"
            )
            == 0.0
            and ma_calibration_v2.get("external_reproduction_complete") is False
            and ma_calibration_v2.get("human_olfactory_90_percent_certified") is False,
            "Ma 2021 mixture calibration v2 contract changed",
        )

    universal_paths = {
        "v1": ROOT / "benchmarks" / "universal_intensity_model_v1.json",
        "v2": ROOT / "benchmarks" / "universal_intensity_hybrid_v2.json",
        "physchem": ROOT / "benchmarks" / "universal_intensity_physchem_v1.json",
        "thresholds": ROOT / "benchmarks" / "universal_odor_thresholds_v2.json",
        "v3": ROOT / "benchmarks" / "universal_intensity_transport_v3.json",
        "v4": ROOT / "benchmarks" / "universal_intensity_threshold_v4.json",
        "v1_script": ROOT / "scripts" / "build_universal_intensity_model_v1.py",
        "v2_script": ROOT / "scripts" / "build_universal_intensity_hybrid_v2.py",
        "physchem_script": ROOT
        / "scripts"
        / "acquire_universal_intensity_physchem_v1.py",
        "threshold_script": ROOT
        / "scripts"
        / "acquire_universal_odor_thresholds_v2.py",
        "v3_script": ROOT / "scripts" / "build_universal_intensity_transport_v3.py",
        "v4_script": ROOT / "scripts" / "build_universal_intensity_threshold_v4.py",
    }
    require(
        all(path.is_file() for path in universal_paths.values()),
        "universal intensity evidence is incomplete",
    )
    if all(path.is_file() for path in universal_paths.values()):
        universal_v1 = json.loads(universal_paths["v1"].read_text(encoding="utf-8"))
        universal_v2 = json.loads(universal_paths["v2"].read_text(encoding="utf-8"))
        universal_physchem = json.loads(
            universal_paths["physchem"].read_text(encoding="utf-8")
        )
        universal_thresholds = json.loads(
            universal_paths["thresholds"].read_text(encoding="utf-8")
        )
        universal_v3 = json.loads(universal_paths["v3"].read_text(encoding="utf-8"))
        universal_v4 = json.loads(universal_paths["v4"].read_text(encoding="utf-8"))
        require(
            universal_v1.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["v1_script"])
            and universal_v1.get("selection", {}).get("selected_candidate")
            == "ridge_physical_1000"
            and universal_v1.get("retrospective_external_gate", {}).get("passed")
            is False
            and universal_v1.get("final_model", {}).get("runtime_primary_score_weight")
            == 0.0
            and universal_v1.get("human_olfactory_90_percent_certified") is False,
            "universal intensity v1 contract changed",
        )
        universal_v2_mono = universal_v2.get("ma_retrospective_evaluation", {}).get(
            "monomolecular", {}
        )
        universal_v2_mix = universal_v2.get("ma_retrospective_evaluation", {}).get(
            "binary_mixture", {}
        )
        require(
            universal_v2.get("source_binding", {}).get("v1_report_sha256")
            == sha256(universal_paths["v1"])
            and universal_v2.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["v2_script"])
            and universal_v2.get("retrospective_repair_gate", {}).get("passed") is True
            and universal_v2.get("prospective_external_gate", {}).get("passed") is False
            and abs(
                float(universal_v2_mono.get("hybrid_equal", {}).get("mae", 0.0))
                - 0.06904540502385427
            )
            < 1e-12
            and abs(
                float(universal_v2_mix.get("hybrid_equal::fechner", {}).get("mae", 0.0))
                - 0.4759061654447821
            )
            < 1e-12
            and universal_v2.get("ma_retrospective_evaluation", {})
            .get("component_bootstrap", {})
            .get("baseline_minus_primary_mae_95_interval", [0.0])[0]
            > 0.0
            and universal_v2.get("ma_retrospective_evaluation", {})
            .get("mixture_bootstrap", {})
            .get("baseline_minus_primary_mae_95_interval", [0.0])[0]
            > 0.0
            and universal_v2.get("runtime", {}).get("primary_score_weight") == 0.0,
            "universal intensity hybrid v2 contract changed",
        )
        require(
            universal_physchem.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["physchem_script"])
            and universal_physchem.get("records_sha256")
            == hashlib.sha256(
                json.dumps(
                    universal_physchem.get("records", []),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            and universal_physchem.get("coverage", {}).get("structures") == 575
            and universal_physchem.get("coverage", {}).get("vapor_pressure") == 188
            and universal_physchem.get("coverage", {}).get("boiling_point") == 513,
            "universal intensity physicochemical registry changed",
        )
        require(
            universal_thresholds.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["threshold_script"])
            and universal_thresholds.get("records_sha256")
            == hashlib.sha256(
                json.dumps(
                    universal_thresholds.get("records", []),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            and universal_thresholds.get("coverage", {}).get("structures") == 556
            and universal_thresholds.get("coverage", {}).get("thresholds") == 97,
            "universal odor-threshold registry changed",
        )
        require(
            universal_v3.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["v3_script"])
            and universal_v3.get("source_binding", {}).get("physchem", {}).get("sha256")
            == sha256(universal_paths["physchem"])
            and universal_v3.get("retrospective_transport_gate", {}).get("passed")
            is False
            and universal_v3.get("runtime", {}).get("primary_score_weight") == 0.0
            and universal_v3.get("prospective_external_gate", {}).get("passed")
            is False,
            "universal transport v3 rejection changed",
        )
        require(
            universal_v4.get("implementation", {}).get("script_sha256")
            == sha256(universal_paths["v4_script"])
            and universal_v4.get("source_binding", {})
            .get("thresholds", {})
            .get("sha256")
            == sha256(universal_paths["thresholds"])
            and universal_v4.get("retrospective_threshold_gate", {}).get("passed")
            is False
            and universal_v4.get("runtime", {}).get("primary_score_weight") == 0.0
            and universal_v4.get("prospective_external_gate", {}).get("passed")
            is False,
            "universal threshold v4 rejection changed",
        )

    minnesota_paths = {
        "predictions": ROOT
        / "benchmarks"
        / "minnesota_intensity_blind_predictions_v1.json",
        "seal": ROOT
        / "benchmarks"
        / "minnesota_intensity_blind_prediction_seal_v1.json",
        "receipt": ROOT
        / "benchmarks"
        / "minnesota_intensity_blind_outcome_acquisition_v1.json",
        "report": ROOT / "benchmarks" / "minnesota_intensity_blind_benchmark_v1.json",
        "scoring_adjudication": ROOT
        / "benchmarks"
        / "minnesota_intensity_blind_scoring_adjudication_v1.json",
        "timestamp": ROOT
        / "benchmarks"
        / "minnesota_intensity_blind_timestamp_v1"
        / "seal.tsr",
        "parent_script": ROOT
        / "scripts"
        / "blind_minnesota_intensity_matching_benchmark.py",
        "adjudicator": ROOT
        / "scripts"
        / "adjudicate_minnesota_intensity_acquisition.py",
        "scoring_adjudicator": ROOT
        / "scripts"
        / "adjudicate_minnesota_intensity_scoring.py",
    }
    require(
        all(path.is_file() for path in minnesota_paths.values()),
        "Minnesota blind intensity evidence is incomplete",
    )
    if all(path.is_file() for path in minnesota_paths.values()):
        mn_predictions = json.loads(
            minnesota_paths["predictions"].read_text(encoding="utf-8")
        )
        mn_seal = json.loads(minnesota_paths["seal"].read_text(encoding="utf-8"))
        mn_receipt = json.loads(minnesota_paths["receipt"].read_text(encoding="utf-8"))
        mn_report = json.loads(minnesota_paths["report"].read_text(encoding="utf-8"))
        mn_scoring = json.loads(
            minnesota_paths["scoring_adjudication"].read_text(encoding="utf-8")
        )
        mn_rows_hash = hashlib.sha256(
            json.dumps(
                mn_predictions.get("predictions", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        require(
            mn_seal.get("prediction_file_sha256")
            == sha256(minnesota_paths["predictions"])
            and mn_seal.get("prediction_rows_sha256") == mn_rows_hash
            and mn_seal.get("benchmark_script_sha256")
            == sha256(minnesota_paths["parent_script"])
            and mn_predictions.get("blind_contract", {}).get("prediction_rows_sha256")
            == mn_rows_hash
            and mn_predictions.get("release_gate", {}).get("passed") is True,
            "Minnesota blind prediction seal integrity failed",
        )
        adjudication = mn_receipt.get("acquisition_adjudication", {})
        require(
            mn_receipt.get("prediction_sha256")
            == sha256(minnesota_paths["predictions"])
            and mn_receipt.get("seal_sha256") == sha256(minnesota_paths["seal"])
            and mn_receipt.get("timestamp", {}).get("response_sha256")
            == sha256(minnesota_paths["timestamp"])
            and adjudication.get("parent_script_sha256")
            == sha256(minnesota_paths["parent_script"])
            and adjudication.get("adjudicator_script_sha256")
            == sha256(minnesota_paths["adjudicator"])
            and adjudication.get("readme_expected_rows", {}).get("retest") == 360
            and adjudication.get("repository_actual_rows", {}).get("retest") == 250
            and adjudication.get("prediction_rows_changed") is False
            and adjudication.get("outcome_rows_changed") is False
            and adjudication.get("scoring_contract_changed") is False,
            "Minnesota acquisition adjudication changed",
        )
        require(
            mn_report.get("blind_integrity", {}).get("prediction_sha256")
            == sha256(minnesota_paths["predictions"])
            and mn_report.get("blind_integrity", {}).get("receipt_sha256")
            == sha256(minnesota_paths["receipt"])
            and mn_report.get("blind_integrity", {})
            .get("timestamp", {})
            .get("response_sha256")
            == sha256(minnesota_paths["timestamp"])
            and mn_report.get("dataset", {}).get("parser", {}).get("raw_rows") == 1630
            and abs(
                float(
                    mn_report.get("results", {})
                    .get("concentration_preserving_equal_hybrid", {})
                    .get("log10_mae", 0.0)
                )
                - 1.747698487880883
            )
            < 1e-12
            and abs(
                float(
                    mn_report.get("results", {})
                    .get("constant_5ppm_baseline", {})
                    .get("log10_mae", 0.0)
                )
                - 1.0552864639302861
            )
            < 1e-12
            and mn_report.get("external_concentration_gate", {}).get("passed") is False
            and mn_report.get("mixture_intensity_external_gate", {}).get("passed")
            is False
            and mn_report.get("runtime_primary_score_weight") == 0.0
            and mn_report.get("human_olfactory_90_percent_certified") is False,
            "Minnesota blind intensity result contract changed",
        )
        corrected_primary = mn_scoring.get("corrected_results", {}).get(
            "concentration_preserving_equal_hybrid", {}
        )
        corrected_baseline = mn_scoring.get("corrected_results", {}).get(
            "constant_5ppm_baseline", {}
        )
        require(
            mn_scoring.get("source_binding", {}).get("predictions_sha256")
            == sha256(minnesota_paths["predictions"])
            and mn_scoring.get("source_binding", {}).get("parent_report_sha256")
            == sha256(minnesota_paths["report"])
            and mn_scoring.get("implementation", {}).get("script_sha256")
            == sha256(minnesota_paths["scoring_adjudicator"])
            and mn_scoring.get("adjudication", {}).get("prediction_values_changed")
            is False
            and mn_scoring.get("adjudication", {}).get("outcome_values_changed")
            is False
            and mn_scoring.get("adjudication", {}).get(
                "metric_weighting_changed_to_match_seal"
            )
            is True
            and abs(float(corrected_primary.get("log10_mae", 0.0)) - 1.6686539145386021)
            < 1e-12
            and abs(
                float(corrected_baseline.get("log10_mae", 0.0)) - 1.0707435533869036
            )
            < 1e-12
            and mn_scoring.get("corrected_external_gate", {}).get("passed") is False
            and mn_scoring.get("runtime_primary_score_weight") == 0.0
            and mn_scoring.get("human_olfactory_90_percent_certified") is False,
            "Minnesota compound-balanced scoring adjudication changed",
        )

    dream_paths = {
        "report": ROOT / "benchmarks" / "dream_mixture_2025_retrospective_v1.json",
        "runtime": ROOT / "benchmarks" / "dream_mixture_2025_research_runtime_v1.json",
        "script": ROOT / "scripts" / "benchmark_dream_mixture_2025.py",
    }
    require(
        all(path.is_file() for path in dream_paths.values()),
        "DREAM mixture retrospective evidence is incomplete",
    )
    dream_training_pairs = 0
    dream_test_pairs = 0
    dream_validation_pairs = 0
    if all(path.is_file() for path in dream_paths.values()):
        dream_report = json.loads(dream_paths["report"].read_text(encoding="utf-8"))
        dream_runtime = json.loads(dream_paths["runtime"].read_text(encoding="utf-8"))
        dream_source = dream_report.get("source", {})
        dream_pommix = dream_source.get("pommix", {})
        dream_dataset = dream_report.get("dataset", {})
        dream_test = dream_report.get("test", {})
        dream_validation = dream_report.get("validation", {})
        dream_release = dream_report.get("release_gate", {})
        dream_timing = dream_report.get("timing", {})
        dream_implementation = dream_report.get("implementation", {})
        dream_training_pairs = int(dream_dataset.get("training_pairs", 0))
        dream_test_pairs = int(dream_dataset.get("test_pairs", 0))
        dream_validation_pairs = int(dream_dataset.get("validation_pairs", 0))
        source_files = dream_source.get("files", {})
        require(
            dream_report.get("schema") == "dream-mixture-retrospective-v1"
            and dream_report.get("implementation", {}).get("script_sha256")
            == sha256(dream_paths["script"])
            and dream_report.get("runtime", {}).get("sha256")
            == sha256(dream_paths["runtime"])
            and dream_source.get("git_commit")
            == "d4294949fdc55d6bab145e8d100d58c87daf1bc6"
            and dream_source.get("root_license_file_present") is False
            and dream_source.get("license_status") == "not_declared_in_repository_root"
            and dream_pommix.get("git_commit")
            == "2558557ac4793ce982d0bd26cfa79cdf4dfbbb6f"
            and dream_pommix.get("license") == "MIT"
            and dream_pommix.get("license_file_sha256")
            == "e7e0f16526da0b53905dd277a585d3e407d7ce93125e38429785da04593fbebe"
            and dream_pommix.get("weights_sha256")
            == "31b8d75daae6a4a36876a14373f43803356af05a8b111751d95ac664026d83fc"
            and dream_pommix.get("embedding_rows_sha256")
            == "2f74ea513256da7772cb6cde51eb6af407ff532e83350edc6a78a7435f650e7b"
            and dream_pommix.get("descriptastorus_commit")
            == "9b133e2c91bb6a67df53db4cba992776db219ab7"
            and dream_pommix.get("descriptastorus_version") == "2.5.0.25"
            and str(dream_pommix.get("torch_version", "")).split("+", 1)[0] == "2.8.0"
            and dream_pommix.get("torch_geometric_version") == "2.7.0"
            and dream_pommix.get("device") == "cpu"
            and dream_pommix.get("embedding_dimensions") == 196
            and dream_pommix.get("molecules") == 235
            and isinstance(source_files, dict)
            and len(source_files) == 11
            and all(
                isinstance(value, dict)
                and len(str(value.get("sha256", ""))) == 64
                and int(value.get("bytes", 0)) > 0
                for value in source_files.values()
            )
            and dream_training_pairs == 730
            and dream_dataset.get("upstream_readme_claimed_training_pairs") == 507
            and dream_dataset.get("readme_training_pair_count_matches_repository")
            is False
            and dream_test_pairs == 46
            and dream_validation_pairs == 50,
            "DREAM mixture source or dataset contract changed",
        )
        require(
            dream_timing.get("development_used_test_or_validation_labels") is True
            and dream_timing.get(
                "formal_candidate_ranking_used_test_or_validation_labels"
            )
            is False
            and dream_report.get("selection", {}).get("development_outcome_aware")
            is True
            and dream_report.get("selection", {}).get("test_labels_used_for_selection")
            is False
            and dream_report.get("selection", {}).get(
                "validation_labels_used_for_selection"
            )
            is False
            and int(dream_implementation.get("portable_runtime_rows_checked", 0)) == 96
            and dream_implementation.get("numpy_version") == "2.2.6"
            and dream_implementation.get("pandas_version") == "2.3.2"
            and dream_implementation.get("scipy_version") == "1.15.2"
            and dream_implementation.get("sklearn_version") == "1.7.1"
            and float(
                dream_implementation.get(
                    "portable_runtime_equivalence_max_abs_error", 1.0
                )
            )
            <= 1e-12,
            "DREAM mixture outcome-awareness or runtime-equivalence boundary changed",
        )
        require(
            abs(
                float(dream_test.get("candidate", {}).get("pearson", 0.0))
                - 0.40029015265804724
            )
            < 1e-12
            and abs(
                float(dream_test.get("current_frozen_r2", {}).get("pearson", 0.0))
                - 0.03937651497652005
            )
            < 1e-12
            and abs(
                float(dream_test.get("candidate", {}).get("rmse", 0.0))
                - 0.10056515609337634
            )
            < 1e-12
            and dream_test.get("gate_passed") is False
            and abs(
                float(dream_validation.get("candidate", {}).get("pearson", 0.0))
                - 0.6362301219746657
            )
            < 1e-12
            and abs(
                float(
                    dream_validation.get("fixed_public_top6_ensemble", {}).get(
                        "pearson", 0.0
                    )
                )
                - 0.473600505159714
            )
            < 1e-12
            and abs(
                float(
                    dream_validation.get(
                        "candidate_human_ceiling_normalized_pearson", 0.0
                    )
                )
                - 0.6894414013918638
            )
            < 1e-12
            and dream_validation.get("gate_passed") is False
            and abs(
                float(
                    dream_validation.get("two_way_subject_pair_bootstrap", {}).get(
                        "candidate_minus_baseline_pearson_95_interval", [0.0]
                    )[0]
                )
                - (-0.017568446895495252)
            )
            < 1e-12
            and abs(
                float(
                    dream_validation.get("two_way_subject_pair_bootstrap", {}).get(
                        "baseline_minus_candidate_rmse_95_interval", [0.0]
                    )[0]
                )
                - 0.0047688702924176
            )
            < 1e-12
            and dream_validation.get("two_way_subject_pair_bootstrap", {}).get(
                "valid_human_ceiling_draws"
            )
            == 19999
            and abs(
                float(
                    dream_validation.get(
                        "candidate_human_ceiling_normalized_pearson_95_interval",
                        [0.0],
                    )[1]
                )
                - 0.8981443076097063
            )
            < 1e-12
            and dream_validation.get("human_ceiling_90_percent_gate", {}).get("passed")
            is False
            and dream_report.get("internal_large_improvement_gate", {}).get("passed")
            is False
            and dream_release.get("passed") is False
            and dream_release.get("runtime_primary_score_weight") == 0.0,
            "DREAM mixture result or release boundary changed",
        )
        require(
            dream_runtime.get("schema") == "dream-mixture-portable-ridge-research/v1"
            and dream_runtime.get("allow_pickle") is False
            and dream_runtime.get("external_openpom_profile_registry_required") is True
            and dream_runtime.get("external_pommix_embedding_model_required") is True
            and dream_runtime.get("runtime_primary_score_weight") == 0.0
            and dream_runtime.get("human_olfactory_90_percent_certified") is False,
            "DREAM mixture research runtime contract changed",
        )

    pair_paths = {
        "report": ROOT / "benchmarks" / "dream_pair_ensemble_retrospective_v2.json",
        "runtime": ROOT / "benchmarks" / "dream_pair_ensemble_research_runtime_v2.json",
        "script": ROOT / "scripts" / "benchmark_dream_pair_ensemble_v2.py",
    }
    require(
        all(path.is_file() for path in pair_paths.values()),
        "DREAM odor-pair ensemble evidence is incomplete",
    )
    dream_pair_training_pairs = 0
    if all(path.is_file() for path in pair_paths.values()):
        pair_report = json.loads(pair_paths["report"].read_text(encoding="utf-8"))
        pair_runtime = json.loads(pair_paths["runtime"].read_text(encoding="utf-8"))
        pair_source = pair_report.get("source", {}).get("odor_pair", {})
        pair_dataset = pair_report.get("dataset", {})
        pair_test = pair_report.get("test", {})
        pair_validation = pair_report.get("validation", {})
        pair_gates = pair_report.get("gates", {})
        pair_implementation = pair_report.get("implementation", {})
        dream_pair_training_pairs = int(pair_dataset.get("training_pairs", 0))
        require(
            pair_report.get("schema") == "dream-pair-ensemble-retrospective/v2"
            and pair_implementation.get("script_sha256") == sha256(pair_paths["script"])
            and pair_report.get("runtime", {}).get("sha256")
            == sha256(pair_paths["runtime"])
            and pair_source.get("git_commit")
            == "32c25530535aa8354107ee6f587afd691ba6c1f0"
            and pair_source.get("license") == "MIT"
            and pair_source.get("license_holder_fields_complete") is False
            and pair_source.get("redistribution_relied_on") is False
            and pair_source.get("license_sha256")
            == "19200a6a9407e592065a5a504c0eefe58adf102c9ac5aa2151bd6f257faa7a9c"
            and pair_source.get("source_tree_sha256")
            == "fb23e987a2fc70d755735fad75466215eb366879ef39d49d351050f5e2152a3f"
            and pair_source.get("weights_sha256")
            == "50a2b0e2bb54d7129d5dcad0cff71d2fb04c6b1d82a56e877e00ba9cc7c43389"
            and pair_source.get("generated_embedding_rows_sha256")
            == "39aee0961908ae761cb1e4ac2dec7b06c1bcb7befedbb0204789d6b954d97ecc"
            and pair_source.get("ogb_version") == "1.3.6"
            and pair_source.get("embedding_dimensions") == 128
            and float(pair_source.get("precomputed_reproduction_max_abs_error", 1.0))
            <= 1.5e-5
            and pair_source.get("generated_training_mixtures") == 730
            and pair_source.get("generated_test_mixtures") == 92
            and pair_source.get("generated_validation_mixtures") == 76
            and dream_pair_training_pairs == 730
            and pair_dataset.get("test_pairs") == 46
            and pair_dataset.get("validation_pairs") == 50,
            "DREAM odor-pair source, data, or implementation contract changed",
        )
        require(
            pair_report.get("timing", {}).get(
                "development_used_test_and_validation_outcomes"
            )
            is True
            and pair_report.get("timing", {}).get(
                "candidate_weights_selected_after_outcomes"
            )
            is True
            and pair_report.get("timing", {}).get("prospective_or_outcome_unopened")
            is False
            and pair_report.get("timing", {}).get(
                "post_selection_intervals_descriptive_only"
            )
            is True
            and pair_report.get("timing", {}).get("inferentially_valid_for_promotion")
            is False
            and pair_report.get("selection", {}).get("eligible_for_promotion") is False
            and pair_report.get("selection", {}).get("test_labels_used_for_selection")
            is True
            and pair_report.get("selection", {}).get(
                "validation_labels_used_for_selection"
            )
            is True
            and float(pair_report.get("selection", {}).get("weight_grid_step", 0.0))
            == 0.05
            and len(pair_report.get("selection", {}).get("weight_search", [])) == 21,
            "DREAM odor-pair outcome-awareness boundary changed",
        )
        require(
            abs(
                float(pair_test.get("candidate", {}).get("pearson", 0.0))
                - 0.43038429417194257
            )
            < 1e-12
            and abs(
                float(pair_test.get("candidate", {}).get("rmse", 0.0))
                - 0.09966599045522863
            )
            < 1e-12
            and abs(
                float(pair_validation.get("candidate", {}).get("pearson", 0.0))
                - 0.6397441933198227
            )
            < 1e-12
            and abs(
                float(pair_validation.get("candidate", {}).get("rmse", 0.0))
                - 0.11192423618426466
            )
            < 1e-12
            and abs(
                float(
                    pair_validation.get(
                        "candidate_human_ceiling_normalized_pearson", 0.0
                    )
                )
                - 0.693249373050415
            )
            < 1e-12
            and abs(
                float(
                    pair_validation.get(
                        "candidate_human_ceiling_normalized_pearson_95_interval",
                        [0.0],
                    )[0]
                )
                - 0.4018898577566281
            )
            < 1e-12
            and abs(
                float(
                    pair_validation.get(
                        "candidate_human_ceiling_normalized_pearson_95_interval",
                        [0.0, 0.0],
                    )[1]
                )
                - 0.8971098943596322
            )
            < 1e-12
            and pair_gates.get("point_pareto", {}).get("passed") is True
            and pair_gates.get("statistical_improvement", {}).get("passed") is False
            and pair_gates.get("human_ceiling_90_percent", {}).get("passed") is False
            and pair_gates.get("production", {}).get("passed") is False
            and pair_gates.get("production", {}).get("runtime_primary_score_weight")
            == 0.0,
            "DREAM odor-pair result or release boundary changed",
        )
        members = pair_runtime.get("members", [])
        require(
            pair_runtime.get("schema") == "dream-pair-ensemble-portable-ridge/v2"
            and pair_runtime.get("allow_pickle") is False
            and pair_runtime.get("external_pommix_embedding_model_required") is True
            and pair_runtime.get("external_odor_pair_embedding_model_required") is True
            and pair_runtime.get("external_model_load_policy")
            == "torch_weights_only_hash_pinned"
            and pair_runtime.get("runtime_primary_score_weight") == 0.0
            and pair_runtime.get("human_olfactory_90_percent_certified") is False
            and len(members) == 2
            and [float(member.get("alpha", 0.0)) for member in members]
            == [30000.0, 100000.0]
            and [float(member.get("weight", 0.0)) for member in members] == [0.6, 0.4]
            and int(pair_implementation.get("portable_runtime_rows_checked", 0)) == 96
            and float(
                pair_implementation.get(
                    "portable_runtime_equivalence_max_abs_error", 1.0
                )
            )
            <= 1e-12,
            "DREAM odor-pair portable runtime contract changed",
        )

    concentration_manifest = json.loads(
        (DATA / "concentration_response_manifest.json").read_text(encoding="utf-8")
    )
    concentration_path = DATA / concentration_manifest["runtime_file"]
    require(
        sha256(concentration_path) == concentration_manifest["runtime_sha256"],
        "concentration response runtime hash mismatch",
    )
    concentration_release = concentration_manifest.get("release_gate", {})
    require(
        bool(concentration_release.get("passed")),
        "concentration response release gate is closed",
    )
    require(
        bool(concentration_release.get("checks"))
        and all(
            bool(value) for value in concentration_release.get("checks", {}).values()
        ),
        "concentration response release checks are incomplete",
    )
    require(
        concentration_manifest.get("algorithm") == "concentration_only_ridge",
        "unapproved concentration algorithm",
    )
    require(
        float(concentration_manifest.get("structure_specific_weight", -1.0)) == 0.0,
        "unvalidated structure-specific concentration weight is nonzero",
    )
    portable_concentration = json.loads(concentration_path.read_text(encoding="utf-8"))
    require(
        portable_concentration.get("format")
        == "standard_scaler_plus_ridge_coefficients_v1",
        "portable concentration-response parameters are missing",
    )
    require(
        portable_concentration.get("allow_pickle") is False
        and portable_concentration.get("source_training_artifact_required_at_runtime")
        is False
        and concentration_manifest.get("distribution_contract", {}).get(
            "source_model_packaged"
        )
        is False
        and concentration_manifest.get("distribution_contract", {}).get(
            "pickle_deserialization_allowed"
        )
        is False,
        "unsafe concentration-response distribution contract",
    )
    continual_policy = json.loads(
        (DATA / "continuous_improvement_policy.json").read_text(encoding="utf-8")
    )
    continual_production = continual_policy.get("production_contract", {})
    continual_external = continual_policy.get("external_evidence_contract", {})
    continual_gates = set(continual_policy.get("required_scientific_gates", []))
    require(
        continual_policy.get("schema_version") == "1.0"
        and continual_policy.get("allowed_evidence_class")
        == "prospective_external_human"
        and {
            "prediction_sealed_before_outcome",
            "prospective_external",
            "molecule_cold",
            "scaffold_cold",
            "source_cold",
            "baseline_runtime_parity",
            "monotone_concentration_response",
            "bootstrap_mae_gain",
            "rank_noninferiority",
            "portable_runtime_parity",
        }.issubset(continual_gates)
        and int(continual_policy.get("minimum_external_sources", 0)) >= 2
        and int(continual_policy.get("minimum_evaluation_targets", 0)) >= 30
        and int(continual_policy.get("minimum_evaluation_rows", 0)) >= 100
        and int(continual_policy.get("maximum_evaluation_rows", 0))
        >= int(continual_policy.get("minimum_evaluation_rows", 0))
        and int(continual_policy.get("maximum_training_rows", 0))
        >= int(continual_policy.get("maximum_evaluation_rows", 0))
        and int(continual_policy.get("minimum_bootstrap_draws", 0)) >= 2000
        and int(continual_policy.get("maximum_bootstrap_draws", 0))
        >= int(continual_policy.get("minimum_bootstrap_draws", 0))
        and int(continual_policy.get("maximum_evidence_artifact_bytes", 0)) > 0
        and int(continual_policy.get("maximum_candidate_bundle_bytes", 0))
        >= int(continual_policy.get("maximum_evidence_artifact_bytes", 0))
        and float(
            continual_policy.get(
                "minimum_baseline_minus_candidate_mae_bootstrap_lower", -1.0
            )
        )
        == 0.0
        and continual_production.get("signed_production_authorization_required") is True
        and continual_production.get("unsafe_serialization_allowed") is False
        and continual_production.get("human_olfactory_90_percent_certified") is False
        and continual_external.get("signed_acquisition_required") is True
        and continual_external.get("authorization_artifact_type")
        == "prospective_dataset_acquisition"
        and continual_external.get(
            "production_signer_must_differ_from_acquisition_signer"
        )
        is True,
        "continual-improvement fail-closed policy changed",
    )
    with np.load(io.BytesIO(component_bytes), allow_pickle=False) as values:
        descriptors = values["descriptors"]
        require(
            descriptors.shape == (72, 217),
            f"R2 component descriptor shape changed: {descriptors.shape}",
        )
        require(
            bool(np.isfinite(descriptors).all()),
            "R2 component descriptors contain non-finite values",
        )

    continual_root = ROOT / "benchmarks" / "continuous_improvement"
    continual_event_count = 0
    if (continual_root / "registry.json").is_file():
        try:
            continual_status = ContinuousImprovementController(continual_root).status()
            continual_verification = continual_status.get("verification", {})
            continual_event_count = int(continual_verification.get("event_count", 0))
            require(
                continual_verification.get("valid") is True
                and continual_event_count >= 1,
                "continual-improvement registry or audit chain is invalid",
            )
        except Exception as error:  # noqa: BLE001 - audit must report, not crash
            require(False, f"continual-improvement registry audit failed: {error}")

    prospective_root = ROOT / "benchmarks" / "prospective_formula_blind_study_v1"
    prospective_counts = {
        "formulas": 0,
        "pairs": 0,
        "participants": 0,
        "assignments": 0,
    }
    prospective_timestamp_paths = {
        "query": prospective_root / "timestamp" / "seal.tsq",
        "response": prospective_root / "timestamp" / "seal.tsr",
        "ca": prospective_root / "timestamp" / "tsa-ca.pem",
        "tsa": prospective_root / "timestamp" / "tsa.crt",
        "verification": prospective_root / "timestamp" / "verification.json",
    }
    require(
        prospective_root.is_dir()
        and all(path.is_file() for path in prospective_timestamp_paths.values()),
        "prospective formula blind study or timestamp evidence is incomplete",
    )
    if prospective_root.is_dir() and all(
        path.is_file() for path in prospective_timestamp_paths.values()
    ):
        try:
            prospective = verify_study_seal(prospective_root)
            prospective_predictions = prospective["predictions"]
            prospective_assignments = prospective["assignments"]
            timestamp_verification = json.loads(
                prospective_timestamp_paths["verification"].read_text(encoding="utf-8")
            )
            participant_counts: dict[str, int] = {}
            pair_counts: dict[str, int] = {}
            for row in prospective_assignments:
                participant_counts[row["participant_id"]] = (
                    participant_counts.get(row["participant_id"], 0) + 1
                )
                pair_counts[row["pair_id"]] = pair_counts.get(row["pair_id"], 0) + 1
            prospective_counts = {
                "formulas": len(
                    prospective.get("formula_manifest", {}).get("formulas", [])
                ),
                "pairs": len(prospective_predictions.get("pairs", [])),
                "participants": len(participant_counts),
                "assignments": len(prospective_assignments),
            }
            require(
                prospective["seal"].get("study_id") == "PFBS-20260828-V1"
                and prospective_predictions.get("human_accuracy_claim") is False
                and prospective_predictions.get("outcome_data_accessed") is False
                and prospective_counts
                == {
                    "formulas": 24,
                    "pairs": 120,
                    "participants": 80,
                    "assignments": 2400,
                }
                and set(participant_counts.values()) == {30}
                and set(pair_counts.values()) == {20}
                and not (prospective_root / "external" / "human_outcomes.csv").exists(),
                "prospective formula blind design or outcome-unopened state changed",
            )
            timestamp = timestamp_verification.get("timestamp", {})
            require(
                timestamp_verification.get("study_id") == "PFBS-20260828-V1"
                and timestamp_verification.get("seal_sha256")
                == sha256(prospective_root / "seal.json")
                and timestamp_verification.get("human_outcome_present_at_verification")
                is False
                and timestamp_verification.get(
                    "manufacturing_evidence_present_at_verification"
                )
                is False
                and timestamp.get("verified") is True
                and timestamp.get("response_sha256")
                == sha256(prospective_timestamp_paths["response"])
                and timestamp.get("ca_sha256")
                == sha256(prospective_timestamp_paths["ca"])
                == "2151b61137ffa86bf664691ba67e7da0b19f98c758e3d228d5d8ebf27e044438"
                and timestamp.get("tsa_sha256")
                == sha256(prospective_timestamp_paths["tsa"])
                == "8bfb0305bb64e2571ca507552ef3245cb1c2fee8728e0ff8689225081ea13467",
                "prospective formula blind timestamp record changed",
            )
        except Exception as error:  # noqa: BLE001 - audit must report, not crash
            require(False, f"prospective formula blind audit failed: {error}")

    bushdid_accuracy_paths = {
        "report": ROOT / "benchmarks" / "bushdid_accuracy_v4.json",
        "model": ROOT / "benchmarks" / "bushdid_accuracy_v4_catboost.json",
        "script": ROOT / "scripts" / "experiment_bushdid_accuracy_v4.py",
    }
    require(
        all(path.is_file() for path in bushdid_accuracy_paths.values()),
        "Bushdid accuracy v4 evidence is incomplete",
    )
    bushdid_accuracy_percent = 0.0
    if all(path.is_file() for path in bushdid_accuracy_paths.values()):
        bushdid_accuracy = json.loads(
            bushdid_accuracy_paths["report"].read_text(encoding="utf-8")
        )
        candidate = bushdid_accuracy.get("results", {}).get(
            "crossfit_compact_candidate", {}
        )
        bushdid_accuracy_percent = float(
            candidate.get("absolute_accuracy_percent", 0.0)
        )
        require(
            bushdid_accuracy.get("schema") == "bushdid-outcome-aware-accuracy-v4"
            and bushdid_accuracy.get("status")
            == "outcome_aware_retrospective_development"
            and bushdid_accuracy.get("implementation", {}).get("script_sha256")
            == sha256(bushdid_accuracy_paths["script"])
            and bushdid_accuracy.get("artifact", {}).get("model_sha256")
            == sha256(bushdid_accuracy_paths["model"])
            and bushdid_accuracy.get("artifact", {}).get("model_bytes")
            == bushdid_accuracy_paths["model"].stat().st_size
            and abs(bushdid_accuracy_percent - 89.57880716951861) < 1e-12
            and abs(float(candidate.get("spearman", 0.0)) - 0.665136791076642) < 1e-12
            and bushdid_accuracy.get("target_gate", {}).get("passed") is False
            and bushdid_accuracy.get("artifact", {}).get("runtime_primary_score_weight")
            == 0.0,
            "Bushdid accuracy v4 contract changed",
        )

    industrial_registry_paths = {
        "report": ROOT / "benchmarks" / "industrial_ingredient_registry_v1.json",
        "database": ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db",
        "builder": ROOT / "scripts" / "build_industrial_ingredient_registry_v1.py",
    }
    require(
        all(path.is_file() for path in industrial_registry_paths.values()),
        "industrial ingredient registry evidence is incomplete",
    )
    industrial_registry_counts = {
        "reference_molecules": 0,
        "source_links": 0,
        "descriptor_assertions": 0,
        "prototype_safe_active": 0,
        "prototype_conditional_active": 0,
        "prototype_active_total": 0,
        "safety_screened": 0,
        "promotion_candidates_total": 0,
        "promotion_evidence_pending": 0,
        "promotion_structural_review_required": 0,
    }
    if all(path.is_file() for path in industrial_registry_paths.values()):
        industrial_registry = json.loads(
            industrial_registry_paths["report"].read_text(encoding="utf-8")
        )
        with sqlite3.connect(
            industrial_registry_paths["database"].resolve().as_uri() + "?mode=ro",
            uri=True,
        ) as connection:
            registry_integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            registry_foreign_keys = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            industrial_registry_counts = {
                "reference_molecules": scalar(
                    connection, "SELECT COUNT(*) FROM ingredients"
                ),
                "source_links": scalar(
                    connection, "SELECT COUNT(*) FROM ingredient_sources"
                ),
                "descriptor_assertions": scalar(
                    connection, "SELECT COUNT(*) FROM odor_descriptors"
                ),
                "prototype_safe_active": scalar(
                    connection,
                    "SELECT COUNT(*) FROM formulation_materials "
                    "WHERE formulation_tier='prototype_safe_active'",
                ),
                "prototype_conditional_active": scalar(
                    connection,
                    "SELECT COUNT(*) FROM formulation_materials "
                    "WHERE formulation_tier='prototype_conditional_active'",
                ),
                "prototype_active_total": scalar(
                    connection,
                    "SELECT COUNT(*) FROM formulation_materials WHERE "
                    "formulation_tier IN "
                    "('prototype_safe_active','prototype_conditional_active')",
                ),
                "safety_screened": scalar(
                    connection, "SELECT COUNT(*) FROM safety_screening"
                ),
                "promotion_candidates_total": scalar(
                    connection, "SELECT COUNT(*) FROM promotion_candidates"
                ),
                "promotion_evidence_pending": scalar(
                    connection,
                    "SELECT COUNT(*) FROM promotion_candidates "
                    "WHERE promotion_status='evidence_pending'",
                ),
                "promotion_structural_review_required": scalar(
                    connection,
                    "SELECT COUNT(*) FROM promotion_candidates "
                    "WHERE promotion_status='structural_review_required'",
                ),
            }
        require(
            registry_integrity == "ok"
            and not registry_foreign_keys
            and industrial_registry.get("schema")
            == "industrial-ingredient-registry-v1.2"
            and industrial_registry.get("implementation", {}).get("script_sha256")
            == sha256(industrial_registry_paths["builder"])
            and industrial_registry.get("database", {}).get("sha256")
            == sha256(industrial_registry_paths["database"])
            and industrial_registry_counts
            == {
                "reference_molecules": 29_240,
                "source_links": 36_327,
                "descriptor_assertions": 71_458,
                "prototype_safe_active": 29,
                "prototype_conditional_active": 5,
                "prototype_active_total": 34,
                "safety_screened": 29_240,
                "promotion_candidates_total": 29_212,
                "promotion_evidence_pending": 20_757,
                "promotion_structural_review_required": 8_455,
            }
            and industrial_registry.get("tier_contract", {}).get(
                "reference_molecules_are_formula_eligible"
            )
            is False
            and industrial_registry.get("tier_contract", {}).get(
                "all_reference_molecules_have_safety_screening"
            )
            is True
            and industrial_registry.get("tier_contract", {}).get(
                "all_unlinked_molecules_have_promotion_path"
            )
            is True
            and industrial_registry.get("tier_contract", {}).get(
                "qualified_or_commercial_materials"
            )
            == 0,
            "industrial ingredient registry contract changed",
        )

    counts = {
        "reference": reference_counts,
        "scientific": scientific_counts,
        "nonhuman_hub": hub_counts,
        "epa": epa_counts,
        "headspace_sensory_hub": headspace_counts,
        "opera_vp_strict_scaffold_test_molecules": 93,
        "natural_materials": len(natural.get("materials", [])),
        "threshold_exact_structure_matches": threshold.get("matched_record_count", 0),
        "r2_component_rows": int(component_manifest.get("component_row_count", 0)),
        "r2_ensemble_members": len(ensemble_manifest.get("members", [])),
        "concentration_response_records": int(
            concentration_manifest.get("training_records", 0)
        ),
        "continual_improvement_audit_events": continual_event_count,
        "dream_mixture_training_pairs": dream_training_pairs,
        "dream_mixture_test_pairs": dream_test_pairs,
        "dream_mixture_validation_pairs": dream_validation_pairs,
        "dream_pair_ensemble_training_pairs": dream_pair_training_pairs,
        "dream_accuracy_v4_candidates": 4960,
        "dream_set_encoder_v5_candidates": 4,
        "prospective_formula_blind_study": prospective_counts,
        "bushdid_accuracy_v4_percent": bushdid_accuracy_percent,
        "industrial_ingredient_registry": industrial_registry_counts,
    }
    result = {
        "passed": not failures,
        "failures": failures,
        "verified": verified,
        "counts": counts,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
