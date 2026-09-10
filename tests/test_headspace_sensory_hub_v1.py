from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from fragrance_ai.research.headspace import (
    AqueousHeadspaceInput,
    HeadspaceInput,
    HeadspaceSensoryHub,
    load_calibrated_response_exponent,
)


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "benchmarks" / "headspace_sensory_hub_v1.db"
HUB_REPORT = ROOT / "benchmarks" / "headspace_sensory_hub_v1.json"
CALIBRATION = ROOT / "benchmarks" / "concentration_headspace_calibration_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hub_is_source_bound_integral_and_kept_out_of_the_wheel() -> None:
    report = json.loads(HUB_REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "headspace-sensory-hub/v1"
    assert report["database"]["sha256"] == sha256(HUB)
    assert report["database"]["packaged_in_wheel"] is False
    assert report["counts"] == {
        "molecule_source_links": 2449,
        "molecules": 1642,
        "molecules_with_physchem": 869,
        "physchem_observations": 27283,
        "sensory_observations": 109688,
        "source_files": 29,
        "stimuli": 2689,
        "stimulus_components": 21708,
        "stimulus_dilutions": 1473,
    }
    assert report["source"]["pyrfume_commit"] == (
        "8054ea98ed675005ec10e67359902f500e4911b0"
    )
    assert report["source"]["opera_license"] == "CC0"
    assert report["claim_boundary"]["human_olfactory_90_percent_certified"] is False
    with sqlite3.connect(HUB.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        units = connection.execute(
            "SELECT DISTINCT unit FROM sensory_observations "
            "WHERE dataset='abraham_2012'"
        ).fetchall()
        assert units == [("log10_inverse_ppmv",)]
        dilution_counts = dict(
            connection.execute(
                "SELECT parse_status, COUNT(*) FROM stimulus_dilutions "
                "GROUP BY parse_status"
            ).fetchall()
        )
        assert sum(dilution_counts.values()) == 1473
        assert dilution_counts["parsed_percent"] > 0
        assert dilution_counts["parsed_fraction"] > 0
        assert dilution_counts["non_numeric_solid"] > 0
    package_configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "headspace_sensory_hub_v1.db" not in package_configuration


def test_measured_raoult_and_henry_paths_are_unit_consistent() -> None:
    exponent = load_calibrated_response_exponent(CALIBRATION)
    with HeadspaceSensoryHub(HUB, report=HUB_REPORT) as hub:
        vapor = hub.vapor_pressure(8908)
        assert vapor is not None
        assert vapor.evidence_class == "measured_opera_vapor_pressure_median"
        assert vapor.pressure_pa == pytest.approx(176.40180316146362)
        assert hub.odor_threshold_ppmv(8908) == pytest.approx(0.014851033121245793)
        result = hub.raoult_headspace(
            [HeadspaceInput(8908, 0.001), HeadspaceInput(8175, 0.001)],
            response_exponent=exponent,
        )
        assert result.status == "complete_equilibrium_reference"
        assert result.vapor_pressure_coverage_percent == 100.0
        assert result.odor_threshold_coverage_percent == 100.0
        assert sum(item.headspace_share or 0.0 for item in result.components) == pytest.approx(
            1.0
        )
        assert all(item.odor_activity_value is not None for item in result.components)
        aqueous = hub.aqueous_headspace(
            [AqueousHeadspaceInput(8908, 0.001)], response_exponent=exponent
        )
        assert aqueous.method == "dilute_aqueous_henry_equilibrium"
        assert aqueous.components[0].gas_ppmv == pytest.approx(0.53)
        assert aqueous.human_olfactory_accuracy_claimed is False


def test_missing_partition_data_stays_missing_and_partial() -> None:
    with HeadspaceSensoryHub(HUB, report=HUB_REPORT) as hub:
        result = hub.raoult_headspace(
            [HeadspaceInput(8908, 0.001), HeadspaceInput(5363388, 0.001)]
        )
        assert result.status == "partial_equilibrium_reference"
        assert result.vapor_pressure_coverage_percent == pytest.approx(50.0)
        missing = next(item for item in result.components if item.cid == 5363388)
        assert missing.partial_pressure_pa is None
        assert missing.partition_evidence_class == "missing"
        assert "one_or_more_components_missing_vapor_partition_data" in result.flags


def test_concentration_calibration_preserves_holdout_and_claim_boundary() -> None:
    document = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert document["schema"] == "concentration-headspace-calibration/v1"
    assert document["source"]["hub_sha256"] == sha256(HUB)
    assert document["implementation"]["script_sha256"] == sha256(
        ROOT / "scripts" / "calibrate_concentration_headspace_v1.py"
    )
    assert document["split"]["molecule_overlap"] == 0
    assert document["split"]["selection_used_holdout_outcomes"] is False
    assert document["split"]["training_molecules"] == 378
    assert document["split"]["holdout_molecules"] == 102
    for direction in ("low_to_high", "high_to_low"):
        result = document["holdout"][direction]
        assert result["candidate"]["mae"] < result["unchanged_intensity_baseline"]["mae"]
        assert result["paired_molecule_bootstrap"][
            "baseline_minus_candidate_mae_95_interval"
        ][0] > 0.0
    assert document["gates"]["molecule_holdout_transfer"]["passed"] is True
    assert document["gates"]["production"]["passed"] is False
    assert document["runtime_primary_score_weight"] == 0.0
    assert document["claim_boundary"]["human_olfactory_90_percent_certified"] is False
