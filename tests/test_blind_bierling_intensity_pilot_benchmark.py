from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.blind_bierling_intensity_pilot_benchmark import (
    BOOTSTRAP_DRAWS,
    CANDIDATES,
    GRID_FRACTIONS,
    GRID_LOG10,
    PILOT_BYTES,
    PILOT_FILE,
    PILOT_MD5,
    PILOT_URL,
    _curve_prediction,
    _monotonic_by_molecule,
    assert_pilot_absent,
    math_log10,
    parse_concentration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_intensity_pilot_contract_and_grid_are_frozen():
    assert PILOT_FILE == "intensity_piloting.csv"
    assert PILOT_BYTES == 37_054
    assert PILOT_MD5 == "3f7e0317b8e05c84545a1d9adb00e93c"
    assert PILOT_URL.endswith("/intensity_piloting.csv/content")
    assert BOOTSTRAP_DRAWS == 1000
    assert len(GRID_LOG10) == len(GRID_FRACTIONS) == 241
    assert GRID_LOG10[0] == -6.0 and GRID_LOG10[-1] == 0.0
    assert len({row["name"] for row in CANDIDATES}) == len(CANDIDATES)


def test_concentration_parser_is_dimensionless_and_fail_closed():
    assert parse_concentration("1/100") == pytest.approx(0.01)
    assert parse_concentration(" 1 / 10 ") == pytest.approx(0.1)
    assert parse_concentration("undiluted") == 1.0
    with pytest.raises(ValueError):
        parse_concentration("2g/10ml")
    with pytest.raises(ValueError):
        parse_concentration("0/10")
    with pytest.raises(ValueError):
        math_log10(0.0)


def test_curve_interpolation_and_outcome_absence(tmp_path):
    row = {
        "grid_log10_fraction": [-2.0, -1.0, 0.0],
        "curve": [10.0, 30.0, 50.0],
    }
    assert _curve_prediction(row, "curve", 0.1) == 30.0
    with pytest.raises(RuntimeError):
        _curve_prediction(row, "curve", 0.001)
    target = tmp_path / PILOT_FILE
    assert_pilot_absent(target)
    (tmp_path / "intensity_piloting.xlsx").write_bytes(b"outcome")
    with pytest.raises(RuntimeError):
        assert_pilot_absent(target)


def test_molecule_curve_projection_is_monotonic_without_cross_molecule_mixing():
    prediction = np.asarray([30.0, 20.0, 10.0, 40.0])
    smiles = ["A", "A", "B", "B"]
    fractions = np.asarray([0.1, 0.01, 0.01, 0.1])
    projected = _monotonic_by_molecule(prediction, smiles, fractions)
    assert projected.tolist() == [30.0, 20.0, 10.0, 40.0]
    reversed_prediction = np.asarray([10.0, 20.0, 50.0, 40.0])
    projected = _monotonic_by_molecule(reversed_prediction, smiles, fractions)
    assert projected.tolist() == [20.0, 20.0, 50.0, 50.0]


def test_published_intensity_result_remains_fail_closed_when_present():
    report = ROOT / "benchmarks" / "bierling_2025_intensity_blind_benchmark_v1.json"
    if not report.is_file():
        return
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["human_olfactory_90_percent_certified"] is False
    assert result["mixture_or_recipe_validated"] is False
    assert result["blind_integrity"]["timestamp"]["verified"] is True
    assert np.isfinite(
        result["results"]["strict_structure_concentration_curve"]["row_spearman"]
    )
    assert result["population"] == {
        "participants": 100,
        "ratings": 964,
        "molecules": 73,
        "conditions": 75,
        "concentration_minimum": 0.01,
        "concentration_maximum": 1.0,
    }
    anchored = result["results"]["condition_transfer_anchored_curve"]
    ravia = result["results"]["frozen_ravia_global_curve"]
    assert anchored["row_spearman"] == pytest.approx(0.5541519097654776)
    assert anchored["mae"] == pytest.approx(21.646485383565675)
    assert ravia["row_spearman"] == pytest.approx(-0.1184298164315976)
    assert ravia["mae"] == pytest.approx(12.960351980243308)
    assert result["condition_transfer_improvement_gate"]["passed"] is False
    assert result["strict_external_gate"]["passed"] is False
    assert result["parser_adjudication"]["developed_after_pilot_opened"] is True
