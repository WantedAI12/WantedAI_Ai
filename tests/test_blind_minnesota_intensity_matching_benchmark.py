from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.blind_minnesota_intensity_matching_benchmark import (
    COMPOUNDS,
    FILES,
    GRID_LOG10_PPM,
    _match_log_ppm,
    _monotonic,
    assert_outcomes_absent,
)


def test_minnesota_contract_has_four_targets_one_standard_and_three_files():
    assert len(COMPOUNDS) == 5
    assert {row["code"] for row in COMPOUNDS} == {"BA", "DD", "F", "M", "AP"}
    assert len(FILES) == 3
    assert sum(row["expected_rows"] for row in FILES) == 1740


def test_monotonic_projection_and_match_interpolation_are_bounded():
    raw = np.linspace(0.1, 0.9, len(GRID_LOG10_PPM))
    raw[20] = 0.0
    curve = _monotonic(raw)
    assert np.all(np.diff(curve) >= 0.0)
    match, status = _match_log_ppm(curve, 0.5)
    assert GRID_LOG10_PPM[0] < match < GRID_LOG10_PPM[-1]
    assert status == "interpolated"


def test_minnesota_outcome_absence_checks_all_three_files(tmp_path: Path):
    assert_outcomes_absent(tmp_path)
    (tmp_path / str(FILES[1]["filename"])).write_text("outcome", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_outcomes_absent(tmp_path)


def test_published_minnesota_external_gate_remains_failed():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "minnesota_intensity_blind_benchmark_v1.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["external_concentration_gate"]["passed"] is False
    assert value["mixture_intensity_external_gate"]["passed"] is False
    assert value["runtime_primary_score_weight"] == 0.0
    assert value["human_olfactory_90_percent_certified"] is False
