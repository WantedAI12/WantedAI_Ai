from __future__ import annotations

import pytest

from scripts.acquire_universal_odor_thresholds_v2 import parse_threshold_ppm


def test_threshold_parser_handles_bracketed_ppm_and_prefers_low_value():
    assert parse_threshold_ppm(["Odor Threshold Low: 0.00008 [ppm]"], 104.17) == pytest.approx(
        0.00008
    )
    value = parse_threshold_ppm(
        ["0.001 mg/cu m (Odor low) 9 mg/cu m (Odor high)"], 88.11
    )
    assert value == pytest.approx(0.001 * 24.45 / 88.11)


def test_threshold_parser_converts_ppb_and_keeps_missing_null():
    assert parse_threshold_ppm(["Detection threshold 2 ppb"], 100.0) == pytest.approx(
        0.002
    )
    assert parse_threshold_ppm([], 100.0) is None
    with pytest.raises(ValueError):
        parse_threshold_ppm(["1 ppm"], 0.0)
