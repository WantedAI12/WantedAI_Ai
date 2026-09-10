import zipfile

import pytest

from scripts.benchmark_service_load import (
    latency_summary,
    percentile,
    wheel_source_mismatches,
)


def test_percentile_interpolates_and_validates_inputs():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.1)


def test_latency_summary_has_stable_units():
    summary = latency_summary([0.001, 0.002, 0.003])
    assert summary["minimum_ms"] == 1.0
    assert summary["p50_ms"] == 2.0
    assert summary["maximum_ms"] == 3.0


def test_wheel_source_binding_detects_package_drift(tmp_path):
    root = tmp_path / "source"
    package = root / "fragrance_ai"
    package.mkdir(parents=True)
    source = package / "api.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fragrance_ai/api.py", source.read_bytes())
    assert wheel_source_mismatches(wheel, root) == []
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert wheel_source_mismatches(wheel, root) == ["fragrance_ai/api.py"]
