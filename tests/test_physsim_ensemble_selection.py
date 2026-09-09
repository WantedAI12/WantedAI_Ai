import json
from pathlib import Path

import pytest

from scripts.build_physsim_ensemble_evidence import _select_member_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _report(name: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / "benchmarks" / name).read_text(encoding="utf-8")
    )


def test_two_seed_weights_are_selected_from_development_only_on_coarse_grid():
    weights, candidates = _select_member_weights(
        _report("physsim_r2_transfer_development_calibration.json"),
        _report("physsim_r2_transfer_development_seed_20260713.json"),
    )

    assert weights == pytest.approx({20260715: 0.3, 20260713: 0.7})
    assert len(candidates) == 11
    selected = next(
        row for row in candidates if row["seed_20260715_weight"] == 0.3
    )
    assert selected["balanced_mean_spearman"] == max(
        row["balanced_mean_spearman"] for row in candidates
    )
    assert set(selected["protocol_metrics"]) == {
        "molecule_disjoint",
        "scaffold_disjoint",
    }
