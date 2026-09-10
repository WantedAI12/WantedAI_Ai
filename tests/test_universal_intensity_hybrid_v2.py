from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_universal_intensity_hybrid_v2 import _equal_centered_hybrid


def test_equal_centered_hybrid_preserves_anchor_mean_and_is_identity_free():
    universal = np.asarray([0.2, 0.4, 0.7, 0.9])
    humanpom = np.asarray([0.5, 0.6, 0.7, 0.8])
    result = _equal_centered_hybrid(universal, humanpom)
    assert float(result.mean()) == pytest.approx(float(humanpom.mean()))
    expected = humanpom.mean() + 0.5 * (humanpom - humanpom.mean()) + 0.5 * (
        universal - universal.mean()
    )
    assert result.tolist() == pytest.approx(expected.tolist())


def test_equal_centered_hybrid_rejects_bad_shapes_and_nonfinite_values():
    with pytest.raises(ValueError):
        _equal_centered_hybrid(np.asarray([0.1, 0.2]), np.asarray([0.1]))
    with pytest.raises(ValueError):
        _equal_centered_hybrid(
            np.asarray([0.1, np.nan, 0.3]), np.asarray([0.1, 0.2, 0.3])
        )


def test_published_hybrid_v2_is_retrospective_only():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "universal_intensity_hybrid_v2.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["retrospective_repair_gate"]["passed"] is True
    assert value["prospective_external_gate"]["passed"] is False
    assert value["runtime"]["primary_score_weight"] == 0.0
