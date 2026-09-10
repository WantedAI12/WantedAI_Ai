import numpy as np
import pytest

from fragrance_ai.recommender.optimizer import (
    NoFeasibleFormula,
    _project_capped_simplex,
)


def _bisection_reference(values: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    low = float(np.min(values - caps)) - total
    high = float(np.max(values)) + total
    for _ in range(80):
        middle = (low + high) / 2.0
        projected = np.clip(values - middle, 0.0, caps)
        if projected.sum() > total:
            low = middle
        else:
            high = middle
    return np.clip(values - high, 0.0, caps)


def test_exact_capped_simplex_projection_matches_reference():
    random = np.random.default_rng(20260803)
    for width in range(1, 15):
        for _ in range(50):
            values = random.normal(0.0, 12.0, width)
            caps = random.uniform(0.01, 40.0, width)
            total = random.uniform(0.0, float(caps.sum()))
            actual = _project_capped_simplex(values, caps, total)
            expected = _bisection_reference(values, caps, total)
            assert actual == pytest.approx(expected, abs=1e-8)
            assert float(actual.sum()) == pytest.approx(total, abs=1e-7)
            assert np.all(actual >= -1e-10)
            assert np.all(actual <= caps + 1e-10)


def test_capped_simplex_rejects_infeasible_total():
    with pytest.raises(NoFeasibleFormula):
        _project_capped_simplex(
            np.asarray([1.0, 2.0]),
            np.asarray([10.0, 10.0]),
            21.0,
        )
