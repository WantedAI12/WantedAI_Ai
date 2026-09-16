import numpy as np
import pytest
from fragrance_ai.recommender.simplex_geometry import (
    tangent_gradient,
    torch_tangent_gradient,
    GEOMETRIES,
)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_gradient_is_intrinsic_and_matches_torch(geometry):
    import torch

    rng = np.random.default_rng(761021)
    g = rng.normal(size=(2, 11))
    w = rng.dirichlet(np.ones(11), size=2)
    w[0, 0] = 0.0
    mask = np.ones_like(w, bool)
    mask[:, 2] = False
    product = np.array([0, 1])
    actual = tangent_gradient(g, w, mask, product, geometry)
    shifted = tangent_gradient(g + 17.0, w, mask, product, geometry)
    np.testing.assert_allclose(actual, shifted, atol=5e-15)
    np.testing.assert_allclose(actual.sum(-1), 0.0, atol=1e-15)
    assert np.all(actual[:, 2] == 0.0)
    expected = torch_tangent_gradient(
        torch.tensor(g),
        torch.tensor(w),
        torch.tensor(mask),
        torch.tensor(product),
        geometry,
    ).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert np.all((g * actual).sum(-1) >= 0.0)


def test_constant_objective_gradient_cannot_move_a_feasible_recipe():
    g = np.ones((1, 4)) * 100
    for geometry in GEOMETRIES:
        np.testing.assert_array_equal(
            tangent_gradient(
                g,
                np.array([[0.0, 0.2, 0.3, 0.5]]),
                np.ones((1, 4), bool),
                np.array([0]),
                geometry,
            ),
            0.0,
        )
