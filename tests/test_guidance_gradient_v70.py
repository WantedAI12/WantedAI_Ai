import numpy as np
import pytest
from fragrance_ai.recommender.formulation_guidance import projected_guidance_gradient


@pytest.mark.parametrize('avoid', [[], [1, 4]])
def test_guidance_derivative_matches_central_difference_and_exact_score(avoid):
    rng = np.random.default_rng(7051)
    shapes = rng.uniform(.001, .05, (7, 2, 5))
    strengths = np.array([1., .1, .5, 1., .9, .2, 1.])
    weights = rng.uniform(.1, 1, 7)
    weights /= weights.sum()
    target = np.array([.5, 0., .3, .2, 0.])
    function = projected_guidance_gradient(shapes, strengths, target, avoid)
    score, jac = function(weights)
    finite = []
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = 1e-7
        finite.append((function(weights+delta)[0]-function(weights-delta)[0])/2e-7)
    np.testing.assert_allclose(jac, finite, atol=2e-6, rtol=2e-5)
    mass = weights*strengths
    projected = np.einsum('n,nhd->hd', mass/mass.sum(), shapes)
    cosine = (projected@target)/(np.linalg.norm(projected, axis=1)*np.linalg.norm(target))
    expected = np.clip(100*(cosine*projected.sum(1)-projected[:, avoid].sum(1)/projected.sum(1)), 0, 100).min()
    assert score == pytest.approx(expected, abs=1e-10)
    assert weights@jac == pytest.approx(0., abs=1e-9)


def test_negative_and_empty_guidance_inputs_are_not_scored_as_valid():
    with pytest.raises(ValueError, match='target'):
        projected_guidance_gradient(np.ones((2, 2, 3)), np.ones(2), np.zeros(3), [])
    fn = projected_guidance_gradient(np.ones((2, 2, 3))*.1, np.ones(2), np.ones(3), [])
    with pytest.raises(ValueError, match='mass'):
        fn(np.array([-.1, 1.1]))
