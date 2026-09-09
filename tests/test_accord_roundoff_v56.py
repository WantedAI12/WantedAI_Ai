"""Regression for the all-donor transfer that crashed an aquatic/earthy request."""
import numpy as np
import pytest

from fragrance_ai.recommender.accord_trials import accord_trials
from fragrance_ai.recommender.profile_match import compare_profiles


def test_complete_transfer_never_creates_negative_recipe_mass():
    weights = np.array([0.3475612584698452, 0.2443080743974399,
                        0.02120605176345119, 0.3869246153692636])
    before = weights.copy()
    profiles = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    trials = list(accord_trials(weights, profiles, np.ones(4), [0., 1.], np.zeros(4), np.ones(4)))
    assert len(trials) == 5
    for candidate in trials:
        assert np.all(candidate >= 0.)
        assert np.all(candidate <= 1.)
        assert abs(candidate.sum()-weights.sum()) < 1e-15
        result = compare_profiles([0., 1.], candidate@profiles, dimensions=('donor','receiver'))
        assert result.score is not None
    np.testing.assert_array_equal(weights, before)


def test_bound_roundoff_repair_preserves_group_mass_and_floors():
    rng = np.random.default_rng(56)
    profiles = np.tile(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]]), (2, 1))
    groups = np.repeat([0, 1], 4)
    for _ in range(60):
        weights = np.r_[rng.dirichlet(np.ones(4))*.4, rng.dirichlet(np.ones(4))*.6]
        lower = weights*.1
        caps = np.minimum(1., weights+.3)
        for candidate in accord_trials(weights, profiles, np.ones(8), [0., 1.], lower, caps,
                                      budget_groups=groups):
            assert np.all(candidate >= lower)
            assert np.all(candidate <= caps)
            for group in (0, 1):
                assert abs(candidate[groups == group].sum()-weights[groups == group].sum()) < 1e-15


def test_actual_negative_input_is_rejected_not_repaired():
    with pytest.raises(ValueError,match='nonnegative'):
        list(accord_trials([-.01, 1.01], np.eye(2), [1., 1.], [0., 1.], [0., 0.], [1., 1.]))
