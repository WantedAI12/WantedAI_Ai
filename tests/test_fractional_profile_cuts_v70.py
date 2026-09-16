import numpy as np
import pytest

from fragrance_ai.recommender.fractional_profile_cuts import solve_profile_cuts


def problem(n=300):
    shapes = np.zeros((2, n, 4))
    shapes[:, :n//2, 0] = 1.
    shapes[:, n//2:, 1] = 1.
    target = np.zeros((2, 3, 4))
    target[:, :, :2] = .5
    return dict(shapes=shapes, wanted=target, responses=np.ones((3, n)),
                fixed=np.ones((1, n)), rhs=np.array([1.]),
                equality=np.ones((1, n)), equality_rhs=np.array([1.]),
                bounds=[(0., 1.)]*n, objective=np.linspace(1., 2., n), level=.95)


def test_every_material_and_endpoint_participates_without_slack_expansion():
    args = problem()
    result = solve_profile_cuts(**args)
    assert result.success and result.x.sum() == pytest.approx(1.)
    assert .45-1e-8 <= result.x[:150].sum() <= .55+1e-8
    assert result.full_pool_count == 300 and result.full_endpoint_count == 4
    assert result.cut_count > 0 and result.linear_solves < 8


def test_unrequested_odors_are_not_dropped_to_make_the_target_pass():
    args = problem()
    args['shapes'] *= .8
    args['shapes'][:, :, 3] = .2
    result = solve_profile_cuts(**args)
    assert not result.success and result.status == 2
    assert result.candidate_score is None or result.candidate_score <= .8+1e-8


def test_material_caps_and_physical_inequalities_remain_binding():
    args = problem()
    args['bounds'][150:] = [(0., 0.)]*150
    result = solve_profile_cuts(**args)
    assert not result.success and result.status == 2


def test_work_limit_is_not_infeasibility_or_a_fabricated_recipe():
    result = solve_profile_cuts(**problem(), maximum_seconds=1e-12)
    assert not result.success and result.status == 1 and result.x is None


def test_all_transport_windows_must_satisfy_the_same_target():
    args = problem()
    args['responses'][1, :150] = 100.
    result = solve_profile_cuts(**args)
    assert not result.success


def test_missing_disabled_shapes_are_allowed_but_not_selected():
    args = problem()
    args['shapes'][:, 0] = 0.
    args['bounds'][0] = (0., 0.)
    result = solve_profile_cuts(**args)
    assert result.success and result.x[0] == 0
