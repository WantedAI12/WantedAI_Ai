"""Synthetic numerical checks; these are not measured perfume performances."""

import numpy as np
import pytest

from fragrance_ai.recommender.blend_correction import correct_blend


def problem():
    return dict(shapes=np.eye(2)[None], wanted=np.array([[[.6, .4]]]),
                responses=np.ones((1, 2)), fixed=np.array([[1., 2.]]), rhs=np.array([2.]),
                equality=np.ones((1, 2)), equality_rhs=np.array([1.]),
                bounds=[(0., 1.), (0., 1.)], initial=np.array([.5, .5]),
                target=.9, avoided=np.zeros((1, 2)), background=np.array([[.5, .5]]))


def test_correction_changes_mass_and_satisfies_background_together():
    args = problem()
    corrected, report = correct_blend(**args)
    assert report['target_met'] and report['reported_score_offset'] == 0.
    assert report['changed_materials'] == 2
    assert corrected.sum() == pytest.approx(1.)
    assert corrected[0] > args['initial'][0]
    assert np.array_equal(args['initial'], [.5, .5])
    assert report['selected_score'] >= 90.


def test_passing_formula_is_not_changed():
    args = problem()
    args['initial'] = np.array([.6, .4])
    corrected, report = correct_blend(**args)
    np.testing.assert_array_equal(corrected, args['initial'])
    assert report['linear_solves'] == report['changed_materials'] == 0


def test_correction_cannot_activate_excluded_ingredient_or_invent_target_pass():
    args = problem()
    args.update(initial=np.array([0., 1.]), bounds=[(0., 0.), (0., 1.)], background=None)
    corrected, report = correct_blend(**args)
    assert corrected[0] == 0
    assert corrected.sum() == pytest.approx(1.)
    assert report['target_met'] is False
    assert report['selected_score'] == pytest.approx(40.)


def test_physical_caps_and_price_are_preserved():
    args = problem()
    args.update(bounds=[(.4, .55), (.45, .6)], rhs=np.array([1.55]))
    corrected, report = correct_blend(**args)
    assert .45 - 1e-8 <= corrected[0] <= .55 + 1e-8
    assert args['fixed'] @ corrected <= args['rhs'] + 1e-9
    assert report['reported_score_offset'] == 0


@pytest.mark.parametrize('key,value', [('target', 0.), ('target', float('nan')),
                                    ('maximum_seconds', -1.), ('maximum_rounds', 0)])
def test_invalid_controls_fail_closed(key, value):
    args = problem()
    args[key] = value
    with pytest.raises(ValueError):
        correct_blend(**args)


def physical_problem():
    from types import SimpleNamespace
    from fragrance_ai.recommender.nonlinear_inverse import NonlinearDoseObjective
    engine = object.__new__(NonlinearDoseObjective)
    engine.draws = 2
    engine.model = SimpleNamespace(gain=np.ones(2), base_moles=1., total_moles=np.ones(2),
        coefficients=np.ones((2, 2, 2)), transport=np.ones(2), suppression=np.zeros(2),
        interaction=np.zeros((2, 2)))
    p = np.eye(2)[None]
    q = np.broadcast_to([.7, .3], (1, 3, 2)).copy()
    return engine, p, q


def test_release_factors_reconstruct_the_anchor_without_changing_physics():
    from fragrance_ai.recommender.physical_blend_correction import release_factors
    engine, p, _ = physical_problem()
    for w in (np.array([.4, .6]), np.array([1., 0.])):
        factors, _, expected = release_factors(engine, p, w)
        actual = np.einsum('tn,n,hnd->htd', factors, w, p) / (factors @ w)[None, :, None]
        np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_physical_correction_is_a_real_mass_change_rechecked_by_original_model():
    from fragrance_ai.recommender.physical_blend_correction import correct_physical_blend
    from fragrance_ai.recommender.nonlinear_inverse import profile_loss
    engine, p, q = physical_problem()
    initial, tw, mask = np.array([.5, .5]), np.array([0., .5, .5]), np.zeros((3, 2))
    result, report = correct_physical_blend(engine, p, q, tw, mask, initial,
        np.zeros(2), np.ones(2), np.ones(2), 2., target_score=90.)
    assert report['changed_materials'] == 2 and report['score_offset'] == 0
    fresh, _ = engine.predict(p, result)
    fresh_score = 100 * (1 - profile_loss(fresh, q, tw, mask)[0])
    assert fresh_score == pytest.approx(report['selected_score'])
    assert fresh_score > report['starting_score']
    assert result.sum() == pytest.approx(1.)
