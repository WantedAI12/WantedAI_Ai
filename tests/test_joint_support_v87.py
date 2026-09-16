import numpy as np
import pytest

from fragrance_ai.recommender.failure_inverse_v87 import expanded_reference_support, quantize_physical_recipe


def test_column_expansion_keeps_required_materials_and_prices_score_and_signal():
    w = np.array([.5, .5, 0., 0., 0., 0.])
    lo = np.array([.1, 0., 0., 0., 0., 0.])
    hi = np.array([1., 1., 1., 1., 0., 1.])
    g = np.array([0., 0., -10., 0., -100., 2.])
    signal = np.array([[0., 0., 0., 20., 100., 0.]])
    selected = expanded_reference_support(g, signal, np.ones(1), w, lo, hi, 4,
                                          profile_columns=1, signal_columns=1)
    assert selected.tolist() == [0, 1, 2, 3]
    assert expanded_reference_support(g, signal, np.ones(1), w, lo, hi, 2).tolist() == [0, 1]


def test_column_expansion_does_not_discard_existing_mass_to_fit_a_smaller_limit():
    with pytest.raises(ValueError, match='support exceeds'):
        expanded_reference_support(np.zeros(2), np.zeros((1, 2)), np.ones(1),
                                   np.array([.5, .5]), np.zeros(2), np.ones(2), 1)


def test_rounded_recipe_uses_same_physical_model_and_mass_units():
    from tests.test_reference_inverse_v81 import inputs
    from fragrance_ai.recommender.nonlinear_inverse import profile_loss
    _, profiles, engine, target, tw, mask = inputs()
    w = np.array([.2, .5, .3])
    signal, _ = engine.signal_value_gradient(w)
    rounded, report = quantize_physical_recipe(engine, profiles, target, tw, mask, w,
        np.zeros(3), np.ones(3), np.ones(3), 2., signal_floor=signal*.99, anchor=w)
    assert rounded is not None and rounded.sum() == pytest.approx(1.)
    predicted, _ = engine.predict(profiles, rounded)
    assert report['quantized_exact_score'] == pytest.approx(100*(1-profile_loss(predicted, target, tw, mask)[0]))
    assert np.all(engine.signal_value_gradient(rounded)[0] >= signal*.99)


def test_impossible_rounding_guard_never_returns_a_recipe():
    from tests.test_reference_inverse_v81 import inputs
    _, profiles, engine, target, tw, mask = inputs()
    w = np.array([.2, .5, .3])
    rounded, report = quantize_physical_recipe(engine, profiles, target, tw, mask, w,
        np.zeros(3), np.ones(3), np.ones(3), 2., signal_floor=np.full(5, 1e10), anchor=w)
    assert rounded is None and report['output_quantization'] == 'no_guard_preserving_rounded_doses'


def test_physical_rounding_choices_preserve_the_original_mass_and_stock_caps():
    from fragrance_ai.recommender.failure_inverse_v87 import quantize_recipe
    w = np.array([.40000049, .59999949, .00000002])
    values = np.array([1., 0., 0.])
    rounded = quantize_recipe(w, np.zeros(3), np.ones(3), np.ones(3), 2.,
        rounding_cost=np.array([-1., 1., 10.]), binary_bands=[(values, 1., 1.)])
    np.testing.assert_allclose(rounded, [.400001, .599999, 0.], atol=1e-12)
    blocked = quantize_recipe(w, np.zeros(3), np.array([.40000049, 1., 1.]), np.ones(3), 2.,
        rounding_cost=np.array([-1., 1., 10.]), binary_bands=[(values, 1., 1.)])
    assert blocked is None


def test_finite_coordinate_rounding_reports_fresh_not_linearized_score():
    from tests.test_reference_inverse_v81 import inputs
    from fragrance_ai.recommender.nonlinear_inverse import profile_loss
    _, profiles, engine, _, tw, mask = inputs()
    w = np.array([.40000049, .59999949, .00000002])
    target, _ = engine.predict(profiles, w)
    rounded, report = quantize_physical_recipe(engine, profiles, target, tw, mask, w,
        np.zeros(3), np.ones(3), np.ones(3), 2., target_score=100.)
    assert rounded is not None and rounded.sum() == pytest.approx(1.)
    predicted, _ = engine.predict(profiles, rounded)
    assert report['quantized_exact_score'] == pytest.approx(100*(1-profile_loss(predicted, target, tw, mask)[0]))


def test_explicit_phases_use_their_own_old_worst_head_not_the_overall_worst_time():
    from fragrance_ai.recommender.failure_inverse_v87 import preservation_ceilings
    losses = np.array([[.3, .1, .4, .2], [.4, .2, .6, .3]])
    actual = preservation_ceilings(losses, [0, .3, .4, .3],
        ['overall', 'opening', 'heart', 'drydown'], {'opening', 'drydown'})
    np.testing.assert_allclose(actual, [[1., .2, .6, .3], [1., .2, .6, .3]])


@pytest.mark.parametrize('repair_mode,full_reference,expected', [(False, True, 12), (True, True, 100), (False, False, 12)])
def test_only_failed_request_repair_uses_the_new_full_support_policy(monkeypatch, repair_mode, full_reference, expected):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.failure_recovery import recovery_context
    engine = NaturalLanguagePerfumeryAI(require_full_profile_match=True)
    if full_reference:
        engine.perception_guidance = SimpleNamespace(complete_reference_bank=SimpleNamespace(odor_space=object()))
    monkeypatch.setattr('fragrance_ai.recommender.perception_runtime.assert_provider_current', lambda _: None)
    observed = []
    class Captured(Exception):
        pass
    def run(text, constraints, *args, **kwargs):
        observed.append(constraints.max_ingredients)
        raise Captured()
    monkeypatch.setattr(engine, '_create_recipe_impl', run)
    with recovery_context() if repair_mode else nullcontext():
        with pytest.raises(Captured):
            engine.create_recipe('opening citrus, heart floral, drydown woody', RecipeConstraints(max_ingredients=100))
    assert observed == [expected]
