"""Mass/occupation-moment checks against independent matrix/ODE references."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from fragrance_ai.recommender.exposure_transport import (
    constant_moment_kernel,
    compose_moments,
    exposure_trajectory,
)
from fragrance_ai.platform.unified_product_inputs import (
    UnifiedProductRequest,
    UnifiedProductContext,
    context_id,
)
from tests.test_unified_product_v60 import (
    model as model,
    make_predictor,
    request_payload,
)


@pytest.mark.parametrize(
    "rates",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 1.0),
        (0.2, 0.01, 0.03, 0.5, 0.0),
        (1e4, 1e-6, 0.0, 1e-4, 1.0),
    ],
)
def test_augmented_kernel_matches_matrix_exponential_and_semigroup(rates):
    e, u, h, r, v = rates
    generator = np.zeros((7, 7))
    generator[:2, :2] = [[-e - u - h, r], [e, -r - v]]
    generator[2, 0], generator[3, 1], generator[4, 0] = u, v, h
    generator[5, 0], generator[6, 1] = 1.0, 1.0
    expected = expm(generator * 0.7)[:, :2].T
    values = [np.array([value]) for value in rates]
    actual = constant_moment_kernel(*values, 0.7)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)
    composed = compose_moments(
        constant_moment_kernel(*values, 0.3), constant_moment_kernel(*values, 0.4)
    )[0]
    np.testing.assert_allclose(actual, composed, atol=2e-12, rtol=1e-10)


def test_fast_peak_integral_is_not_erased_between_plotting_points():
    rates = np.array([[1e4, 0.0, 0.0, 0.0, 1e4, 0.0]])
    fractions = np.array([[0.0, 0.0]])
    initial = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    states, final, moments, exposure, detail = exposure_trajectory(
        rates, fractions, [0.0, 480.0], duration=480.0, initial=initial
    )
    assert exposure[0, 1] == pytest.approx(1e-4, rel=1e-12)
    assert exposure[0, 0] == pytest.approx(1e-4, rel=1e-12)
    assert np.trapezoid(states[:, 0, 1], [0.0, 480.0]) == 0.0
    assert final.sum() == pytest.approx(1.0, abs=1e-13)
    assert moments[-1, 0, 1] == exposure[0, 1]
    assert detail["local_dimensionless_defect_max"] == 0.0


def test_drying_reaction_return_and_exposure_match_independent_ode():
    rates = np.array([[0.4, 0.02, 0.03, 0.1, 0.05, 0.3]])
    fractions = np.array([[0.8, 0.1]])
    initial = np.array([[0.014, 0.006, 0.0, 0.0, 0.0, 0.002]])
    times = np.array([0.0, 0.017, 1.3, 8.9, 30.0])

    def rhs(t, state):
        capacity = 0.2 + 0.8 * np.exp(-0.3 * t)
        e, u, h = 0.4 / capacity, 0.02 / capacity, 0.03 * (1 - 0.1 / capacity)
        film, air = state[:2]
        return [
            -(e + u + h) * film + 0.1 * air,
            e * film - 0.15 * air,
            u * film,
            0.05 * air,
            h * film,
            0.0,
            film,
            air,
        ]

    reference = solve_ivp(
        rhs,
        (0.0, 30.0),
        np.r_[initial[0], 0.0, 0.0],
        method="DOP853",
        rtol=2e-12,
        atol=1e-14,
        t_eval=times,
    )
    assert reference.success
    actual, _, moments, _, detail = exposure_trajectory(
        rates, fractions, times, duration=30.0, initial=initial, rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        np.c_[actual[:, 0], moments[:, 0]], reference.y.T, atol=3e-8, rtol=5e-6
    )
    assert actual.min() >= 0 and np.all(np.diff(actual[:, 0, 2:], axis=0) >= 0)
    assert np.all(np.diff(moments[:, 0], axis=0) >= 0)
    np.testing.assert_allclose(actual.sum(axis=2), initial.sum(), atol=1e-13)
    assert detail["local_dimensionless_defect_max"] <= 1.01e-8


def test_display_sampling_cannot_change_final_state_or_integral():
    rates = np.array([[0.4, 0.02, 0.03, 0.1, 0.05, 0.3]])
    fractions = np.array([[0.8, 0.1]])
    initial = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    a = exposure_trajectory(
        rates, fractions, [0.0, 1.3, 30.0], duration=30.0, initial=initial
    )
    b = exposure_trajectory(
        rates,
        fractions,
        [0.0, 0.01, 0.9, 1.3, 17.0, 30.0],
        duration=30.0,
        initial=initial,
    )
    np.testing.assert_array_equal(a[0], b[0][[0, 3, 5]])
    np.testing.assert_array_equal(a[2], b[2][[0, 3, 5]])
    np.testing.assert_array_equal(a[3], b[3])


def test_work_exhaustion_is_an_error_not_a_silent_coarse_prediction():
    with pytest.raises(ValueError, match="budget"):
        exposure_trajectory(
            np.array([[0.4, 0.02, 0.03, 0.1, 0.05, 0.3]]),
            np.array([[0.8, 0.1]]),
            [0.0, 30.0],
            duration=30.0,
            initial=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            work_budget={"remaining_material_transitions": 1},
        )


@pytest.mark.parametrize("product", ["perfume", "body_lotion", "body_wash"])
def test_integrals_are_connected_to_each_product_and_rinse_keeps_past_exposure(
    model, product
):
    predictor = make_predictor(model)
    payload = request_payload(product)
    a = predictor.predict(UnifiedProductRequest(**payload))
    b_payload = deepcopy(payload)
    b_payload["times_minutes"] = [0.0, 0.01, 0.1, 0.7, 1.0, 1.2, 2.0, 2.9, 3.0]
    b = predictor.predict(UnifiedProductRequest(**b_payload))
    assert (
        a["integrated_exposure"]["materials"] == b["integrated_exposure"]["materials"]
    )
    assert a["model"]["learned_transport_applied_to_finished_product"] is False
    assert all(
        row["air_exposure_mg_min_m3"] > 0
        for row in a["integrated_exposure"]["materials"]
    )
    assert a["human_similarity_percent"] is None
    if product == "body_wash":
        before = request_payload("body_lotion")
        before["context"]["product_type"] = "body_wash"
        before["stages"][0]["rinse_retained_film_fractions"] = dict.fromkeys(
            [item["ingredient_id"] for item in before["components"]], 1.0
        )
        before["stages"][0]["rinse_source_reference"] = "unit retention test fixture"
        before["parameter_context_id"] = context_id(
            UnifiedProductContext(**before["context"])
        )
        unwashed = predictor.predict(UnifiedProductRequest(**before))
        at_rinse = a["temporal_profile"][2]["materials"]
        previous = unwashed["temporal_profile"][2]["materials"]
        for first, second in zip(at_rinse, previous):
            assert (
                first["cumulative_air_exposure_mg_min_m3"]
                == second["cumulative_air_exposure_mg_min_m3"]
            )
        assert (
            a["integrated_exposure"]["materials"][0]["air_exposure_mg_min_m3"]
            < unwashed["integrated_exposure"]["materials"][0]["air_exposure_mg_min_m3"]
        )
