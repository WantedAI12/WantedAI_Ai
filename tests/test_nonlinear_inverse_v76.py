import numpy as np
import pytest

from tests.test_dose_refinement import materials
from fragrance_ai.recommender.nonlinear_inverse import NonlinearDoseObjective


@pytest.mark.parametrize("concentration", [3.0, 15.0, 30.0])
def test_physical_scalar_vjp_matches_independent_differences(concentration):
    items = materials()
    physics = NonlinearDoseObjective(items, {}, concentration, draws=16)
    p = np.array([i.vector() for i in items])[None, None]
    rng = np.random.default_rng(34)
    q = rng.dirichlet(np.ones(19), size=(1, 1, 6))
    w = np.array([[0.2, 0.35, 0.45]])
    times = np.array([[0, 0.1, 0.2, 0.3, 0.2, 0.2]])
    mask = np.zeros_like(q)
    mask[..., 3] = 1
    args = dict(time_weights=times, avoided=mask)
    loss, grad, pred = physics(p, q, None, w, np.array([0]), **args)
    finite = np.array(
        [
            (
                physics(p, q, None, w + row * 1e-6, np.array([0]), **args)[0]
                - physics(p, q, None, w - row * 1e-6, np.array([0]), **args)[0]
            )
            / 2e-6
            for row in np.eye(3)
        ]
    ).T
    np.testing.assert_allclose(grad, finite, atol=2e-7, rtol=1e-5)
    state = physics.model.evaluate(w[0])
    np.testing.assert_allclose(pred[0, 0, 0], state.nominal, atol=1e-14)
    np.testing.assert_allclose(pred[0, 0, 1:], state.temporal, atol=1e-14)
    import torch

    target = physics.torch("cpu")
    tw = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    tl, tg, tp = target(
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(q, dtype=torch.float32),
        None,
        tw,
        torch.tensor([0]),
        time_weights=torch.tensor(times, dtype=torch.float32),
        avoided=torch.tensor(mask, dtype=torch.float32),
    )
    numerical_autograd = torch.autograd.grad(tl.sum(), tw, retain_graph=True)[0]
    np.testing.assert_allclose(
        tg.detach(), numerical_autograd.detach(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(tl.detach(), loss, atol=2e-6)
    np.testing.assert_allclose(tg.detach(), grad, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(tp.detach(), pred, atol=2e-6)


def test_physical_objective_is_not_normalized_linear_response():
    from fragrance_ai.recommender.exposure_normalization import normalized_exposure

    items = materials()
    physics = NonlinearDoseObjective(items, {}, 15.0)
    profiles = np.array([i.vector() for i in items])[None, None]
    weights = np.array([[0.2, 0.3, 0.5]])
    physical, _ = physics.predict(profiles[0], weights[0])
    responses = np.vstack((physics.model.gain, physics.model.coefficients.mean(0)))[
        None
    ]
    linear, _ = normalized_exposure(profiles, responses, weights)
    assert np.max(np.abs(physical - linear[0])) > 0.01


def test_sparse_contraction_preserves_dense_prediction_and_inactive_entry_gradients():
    from dataclasses import replace

    source = materials()
    items = [replace(source[i % 3], ingredient_id=f"sparse-{i}") for i in range(41)]
    physics = NonlinearDoseObjective(items, {}, 15.0, draws=16)
    p = np.array([i.vector() for i in items])[None]
    w = np.zeros(41)
    w[[0, 7, 13]] = [0.2, 0.3, 0.5]
    actual, _ = physics.predict(p, w)
    m = physics.model
    a = w * m.coefficients / (m.base_moles + w @ m.total_moles)
    response = a**0.55 / (1 + a**0.55) * m.transport
    suppressed = response / (
        1 + m.suppression[:, None, None] * (response @ m.interaction.T)
    )
    raw = np.einsum("kti,hid->kthd", suppressed, p)
    dense = (raw / raw.sum(-1, keepdims=True)).mean(0).transpose(1, 0, 2)
    np.testing.assert_allclose(actual[:, 1:], dense, atol=2e-15)
    q = np.broadcast_to(np.roll(actual, 2, axis=-1), (1, *actual.shape))
    args = {"time_weights": np.array([[0, 0.2, 0.2, 0.2, 0.2, 0.2]])}
    _, gradient, _ = physics(p[None], q, None, w[None], np.array([0]), **args)
    assert np.isfinite(gradient).all() and np.any(np.abs(gradient[0, w == 0]) > 0)
