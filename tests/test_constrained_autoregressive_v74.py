import numpy as np
import pytest

from fragrance_ai.recommender.constrained_autoregressive import (
    capped_simplex,
    project_constraints,
    features,
    rollout,
)


def problem(n=9, dimensions=19, heads=1):
    rng = np.random.default_rng(7414 + n)
    p = rng.uniform(0.01, 1, (2, heads, n, dimensions)).astype(np.float32)
    p /= p.sum(-1, keepdims=True)
    q = rng.uniform(0.01, 1, (2, heads, 3, dimensions)).astype(np.float32)
    q /= q.sum(-1, keepdims=True)
    r = rng.uniform(0.1, 2, (2, 3, n)).astype(np.float32)
    w = np.zeros((2, n), np.float32)
    w[:, :3] = [0.2, 0.3, 0.5]
    lower = np.zeros_like(w)
    upper = np.full_like(w, 0.6)
    prices = rng.uniform(10, 100, w.shape).astype(np.float32)
    budget = np.full(2, 75.0, np.float32)
    latent = rng.normal(0, 0.1, (2, n, 256)).astype(np.float32)
    return latent, p, q, r, w, np.array([0, 1]), lower, upper, prices, budget


def test_capped_projection_mass_bounds_and_invalid_inputs():
    v = np.array([[2.0, -0.5, 0.3, 0.2]])
    l = np.array([[0.0, 0.05, 0.0, 0.1]])
    u = np.array([[0.4, 0.5, 0.4, 0.2]])
    w = capped_simplex(v, l, u)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-12)
    assert np.all(w >= l) and np.all(w <= u)
    for bad in (np.full_like(v, np.nan), np.full_like(v, np.inf)):
        with pytest.raises(ValueError):
            capped_simplex(bad, l, u)
    with pytest.raises(ValueError):
        capped_simplex(v, l, u * 0.1)


def test_price_repair_and_infeasible_budget():
    v = np.array([[0.9, 0.05, 0.05]])
    l, u = np.zeros_like(v), np.ones_like(v)
    prices = np.array([[100.0, 10.0, 30.0]])
    w = project_constraints(v, l, u, prices, np.array([35.0]))
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-12)
    assert (w * prices).sum() <= 35.0 + 1e-10
    with pytest.raises(ValueError, match="budget"):
        project_constraints(v, l, u, prices, np.array([9.0]))


def test_capped_projection_backward_matches_finite_difference():
    torch = pytest.importorskip("torch")
    from fragrance_ai.research.constrained_autoregressive_network import _CappedSimplex

    value = torch.tensor(
        [[0.12, 0.4, 0.32, 0.05]], dtype=torch.double, requires_grad=True
    )
    lower, upper = torch.zeros_like(value), torch.full_like(value, 0.6)
    assert torch.autograd.gradcheck(
        lambda x: _CappedSimplex.apply(x, lower, upper), (value,), atol=1e-5
    )


@pytest.mark.parametrize(
    "n,dimensions,heads", [(9, 19, 1), (31, 146, 2), (2048, 19, 1)]
)
def test_cpu_export_parity_and_large_sparse_candidate_pool(n, dimensions, heads):
    torch = pytest.importorskip("torch")
    from fragrance_ai.research.constrained_autoregressive_network import (
        ConstrainedAutoregressiveCell,
    )

    torch.set_num_threads(1)
    torch.manual_seed(74)
    values = problem(n, dimensions, heads)
    model = ConstrainedAutoregressiveCell().eval()
    arrays = {
        "autoregressive." + k: v.detach().numpy() for k, v in model.state_dict().items()
    }
    actual, report = rollout(arrays, *values)
    with torch.no_grad():
        expected, _ = model(*[torch.as_tensor(x) for x in values])
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-5)
    latent, p, q, r, w, product, lower, upper, prices, budget = values
    initial = project_constraints(w, lower, upper, prices, budget)
    _, before, _ = features(
        p,
        q,
        r,
        initial,
        initial * 0,
        product,
        0,
        upper > lower,
        lower,
        upper,
        prices,
        budget,
    )
    _, after, _ = features(
        p,
        q,
        r,
        actual,
        actual * 0,
        product,
        0,
        upper > lower,
        lower,
        upper,
        prices,
        budget,
    )
    assert np.all(after <= before + 1e-6)
    np.testing.assert_allclose(actual.sum(-1), 1.0, atol=1e-7)
    assert np.all(actual >= lower - 1e-10) and np.all(actual <= upper + 1e-10)
    assert np.all((actual * prices).sum(-1) <= budget + 1e-6)
    assert report["caps_and_price_in_each_neural_step"]


def test_smooth_worst_case_gradient_matches_finite_difference():
    _, p, q, r, w, product, lower, upper, prices, budget = problem()
    p, q, r = [v.astype(np.float64) for v in (p, q, r)]
    w = np.full_like(w, 1 / w.shape[-1], dtype=np.float64)
    mask = upper > lower

    def loss(value):
        _, _, pred = features(
            p, q, r, value, value * 0, product, 0, mask, lower, upper, prices, budget
        )
        tv = 0.5 * np.abs(pred - q).sum(-1)
        cosine = (pred * q).sum(-1) / (
            np.linalg.norm(pred, axis=-1) * np.linalg.norm(q, axis=-1)
        )
        losses = np.stack((tv, 1 - cosine), -1).reshape(2, -1)
        return np.log(np.exp(64 * losses).sum(-1)) / 64

    numerical = np.zeros_like(w)
    for i in range(w.shape[-1]):
        delta = np.zeros_like(w)
        delta[:, i] = 1e-6
        numerical[:, i] = (loss(w + delta) - loss(w - delta)) / 2e-6
    f, _, _ = features(
        p, q, r, w, w * 0, product, 0, mask, lower, upper, prices, budget
    )
    np.testing.assert_allclose(
        f[:, :, 16], numerical / np.abs(numerical).max(-1, keepdims=True), atol=1e-6
    )


def test_feedback_does_not_stall_on_one_depleted_coordinate():
    from fragrance_ai.recommender.autoregressive_refinement import ResidualFeedback
    feedback=ResidualFeedback(target=95.)
    feedback.observe([.2,.2,.6],[.5,-.5],50.)
    feedback.observe([0.,.25,.75],[.3,-.3],70.)
    proposed=feedback.proposal()
    assert proposed is not None
    assert np.all(proposed>=0)
    assert proposed.sum()==pytest.approx(1.)
    assert np.abs(proposed-[0.,.25,.75]).sum()<=.15+1e-10
