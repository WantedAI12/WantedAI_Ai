import numpy as np
import pytest
from fragrance_ai.recommender.constrained_autoregressive import capped_simplex


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_absent_materials_appear_when_reprojecting_a_feasible_recipe(dtype):
    rng = np.random.default_rng(761025)
    values = np.zeros((16, 3830), dtype=dtype)
    for row in values:
        row[:10] = rng.dirichlet(np.ones(10)).astype(dtype)
    lower = np.zeros_like(values)
    upper = np.ones_like(values)
    actual = capped_simplex(values, lower, upper)
    assert np.all(actual[:, 10:] == 0.0)
    np.testing.assert_allclose(actual.sum(-1), 1.0, atol=3e-16)
    import torch
    from fragrance_ai.research.constrained_autoregressive_network import _CappedSimplex

    tensor = _CappedSimplex.apply(
        torch.tensor(values), torch.tensor(lower), torch.tensor(upper)
    ).numpy()
    assert np.all(tensor[:, 10:] == 0.0)
    np.testing.assert_allclose(
        tensor, actual, atol=1e-7 if dtype == np.float32 else 1e-15
    )


def test_real_trace_dose_is_preserved_and_has_nonzero_learning_gradient():
    import torch
    from fragrance_ai.research.constrained_autoregressive_network import _CappedSimplex

    values = torch.tensor(
        [[1e-9, 0.4, 0.6 - 1e-9, 0.0]], dtype=torch.float64, requires_grad=True
    )
    output = _CappedSimplex.apply(
        values, torch.zeros_like(values), torch.ones_like(values)
    )
    output[0, 0].backward()
    assert output[0, 0].item() == pytest.approx(1e-9, rel=0, abs=1e-20)
    assert values.grad[0, 0].item() > 0.0
    assert output[0, -1] == 0.0


def test_cuda_training_projection_uses_the_same_operator_and_trace_gradient():
    import torch
    from fragrance_ai.research.constrained_autoregressive_network import _CappedSimplex
    if not torch.cuda.is_available():
        pytest.skip('CUDA training bridge requires a GPU')
    v=np.zeros((1,3830),np.float32);v[0,:4]=[1e-9,.2,.3,.5]
    low=np.zeros_like(v);high=np.ones_like(v)
    x=torch.tensor(v,device='cuda',requires_grad=True)
    y=_CappedSimplex.apply(x,torch.tensor(low,device='cuda'),torch.tensor(high,device='cuda'))
    expected=capped_simplex(v,low,high).astype(np.float32)
    np.testing.assert_array_equal(y.detach().cpu(),expected)
    y[0,0].backward()
    assert x.grad[0,0]>0
    assert torch.count_nonzero(y[0,4:])==0


def test_outside_simplex_projection_can_legitimately_add_required_mass():
    v = np.array([[0.2, 0.0, 0.0]])
    result = capped_simplex(v, np.zeros_like(v), np.ones_like(v))
    np.testing.assert_allclose(result, [[0.2 + 0.8 / 3, 0.8 / 3, 0.8 / 3]], atol=1e-14)


def test_projection_respects_tight_bounds_without_dropping_positive_traces():
    v = np.array([[1e-12, 0.3, 0.7 - 1e-12]])
    result = capped_simplex(
        v, np.array([[1e-12, 0.3, 0.0]]), np.array([[0.1, 0.3, 0.9]])
    )
    assert result[0, 0] >= 1e-12 and result[0, 1] == 0.3
    np.testing.assert_allclose(result.sum(-1), 1.0, atol=2e-16)
