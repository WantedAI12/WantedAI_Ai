import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fragrance_ai.research.mixture_mlp import DistributionResidual, MixtureMLP  # noqa: E402
from fragrance_ai.research.mixture_mlp_numpy import NumpyMixtureMLP  # noqa: E402
from fragrance_ai.research.physmix_comparison import PairComparison, parameter_counts  # noqa: E402
from tests.test_physmix_comparison_v83 import network  # noqa: E402


def inputs():
    torch.manual_seed(8401)
    return torch.randn(2, 5, 256), torch.rand(2, 5), torch.randn(2, 5, 256), torch.rand(2, 5), torch.zeros(2, 64)


def test_old_control_initialization_is_preserved_exactly():
    _, arrays = network()
    torch.manual_seed(99)
    previous = PairComparison(arrays, "capacity_control").eval()
    torch.manual_seed(99)
    current = MixtureMLP(arrays, "pooled_mlp").eval()
    for name, value in previous.state_dict().items():
        torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(previous(*inputs()), current(*inputs()), rtol=0, atol=0)


def test_parameter_matched_arms_have_the_same_initial_prediction():
    _, arrays = network()
    reference = MixtureMLP(arrays, "pooled_mlp").eval()
    count = parameter_counts(reference)["interaction"]
    for mode in MixtureMLP.MODES:
        model = MixtureMLP(arrays, mode).eval()
        model.head.load_state_dict(reference.head.state_dict())
        assert abs(parameter_counts(model)["interaction"] / count - 1) < .01
        torch.testing.assert_close(model(*inputs()), reference(*inputs()))


@pytest.mark.parametrize("mode", MixtureMLP.MODES)
def test_cpu_export_and_cached_forward_match_and_pair_is_symmetric(mode):
    _, arrays = network()
    model = MixtureMLP(arrays, mode).eval()
    model.gate.data.fill_(.7)
    a, wa, b, wb, c = inputs()
    with torch.no_grad():
        expected = model(a, wa, b, wb, c)
        cached = model.forward_precomputed((a, wa, *model.backbone(a, wa, c)), (b, wb, *model.backbone(b, wb, c)))
    portable = NumpyMixtureMLP({k: v.detach().numpy() for k, v in model.state_dict().items()}, mode)
    actual = portable(*[x.numpy() for x in (a, wa, b, wb, c)])
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected, cached, rtol=0, atol=0)
    torch.testing.assert_close(expected, model(b, wb, a, wa, c))


@pytest.mark.parametrize("covariance", [False, True])
def test_distribution_is_order_zero_and_row_split_invariant_with_correct_dose_gradient(covariance):
    torch.manual_seed(8402)
    block = DistributionResidual(covariance=covariance).double()
    e = torch.randn(1, 3, 256, dtype=torch.double)
    w = torch.tensor([[.2, .3, .5]], dtype=torch.double, requires_grad=True)
    expected = block.moments(e, w)
    torch.testing.assert_close(block.moments(e[:, [2, 0, 1]], w[:, [2, 0, 1]]), expected)
    ghost = torch.cat((e, torch.full((1, 1, 256), 1000., dtype=torch.double)), 1)
    torch.testing.assert_close(block.moments(ghost, torch.cat((w, torch.zeros(1, 1, dtype=torch.double)), 1)), expected)
    torch.testing.assert_close(block.moments(e[:, [0, 1, 1, 2]], torch.tensor([[.2, .1, .2, .5]], dtype=torch.double)), expected)
    assert torch.autograd.gradcheck(lambda value: block.moments(e, value), w, fast_mode=True)


def test_prepool_transform_distinguishes_a_mean_diagonal_variance_collision():
    # Same marginal first/second moments, opposite cross-channel covariance.
    torch.manual_seed(8403)
    block = DistributionResidual(covariance=True).double()
    a = torch.zeros(1, 2, 256, dtype=torch.double)
    b = torch.zeros_like(a)
    a[0, :, :2] = torch.tensor([[1., 1.], [-1., -1.]])
    b[0, :, :2] = torch.tensor([[1., -1.], [-1., 1.]])
    w = torch.full((1, 2), .5, dtype=torch.double)
    torch.testing.assert_close(a.mean(1), b.mean(1))
    torch.testing.assert_close(a.square().mean(1), b.square().mean(1))
    assert torch.linalg.vector_norm(block.moments(a, w) - block.moments(b, w)) > 1e-3


@pytest.mark.parametrize("mode", MixtureMLP.MODES)
def test_runtime_rejects_invalid_or_empty_amounts(mode):
    _, arrays = network()
    model = MixtureMLP(arrays, mode)
    portable = NumpyMixtureMLP({k: v.detach().numpy() for k, v in model.state_dict().items()}, mode)
    for invalid in (0., -1., float("nan"), float("inf")):
        a, wa, b, wb, c = inputs()
        wa[:] = invalid
        with pytest.raises(ValueError):
            model(a, wa, b, wb, c)
        with pytest.raises(ValueError):
            portable(*[x.numpy() for x in (a, wa, b, wb, c)])


def test_mixture_holdout_excludes_test_identities_from_training_and_selection():
    from fragrance_ai.research.r2_physsim import MixturePair
    from scripts.compare_mixture_mlp_v84 import mixture_partitions
    from scripts.compare_physmix_v83 import mixture_key, pair_key
    pairs = [MixturePair((str(i),), (str(j),), .5, f"{i}-{j}") for i in range(16) for j in range(i + 1, 16)]
    pairs.append(MixturePair(pairs[0].mixture_b, pairs[0].mixture_a, .5, "reverse"))
    splits = mixture_partitions(pairs, 42)
    assert {i for row in splits for i in row["test"]} == set(range(len(pairs)))
    assert all(len(row["both_unseen"]) > 0 for row in splits)
    for row in splits:
        seen = {mixture_key(m) for i in row["train"] + row["validation"] for m in (pairs[i].mixture_a, pairs[i].mixture_b)}
        train_pairs = {pair_key(pairs[i]) for i in row["train"] + row["validation"]}
        for i in row["test"]:
            assert any(mixture_key(m) not in seen for m in (pairs[i].mixture_a, pairs[i].mixture_b))
            assert pair_key(pairs[i]) not in train_pairs
