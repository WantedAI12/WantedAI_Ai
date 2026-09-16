import numpy as np
import pytest

from fragrance_ai.recommender.emulsion_science import normal_log_bins
from fragrance_ai.recommender.full_reference_guidance import agreement_gradient
from fragrance_ai.recommender.formulation_core import forward_arrays
from fragrance_ai.research.formulation_network import FormulationNetwork, export_arrays


def test_goal_changes_policy_deficit_not_fixed_formula_odor():
    import torch

    torch.manual_seed(761)
    network = FormulationNetwork(
        1599, 450, 292, 21, blend_outputs=109, aqueous_outputs=18
    ).eval()
    rng = np.random.default_rng(11)
    molecules = rng.normal(size=(2, 3, 1599)).astype(np.float32)
    molecules[1] = molecules[0]
    masses = np.tile([0.2, 0.3, 0.5], (2, 1)).astype(np.float32)
    context = np.zeros((2, 64), np.float32)
    context[:, 0] = 1
    context[:, 12] = 2
    context[:, 63] = np.log10(0.15)
    context[0, 25] = 1
    context[1, 27] = 1
    context[:, 45] = 1
    si = np.zeros((2, 0), np.int64)
    sv = np.zeros((2, 0, 12), np.float32)
    with torch.no_grad():
        output = network(
            *[torch.from_numpy(x) for x in (molecules, masses, context, si, sv)]
        )
    result = forward_arrays(export_arrays(network), molecules, masses, context, si, sv)
    for key in result:
        np.testing.assert_allclose(
            result[key], output[key].numpy(), atol=2e-5, rtol=2e-5
        )
        if key != "revision":
            np.testing.assert_array_equal(result[key][0], result[key][1])
    np.testing.assert_array_equal(
        result["revision"], context[:, 25:44] - context[:, 44:63]
    )


def test_reference_full_profile_derivative_has_no_missing_axes():
    rng = np.random.default_rng(12)
    basis = rng.dirichlet(np.ones(146), size=(4, 10))
    target = rng.dirichlet(np.ones(146), size=10)
    strengths = np.array([1.0, 0.1, 0.8, 1.0])
    mask = np.zeros_like(target)
    mask[:, 7] = 1
    function = agreement_gradient(basis, strengths, target, mask)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    score, grad = function(x)
    finite = np.array(
        [
            (function(x + row * 1e-6)[0] - function(x - row * 1e-6)[0]) / 2e-6
            for row in np.eye(4)
        ]
    )
    assert 0 <= score <= 100
    np.testing.assert_allclose(grad, finite, atol=1e-6, rtol=1e-5)


def test_lognormal_integrates_irregular_bins_not_point_densities():
    from scipy.special import ndtr

    centers = np.array([-3.0, -0.2, 0.1, 2.0])
    values = np.exp(normal_log_bins(centers, np.array([0.0]), np.array([1.0])))[0]
    edges = np.r_[-np.inf, (centers[:-1] + centers[1:]) / 2, np.inf]
    np.testing.assert_allclose(values, np.diff(ndtr(edges)), atol=1e-15)
    assert values.sum() == pytest.approx(1.0)


def test_reference_gradient_accepts_arraylike_strengths_and_rejects_absent_goal():
    basis = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    function = agreement_gradient(basis, [1.0, 1.0], [[0.5, 0.5]], [[0.0, 0.0]])
    assert function([0.5, 0.5])[0] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="normalized positive targets"):
        agreement_gradient(basis, [1.0, 1.0], [[0.0, 0.0]], [[0.0, 0.0]])


@pytest.mark.parametrize(
    "bad", [None, [], "invalid", {"water": True}, {"water": float("nan")}]
)
def test_observed_request_malformed_percentages_are_validation_errors(bad):
    from pydantic import ValidationError
    from fragrance_ai.platform.observed_formulation_inputs import ObservedLiquidRequest

    with pytest.raises(ValidationError):
        ObservedLiquidRequest(reference_protocol="reference", ingredient_percent=bad)


def test_annotation_missingness_is_never_returned_as_an_odor_label():
    from types import SimpleNamespace
    from fragrance_ai.recommender.formulation_core import SYSTEM_VERSION
    from fragrance_ai.recommender.observed_formulation import predict_pair

    core = SimpleNamespace(
        version=SYSTEM_VERSION,
        sha256="a" * 64,
        features=lambda _: np.zeros((2, 1599)),
        forward=lambda *args: {"blend": np.zeros((1, 4))},
        manifest={
            "blend_labels": ["", "No odor group found for these", "floral", "odorless"]
        },
    )
    result = predict_pair(core, "C", "CC")
    assert set(result["labels"]) == {"floral", "odorless"}
    assert set(result["non_odor_annotation_metadata_scores"]) == {
        "missing_note_field",
        "unclassified_odor_group",
    }
    assert result["annotation_missingness_is_not_odorlessness"] is True
