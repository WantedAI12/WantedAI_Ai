import numpy as np
import pytest

from fragrance_ai.recommender.aligned_autoregressive import objective, rollout
from fragrance_ai.recommender.exposure_normalization import normalized_exposure
from tests.test_constrained_autoregressive_v74 import problem
from tests.test_formulation_core_v69 import fixture as shared_core_fixture

base_fixture = shared_core_fixture


@pytest.mark.parametrize("tiny", [1e-18, 1e-200])
def test_weak_positive_exposure_keeps_unit_shape(tiny):
    p = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    r = np.array([[[tiny, 1.0]]])
    w = np.array([[1.0, 0.0]])
    actual, _ = normalized_exposure(p, r, w)
    np.testing.assert_allclose(actual, [[[[1.0, 0.0]]]], atol=1e-14)
    with pytest.raises(ValueError, match="headspace"):
        normalized_exposure(p, r, w * 0)


def test_torch_normalization_and_gradient_match_weak_positive_exposure():
    torch = pytest.importorskip("torch")
    from fragrance_ai.recommender.exposure_normalization import (
        torch_normalized_exposure,
    )

    p = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.double)
    r = torch.tensor([[[1e-18, 2e-18]]], dtype=torch.double)
    w = torch.tensor([[0.4, 0.6]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: torch_normalized_exposure(p, r, x)[0], (w,)
    )


def test_latest_anchor_recovers_improving_partial_recipe():
    from types import SimpleNamespace
    from fragrance_ai.recommender.adaptive_pyramid import refinement_blends

    baseline = [
        SimpleNamespace(ingredient_id="a", concentrate_percent=80.0),
        SimpleNamespace(ingredient_id="b", concentrate_percent=20.0),
    ]
    proposal = {
        "weights_percent": {"a": 15.0, "b": 85.0},
        "anchor_weights_percent": {"a": 20.0, "b": 80.0},
    }
    results = dict(refinement_blends(baseline, proposal, 2, {"a", "b"}))
    assert results[0.25] == {"a": 18.75, "b": 81.25}
    assert proposal["anchor_weights_percent"] == {"a": 20.0, "b": 80.0}


def test_product_objective_matches_public_aggregation_and_avoidance():
    p = np.array([[[[1.0, 0.0], [0.0, 1.0]]]] * 2)
    r = np.array([[[1.0, 1.0], [1.0, 4.0], [4.0, 1.0]]] * 2)
    w = np.full((2, 2), 0.5)
    q = np.array([[[[0.5, 0.5], [0.2, 0.8], [0.2, 0.8]]]] * 2)
    time = np.array([[0.0, 0.8, 0.2]] * 2)
    loss, _, pred = objective(p, q, r, w, np.array([0, 1]), time_weights=time)
    from fragrance_ai.recommender.profile_match import compare_profiles

    point = np.array(
        [
            [
                compare_profiles(q[b, 0, t], pred[b, 0, t], dimensions=("a", "b")).score
                for t in range(3)
            ]
            for b in range(2)
        ]
    )
    assert loss[0] == pytest.approx(
        1 - min(point[0, 0], point[0, 1:] @ time[0, 1:]) / 100
    )
    assert loss[1] == pytest.approx(1 - point[1].min() / 100)
    avoid = np.zeros_like(q)
    avoid[:, :, :, 1] = 1
    avoided, _, _ = objective(
        p, q, r, w, np.array([0, 1]), time_weights=time, avoided=avoid
    )
    assert np.all(avoided >= loss)


@pytest.mark.parametrize(
    "n,dimensions,heads", [(9, 19, 1), (31, 146, 2), (2048, 19, 1)]
)
def test_retrained_cpu_operator_matches_torch_and_accepted_loss(n, dimensions, heads):
    torch = pytest.importorskip("torch")
    from fragrance_ai.research.aligned_autoregressive_network import (
        AlignedAutoregressiveCell,
    )

    torch.set_num_threads(1)
    torch.manual_seed(75)
    model = AlignedAutoregressiveCell().eval()
    values = problem(n, dimensions, heads)
    arrays = {
        "autoregressive." + k: v.detach().numpy() for k, v in model.state_dict().items()
    }
    actual, report = rollout(arrays, *values, steps=3)
    with torch.no_grad():
        expected, trace = model(*[torch.as_tensor(v) for v in values], steps=3)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-5)
    _, p, q, r, _, product, lower, upper, prices, budget = values
    last, _, _ = objective(p, q, r, actual, product)
    np.testing.assert_allclose(last, trace[:, -1].numpy(), atol=1e-6)
    assert (np.diff(trace.numpy(), axis=1) <= 1e-7).all()
    np.testing.assert_allclose(actual.sum(-1), 1.0, atol=1e-7)
    assert np.all(actual >= lower - 1e-10) and np.all(actual <= upper + 1e-10)
    assert np.all((actual * prices).sum(-1) <= budget + 1e-5)
    assert "backtracked_steps" in report


def test_core_binds_aligned_schema_and_rejects_missing_refit_evidence(
    base_fixture,
    tmp_path,
):
    import hashlib
    import json
    from fragrance_ai.research.aligned_autoregressive_network import (
        AlignedAutoregressiveCell,
    )
    from fragrance_ai.recommender.formulation_core import FormulationCore

    path, _, _, arrays = base_fixture
    manifest = json.loads(path.read_text())
    model = AlignedAutoregressiveCell()
    arrays = {
        **arrays,
        **{
            "autoregressive." + k: v.detach().numpy()
            for k, v in model.state_dict().items()
        },
    }
    np.savez_compressed(tmp_path / "weights.npz", **arrays)
    manifest["schema"] = "shared-formulation-core/v75"
    manifest["weights"] = {
        "path": "weights.npz",
        "sha256": hashlib.sha256((tmp_path / "weights.npz").read_bytes()).hexdigest(),
    }
    manifest["autoregressive"] = {
        "base_heads_frozen_and_byte_identical": True,
        "default_steps": 8,
    }
    manifest["evaluation"]["gates"].update(
        dict.fromkeys(
            (
                "autoregressive_vs_initial",
                "autoregressive_vs_fixed_optimizer",
                "autoregressive_cpu_export",
                "frozen_backbone_identity",
                "autoregressive_constraints",
                "autoregressive_vs_v73",
            ),
            True,
        )
    )

    def load():
        raw = json.dumps(manifest).encode()
        target = tmp_path / "model.json"
        target.write_bytes(raw)
        return FormulationCore(target, hashlib.sha256(raw).hexdigest())

    assert load().version == "shared-formulation-core/v75"
    manifest["profile_refit"] = {
        "schema": "isolated-molecular-profile-refit/v75",
        "nonmolecular_parameter_arrays_exact": True,
        "source_disjoint_profile_improvement": True,
    }
    manifest["autoregressive"]["base_heads_frozen_and_byte_identical"] = False
    with pytest.raises(ValueError, match="gates"):
        load()
    manifest["evaluation"]["gates"].update(
        profile_refit_verified=True,
        autoregressive_refit_revalidated=True,
        frozen_backbone_identity=False,
    )
    assert load().version == "shared-formulation-core/v75"


def test_reference_profiles_expand_proposals_without_changing_requested_target(
    monkeypatch,
):
    from types import SimpleNamespace
    from tests.test_dose_refinement import setup_case
    from fragrance_ai.recommender.reference_profile_refinement import (
        reference_profile_seeds,
    )
    from fragrance_ai.recommender.fractional_transfers import TransferPhysics

    items, brief, _, _ = setup_case("floral")
    requested = dict(brief.target_profile)
    profiles = np.full((2, 146), 1 / 146)
    calls = []

    def proposal(items, p, q, r, w, **kwargs):
        calls.append((p.shape, q.shape, kwargs))
        return w / w.sum(), {"steps": 8}

    core = SimpleNamespace(
        version="shared-formulation-core/v75",
        sha256="a" * 64,
        autoregressive_proposal=proposal,
    )
    guide = SimpleNamespace(
        provider=SimpleNamespace(core=core),
        shapes=SimpleNamespace(
            prefetch=lambda x: None, shape=lambda i: profiles.copy()
        ),
    )
    bank = SimpleNamespace(
        parent_sha256=core.sha256,
        sha256="b" * 64,
        endpoints=[str(i) for i in range(146)],
        targets=lambda brief, rows: (
            [{"profiles": profiles.copy(), "avoided": []} for _ in rows],
            [],
        ),
    )
    original = {item.ingredient_id: w for item, w in zip(items, [35.0, 30.0, 35.0])}
    seeds, report = reference_profile_seeds(
        items, brief, TransferPhysics(items, {}, 15), original, {}, guide, bank=bank
    )
    assert len(seeds) == 2 and calls[0][0] == (2, 3, 146) and calls[0][1] == (2, 6, 146)
    assert brief.target_profile == requested
    assert report["legacy_requested_target_and_score_unchanged"]
    assert not report["target_fit_to_candidate_catalogue"]


def test_frontier_diagnostics_never_relax_actual_request_or_materials(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from tests.test_dose_refinement import setup_case
    from fragrance_ai.recommender import global_profile_search
    from fragrance_ai.recommender.profile_frontier import explain_profile_frontier

    items, brief, _, _ = setup_case("floral")
    before = asdict(brief), [asdict(item) for item in items]
    calls = []

    def solve(candidates, request, **kwargs):
        calls.append((candidates, request))
        return SimpleNamespace(certified_overlap_upper_score=80.0)

    monkeypatch.setattr(global_profile_search, "optimize_full_pool", solve)
    result = explain_profile_frontier(
        items,
        brief,
        {
            "status": "validated_fractional_cap_cost_upper",
            "upper_score": 70.0,
            "legacy_render_allowance_points": 1.0,
        },
    )
    assert result["status"] == "profile_geometry_or_target_expression_bottleneck"
    assert result["without_dose_caps_and_budget_upper_score"] == 81.0
    assert not result["actual_request_or_material_limits_changed"]
    assert before == (asdict(brief), [asdict(item) for item in items])
    assert all(item.max_concentrate_percent == 100.0 for item in calls[0][0])


def test_nonlinear_polish_uses_current_anchor_and_exact_guidance_floor(monkeypatch):
    from types import SimpleNamespace
    from tests.test_dose_refinement import setup_case
    from fragrance_ai.recommender import dose_refinement

    items, brief, _, policy = setup_case("floral")
    anchor = {
        item.ingredient_id: value for item, value in zip(items, [20.0, 50.0, 30.0])
    }
    seed = {item.ingredient_id: value for item, value in zip(items, [15.0, 55.0, 30.0])}
    calls = []
    guide = SimpleNamespace(
        enabled=True, evaluate=lambda w, ingredients, exact: {"score": 87.25}
    )

    def optimize(*args, **kwargs):
        calls.append((args, kwargs))
        return {"weights_percent": seed}

    monkeypatch.setattr(dose_refinement, "optimize_dose_support", optimize)
    result = dose_refinement.polish_neural_seed(
        items, brief, {}, seed, anchor, policy, {}, guidance=guide
    )
    assert result["weights_percent"] == seed
    assert calls[0][1]["guidance"] is guide
    assert calls[0][1]["minimum_guidance"] == 87.25
    baseline = dose_refinement.DoseModel(
        items, {}, brief.constraints.product_concentration_percent
    ).evaluate(np.array([0.2, 0.5, 0.3]))
    np.testing.assert_allclose(calls[0][0][4].nominal, baseline.nominal)
