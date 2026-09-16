from dataclasses import replace
import pytest

from fragrance_ai.research.replay_search import (
    Observation, Policy, UnsupportedReplay, choose, improve_policy, replay, run_live,
)


def world():
    return {branch: [Observation(branch, i, -(floor + 1 / (i + 5)), .1, terminal=i == 30)
                     for i in range(1, 31)] for branch, floor in (("a", .1), ("b", .4), ("c", .2))}


def test_replay_and_live_share_identical_decisions():
    history = world()
    policy = Policy(warmup=4, patience=8)
    cursors = dict.fromkeys(history, 0)

    def advance(branch):
        obs = history[branch][cursors[branch]]
        cursors[branch] += 1
        return obs

    assert run_live(tuple(history), advance, policy, budget=45) == replay(history, policy, budget=45)


def test_future_outcomes_cannot_change_a_prefix_decision():
    first = world()
    second = {b: [o if o.step <= 4 else replace(o, score=1000.) for o in row] for b, row in first.items()}
    policy = Policy(warmup=4)
    a, b = replay(first, policy, budget=12), replay(second, policy, budget=12)
    assert a == b
    prefix = {name: tuple(row[:4]) for name, row in first.items()}
    assert choose(policy, prefix, tuple(prefix)) == choose(policy, prefix, tuple(prefix))


def test_missing_recorded_children_are_unsupported_not_fake_success():
    data = {"a": [Observation("a", 1, .3, .1)]}
    with pytest.raises(UnsupportedReplay):
        replay(data, Policy(kind="round_robin"), budget=2)
    data["a"][0] = replace(data["a"][0], terminal=True)
    assert replay(data, Policy(kind="round_robin"), budget=2)["work"] == 1


def test_failed_observations_cannot_win_and_best_anchor_is_preserved():
    data = {"a": [Observation("a", 1, .8, .1), Observation("a", 2, None, .1, status="numeric_failure", terminal=True)],
            "b": [Observation("b", 1, .2, .1, terminal=True)]}
    result = replay(data, Policy(kind="round_robin"), budget=4)
    assert result["best"]["score"] == .8
    assert result["work"] == 3
    assert result["seconds"] == pytest.approx(.3)


def test_terminated_best_model_still_prevents_spending_on_inferior_frontiers():
    prefix = {"done": tuple(Observation("done", i, .9, .1, terminal=i == 5) for i in range(1, 6)),
              "weak": tuple(Observation("weak", i, .2, .1) for i in range(1, 6))}
    old = Policy(revision=1, warmup=4, exploration=0., patience=12)
    new = Policy(revision=2, warmup=4, exploration=0., patience=12)
    assert choose(old, prefix, ("weak",))[0] == "weak"
    assert choose(new, prefix, ("weak",))[0] is None


def test_policy_learning_includes_incumbent_and_respects_frozen_quality_band():
    policy, record = improve_policy([world()], budget=45)
    assert record["evaluated_policies"] == 109
    assert policy.digest == record["selection"]["policy_sha256"]
    assert record["selection"]["score"] >= max(r["score"] for r in record["candidates"]) - .001
    assert any(r["policy"]["kind"] == "round_robin" for r in record["candidates"])


@pytest.mark.parametrize("kwargs", [{"step": 0}, {"seconds": -1}, {"seconds": float("nan")},
                                    {"score": float("inf")}, {"score": None}, {"status": "error"}])
def test_invalid_scored_attempts_are_rejected(kwargs):
    values = {"branch": "a", "step": 1, "score": .4, "seconds": .1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        Observation(**values)


def test_nested_policy_worlds_exclude_every_outer_selection_and_test_index():
    from fragrance_ai.research.r2_physsim import MixturePair
    from scripts.compare_mixture_mlp_v84 import mixture_partitions
    from scripts.train_replay_mlp_v85 import inner_split
    pairs = [MixturePair((str(i),), (str(j),), .5, str((i, j))) for i in range(12) for j in range(i + 1, 12)]
    for outer in mixture_partitions(pairs, 851):
        inner = inner_split(pairs, outer["train"], 852)
        assert set(inner["train"] + inner["validation"]) == set(outer["train"])
        assert not set(inner["train"] + inner["validation"]) & set(outer["test"] + outer["validation"])


def test_live_training_is_independent_of_branch_interleaving():
    torch = pytest.importorskip("torch")
    import numpy as np
    from fragrance_ai.research.mixture_replay_training import PairBank, TrainingBranch, validation_ensemble
    from fragrance_ai.research.physmix_comparison import FrozenAggregation
    from tests.test_physmix_comparison_v83 import network
    _, arrays = network()
    bank = object.__new__(PairBank)
    bank.device, bank.arrays = torch.device("cpu"), arrays
    e, w, context = torch.randn(8, 4, 256), torch.rand(8, 4), torch.zeros(8, 64)
    with torch.no_grad():
        bank.cache = (e, w, *FrozenAggregation(arrays)(e, w, context))
    bank.datasets = {"snitz": (torch.arange(16) % 8, (torch.arange(16) + 3) % 8, torch.linspace(.2, .8, 16))}
    args = (bank, "set_mlp", list(range(12)), list(range(12, 16)), 855)
    direct, interleaved = TrainingBranch(*args), TrainingBranch(*args)
    unrelated = TrainingBranch(bank, "normalized_mlp", list(range(12)), list(range(12, 16)), 856)
    direct.advance()
    direct.advance()
    interleaved.advance()
    unrelated.advance()
    interleaved.advance()
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, interleaved.model.state_dict()[name], rtol=0, atol=0)
    predictions = np.array([[.2, .4, .6, .8], [.8, .6, .4, .2]])
    weights, _ = validation_ensemble(predictions, predictions[0])
    assert np.all(weights >= 0) and weights.sum() == pytest.approx(1.)
    assert np.mean((weights @ predictions - predictions[0])**2) <= 1e-12


@pytest.mark.parametrize("bad", [[], [0., 0.], [-1., 1.], [float("nan"), 1.], [True, 1.], ["1", 1.]])
def test_relative_composition_rejects_invalid_amounts(bad):
    pytest.importorskip("rdkit")
    from fragrance_ai.recommender.replay_mixture import ReplayMixtureModel
    with pytest.raises(ValueError):
        ReplayMixtureModel._composition(["CCO", "CC"], bad)


def test_relative_composition_is_duplicate_split_and_unit_invariant():
    pytest.importorskip("rdkit")
    import numpy as np
    from fragrance_ai.recommender.replay_mixture import ReplayMixtureModel
    g, w = ReplayMixtureModel._composition(["CCO", "CC"], [20., 80.])
    h, v = ReplayMixtureModel._composition(["CC", "OCC", "CCO"], [.8, .1, .1])
    assert g == h
    np.testing.assert_allclose(w, v, rtol=0, atol=0)


def test_identity_cache_reuses_validation_but_never_reuses_amounts():
    pytest.importorskip("rdkit")
    import numpy as np
    from fragrance_ai.recommender.replay_mixture import ReplayMixtureModel, _canonical_graph
    _canonical_graph.cache_clear()
    _, first = ReplayMixtureModel._composition(["CCO", "CC"], [20., 80.])
    _, second = ReplayMixtureModel._composition(["CCO", "CC"], [80., 20.])
    assert _canonical_graph.cache_info().misses == 2
    assert _canonical_graph.cache_info().hits == 2
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError):
        ReplayMixtureModel._composition(["CCO", "CC"], [-1., 1.])
