from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('width',[96,128])
def test_wider_recurrent_state_preserves_previous_controls_before_training(width):
    torch = pytest.importorskip('torch')
    from fragrance_ai.research.aligned_autoregressive_network import AlignedAutoregressiveCell
    from fragrance_ai.research.autoregressive_capacity import initialize_from_controller
    torch.manual_seed(821)
    old = AlignedAutoregressiveCell()
    with torch.no_grad():
        old.controls.weight.normal_(std=.2)
    wide = AlignedAutoregressiveCell(hidden_size=width)
    initialize_from_controller(wide,old.state_dict())
    old_state = torch.zeros(7,48)
    new_state = torch.zeros(7,width)
    for _ in range(5):
        value = torch.randn(7,88)
        old_state = old.cell(value,old_state)
        new_state = wide.cell(value,new_state)
        torch.testing.assert_close(wide.controls(new_state),old.controls(old_state),rtol=1e-5,atol=1e-6)
    wide.controls(new_state).sum().backward()
    assert wide.controls.weight.grad[:,48:].abs().sum()>0


@pytest.mark.parametrize('width',[48,96,128])
def test_dynamic_width_cpu_export_matches_torch(width):
    torch = pytest.importorskip('torch')
    from fragrance_ai.research.aligned_autoregressive_network import AlignedAutoregressiveCell
    from fragrance_ai.recommender.aligned_autoregressive import rollout
    from tests.test_constrained_autoregressive_v74 import problem
    torch.manual_seed(823)
    model = AlignedAutoregressiveCell(hidden_size=width).eval()
    values = problem(11,146,2)
    arrays = {'autoregressive.'+k:v.detach().numpy() for k,v in model.state_dict().items()}
    actual,_ = rollout(arrays,*values,steps=3)
    with torch.no_grad():
        expected,_ = model(*[torch.as_tensor(v) for v in values],steps=3)
    np.testing.assert_allclose(actual,expected.numpy(),atol=2e-5,rtol=2e-5)


def test_nested_work_budgets_are_request_local_and_do_not_change_targets():
    from fragrance_ai.recommender.search_budget import ACTIVE,allowance,governed
    observed = []
    @governed('perfume')
    def inner():
        observed.append(id(ACTIVE.get()))
        assert allowance('cone',8)==24.
        return SimpleNamespace(calculated_profile_similarity=94.7,score_contract={'target':95.})
    @governed('perfume')
    def outer():
        observed.append(id(ACTIVE.get()))
        return inner()
    result = outer()
    assert observed[0]==observed[1] and ACTIVE.get() is None
    assert result.score_contract['target']==95.
    assert result.score_contract['search_work_budget']['score_threshold_changed'] is False
    assert result.score_contract['search_work_budget']['verified_best_scores']['returned_profile']==94.7


def test_work_budget_restores_context_on_failure():
    from fragrance_ai.recommender.search_budget import ACTIVE,governed
    @governed('body_lotion')
    def fails():
        raise ValueError('expected')
    with pytest.raises(ValueError):
        fails()
    assert ACTIVE.get() is None


def test_budget_clock_can_stop_work_without_saying_a_formula_is_infeasible(monkeypatch):
    import fragrance_ai.recommender.search_budget as module
    clock = [100.]
    monkeypatch.setattr(module,'monotonic',lambda:clock[0])
    budget = module.SearchBudget('perfume',total_seconds=20.,started=100.)
    token = module.ACTIVE.set(budget)
    try:
        assert module.allowance('physical_polish',8)==18.
        clock[0] = 119.
        assert module.allowance('physical_polish',8)==1.
        clock[0] = 121.
        assert module.exhausted() and budget.report()['budget_exhausted']
        assert 'infeasible' not in budget.report()
    finally:
        module.ACTIVE.reset(token)
