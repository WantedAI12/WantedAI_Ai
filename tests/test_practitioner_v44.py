from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

from fragrance_ai.recommender.accord_trials import accord_trials
from fragrance_ai.recommender.lotion_conic import solve_full_profile
from fragrance_ai.recommender.lotion_optimizer import _compact_profile_residuals, optimize_lotion
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe, build_estimated_lotion_inputs
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture


def test_accord_trials_preserve_internal_ratios_mass_bounds_and_note_budget():
    weights = np.array([60., 20., 10., 10.])
    profiles = np.array([[1, 0], [1, 0], [0, 1], [0, 1.]])
    trials = list(accord_trials(weights, profiles, np.ones(4), [.5, .5], np.zeros(4), np.ones(4)*100))
    assert trials
    assert any(abs((p@profiles)[0]-50) < abs((weights@profiles)[0]-50) for p in trials)
    for p in trials:
        assert p.sum() == pytest.approx(100)
        assert np.all(p >= 0) and np.all(p <= 100)
        assert p[0] == pytest.approx(3*p[1])
        assert p[2] == pytest.approx(p[3])
    assert not list(accord_trials(weights, profiles, np.ones(4), [.5,.5], np.zeros(4),
        np.ones(4)*100, budget_groups=['top','top','base','base']))


def test_product_response_changes_direction_not_a_fixed_note_ratio():
    w, p = np.array([.5,.5]), np.eye(2)
    forward = list(accord_trials(w, p, [10,1], [.5,.5], [0,0], [1,1]))
    backward = list(accord_trials(w, p, [1,10], [.5,.5], [0,0], [1,1]))
    assert forward and all(x[0] < .5 for x in forward)
    assert backward and all(x[0] > .5 for x in backward)


def test_required_floor_and_caps_apply_to_entire_blocks():
    trials = list(accord_trials([.8,.2], np.eye(2), [1,1], [.5,.5], [.7,0], [1,.25]))
    assert trials and len(trials) <= 32
    assert all(x[0] >= .7-1e-12 and x[1] <= .25+1e-12 for x in trials)


@pytest.mark.parametrize('scale', [1., 1e-12])
def test_cone_recomputes_cosine_and_preserves_mass_and_cap(scale):
    pytest.importorskip('clarabel')
    p, t, r = np.eye(2), np.array([[.5,.5]]), np.array([[scale,scale]])
    rows, sums, outside = _compact_profile_residuals(p,t,r)
    ns = sums.shape[1]
    w, report = solve_full_profile(profiles=p, targets=t, responses=r, target_score=99.,
        residual_rows=rows, slack_sums=sums, outside=outside, avoid_vectors=np.zeros_like(t),
        fixed_rows=sparse.csr_matrix((0,2+ns)), fixed_rhs=np.array([]),
        a_eq=sparse.csr_matrix([[1.,1.,0.,0.]]), eq_rhs=np.array([1.]),
        bounds=[(0,.6),(0,1)]+[(0,None)]*ns, objective=np.array([0.,1.,0.,0.]))
    assert report['status'] in ('Solved','AlmostSolved')
    assert w is not None and w.sum() == pytest.approx(1.,abs=1e-8)
    assert 0 <= w[0] <= .6+1e-8
    cosine = w@t[0]/np.linalg.norm(w)/np.linalg.norm(t[0])
    overlap = 1-.5*np.abs(w/w.sum()-t[0]).sum()
    assert min(cosine,overlap) >= .99-1e-8


def test_block_recovery_changes_actual_lotion_after_lp_failure(monkeypatch):
    monkeypatch.setattr('fragrance_ai.recommender.lotion_optimizer.linprog',
        lambda *a, **k: SimpleNamespace(status=1, success=False))
    monkeypatch.setattr('fragrance_ai.recommender.lotion_conic.solve_full_profile',
        lambda **k: (None, {'solver_calls':0, 'status':'unavailable'}))
    value, catalog = fixture()
    result = optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result['score'] > result['baseline_score']+30
    assert result['accord_refinement']['accepted'] > 0
    assert result['search_incomplete']  # A failed LP is not hidden by a better candidate.
    assert result['score'] == min(x['score'] for x in result['timepoint_assessments'])
    assert not result['manufacturing_approved']
    assert result['conic_recovery']['status'] == 'experimental_solver_disabled'


@pytest.mark.parametrize('values', [[True], [.5,.5], [0], [3.1], [float('nan')]])
def test_dose_trial_inputs_are_bounded_unique_finite(values):
    with pytest.raises(ValueError):
        LotionEstimateRequest(brief='citrus woody', dose_trials={'concentrations_percent':values})


def test_real_dose_context_recomputed_and_input_unchanged():
    from fragrance_ai.recommender.catalog import IngredientCatalog
    catalog = IngredientCatalog.load_builtin()
    a = LotionEstimateRequest(brief='citrus woody')
    before = a.model_dump()
    first, _, _ = build_estimated_lotion_inputs(a, catalog)
    second, _, _ = build_estimated_lotion_inputs(a.model_copy(update={'fragrance_concentration_percent':.25}), catalog)
    assert a.model_dump() == before
    assert first.simulation.parameter_context_id != second.simulation.parameter_context_id
    assert second.simulation.application_context.fragrance_concentration_percent == .25


def test_dose_selection_preserves_failed_trials_and_does_not_mutate_request(monkeypatch):
    # Selection-unit test only; actual chemistry is exercised separately.
    def estimate(request, *a, **k):
        dose = request.fragrance_concentration_percent
        if dose == .1:
            raise ValueError('explicit concentration conflict')
        return {'status':'research_candidate_only', 'score': { .5:80., .25:81., 1.:79. }[dose],
                'profile_target_met':False, 'solver_calls':1, 'perception_model':{}}
    monkeypatch.setattr('fragrance_ai.recommender.lotion_estimation._estimate_lotion_recipe', estimate)
    request = LotionEstimateRequest(brief='citrus woody', dose_trials={'concentrations_percent':[.1,.25,1.]})
    before = request.model_dump()
    result = estimate_lotion_recipe(request, None)
    assert request.model_dump() == before
    assert result['score'] == 81 and result['dose_design']['selected_dose_percent'] == .25
    assert len(result['dose_design']['evaluated_variants']) == 3
    assert len(result['dose_design']['variant_errors']) == 1
    assert result['dose_design']['fixed_dose_score'] == 80
    assert not result['profile_target_met']
