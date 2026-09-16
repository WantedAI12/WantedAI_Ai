"""Exact-loss parity, constrained inversion and no fabricated target improvements."""
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.nonlinear_inverse import NonlinearDoseObjective, profile_loss
from fragrance_ai.recommender.reference_inverse import nominal_seed, polish, smooth_profile_loss, quantize_recipe


def inputs():
    items = IngredientCatalog.load_builtin().ingredients[:3]
    p = np.array([[[.8,.15,.05],[.1,.8,.1],[.1,.2,.7]],
                  [[.7,.2,.1],[.15,.75,.1],[.12,.18,.7]]])
    engine = NonlinearDoseObjective(items,{},15.,draws=16)
    target,_ = engine.predict(p,np.array([.2,.5,.3]))
    tw = np.array([0.,.1,.1,.3,.25,.25])
    return items,p,engine,target,tw,np.zeros_like(target)


def test_smooth_loss_gradient_and_exact_upper_approximation():
    _,p,engine,q,tw,mask = inputs()
    pred,_ = engine.predict(p,np.array([.45,.2,.35]))
    smooth,grad = smooth_profile_loss(pred,q,tw,mask)
    assert smooth >= profile_loss(pred,q,tw,mask)[0]
    rng = np.random.default_rng(19)
    direction = rng.normal(size=pred.shape)*.01
    step = 1e-6
    finite = (smooth_profile_loss(pred+step*direction,q,tw,mask)[0]
        -smooth_profile_loss(pred-step*direction,q,tw,mask)[0])/(2*step)
    assert np.sum(grad*direction)==pytest.approx(finite,rel=1e-5,abs=1e-7)


@pytest.mark.parametrize('weights', [[.45,.2,.35],[.5,0.,.5]])
def test_hill_coordinate_gradient_including_exact_zero_entry(weights):
    _,p,engine,q,tw,mask = inputs()
    w = np.array(weights)
    z = w**.55
    loss,grad,pred = engine(p[None],q[None],None,w[None],[0],time_weights=tw[None],
        avoided=mask[None],loss_function=smooth_profile_loss,gradient_coordinates='hill_power')
    def f(value):
        pred,_ = engine.predict(p,value**(1/.55))
        return smooth_profile_loss(pred,q,tw,mask)[0]
    for i in range(3):
        delta = np.eye(3)[i]*1e-7
        finite = (f(z+delta)-f(z-delta))/(2e-7) if z[i]>0 else (f(z+delta)-f(z))/1e-7
        assert grad[0,i]==pytest.approx(finite,rel=.002,abs=1e-5)


def test_nominal_lp_solves_unequal_gains_and_caps_without_a_wrong_mass_basis():
    p = np.stack([np.eye(3)]*2)
    w_true = np.array([.2,.3,.5])
    gains = np.array([1.,2.,4.])
    q = np.stack([w_true*gains/(w_true@gains)]*2)
    w,report = nominal_seed(p,q,gains,np.array([.1,0,0]),np.array([.4,.4,.8]),
        np.array([20.,40.,60.]),60.)
    assert w is not None and report['nominal_tv_seed_score']==pytest.approx(100.)
    np.testing.assert_allclose(w,w_true,atol=1e-8)


def test_infeasible_caps_do_not_produce_a_formula_or_success_claim():
    p = np.stack([np.eye(3)]*2)
    w,report = nominal_seed(p,np.ones((2,3))/3,np.ones(3),np.zeros(3),np.full(3,.2),np.ones(3),10.)
    assert w is None and report['status']!=0


def test_physical_polish_improves_the_unchanged_exact_objective():
    items,p,engine,q,tw,mask = inputs()
    initial = np.array([.5,.2,.3])
    w,report = polish(items,{},15.,p,q,tw,mask,initial,np.zeros(3),np.ones(3),
        np.array([10.,20.,30.]),30.,draws=16,maxiter=50)
    assert w is not None and report['selected_exact_score'] > report['starting_exact_score']+1
    pred,_ = engine.predict(p,w)
    assert report['selected_exact_score']==pytest.approx(100*(1-profile_loss(pred,q,tw,mask)[0]))
    assert report['selected_exact_score']>99
    assert w.sum()==pytest.approx(1.) and np.all(w>=0)


def test_full_reference_head_switch_is_not_a_changed_target():
    from fragrance_ai.recommender.adaptive_pyramid import check_adaptive_response
    from tests.test_odor_space_v77 import brief
    b = brief({'citrus':1.})
    targets = [{'profiles':[[.8,.2],[.6,.4]],'avoided':[]}]*2
    def assessment(head):
        row = {'target_profile':targets[0]['profiles'][head], 'avoided_mass':0.,'score':90.,
            'minutes':0,'phase':'opening','weight':1.}
        return {'nominal':dict(row),'temporal':[dict(row)],'reference_assessment':{
            'endpoints':['a','b'],'intent':{'targets':targets},
            'predicted_profiles':[[[.7,.3],[.7,.3]],[[.5,.5],[.5,.5]]]}}
    old,new = assessment(0),assessment(1)
    point = SimpleNamespace(total_relative_intensity=1.,relative_to_opening_intensity_percent=100.)
    twin = SimpleNamespace(temporal_points=[point])
    item = IngredientCatalog.load_builtin().ingredients[0]
    line = SimpleNamespace(ingredient_id=item.ingredient_id,concentrate_percent=100.,pyramid=item.pyramid)
    policy = {'intensity_range':None,'diffusion_range':None,'signal_retention_floor':.9,'long_lasting':False}
    assert check_adaptive_response(b,old,new,twin,twin,[line],{item.ingredient_id:item},policy)==[]
    new['reference_assessment']['intent']['targets'] = [{'profiles':[[.9,.1],[.6,.4]],'avoided':[]}]*2
    assert 'odor_target_changed' in check_adaptive_response(b,old,new,twin,twin,[line],{item.ingredient_id:item},policy)


@pytest.mark.parametrize('w',[[.4,.2,.4],[.5,0.,.5]])
def test_signal_hill_derivative_and_forward_parity(w):
    _,p,engine,_,_,_ = inputs()
    w = np.asarray(w)
    z = w**.55
    signal,gradient = engine.signal_value_gradient(w)
    _,cache = engine.predict(p,w)
    np.testing.assert_allclose(signal,cache[3].sum(-1).mean(0),rtol=1e-13)
    for i in range(len(w)):
        delta = np.eye(len(w))[i]*1e-7
        right = engine.signal_value_gradient((z+delta)**(1/.55))[0]
        left = engine.signal_value_gradient((z-delta)**(1/.55))[0] if w[i]>0 else signal
        finite = (right-left)/(2e-7 if w[i]>0 else 1e-7)
        np.testing.assert_allclose(gradient[:,i],finite,rtol=.002,atol=1e-7)


def test_joint_rounding_preserves_caps_mass_and_zero_row_removal():
    w = np.array([.40000049,.59999949,.00000002])
    result = quantize_recipe(w,np.zeros(3),np.array([.40000049,1,1]),np.array([1,2,3]),3.)
    np.testing.assert_allclose(result,[.4,.6,0.],atol=1e-12)
    assert result.sum()==1. and np.all(result<=np.array([.40000049,1,1]))


def test_joint_rounding_respects_explicit_note_and_price_rows():
    w = np.array([.1+.4e-6,.2-.4e-6,.7])
    result = quantize_recipe(w,np.zeros(3),np.ones(3),np.array([10.,20.,30.]),26.,
        bands=[(np.array([1,1,0]),.3,.3)])
    assert result is not None and result[:2].sum()==pytest.approx(.3)
    assert result@np.array([10.,20.,30.])<=26.+1e-8


def test_infeasible_preservation_does_not_return_a_high_scoring_seed():
    import json
    items,p,engine,q,tw,mask = inputs()
    w,report = polish(items,{},15.,p,q,tw,mask,np.array([.2,.5,.3]),np.zeros(3),np.ones(3),
        np.ones(3),10.,draws=16,maxiter=2,signal_floor=np.full(5,1e10))
    assert w is None and report['selected_exact_score'] is None
    json.dumps(report,allow_nan=False)


def test_lotion_fractional_refinement_uses_all_scenarios_and_exact_score():
    from fragrance_ai.recommender.lotion_fractional_inverse import refine
    from fragrance_ai.recommender.corrective_profile_search import profile_derivatives
    p = np.stack([np.eye(3)]*2)
    r = np.array([[1.,2.,3.],[3.,2.,1.]])
    truth = np.array([.2,.5,.3])
    q = np.stack([r*truth/(r@truth)[:,None]]*2)
    mask = np.zeros((2,3))
    bg = np.ones((2,3))/3
    w,report = refine(shapes=p,wanted=q,responses=r,fixed=np.ones((1,3)),rhs=np.array([2.]),
        equality=np.ones((1,3)),equality_rhs=np.ones(1),bounds=np.array([[0.,1.]]*3),
        initial=np.array([.4,.2,.4]),avoided=mask,background=bg,target=.95,max_seconds=10.)
    assert w is not None and report['target_met']
    values,_,contrast,_ = profile_derivatives(p,q,r,w,mask,bg)
    assert report['selected_score']==pytest.approx(100*values.min())
    assert report['all_material_columns']==3 and np.all(contrast>1e-8)
    assert w.sum()==pytest.approx(1.)


def test_failed_column_expansion_retains_an_earlier_feasible_incumbent(monkeypatch):
    from scipy import sparse
    from scipy.optimize import OptimizeResult
    import fragrance_ai.recommender.lotion_reference_search as module
    count = [0]
    def solve(objective,parameters,diagnostics):
        count[0] += 1
        diagnostics['linear_solves'] += 1
        if count[0]>1:
            return OptimizeResult(success=False,status=1,x=None,message='time limit')
        x = np.zeros(len(objective))
        x[0] = 1
        return OptimizeResult(success=True,status=0,x=x,fun=0.,
            ineqlin=SimpleNamespace(marginals=np.ones(1)),eqlin=SimpleNamespace(marginals=np.zeros(1)))
    monkeypatch.setattr(module,'_solve_with_time_recovery',solve)
    row = np.zeros((1,66))
    row[0,64:] = 1.
    diagnostics = {'linear_solves':0,'maximum_working_materials':0,'full_pool_pricing_passes':0}
    result = module.column_linprog(np.zeros(66),material_count=66,initial_columns=[0],
        column_order=list(range(66)),diagnostics=diagnostics,A_ub=sparse.csr_matrix(row),b_ub=np.ones(1),
        A_eq=sparse.csr_matrix(np.ones((1,66))),b_eq=np.ones(1),bounds=[(0.,1.)]*66)
    assert result.success and result.x[0]==1. and result.x.sum()==1.
    assert not result.full_pool_optimality_certified
    assert diagnostics['feasible_incumbent_retained_after_expansion_failure']


@pytest.mark.parametrize('kind',['profile','avoidance'])
@pytest.mark.parametrize('weights',[[.2,.4,.4],[.5,0.,.5]])
def test_pointwise_hill_constraints_have_analytic_finite_derivatives(kind,weights):
    from fragrance_ai.recommender.reference_inverse import point_value_gradient
    _,p,engine,q,_,mask = inputs()
    mask[...,1] = 1.
    w = np.asarray(weights)
    z = w**.55
    values,gradient = point_value_gradient(engine,p,q,mask,w,kind=kind)
    for i in range(3):
        delta = np.eye(3)[i]*1e-7
        right = point_value_gradient(engine,p,q,mask,(z+delta)**(1/.55),kind=kind)[0]
        left = point_value_gradient(engine,p,q,mask,(z-delta)**(1/.55),kind=kind)[0] if w[i]>0 else values
        finite = (right-left)/(2e-7 if w[i]>0 else 1e-7)
        np.testing.assert_allclose(gradient[...,i],finite,rtol=.004,atol=1e-5)


def test_rounded_incumbent_is_repaired_without_weakening_constraints():
    items,p,_,q,tw,mask = inputs()
    w,report = polish(items,{},15.,p,q,tw,mask,np.array([.2,.5,.300001]),np.zeros(3),np.ones(3),
        np.ones(3),10.,draws=16,maxiter=2)
    assert report['roundoff_mass_repaired'] and w is not None
    assert w.sum()==pytest.approx(1.,abs=1e-10)
