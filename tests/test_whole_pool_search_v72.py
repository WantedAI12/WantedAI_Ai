from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from fragrance_ai.recommender.fractional_transfers import TransferPhysics, fractional_transfer_seeds
from fragrance_ai.recommender.lotion_conic import solve_full_profile
from fragrance_ai.recommender.corrective_profile_search import profile_derivatives
from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer
from fragrance_ai.recommender.formulation_guidance import SharedRecipeSession
from tests.test_dose_refinement import setup_case


@pytest.mark.parametrize('dose',[0., .000001, .01, .25])
def test_partial_transfer_matches_independent_central_recipe_physics(dose):
    items,brief,_,_=setup_case()
    receiver=replace(items[0],ingredient_id='receiver',name='receiver',profile={'green':1.},active_strength_percent=30.)
    items=[*items,receiver]
    model=TransferPhysics(items,{},brief.constraints.product_concentration_percent)
    original={0:.25,1:.4,2:.35}
    nominal,temporal,_=model.transfer(original,0,np.array([3]),np.array([dose]))
    weights={0:.25-dose,1:.4,2:.35,3:dose}
    selected=[items[i] for i,value in weights.items() if value > 0]
    percent=np.array([weights[i]*100 for i,value in weights.items() if value > 0])
    variant=ConstrainedFormulaOptimizer().variant_from_weights(selected,brief,percent)
    prepared=model.engine._prepare(variant[0],{i.ingredient_id:i for i in items},{})
    curves=model.engine._build_ingredient_temporal_profiles(prepared,model.engine._interaction_matrix(prepared))
    expected=[]
    for t in range(5):
        row=sum(curve.points[t].odor_contribution_percent/100*next(i for i in items if i.ingredient_id==curve.ingredient_id).vector() for curve in curves)
        expected.append(row/row.sum())
    np.testing.assert_allclose(temporal[0],expected,atol=3e-8,rtol=3e-7)
    expected_nominal=ConstrainedFormulaOptimizer._achieved_profile(percent,selected)
    np.testing.assert_allclose(nominal[0],expected_nominal,atol=1e-12,rtol=1e-12)


def test_low_cap_materials_are_available_as_partial_not_only_whole_replacements():
    items,brief,_,_=setup_case('floral')
    trace=replace(items[1],ingredient_id='trace',name='trace',max_concentrate_percent=.1,odor_impact=2000.)
    items=[*items,trace]
    seeds,report=fractional_transfer_seeds(items,brief,{}, {'top':35.,'heart':30.,'base':35.},{},maximum=5)
    assert report['receiver_count']==1 and report['dose_proposals_evaluated']>0
    assert seeds and any('trace' in seed for seed in seeds)
    for seed in seeds:
        assert sum(seed.values())==pytest.approx(100.)
        assert 0 < seed['trace'] <= .1+1e-10
        assert len(seed)>=3


def test_explicit_material_minima_survive_all_partial_transfers():
    items,brief,_,_=setup_case('floral')
    trace=replace(items[1],ingredient_id='trace',name='trace',max_concentrate_percent=.1,odor_impact=2000.)
    items=[*items,trace]
    seeds,_=fractional_transfer_seeds(items,brief,{}, {'top':35.,'heart':30.,'base':35.}, {'top':35.,'base':35.})
    assert all(seed['top']>=35. and seed['base']>=35. for seed in seeds)


def test_partial_neural_batch_agrees_with_each_full_formula_evaluation():
    rng=np.random.default_rng(7201)
    shapes=rng.uniform(.01,1,(6,2,5))
    shapes/=shapes.sum(-1,keepdims=True)
    items=[SimpleNamespace(ingredient_id=str(i),active_strength_percent=100./(i+1)) for i in range(6)]
    session=SharedRecipeSession.__new__(SharedRecipeSession)
    session.enabled,session.target=True,np.array([.7,.3,0.])
    session.projection=np.zeros((5,3))
    session.projection[:3,:3]=np.eye(3)
    session.avoided=[1]
    session.shapes=SimpleNamespace(prefetch=lambda rows:None,shape=lambda item:shapes[int(item.ingredient_id)])
    session.grid_calls,session.exact_calls=0,0
    session.provider=SimpleNamespace(core=SimpleNamespace(sha256='a'*64),endpoints=list('abcde'))
    amounts=np.array([.0001,.2,4.,50.])
    values=session.transfer_scores({'0':60.,'1':40.},items,'0',items[2:],amounts)
    expected=[session.evaluate([60.-dose,40.,dose],[items[0],items[1],receiver])['score'] for dose,receiver in zip(amounts,items[2:])]
    np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)


def cone_problem():
    h,t,n,d=2,2,300,3
    shapes=np.zeros((h,n,d))
    shapes[:,:150,0]=1.
    shapes[:,150:,1]=1.
    wanted=np.zeros((h,t,d))
    wanted[:,:,:2]=.5
    responses=np.ones((t,n))
    residual=np.vstack([(shapes[head]-wanted[head,when]).T*responses[when] for head in range(h) for when in range(t)])
    ns=h*t*d
    identity=sparse.eye(ns,format='csr')
    absolute=sparse.vstack((sparse.hstack((sparse.csr_matrix(residual),-identity)),sparse.hstack((-sparse.csr_matrix(residual),-identity))),format='csr')
    sums=sparse.csr_matrix((np.ones(ns),(np.repeat(np.arange(h*t),d),np.arange(ns))),shape=(h*t,ns))
    return dict(profiles=shapes,targets=wanted,responses=responses,target_score=99.,residual_rows=absolute,
        slack_sums=sums,outside=np.zeros((h*t,n)),avoid_vectors=np.zeros((h*t,d)),fixed_rows=sparse.csr_matrix((0,n+ns)),
        fixed_rhs=np.array([]),a_eq=sparse.csr_matrix([np.r_[np.ones(n),np.zeros(ns)]]),eq_rhs=np.array([1.]),
        bounds=[(0.,1.)]*n+[(0.,None)]*ns,objective=np.r_[np.arange(n)/n,np.zeros(ns)],
        background=np.array([[0.,0.,1.]]*h),initial_columns=list(range(64)),column_order=list(range(n)),time_limit=5.)


def test_cone_keeps_both_heads_and_finds_necessary_omitted_materials():
    pytest.importorskip('clarabel')
    args=cone_problem()
    weights,report=solve_full_profile(**args)
    assert weights is not None and report['status'] in ('Solved','AlmostSolved')
    assert report['material_columns_priced']==300
    assert report['maximum_working_materials']<300
    values,_,contrast,_=profile_derivatives(args['profiles'],args['targets'],args['responses'],weights,background=args['background'])
    assert values.min()>=.99-1e-8 and contrast.min()>1e-8
    assert weights.sum()==pytest.approx(1.,abs=1e-8)


def test_feasible_specific_candidate_survives_higher_nonspecific_intermediate(monkeypatch):
    clarabel=pytest.importorskip('clarabel')
    args=cone_problem()
    args['target_score']=95.
    args['profiles'][:,:64]=[.5,.49,.01]
    args['profiles'][:,64:]=[.47,.53,0.]
    args['background']=np.array([[.5,.49,.01]]*2)
    calls=[]

    def solver(_p,costs,*_values):
        # Both intermediates obey physical mass/bounds. The first is almost
        # the requested shape but indistinguishable from background (99).
        # The second is specific and passes the complete 95 gate (97).
        x=np.zeros(len(costs))
        x[0 if not calls else 64]=1.
        status='PrimalInfeasible' if not calls else 'MaxTime'
        calls.append(status)
        return SimpleNamespace(solve=lambda:SimpleNamespace(x=x,status=status,iterations=1,z=None))

    monkeypatch.setattr(clarabel,'DefaultSolver',solver)
    weights,report=solve_full_profile(**args)
    assert calls==['PrimalInfeasible','MaxTime']
    assert report['feasible_before_cost_optimum'] and report['solver_incomplete']
    values,_,contrast,_=profile_derivatives(args['profiles'],args['targets'],args['responses'],weights,background=args['background'])
    assert values.min() >= .95 and contrast.min()>1e-8
    assert weights[64]==pytest.approx(1.)


def test_chemical_descriptor_cache_keeps_values_and_native_profiles_independent():
    from fragrance_ai.research.conditional_profiles import molecule_features,_molecular_descriptor_tuple
    _molecular_descriptor_tuple.cache_clear()
    native={'profile':[1.]+[0.]*18,'odor_impact':1.}
    first=molecule_features('CCO',native)
    canonical,bits,physical=_molecular_descriptor_tuple.__wrapped__('CCO')
    assert first=={'canonical_smiles':canonical,'fingerprint_bits':list(bits),'physical':list(physical),'native':native}
    first['physical'][0]=-1.
    first['fingerprint_bits'].clear()
    changed={'profile':[0.,1.]+[0.]*17,'odor_impact':2.}
    second=molecule_features('CCO',changed)
    assert second['physical'][0]>0 and second['fingerprint_bits']
    assert second['native'] is changed
    assert _molecular_descriptor_tuple.cache_info().hits>=1
    with pytest.raises(ValueError,match='native profile'):
        molecule_features('CCO',{'profile':[1.]})
