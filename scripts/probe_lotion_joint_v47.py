"""Bounded joint oil/recipe SQP diagnostic in the unchanged design envelope."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def joint_step(state, weights, oil, *, radius=5.):
    from fragrance_ai.recommender.lotion_numerics import conditioned_linprog
    n = len(weights)
    r,dr,profiles,targets = (state[k] for k in ('responses','response_derivative','profiles','targets'))
    count,ns = len(r),state['n_slack']
    q = state['assessments'](weights)[0]/100.
    denominator = r@weights
    dden = dr@weights
    dp = (dr*weights)@profiles
    times,axes = np.nonzero(targets > 0)
    derivative_residual = dp[times,axes]-targets[times,axes]*dden[times]
    residual = sparse.hstack([state['residual_rows'],sparse.csr_matrix(np.r_[derivative_residual,-derivative_residual][:,None]),
                             sparse.csr_matrix((2*ns,1))],format='csr')
    doutside = np.sum(dp*(targets == 0),axis=1)
    overlap = sparse.hstack([
        sparse.csr_matrix(state['outside']-2*(1-q)*r),state['slack_sums'],
        sparse.csr_matrix((doutside-2*(1-q)*dden)[:,None]),
        sparse.csr_matrix((2*denominator)[:,None])],format='csr')
    negative = (state['avoid_vectors']@profiles.T-(1-q))*r
    avoidance = sparse.hstack([sparse.csr_matrix(negative),sparse.csr_matrix((count,ns)),
        sparse.csr_matrix((np.sum(state['avoid_vectors']*dp,axis=1)-(1-q)*dden)[:,None]),
        sparse.csr_matrix(denominator[:,None])],format='csr')
    fixed_derivative = np.r_[0.,-(state['physical_derivative']@weights)/state['physical_scale']]
    if state['request'].maximum_modeled_uptake_mg_cm2 is not None:
        fixed_derivative = np.r_[fixed_derivative,
            (state['uptake_derivative']@weights)/state['request'].maximum_modeled_uptake_mg_cm2]
    fixed = sparse.hstack([state['fixed_rows'],sparse.csr_matrix(fixed_derivative[:,None]),
                          sparse.csr_matrix((len(fixed_derivative),1))],format='csr')
    matrix = sparse.vstack([residual,overlap,avoidance,fixed],format='csr')
    rhs = np.r_[np.zeros(2*ns+2*count),state['fixed_rhs']]
    result = conditioned_linprog(linprog,np.r_[np.zeros(n+ns+1),-1.],A_ub=matrix,b_ub=rhs,
        A_eq=sparse.hstack([state['a_eq'],sparse.csr_matrix((state['a_eq'].shape[0],2))]),b_eq=state['eq_rhs'],
        bounds=state['bounds']+[(max(10.-oil,-radius),min(30.-oil,radius)),(0.,1.)],scales=np.ones(n+ns+2))
    report = {'solver_status':int(result.status),'baseline_score':100*q,'radius':radius,'predicted_gain':None}
    if result.status != 0 or not result.success or result.x is None:
        return None,None,report
    proposal = np.maximum(0.,result.x[:n]); proposal /= proposal.sum()
    next_oil = oil+float(result.x[-2])
    report.update(oil_delta=float(result.x[-2]),predicted_gain=float(100*result.x[-1]))
    return proposal,next_oil,report


def main():
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest,build_estimated_lotion_inputs
    from fragrance_ai.recommender import lotion_optimizer as module
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--brief',action='append',required=True)
    p.add_argument('--oil',type=float,default=20.)
    args = p.parse_args()
    if args.output.exists(): p.error('new evidence file required')
    paths = [Path(__file__),ROOT/'fragrance_ai/recommender/lotion_optimizer.py',
             ROOT/'fragrance_ai/recommender/lotion_estimation.py',ROOT/'fragrance_ai/recommender/lotion.py',
             ROOT/'fragrance_ai/recommender/lotion_numerics.py',ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json']
    source = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    catalog = read_catalog(ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    original = module.linprog
    class BasisReady(Exception): pass
    def at_oil(brief,oil,basis_only=False,seed=None):
        request,scenarios,_ = build_estimated_lotion_inputs(LotionEstimateRequest(brief=brief,
            registry_pool='conditional_research',max_risk_tier=2),catalog,oil_base_percent=oil)
        state = {}
        def capture(objective,**kwargs):
            frame = inspect.currentframe().f_back.f_back
            if frame.f_code.co_name == '_optimize_lotion_transport':
                state.update(frame.f_locals)
                if basis_only: raise BasisReady()
            return original(objective,**kwargs)
        module.linprog = capture
        result = None
        try:
            result = module.optimize_lotion(request,catalog,transport_scenarios=scenarios,
                _transport_only=True,reuse_transport_basis=True,_incumbent_recipe=seed)
        except BasisReady:
            pass
        finally:
            module.linprog = original
        return state,result
    rows = []
    for brief in args.brief:
        started = time.perf_counter()
        oil = args.oil
        state,baseline = at_oil(brief,oil)
        by_id = {r['ingredient_id']:r['concentrate_percent']/100 for r in baseline['recipe'] or baseline['closest_candidate']}
        weights = np.array([by_id.get(i.ingredient_id,0.) for i in state['pool']])
        score = state['assessments'](weights)[0]
        trials = []
        for step in range(3):
            if score+1e-8 >= 95.: break
            lower,upper = max(10.,oil-.01),min(30.,oil+.01)
            low,_ = at_oil(brief,lower,True)
            high,_ = at_oil(brief,upper,True)
            assert [i.ingredient_id for i in low['pool']] == [i.ingredient_id for i in high['pool']] == [i.ingredient_id for i in state['pool']]
            raw = lambda s: s['physical']/s['thresholds'] if s['all_thresholds'] else s['physical']
            state['response_derivative'] = (raw(high)-raw(low))/(upper-lower)/state['response_scale'][:,None]
            state['physical_derivative'] = (high['physical']-low['physical'])/(upper-lower)
            state['uptake_derivative'] = (np.asarray(high['uptake_blocks'])-np.asarray(low['uptake_blocks']))/(upper-lower)
            improved = False
            for radius in (5.,1.):
                candidate,new_oil,report = joint_step(state,weights,oil,radius=radius)
                if candidate is None:
                    trials.append(report)
                    continue
                actual,_ = at_oil(brief,new_oil,True)
                valid = bool(actual['valid'](candidate))
                actual_score = actual['assessments'](candidate)[0]
                report['linearized_candidate_actual_score'] = actual_score
                # Correct the neglected oil*recipe cross term by re-solving
                # recipe weights at the proposed oil ratio, with current exact
                # physics and the same constraints. Reuse its transport basis.
                seed = [{'ingredient_id':i.ingredient_id,'concentrate_percent':float(w*100)}
                    for i,w in zip(actual['pool'],candidate) if w > 0] if valid else None
                refined,refit = at_oil(brief,new_oil,seed=seed)
                if refit['score'] > actual_score:
                    lookup = {r['ingredient_id']:r['concentrate_percent']/100 for r in refit['recipe'] or refit['closest_candidate']}
                    candidate = np.array([lookup.get(i.ingredient_id,0.) for i in refined['pool']])
                    actual = refined
                    valid = bool(actual['valid'](candidate))
                    actual_score = actual['assessments'](candidate)[0]
                report.update(oil=new_oil,actual_score=actual_score,valid=valid,accepted=valid and actual_score > score+1e-7)
                trials.append(report)
                if report['accepted']:
                    state,weights,oil,score = actual,candidate,new_oil,actual_score
                    improved = True
                    break
            if not improved: break
        row = {'brief':brief,'initial_oil':args.oil,'baseline_score':baseline['score'],
            'selected_oil':oil,'score':score,'passed95':score+1e-8 >= 95.,'trials':trials,'seconds':time.perf_counter()-started}
        if score > baseline['score']+1e-7:
            # Fresh transport at the proposed formula, independent of all LP
            # response matrices. Keep every timepoint and scenario.
            fresh_scores = []
            for basis in state['basis_requests']:
                supplied = {r.ingredient_id:r for r in basis.materials}
                selected = [i for i,w in zip(state['pool'],weights) if w > 0]
                material = [supplied[i.ingredient_id].model_copy(update={'concentrate_percent':float(w*100)})
                    for i,w in zip(state['pool'],weights) if w > 0]
                result = module.simulate_lotion(basis.model_copy(update={'materials':material}),module.IngredientCatalog(selected))
                for target,point in zip(state['prepared']['evaluation_targets'],result['temporal_profile'][1:]):
                    fresh_scores.append(module.compare_profiles(target['target_profile'],point['scent_profile'],avoided=target['avoided']).score)
            row['fresh_score'] = min(fresh_scores)
            row['fresh_passed95'] = min(fresh_scores)+1e-8 >= 95.
            row['formula'] = [{'ingredient_id':i.ingredient_id,'concentrate_percent':float(w*100)}
                for i,w in zip(state['pool'],weights) if w > 0]
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'formula'}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha for path,sha in source.items())
    args.output.write_text(json.dumps({'scope':'bounded_joint_design_probe_not_full400',
        'source_sha256':source,'source_unchanged':True,'rows':rows},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
