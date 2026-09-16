"""All-column fractional residual refinement of the existing lotion model.

At a fixed loss lambda, each row N_ht(x)-lambda*D_t(x) is convex
piecewise linear. An epigraph LP minimizes the worst normalized residual.
Every accepted iterate is checked using the original normalized profiles,
including cosine, excluded notes and the existing background discrimination.
No primal optimum is presented as a formal infeasibility certificate.
"""
from time import monotonic
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .lotion_numerics import conditioned_linprog, response_variable_scales


def refine(*, shapes, wanted, responses, fixed, rhs, equality, equality_rhs,
           bounds, initial, avoided, background, target, max_seconds=12., max_rounds=4):
    from .corrective_profile_search import profile_derivatives
    start = monotonic()
    shapes,wanted,responses,fixed,rhs,equality,equality_rhs = map(lambda x:np.asarray(x,float),
        (shapes,wanted,responses,fixed,rhs,equality,equality_rhs))
    if shapes.ndim!=3:
        raise ValueError('full head/material/descriptor profiles required')
    h,n,d = shapes.shape
    t = len(responses)
    lo,hi = np.asarray(bounds,float).T
    if (wanted.shape!=(h,t,d) or responses.shape!=(t,n) or lo.shape!=(n,) or hi.shape!=(n,)
            or fixed.shape!=(len(rhs),n) or equality.shape!=(len(equality_rhs),n)
            or not 0<target<=1 or not max_seconds>0 or max_rounds<1
            or any(not np.isfinite(x).all() for x in (shapes,wanted,responses,fixed,rhs,equality,equality_rhs,lo,hi))
            or np.any(shapes<0) or np.any(wanted<0) or np.any(responses<0)
            or np.any(lo<0) or np.any(hi<lo)
            or not np.allclose(shapes[:,hi>0].sum(-1),1,atol=1e-7)
            or not np.allclose(wanted.sum(-1),1,atol=1e-7)):
        raise ValueError('finite normalized profiles and unchanged feasible constraint coordinates required')
    def valid(w):
        return (w is not None and w.shape==(n,) and np.isfinite(w).all() and np.all(w>=lo-1e-10)
            and np.all(w<=hi+1e-10) and np.all(fixed@w<=rhs+1e-9)
            and np.all(np.abs(equality@w-equality_rhs)<=1e-8) and np.all(responses@w>0))
    report = {'version':'lotion-fractional-residual/v81','all_material_columns':n,
        'all_descriptor_axes':d,'linear_solves':0,'accepted_steps':0,'iterations':[],
        'target_and_forward_model_unchanged':True,'formal_certificate':False}
    if not valid(initial):
        return None,{**report,'status':'feasible_incumbent_required'}
    best = np.array(initial,copy=True)
    def evaluate(w):
        values,_,contrast,_ = profile_derivatives(shapes,wanted,responses,w,avoided,background)
        score = float(values.min())
        return score,bool(np.all(contrast>1e-8))
    best_score,specific = evaluate(best)
    report['starting_score'] = 100*best_score
    # Descriptor residual slack variables retain all endpoints, not just the
    # endpoints requested by name. x remains the as-supplied mass fraction.
    ns = h*t*d
    delta = np.vstack([(shapes[a]-wanted[a,b]).T*responses[b] for a in range(h) for b in range(t)])
    residual = sparse.csr_matrix(delta)
    identity = sparse.eye(ns,format='csr')
    zeros = sparse.csr_matrix((ns,1))
    absolute = sparse.vstack((sparse.hstack((residual,-identity,zeros)),
        sparse.hstack((-residual,-identity,zeros))),format='csr')
    groups = sparse.kron(sparse.eye(h*t),np.ones((1,d)),format='csr')*.5
    fixed_rows = sparse.hstack((sparse.csr_matrix(fixed),sparse.csr_matrix((len(rhs),ns+1))),format='csr')
    equal = sparse.hstack((sparse.csr_matrix(equality),sparse.csr_matrix((len(equality_rhs),ns+1))),format='csr')
    repeated = np.tile(responses,(h,1))
    avoid_coeff = np.einsum('hnd,td->htn',shapes,avoided).reshape(h*t,n)*repeated
    full_bounds = [*map(tuple,bounds),*[(0.,None)]*ns,(None,None)]
    objective = np.r_[np.zeros(n+ns),1.]
    scale = np.r_[response_variable_scales(responses,ns),1.]
    columns = {'linear_solves':0,'maximum_working_materials':0,'full_pool_pricing_passes':0}
    report['column_search'] = columns
    for _ in range(max_rounds):
        remaining = max_seconds-(monotonic()-start)
        if remaining<=0 or best_score+1e-9>=target and specific:
            break
        level = 1-best_score
        denominators = np.maximum(repeated@best,1e-30)
        variation = sparse.hstack((-level*sparse.csr_matrix(repeated),groups,-denominators[:,None]),format='csr')
        avoidance = sparse.hstack((sparse.csr_matrix(avoid_coeff-level*repeated),
            sparse.csr_matrix((h*t,ns)),-denominators[:,None]),format='csr')
        def solve(costs,**kwargs):
            if n>256:
                from .lotion_reference_search import column_linprog
                return column_linprog(costs,material_count=n,initial_columns=np.flatnonzero(best>0).tolist(),
                    column_order=np.argsort(-best,kind='stable').tolist(),diagnostics=columns,**kwargs)
            return linprog(costs,**{**kwargs,'method':'highs-ds','options':{
                **kwargs.get('options',{}),'time_limit':min(5.,remaining)}})
        result = conditioned_linprog(solve,objective,
            A_ub=sparse.vstack((absolute,variation,avoidance,fixed_rows),format='csr'),
            b_ub=np.r_[np.zeros(2*ns+2*h*t),rhs],A_eq=equal,b_eq=equality_rhs,
            bounds=full_bounds,scales=scale)
        report['linear_solves'] += 1
        row = {'status':int(result.status),'success':bool(result.success)}
        report['iterations'].append(row)
        if not result.success or result.x is None or not valid(result.x[:n]):
            break
        proposed = np.maximum(result.x[:n],0.)
        previous_rank = (best_score+1e-9>=target and specific,best_score)
        anchor = best.copy()
        for fraction in (1.,.5,.25,.125):
            trial = anchor+fraction*(proposed-anchor)
            if not valid(trial):
                continue
            score,is_specific = evaluate(trial)
            if (score+1e-9>=target and is_specific,score)>(best_score+1e-9>=target and specific,best_score):
                best,best_score,specific = trial,score,is_specific
        row['fresh_score'] = 100*best_score
        if (best_score+1e-9>=target and specific,best_score)<=previous_rank:
            break
        report['accepted_steps'] += 1
    report.update(selected_score=100*best_score,target_met=bool(best_score+1e-9>=target and specific),
        seconds=monotonic()-start,status='verified_iterates_only')
    return best,report
