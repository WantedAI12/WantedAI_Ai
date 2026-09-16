"""Joint full-profile and background feasibility over every material column."""

from time import monotonic

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .corrective_profile_search import profile_derivatives
from .lotion_numerics import conditioned_linprog, response_variable_scales


def solve_reference_feasibility(*, shapes, wanted, responses, fixed, rhs, equality,
                                equality_rhs, bounds, target, avoided, background,
                                maximum_seconds=40., maximum_rounds=8):
    start = monotonic()
    s, q, r, a, b, e, f, mask, bg = map(lambda x:np.asarray(x,float),
        (shapes,wanted,responses,fixed,rhs,equality,equality_rhs,avoided,background))
    h,n,d = s.shape
    t = len(r)
    if (q.shape!=(h,t,d) or r.shape!=(t,n) or mask.shape!=(t,d) or bg.shape!=(h,d)
            or a.shape!=(len(b),n) or e.shape!=(len(f),n) or np.shape(bounds)!=(n,2)
            or not 0<target<=1 or maximum_seconds<=0
            or any(not np.isfinite(v).all() for v in (s,q,r,a,b,e,f,mask,bg))
            or np.any(s<0) or np.any(q<0) or np.any(r<0)
            or not np.allclose(q.sum(-1),1.,atol=1e-7)):
        raise ValueError('finite complete aligned reference problem required')
    ns=h*t*d
    repeated=np.tile(r,(h,1))
    residual=np.vstack([(s[i]-q[i,j]).T*r[j] for i in range(h) for j in range(t)])
    absolute=sparse.vstack((sparse.hstack((sparse.csr_matrix(residual),-sparse.eye(ns))),
                           sparse.hstack((-sparse.csr_matrix(residual),-sparse.eye(ns)))),format='csr')
    groups=.5*sparse.kron(sparse.eye(h*t),np.ones((1,d)),format='csr')
    overlap=sparse.hstack((-(1-target)*sparse.csr_matrix(repeated),groups),format='csr')
    avoid=np.einsum('hnd,td->htn',s,mask).reshape(h*t,n)*repeated-(1-target)*repeated
    unit=q/np.linalg.norm(q,axis=-1,keepdims=True)
    from .reference_discrimination import contrast_direction
    contrast=contrast_direction(q,bg[:,None])
    discrimination=(2e-8-np.einsum('hnd,htd->htn',s,contrast).reshape(h*t,n))*repeated
    matrix=sparse.vstack((absolute,overlap,sparse.hstack((sparse.csr_matrix(np.vstack((avoid,discrimination,a))),
                         sparse.csr_matrix((2*h*t+len(b),ns))))),format='csr')
    limits=np.r_[np.zeros(2*ns+3*h*t),b]
    eq=sparse.hstack((sparse.csr_matrix(e),sparse.csr_matrix((len(f),ns))),format='csr')
    full_bounds=[*bounds,*[(0.,None)]*ns]
    scales=response_variable_scales(r,ns)
    # Feasibility first. No cheap generic/background recipe can win because
    # price was optimized before the requested identity constraints.
    costs=np.r_[np.zeros(n),np.ones(ns)]
    report={'version':'joint-reference-feasibility/v89','all_material_columns':n,
            'all_descriptor_axes':d,'background_constraints_in_initial_problem':True,
            'linear_solves':0,'cosine_cuts':0,'formal_infeasibility_certificate':False}
    lo,hi=np.asarray(bounds).T
    best,best_score=None,-1.
    for _ in range(maximum_rounds):
        remaining=maximum_seconds-(monotonic()-start)
        if remaining<=0:
            break
        def solver(objective,**kwargs):
            return linprog(objective,**{**kwargs,'options':{**kwargs.get('options',{}),'time_limit':remaining}})
        result=conditioned_linprog(solver,costs,A_ub=matrix,b_ub=limits,A_eq=eq,b_eq=f,bounds=full_bounds,scales=scales)
        report['linear_solves']+=1
        report['solver_status']=int(result.status)
        if result.x is None:
            break
        w=result.x[:n]
        if (not np.isfinite(w).all() or np.any(w<lo-1e-10) or np.any(w>hi+1e-10)
                or np.any(a@w>b+1e-9) or np.any(np.abs(e@w-f)>1e-8) or np.any(r@w<=0)):
            break
        values,_,margins,_=profile_derivatives(s,q,r,w,mask,bg)
        score=float(values.min())
        passed=score+1e-10>=target and bool(np.all(margins>1e-8))
        if score>best_score:
            best,best_score=w.copy(),score
        if passed:
            return w,{**report,'status':'fresh_joint_feasibility_pass','score':100*score,
                      'minimum_background_margin':float(margins.min()),'seconds':monotonic()-start}
        pred=np.einsum('tn,n,hnd->htd',r,w,s,optimize=True)/(r@w)[None,:,None]
        direction=pred/np.linalg.norm(pred,axis=-1,keepdims=True)
        cosine=(unit*direction).sum(-1)
        cuts=[]
        for i,j in zip(*np.where(cosine<target)):
            cuts.append(np.r_[(s[i]@(target*direction[i,j]-unit[i,j]))*r[j],np.zeros(ns)])
        if not cuts:
            break
        report['cosine_cuts']+=len(cuts)
        matrix=sparse.vstack((matrix,sparse.csr_matrix(np.asarray(cuts))),format='csr')
        limits=np.r_[limits,np.zeros(len(cuts))]
    return best,{**report,'status':'no_joint_pass_found','score':100*best_score if best is not None else None,
                 'seconds':monotonic()-start}
