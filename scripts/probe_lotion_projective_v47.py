"""Equivalent homogeneous feasibility coordinates; no altered lotion targets."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def projective_feasibility(objective, *, normalizer, A_ub, b_ub, A_eq, b_eq, bounds, time_limit=2.):
    """x=z/t; a positive normalizer fixes the homogeneous scale.

    A*x<=b becomes A*z-b*t<=0; l<=x<=u becomes l*t<=z<=u*t.
    This is feasibility only, not an equivalent transformed cost minimum.
    """
    n = len(objective)
    blocks = [sparse.hstack([A_ub, -sparse.csr_matrix(np.asarray(b_ub)[:, None])], format='csr')]
    rows, cols, data = [], [], []
    for i, (lo, hi) in enumerate(bounds):
        for sign, value in ((-1., lo), (1., hi)):
            if value is not None:
                row = len(rows)//2
                rows += [row, row]; cols += [i,n]; data += [sign,-sign*value]
    blocks.append(sparse.csr_matrix((data,(rows,cols)),shape=(len(rows)//2,n+1)))
    matrix = sparse.vstack(blocks,format='csr')
    equality = sparse.vstack([
        sparse.hstack([A_eq,-sparse.csr_matrix(np.asarray(b_eq)[:,None])]),
        sparse.csr_matrix(np.r_[normalizer,0.][None,:])],format='csr')
    erhs = np.r_[np.zeros(A_eq.shape[0]),1.]
    def normalize(matrix, rhs):
        norm = np.maximum(np.abs(matrix).max(axis=1).toarray().ravel(),np.abs(rhs))
        norm[norm == 0] = 1.
        return matrix.multiply((1./norm)[:,None]).tocsr(),rhs/norm
    matrix,rhs = normalize(matrix,np.zeros(matrix.shape[0]))
    equality,erhs = normalize(equality,erhs)
    started = time.perf_counter()
    result = linprog(np.zeros(n+1),A_ub=matrix,b_ub=rhs,A_eq=equality,b_eq=erhs,
        bounds=[(0.,None)]*n+[(1e-12,None)],method='highs-ipm',
        options={'presolve':True,'time_limit':time_limit,'small_matrix_value':1e-12,
                 'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    report = {'status':int(result.status),'seconds':time.perf_counter()-started,'recovered':False}
    if result.status != 0 or not result.success or result.x is None or result.x[-1] <= 0:
        return None,report
    x = result.x[:-1]/result.x[-1]
    report.update(scale=float(result.x[-1]),max_linear_violation=float(np.max(A_ub@x-b_ub)),
                  max_equality_violation=float(np.max(np.abs(A_eq@x-b_eq))))
    return x,report


def main():
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest,build_estimated_lotion_inputs
    from fragrance_ai.recommender import lotion_optimizer as module
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--brief',action='append',required=True)
    p.add_argument('--oil',type=float,default=30.)
    args = p.parse_args()
    if args.output.exists(): p.error('new evidence file required')
    catalog = read_catalog(ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    rows = []
    for brief in args.brief:
        request,scenarios,_ = build_estimated_lotion_inputs(LotionEstimateRequest(brief=brief,
            registry_pool='conditional_research',max_risk_tier=2),catalog,oil_base_percent=args.oil)
        captured = {}
        original = module.linprog
        def capture(objective,**kwargs):
            local = inspect.currentframe().f_back.f_locals
            if local.get('overlap') == .95 and not captured:
                captured.update(objective=np.asarray(objective),
                    kwargs={key:kwargs[key] for key in ('A_ub','b_ub','A_eq','b_eq','bounds')},
                    responses=local['responses'],valid=local['valid'],assessments=local['assessments'])
            return original(objective,**kwargs)
        module.linprog = capture
        try:
            baseline = module.optimize_lotion(request,catalog,transport_scenarios=scenarios,
                _transport_only=True,reuse_transport_basis=True)
        finally:
            module.linprog = original
        result = {'brief':brief,'oil':args.oil,'score':baseline['score'],
                  'passed95':baseline['profile_target_met'],'probes':[]}
        if captured:
            count = captured['responses'].shape[1]
            norm = np.r_[np.max(captured['responses'],axis=0),np.zeros(len(captured['objective'])-count)]
            x,report = projective_feasibility(captured['objective'],normalizer=norm,**captured['kwargs'])
            if x is not None:
                w = np.maximum(0.,x[:count]); w /= w.sum()
                score,checks = captured['assessments'](w)
                report.update(score=score,valid=bool(captured['valid'](w)),
                              recovered=bool(captured['valid'](w) and score+1e-8 >= 95))
            result['probes'].append(report)
        rows.append(result)
        print(json.dumps(result),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'equivalent_coordinate_probe_not_final_recipe_or_full400',
        'source':'https://www.seas.ucla.edu/~vandenbe/ee236a/lectures/lfp.pdf','rows':rows},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
