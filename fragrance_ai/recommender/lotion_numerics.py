"""Bounded equivalent coordinates for badly scaled lotion linear programs."""
import numpy as np
from scipy import sparse


def response_variable_scales(responses, slack_count):
    maxima = np.max(responses, axis=0)
    positive = maxima[maxima > 0]
    floor = float(positive.min()) if len(positive) else 1.
    scales = np.ones(len(maxima))
    np.divide(floor, maxima, out=scales, where=maxima > 0)
    return np.r_[np.clip(scales, 1e-12, 1.), np.full(slack_count, max(floor, 1e-12))]


def conditioned_linprog(solver, objective, *, A_ub, b_ub, A_eq, b_eq, bounds, scales):
    """x=D*y with positive D; no candidate, constraint or threshold changes."""
    scales = np.asarray(scales, dtype=float)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('finite positive coordinate scales required')
    def transform(matrix, rhs):
        changed = sparse.csr_matrix(matrix).multiply(scales).tocsr()
        norms = np.maximum(np.abs(changed).max(axis=1).toarray().ravel(), np.abs(rhs))
        norms[norms == 0] = 1.
        return changed.multiply((1./norms)[:,None]).tocsr(), np.asarray(rhs)/norms
    matrix, rhs = transform(A_ub, b_ub)
    equality, eq_rhs = transform(A_eq, b_eq)
    costs = objective*scales
    costs = costs/max(float(np.max(np.abs(costs))), 1e-300)
    transformed_bounds = [(lo/s if lo is not None else None, hi/s if hi is not None else None)
        for (lo,hi),s in zip(bounds, scales)]
    result = solver(costs, A_ub=matrix, b_ub=rhs, A_eq=equality, b_eq=eq_rhs,
        bounds=transformed_bounds, method='highs-ipm', options={'time_limit':2., 'presolve':True,
            'small_matrix_value':1e-12, 'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
    if getattr(result, 'x', None) is not None:
        result.x = np.asarray(result.x)*scales
        result.fun = float(objective@result.x)
    return result
