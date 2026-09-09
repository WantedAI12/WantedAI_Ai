"""Direct full-profile cone feasibility over the existing lotion transport model.

For fixed q, cosine >= q is a second-order cone. Overlap, avoidance, cost,
mass balance, dose and ingredient limits remain the existing linear rows.
The final caller rechecks actual returned recipe curves; solver status alone
is neither a feasible recipe nor a formal global infeasibility certificate.
"""
import time

import numpy as np
from scipy import sparse

from .lotion_numerics import response_variable_scales


def solve_full_profile(*, profiles, targets, responses, target_score, residual_rows, slack_sums,
                       outside, avoid_vectors, fixed_rows, fixed_rhs, a_eq, eq_rhs, bounds,
                       objective, time_limit=2.):
    report = {'version': 'lotion-full-profile-socp/v1', 'attempted': False, 'accepted': False,
              'solver': 'clarabel', 'solver_calls': 0, 'formal_certificate': False,
              'target_score': float(target_score), 'all_profile_axes_retained': True,
              'acceptance_threshold_modified': False, 'solver_incomplete': False}
    try:
        import clarabel
    except ImportError:
        return None, {**report, 'status': 'optional_conic_solver_unavailable'}
    if not 0 < target_score <= 100 or not np.isfinite(target_score):
        raise ValueError('finite positive cone target at most 100 required')
    n, d = profiles.shape
    count, ns = len(responses), slack_sums.shape[1]
    q, width = target_score/100., n+ns
    if targets.shape != (count, d) or len(bounds) != width:
        raise ValueError('conic profile shape mismatch')
    if any(not np.isfinite(x).all() for x in (profiles, targets, responses)) or np.any(responses < 0):
        raise ValueError('finite conic profile and response data required')
    # An inexpensive, optimistic bound avoids impossible pure-axis requests.
    # It ignores all physical constraints and is not a global solver result.
    upper = 100*np.minimum(targets, profiles.max(axis=0)[None, :]).sum(axis=1).min()
    report['coordinate_upper_score'] = float(upper)
    if upper+1e-7 < target_score:
        return None, {**report, 'status': 'excluded_by_fixed_profile_coordinate_bound'}
    variation = sparse.hstack([sparse.csr_matrix(outside-2*(1-q)*responses), slack_sums], format='csr')
    avoidance = sparse.hstack([sparse.csr_matrix((avoid_vectors@profiles.T-(1-q))*responses),
                               sparse.csr_matrix((count, ns))], format='csr')
    linear = sparse.vstack([residual_rows, variation, avoidance, fixed_rows], format='csr')
    linear_rhs = np.r_[np.zeros(residual_rows.shape[0]+2*count), fixed_rhs]
    # Explicit variable bounds are part of the cone system, never solver hints.
    bound_rows, bound_cols, bound_values, bound_rhs = [], [], [], []
    for i, (lo, hi) in enumerate(bounds):
        for sign, value in ((-1., lo), (1., hi)):
            if value is not None:
                bound_rows.append(len(bound_rhs)); bound_cols.append(i)
                bound_values.append(sign); bound_rhs.append(sign*value)
    bounded = sparse.csr_matrix((bound_values, (bound_rows, bound_cols)), shape=(len(bound_rhs), width))
    linear = sparse.vstack([linear, bounded], format='csr')
    linear_rhs = np.r_[linear_rhs, bound_rhs]
    scales = response_variable_scales(responses, ns)

    def scale_linear(matrix, rhs):
        matrix = sparse.csr_matrix(matrix).multiply(scales).tocsr()
        norms = np.maximum(np.abs(matrix).max(axis=1).toarray().ravel(), np.abs(rhs))
        norms[norms == 0] = 1.
        return matrix.multiply((1./norms)[:, None]).tocsc(), np.asarray(rhs)/norms

    equality, equality_rhs = scale_linear(a_eq, eq_rhs)
    linear, linear_rhs = scale_linear(linear, linear_rhs)
    blocks, right = [equality, linear], [equality_rhs, linear_rhs]
    cones = [clarabel.ZeroConeT(equality.shape[0]), clarabel.NonnegativeConeT(linear.shape[0])]
    for t in range(count):
        target_norm = float(np.linalg.norm(targets[t]))
        if target_norm <= 0:
            raise ValueError('nonempty cone target required')
        # p = profiles.T @ (responses[t] * weights).
        # [target.p / ||target||, q*p] belongs to the Lorentz cone.
        prediction = profiles.T*responses[t][None, :]
        soc = np.vstack([(targets[t]@prediction)/target_norm, q*prediction])
        block = sparse.hstack([-sparse.csr_matrix(soc), sparse.csr_matrix((d+1, ns))], format='csr')
        block = block.multiply(scales).tocsc()
        norm = max(float(np.abs(block).max()), 1e-300)
        # One common scale per cone: independent row scaling changes geometry.
        blocks.append(block/norm); right.append(np.zeros(d+1))
        cones.append(clarabel.SecondOrderConeT(d+1))
    matrix, rhs = sparse.vstack(blocks, format='csc'), np.concatenate(right)
    costs = np.asarray(objective)*scales
    costs /= max(float(np.abs(costs).max()), 1e-300)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.time_limit = time_limit
    settings.max_iter = 150
    settings.tol_feas = 1e-10
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    started = time.perf_counter()
    report.update(attempted=True, solver_calls=1, solver_version=clarabel.__version__,
                  cone_count=len(cones), decision_variables=width)
    solution = clarabel.DefaultSolver(sparse.csc_matrix((width, width)), costs, matrix, rhs, cones, settings).solve()
    status = str(solution.status)
    report.update(status=status, seconds=time.perf_counter()-started, iterations=solution.iterations,
                  solver_incomplete=status not in ('Solved', 'PrimalInfeasible'))
    if status not in ('Solved', 'AlmostSolved') or solution.x is None:
        return None, report
    weights = np.asarray(solution.x[:n])*scales[:n]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return None, {**report, 'status': 'invalid_conic_candidate', 'solver_incomplete': True}
    return weights, report
