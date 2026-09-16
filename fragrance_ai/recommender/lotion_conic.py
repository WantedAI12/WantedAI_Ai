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
                       objective, time_limit=2., background=None,
                       initial_columns=None, column_order=None):
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
    profiles, targets, responses = map(np.asarray, (profiles, targets, responses))
    if profiles.ndim == 2:
        n, d = profiles.shape
        profile_rows = np.broadcast_to(profiles, (len(responses), n, d))
        background_rows = (np.broadcast_to(np.asarray(background), (len(responses), d))
                           if background is not None else None)
    elif profiles.ndim == 3:
        heads, n, d = profiles.shape
        times = len(responses)
        if targets.shape != (heads, times, d):
            raise ValueError('all reference heads and transport scenarios must align')
        profile_rows = np.broadcast_to(profiles[:, None], (heads, times, n, d)).reshape(-1,n,d)
        targets = targets.reshape(-1,d)
        responses = np.tile(responses,(heads,1))
        background_rows = (np.broadcast_to(np.asarray(background)[:,None], (heads,times,d)).reshape(-1,d)
                           if background is not None else None)
    else:
        raise ValueError('profile matrix or head/material/endpoint tensor required')
    count, ns = len(responses), slack_sums.shape[1]
    q, width = target_score/100., n+ns
    if targets.shape != (count, d) or len(bounds) != width:
        raise ValueError('conic profile shape mismatch')
    if any(not np.isfinite(x).all() for x in (profiles, targets, responses)) or np.any(responses < 0):
        raise ValueError('finite conic profile and response data required')
    # An inexpensive, optimistic bound avoids impossible pure-axis requests.
    # It ignores all physical constraints and is not a global solver result.
    upper = 100*np.minimum(targets, profile_rows.max(axis=1)).sum(axis=1).min()
    report['coordinate_upper_score'] = float(upper)
    if upper+1e-7 < target_score:
        return None, {**report, 'status': 'excluded_by_fixed_profile_coordinate_bound'}
    variation = sparse.hstack([sparse.csr_matrix(outside-2*(1-q)*responses), slack_sums], format='csr')
    avoidance = sparse.hstack([sparse.csr_matrix((np.einsum('rd,rnd->rn',avoid_vectors,profile_rows)-(1-q))*responses),
                               sparse.csr_matrix((count, ns))], format='csr')
    linear = sparse.vstack([residual_rows, variation, avoidance, fixed_rows], format='csr')
    linear_rhs = np.r_[np.zeros(residual_rows.shape[0]+2*count), fixed_rhs]
    # Explicit variable bounds are part of the cone system, never solver hints.
    bound_rows, bound_cols, bound_values, bound_rhs = [], [], [], []
    for i, (lo, hi) in enumerate(bounds):
        for sign, value in ((-1., lo), (1., hi)):
            if value is not None:
                bound_rows.append(len(bound_rhs))
                bound_cols.append(i)
                bound_values.append(sign)
                bound_rhs.append(sign*value)
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
        prediction = profile_rows[t].T*responses[t][None, :]
        soc = np.vstack([(targets[t]@prediction)/target_norm, q*prediction])
        block = sparse.hstack([-sparse.csr_matrix(soc), sparse.csr_matrix((d+1, ns))], format='csr')
        block = block.multiply(scales).tocsc()
        norm = max(float(np.abs(block).max()), 1e-300)
        # One common scale per cone: independent row scaling changes geometry.
        blocks.append(block/norm)
        right.append(np.zeros(d+1))
        cones.append(clarabel.SecondOrderConeT(d+1))
        if background_rows is not None:
            background_norm = float(np.linalg.norm(background_rows[t]))
            if background_norm <= 0 or not np.isfinite(background_norm):
                raise ValueError('finite nonempty background profile required')
            from .reference_discrimination import contrast_direction
            contrast = contrast_direction(targets[t],background_rows[t])
            # Preserve the final strict background-discrimination condition.
            # A conservative numeric margin cannot certify infeasibility of
            # the original open inequality, so formal_certificate stays false.
            discrimination = np.vstack([contrast@prediction,2e-8*prediction])
            block = sparse.hstack([-sparse.csr_matrix(discrimination),sparse.csr_matrix((d+1,ns))],format='csr').multiply(scales).tocsc()
            norm = max(float(np.abs(block).max()),1e-300)
            blocks.append(block/norm)
            right.append(np.zeros(d+1))
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
    active = set(range(n)) if initial_columns is None else set(initial_columns)
    active.update(i for i,(low,_) in enumerate(bounds[:n]) if low and low > 0)
    order = list(range(n)) if column_order is None else list(column_order)
    active.update(order[:min(64,n)])
    if not active or any(i < 0 or i >= n for i in active) or set(order) != set(range(n)):
        raise ValueError('valid full-pool cone column order required')
    report.update(full_pool_materials=n, material_columns_priced=0,
                  maximum_working_materials=0, permanent_candidate_shortlist=False)
    selected_columns = None
    status = 'NotStarted'
    best_weights, best_rank = None, (False, -1.)

    def capture_candidate(solution, columns):
        nonlocal best_weights, best_rank
        if solution.x is None or len(solution.x) != len(columns) or not np.isfinite(solution.x).all():
            return False
        full = np.zeros(width)
        full[columns] = solution.x
        weights = np.maximum(0.,full[:n]*scales[:n])
        low = np.asarray([0. if b[0] is None else b[0] for b in bounds[:n]])
        high = np.asarray([np.inf if b[1] is None else b[1] for b in bounds[:n]])
        eq_error = np.asarray(a_eq[:,:n]@weights-eq_rhs).ravel()
        fixed_value = np.asarray(fixed_rows[:,:n]@weights).ravel()
        tolerance = 1e-9*np.maximum(np.maximum(np.abs(fixed_value),np.abs(fixed_rhs)),1e-12)
        if (np.any(weights < low-1e-10) or np.any(weights > high+1e-10)
                or np.any(np.abs(eq_error)>1e-8) or np.any(fixed_value-np.asarray(fixed_rhs)>tolerance)):
            return False
        weighted = responses*weights
        total = weighted.sum(-1)
        if np.any(total <= 0):
            return False
        predicted = np.einsum('rn,rnd->rd',weighted,profile_rows)/total[:,None]
        cosine = np.einsum('rd,rd->r',predicted,targets)/(np.linalg.norm(predicted,axis=1)*np.linalg.norm(targets,axis=1))
        overlap = np.minimum(predicted,targets).sum(-1)
        avoidance = 1-(predicted*avoid_vectors).sum(-1)
        quality = float(np.minimum(np.minimum(cosine,overlap),avoidance).min()*100)
        specific = True
        if background_rows is not None:
            from .reference_discrimination import contrast_direction
            margin=np.einsum('rd,rd->r',predicted,contrast_direction(targets,background_rows))/np.linalg.norm(predicted,axis=1)
            specific = bool(np.all(margin>1e-8))
        passes = quality+1e-8 >= target_score and specific
        # The retained incumbent must obey the SAME ranking as the caller.
        # A high-scoring generic/background shape cannot replace a slightly
        # lower-scoring specific formula which passes all acceptance checks.
        rank = (passes, quality)
        if rank > best_rank:
            best_weights,best_rank = weights,rank
            report['fresh_candidate_score'] = quality
            report['fresh_candidate_specific'] = specific
            report['fresh_candidate_passed'] = passes
        return passes
    # The auxiliary variables retain every odor endpoint. Only MATERIAL
    # columns are introduced incrementally, using dual prices over the full
    # original cone matrix. No fixed odor shortlist becomes the universe.
    for expansion in range(10):
        remaining = time_limit-(time.perf_counter()-started)
        if remaining <= 0:
            status = 'MaxTime'
            break
        columns = np.asarray(sorted(active)+list(range(n,width)),dtype=int)
        restricted = matrix[:,columns]
        eq_count, linear_count = equality.shape[0], linear.shape[0]
        eq_nonzero = np.asarray(restricted[:eq_count].getnnz(axis=1)).ravel() > 0
        linear_nonzero = np.asarray(restricted[eq_count:eq_count+linear_count].getnnz(axis=1)).ravel() > 0
        eq_keep = np.flatnonzero(eq_nonzero | (rhs[:eq_count] != 0))
        linear_keep = np.flatnonzero(linear_nonzero | (rhs[eq_count:eq_count+linear_count] < 0))+eq_count
        keep = np.r_[eq_keep,linear_keep,np.arange(eq_count+linear_count,len(rhs))]
        working_cones = [clarabel.ZeroConeT(len(eq_keep)),clarabel.NonnegativeConeT(len(linear_keep)),*cones[2:]]
        settings.time_limit = remaining
        solution = clarabel.DefaultSolver(sparse.csc_matrix((len(columns),len(columns))), costs[columns],
            restricted[keep].tocsc(),rhs[keep],working_cones,settings).solve()
        status = str(solution.status)
        report['solver_calls'] = expansion+1
        report['maximum_working_materials'] = max(report['maximum_working_materials'],len(active))
        report['iterations'] = solution.iterations
        if capture_candidate(solution,columns):
            # Feasibility is enough; unfinished cost minimization must not
            # discard a formula passing every original profile/physical row.
            selected_columns = columns
            report['feasible_before_cost_optimum'] = status not in ('Solved','AlmostSolved')
            break
        if status in ('Solved','AlmostSolved') and solution.x is not None:
            selected_columns = columns
            break
        if len(active) == n or status not in ('PrimalInfeasible','AlmostPrimalInfeasible','MaxIterations','MaxTime','InsufficientProgress','NumericalError'):
            break
        missing = np.asarray([i for i in range(n) if i not in active and bounds[i][1] != 0],dtype=int)
        if not len(missing):
            break
        additions = []
        if solution.z is not None and len(solution.z) == len(keep) and np.isfinite(solution.z).all():
            dual = np.zeros(len(rhs))
            dual[keep] = solution.z
            reduced = matrix[:,missing].T@dual
            if status not in ('PrimalInfeasible','AlmostPrimalInfeasible'):
                reduced = reduced+costs[missing]
            report['material_columns_priced'] = n
            eligible = np.flatnonzero(reduced < -1e-9)
            additions = missing[eligible[np.argsort(reduced[eligible],kind='stable')[:max(64,len(active)//2)]]].tolist()
        if not additions:
            additions = [i for i in order if i not in active][:max(64,len(active)//2)]
        active.update(additions)
    report.update(status=status, seconds=time.perf_counter()-started,
                  solver_incomplete=selected_columns is None or status not in ('Solved','AlmostSolved'),
                  global_cost_optimality_claimed=False,
                  restricted_infeasibility_is_not_full_infeasibility=True)
    if selected_columns is None:
        return best_weights, report
    if report.get('feasible_before_cost_optimum') and best_weights is not None:
        return best_weights, report
    full = np.zeros(width)
    full[selected_columns] = solution.x
    weights = full[:n]*scales[:n]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return None, {**report, 'status': 'invalid_conic_candidate', 'solver_incomplete': True}
    return weights, report
