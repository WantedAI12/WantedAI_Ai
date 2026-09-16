"""Fully corrective column search for multi-scenario fractional odor profiles.

Every material is priced in each linearized max-min oracle. Its active face
is then optimized against the EXACT normalized profiles with analytic
derivatives. The full pool is never replaced with an odor-ranked shortlist.
Only feasible, freshly evaluated improvements survive; no optimality claim
is made when the finite local search stops.
"""
from time import monotonic
import warnings

import numpy as np
from scipy.optimize import linprog, minimize, OptimizeResult


class _SearchDeadline(Exception):
    def __init__(self, values):
        self.values = values.copy()


def profile_derivatives(shapes, wanted, responses, weights, avoided=None, background=None, *, smooth_epsilon=0.):
    shapes, wanted, responses, weights = map(np.asarray, (shapes, wanted, responses, weights))
    totals = responses@weights
    if np.any(totals <= 0):
        raise ValueError('positive modeled exposure required')
    predicted = np.einsum('tn,n,hnd->htd', responses, weights, shapes, optimize=True)/totals[None, :, None]
    norms = np.linalg.norm(predicted, axis=-1)
    target_unit = wanted/np.linalg.norm(wanted, axis=-1, keepdims=True)
    cosine = np.einsum('htd,htd->ht', predicted, target_unit)/norms
    if smooth_epsilon > 0:
        distance = np.sqrt((predicted-wanted)**2+smooth_epsilon**2)
        # This is a conservative smooth LOWER bound on exact overlap. It is
        # used for search only; no higher score is manufactured by smoothing.
        overlap = 1-.5*distance.sum(-1)
        overlap_gradient = -.5*(predicted-wanted)/distance
    else:
        overlap = np.minimum(predicted, wanted).sum(-1)
        overlap_gradient = (predicted < wanted).astype(float)
    if avoided is None:
        avoided = np.zeros(wanted.shape[1:])
    avoided = np.asarray(avoided)
    avoidance = 1-np.einsum('htd,td->ht', predicted, avoided)
    values = np.stack((overlap, cosine, avoidance), axis=-1)
    profile_gradients = np.stack((overlap_gradient,
        target_unit/norms[..., None]-cosine[..., None]*predicted/norms[..., None]**2,
        np.broadcast_to(-avoided, predicted.shape)), axis=-2)

    def mass_gradient(gradient):
        contracted = np.einsum('htkd,hnd->htkn', gradient, shapes, optimize=True)
        center = np.einsum('htkd,htd->htk', gradient, predicted)
        return (contracted-center[..., None])*responses[None, :, None, :]/totals[None, :, None, None]

    jacobian = mass_gradient(profile_gradients)
    contrast_values, contrast_jacobian = np.zeros(0), np.zeros((0, len(weights)))
    if background is not None:
        background = np.asarray(background)
        from .reference_discrimination import contrast_direction
        contrast = contrast_direction(wanted,background[:,None])
        contrast_values = np.einsum('htd,htd->ht', contrast, predicted)/norms
        gradient = contrast/norms[..., None]-contrast_values[..., None]*predicted/norms[..., None]**2
        contrast_jacobian = mass_gradient(gradient[:, :, None]).reshape(-1, len(weights))
        contrast_values = contrast_values.ravel()
    return values.ravel(), jacobian.reshape(-1, len(weights)), contrast_values, contrast_jacobian


def corrective_profile_search(*, shapes, wanted, responses, fixed, rhs, equality,
        equality_rhs, bounds, initial, target, avoided=None, background=None,
        maximum_seconds=10., maximum_rounds=10):
    started = monotonic()
    linear_solves, nonlinear_solves = 0, 0
    n = len(bounds)
    lo = np.asarray([0. if b[0] is None else b[0] for b in bounds])
    hi = np.asarray([1. if b[1] is None else b[1] for b in bounds])
    fixed, rhs, equality, equality_rhs = map(np.asarray, (fixed, rhs, equality, equality_rhs))
    if not (0 < target <= 1 and maximum_seconds > 0 and maximum_rounds >= 1):
        raise ValueError('invalid corrective search configuration')
    if any(not np.isfinite(v).all() for v in (lo, hi, fixed, rhs, equality, equality_rhs)) or np.any(lo > hi):
        raise ValueError('finite consistent full-pool constraints required')

    def valid(x):
        return (x is not None and np.shape(x) == (n,) and np.isfinite(x).all()
                and np.all(x >= lo-1e-10) and np.all(x <= hi+1e-10)
                and np.all(fixed@x <= rhs+1e-9)
                and np.all(np.abs(equality@x-equality_rhs) <= 1e-8))

    best = np.asarray(initial, float).copy() if initial is not None else None
    if not valid(best):
        linear_solves += 1
        feasible = linprog(np.zeros(n), A_ub=fixed, b_ub=rhs, A_eq=equality,
                           b_eq=equality_rhs, bounds=bounds, method='highs', options={'time_limit': 2.})
        best = feasible.x if feasible.success and valid(feasible.x) else None
    if best is None:
        return OptimizeResult(success=False, status=1, x=None, candidate=None,
            candidate_score=None, full_pool_count=n, priced_columns=0, rounds=0,
            linear_solves=linear_solves, nonlinear_solves=0,
            elapsed_seconds=monotonic()-started, message='no feasible initial formula')

    def evaluate(x):
        return profile_derivatives(shapes, wanted, responses, x, avoided, background)

    values, jac, contrast, contrast_jac = evaluate(best)
    best_score = float(values.min())
    specific = bool(np.all(contrast > 1e-8))
    visited, priced, rounds = set(), 0, 0
    for iteration in range(maximum_rounds):
        if (best_score+1e-9 >= target and specific) or monotonic()-started >= maximum_seconds:
            break
        rounds = iteration+1
        # One oracle contains EVERY original material and every score component.
        # The epigraph variable is a local search proposal, never a score bound.
        search_values, search_jac, _, _ = profile_derivatives(shapes, wanted, responses, best,
            avoided, background, smooth_epsilon=1e-5)
        rows = [np.c_[-search_jac, np.ones(len(search_values))], np.c_[fixed, np.zeros(len(rhs))]]
        limits = [search_values-search_jac@best, rhs]
        if len(contrast):
            rows.append(np.c_[-contrast_jac, np.zeros(len(contrast))])
            limits.append(contrast-contrast_jac@best-2e-8)
        matrix, limits = np.vstack(rows), np.concatenate(limits)
        norms = np.maximum(np.abs(matrix).max(1), np.abs(limits))
        norms = np.maximum(norms, 1e-15)
        linear_solves += 1
        proposal = linprog(np.r_[np.zeros(n), -1.], A_ub=matrix/norms[:, None], b_ub=limits/norms,
            A_eq=np.c_[equality, np.zeros(len(equality_rhs))], b_eq=equality_rhs,
            bounds=[*bounds, (0., 1.)], method='highs', options={'time_limit': 2.,
                'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        priced += n
        if not proposal.success or proposal.x is None:
            break
        support = np.flatnonzero((best > 1e-12) | (proposal.x[:-1] > 1e-12) | (lo > 0))
        signature = tuple(support)
        if signature in visited:
            break
        visited.add(signature)
        initial_values = np.r_[best[support], min(best_score, target)]
        local_shapes, local_responses = shapes[:, support], responses[:, support]
        cached_key, cached = None, None

        def local(v):
            nonlocal cached_key, cached
            key = v.tobytes()
            if key != cached_key:
                scores, gradients, discriminants, discriminant_jac = profile_derivatives(
                    local_shapes, wanted, local_responses, v[:-1], avoided, background, smooth_epsilon=1e-5)
                cached = (np.r_[scores-v[-1], discriminants-2e-8],
                    np.vstack((np.c_[gradients, -np.ones(len(scores))],
                               np.c_[discriminant_jac, np.zeros(len(discriminants))])))
                cached_key = key
            return cached

        def deadline(v):
            if monotonic()-started >= maximum_seconds:
                raise _SearchDeadline(v)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='^Values in x were outside bounds during a minimize step, clipping to bounds$',
                                    category=RuntimeWarning, module=r'scipy\.optimize\._slsqp_py')
            nonlinear_solves += 1
            try:
                solved = minimize(lambda v: -v[-1], initial_values, jac=lambda v: np.r_[np.zeros(len(support)), -1.],
                    method='SLSQP', callback=deadline, bounds=[*[bounds[i] for i in support], (0., target)], constraints=[
                    {'type':'eq', 'fun':lambda v: equality[:, support]@v[:-1]-equality_rhs,
                     'jac':lambda v: np.c_[equality[:, support], np.zeros(len(equality_rhs))]},
                    {'type':'ineq', 'fun':lambda v: rhs-fixed[:, support]@v[:-1],
                     'jac':lambda v: np.c_[-fixed[:, support], np.zeros(len(rhs))]},
                    {'type':'ineq', 'fun':lambda v: local(v)[0], 'jac':lambda v: local(v)[1]}],
                    options={'maxiter': 100, 'ftol': 1e-10})
            except _SearchDeadline as stopped:
                solved = OptimizeResult(success=False, status=1, x=stopped.values)
        x = np.zeros(n)
        if solved.x is not None and np.isfinite(solved.x).all():
            x[support] = solved.x[:-1]
            # Compare every intermediate point on the feasible chord as well;
            # an unconverged local iterate never erases a verified incumbent.
            for fraction in (1., .5, .25, .125):
                trial = (1-fraction)*best+fraction*x
                if not valid(trial):
                    continue
                current, _, current_contrast, _ = evaluate(trial)
                score = float(current.min())
                current_specific = bool(np.all(current_contrast > 1e-8))
                if (score+1e-9 >= target and current_specific, score) > (best_score+1e-9 >= target and specific, best_score):
                    best, best_score, specific = trial, score, current_specific
        values, jac, contrast, contrast_jac = evaluate(best)
    passed = best_score+1e-9 >= target and specific
    return OptimizeResult(success=passed, status=0 if passed else 1,
        x=best if passed else None, candidate=best, candidate_score=best_score,
        full_pool_count=n, priced_columns=priced, rounds=rounds,
        linear_solves=linear_solves, nonlinear_solves=nonlinear_solves,
        elapsed_seconds=monotonic()-started, message='exact full-profile check; local search, not global certificate')
