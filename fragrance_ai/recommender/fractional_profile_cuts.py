"""Full-pool outer approximation of normalized profile feasibility.

For each time/head, L1(Dx) <= 2(1-q) r.x is represented by supporting
halfspaces sign(Dx_k).Dx <= 2(1-q) r.x. No descriptor is removed. Cosine
cones use their supporting planes too. Only a fresh full-vector check accepts
the solution; finite cutting-plane work is not a proof of infeasibility.
"""
from time import monotonic

import numpy as np
from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import csr_matrix


def solve_profile_cuts(*, shapes, wanted, responses, fixed, rhs, equality,
                       equality_rhs, bounds, objective, level, background=None,
                       initial=None, maximum_seconds=12., maximum_rounds=48,
                       linear_solver=linprog):
    shapes, wanted, responses = (np.asarray(x, dtype=float) for x in (shapes, wanted, responses))
    heads, n, dimensions = shapes.shape
    times = responses.shape[0]
    disabled = np.asarray([bound[1] == 0 for bound in bounds], dtype=bool)
    if (wanted.shape != (heads, times, dimensions) or responses.shape[1] != n
            or not 0 <= level <= 1 or maximum_seconds <= 0 or maximum_rounds < 1
            or any(not np.isfinite(x).all() or np.any(x < 0) for x in (shapes, wanted, responses))
            or disabled.shape != (n,)
            or not np.allclose(shapes[:, ~disabled].sum(axis=2), 1., atol=1e-7)
            or not np.allclose(wanted.sum(axis=2), 1., atol=1e-7)):
        raise ValueError('normalized finite full profiles and aligned responses required')
    target_unit = wanted/np.linalg.norm(wanted, axis=2, keepdims=True)
    contrast = None
    if background is not None:
        background = np.asarray(background, dtype=float)
        if background.shape != (heads, dimensions) or not np.isfinite(background).all() or np.any(background < 0) or np.any(background.sum(axis=1) <= 0):
            raise ValueError('invalid profile background')
        from .reference_discrimination import contrast_direction
        contrast = contrast_direction(wanted,background[:,None,:])
    fixed, rhs = np.asarray(fixed, dtype=float), np.asarray(rhs, dtype=float)
    equality, equality_rhs = np.asarray(equality, dtype=float), np.asarray(equality_rhs, dtype=float)
    objective = np.asarray(objective, dtype=float)
    lo = np.array([0. if value[0] is None else value[0] for value in bounds])
    hi = np.array([np.inf if value[1] is None else value[1] for value in bounds])
    if any(not np.isfinite(x).all() for x in (fixed, rhs, equality, equality_rhs, objective, lo)) or np.any(lo > hi):
        raise ValueError('invalid linear constraints')
    if objective.shape != (n,) or fixed.shape != (len(rhs), n) or equality.shape != (len(equality_rhs), n):
        raise ValueError('linear constraint shape mismatch')
    rows, keys = [], set()
    started, solves = monotonic(), 0
    best, best_score = None, -np.inf

    def valid(x):
        return (x is not None and np.isfinite(x).all() and x.shape == (n,)
                and np.all(x >= lo-1e-10) and np.all(x <= hi+1e-10)
                and np.all(fixed@x <= rhs+1e-9)
                and np.all(np.abs(equality@x-equality_rhs) <= 1e-8))

    def predict(x):
        weighted = responses*x
        total = weighted.sum(axis=1)
        if np.any(total <= 0):
            return None
        return np.einsum('tn,hnd->htd', weighted/total[:, None], shapes, optimize=True)

    def add(row):
        scale = float(np.max(np.abs(row)))
        if scale <= 1e-15:
            return
        normalized = row/scale
        key = np.round(normalized, 11).tobytes()
        if key not in keys:
            keys.add(key)
            rows.append(normalized)

    def inspect(x):
        nonlocal best, best_score
        predicted = predict(x)
        if predicted is None:
            return False
        unit = predicted/np.linalg.norm(predicted, axis=2, keepdims=True)
        overlap = np.minimum(predicted, wanted).sum(axis=2)
        cosine = np.einsum('htd,htd->ht', target_unit, unit)
        score = float(np.minimum(overlap, cosine).min())
        if score > best_score:
            best, best_score = x.copy(), score
        specific = contrast is None or np.all(np.einsum('htd,htd->ht', contrast, unit) > 1e-8)
        if score+1e-9 >= level and specific:
            return True
        for h in range(heads):
            for t in range(times):
                if overlap[h, t]+1e-10 < level:
                    delta = shapes[h]-wanted[h, t]
                    sign = np.sign(predicted[h, t]-wanted[h, t])
                    add((delta@sign-2*(1-level))*responses[t])
                if cosine[h, t]+1e-10 < level:
                    add((shapes[h]@(level*unit[h, t]-target_unit[h, t]))*responses[t])
                if contrast is not None and float(contrast[h, t]@unit[h, t]) <= 1e-8:
                    add((shapes[h]@(2e-8*unit[h, t]-contrast[h, t]))*responses[t])
        return False

    def finish(success, status, message, x=None):
        return OptimizeResult(success=success, status=status, message=message, x=x,
                              candidate=best, candidate_score=best_score if best is not None else None,
                              cut_count=len(rows), linear_solves=solves, elapsed_seconds=monotonic()-started,
                              full_pool_count=n, full_endpoint_count=dimensions)

    if initial is not None and valid(np.asarray(initial)) and inspect(np.asarray(initial)):
        return finish(True, 0, 'initial full-profile feasible', np.asarray(initial).copy())
    for _ in range(maximum_rounds):
        remaining = maximum_seconds-(monotonic()-started)
        if remaining <= 0:
            return finish(False, 1, 'full-profile separation work limit')
        matrix = np.vstack([fixed, *[row[None, :] for row in rows]]) if rows else fixed
        limits = np.r_[rhs, np.zeros(len(rows))]
        scales = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(limits))
        scales = np.maximum(scales, 1e-15)
        result = linear_solver(objective, A_ub=csr_matrix(matrix/scales[:, None]), b_ub=limits/scales,
                         A_eq=csr_matrix(equality), b_eq=equality_rhs, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': min(2., remaining),
                                  'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        solves += 1
        if not result.success or result.x is None:
            # The numerical strict-background margin is stronger than >0.
            status = 2 if result.status == 2 and contrast is None else 1
            return finish(False, status, 'outer relaxation: '+str(result.message))
        x = result.x
        if not valid(x):
            return finish(False, 1, 'primal residual check failed')
        previous = len(rows)
        if inspect(x):
            return finish(True, 0, 'fresh full-profile feasibility verified', x)
        if len(rows) == previous:
            return finish(False, 1, 'separation stalled; no full feasibility claim')
    return finish(False, 1, 'full-profile separation round limit')
