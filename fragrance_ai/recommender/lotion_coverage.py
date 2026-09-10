"""Optimistic fixed-profile limits, independent of transport and recipe constraints."""
import numpy as np
from scipy.optimize import linprog

from .models import profile_vector


def profile_overlap_upper(profiles, target):
    """A feasible dual inequality, not an unchecked primal optimum.

    For any q in [0,1], overlap(t,p) <= t.q + max_i a_i.(1-q).
    Recomputing this expression after clipping q remains an upper bound even
    when the numerical optimizer stops early. All axes participate.
    """
    matrix, target = np.asarray(profiles, dtype=float), np.asarray(target, dtype=float)
    if (matrix.ndim != 2 or target.shape != (matrix.shape[1],) or not len(matrix)
            or not np.isfinite(matrix).all() or not np.isfinite(target).all()
            or np.any(matrix < 0) or np.any(target < 0) or target.sum() <= 0
            or np.any(matrix.sum(axis=1) <= 0)):
        raise ValueError('finite nonnegative nonempty profiles required')
    matrix = np.unique(matrix / matrix.sum(axis=1)[:,None], axis=0)
    target = target / target.sum()
    d = len(target)
    result = linprog(np.r_[target, 1.], A_ub=np.c_[-matrix, -np.ones(len(matrix))],
        b_ub=-np.ones(len(matrix)), bounds=[(0.,1.)]*(d+1), method='highs',
        options={'time_limit':.2})
    if result.x is None or not np.isfinite(result.x).all():
        return 100.
    q = np.clip(result.x[:d], 0., 1.)
    return min(100., 100.*float(target@q + np.max(matrix@(1.-q))) + 1e-6)


def lotion_profile_coverage(catalog, evaluation_targets, target_score=95.):
    # The caller supplies a superset of every candidate allowed in the search.
    profiles = np.array([i.vector() for i in catalog.ingredients if sum(i.profile.values()) > 0])
    bounds, seen = [], {}
    for row in evaluation_targets:
        vector = profile_vector(row['target_profile'])
        key = tuple(vector)
        if key not in seen:
            seen[key] = profile_overlap_upper(profiles, vector)
        bounds.append({'minutes':row['minutes'], 'phase':row['phase'], 'upper_percent':seen[key]})
    upper = min(r['upper_percent'] for r in bounds) if bounds else 100.
    return {'optimistic_profile_upper_percent':upper, 'target_excluded':upper < target_score-1e-8,
        'scope':'supplied_profile_pool; ignores_transport_caps_cost_and_avoidance',
        'bound_method':'recomputed_clipped_dual_overlap_inequality_with_numeric_margin',
        'timepoint_bounds':bounds, 'human_accuracy_measured':False}
