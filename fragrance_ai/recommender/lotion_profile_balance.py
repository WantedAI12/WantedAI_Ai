"""Full learned note-balance refinement over actual lotion transport responses.

The checkpoint remains a stock-reference prior. This corrects a mathematical
target-affinity degeneracy, not missing lotion sensory calibration. Unmapped
endpoint mass is retained; unknown molecules receive conservative bounds.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .lotion_evaluation import LOTION_PROJECTION, refinement_score_floors
from .lotion_numerics import conditioned_linprog

VERSION = 'lotion-learned-full-balance/v2'
AXES = tuple(LOTION_PROJECTION) + ('unmapped_endpoint_mass',)


def prepare_balance(predictor, pool, target_rows):
    mapping = getattr(predictor, 'projection', LOTION_PROJECTION)
    axes = tuple(mapping) + ('unmapped_endpoint_mass',)
    if any((set(row['avoided']) | {k for k, v in row['target_profile'].items() if v > 0})
           - set(mapping) for row in target_rows):
        raise ValueError('unsupported learned profile target')
    endpoints = predictor.provider.endpoints
    projection = np.zeros((len(endpoints), len(axes)))
    for j, axis in enumerate(axes[:-1]):
        for name in mapping[axis]:
            projection[endpoints.index(name), j] = 1.
    if np.any(projection.sum(axis=1) > 1):
        raise ValueError('overlapping learned note projection')
    projection[:, -1] = 1.-projection.sum(axis=1)
    shapes = np.zeros((getattr(predictor, 'reference_count', 3), len(pool), len(axes)))
    missing = np.zeros(len(pool))
    for i, item in enumerate(pool):
        shape = predictor.shape(item)
        if shape is None:
            shapes[:, i, -1] = 1.
            missing[i] = 1.
        else:
            shapes[:, i] = shape @ projection
    targets = np.array([[row['target_profile'].get(axis, 0.) for axis in axes] for row in target_rows])
    if not np.isfinite(targets).all() or np.any(targets < 0) or np.any(targets.sum(axis=1) <= 0):
        raise ValueError('positive finite learned profile targets required')
    targets /= targets.sum(axis=1)[:, None]
    avoided = np.array([[float(axis in row['avoided']) for axis in axes] for row in target_rows])
    # Unknown mass could lie entirely in a prohibited axis. If no axis is
    # prohibited it is still penalized by the conservative overlap bound.
    avoidance = np.einsum('rnd,td->rtn', shapes, avoided)
    avoidance += (avoided.sum(axis=1) > 0)[None, :, None]*missing[None, None, :]
    return shapes, targets, avoidance


def profile_values(shapes, targets, avoidance, responses, weights):
    contribution = responses*weights
    total = contribution.sum(axis=1)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError('positive finite lotion headspace required')
    predicted = np.einsum('tn,rnd->rtd', contribution, shapes)/total[None, :, None]
    overlap = np.minimum(predicted, targets[None, :, :]).sum(axis=2)
    avoided = np.sum(avoidance*contribution[None, :, :], axis=2)/total[None, :]
    return np.clip(np.minimum(overlap, 1.-avoided), 0., 1.)


def fresh_curve_profile_match(results, target_rows, pool, predictor):
    shapes, targets, avoidance = prepare_balance(predictor, pool, target_rows)
    ids = {item.ingredient_id: i for i, item in enumerate(pool)}
    points = [point for result in results for point in result['temporal_profile'][1:]]
    if len(points) != len(target_rows):
        raise ValueError('learned balance final timepoint mismatch')
    responses = np.zeros((len(points), len(pool)))
    for t, point in enumerate(points):
        key = 'odor_activity_proxy' if point['profile_basis'] == 'linear_odor_activity_proxy' else 'air_concentration_mg_m3'
        for row in point['materials']:
            responses[t, ids[row['ingredient_id']]] = row[key]
    return profile_values(shapes, targets, avoidance, responses, np.ones(len(pool)))


def refine_profile_balance(*, predictor, pool, responses, target_rows, best,
                           fixed, rhs, a_eq, eq_rhs, bounds, prices, valid, assessments, learned, target_score=95.,
                           anchor_strict_floors=None, anchor_affinities=None):
    from .lotion_optimizer import _compact_profile_residuals
    shapes, targets, avoidance = prepare_balance(predictor, pool, target_rows)
    nr = len(shapes)
    def evaluate(w):
        return profile_values(shapes, targets, avoidance, responses, w)

    initial = best.copy()
    initial_values = evaluate(initial)
    original_scores = np.array([row['score'] for row in assessments(initial)[1]])
    floors = refinement_score_floors(original_scores, target_score)
    if anchor_strict_floors is not None:
        floors = np.maximum(floors, anchor_strict_floors)
    old_affinity = learned(initial)
    protected_affinity = old_affinity if anchor_affinities is None else np.maximum(old_affinity, anchor_affinities)
    report = {'version': VERSION, 'score_kind': 'full_learned_note_overlap_lower_bound_not_human_similarity',
              'axes': list(getattr(predictor, 'projection', LOTION_PROJECTION)) + ['unmapped_endpoint_mass'],
              'baseline_score': float(100*initial_values.min()),
              'baseline_reference_scores': (100*initial_values).tolist(),
              'previous_strict_scores': original_scores.tolist(), 'required_strict_scores': floors.tolist(),
              'previous_reference_affinities': (100*old_affinity).tolist(),
              'selected_score': float(100*initial_values.min()), 'recipe_changed': False,
              'solver_calls': 0, 'solver_statuses': [], 'solver_incomplete': False,
              'unmapped_endpoint_mass_dropped': False, 'human_similarity_percent': None}
    if initial_values.min() >= 1.-1e-8:
        report['status'] = 'already_at_model_upper_bound'
        return best, report
    blocks = [_compact_profile_residuals(shapes[r], targets, responses) for r in range(nr)]
    n, old_width = len(pool), fixed.shape[1]
    ns = blocks[0][1].shape[1]
    width = old_width+nr*ns

    def extend(matrix):
        return sparse.hstack([matrix, sparse.csr_matrix((matrix.shape[0], nr*ns))], format='csr')

    def place(matrix, r):
        return sparse.hstack([matrix[:, :n], sparse.csr_matrix((matrix.shape[0], old_width-n+r*ns)),
                              matrix[:, n:], sparse.csr_matrix((matrix.shape[0], (nr-1-r)*ns))], format='csr')

    constant = sparse.vstack([extend(fixed)] + [place(b[0], r) for r, b in enumerate(blocks)], format='csr')
    base_rhs = np.r_[rhs, np.zeros(constant.shape[0]-fixed.shape[0])]

    def quality_rows(level, preserve_initial=True):
        rows = []
        for r, (_, sums, outside) in enumerate(blocks):
            limits = np.maximum(initial_values[r], level) if preserve_initial else np.full(len(responses), level)
            rows.append(place(sparse.hstack([sparse.csr_matrix(outside-2*(1-limits[:, None])*responses), sums], format='csr'), r))
            rows.append(sparse.hstack([sparse.csr_matrix((avoidance[r]-(1-limits[:, None]))*responses),
                                       sparse.csr_matrix((len(responses), width-n))], format='csr'))
        return sparse.vstack(rows, format='csr')

    lo = float(initial_values.min())
    # Each mixture coordinate is bounded above by the largest candidate
    # coordinate. Ignore cost/transport restrictions here: this is only an
    # optimistic bound used to avoid futile solver probes, not a certificate.
    upper = np.minimum(shapes.max(axis=1)[:, None, :], targets[None, :, :]).sum(axis=2)
    hi = max(lo, min(1., float(upper.min())+1e-10))
    report['optimistic_full_balance_upper_score'] = 100*hi
    def bounded_solver(*args, **kwargs):
        kwargs['options'] = {**kwargs['options'], 'time_limit': .5}
        kwargs['method'] = 'highs-ds'
        return linprog(*args, **kwargs)

    # Preserve all starting curves separately from maximizing their minimum.
    # A generalized fractional step avoids spending an entire online budget
    # proving infeasible far-away quality levels on a large material pool.
    guard_rows = quality_rows(lo)
    guarded = sparse.vstack([constant, guard_rows], format='csr')
    guarded_rhs = np.r_[base_rhs, np.zeros(guard_rows.shape[0])]
    report['method'] = 'guarded_full_profile_fractional_lp'
    for step in range(5):
        if hi-lo < 1e-5:
            break
        rows = quality_rows(lo, preserve_initial=False)
        denominators = responses@best
        improvement = np.tile(np.r_[2*denominators, denominators], nr)
        matrix = sparse.vstack([
            sparse.hstack([guarded, sparse.csr_matrix((guarded.shape[0], 1))]),
            sparse.hstack([rows, sparse.csr_matrix(improvement[:, None])])], format='csr')
        equality = sparse.hstack([extend(a_eq), sparse.csr_matrix((a_eq.shape[0], 1))], format='csr')
        result = conditioned_linprog(bounded_solver, np.r_[np.zeros(width), -1.],
            A_ub=matrix, b_ub=np.r_[guarded_rhs, np.zeros(rows.shape[0])],
            A_eq=equality, b_eq=eq_rhs, bounds=bounds+[(0., None)]*(nr*ns)+[(0., 1.)], scales=np.ones(width+1))
        report['solver_calls'] += 1
        report['solver_statuses'].append(int(result.status))
        if result.status == 2:
            break
        if result.status != 0 or not result.success or result.x is None:
            report['solver_incomplete'] = True
            break
        candidate = np.maximum(0., result.x[:n])
        if not np.isfinite(candidate).all() or candidate.sum() <= 0:
            report['solver_incomplete'] = True
            break
        candidate /= candidate.sum()
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            proposal = best+fraction*(candidate-best)
            values = evaluate(proposal)
            strict = np.array([row['score'] for row in assessments(proposal)[1]])
            if (valid(proposal) and np.all(strict+1e-8 >= floors)
                    and np.all(learned(proposal)+1e-9 >= protected_affinity)
                    and np.all(values+1e-9 >= initial_values) and values.min() > lo+1e-7):
                best = proposal.copy()
                lo = float(values.min())
                accepted = True
                break
        if not accepted:
            break
    report.update(selected_score=float(100*evaluate(best).min()), selected_reference_scores=(100*evaluate(best)).tolist(),
                  recipe_changed=bool(np.max(np.abs(best-initial)) > 1e-10),
                  status='verified_profile_balance_improvement' if np.max(np.abs(best-initial)) > 1e-10 else 'no_verified_profile_improvement')
    return best, report
