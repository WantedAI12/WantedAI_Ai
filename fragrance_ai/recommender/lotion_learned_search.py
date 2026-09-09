"""Secondary robust learned affinity search under quality and user budget guards.

Stock-reference shapes are an unvalidated research prior, not lotion sensory
truth. Missing graphs receive the worst endpoint affinity, never zero padding
followed by normalization. All screened candidates remain in the search pool.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .lotion_evaluation import LOTION_PROJECTION as PROJECTION
from .lotion_evaluation import refinement_score_floors
from .lotion_numerics import conditioned_linprog


def _roundoff_difference(left, right):
    """Do not amplify subtraction roundoff into a unit LP constraint.

    The cutoff scales with the operands, not with transported mass or an
    absolute epsilon. Every proposal still passes the original exact guards.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    delta = left-right
    uncertainty = 8*np.finfo(float).eps*np.maximum(np.abs(left), np.abs(right))
    return np.where(np.abs(delta) <= uncertainty, 0., delta)


def fractional_refinement(*, best, learned, affinity, responses, fixed, rhs, affinity_rows,
                          residual_rows, slack_sums, outside, avoid_vectors, profiles,
                          a_eq, eq_rhs, bounds, prices, valid, assessments, target_score=95.,
                          anchor_strict_floors=None, anchor_affinities=None):
    """Generalized fractional steps with exact guards and cost-only polishing.

    Keep the previous solver's incumbent. Equivalent row normalization avoids
    silently discarding weak response constraints. The acceptance target and
    worst-point score stay protected while non-worst surplus may be adjusted.
    """
    n, ns = len(best), slack_sums.shape[1]
    previous = best.copy()
    previous_values = learned(best)
    previous_floors = np.array([x['score'] for x in assessments(best)[1]])
    protected_floors = refinement_score_floors(previous_floors, target_score)
    if anchor_strict_floors is not None:
        protected_floors = np.maximum(protected_floors, anchor_strict_floors)
    protected_affinities = previous_values if anchor_affinities is None else np.maximum(previous_values, anchor_affinities)
    overlap = protected_floors/100.
    guards = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(outside-2*(1-overlap[:, None])*responses), slack_sums]),
        sparse.hstack([sparse.csr_matrix((avoid_vectors@profiles.T-(1-overlap[:, None]))*responses),
                       sparse.csr_matrix((len(responses), ns))]),
        affinity_rows(protected_affinities)], format='csr')
    fixed = sparse.vstack([fixed, guards], format='csr')
    rhs = np.r_[rhs, np.zeros(guards.shape[0])]
    diagnostics = {'method': 'guarded_generalized_fractional_lp_and_cost_polish', 'solver_calls': 0,
                   'solver_statuses': [], 'accepted_steps': 0, 'cost_polish_applied': False,
                   'previous_affinity': float(100*previous_values.min()),
                   'previous_cost_per_kg': float(prices@previous),
                   'previous_strict_scores': previous_floors.tolist(),
                   'required_strict_scores': protected_floors.tolist(),
                   'previous_reference_affinities': (100*previous_values).tolist(),
                   'anchor_policy': 'original_and_stage_guards_rechecked_no_cumulative_tolerance'}

    def acceptable(candidate, reference_values, reference_floors):
        if not valid(candidate):
            return False
        actual = np.array([x['score'] for x in assessments(candidate)[1]])
        # Checking only the immediately preceding point spends the numerical
        # tolerance again on every iteration. Always retain the original
        # and stage-entry floors, which the final transport check also uses.
        return bool(np.all(actual+1e-8 >= np.maximum(reference_floors, protected_floors))
                    and np.all(learned(candidate)+1e-9 >= np.maximum(reference_values, protected_affinities)))

    for _ in range(5):
        values = learned(best)
        level = float(values.min())
        if level >= 1.-1e-8:
            break
        denominator = responses@best
        # z measures the minimum improvement using the current physical
        # denominator. A positive z improves every ratio above current min.
        gaps = ((level-affinity)*responses[None, :, :]/denominator[None, :, None]).reshape(-1, n)
        improvement = sparse.hstack([sparse.csr_matrix(gaps), sparse.csr_matrix((len(gaps), ns)),
                                     sparse.csr_matrix(np.ones((len(gaps), 1)))])
        matrix = sparse.vstack([sparse.hstack([fixed, sparse.csr_matrix((fixed.shape[0], 1))]), improvement], format='csr')
        result = conditioned_linprog(linprog, np.r_[np.zeros(n+ns), -1.], A_ub=matrix,
            b_ub=np.r_[rhs, np.zeros(len(gaps))],
            A_eq=sparse.hstack([a_eq, sparse.csr_matrix((a_eq.shape[0], 1))]), b_eq=eq_rhs,
            bounds=bounds+[(0., 2.)], scales=np.ones(n+ns+1))
        diagnostics['solver_calls'] += 1
        diagnostics['solver_statuses'].append(int(result.status))
        if result.status != 0 or not result.success or result.x is None:
            break
        candidate = np.maximum(0., result.x[:n])
        if not np.isfinite(candidate).all() or candidate.sum() <= 0:
            break
        candidate /= candidate.sum()
        current_floors = refinement_score_floors([x['score'] for x in assessments(best)[1]], target_score)
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            proposal = best+(candidate-best)*fraction
            if (acceptable(proposal, values, current_floors)
                    and learned(proposal).min() > level+1e-7):
                best, accepted = proposal.copy(), True
                diagnostics['accepted_steps'] += 1
                break
        if not accepted:
            break
    # Remove unnecessary cost at the achieved per-scenario affinity; never
    # trade quality for a cheaper blend or report a price estimate as a quote.
    if np.max(np.abs(best-previous)) > 1e-10:
        values = learned(best)
        floors = np.array([x['score'] for x in assessments(best)[1]])
        q = floors/100.
        polish_guards = sparse.vstack([affinity_rows(values),
            sparse.hstack([sparse.csr_matrix(outside-2*(1-q[:, None])*responses), slack_sums]),
            sparse.hstack([sparse.csr_matrix((avoid_vectors@profiles.T-(1-q[:, None]))*responses),
                           sparse.csr_matrix((len(responses), ns))])], format='csr')
        result = conditioned_linprog(linprog, np.r_[prices, np.zeros(ns)],
            A_ub=sparse.vstack([fixed, polish_guards], format='csr'), b_ub=np.r_[rhs, np.zeros(polish_guards.shape[0])],
            A_eq=a_eq, b_eq=eq_rhs, bounds=bounds, scales=np.ones(n+ns))
        diagnostics['solver_calls'] += 1
        diagnostics['solver_statuses'].append(int(result.status))
        if result.status == 0 and result.success and result.x is not None:
            candidate = np.maximum(0., result.x[:n])
            if np.isfinite(candidate).all() and candidate.sum() > 0:
                candidate /= candidate.sum()
                if acceptable(candidate, values, floors) and prices@candidate < prices@best-1e-7:
                    best = candidate
                    diagnostics['cost_polish_applied'] = True
    return best, diagnostics


def target_coefficients(target_rows, endpoints, projection=PROJECTION):
    coefficients = np.zeros((len(target_rows), len(endpoints)))
    for t, row in enumerate(target_rows):
        largest = max(row['target_profile'].values(), default=0.)
        if largest <= 0:
            raise ValueError('learned lotion objective requires a positive target')
        partial = any(weight > 0 and axis not in projection for axis, weight in row['target_profile'].items())
        # Unsupported target mass gets no credit and is not renormalized to
        # a perfect known-axis score. This is a secondary partial objective;
        # all original axes still participate in the independent strict score.
        denominator = sum(row['target_profile'].values()) if partial else largest
        for axis, weight in row['target_profile'].items():
            if weight > 0 and axis in projection:
                for name in projection[axis]:
                    coefficients[t, endpoints.index(name)] = weight/denominator
        for axis in row['avoided']:
            if axis not in projection:
                raise ValueError('unmodeled avoidance cannot be silently omitted')
            for name in projection[axis]:
                coefficients[t, endpoints.index(name)] = -1.
    return coefficients


def fresh_curve_affinity(results, target_rows, pool, predictor):
    """Re-evaluate from final simulated curves, not the LP response matrix."""
    known = {item.ingredient_id: item for item in pool}
    coefficients = target_coefficients(target_rows, predictor.provider.endpoints, getattr(predictor, 'projection', PROJECTION))
    points = [point for result in results for point in result['temporal_profile'][1:]]
    if len(points) != len(target_rows):
        raise ValueError('learned lotion final timepoint mismatch')
    values = np.zeros((getattr(predictor, 'reference_count', 3), len(points)))
    for t, point in enumerate(points):
        key = 'odor_activity_proxy' if point['profile_basis'] == 'linear_odor_activity_proxy' else 'air_concentration_mg_m3'
        total = sum(row[key] for row in point['materials'])
        if not np.isfinite(total) or total <= 0:
            raise ValueError('learned lotion final headspace is invalid')
        for row in point['materials']:
            shape = predictor.shape(known[row['ingredient_id']])
            values[:, t] += (shape@coefficients[t] if shape is not None else -1.)*(row[key]/total)
    return values


def refine_lotion_weights(*, predictor, pool, responses, target_rows, incumbent,
                          residual_rows, slack_sums, outside, avoid_vectors, profiles,
                          fixed_rows, fixed_rhs, a_eq, eq_rhs, bounds, prices, valid, assessments, target_score=95.):
    projection, nr = getattr(predictor, 'projection', PROJECTION), getattr(predictor, 'reference_count', 3)
    report = {'version': 'lotion-learned-refinement/v5', 'enabled': True, 'recipe_changed': False,
              'status': 'no_verified_improvement', 'solver_calls': 0, 'solver_incomplete': False,
              'proposal_solver_statuses': [],
              'score_kind': 'worst_reference_profile_target_affinity_not_similarity',
              'affinity_range': [-100., 100.], 'acceptance_threshold_modified': False,
              'human_similarity_percent': None, 'candidate_count': len(pool),
              'reference_profile_scenarios': list(getattr(predictor, 'reference_scenarios', ())),
              'reference_profile_family': getattr(predictor, 'bridge_version', None)}
    unsupported = sorted({axis for row in target_rows for axis in
        list(row['avoided']) + [key for key, value in row['target_profile'].items() if value > 0]
        if axis not in projection})
    unsupported_avoided = sorted({axis for row in target_rows for axis in row['avoided'] if axis not in projection})
    coverage = [sum(v for k,v in row['target_profile'].items() if k in projection)
                /sum(row['target_profile'].values()) for row in target_rows]
    report.update(unsupported_target_axes=unsupported, unsupported_avoided_axes=unsupported_avoided,
        modeled_target_mass_fractions=coverage, all_target_axes_modeled=not unsupported,
        partial_target_guidance=bool(unsupported) and not unsupported_avoided and min(coverage) > 0)
    if unsupported and (unsupported_avoided or min(coverage) <= 0):
        report.update(status='unsupported_target_axes', unsupported_target_axes=unsupported)
        return incumbent, report
    if unsupported:
        report['score_kind'] = 'partial_known_axis_affinity_unmodeled_target_mass_gets_no_credit'
    endpoints = predictor.provider.endpoints
    coefficients = target_coefficients(target_rows, endpoints, projection)
    # -1 is a conservative bound on any unknown normalized endpoint shape.
    affinity = np.full((nr, len(target_rows), len(pool)), -1.)
    missing = []
    predictor.prefetch(pool)
    for i, item in enumerate(pool):
        shape = predictor.shape(item)
        if shape is None:
            missing.append(item.ingredient_id)
        else:
            affinity[:, :, i] = shape @ coefficients.T
    report['unmapped_candidate_ids'] = missing
    if len(missing) == len(pool):
        report['status'] = 'no_supported_candidate'
        return incumbent, report
    def learned(weights):
        contribution = responses * weights
        return np.sum(affinity*contribution[None, :, :], axis=2) / contribution.sum(axis=1)[None, :]
    baseline_values = learned(incumbent)
    _, checks = assessments(incumbent)
    original_scores = np.array([row['score'] for row in checks])
    floors = refinement_score_floors(original_scores, target_score)
    report.update(original_strict_scores=original_scores.tolist(), required_strict_scores=floors.tolist(),
        strict_score_policy='preserve_minimum_failed_points_and_attained_target', target_score=target_score)
    n, ns = len(pool), slack_sums.shape[1]
    overlap = floors/100.
    variation = sparse.hstack([sparse.csr_matrix(outside-2*(1-overlap[:, None])*responses), slack_sums])
    avoidance = sparse.hstack([sparse.csr_matrix((avoid_vectors@profiles.T-(1-overlap[:, None]))*responses),
                               sparse.csr_matrix((len(target_rows), ns))])
    # fixed_rows retains the user's cost cap. Do not freeze the cheapest
    # incumbent's cost: that can uniquely fix its recipe and erase all search.
    fixed = sparse.vstack([residual_rows, variation, avoidance, fixed_rows], format='csr')
    rhs = np.r_[np.zeros(residual_rows.shape[0]+2*len(target_rows)), fixed_rhs]
    # Preserve every learned scenario as well as improving the worst one.
    def affinity_rows(levels):
        values = _roundoff_difference(levels[:, :, None], affinity)*responses[None, :, :]
        return sparse.hstack([sparse.csr_matrix(values.reshape(-1, n)),
                              sparse.csr_matrix((nr*len(target_rows), ns))], format='csr')
    fixed = sparse.vstack([fixed, affinity_rows(baseline_values)], format='csr')
    rhs = np.r_[rhs, np.zeros(nr*len(target_rows))]
    best = incumbent.copy()
    lo, hi = float(baseline_values.min()), 1.
    report.update(baseline_affinity=100*lo, baseline_strict_score=float(floors.min()),
                  baseline_cost_per_kg=float(prices@incumbent), baseline_reference_affinities=(100*baseline_values).tolist())
    for probe in range(9):
        if hi-lo < 1e-5:
            break
        # A coarse global bisection alone misses improvements smaller than
        # its 1/256 interval. End with a local probe, without relaxing guards.
        middle = min(hi, lo+1e-6) if probe == 8 else (lo+hi)/2.
        matrix = sparse.vstack([fixed, affinity_rows(np.full_like(baseline_values, middle))], format='csr')
        result = linprog(np.r_[prices/max(float(prices.max()), 1.), np.zeros(ns)],
            A_ub=matrix, b_ub=np.r_[rhs, np.zeros(nr*len(target_rows))], A_eq=a_eq, b_eq=eq_rhs,
            bounds=bounds, method='highs', options={'time_limit': .5,
                'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        report['solver_calls'] += 1
        report['proposal_solver_statuses'].append(int(result.status))
        if result.status == 2:
            hi = middle
            continue
        if result.status != 0 or not result.success or result.x is None:
            report['solver_incomplete'] = True
            break
        candidate = np.maximum(0., result.x[:n])
        if not np.isfinite(candidate).all() or candidate.sum() <= 0:
            report['solver_incomplete'] = True
            break
        candidate /= candidate.sum()
        accepted = False
        # Exact cosine guards may be stricter than the LP overlap surrogate.
        for fraction in (1., .5, .25, .125, .0625):
            proposal = incumbent+(candidate-incumbent)*fraction
            if not valid(proposal):
                continue
            _, actual = assessments(proposal)
            values = learned(proposal)
            if (np.any(np.array([x['score'] for x in actual])+1e-8 < floors)
                    or np.any(values+1e-9 < baseline_values)
                    or values.min() <= learned(best).min()+1e-7):
                continue
            best, accepted = proposal.copy(), True
            lo = max(lo, float(values.min()))
            break
        if not accepted:
            hi = middle
    best, refinement = fractional_refinement(best=best, learned=learned, affinity=affinity, responses=responses,
        fixed=fixed, rhs=rhs, affinity_rows=affinity_rows, residual_rows=residual_rows,
        slack_sums=slack_sums, outside=outside, avoid_vectors=avoid_vectors, profiles=profiles,
        a_eq=a_eq, eq_rhs=eq_rhs, bounds=bounds, prices=prices, valid=valid, assessments=assessments,
        target_score=target_score, anchor_strict_floors=floors, anchor_affinities=baseline_values)
    report['precision_refinement'] = refinement
    report['solver_calls'] += refinement['solver_calls']
    report['solver_incomplete'] = report['solver_incomplete'] or any(x not in (0, 2) for x in refinement['solver_statuses'])
    # A verified incumbent remains usable after an earlier solver timeout or
    # numerical stop. Mixed targets need a balance objective even in that case.
    # Single-note targets already order known shapes by that note's mass.
    if unsupported:
        report['profile_balance_status'] = 'partial_affinity_only_all_axes_retained_in_strict_evaluation'
    elif any(sum(value > 0 for value in row['target_profile'].values()) > 1 for row in target_rows):
        from .lotion_profile_balance import refine_profile_balance
        current_floors = refinement_score_floors([row['score'] for row in assessments(best)[1]], target_score)/100.
        balance_guards = sparse.vstack([
            sparse.hstack([sparse.csr_matrix(outside-2*(1-current_floors[:, None])*responses), slack_sums]),
            sparse.hstack([sparse.csr_matrix((avoid_vectors@profiles.T-(1-current_floors[:, None]))*responses),
                           sparse.csr_matrix((len(responses), ns))]), affinity_rows(learned(best))], format='csr')
        best, balance = refine_profile_balance(predictor=predictor, pool=pool, responses=responses,
            target_rows=target_rows, best=best, fixed=sparse.vstack([fixed, balance_guards], format='csr'),
            rhs=np.r_[rhs, np.zeros(balance_guards.shape[0])], a_eq=a_eq, eq_rhs=eq_rhs, bounds=bounds,
            prices=prices, valid=valid, assessments=assessments, learned=learned, target_score=target_score,
            anchor_strict_floors=np.maximum(floors, refinement['required_strict_scores']),
            anchor_affinities=np.maximum(baseline_values, np.asarray(refinement['previous_reference_affinities'])/100.))
        report['profile_balance'] = balance
        report['solver_calls'] += balance['solver_calls']
        report['solver_incomplete'] = report['solver_incomplete'] or balance['solver_incomplete']
    else:
        report['profile_balance_status'] = 'single_note_target_uses_existing_affinity_search'
    report.update(recipe_changed=bool(np.max(np.abs(best-incumbent)) > 1e-10),
                  selected_affinity=float(100*learned(best).min()),
                  selected_strict_score=assessments(best)[0], selected_cost_per_kg=float(prices@best))
    report['cost_change_per_kg'] = report['selected_cost_per_kg']-report['baseline_cost_per_kg']
    report['cost_policy'] = 'user_budget_cap_preserved_not_incumbent_cost_frozen'
    if report['recipe_changed']:
        report['status'] = 'verified_nonregressing_research_refinement'
    return best, report
