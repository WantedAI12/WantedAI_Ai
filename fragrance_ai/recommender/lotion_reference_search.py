"""Whole-pool inverse design against independently frozen full odor references.

Positive finite-dose transport is the authoritative forward model. The learned
Atlas profiles enter the objective itself, not only an after-the-fact report.
Linear constraints implement the SAME full-vector L1 overlap as the final check;
no descriptor subset, score clipping, target adaptation or oracle recipe.
"""
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeResult

from .adaptive_pyramid import explicit_pyramid
from .catalog import IngredientCatalog, normalize_name
from .lotion import _simulate_lotion_transport, EXPOSURE_INTEGRATION_VERSION
from .lotion_evaluation import compare_lotion_profiles
from .lotion_incumbent import incumbent_weights
from .lotion_reference_objective import exposure_groups
from .lotion_numerics import conditioned_linprog, response_variable_scales
from .search_budget import governed, allowance, exhausted
from .reference_discrimination import using_reference_policy


def _solve_with_time_recovery(objective, parameters, diagnostics):
    """Retry a timed-out working LP before making its column set larger.

    The retry changes only the wall-clock allowance, not the optimization
    problem or feasibility tolerances. Reserve at most nine extra seconds
    across the whole inverse-design search, including elastic Phase I.
    """
    if exhausted():
        diagnostics['skipped_after_deadline'] = diagnostics.get('skipped_after_deadline',0)+1
        return OptimizeResult(success=False,status=1,x=None,message='request computation budget exhausted')
    result = linprog(objective, **parameters)
    diagnostics['linear_solves'] += 1
    timed_out = result.status == 1 and 'time limit' in str(result.message).lower()
    if not timed_out:
        return result
    diagnostics['time_limit_hits'] = diagnostics.get('time_limit_hits', 0) + 1
    reserved = diagnostics.get('time_limit_retry_reserved_seconds', 0.)
    retry = allowance('linear_retry',3.)
    retry_total = 24. if retry>3. else 9.
    if exhausted() or reserved + retry > retry_total:
        diagnostics['time_limit_retry_budget_exhausted'] = True
        return result
    diagnostics['time_limit_retry_reserved_seconds'] = reserved + retry
    diagnostics['time_limit_retries'] = diagnostics.get('time_limit_retries', 0) + 1
    retry_parameters = {**parameters, 'options': {
        **parameters.get('options', {}), 'time_limit': retry}}
    result = linprog(objective, **retry_parameters)
    diagnostics['linear_solves'] += 1
    if result.success:
        diagnostics['time_limit_retry_successes'] = diagnostics.get('time_limit_retry_successes', 0) + 1
    return result


def _feasibility_columns(parameters, full_parameters, missing, diagnostics):
    """Elastic Phase I dual pricing; artificial violations never form a recipe.

    Original variable bounds remain fixed. Nonnegative inequality violations
    and signed equality violations make the restricted row system feasible.
    An omitted zero-lower-bound column can reduce that violation exactly when
    its Phase I reduced cost is negative. No scent ranking is used here.
    """
    width = len(parameters['bounds'])
    m, k = parameters['A_ub'].shape[0], parameters['A_eq'].shape[0]
    artificial = m + 2*k
    previous_solves = diagnostics['linear_solves']
    phase = _solve_with_time_recovery(np.r_[np.zeros(width), np.ones(artificial)], {
        'A_ub': sparse.hstack((parameters['A_ub'], -sparse.eye(m), sparse.csr_matrix((m, 2*k))), format='csr'),
        'b_ub': parameters['b_ub'],
        'A_eq': sparse.hstack((parameters['A_eq'], sparse.csr_matrix((k, m)), sparse.eye(k), -sparse.eye(k)), format='csr'),
        'b_eq': parameters['b_eq'], 'bounds': parameters['bounds']+[(0., None)]*artificial,
        'method': 'highs-ds', 'options': parameters['options']}, diagnostics)
    diagnostics['phase_one_solves'] = diagnostics.get('phase_one_solves', 0) + diagnostics['linear_solves'] - previous_solves
    if not phase.success or phase.x is None:
        return []
    reduced = (-full_parameters['A_ub'][:, missing].T@phase.ineqlin.marginals
               -full_parameters['A_eq'][:, missing].T@phase.eqlin.marginals)
    diagnostics['phase_one_pricing_passes'] = diagnostics.get('phase_one_pricing_passes', 0) + 1
    negative = np.flatnonzero(reduced < -1e-9)
    return missing[negative[np.argsort(reduced[negative], kind='stable')[:64]]].tolist()


def column_linprog(objective, *, material_count, initial_columns, column_order, diagnostics, **kwargs):
    """Bounded working LP; every omitted material gets reduced-cost pricing.

    This does not impose a recipe cardinality limit or change any full LP
    constraint. Restricted infeasibility is NEVER reported as full-pool
    infeasibility. On exhausted expansion it is explicitly incomplete.
    """
    objective = np.asarray(objective)
    n = material_count
    active = set(initial_columns) | {i for i,(lo,_) in enumerate(kwargs['bounds'][:n]) if lo and lo > 0}
    active.update(column_order[:min(64,n)])
    auxiliary = list(range(n,len(objective)))
    retained = None
    def retain_or_fail(failed):
        if retained is not None:
            diagnostics['feasible_incumbent_retained_after_expansion_failure'] = True
            diagnostics['expansion_budget_reached'] = True
            return OptimizeResult(**{**dict(retained),
                'message':'feasible restricted incumbent retained; full optimality not established',
                'full_pool_optimality_certified':False})
        return failed
    for attempt in range(6):
        indices = np.asarray(sorted(active)+auxiliary, dtype=int)
        parameters = {**kwargs, 'A_ub': kwargs['A_ub'][:,indices],
            'A_eq': kwargs['A_eq'][:,indices], 'bounds': [kwargs['bounds'][i] for i in indices]}
        # Dense presolve/IPM fill-in dominated the full 146-D experiment.
        # Dual simplex with explicit work bounds avoids that unbounded setup.
        parameters['method'] = 'highs-ds'
        parameters['options'] = {**kwargs.get('options', {}), 'presolve': False,
            'time_limit':allowance('linear',.75,len(active))}
        result = _solve_with_time_recovery(objective[indices], parameters, diagnostics)
        diagnostics['maximum_working_materials'] = max(diagnostics['maximum_working_materials'],len(active))
        if result.success and result.x is not None:
            full = np.zeros(len(objective))
            full[indices] = result.x
            # HiGHS multipliers correspond to the already conditioned rows.
            # No hand-chosen odor shortlist determines the final candidate set.
            reduced = (objective[:n] - kwargs['A_ub'][:,:n].T@result.ineqlin.marginals
                       - kwargs['A_eq'][:,:n].T@result.eqlin.marginals)
            missing = np.asarray([i for i in range(n) if i not in active and kwargs['bounds'][i][1] != 0],dtype=int)
            diagnostics['full_pool_pricing_passes'] += 1
            diagnostics['material_columns_priced'] = n
            negative = missing[reduced[missing] < -1e-9]
            result.x = full
            if np.isfinite(full).all() and (retained is None or objective@full < objective@retained.x):
                retained = OptimizeResult(**dict(result))
            if not len(negative):
                return result
            if attempt == 5:
                # Keep a feasible proposal, but do not claim a full cost optimum.
                diagnostics['expansion_budget_reached'] = True
                return result
            active.update(negative[np.argsort(reduced[negative])[:64]].tolist())
        else:
            if len(active) == n:
                return retain_or_fail(result)
            if attempt == 5 or result.status not in (1,2,4):
                diagnostics['expansion_budget_reached'] = True
                return retain_or_fail(OptimizeResult(success=False,status=1,x=None,
                    message='working-column expansion incomplete; full feasibility not established'))
            if result.status == 2:
                missing = np.asarray([i for i in range(n) if i not in active and kwargs['bounds'][i][1] != 0], dtype=int)
                additions = _feasibility_columns(parameters, kwargs, missing, diagnostics)
                diagnostics['material_columns_priced'] = n
                if additions:
                    active.update(additions)
                    continue
            # Failed Phase I or numerical/iteration limits cannot certify full
            # infeasibility. Keep the bounded independent expansion fallback.
            count = min(n, max(128, 2*len(active)))
            active.update(column_order[:count])
    raise AssertionError('unreachable column loop')


@governed('body_lotion')
@using_reference_policy
def optimize_observed_reference(*, request, prepared, brief, pool, basis_requests,
        responses, physical, uptake_blocks, predictor, bank, baseline,
        cache_status, reused_simulations, exposure_blocks, incumbent_recipe=None):
    targets, unsupported = bank.targets(brief, prepared['evaluation_targets'])
    from .odor_space import target_coverage
    coverage = target_coverage(targets, unsupported)
    # Partial handling is explicit in the hierarchical compiler. Preserve the
    # older bank contract whose missing strings have no polarity/type.
    if getattr(bank, 'odor_space', None) is None and unsupported:
        coverage['searchable'] = False
    contract = bank.contract()
    prepared = {**prepared, 'product_model': contract,
        'intent': {**prepared['intent'], 'representation': {
            **prepared['intent']['representation'], 'weight_semantics': 'semantic_query_weights_for_observed_full_profiles',
            'unrequested_axes_remain_in_strict_comparison': False,
            'zero_target_axis_semantics': 'not_requested_not_zero_in_observed_reference',
            'learned_coverage_is_not_primary_19_axis_score': False}}}
    common = {'schema_version': 'lotion-optimization-1', 'product_model': contract,
        'preparation': prepared, 'perceptual_evaluation': {**contract,
            'unsupported_requirements': unsupported, 'target_coverage': coverage,
            'absolute_intensity_target_evaluated': False,
            'minimum_odor_activity_proxy': 1., 'target_threshold': request.target_similarity,
            'presence_scope': 'mean_window_odor_activity_and_separate_pointwise_air_concentration_floor',
            'minimum_air_concentration_mg_m3': request.minimum_air_concentration_mg_m3,
            'air_concentration_floor_scope': 'every_requested_output_time_except_zero',
            'exposure_integration_version': EXPOSURE_INTEGRATION_VERSION},
        'score_kind': 'exposure_integrated_observed_reference_agreement_not_user_similarity',
        'human_similarity_percent': None, 'manufacturing_approved': False,
        'all_user_requirements_verified': False, 'external_api_calls': 0}
    if not coverage['searchable']:
        return {**common, 'status': 'insufficient_observed_target_coverage', 'score': None,
            'profile_target_met': False, 'recipe': [], 'closest_candidate': [],
            'solver_calls': 0, 'search_incomplete': False,
            'transport_simulation_calls': len(basis_requests)-reused_simulations,
            'reused_basis_simulations': reused_simulations, 'basis_cache_status': cache_status}
    if any(s.profile_weighting != 'odor_activity' for s in basis_requests):
        raise ValueError('observed-reference design requires complete odor-threshold weighting')
    n, d = len(pool), len(bank.endpoints)
    original_targets = targets
    original_target_rows = prepared['evaluation_targets']
    groups = exposure_groups(brief, original_target_rows, request.transition_schedule)
    groups = [{**g, 'aggregation': EXPOSURE_INTEGRATION_VERSION} for g in groups]
    windows = [(g['window_start_minutes'], g['minutes']) for g in groups]
    targets = [original_targets[g['target_index']] for g in groups]
    thresholds = np.vstack([np.tile([m.odor_threshold_mg_m3 for m in s.materials],
                                    (len(groups), 1)) for s in basis_requests])
    absolute_oav = np.vstack(exposure_blocks)/thresholds
    if absolute_oav.shape != (len(groups)*len(basis_requests), n) or not np.isfinite(absolute_oav).all() or np.any(absolute_oav < 0):
        raise ValueError('finite nonnegative analytic exposure basis required')
    responses = absolute_oav/np.maximum(absolute_oav.max(axis=1,keepdims=True),1e-300)
    t_count = len(responses)
    target_info = targets * len(basis_requests)
    wanted = np.stack([t['profiles'] for t in target_info], axis=1)  # head,time,descriptor
    predictor.prefetch(pool)
    shapes = np.zeros((2, n, d))
    covered = np.ones(n, bool)
    for i, item in enumerate(pool):
        value = predictor.shape(item)
        if value is None:
            covered[i] = False
        else:
            shapes[:, i] = value
    from .lotion_coverage import profile_overlap_upper
    bounds_seen, profile_bounds = {}, []
    for target_info_row in targets:
        for h in range(2):
            vector = target_info_row['profiles'][h]
            key = (h, tuple(vector))
            if key not in bounds_seen:
                bounds_seen[key] = (profile_overlap_upper(shapes[h, covered], vector)
                                   if covered.any() else 0.)
            profile_bounds.append(bounds_seen[key])
    profile_upper = min(profile_bounds)
    common['perceptual_evaluation']['profile_coverage'] = {
        'optimistic_profile_upper_percent': profile_upper,
        'target_excluded': profile_upper < request.target_similarity-1e-8,
        'scope': 'full_146_endpoint_learned_profile_hull_ignoring_transport_and_cost',
        'bound_method': 'recomputed_clipped_dual_overlap_inequality_with_numeric_margin',
        'human_accuracy_measured': False}
    caps = np.minimum(1., np.asarray([i.as_supplied_cap_percent()/100 for i in pool]))
    # Missing learned identities are exposed, not replaced with a favorable
    # zero-odor vector. They cannot enter a scored formula in this mode.
    caps[~covered] = 0
    required = {normalize_name(x) for x in brief.requested_ingredients}
    lower = np.asarray([1e-6 if normalize_name(i.name) in required else 0. for i in pool])
    if np.any(lower > caps):
        raise ValueError('requested material lacks a learned full-odor profile')
    prices = np.asarray([i.price_per_kg for i in pool])
    notes, _ = explicit_pyramid(request.brief)
    eq = np.asarray([np.ones(n)] + [np.asarray([float(i.pyramid == note) for i in pool]) for note in notes])
    eq_rhs = np.asarray([1.] + [notes[n]/100 for n in notes])
    fixed = [prices/request.max_formula_cost_per_kg]
    rhs = [1.]
    for matrix, floor in ((physical, request.minimum_air_concentration_mg_m3), (absolute_oav, 1.)):
        scale = np.maximum(matrix.max(axis=1), 1e-300)
        fixed.extend(-matrix/scale[:, None])
        rhs.extend(-floor/scale)
    if request.maximum_modeled_uptake_mg_cm2 is not None:
        fixed.extend(np.asarray(uptake_blocks)/request.maximum_modeled_uptake_mg_cm2)
        rhs.extend(np.ones(len(basis_requests)))
    fixed, rhs = np.asarray(fixed), np.asarray(rhs)
    avoid = np.asarray([[float(e in t['avoided']) for e in bank.endpoints] for t in target_info])
    avoidance = np.einsum('hnd,td->htn', shapes, avoid)
    bounds = list(zip(lower, caps))

    def valid(x):
        return (np.isfinite(x).all() and np.all(x >= lower-1e-10) and np.all(x <= caps+1e-10)
            and np.max(np.abs(eq@x-eq_rhs)) <= 1e-8
            and np.all(fixed@x <= rhs+1e-9)
            and np.all(physical@x >= request.minimum_air_concentration_mg_m3*(1-1e-8))
            and np.all(absolute_oav@x >= 1.-1e-8))

    def predictions(x):
        weights = responses*x
        den = weights.sum(axis=1)
        if np.any(den <= 0):
            raise ValueError('empty final headspace')
        return np.einsum('tn,hnd->htd', weights/den[:, None], shapes, optimize=True)

    def evaluate(x, detail=False):
        predicted = predictions(x)
        # Fast exact overlap; the final comparison also rechecks cosine and bans.
        overlaps = 100*np.minimum(predicted, wanted).sum(axis=2)
        cosines = 100*np.einsum('htd,htd->ht', predicted, wanted)/(
            np.linalg.norm(predicted, axis=2)*np.linalg.norm(wanted, axis=2))
        avoided = 100*(1-np.einsum('htd,td->ht', predicted, avoid))
        score = float(np.clip(np.minimum(np.minimum(overlaps, cosines), avoided).min(), 0, 100))
        return (score, predicted, overlaps) if detail else score

    unit_targets = wanted/np.linalg.norm(wanted, axis=2, keepdims=True)
    from .reference_discrimination import contrast_direction
    discrimination = contrast_direction(wanted,bank.background[:,None,:])

    def is_specific(predicted):
        differences = np.einsum('htd,htd->ht', discrimination, predicted)/np.linalg.norm(predicted, axis=2)
        return bool(np.all(differences > 1e-8))

    from .autoregressive_refinement import ResidualFeedback
    feedback = ResidualFeedback(target=request.target_similarity)
    best, best_score, best_specific = None, -1., False
    numerical_repairs={'applied':0,'rejected':0}
    from .retired_blends import RetiredBlendFilter
    retired_blends = RetiredBlendFilter('body_lotion',
        basis_requests[0].application_context.fragrance_concentration_percent)

    def consider(x, score, predicted):
        nonlocal best, best_score, best_specific
        from .numerical_feasibility import canonical_solver_weights
        canonical=canonical_solver_weights(x,lower,caps,valid)
        if canonical is None:
            numerical_repairs['rejected']+=1
            return
        if not np.array_equal(canonical,x):
            numerical_repairs['applied']+=1
            x=canonical
            score,predicted,_=evaluate(x,True)
        if retired_blends.reject_weights(pool, x):
            return
        specific = is_specific(predicted)
        # A numerically high but background-indistinguishable candidate must
        # not displace a recipe satisfying the unchanged final acceptance gate.
        rank = (score+1e-8 >= request.target_similarity and specific, score)
        previous = (best_score+1e-8 >= request.target_similarity and best_specific, best_score)
        if rank > previous:
            best, best_score, best_specific = x.copy(), score, specific
            feedback.observe(x, (wanted-predicted).ravel(), score, qualified=bool(rank[0]))

    def target_reached():
        return best_score+1e-8 >= request.target_similarity and best_specific

    seed, seed_status = incumbent_weights(incumbent_recipe, pool)
    incumbent_score = evaluate(seed) if seed is not None and valid(seed) else None
    for x in (baseline, seed):
        if x is not None and valid(x):
            score, predicted, _ = evaluate(x, True)
            consider(x, score, predicted)
    # Exact absolute residual system. A descriptor whose target is outside
    # every candidate's range has a known sign and needs no slack. It remains
    # in the objective, unlike dropping an unrequested or difficult note.
    residuals, residual_times = [], []
    outside = np.zeros((2*t_count, n))
    for h in range(2):
        minimum = shapes[h, covered].min(axis=0) if covered.any() else np.zeros(d)
        maximum = shapes[h, covered].max(axis=0) if covered.any() else np.zeros(d)
        for t in range(t_count):
            q = h*t_count+t
            delta = shapes[h]-wanted[h,t]
            positive, negative = minimum > wanted[h,t], maximum < wanted[h,t]
            unknown = ~(positive | negative)
            outside[q] = (delta[:,positive].sum(axis=1)-delta[:,negative].sum(axis=1))*responses[t]
            for axis in np.flatnonzero(unknown):
                residuals.append(delta[:,axis]*responses[t])
                residual_times.append(q)
    n_slack = len(residuals)
    identity = sparse.eye(n_slack, format='csr')
    residual_matrix = sparse.csr_matrix(np.asarray(residuals).reshape(n_slack,n))
    absolute_rows = sparse.vstack((sparse.hstack((residual_matrix,-identity)),
                                  sparse.hstack((-residual_matrix,-identity))), format='csr')
    slack_sums = sparse.csr_matrix((np.ones(n_slack),(residual_times,np.arange(n_slack))),
                                   shape=(2*t_count,n_slack))
    physical_rows = sparse.hstack((sparse.csr_matrix(fixed),sparse.csr_matrix((len(fixed),n_slack))),format='csr')
    equality = sparse.hstack((sparse.csr_matrix(eq),sparse.csr_matrix((len(eq),n_slack))),format='csr')
    repeated_responses = np.tile(responses,(2,1))
    full_bounds = bounds+[(0.,None)]*n_slack
    scales = response_variable_scales(responses,n_slack)
    objective = np.r_[prices/max(1.,prices.max()),np.zeros(n_slack)]
    from .odor_expression import expression_utility
    fine_utility, fine_diagnostic = expression_utility(pool,brief)
    if np.ptp(fine_utility)>1e-12:
        objective[:n] -= .15*fine_utility
    calls, incomplete = 0, False
    # Rank every candidate using the same complete reference shapes. Include
    # physically persistent, cheap and explicitly requested seeds as well.
    affinities = np.einsum('hnd,htd->htn',shapes,wanted,optimize=True)/np.maximum(
        np.linalg.norm(shapes,axis=2)[:,None,:]*np.linalg.norm(wanted,axis=2)[:,:,None],1e-300)
    ranking = np.argsort(-(affinities.min(axis=(0,1))+.05*fine_utility),kind='stable').tolist()
    initial = set(np.flatnonzero(best > 0).tolist()) if best is not None else set()
    initial.update(np.argsort(prices)[:8].tolist())
    initial.update(np.argmax(absolute_oav,axis=1).tolist())
    initial.update(np.argmax(affinities.reshape(-1,n),axis=1).tolist())
    for note in notes:
        eligible = [i for i,item in enumerate(pool) if item.pyramid == note and caps[i] > 0]
        initial.update(sorted(eligible,key=lambda i:prices[i])[:8])
    column_diagnostics = {'linear_solves':0,'maximum_working_materials':0,
        'full_pool_pricing_passes':0,'material_columns_priced':0,'expansion_budget_reached':False,
        'all_candidate_materials_ranked':n,'permanent_candidate_shortlist':False,
        'phase_one_solves':0, 'phase_one_pricing_passes':0,
        'cosine_separation_cuts':0, 'background_separation_cuts':0,
        'maximum_separation_rounds':0, 'numerical_state_repairs':numerical_repairs}

    # The API target is 95, not the historical search-only 95.2 buffer.
    # A 0.0001-point numeric margin is enough for the unrounded lotion output;
    # the full fresh forward calculation remains the acceptance authority.
    goal_level = min(1.,request.target_similarity/100+1e-6)
    from .failure_recovery import dose_correction_active
    if dose_correction_active() and not target_reached() and not exhausted() and profile_upper+1e-8 >= request.target_similarity:
        from .blend_correction import correct_blend
        correction, correction_report = correct_blend(
            shapes=shapes, wanted=wanted, responses=responses, fixed=fixed, rhs=rhs,
            equality=eq, equality_rhs=eq_rhs, bounds=bounds, initial=best,
            target=goal_level, avoided=avoid, background=bank.background,
            maximum_seconds=allowance('blend_correction',45.,n), maximum_rounds=96)
        column_diagnostics['blend_correction'] = correction_report
        column_diagnostics['linear_solves'] += correction_report['linear_solves']
        if correction is not None and valid(correction):
            value, predicted, _ = evaluate(correction, True)
            consider(correction, value, predicted)
    if not target_reached() and not exhausted() and profile_upper+1e-8 >= request.target_similarity:
        from .lotion_conic import solve_full_profile
        conic, conic_report = solve_full_profile(profiles=shapes,targets=wanted,responses=responses,
            target_score=100*goal_level,residual_rows=absolute_rows,slack_sums=slack_sums,
            outside=outside,avoid_vectors=np.tile(avoid,(2,1)),fixed_rows=physical_rows,fixed_rhs=rhs,
            a_eq=equality,eq_rhs=eq_rhs,bounds=full_bounds,objective=objective,
            background=bank.background,time_limit=allowance('cone',8.,n),initial_columns=sorted(initial),column_order=ranking)
        column_diagnostics['full_reference_cone'] = conic_report
        column_diagnostics['conic_solver_calls'] = conic_report.get('solver_calls',0)
        if conic is not None and valid(conic):
            value,predicted,_ = evaluate(conic,True)
            before = best_score
            consider(conic,value,predicted)
            conic_report.update(candidate_accepted=best_score>before,target_met=target_reached(),fresh_score=value)

    if not target_reached() and not exhausted():
        from .corrective_profile_search import corrective_profile_search
        corrected = corrective_profile_search(shapes=shapes, wanted=wanted,
            responses=responses, fixed=fixed, rhs=rhs, equality=eq,
            equality_rhs=eq_rhs, bounds=bounds, initial=best,
            target=goal_level, avoided=avoid,
            background=bank.background, maximum_seconds=allowance('corrective',8.,n))
        column_diagnostics['corrective_full_pool'] = {
            'status': int(corrected.status), 'success': bool(corrected.success),
            'rounds': corrected.rounds, 'priced_columns': corrected.priced_columns,
            'seconds': corrected.elapsed_seconds, 'candidate_score': corrected.candidate_score,
            'full_pool_count': n, 'full_endpoint_count': d,
            'permanent_candidate_shortlist': False,
            'linear_solves': corrected.linear_solves, 'nonlinear_solves': corrected.nonlinear_solves}
        column_diagnostics['linear_solves'] += corrected.linear_solves
        column_diagnostics['nonlinear_solves'] = corrected.nonlinear_solves
        corrected_candidate = corrected.x if corrected.success else corrected.candidate
        if corrected_candidate is not None and valid(corrected_candidate):
            value, predicted, _ = evaluate(corrected_candidate, True)
            consider(corrected_candidate, value, predicted)

    # Solve the same normalized profile inequalities without thousands of
    # descriptor slack variables first. Every omitted halfspace is separated
    # against every endpoint, time and learned head before accepting a recipe.
    # The original solver remains a recovery path; neither score nor caps change.
    if n > 256 and not target_reached() and not exhausted():
        from .fractional_profile_cuts import solve_profile_cuts
        def compact_solver(costs, **kwargs):
            # The compact problem has only material columns, not thousands of
            # descriptor slacks. Solve ALL of them in one conditioned LP.
            # Unscaled tiny release coefficients were dropped by presolve in
            # this path, and repeated restricted solves could exhaust Phase I
            # before a jointly feasible blend was ever considered.
            result = conditioned_linprog(linprog, costs,
                A_ub=kwargs['A_ub'], b_ub=kwargs['b_ub'],
                A_eq=kwargs['A_eq'], b_eq=kwargs['b_eq'], bounds=kwargs['bounds'],
                scales=response_variable_scales(responses, 0))
            column_diagnostics['linear_solves'] += 1
            column_diagnostics['compact_full_pool_solves'] = column_diagnostics.get('compact_full_pool_solves', 0)+1
            column_diagnostics['maximum_working_materials'] = max(column_diagnostics['maximum_working_materials'], n)
            column_diagnostics['material_columns_priced'] = n
            if result.success and result.x is not None:
                initial.update(np.flatnonzero(result.x > 0).tolist())
            return result
        direct_level = goal_level
        direct_avoid = ((avoidance-(1-direct_level))*responses[None]).reshape(-1,n)
        direct = solve_profile_cuts(
            shapes=shapes, wanted=wanted, responses=responses,
            fixed=np.vstack((fixed, direct_avoid)), rhs=np.r_[rhs, np.zeros(len(direct_avoid))],
            equality=eq, equality_rhs=eq_rhs, bounds=bounds, objective=objective[:n],
            level=direct_level, background=bank.background,
            initial=best, maximum_seconds=allowance('profile_cuts',12.,n), linear_solver=compact_solver)
        column_diagnostics['full_profile_separation'] = {
            'status': int(direct.status), 'success': bool(direct.success),
            'linear_solves': direct.linear_solves, 'cuts': direct.cut_count,
            'seconds': direct.elapsed_seconds, 'candidate_score': direct.candidate_score,
            'full_pool_count': n, 'full_endpoint_count': d}
        column_diagnostics['all_candidate_materials_ranked'] = n
        direct_candidate = direct.x if direct.success else direct.candidate
        if direct_candidate is not None and valid(direct_candidate):
            value, predicted, _ = evaluate(direct_candidate, True)
            consider(direct_candidate, value, predicted)

    def restricted_solver(costs, **kwargs):
        return column_linprog(costs,material_count=n,initial_columns=initial,column_order=ranking,
                              diagnostics=column_diagnostics,**kwargs)

    def solve(level):
        nonlocal calls, incomplete
        # Below the acceptance threshold this is closest-candidate search, not
        # an approval attempt. Requiring background discrimination there can
        # discard a useful lower-bound solution and spend work on a recipe
        # that cannot be approved anyway. The final gate remains unchanged.
        require_discrimination = level*100+1e-7 >= request.target_similarity
        overlap_rows = sparse.hstack((sparse.csr_matrix(outside-2*(1-level)*repeated_responses),slack_sums),format='csr')
        avoid_rows = sparse.hstack((sparse.csr_matrix(((avoidance-(1-level))*responses[None]).reshape(-1,n)),
                                   sparse.csr_matrix((2*t_count,n_slack))),format='csr')
        matrix = sparse.vstack((absolute_rows,overlap_rows,avoid_rows,physical_rows),format='csr')
        limits = np.r_[np.zeros(2*n_slack+4*t_count),rhs]
        cuts, used_background_margin = [], False
        for separation_round in range(12):
            column_diagnostics['maximum_separation_rounds'] = max(
                column_diagnostics['maximum_separation_rounds'], separation_round+1)
            current = sparse.vstack((matrix, sparse.csr_matrix(np.asarray(cuts))), format='csr') if cuts else matrix
            result = conditioned_linprog(restricted_solver, objective, A_ub=current,
                b_ub=np.r_[limits, np.zeros(len(cuts))],
                A_eq=equality,b_eq=eq_rhs,bounds=full_bounds,scales=scales)
            calls += 1
            if result.status == 2 and not used_background_margin:
                return False
            if not result.success or result.x is None:
                # A strict-discrimination numerical margin is a recovery aid,
                # not a certificate that the original full problem is impossible.
                incomplete = True
                return None
            x = np.maximum(0.,result.x[:n])
            if not np.isfinite(x).all() or x.sum() <= 0:
                incomplete = True
                return None
            x /= x.sum()
            if not valid(x):
                incomplete = True
                return None
            score, predicted, overlaps = evaluate(x,True)
            consider(x, score, predicted)
            initial.update(np.flatnonzero(x > 0).tolist())
            if float(overlaps.min())+1e-7 < level*100:
                incomplete = True
                return None
            if score+1e-7 >= level*100 and (not require_discrimination or is_specific(predicted)):
                return True
            previous_cut_count = len(cuts)
            for h in range(2):
                for t in range(t_count):
                    direction = predicted[h,t]/np.linalg.norm(predicted[h,t])
                    gradients = []
                    if float(unit_targets[h,t]@direction)+1e-9 < level:
                        # q*||p|| - unit(target).p <= 0 is a convex cone.
                        # Its tangent is a necessary linear inequality, so it
                        # cannot discard any cosine-feasible composition.
                        gradients.append(level*direction-unit_targets[h,t])
                        column_diagnostics['cosine_separation_cuts'] += 1
                    if require_discrimination and float(discrimination[h,t]@direction) <= 1e-8:
                        # Same cone separation for the final background guard.
                        # 2e-8 gives numerical room over its strict >1e-8 check;
                        # infeasibility after this margin is only incomplete.
                        gradients.append(2e-8*direction-discrimination[h,t])
                        column_diagnostics['background_separation_cuts'] += 1
                        used_background_margin = True
                    for gradient in gradients:
                        row = (shapes[h]@gradient)*responses[t]
                        magnitude = float(np.max(np.abs(row)))
                        if magnitude > 0:
                            cuts.append(np.r_[row/magnitude, np.zeros(n_slack)])
            if len(cuts) == previous_cut_count:
                break
        incomplete = True
        return None
    target = request.target_similarity/100
    lo, hi = 0., profile_upper/100
    if not target_reached() and profile_upper+1e-8 >= request.target_similarity:
        if solve(goal_level) is False:
            hi = min(hi, goal_level)
        if not target_reached():
            solve(target)
    elif not target_reached() and best is None:
        solve(0.)
    if not target_reached():
        # Keep the established bracket trajectory: feasible cost-minimizing
        # proposals also seed column discovery. Jumping to the incumbent score
        # can lose those seeds under the bounded solver budget.
        for _ in range(9):
            if hi-lo <= 1e-4 or exhausted():
                break
            mid = (lo+hi)/2
            result = solve(mid)
            if result is None:
                break
            if result:
                lo = mid
            else:
                hi = mid
    if best is not None and not target_reached() and not exhausted() and profile_upper > best_score+1e-8:
        from .lotion_fractional_inverse import refine
        improved, inverse_report = refine(shapes=shapes,wanted=wanted,responses=responses,
            fixed=fixed,rhs=rhs,equality=eq,equality_rhs=eq_rhs,bounds=bounds,
            initial=best,avoided=avoid,background=bank.background,target=goal_level,
            max_seconds=allowance('fractional_inverse',12.,n))
        column_diagnostics['fractional_residual_inverse'] = inverse_report
        column_diagnostics['linear_solves'] += inverse_report.get('column_search',{}).get('linear_solves') or inverse_report['linear_solves']
        if improved is not None and valid(improved):
            value,predicted,_ = evaluate(improved,True)
            consider(improved,value,predicted)
    if best is not None and not target_reached() and not exhausted() and profile_upper > best_score+1e-8:
        from time import monotonic
        from .corrective_profile_search import corrective_profile_search
        feedback_started = monotonic()
        feedback_status = 'feedback_round_budget_reached'
        feedback_rounds = 0
        selected_core=getattr(predictor.provider,'core',None)
        aligned=getattr(selected_core,'version',None) in ('shared-formulation-core/v75','shared-formulation-core/v76')
        maximum_rounds,maximum_seconds=(6,allowance('feedback',16.,n)) if aligned else (3,8.)
        for feedback_round in range(maximum_rounds):
            if target_reached():
                feedback_status = 'requested_target_met'
                break
            remaining = maximum_seconds-(monotonic()-feedback_started)
            if remaining <= 0:
                feedback_status = 'feedback_work_budget_reached'
                break
            feedback_rounds = feedback_round+1
            before_rank = (target_reached(), best_score)
            core=getattr(predictor.provider,'core',None)
            if core is not None and hasattr(core,'autoregressive_proposal'):
                # Missing full-profile rows have zero caps and cannot be made
                # into neural candidates merely by filling them with zeros.
                supported=np.flatnonzero(covered & (caps>0))
                learned=core.autoregressive_proposal([pool[i] for i in supported],shapes[:,supported],wanted,
                    responses[:,supported],best[supported],product='body_lotion',minimum_fractions=lower[supported],
                    maximum_fractions=caps[supported],prices=prices[supported],price_budget=request.max_formula_cost_per_kg,
                    avoided=np.broadcast_to(avoid[None],wanted.shape))
                if learned is not None:
                    desired,neural_report=learned
                    column_diagnostics['neural_autoregressive']=neural_report
                    usage=column_diagnostics.setdefault('neural_usage',{'calls':0,'projection_rejections':0,
                        'projected_seeds':0,'full_profile_evaluations':0,'adopted_improvements':0})
                    usage['calls']+=1
                    full_desired=np.zeros(n)
                    full_desired[supported]=desired
                    # L1 projection into the ORIGINAL complete physical
                    # feasible polytope; no rule is waived by a neural output.
                    identity=sparse.eye(n,format='csr')
                    projected=conditioned_linprog(linprog,np.r_[np.zeros(n),np.ones(n)],
                        A_ub=sparse.vstack((sparse.hstack((sparse.csr_matrix(fixed),sparse.csr_matrix(fixed.shape))),
                            sparse.hstack((identity,-identity)),sparse.hstack((-identity,-identity))),format='csr'),
                        b_ub=np.r_[rhs,full_desired,-full_desired],
                        A_eq=sparse.hstack((sparse.csr_matrix(eq),sparse.csr_matrix(eq.shape)),format='csr'),b_eq=eq_rhs,
                        bounds=[*bounds,*[(0.,None)]*n],scales=np.r_[response_variable_scales(responses,0),
                            response_variable_scales(responses,0)])
                    column_diagnostics['linear_solves']+=1
                    if projected.success and projected.x is not None and valid(projected.x[:n]):
                        usage['projected_seeds']+=1
                        neural_before=(target_reached(),best_score)
                        anchor=best.copy()
                        for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125):
                            candidate=anchor+fraction*(projected.x[:n]-anchor)
                            if not valid(candidate):
                                continue
                            usage['full_profile_evaluations']+=1
                            value,predicted,_=evaluate(candidate,True)
                            consider(candidate,value,predicted)
                            if target_reached():
                                break
                        if (target_reached(),best_score)>neural_before:
                            usage['adopted_improvements']+=1
                    else:
                        usage['projection_rejections']+=1
            extrapolated = feedback.proposal()
            if extrapolated is not None and valid(extrapolated):
                value, predicted, _ = evaluate(extrapolated, True)
                consider(extrapolated, value, predicted)
            if not target_reached():
                # Reuse the same complete transport basis, but feed the latest
                # verified composition into the next physical correction.
                remaining = maximum_seconds-(monotonic()-feedback_started)
                if remaining <= 0:
                    feedback_status = 'feedback_work_budget_reached'
                    break
                corrected = corrective_profile_search(shapes=shapes, wanted=wanted,
                    responses=responses, fixed=fixed, rhs=rhs, equality=eq,
                    equality_rhs=eq_rhs, bounds=bounds, initial=best,
                    target=goal_level, avoided=avoid, background=bank.background,
                    maximum_seconds=min(allowance('feedback_corrective',2.5,n),remaining), maximum_rounds=2)
                column_diagnostics['linear_solves'] += corrected.linear_solves
                column_diagnostics['nonlinear_solves'] = column_diagnostics.get('nonlinear_solves',0)+corrected.nonlinear_solves
                candidate = corrected.x if corrected.success else corrected.candidate
                if candidate is not None and valid(candidate):
                    value, predicted, _ = evaluate(candidate, True)
                    consider(candidate, value, predicted)
            if (target_reached(), best_score) <= before_rank:
                feedback_status = 'verified_score_plateau'
                break
        column_diagnostics['autoregressive_refinement'] = feedback.report(
            status='requested_target_met' if target_reached() else feedback_status,
            rounds=feedback_rounds, maximum_rounds=maximum_rounds, maximum_additional_seconds=maximum_seconds,
            seconds=monotonic()-feedback_started,
            full_pool_considered=n, full_endpoint_count=d,
            budget_kind='between_solver_steps_not_hard_preemption')
    else:
        column_diagnostics['autoregressive_refinement'] = feedback.report(
            status='requested_target_met' if target_reached() else 'no_improving_feasible_feedback',rounds=0)
    if retired_blends.rejected:
        common['retired_blends'] = retired_blends.report()
    if best is None:
        return {**common, 'status': 'search_incomplete' if incomplete else 'no_feasible_research_formula',
            'score': None, 'profile_target_met': False, 'recipe': [], 'closest_candidate': [],
            'solver_calls': column_diagnostics['linear_solves']+column_diagnostics.get('conic_solver_calls',0), 'search_incomplete': incomplete,
            'transport_simulation_calls': len(basis_requests)-reused_simulations,
            'reused_basis_simulations': reused_simulations, 'basis_cache_status': cache_status}

    selected_indices = np.flatnonzero(best > 0)
    selected = [(pool[i], best[i]) for i in selected_indices]
    selected_catalog = IngredientCatalog([i for i, _ in selected])
    simulations, final_checks, legacy_checks = [], [], []
    physical_ok = True
    from .lotion_surrogate import attach_release_prediction
    for s_index, basis in enumerate(basis_requests):
        by_id = {m.ingredient_id: m for m in basis.materials}
        materials = [by_id[item.ingredient_id].model_copy(update={'concentrate_percent': float(w*100)}) for item,w in selected]
        actual_request = basis.model_copy(update={'materials': materials})
        actual = _simulate_lotion_transport(actual_request, selected_catalog, exposure_windows=windows)
        physical_ok &= actual['status'] != 'outside_open_sink_assumption'
        temporal = actual['temporal_profile'][1:]
        activities = np.asarray([[r['mean_odor_activity_proxy'] for r in p['materials']] for p in actual['exposure_windows']])
        for j, group in enumerate(groups):
            t = s_index*len(targets)+j
            transport = activities[j]
            physical_ok &= float(transport.sum())+1e-8 >= 1.
            mixture = np.einsum('n,hnd->hd', transport/transport.sum(), shapes[:, selected_indices])
            for h in range(2):
                check = bank.compare(target_info[t], mixture[h], h)
                final_checks.append({**check, **{k:v for k,v in group.items() if k not in ('weights','target_index')},
                    'scenario_index': s_index, 'reference_head': ('applicability','use')[h],
                    'total_odor_activity_proxy': float(transport.sum()), 'concepts': target_info[t]['concepts']})
        for row,point in zip(original_target_rows,temporal):
            physical_ok &= point['total_air_concentration_mg_m3'] >= request.minimum_air_concentration_mg_m3*(1-1e-8)
            legacy_checks.append({'minutes': point['minutes'], 'scenario_index': s_index,
                **compare_lotion_profiles(row['target_profile'], point['scent_profile'],
                    avoided=row['avoided']).to_dict()})
        simulations.append(attach_release_prediction(actual, actual_request, selected_catalog,
            provider=predictor.provider, shape_predictor=predictor))
    bank.assert_current()
    if hasattr(predictor,'assert_observations_current'):
        predictor.assert_observations_current()
    score = min(c['score'] for c in final_checks)
    if abs(score-best_score) > 1e-6:
        raise ValueError('fresh final-mixture reference score differs from search prediction')
    specific = all(c['reference_more_specific_than_background'] for c in final_checks)
    passed = bool(coverage['complete'] and score+1e-8 >= request.target_similarity and physical_ok and specific)
    lines = [{'ingredient_id': item.ingredient_id, 'name': item.name, 'concentrate_percent': float(w*100),
        'finished_product_percent': float(w*basis_requests[0].application_context.fragrance_concentration_percent),
        'pyramid': item.pyramid, 'price_per_kg': item.price_per_kg} for item,w in selected]
    digest = hashlib.sha256(json.dumps({'request': request.model_dump(mode='json'), 'recipe': lines,
        'reference': bank.sha256, 'predictor': bank.parent_sha256,
        'component_observations': getattr(getattr(predictor,'observations',None),'sha256',None),
        'exposure_integration': EXPOSURE_INTEGRATION_VERSION}, sort_keys=True).encode()).hexdigest()
    common['perceptual_evaluation'].update(
        status='reference_profile_evaluated', physical_presence_passed=bool(physical_ok),
        background_discrimination_passed=specific, full_endpoint_count=d,
        candidate_pool_count=n, learned_profile_covered_count=int(covered.sum()),
        missing_learned_material_ids=[i.ingredient_id for i,c in zip(pool,covered) if not c],
        component_reference_observations=(predictor.observation_summary() if hasattr(predictor,'observation_summary') else {'configured':False}),
        targets=[{'phase': r['phase'], 'minutes': r['minutes'], 'concepts': t['concepts'],
                  'window_start_minutes':r['window_start_minutes'], 'integration_weights':r['weights'].tolist(),
                  'integration_weights_used': False, 'integration_weights_semantics': 'legacy_display_trapezoid_diagnostic_only',
                  'aggregation': EXPOSURE_INTEGRATION_VERSION,
                  'reference_profiles': t['profiles'].tolist(), 'avoided': t['avoided'],
                  'source': t.get('source', 'observed_conditional_reference_not_measured_user_target'),
                  'annotation_conditioned_concepts': t.get('annotation_conditioned_concepts', []),
                  'reference_evidence': t.get('reference_evidence', {}),
                  'reference_identity_support': t.get('reference_identity_support', {})}
                 for r,t in zip(groups, targets)])
    return {**common, 'status': 'research_profile_target_met' if passed else 'research_candidate_only' if coverage['complete'] else 'partial_source_reference_candidate',
        'score': score if coverage['complete'] else None,
        'partial_profile_score': None if coverage['complete'] else score,
        'score_scope': coverage['score_scope'], 'profile_target_met': passed, 'formula_id': digest,
        'recipe': lines if passed else [], 'closest_candidate': [] if passed else lines,
        'timepoint_assessments': final_checks,
        'legacy_evaluation': {'score': min(c['score'] for c in legacy_checks), 'timepoint_assessments': legacy_checks,
            'version': 'body-lotion-transport-profile/v2', 'used_for_recipe_selection': False},
        'incumbent_reuse': {'status': seed_status, 'current_score': incumbent_score,
            'old_score_reused': False, 'used': seed is not None},
        'baseline_score': incumbent_score, 'simulation': simulations[0], 'scenario_simulations': simulations[1:],
        'transport_scenario_count': len(basis_requests), 'solver_calls': column_diagnostics['linear_solves']+column_diagnostics.get('conic_solver_calls',0),
        'solver_recovery_calls': 0, 'solver_conditioning_calls': 0, 'search_incomplete': incomplete,
        'exact_profile_slack_variables': n_slack, 'estimated_concentrate_cost_per_kg': float(prices@best),
        'material_column_search': column_diagnostics,
        'fine_expression_search':fine_diagnostic,
        'transport_simulation_calls': 2*len(basis_requests)-reused_simulations,
        'reused_basis_simulations': reused_simulations, 'basis_cache_status': cache_status,
        'search_kind': 'full_146_endpoint_observed_reference_exact_lp',
        'limitations': ['Source-observed reference profiles are not user-specific measured targets.',
            'Transport-weighted component shapes do not validate lotion masking or synergy.',
            'OAV presence uses existing estimated thresholds, not measured human detectability.',
            'V58 independent-time predictions remain auxiliary and may report trajectory warnings.']}
