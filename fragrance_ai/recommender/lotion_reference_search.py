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
from .lotion import _simulate_lotion_transport
from .lotion_evaluation import compare_lotion_profiles
from .lotion_incumbent import incumbent_weights
from .lotion_reference_objective import VERSION, exposure_groups
from .lotion_numerics import conditioned_linprog, response_variable_scales


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
    for attempt in range(6):
        indices = np.asarray(sorted(active)+auxiliary, dtype=int)
        parameters = {**kwargs, 'A_ub': kwargs['A_ub'][:,indices],
            'A_eq': kwargs['A_eq'][:,indices], 'bounds': [kwargs['bounds'][i] for i in indices]}
        # Dense presolve/IPM fill-in dominated the full 146-D experiment.
        # Dual simplex with explicit work bounds avoids that unbounded setup.
        parameters['method'] = 'highs-ds'
        parameters['options'] = {**kwargs.get('options', {}), 'presolve': False, 'time_limit': .75}
        result = linprog(objective[indices], **parameters)
        diagnostics['linear_solves'] += 1
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
            if not len(negative):
                return result
            if attempt == 5:
                # Keep a feasible proposal, but do not claim a full cost optimum.
                diagnostics['expansion_budget_reached'] = True
                return result
            active.update(negative[np.argsort(reduced[negative])[:64]].tolist())
        else:
            if len(active) == n:
                return result
            if attempt == 5 or result.status not in (1,2,4):
                diagnostics['expansion_budget_reached'] = True
                return OptimizeResult(success=False,status=1,x=None,
                    message='working-column expansion incomplete; full feasibility not established')
            # Feasibility has no ordinary reduced-cost certificate. Expand the
            # independent ranking rather than treating the missing columns as banned.
            count = min(n, max(128, 2*len(active)))
            active.update(column_order[:count])
    raise AssertionError('unreachable column loop')


def optimize_observed_reference(*, request, prepared, brief, pool, basis_requests,
        responses, physical, uptake_blocks, predictor, bank, baseline,
        cache_status, reused_simulations, incumbent_recipe=None):
    targets, unsupported = bank.targets(brief, prepared['evaluation_targets'])
    contract = bank.contract()
    prepared = {**prepared, 'product_model': contract,
        'intent': {**prepared['intent'], 'representation': {
            **prepared['intent']['representation'], 'weight_semantics': 'semantic_query_weights_for_observed_full_profiles',
            'unrequested_axes_remain_in_strict_comparison': False,
            'zero_target_axis_semantics': 'not_requested_not_zero_in_observed_reference',
            'learned_coverage_is_not_primary_19_axis_score': False}}}
    common = {'schema_version': 'lotion-optimization-1', 'product_model': contract,
        'preparation': prepared, 'perceptual_evaluation': {**contract,
            'unsupported_requirements': unsupported, 'absolute_intensity_target_evaluated': False,
            'minimum_odor_activity_proxy': 1., 'target_threshold': request.target_similarity,
            'presence_scope': 'mean_over_requested_exposure_window_not_every_timepoint'},
        'score_kind': 'exposure_integrated_observed_reference_agreement_not_user_similarity',
        'human_similarity_percent': None, 'manufacturing_approved': False,
        'all_user_requirements_verified': False, 'external_api_calls': 0}
    if unsupported or any(t is None for t in targets):
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
    targets = [original_targets[g['target_index']] for g in groups]
    integration = sparse.block_diag([np.asarray([g['weights'] for g in groups])]*len(basis_requests),format='csr')
    thresholds = np.vstack([np.tile([m.odor_threshold_mg_m3 for m in s.materials],
                                    (len(original_targets), 1)) for s in basis_requests])
    original_oav = physical/thresholds
    absolute_oav = integration@original_oav
    responses = absolute_oav/np.maximum(absolute_oav.max(axis=1,keepdims=True),1e-300)
    t_count = len(responses)
    target_rows = [{**{k:v for k,v in g.items() if k != 'weights'}, 'scenario_index':i}
                   for i in range(len(basis_requests)) for g in groups]
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

    best, best_score = None, -1.
    seed, seed_status = incumbent_weights(incumbent_recipe, pool)
    incumbent_score = evaluate(seed) if seed is not None and valid(seed) else None
    for x in (baseline, seed):
        if x is not None and valid(x):
            score = evaluate(x)
            if score > best_score:
                best, best_score = x.copy(), score
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
        'all_candidate_materials_ranked':n,'permanent_candidate_shortlist':False}

    def restricted_solver(costs, **kwargs):
        return column_linprog(costs,material_count=n,initial_columns=initial,column_order=ranking,
                              diagnostics=column_diagnostics,**kwargs)

    def solve(level):
        nonlocal best, best_score, calls, incomplete
        overlap_rows = sparse.hstack((sparse.csr_matrix(outside-2*(1-level)*repeated_responses),slack_sums),format='csr')
        avoid_rows = sparse.hstack((sparse.csr_matrix(((avoidance-(1-level))*responses[None]).reshape(-1,n)),
                                   sparse.csr_matrix((2*t_count,n_slack))),format='csr')
        matrix = sparse.vstack((absolute_rows,overlap_rows,avoid_rows,physical_rows),format='csr')
        limits = np.r_[np.zeros(2*n_slack+4*t_count),rhs]
        result = conditioned_linprog(restricted_solver, objective, A_ub=matrix,b_ub=limits,
            A_eq=equality,b_eq=eq_rhs,bounds=full_bounds,scales=scales)
        calls += 1
        if result.status == 2:
            return False
        if not result.success or result.x is None:
            incomplete = True
            return None
        x = np.maximum(0.,result.x[:n])
        x /= x.sum()
        if not valid(x):
            incomplete = True
            return None
        score, _, overlaps = evaluate(x,True)
        if score > best_score+1e-9:
            best,best_score = x.copy(),score
            initial.update(np.flatnonzero(best > 0).tolist())
        if float(overlaps.min())+1e-7 < level*100:
            incomplete = True
            return None
        return True
    target = request.target_similarity/100
    lo, hi = 0., profile_upper/100
    if profile_upper+1e-8 >= request.target_similarity:
        if solve(min(1., target+.002)) is False:
            hi = min(hi, target+.002)
        if best_score+1e-8 < request.target_similarity:
            solve(target)
    elif best is None:
        solve(0.)
    if best_score+1e-8 < request.target_similarity:
        for _ in range(9):
            mid = (lo+hi)/2
            result = solve(mid)
            if result is None:
                break
            if result:
                lo = mid
            else:
                hi = mid
    if best is None:
        return {**common, 'status': 'search_incomplete' if incomplete else 'no_feasible_research_formula',
            'score': None, 'profile_target_met': False, 'recipe': [], 'closest_candidate': [],
            'solver_calls': column_diagnostics['linear_solves'], 'search_incomplete': incomplete,
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
        actual = _simulate_lotion_transport(actual_request, selected_catalog)
        physical_ok &= actual['status'] != 'outside_open_sink_assumption'
        temporal = actual['temporal_profile'][1:]
        activities = np.asarray([[r['odor_activity_proxy'] for r in p['materials']] for p in temporal])
        for j, group in enumerate(groups):
            t = s_index*len(targets)+j
            transport = group['weights']@activities
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
    score = min(c['score'] for c in final_checks)
    if abs(score-best_score) > 1e-6:
        raise ValueError('fresh final-mixture reference score differs from search prediction')
    specific = all(c['reference_more_specific_than_background'] for c in final_checks)
    passed = bool(score+1e-8 >= request.target_similarity and physical_ok and specific)
    lines = [{'ingredient_id': item.ingredient_id, 'name': item.name, 'concentrate_percent': float(w*100),
        'finished_product_percent': float(w*basis_requests[0].application_context.fragrance_concentration_percent),
        'pyramid': item.pyramid, 'price_per_kg': item.price_per_kg} for item,w in selected]
    digest = hashlib.sha256(json.dumps({'request': request.model_dump(mode='json'), 'recipe': lines,
        'reference': bank.sha256, 'predictor': bank.parent_sha256}, sort_keys=True).encode()).hexdigest()
    common['perceptual_evaluation'].update(
        status='reference_profile_evaluated', physical_presence_passed=bool(physical_ok),
        background_discrimination_passed=specific, full_endpoint_count=d,
        candidate_pool_count=n, learned_profile_covered_count=int(covered.sum()),
        missing_learned_material_ids=[i.ingredient_id for i,c in zip(pool,covered) if not c],
        targets=[{'phase': r['phase'], 'minutes': r['minutes'], 'concepts': t['concepts'],
                  'window_start_minutes':r['window_start_minutes'], 'integration_weights':r['weights'].tolist(),
                  'reference_profiles': t['profiles'].tolist(), 'avoided': t['avoided']}
                 for r,t in zip(groups, targets)])
    return {**common, 'status': 'research_profile_target_met' if passed else 'research_candidate_only',
        'score': score, 'profile_target_met': passed, 'formula_id': digest,
        'recipe': lines if passed else [], 'closest_candidate': [] if passed else lines,
        'timepoint_assessments': final_checks,
        'legacy_evaluation': {'score': min(c['score'] for c in legacy_checks), 'timepoint_assessments': legacy_checks,
            'version': 'body-lotion-transport-profile/v2', 'used_for_recipe_selection': False},
        'incumbent_reuse': {'status': seed_status, 'current_score': incumbent_score,
            'old_score_reused': False, 'used': seed is not None},
        'baseline_score': incumbent_score, 'simulation': simulations[0], 'scenario_simulations': simulations[1:],
        'transport_scenario_count': len(basis_requests), 'solver_calls': column_diagnostics['linear_solves'],
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
