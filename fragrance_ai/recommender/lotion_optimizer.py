"""Fixed-base, data-supplied lotion inverse design; not commercial approval."""
import hashlib
import json
from datetime import date

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .adaptive_pyramid import explicit_pyramid
from .brief_parser import NaturalLanguageBriefParser
from .catalog import IngredientCatalog, normalize_name
from .lotion import _simulate_lotion_transport as simulate_lotion
from .lotion_basis_cache import reuse_basis
from .lotion_numerics import conditioned_linprog, response_variable_scales
from .models import RecipeConstraints, SCENT_DIMENSIONS, profile_vector
from .lotion_evaluation import compare_lotion_profiles as compare_profiles, phase_for_time, evaluation_contract, refinement_score_floors
from .safety import CandidateSafetyScreen
from .intent_controls import apply_intent_controls, target_controls, effective_phase_target, representation_contract


def _compact_profile_residuals(profiles, targets, responses):
    """Exact full-axis L1 system: zero-target residuals have a known sign.

    They remain in ``outside`` rather than being dropped from the score. Only
    positive-target axes need absolute-value slack variables in the LP.
    """
    times, dimensions = np.nonzero(targets > 0)
    residual = (profiles[:, dimensions].T - targets[times, dimensions, None]) * responses[times]
    count = len(times)
    identity = sparse.eye(count, format="csr")
    rows = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(residual), -identity]),
        sparse.hstack([sparse.csr_matrix(-residual), -identity])], format="csr")
    sums = sparse.csr_matrix((np.ones(count), (times, np.arange(count))),
                             shape=(len(targets), count))
    outside = ((targets == 0) @ profiles.T) * responses
    return rows, sums, outside


def prepare_lotion_optimization(request, catalog, parser=None, *, _request_snapshot=None):
    simulation = request.simulation
    known = {item.ingredient_id: item for item in catalog.ingredients}
    supplied = {row.ingredient_id: row for row in simulation.materials}
    if set(supplied) - set(known) or set(request.excluded_ingredient_ids) - set(known):
        raise ValueError("unknown lotion material/exclusion ID")
    constraints = RecipeConstraints(product_category="body_lotion", target_similarity=request.target_similarity,
        product_concentration_percent=simulation.application_context.fragrance_concentration_percent,
        max_formula_cost_per_kg=request.max_formula_cost_per_kg,
        max_ingredient_price_per_kg=request.max_ingredient_price_per_kg,
        min_availability=request.min_availability, max_risk_tier=request.max_risk_tier,
        enable_registry_trace_candidates=request.registry_pool == "conditional_research",
        explicit_bans=set(request.excluded_ingredient_ids))
    snapshot, today = _request_snapshot, date.today()
    controls = target_controls(request)
    reusable = (snapshot is not None and snapshot.get('catalog') is catalog
        and snapshot.get('parser') is parser and snapshot.get('text') == request.brief
        and snapshot.get('constraints') == constraints and snapshot.get('day') == today
        and snapshot.get('intent_controls', {}) == controls and 'raw_eligible' in snapshot)
    brief = snapshot['brief'] if reusable else (parser or NaturalLanguageBriefParser(catalog)).parse(request.brief, constraints)
    if not reusable:
        brief = apply_intent_controls(brief, controls)
    if brief.constraints.product_concentration_percent != simulation.application_context.fragrance_concentration_percent:
        raise ValueError("natural-language concentration differs from the supplied lotion context")
    if brief.constraints.finished_batch_mass_g != constraints.finished_batch_mass_g or brief.constraints.finished_volume_ml != constraints.finished_volume_ml:
        raise ValueError("lotion inverse design does not generate batch/volume manufacturing instructions")
    if reusable:
        eligible, rejected = snapshot['raw_eligible'], dict(snapshot['rejected'])
    else:
        eligible, rejected = CandidateSafetyScreen().screen(catalog, brief, as_of=today)
    eligible = [item for item in eligible if abs(item.active_strength_percent - 100) <= 1e-8 and np.sum(item.vector()) > 0]
    pool = [item for item in eligible if item.ingredient_id in supplied]
    if not pool:
        raise ValueError("no eligible lotion candidates with supplied transport data")
    if brief.constraints.max_ingredients < len(pool):
        raise ValueError("natural-language ingredient-count constraints need a cardinality solver; remove the count or supply a smaller pool")
    required_names = {normalize_name(name) for name in brief.requested_ingredients}
    if not required_names.issubset({normalize_name(item.name) for item in pool}):
        raise ValueError("an explicitly requested material is ineligible or missing lotion transport data")
    times = simulation.times_minutes[1:]
    phases = [phase_for_time(t, request.transition_schedule) for t in times]
    if set(brief.phase_target_profiles) - set(phases):
        raise ValueError("requested scent phase has no evaluation timepoint")
    # A phase containing only a prohibition has no positive profile. Inherit
    # the global positive intent, while retaining its phase-specific avoidance.
    targets = [effective_phase_target(brief, phase) for phase in phases]
    avoided = [sorted(set(brief.avoided_dimensions) | set(brief.phase_avoided_dimensions.get(phase, []))) for phase in phases]
    # These are the package's explicit USD/kg estimate labels, not different
    # currencies. Preserve provenance while joining the core and registry pool.
    usd_labels = {"USD_estimate", "USD_estimate_not_supplier_quote"}
    price_labels = {item.currency for item in pool}
    if not price_labels.issubset(usd_labels):
        raise ValueError("lotion candidate prices use mixed units; provide a consistent catalog")
    unscored = ["human similarity", "regulatory approval", "lotion stability", "skin feel", "human-detectable longevity"]
    if brief.texture_profile:
        unscored.append("requested texture: " + ", ".join(brief.texture_profile))
    if brief.trigeminal_profile:
        unscored.append("requested trigeminal properties: " + ", ".join(brief.trigeminal_profile))
    if brief.intensity != "medium":
        unscored.append("qualitative human intensity: " + brief.intensity)
    prepared = {"schema_version": "lotion-intent-1", "status": "ready_for_research_optimization",
        "product_model": evaluation_contract(),
        "intent": {"original_text": brief.original_text, "target_profile": brief.target_profile,
            "representation": representation_contract(brief),
            "phase_target_profiles": brief.phase_target_profiles, "avoided_dimensions": brief.avoided_dimensions,
            "phase_avoided_dimensions": brief.phase_avoided_dimensions,
            "requested_ingredients": brief.requested_ingredients},
        "evaluation_targets": [{"minutes": time, "phase": phase, "target_profile": target, "avoided": negative}
            for time, phase, target, negative in zip(times, phases, targets, avoided)],
        "eligible_catalog_count": len(eligible), "transport_covered_candidate_count": len(pool),
        "missing_transport_count": len(eligible) - len(pool), "candidate_ids": [item.ingredient_id for item in pool],
        "rejected_candidate_counts": rejected, "price_unit": "USD_estimate",
        "price_currency": "USD", "price_source_labels": sorted(price_labels), "live_supplier_price_verified": False,
        "effective_target": max(95., request.target_similarity), "unscored_requirements": unscored,
        "odor_projection_versions": sorted({item.odor_projection_version or 'explicit-odor-projection-1' for item in pool}),
        "coefficient_scope": simulation.coefficient_scope, "coefficient_scope_reference": simulation.coefficient_scope_reference,
        "parameter_source_verified": False, "manufacturing_approved": False}
    return prepared, brief, pool


def optimize_lotion(request, catalog, parser=None, *, transport_scenarios=None, target_only=False, improve_score_from=None,
                    reuse_transport_basis=False, _request_snapshot=None, perception_guidance=None, _transport_only=False,
                    _shape_predictor=None, _incumbent_recipe=None, experimental_conic_recovery=False):
    from .lotion_perception import attach_lotion_perception, resolve_provider, make_lotion_shape_predictor
    provider = None if _transport_only else resolve_provider(perception_guidance)
    predictor = (_shape_predictor or make_lotion_shape_predictor(provider)) if provider is not None else None
    result = _optimize_lotion_transport(request, catalog, parser, transport_scenarios=transport_scenarios,
        target_only=target_only, improve_score_from=improve_score_from, reuse_transport_basis=reuse_transport_basis,
        _request_snapshot=_request_snapshot, _shape_predictor=predictor, _incumbent_recipe=_incumbent_recipe,
        experimental_conic_recovery=experimental_conic_recovery)
    return attach_lotion_perception(result, catalog, provider, _shape_predictor=predictor)


def _optimize_lotion_transport(request, catalog, parser=None, *, transport_scenarios=None, target_only=False,
                              improve_score_from=None, reuse_transport_basis=False, _request_snapshot=None,
                              _shape_predictor=None, _incumbent_recipe=None, experimental_conic_recovery=False):
    if not isinstance(experimental_conic_recovery, bool):
        raise ValueError('experimental_conic_recovery must be boolean')
    prepared, brief, pool = prepare_lotion_optimization(request, catalog, parser, _request_snapshot=_request_snapshot)
    simulation = request.simulation
    supplied = {row.ingredient_id: row for row in simulation.materials}
    n, dims = len(pool), len(SCENT_DIMENSIONS)
    scenarios = [simulation] + list(transport_scenarios or [])
    if len(scenarios) > 5:
        raise ValueError("at most five explicit transport scenarios")
    for scenario in scenarios:
        if scenario.times_minutes != simulation.times_minutes or scenario.application_context != simulation.application_context:
            raise ValueError("scenario contexts and evaluation times must match")
        if scenario.transport_mode != simulation.transport_mode or scenario.coefficient_scope != simulation.coefficient_scope:
            raise ValueError("scenario transport modes and coefficient scopes must match")
        if {m.ingredient_id for m in scenario.materials} != set(supplied):
            raise ValueError("scenario ingredient coverage must match")
    t_count = len(prepared["evaluation_targets"]) * len(scenarios)
    if t_count*n*dims > 2_000_000:
        raise ValueError("robust lotion optimization matrix budget exceeded")
    baseline = np.array([supplied[item.ingredient_id].concentrate_percent for item in pool])
    baseline /= baseline.sum()
    # One multi-material transport calculation provides the independent linear
    # response columns. Do not rerun transport for each LP or candidate material.
    basis_materials = [supplied[item.ingredient_id].model_copy(update={"concentrate_percent": float(100*weight)})
        for item, weight in zip(pool, baseline)]
    if any(s.profile_weighting != simulation.profile_weighting for s in scenarios):
        raise ValueError("scenario profile weighting must match")
    all_thresholds = all(row.odor_threshold_mg_m3 is not None for s in scenarios for row in s.materials) and simulation.profile_weighting != "air_mass"
    weighting = "odor_activity" if all_thresholds else "air_mass"
    basis_requests, physical_blocks, threshold_blocks, uptake_blocks = [], [], [], []
    for scenario in scenarios:
        scenario_materials = {m.ingredient_id: m for m in scenario.materials}
        materials = [scenario_materials[item.ingredient_id].model_copy(update={"concentrate_percent": float(100*weight)})
                     for item, weight in zip(pool, baseline)]
        candidate = scenario.model_copy(update={"materials": materials, "profile_weighting": weighting})
        basis_requests.append(candidate)
        threshold_blocks.append(np.tile([m.odor_threshold_mg_m3 or np.nan for m in materials],
                                        (len(prepared["evaluation_targets"]), 1)))
    def compute_basis():
        physical_values, uptake_values = [], []
        basis_catalog = IngredientCatalog(pool)
        for candidate in basis_requests:
            result = simulate_lotion(candidate, basis_catalog)
            uptake_values.append((np.array([r['skin_sink_mg_cm2'] for r in result['temporal_profile'][-1]['materials']])/baseline).tolist())
            physical_values.append((np.array([[r['air_concentration_mg_m3'] for r in point['materials']]
                for point in result['temporal_profile'][1:]])/baseline[None,:]).tolist())
        return {'physical':physical_values, 'uptake':uptake_values}
    basis_data, cache_status = reuse_basis(basis_requests, pool, compute_basis) if reuse_transport_basis else (compute_basis(), 'disabled')
    physical_blocks, uptake_blocks = basis_data['physical'], basis_data['uptake']
    reused_simulations = len(scenarios) if cache_status in ('hit','shared') else 0
    physical = np.vstack(physical_blocks)
    thresholds = np.vstack(threshold_blocks)
    responses = physical / thresholds if all_thresholds else physical
    if not np.isfinite(responses).all() or np.any(np.max(responses, axis=1) <= 1e-30):
        raise ValueError("no finite nonzero modeled headspace at one or more requested times")
    response_scale = np.max(responses, axis=1)
    responses = responses / response_scale[:, None]
    from .lotion_reference_objective import configured_reference
    reference_bank = configured_reference(request, _shape_predictor)
    if reference_bank is not None:
        from .lotion_reference_search import optimize_observed_reference
        return optimize_observed_reference(request=request, prepared=prepared, brief=brief, pool=pool,
            basis_requests=basis_requests, responses=responses, physical=physical, uptake_blocks=uptake_blocks,
            predictor=_shape_predictor, bank=reference_bank, baseline=baseline, cache_status=cache_status,
            reused_simulations=reused_simulations, incumbent_recipe=_incumbent_recipe)
    profiles = np.array([item.vector() for item in pool])
    target_rows = [{**row, "scenario_index": i} for i in range(len(scenarios)) for row in prepared["evaluation_targets"]]
    targets = np.array([profile_vector(row["target_profile"]) for row in target_rows])
    avoid_vectors = np.array([[1. if axis in row["avoided"] else 0. for axis in SCENT_DIMENSIONS] for row in target_rows])
    caps = np.minimum(1., np.array([item.as_supplied_cap_percent() / 100. for item in pool]))
    prices = np.array([item.price_per_kg for item in pool])
    cost_cap = brief.constraints.max_formula_cost_per_kg
    lower = np.array([1e-6 if normalize_name(item.name) in {normalize_name(x) for x in brief.requested_ingredients} else 0. for item in pool])
    notes, _ = explicit_pyramid(request.brief)
    eq_rows = [np.ones(n)] + [np.array([1. if item.pyramid == note else 0. for item in pool]) for note in notes]
    eq_rhs = np.array([1.] + [notes[note]/100. for note in notes])
    # Existing pyramid helper returns percentages; no silent note-ratio defaults.
    a_eq_x = np.array(eq_rows)
    residual_rows, slack_sums, outside = _compact_profile_residuals(profiles, targets, responses)
    n_slack = slack_sums.shape[1]
    a_eq = sparse.hstack([sparse.csr_matrix(a_eq_x), sparse.csr_matrix((len(eq_rows), n_slack))], format="csr")
    physical_scale = np.max(physical, axis=1)
    physical_scaled = physical / physical_scale[:, None]
    fixed_x = sparse.vstack([sparse.csr_matrix(prices[None, :] / cost_cap), sparse.csr_matrix(-physical_scaled)], format="csr")
    fixed_rows = sparse.hstack([fixed_x, sparse.csr_matrix((1+t_count, n_slack))], format="csr")
    fixed_rhs = np.r_[1., -request.minimum_air_concentration_mg_m3 / physical_scale]
    if request.maximum_modeled_uptake_mg_cm2 is not None:
        uptake_rows = np.array(uptake_blocks) / request.maximum_modeled_uptake_mg_cm2
        fixed_rows = sparse.vstack([fixed_rows, sparse.hstack([
            sparse.csr_matrix(uptake_rows), sparse.csr_matrix((len(scenarios), n_slack))])], format="csr")
        fixed_rhs = np.r_[fixed_rhs, np.ones(len(scenarios))]
    objective = np.r_[prices / max(float(prices.max()), 1.), np.zeros(n_slack)]
    bounds = list(zip(lower, caps)) + [(0., None)] * n_slack
    calls, incomplete, recovery_calls = 0, False, 0
    conditioning_calls = 0
    prefer_conditioned = False
    scales = response_variable_scales(responses, n_slack)
    poorly_scaled = float(np.min(scales[:n])) <= 1e-10

    def valid(weights):
        return (np.isfinite(weights).all() and np.all(weights >= lower - 1e-10) and np.all(weights <= caps + 1e-10)
            and np.max(np.abs(a_eq_x @ weights - eq_rhs)) <= 1e-8
            and prices @ weights <= cost_cap + 1e-7
            and (request.maximum_modeled_uptake_mg_cm2 is None or
                 np.all(np.array(uptake_blocks) @ weights <= request.maximum_modeled_uptake_mg_cm2 * (1+1e-8)))
            and np.all(physical @ weights >= request.minimum_air_concentration_mg_m3 * (1-1e-8)))

    def assessments(weights):
        values = []
        for t, row in enumerate(target_rows):
            contribution = responses[t] * weights
            predicted = contribution @ profiles
            values.append(compare_profiles(row["target_profile"], predicted, avoided=row["avoided"]).to_dict())
        return min(value["score"] for value in values), values

    best, best_score = None, -1.
    if valid(baseline):
        best, best_score = baseline.copy(), assessments(baseline)[0]

    from .lotion_incumbent import incumbent_weights
    seed, seed_status = incumbent_weights(_incumbent_recipe, pool)
    seed_report = {'status': seed_status, 'used': False, 'old_score_reused': False,
                   'current_score': None, 'extra_transport_simulations': 0}
    if seed is not None:
        if not valid(seed):
            seed_report['status'] = 'rejected_by_current_constraints'
        else:
            seed_score = assessments(seed)[0]
            seed_report.update(status='current_constraints_and_transport_rechecked', current_score=seed_score)
            if seed_score > best_score + 1e-9:
                best, best_score = seed.copy(), seed_score
                seed_report['used'] = True

    def solve(overlap):
        nonlocal calls, incomplete, best, best_score, recovery_calls, conditioning_calls, prefer_conditioned
        variation = sparse.hstack([sparse.csr_matrix(outside-2*(1-overlap)*responses), slack_sums], format="csr")
        avoidance = (avoid_vectors @ profiles.T - (1-overlap)) * responses
        avoidance = sparse.hstack([sparse.csr_matrix(avoidance), sparse.csr_matrix((t_count, n_slack))], format="csr")
        matrix = sparse.vstack([residual_rows, variation, avoidance, fixed_rows], format="csr")
        rhs = np.r_[np.zeros(2*n_slack+2*t_count), fixed_rhs]
        used_conditioning = prefer_conditioned
        if used_conditioning:
            # Once the original coordinates failed their exact normalized
            # check, do not rediscover that failure at every bisection level.
            # This is x=D*y on the SAME rows, bounds and target, not a new score.
            result = conditioned_linprog(linprog, objective, A_ub=matrix, b_ub=rhs, A_eq=a_eq, b_eq=eq_rhs,
                bounds=bounds, scales=scales)
            conditioning_calls += 1
        else:
            result = linprog(objective, A_ub=matrix, b_ub=rhs, A_eq=a_eq, b_eq=eq_rhs, bounds=bounds,
                             method="highs-ds" if n > 1024 else "highs",
                             options={"time_limit": .5, "presolve": n <= 1024,
                                      "small_matrix_value": 1e-12,
                                      "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        calls += 1
        # Numerical failure/time exhaustion is not proof of infeasibility.
        # A different algorithm gets one bounded recovery attempt, using the
        # exact same matrix, pool, constraints, and acceptance score.
        if result.status not in (0, 2):
            result = linprog(objective, A_ub=matrix, b_ub=rhs, A_eq=a_eq, b_eq=eq_rhs, bounds=bounds,
                             method="highs-ipm", options={"time_limit": 2., "presolve": True,
                                 "small_matrix_value": 1e-12,
                                 "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
            calls += 1
            recovery_calls += 1
        def verified(candidate):
            nonlocal best, best_score
            if candidate.status != 0 or not getattr(candidate, 'success', False) or candidate.x is None:
                return None
            weights = np.maximum(0., candidate.x[:n])
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                return None
            weights /= weights.sum()
            if not valid(weights):
                return None
            score, checks = assessments(weights)
            if score > best_score + 1e-9:
                best, best_score = weights.copy(), score
            # Absolute LP feasibility tolerance must not erase normalized
            # overlap/avoidance constraints on very weak response columns.
            if any(c['overlap_score'] + 1e-7 < 100*overlap or
                   100*(1-c['avoided_mass']) + 1e-7 < 100*overlap for c in checks):
                return None
            return weights, score
        checked = verified(result)
        unverified_success = result.status == 0 and checked is None
        target_probe = overlap + 1e-10 >= prepared['effective_target']/100.
        if checked is None and not used_conditioning and conditioning_calls < 2 and (result.status in (0, 4) or
                (target_probe and (result.status == 1 or poorly_scaled))):
            result = conditioned_linprog(linprog, objective, A_ub=matrix, b_ub=rhs, A_eq=a_eq, b_eq=eq_rhs,
                bounds=bounds, scales=scales)
            conditioning_calls += 1
            calls += 1
            recovery_calls += 1
            checked = verified(result)
            if checked is not None:
                prefer_conditioned = True
        if checked is None:
            if result.status == 2 and not unverified_success:
                return False
            incomplete = True
            return None
        weights, score = checked
        if score > best_score + 1e-9:
            best, best_score = weights, score
        return True

    lo, hi = 0., 1.
    target_reached = False
    if request.search_goal == "reach_target" or target_only:
        # Clearance is a numerical guard, never a relaxed acceptance threshold.
        target_overlap = min(1., (prepared["effective_target"] + .2) / 100.)
        status = solve(target_overlap)
        if best_score + 1e-8 >= prepared["effective_target"]:
            lo, target_reached = target_overlap if status else min(target_overlap, best_score/100.), True
        if not target_reached:
            exact_target = prepared["effective_target"] / 100.
            status = solve(exact_target)
            if best_score + 1e-8 >= prepared["effective_target"]:
                lo, target_reached = exact_target if status else min(exact_target, best_score/100.), True
    feasible = False if target_reached or target_only else solve(0.)
    if target_only and not target_reached and improve_score_from is not None:
        # Spend a small improvement budget only above the incumbent's score.
        # These lower search probes NEVER change effective_target or approval.
        if not np.isfinite(improve_score_from) or not 0 <= improve_score_from < prepared['effective_target']:
            raise ValueError('refinement incumbent must be below the unchanged target')
        probe_lo, probe_hi = max(improve_score_from, best_score) / 100., prepared['effective_target'] / 100.
        for _ in range(4):
            middle = (probe_lo+probe_hi)/2.
            status = solve(middle)
            if status is None:
                break
            if status:
                probe_lo, lo = middle, max(lo, middle)
            else:
                probe_hi, hi = middle, min(hi, middle)
    if feasible:
        for _ in range(12):
            middle = (lo + hi) / 2
            status = solve(middle)
            if status is None:
                break
            if status:
                lo = middle
            else:
                hi = middle
    accord_report = {'method': 'product_response_accord_block_trials/v1', 'product': 'body_lotion',
        'trials': 0, 'accepted': 0, 'baseline_score': best_score if best is not None else None,
        'response_basis': weighting, 'human_panel_simulated': False}
    if best is not None and best_score+1e-8 < prepared['effective_target']:
        from .accord_trials import accord_trials
        for _ in range(2):
            incumbent = best.copy()
            scores = assessments(incumbent)[1]
            worst = int(np.argmin([row['score'] for row in scores]))
            for candidate in accord_trials(incumbent, profiles, responses[worst], targets[worst], lower, caps):
                accord_report['trials'] += 1
                if not valid(candidate):
                    continue
                candidate_score = assessments(candidate)[0]
                if candidate_score > best_score+1e-8:
                    best, best_score = candidate, candidate_score
                    accord_report['accepted'] += 1
            if np.array_equal(best, incumbent) or best_score+1e-8 >= prepared['effective_target']:
                break
        lo = max(lo, best_score/100.)
        hi = max(hi, lo)
    accord_report['selected_score'] = best_score if best is not None else None
    # The V44 large-catalog pilot did not recover a pass and frequently hit its
    # CPU budget. Keep the exact-cone experiment opt-in, not default API latency.
    conic_report = {'status': 'existing_target_already_met' if experimental_conic_recovery else
        'experimental_solver_disabled', 'attempted': False, 'accepted': False, 'solver_calls': 0}
    if experimental_conic_recovery and best_score+1e-8 < prepared['effective_target']:
        from .lotion_conic import solve_full_profile
        candidate, conic_report = solve_full_profile(profiles=profiles, targets=targets, responses=responses,
            target_score=prepared['effective_target'], residual_rows=residual_rows, slack_sums=slack_sums,
            outside=outside, avoid_vectors=avoid_vectors, fixed_rows=fixed_rows, fixed_rhs=fixed_rhs,
            a_eq=a_eq, eq_rhs=eq_rhs, bounds=bounds, objective=objective)
        calls += conic_report['solver_calls']
        incomplete = incomplete or conic_report.get('solver_incomplete', False)
        if candidate is not None:
            candidate = np.maximum(0., candidate)
            candidate /= candidate.sum()
            if valid(candidate):
                candidate_score = assessments(candidate)[0]
                conic_report['recomputed_score'] = candidate_score
                if candidate_score > best_score+1e-8:
                    best, best_score = candidate, candidate_score
                    lo = max(lo, best_score/100.)
                    hi = max(hi, lo)
                    conic_report.update(accepted=True, target_met=best_score+1e-8 >= prepared['effective_target'])
            else:
                conic_report['candidate_rejected_by_exact_constraints'] = True
    if best is None:
        return {"schema_version": "lotion-optimization-1", "status": "search_incomplete" if incomplete else
                "target_not_found" if target_only else "no_feasible_research_formula",
                "preparation": prepared, "recipe": [], "closest_candidate": [], "solver_calls": calls,
                "conic_recovery": conic_report, "accord_refinement": accord_report,
                "human_similarity_percent": None, "manufacturing_approved": False}
    learned_report = None
    incumbent = best.copy()
    incumbent_checks = assessments(incumbent)[1]
    if _shape_predictor is not None:
        from .lotion_learned_search import refine_lotion_weights
        best, learned_report = refine_lotion_weights(predictor=_shape_predictor, pool=pool, responses=responses,
            target_rows=target_rows, incumbent=incumbent, residual_rows=residual_rows, slack_sums=slack_sums,
            outside=outside, avoid_vectors=avoid_vectors, profiles=profiles, fixed_rows=fixed_rows, fixed_rhs=fixed_rhs,
            a_eq=a_eq, eq_rhs=eq_rhs, bounds=bounds, prices=prices, valid=valid, assessments=assessments,
            target_score=prepared['effective_target'])
    def final_simulations(weights):
        selected = [(item, weight) for item, weight in zip(pool, weights) if weight > 0]
        results = []
        for basis in basis_requests:
            by_id = {m.ingredient_id: m for m in basis.materials}
            materials = [by_id[item.ingredient_id].model_copy(update={"concentrate_percent": float(weight*100)}) for item, weight in selected]
            results.append(simulate_lotion(basis.model_copy(update={"materials": materials}), IngredientCatalog([item for item, _ in selected])))
        return selected, results
    selected, final_results = final_simulations(best)
    extra_simulations = 0
    if learned_report and learned_report['recipe_changed']:
        from .lotion_learned_search import fresh_curve_affinity
        points = [point for result in final_results for point in result['temporal_profile'][1:]]
        if len(points) != len(target_rows):
            raise ValueError('learned lotion final profile count mismatch')
        fresh_scores = np.asarray([compare_profiles(row['target_profile'], point['scent_profile'] or {},
            avoided=row['avoided']).score for row, point in zip(target_rows, points)])
        guard_checks = {}
        def guard(name, actual, required, tolerance):
            actual, required = np.asarray(actual), np.asarray(required)
            if actual.shape != required.shape or not np.isfinite(actual).all() or not np.isfinite(required).all():
                raise ValueError('learned lotion final guard shape or value invalid')
            margin = float(np.min(actual-required))
            passed = bool(np.all(actual+tolerance >= required))
            guard_checks[name] = {'passed': bool(passed), 'minimum_margin_points': margin}
            return bool(passed)
        protected_scores = refinement_score_floors([check['score'] for check in incumbent_checks], prepared['effective_target'])
        verified = guard('incumbent_strict_scores', fresh_scores, protected_scores, 1e-8)
        verified = verified and all(result['status'] != 'outside_open_sink_assumption' for result in final_results)
        fresh_affinity = 100*fresh_curve_affinity(final_results, target_rows, pool, _shape_predictor)
        verified = guard('incumbent_learned_affinities', fresh_affinity,
                         learned_report['baseline_reference_affinities'], 1e-7) and verified
        affinity_improved = float(fresh_affinity.min()) > learned_report['baseline_affinity']+1e-5
        balance = learned_report.get('profile_balance')
        balance_improved = False
        if balance is not None:
            from .lotion_profile_balance import fresh_curve_profile_match
            fresh_balance = 100*fresh_curve_profile_match(final_results, target_rows, pool, _shape_predictor)
            balance_verified = guard('balance_profile_scores', fresh_balance, balance['baseline_reference_scores'], 1e-7)
            balance_verified = guard('balance_learned_affinities', fresh_affinity,
                                     balance['previous_reference_affinities'], 1e-7) and balance_verified
            balance_verified = guard('balance_strict_scores', fresh_scores,
                balance.get('required_strict_scores', balance['previous_strict_scores']), 1e-8) and balance_verified
            balance_improved = balance_verified and float(fresh_balance.min()) > balance['baseline_score']+1e-5
            verified = verified and balance_verified
            balance.update(fresh_transport_verified=balance_verified, selected_score=float(fresh_balance.min()),
                           selected_reference_scores=fresh_balance.tolist())
        verified = verified and (affinity_improved or balance_improved)
        precision = learned_report.get('precision_refinement')
        if precision:
            verified = guard('precision_learned_affinities', fresh_affinity,
                             precision['previous_reference_affinities'], 1e-7) and verified
            verified = guard('precision_strict_scores', fresh_scores,
                precision.get('required_strict_scores', precision['previous_strict_scores']), 1e-8) and verified
        learned_report['fresh_transport_checks'] = guard_checks
        learned_report['proposed_fresh_strict_score'] = float(fresh_scores.min())
        learned_report['proposed_fresh_affinity'] = float(fresh_affinity.min())
        learned_report['fresh_transport_verified'] = verified
        if verified:
            learned_report['selected_affinity'] = float(fresh_affinity.min())
            learned_report['selected_reference_affinities'] = fresh_affinity.tolist()
        if not verified:
            best = incumbent
            selected, final_results = final_simulations(best)
            extra_simulations = len(scenarios)
            learned_report.update(recipe_changed=False, status='fresh_transport_guard_rejected',
                selected_affinity=learned_report['baseline_affinity'], selected_strict_score=assessments(best)[0],
                selected_cost_per_kg=float(prices@best), cost_change_per_kg=0.)
            if balance is not None:
                balance.update(recipe_changed=False, status='final_transport_rejected', selected_score=None,
                               selected_reference_scores=None)
    final_result = final_results[0]
    final_points = [point for result in final_results for point in result["temporal_profile"][1:]]
    # Recompute from returned final curves rather than trusting the LP objective.
    final_checks = [{"minutes": row["minutes"], "phase": row["phase"], "scenario_index": row["scenario_index"],
        **compare_profiles(row["target_profile"], point["scent_profile"] or {}, avoided=row["avoided"]).to_dict()}
        for row, point in zip(target_rows, final_points)]
    score = min(row["score"] for row in final_checks)
    if learned_report is not None and 'selected_strict_score' in learned_report:
        learned_report['selected_strict_score'] = score
        before = np.asarray(learned_report['original_strict_scores'])
        after = np.asarray([row['score'] for row in final_checks])
        learned_report['timepoint_tradeoffs'] = {
            'declined_within_protected_headroom': int(np.sum(after+1e-8 < before)),
            'largest_decline_points': float(max(0., np.max(before-after))),
            'lost_target_timepoints': int(np.sum((before+1e-8 >= prepared['effective_target'])
                                                & (after+1e-8 < prepared['effective_target']))),
            'minimum_score_change': float(after.min()-before.min()),
        }
    target_met = score + 1e-8 >= prepared["effective_target"]
    lines = [{"ingredient_id": item.ingredient_id, "name": item.name, "concentrate_percent": float(weight*100),
        "finished_product_percent": float(weight*simulation.application_context.fragrance_concentration_percent),
        "pyramid": item.pyramid, "price_per_kg": item.price_per_kg,
        "odor_projection_version": item.odor_projection_version or 'explicit-odor-projection-1'} for item, weight in selected]
    digest = hashlib.sha256(json.dumps({"request": request.model_dump(mode="json"), "lines": lines,
        "profiles": profiles.tolist(), "transport_scenarios": [s.model_dump(mode="json") for s in scenarios[1:]]}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # The trained release path consumes the FINAL selected mixture, never a
    # uniform basis or an earlier candidate. Existing numerical acceptance stays.
    from .lotion_surrogate import attach_release_prediction
    for i, basis in enumerate(basis_requests):
        by_id = {m.ingredient_id: m for m in basis.materials}
        materials = [by_id[item.ingredient_id].model_copy(update={"concentrate_percent": float(weight*100)})
                     for item, weight in selected]
        final_results[i] = attach_release_prediction(final_results[i], basis.model_copy(update={"materials": materials}),
            IngredientCatalog([item for item, _ in selected]),
            provider=_shape_predictor.provider if _shape_predictor is not None else None,
            shape_predictor=_shape_predictor)
    final_result = final_results[0]
    return {"schema_version": "lotion-optimization-1", "status": "research_profile_target_met" if target_met else "research_candidate_only",
        "product_model": evaluation_contract(),
        "formula_id": digest, "preparation": prepared, "recipe": lines if target_met else [], "closest_candidate": [] if target_met else lines,
        "score": score, "score_kind": "minimum_timepoint_lotion_model_profile_agreement_not_human_similarity",
        "profile_target_met": target_met, "timepoint_assessments": final_checks,
        **({'learned_optimization': learned_report} if learned_report is not None else {}),
        "estimated_concentrate_cost_per_kg": float(prices @ best), "simulation": final_result,
        "transport_scenario_count": len(scenarios),
        "scenario_simulations": final_results[1:],
        "solver_calls": calls, "solver_recovery_calls": recovery_calls, "search_incomplete": incomplete,
        "conic_recovery": conic_report,
        "accord_refinement": accord_report,
        "solver_conditioning_calls": conditioning_calls,
        "solver_conditioned_mode": prefer_conditioned,
        "profile_lp_slack_variables": n_slack, "full_axis_slack_equivalent": t_count*dims,
        "attainability": {"numeric_overlap_lower_percent": 100*lo,
            "numeric_model_score_upper_percent": 100*hi,
            "requested_target_excluded_by_numeric_bound": 100*hi < prepared["effective_target"]-1e-8,
            "scope": "current_candidate_pool_and_supplied_transport_only_not_human_accuracy",
            "formal_certificate": False, "search_incomplete": incomplete},
        "search_kind": "max_min_overlap_bisection_with_full_profile_recheck",
        "baseline_score": assessments(baseline)[0] if valid(baseline) else None,
        "incumbent_reuse": seed_report,
        "transport_simulation_calls": 2*len(scenarios)-reused_simulations+extra_simulations,
        "reused_basis_simulations": reused_simulations, "basis_cache_status": cache_status, "external_api_calls": 0,
        "screening_scope": "existing_candidate_policy_and_composition_caps_not_lotion_regulatory_approval",
        "human_similarity_percent": None, "manufacturing_approved": False, "all_user_requirements_verified": False,
        "limitations": ["Only supplied fixed-base transport coefficients participate; missing catalog properties are not invented.",
            "Transport coefficients are caller-supplied and unverified; no lotion calibration dataset was fitted.",
            "This score is not directly interchangeable with the perfume engine's aggregate score.",
            "Normalized profile agreement does not prove human intensity, stability or manufacturing suitability.",
            "LP maximizes a minimum overlap surrogate; final full-vector checks do not prove global optimum."]}
