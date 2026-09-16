"""Enhanced search, invoked only after the unchanged baseline fails."""
import numpy as np

VERSION = "hierarchical-perfume-reference/v77"

def native_proposals(candidates, brief, properties, baseline_lines, policy, minimums, guidance, diagnostics):
    """All eligible reference-covered ingredients enter the trained inverse."""
    from .reference_profile_refinement import reference_profile_seeds
    from .fractional_transfers import TransferPhysics
    # Reuse the exact transport carrier and all-candidate reference seeding.
    original = {line.ingredient_id:line.concentrate_percent for line in baseline_lines}
    transport = TransferPhysics(candidates, properties, brief.constraints.product_concentration_percent)
    seeds, report = reference_profile_seeds(candidates, brief, transport, original, minimums, guidance, bank=guidance.bank)
    diagnostics.update(report, primary_objective=VERSION, legacy_requested_target_and_score_unchanged=False,
        legacy_score_retained_as_separate_diagnostic=True)
    for label, weights in seeds:
        yield {'weights_percent':weights, 'anchor_weights_percent':original,
            'replacement_mode':label, 'allocation_mode':'inferred_full_range',
            'full_pool_considered':len(candidates), 'target_sha256':guidance.intent['sha256']}
    # The learned decoder may abstain on a new semantic target. Use the exact
    # physical gradient over EVERY reference-covered material as a bounded
    # second proposal source, not an after-the-fact score correction.
    from .constrained_autoregressive import project_constraints
    from .autoregressive_refinement import neural_mass_seeds
    guidance.shapes.prefetch(candidates)
    items = [i for i in candidates if guidance.shapes.shape(i) is not None]
    if not set(original) <= {i.ingredient_id for i in items}:
        return
    # Reserve computation for support-changing physical inversion before a
    # sequence of full-pool projected mass steps can consume the request budget.
    from .failure_inverse_v87 import source_fixed_proposals
    yield from source_fixed_proposals(items, brief, properties, original, minimums, policy, guidance, diagnostics)
    function = guidance.gradient_function(items)
    if function is None:
        return
    w = np.array([original.get(i.ingredient_id,0.)/100 for i in items])
    lower = np.array([minimums.get(i.ingredient_id,0.)/100 for i in items])
    upper = np.array([min(1.,i.as_supplied_cap_percent()/100) for i in items])
    prices = np.array([i.price_per_kg for i in items])
    budget = np.array([brief.constraints.max_formula_cost_per_kg])
    score, gradient = function(w)
    diagnostics['physical_polish']={'all_candidate_columns':len(items),'evaluations':1,'accepted_steps':0,
        'starting_score':score,'selected_score':score,'final_draws':brief.constraints.simulation_draws}
    for iteration in range(4):
        scale = max(float(np.max(np.abs(gradient-gradient.mean()))),1e-12)
        moved=False
        for step in (.05,.01,.002,.0004,.00008):
            projected=project_constraints((w+step*(gradient-gradient.mean())/scale)[None],
                lower[None],upper[None],prices[None],budget)[0]
            proposals,_=neural_mass_seeds(items,projected,{i.ingredient_id:float(v*100) for i,v in zip(items,w) if v>0},
                minimums,brief.constraints.max_ingredients)
            if not proposals:
                continue
            weights=proposals[-1][1]
            candidate=np.array([weights.get(i.ingredient_id,0.)/100 for i in items])
            if (np.any(candidate>upper+1e-10) or np.any(candidate<lower-1e-10)
                    or candidate@prices>budget[0]+1e-7 or np.allclose(candidate,w,atol=1e-14,rtol=0)):
                continue
            value, derivative=function(candidate)
            diagnostics['physical_polish']['evaluations']+=1
            if value>score+1e-8:
                w,score,gradient=candidate,value,derivative
                diagnostics['physical_polish']['accepted_steps']+=1
                diagnostics['physical_polish']['selected_score']=score
                yield {'weights_percent':weights,'anchor_weights_percent':original,
                    'replacement_mode':'hierarchical_exact_physics_polish','allocation_mode':'inferred_full_range',
                    'full_pool_considered':len(candidates),'proposal_profile_score':score,
                    'target_sha256':guidance.intent['sha256'],'iteration':iteration+1}
                moved=True
                break
        if not moved or score+1e-8>=brief.constraints.target_similarity:
            break
