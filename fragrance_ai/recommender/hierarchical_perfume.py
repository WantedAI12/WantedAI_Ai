"""One full-reference perfume objective for search, gradients and final output."""
from collections import OrderedDict
import numpy as np

from .full_reference_guidance import FullReferenceSession
from .nonlinear_inverse import NonlinearDoseObjective, profile_loss
from .odor_space import target_report
from .science import TemporalMixtureSimulator, TIMEPOINTS_MINUTES

VERSION = 'hierarchical-perfume-reference/v77'


def target_rows(brief):
    from .models import SCENT_DIMENSIONS
    if not sum(brief.target_profile.values()) and not brief.phase_target_profiles:
        # A recognized but unanchored odor is not a contradictory phase ban.
        # The compiler must report its missing reference before transport runs.
        return [{'phase': 'overall', 'target_profile': brief.target_profile, 'avoided': brief.avoided_dimensions},
            *[{'phase': TemporalMixtureSimulator._phase_for_time(t),
               'target_profile': brief.target_profile, 'avoided': brief.avoided_dimensions}
              for t in TIMEPOINTS_MINUTES]]
    return [{'phase':'overall', 'target_profile':brief.target_profile, 'avoided':brief.avoided_dimensions},
        *[{'phase':TemporalMixtureSimulator._phase_for_time(t),
           'target_profile':dict(zip(SCENT_DIMENSIONS, q)), 'avoided':avoid}
          for t,(q,_,avoid) in zip(TIMEPOINTS_MINUTES, TemporalMixtureSimulator.targets_by_time(brief))]]


def active(provider, brief):
    bank = getattr(provider, 'complete_reference_bank', None)
    return (bank is not None and getattr(bank,'odor_space',None) is not None
            and brief.target_profile_source != 'explicit_structured_relative_weights'
            and not brief.constraints.reference_target_id)


def evaluate_reference(provider, brief, lines, ingredients, properties, draws):
    bank = provider.complete_reference_bank
    report = target_report(bank, brief, target_rows(brief))
    result = {'version': VERSION, 'intent':report, 'score':None, 'status':report['status'],
        'draws':draws, 'target_met':False, 'prediction_kind':'nonlinear_dose_headspace_weighted_source_profiles',
        'actual_human_similarity_measured':False}
    if not report['searchable'] or not lines:
        return result
    selected = [ingredients[line.ingredient_id] for line in lines]
    session = provider.begin_reference_shapes()
    session.prefetch(selected)
    values = [session.shape(item) for item in selected]
    if any(p is None for p in values):
        result.update(status='component_reference_missing',
                      missing_ingredients=[i.ingredient_id for i,p in zip(selected,values) if p is None])
        return result
    p = np.stack(values, axis=1)
    w = np.asarray([line.concentrate_percent/100 for line in lines])
    q = np.stack([row['profiles'] for row in report['targets']], axis=1)
    mask = np.asarray([[name in row['avoided'] for name in bank.endpoints] for row in report['targets']],float)
    objective = NonlinearDoseObjective(selected, properties, brief.constraints.product_concentration_percent, draws=draws)
    prediction, _ = objective.predict(p, w)
    loss, _ = profile_loss(prediction, q, np.r_[0.,TemporalMixtureSimulator.time_weights(brief)], mask)
    score = float(100*(1-loss))
    complete = report['coverage']['complete']
    result.update(status='computed_source_reference_profile' if complete else 'partial_source_reference_candidate',
        score=score if complete else None, partial_profile_score=None if complete else score,
        score_scope=report['coverage']['score_scope'],
        target_met=complete and score+1e-8 >= brief.constraints.target_similarity,
        endpoints=list(bank.endpoints), predicted_profiles=prediction.tolist(),
        aggregation='worst_source_head_of_max_nominal_and_time_weighted_profile_loss',
        target_fixed_before_candidate_search=True, descriptor_count=len(bank.endpoints),
        model_sha256=provider.core.sha256, reference_sha256=bank.sha256)
    return result


class PhysicalReferenceSession(FullReferenceSession):
    """No independent additive guard competing with the physical objective."""
    authoritative_profile_objective = True

    def __init__(self, provider, brief):
        super().__init__(provider, brief)
        self.intent = target_report(self.bank, brief, target_rows(brief))
        self.enabled = self.product_supported and self.intent['searchable']
        self.unsupported_intent = self.intent['unsupported']
        self.properties = None
        self.functions = OrderedDict()
        if self.enabled:
            self.q = np.stack([r['profiles'] for r in self.intent['targets']],axis=1)
            self.mask = np.array([[n in r['avoided'] for n in self.bank.endpoints] for r in self.intent['targets']],float)
            self.tw = np.r_[0., TemporalMixtureSimulator.time_weights(brief)]
            # Parent setup disables its tensors on ANY missing detail. Rebuild
            # them for partial search too; reference seeding uses these fields.
            self.target_matrix = np.concatenate([r['profiles'] for r in self.reference_targets], axis=0)
            self.avoidance = np.repeat(self.mask[1:], 2, axis=0)

    def bind_properties(self, properties):
        self.properties = properties
        self.functions.clear()

    def gradient_function(self, ingredients):
        if not self.enabled or self.properties is None:
            return None
        ids = tuple(i.ingredient_id for i in ingredients)
        if ids in self.functions:
            return self.functions[ids]
        self.shapes.prefetch(ingredients)
        values = [self.shapes.shape(i) for i in ingredients]
        if any(p is None for p in values):
            self.missing_ids.update(i.ingredient_id for i,p in zip(ingredients,values) if p is None)
            return None
        p = np.stack(values,axis=1)
        objective = NonlinearDoseObjective(ingredients, self.properties,
            self.brief.constraints.product_concentration_percent, draws=self.brief.constraints.simulation_draws)
        def score(w):
            loss, gradient, _ = objective(p[None], self.q[None], None, np.asarray(w)[None], [0],
                time_weights=self.tw[None], avoided=np.broadcast_to(self.mask,self.q.shape)[None])
            return float(100*(1-loss[0])), -100*gradient[0]
        self.functions[ids] = score
        while len(self.functions) > 2:
            self.functions.popitem(last=False)
        return score

    def evaluate(self, weights, ingredients, *, exact=False):
        w = np.asarray(weights,float)
        if w.shape != (len(ingredients),) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-100)>.002:
            raise ValueError('finite formula percentages summing to 100 required')
        positive = w > 0
        fun = self.gradient_function([i for i,ok in zip(ingredients,positive) if ok])
        if fun is None:
            return None
        score,_ = fun(w[positive]/100)
        self.exact_calls += int(exact)
        self.grid_calls += int(not exact)
        return {'score':score, 'nominal_score':score, 'checkpoint_sha256':self.provider.core.sha256,
            'reference_sha256':self.bank.sha256, 'target_sha256':self.intent['sha256'],
            'exact_forward':True, 'descriptor_count':len(self.bank.endpoints),
            'prediction_kind':VERSION, 'target_fit_to_candidate_catalogue':False}

    def transfer_scores(self, original, ingredients, donor_id, replacements, amounts_percent):
        amounts = np.asarray(amounts_percent,float)
        if donor_id not in original or amounts.shape != (len(replacements),) or not np.isfinite(amounts).all() or np.any((amounts<0)|(amounts>original[donor_id]+1e-10)):
            raise ValueError('invalid physical reference transfer')
        lookup = {i.ingredient_id:i for i in [*ingredients,*replacements]}
        output = np.full(len(replacements),np.nan)
        for j,(item,amount) in enumerate(zip(replacements,amounts)):
            weights = dict(original)
            weights[donor_id] -= amount
            weights[item.ingredient_id] = weights.get(item.ingredient_id,0.) + amount
            weights = {k:v for k,v in weights.items() if v>0}
            value = self.evaluate(list(weights.values()),[lookup[k] for k in weights])
            if value is not None:
                output[j] = value['score']
        return output

    def report(self, *args, **kwargs):
        value = super().report(*args, **kwargs)
        value.update(guidance_version=VERSION, physical_objective_separate=False,
            primary_and_guidance_objective_identical=True, target_intent=self.intent)
        return value


def native_proposals(candidates, brief, properties, baseline_lines, policy, minimums, guidance, diagnostics):
    """All eligible reference-covered ingredients enter the trained inverse."""
    from .failure_recovery import recovery_active
    if recovery_active():
        from .failure_hierarchical_v87 import native_proposals as recover
        yield from recover(candidates, brief, properties, baseline_lines, policy, minimums, guidance, diagnostics)
        return
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
    from .reference_inverse import source_fixed_proposals
    yield from source_fixed_proposals(items, brief, properties, original, minimums, policy, guidance, diagnostics)
