"""Request-local residual feedback, not self-label training or a new score.

For accepted iterates x_k with residuals r_k, a regularized multisecant step
fits R a = -r_k and proposes x_k + D a. A trust region bounds supplied mass
movement. Product-specific physics and every original constraint must still
verify the proposal; this module cannot approve a formula.
"""
from dataclasses import dataclass

import numpy as np

VERSION = 'autoregressive-residual-refinement/v1'


@dataclass(frozen=True)
class FeedbackState:
    weights: np.ndarray
    residual: np.ndarray
    score: float
    qualified: bool


class ResidualFeedback:
    def __init__(self, *, target, history_size=4, max_step_l1=.15):
        if (not np.isfinite(target) or not 0 < target <= 100
                or not isinstance(history_size, int) or not 2 <= history_size <= 8
                or not np.isfinite(max_step_l1) or not 0 < max_step_l1 <= 1):
            raise ValueError('finite bounded residual-feedback configuration required')
        self.target, self.history_size, self.max_step_l1 = float(target), history_size, max_step_l1
        self.history = []
        self.accepted_updates = self.proposal_count = 0
        self.events = []

    def observe(self, weights, residual, score, *, qualified=False):
        """Only call after the product-specific verifier accepts this state."""
        weights, residual = np.array(weights, float, copy=True), np.array(residual, float, copy=True).ravel()
        if (weights.ndim != 1 or not len(weights) or not len(residual)
                or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0
                or not np.isfinite(residual).all() or not np.isfinite(score) or not 0 <= score <= 100+1e-7
                or not isinstance(qualified, bool)):
            raise ValueError('finite verified feedback state required')
        if qualified and score+1e-7 < self.target:
            raise ValueError('subthreshold state cannot be qualified')
        if self.history:
            previous = self.history[-1]
            if (weights.shape != previous.weights.shape or residual.shape != previous.residual.shape
                    or not np.isclose(weights.sum(), previous.weights.sum(), atol=1e-6, rtol=1e-8)):
                raise ValueError('feedback coordinate or mass contract changed')
            if (qualified, score) <= (previous.qualified, previous.score):
                return False
            self.accepted_updates += 1
        weights.setflags(write=False)
        residual.setflags(write=False)
        self.history.append(FeedbackState(weights, residual, float(score), qualified))
        self.history = self.history[-self.history_size:]
        self.events.append({'score':float(score), 'qualified':qualified,
            'residual_l2':float(np.linalg.norm(residual)), 'residual_max':float(np.abs(residual).max())})
        # A pathological caller cannot turn diagnostics into unbounded storage.
        self.events = self.events[-32:]
        return True

    def proposal(self):
        if len(self.history) < 2 or self.history[-1].qualified:
            return None
        latest = self.history[-1]
        dr = np.column_stack([b.residual-a.residual for a,b in zip(self.history, self.history[1:])])
        dx = np.column_stack([b.weights-a.weights for a,b in zip(self.history, self.history[1:])])
        scale = float(np.linalg.norm(dr, ord=2))
        if not np.isfinite(scale) or scale <= 1e-12:
            return None
        # Scaling and a fixed relative ridge prevent singular/near-collinear
        # accepted trajectories from producing arbitrarily large directions.
        normalized = dr/scale
        coefficients = np.linalg.solve(normalized.T@normalized+1e-4*np.eye(dr.shape[1]),
            -normalized.T@(latest.residual/scale))
        direction = dx@coefficients
        size = float(np.abs(direction).sum())
        if not np.isfinite(direction).all() or size <= 1e-12:
            return None
        factor = min(1., self.max_step_l1*float(latest.weights.sum())/size)
        negative = direction < 0
        if negative.any():
            factor = min(factor, float(np.min(latest.weights[negative]/-direction[negative])))
        if factor <= 1e-12:
            # A depleted coordinate must not cancel the entire direction.
            # Product constraints and the final verifier still decide adoption.
            from .constrained_autoregressive import capped_simplex
            total=float(latest.weights.sum())
            bounded=min(1.,self.max_step_l1*total/size)
            proposal=capped_simplex(((latest.weights+bounded*direction)/total)[None],
                np.zeros((1,len(direction))),np.ones((1,len(direction))))[0]*total
            movement=proposal-latest.weights
            distance=float(np.abs(movement).sum())
            if distance<=1e-12:
                return None
            proposal=latest.weights+min(1.,self.max_step_l1*total/distance)*movement
            self.proposal_count+=1
            return np.maximum(proposal,0.)
        proposal = latest.weights+factor*direction
        if np.any(proposal < -1e-10):
            return None
        self.proposal_count += 1
        return np.maximum(proposal, 0.)

    def report(self, **extra):
        return {'version':VERSION,'target_unchanged':self.target,
            'verified_feedback_updates':self.accepted_updates,'secant_proposals':self.proposal_count,
            'history_capacity':self.history_size,'accepted_history':list(self.events),
            'network_weights_updated':False,'self_generated_training_labels_written':False,
            'external_api_calls':0,'proposal_is_not_approval':True,**extra}


def perfume_residual(assessment, dimensions):
    """Use actual final-draw profiles, including every unrequested axis."""
    rows = [assessment['nominal'], *assessment['temporal']]
    parts = []
    for index,row in enumerate(rows):
        weight = 1. if index == 0 else float(row['weight'])
        target, predicted = row['target_profile'], row['predicted_profile']
        values = np.array([target.get(a,0.)-predicted.get(a,0.) for a in dimensions])
        parts.append(np.sqrt(max(0.,weight))*values)
    return np.concatenate(parts)


def perfume_feedback_weights(values):
    """Normalize only feedback coordinates; never edit a rendered recipe.

    Existing four-decimal concentrate lines can sum to 100 +/- 0.001.
    That established display precision must not crash the feedback observer.
    """
    values=np.array(values,float,copy=True)
    total=float(values.sum())
    if not np.isfinite(values).all() or np.any(values<0) or abs(total-100.)>.00101:
        raise ValueError('invalid rendered formula mass in feedback')
    return values*(100./total)


def neural_mass_seeds(items, predicted, original, minimums, support_budget):
    """Retain the feasible incumbent support when projecting neural proposals.

    A network may concentrate on a few low-cap odorants. Their complement must
    stay available for physical mass balance; otherwise projection can reject
    every neural proposal before the original verifier ever sees it.
    """
    identifiers=[item.ingredient_id for item in items]
    lookup={key:i for i,key in enumerate(identifiers)}
    values=np.asarray(predicted,float)
    if values.shape!=(len(items),) or not np.isfinite(values).all() or np.any(values<0) or values.sum()<=0:
        raise ValueError('finite neural mass proposal required')
    values=values/values.sum()
    current=np.array([original.get(key,0.) for key in identifiers])/100.
    if current.sum()<=0 or not set(original)<=set(lookup):
        raise ValueError('known incumbent required for neural projection')
    current/=current.sum()
    support={lookup[key] for key,value in original.items() if value>0}
    support.update(lookup[key] for key,value in minimums.items() if value>0)
    if len(support)>support_budget:
        return [],{'status':'incumbent_exceeds_current_support_budget'}
    for i in np.argsort(-values,kind='stable'):
        if len(support)>=support_budget:
            break
        if values[i]>1e-10:
            support.add(int(i))
    distance=float(np.abs(values-current).sum())
    fraction=min(1.,.15/max(distance,1e-12))
    damped=(1-fraction)*current+fraction*values
    proposals=[]
    for name,proposal in (('trusted_neural_feedback',damped),('trained_autoregressive_decoder',values)):
        mass=sum(float(proposal[i]) for i in support)
        if mass>0:
            proposals.append((name,{identifiers[i]:float(proposal[i]*100/mass) for i in sorted(support)}))
    return proposals,{'status':'incumbent_support_retained','full_pool_scored':len(items),
        'proposal_support_count':len(support),'permanent_candidate_shortlist':False,
        'neural_trust_fraction':fraction,'maximum_damped_full_pool_mass_l1':.15}
