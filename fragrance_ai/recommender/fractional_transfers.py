"""Whole-pool partial-mass transfers, including low-cap trace ingredients.

Replacing an entire donor silently excluded any receiver capped below that
donor's mass. This search visits every legal receiver at several sub-doses,
retains the donor remainder, and evaluates the full nonlinear central mixture.
These are proposals only; the unchanged sampled forward model accepts them.
"""
from types import SimpleNamespace

import numpy as np

from .models import SCENT_DIMENSIONS, profile_vector
from .science import ATMOSPHERIC_PRESSURE_PA, ETHANOL_MOLECULAR_WEIGHT, TIMEPOINTS_MINUTES, TemporalMixtureSimulator


class TransferPhysics:
    def __init__(self, ingredients, properties, concentration):
        self.items = list(ingredients)
        self.properties = properties
        self.engine = TemporalMixtureSimulator()
        self.inputs = self.engine.prepare_response_inputs(self.items, properties)
        self.vectors = np.asarray([item.vector() for item in self.items])
        self.normalized = self.vectors/np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
        self.moles = concentration*self.inputs.strengths/100/self.inputs.molecular_weights
        self.total_moles = self.moles + concentration*self.inputs.carrier_moles_per_g
        oav = self.moles*self.inputs.activities*self.inputs.vapor_pressures/ATMOSPHERIC_PRESSURE_PA*1e6/self.inputs.thresholds
        self.coefficient = oav[:, None]*np.power(.5, np.asarray(TIMEPOINTS_MINUTES)[None]/self.inputs.half_lives[:, None])
        self.base_moles = max(0., 100-concentration)/ETHANOL_MOLECULAR_WEIGHT
        self.gain = np.asarray([item.odor_impact*item.active_strength_percent/100 for item in self.items])
        self.has_props = np.asarray([properties.get(item.ingredient_id) is not None for item in self.items])
        self.logp = np.asarray([properties[item.ingredient_id].xlogp
            if self.has_props[i] and properties[item.ingredient_id].xlogp is not None else np.nan
            for i, item in enumerate(self.items)])

    def cross(self, left, right):
        chemical = np.exp(-np.abs(self.inputs.molecular_weights[left, None]-self.inputs.molecular_weights[right][None])/300.)
        polarity = np.exp(-np.abs(self.logp[left, None]-self.logp[right][None])/3.)
        polarity = np.where(np.isfinite(polarity), polarity, .5)
        chemical = np.where(self.has_props[left, None]&self.has_props[right][None], chemical*polarity, .5)
        return np.clip(.7*np.clip(self.normalized[left]@self.normalized[right].T, 0., 1.)+.3*chemical, 0., 1.)

    def transfer(self, original, donor, receivers, amounts):
        """Amounts and original weights are concentrate MASS FRACTIONS."""
        ids = np.asarray(list(original), dtype=int)
        weights = np.asarray(list(original.values()), dtype=float)
        receivers, amounts = np.asarray(receivers, dtype=int), np.asarray(amounts, dtype=float)
        donor_weight = float(original[donor])
        if (amounts.shape != receivers.shape or np.any(amounts < 0) or np.any(amounts > donor_weight+1e-12)
                or any(int(i) in original for i in receivers) or not np.isfinite(amounts).all()):
            raise ValueError('partial transfer requires new receiver identities and a legal donor mass')
        rest = ids[ids != donor]
        rest_weights = weights[ids != donor]
        retained = np.maximum(0., donor_weight-amounts)
        denominator = np.maximum(1e-12, self.base_moles+rest_weights@self.total_moles[rest]
            +retained*self.total_moles[donor]+amounts*self.total_moles[receivers])
        rest_oav = (self.coefficient[rest]*rest_weights[:, None]).T
        current_rest = self.engine._odor_response(rest_oav[None]/denominator[:, None, None])*self.inputs.transports[rest]
        current_donor = self.engine._odor_response(retained[:, None]*self.coefficient[donor]/denominator[:, None])*self.inputs.transports[donor]
        current_new = self.engine._odor_response(amounts[:, None]*self.coefficient[receivers]/denominator[:, None])*self.inputs.transports[receivers, None]
        current_donor[retained == 0] = 0.
        current_new[amounts == 0] = 0.
        rr = self.engine._interaction_matrix([SimpleNamespace(ingredient=self.items[i], properties=self.properties.get(self.items[i].ingredient_id)) for i in rest]) if len(rest) else np.empty((0,0))
        dr = self.cross(np.asarray([donor]), rest)[0]
        nr = self.cross(receivers, rest)
        nd = self.cross(receivers, np.asarray([donor]))[:, 0]
        suppressed_rest = current_rest/(1+.2*(current_rest@rr.T+current_donor[:, :, None]*dr[None, None]+current_new[:, :, None]*nr[:, None]))
        suppressed_donor = current_donor/(1+.2*(current_rest@dr+current_new*nd[:, None]))
        suppressed_new = current_new/(1+.2*((current_rest*nr[:, None]).sum(-1)+current_donor*nd[:, None]))
        signal = suppressed_rest.sum(-1)+suppressed_donor+suppressed_new
        temporal = suppressed_rest@self.vectors[rest]+suppressed_donor[:, :, None]*self.vectors[donor]+suppressed_new[:, :, None]*self.vectors[receivers, None]
        temporal /= np.maximum(temporal.sum(-1, keepdims=True), 1e-12)
        nominal = (rest_weights*self.gain[rest])@self.vectors[rest]+retained[:, None]*self.gain[donor]*self.vectors[donor]+amounts[:, None]*self.gain[receivers, None]*self.vectors[receivers]
        nominal /= np.maximum(nominal.sum(-1, keepdims=True), 1e-12)
        return nominal, temporal, signal


def _scores(target, predicted, avoided):
    target = np.asarray(target)
    if target.sum() <= 0:
        return np.zeros(len(predicted))
    target = target/target.sum()
    cosine = predicted@target/np.maximum(np.linalg.norm(predicted, axis=1)*np.linalg.norm(target), 1e-12)
    overlap = np.minimum(target, predicted).sum(-1)
    indices = [SCENT_DIMENSIONS.index(name) for name in set(avoided)]
    return np.minimum(np.minimum(cosine, overlap), 1-predicted[:, indices].sum(-1))


def fractional_transfer_seeds(ingredients, brief, properties, original, minimums, *, maximum=5, guidance=None, minimum_guidance=None):
    items = list(ingredients)
    lookup = {item.ingredient_id:i for i,item in enumerate(items)}
    if not original or not set(original) <= set(lookup):
        return [], {'status':'missing_current_recipe', 'full_pool_count':len(items)}
    model = TransferPhysics(items, properties, brief.constraints.product_concentration_percent)
    weights = {lookup[key]:value/100 for key,value in original.items()}
    prices = np.asarray([item.price_per_kg for item in items])
    caps = np.asarray([item.as_supplied_cap_percent()/100 for item in items])
    original_cost = sum(prices[key]*value for key,value in weights.items())
    targets, time_weights = model.engine.targets_by_time(brief), model.engine.time_weights(brief)
    receiver_ids = np.asarray([i for i in range(len(items)) if i not in weights and caps[i] >= 1e-6], dtype=int)
    diagnostic = {'status':'all_eligible_receiver_caps_visited', 'full_pool_count':len(items),
        'receiver_count':len(receiver_ids), 'dose_proposals_evaluated':0,
        'minimum_resolvable_concentrate_percent':.0001, 'whole_donor_replacement_required':False}
    if not len(receiver_ids):
        return [], diagnostic
    records = []
    for donor, donor_weight in weights.items():
        floor = minimums.get(items[donor].ingredient_id, 0.)/100
        available = donor_weight-floor
        if available < 1e-6:
            continue
        maxima = np.minimum(available, caps[receiver_ids])
        deltas = prices[receiver_ids]-prices[donor]
        budget = brief.constraints.max_formula_cost_per_kg-original_cost
        higher = deltas > 0
        maxima[higher] = np.minimum(maxima[higher], max(0., budget)/deltas[higher])
        # Six logarithmic levels cover trace through material-specific maximum.
        for fraction in (1e-3, 1e-2, .05, .2, .5, 1.):
            amounts = np.floor(maxima*fraction*1e6)/1e6
            legal = amounts >= 1e-6
            choices, doses = receiver_ids[legal], amounts[legal]
            if not len(choices):
                continue
            nominal, temporal, signal = model.transfer(weights, donor, choices, doses)
            base_nominal, base_temporal, base_signal = model.transfer(weights, donor, choices[:1], np.zeros(1))
            phase = np.array([_scores(target, temporal[:,t], avoided) for t,(target,_,avoided) in enumerate(targets)]).T
            baseline_phase = np.array([_scores(target, base_temporal[:,t], avoided)[0] for t,(target,_,avoided) in enumerate(targets)])
            quality = np.minimum(_scores(profile_vector(brief.target_profile), nominal, brief.avoided_dimensions), phase@time_weights)
            baseline_quality = min(float(_scores(profile_vector(brief.target_profile), base_nominal, brief.avoided_dimensions)[0]), float(baseline_phase@time_weights))
            rank = quality-np.maximum(0., baseline_phase-phase)@time_weights-.25*(np.maximum(0., .97-signal/np.maximum(base_signal,1e-12))@time_weights)
            if guidance is not None and minimum_guidance is not None and hasattr(guidance, 'transfer_scores'):
                neural = guidance.transfer_scores(original, items, items[donor].ingredient_id,
                    [items[i] for i in choices], doses*100)
                if neural is not None:
                    rank = np.where(np.isfinite(neural)&(neural+1e-8 >= minimum_guidance), rank, -np.inf)
            diagnostic['dose_proposals_evaluated'] += len(choices)
            order = np.argsort(-rank, kind='stable')[:maximum]
            for j in order:
                if not np.isfinite(rank[j]) or rank[j] <= baseline_quality+1e-6:
                    continue
                candidate = dict(original)
                donor_id, receiver_id = items[donor].ingredient_id, items[choices[j]].ingredient_id
                candidate[donor_id] -= float(doses[j]*100)
                if candidate[donor_id] <= 1e-10:
                    candidate.pop(donor_id)
                candidate[receiver_id] = float(doses[j]*100)
                if len(candidate) <= brief.constraints.max_ingredients:
                    records.append((float(rank[j]-baseline_quality), receiver_id, candidate))
    chosen, used = [], set()
    for _, receiver, candidate in sorted(records, key=lambda row:(-row[0],row[1])):
        if receiver not in used:
            used.add(receiver)
            chosen.append(candidate)
        if len(chosen) >= maximum:
            break
    diagnostic['diverse_seeds_returned'] = len(chosen)
    return chosen, diagnostic
