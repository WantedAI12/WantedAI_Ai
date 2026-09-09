"""Joint-time, actual-dose proposals for the existing scientific simulator.

These are search proposals, not a new public score or new sensory evidence.
The service rechecks every adopted proposal with its unchanged final-draw
simulator, complete-profile score, safety gates and V9 preservation guards.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.optimize import linprog, minimize

from .adaptive_pyramid import DIFFUSION
from .global_profile_search import optimize_full_pool, profile_upper_bound
from .models import SCENT_DIMENSIONS, profile_vector
from .science import ATMOSPHERIC_PRESSURE_PA, ETHANOL_MOLECULAR_WEIGHT, TIMEPOINTS_MINUTES, TemporalMixtureSimulator


@dataclass
class DoseState:
    nominal: np.ndarray
    temporal: np.ndarray
    signal: np.ndarray
    nominal_jac: np.ndarray
    temporal_jac: np.ndarray
    signal_jac: np.ndarray


class DoseModel:
    """Fixed per-material prior samples give smooth, common-random proposals."""

    def __init__(self, ingredients, properties, concentration, draws=32):
        self.ingredients = list(ingredients)
        ingredients = self.ingredients
        self.vectors = np.asarray([item.vector() for item in ingredients])
        self.gain = np.asarray([item.odor_impact * item.active_strength_percent / 100 for item in ingredients])
        simulator = TemporalMixtureSimulator()
        props = [properties.get(item.ingredient_id) for item in ingredients]
        self.moles = np.asarray([concentration * item.active_strength_percent / 100 / (prop.molecular_weight if prop else 180.)
                                 for item, prop in zip(ingredients, props)])
        self.base_moles = max(0., 100 - concentration) / ETHANOL_MOLECULAR_WEIGHT
        self.transport = np.asarray([simulator._air_to_receptor_transport(prop) for prop in props])
        self.interaction = simulator._interaction_matrix([SimpleNamespace(ingredient=item, properties=prop) for item, prop in zip(ingredients, props)])
        self.coefficients = np.empty((draws, len(TIMEPOINTS_MINUTES), len(ingredients)))
        self.suppression = np.clip(np.random.default_rng(98173).normal(.20, .05, draws), .08, .40)
        for index, (item, prop) in enumerate(zip(ingredients, props)):
            seed = int(hashlib.sha256(("dose-prior-search-1:" + item.ingredient_id).encode()).hexdigest()[:16], 16)
            noise = np.random.default_rng(seed).normal(size=(draws, 3))
            pressure, vapor_sigma = simulator._vapor_pressure_prior(item, prop)
            threshold, threshold_sigma = simulator._threshold_prior(item, prop)
            pressure = pressure * np.exp(noise[:, 0] * vapor_sigma)
            threshold = threshold * np.exp(noise[:, 1] * threshold_sigma)
            logp = prop.xlogp if prop and prop.xlogp is not None else 2.
            activity = max(.5, min(3., np.exp(.18 * (logp - 2.)))) * np.exp(noise[:, 2] * (.18 if prop else .50))
            lives = np.asarray([simulator._half_life_minutes(item, prop, value) for value in pressure])
            oav = self.moles[index] * activity * pressure / ATMOSPHERIC_PRESSURE_PA * 1e6 / np.maximum(threshold, 1e-12)
            self.coefficients[:, :, index] = oav[:, None] * np.power(.5, np.asarray(TIMEPOINTS_MINUTES)[None, :] / lives[:, None])
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.moles).all():
            raise ValueError("nonfinite dose-model inputs")

    def evaluate(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.gain.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("invalid dose-model weights")
        denominator = max(1e-12, self.base_moles + weights @ self.moles)
        activity = weights * self.coefficients / denominator
        current = TemporalMixtureSimulator._odor_response(np.maximum(activity, 1e-12)) * self.transport
        slope = .55 * current * (1 - current / self.transport)
        slope = np.where(activity > 1e-12, slope, 0.)
        suppression = 1 + self.suppression[:, None, None] * (current @ self.interaction.T)
        suppressed = current / suppression
        signal = suppressed.sum(axis=2)
        raw = suppressed @ self.vectors
        total = np.maximum(raw.sum(axis=2), 1e-12)
        temporal = raw / total[..., None]
        if len(weights) <= 32:
            # Preserve the existing small-support numerical trajectory.
            log_jac = np.diag(1 / np.maximum(weights, 1e-12)) - self.moles[None, :] / denominator
            current_jac = slope[..., None] * log_jac
            suppressed_jac = (current_jac / suppression[..., None]
                              - (current / suppression ** 2)[..., None] * self.suppression[:, None, None, None]
                              * (self.interaction @ current_jac))
            signal_jac = suppressed_jac.sum(axis=2)
            raw_jac = self.vectors.T @ suppressed_jac
        else:
            # Contract the 19 odor outputs plus total signal BEFORE applying
            # the dose Jacobian. Exact chain rule, without draw*time*N*N
            # intermediates or a draw*time*N^3 matrix product.
            channels = np.c_[self.vectors, np.ones(len(weights))].T
            adjoint = channels / suppression[..., None, :]
            adjoint -= ((channels * (current / suppression ** 2)[..., None, :])
                        * self.suppression[:, None, None, None]) @ self.interaction
            scaled = adjoint * slope[..., None, :]
            output_jac = (scaled / np.maximum(weights, 1e-12)
                          - scaled.sum(axis=-1)[..., None] * (self.moles / denominator))
            raw_jac, signal_jac = output_jac[:, :, :-1, :], output_jac[:, :, -1, :]
        # Vectors are normalized; retain the actual raw sum in the derivative
        # as well so an all-zero descriptor row cannot create fake profile mass.
        total_jac = raw_jac.sum(axis=2)
        temporal_jac = (raw_jac - temporal[..., None] * total_jac[:, :, None, :]) / total[..., None, None]
        nominal_raw = (weights * self.gain) @ self.vectors
        nominal_total = max(1e-12, float(nominal_raw.sum()))
        nominal = nominal_raw / nominal_total
        nominal_jac = (self.vectors.T * self.gain - nominal[:, None] * (self.vectors.sum(axis=1) * self.gain)) / nominal_total
        return DoseState(nominal, temporal.mean(axis=0), signal.mean(axis=0), nominal_jac,
                         temporal_jac.mean(axis=0), signal_jac.mean(axis=0))


def agreement_components(target, predicted, jacobian, avoided=()):
    """The same three full-profile components; values are fractions here."""
    target = profile_vector(target) if isinstance(target, dict) else np.asarray(target)
    if target.sum() <= 0:
        raise ValueError("complete positive target required")
    target = target / target.sum()
    norm = max(1e-12, float(np.linalg.norm(predicted)))
    goal_norm = np.linalg.norm(target)
    cosine = float(target @ predicted / goal_norm / norm)
    cosine_gradient = (target / goal_norm / norm - cosine * predicted / norm ** 2) @ jacobian
    overlap = float(np.minimum(target, predicted).sum())
    overlap_gradient = (predicted < target).astype(float) @ jacobian
    ids = [SCENT_DIMENSIONS.index(name) for name in set(avoided)]
    return np.array([cosine, overlap, 1 - predicted[ids].sum()]), np.vstack([
        cosine_gradient, overlap_gradient, -jacobian[ids].sum(axis=0),
    ])


def linear_constraints(ingredients, brief, policy):
    """All bounds act on physical mass fractions, never normalized odor mass."""
    rows, rhs = [], []
    def upper(coefficients, limit):
        rows.append(np.asarray(coefficients, dtype=float))
        rhs.append(float(limit))
    for group, (low, high) in policy["bounds"].items():
        row = [float(item.pyramid == group) for item in ingredients]
        upper(row, high / 100)
        upper(-np.asarray(row), -low / 100)
    prices = np.asarray([item.price_per_kg for item in ingredients])
    cost_scale = max(1., brief.constraints.max_formula_cost_per_kg)
    upper(prices / cost_scale, brief.constraints.max_formula_cost_per_kg / cost_scale)
    for band, coefficients, saturated in (
        (policy["intensity_range"], [item.odor_impact * item.active_strength_percent / 100 / 2.5 for item in ingredients], True),
        (policy["diffusion_range"], [DIFFUSION[item.pyramid] for item in ingredients], False),
    ):
        if band is not None:
            upper(-np.asarray(coefficients), -band[0])
            if not saturated or band[1] < 1:
                upper(coefficients, band[1])
    return np.asarray(rows), np.asarray(rhs)


def project_seed(ingredients, seed, brief, policy, minimums):
    n = len(ingredients)
    cap = np.asarray([item.as_supplied_cap_percent() / 100 for item in ingredients])
    # Positive supports avoid a singular derivative at exactly zero dose. This
    # is 0.0001% concentrate, at the renderer's precision, not a dose cap.
    lower = np.asarray([max(1e-6, minimums.get(item.ingredient_id, 0.) / 100) for item in ingredients])
    target = np.asarray([seed.get(item.ingredient_id, 0.) / 100 for item in ingredients])
    if np.any(lower > cap):
        return None
    matrix, rhs = linear_constraints(ingredients, brief, policy)
    result = linprog(np.r_[np.zeros(n), np.ones(n)],
                     A_ub=np.vstack([np.c_[matrix, np.zeros_like(matrix)], np.c_[np.eye(n), -np.eye(n)], np.c_[-np.eye(n), -np.eye(n)]]),
                     b_ub=np.r_[rhs, target, -target], A_eq=[np.r_[np.ones(n), np.zeros(n)]], b_eq=[1.],
                     bounds=[*zip(lower, cap), *([(0., None)] * n)], method="highs")
    return (result.x[:n], lower, cap, matrix, rhs) if result.success else None


def optimize_dose_support(ingredients, brief, properties, seed, baseline_state, policy, minimums, max_iterations=100, *, proposal_signal_margin=.97):
    if not 0 <= proposal_signal_margin <= 1:
        raise ValueError("proposal signal margin must be finite and in [0, 1]")
    projected = project_seed(ingredients, seed, brief, policy, minimums)
    if projected is None:
        return None
    weights, lower, cap, matrix, rhs = projected
    model = DoseModel(ingredients, properties, brief.constraints.product_concentration_percent)
    simulator = TemporalMixtureSimulator()
    targets = simulator.targets_by_time(brief)
    time_weights = simulator.time_weights(brief)
    positive = time_weights > 0
    if any(target.sum() <= 0 for (target, _, _), weight in zip(targets, time_weights) if weight > 0):
        return None
    old_nominal, _ = agreement_components(brief.target_profile, baseline_state.nominal, baseline_state.nominal_jac, brief.avoided_dimensions)
    old_scores = np.asarray([min(agreement_components(target, vector, jac, avoided)[0]) if target.sum() > 0 else 0.
                             for (target, _, avoided), vector, jac in zip(targets, baseline_state.temporal, baseline_state.temporal_jac)])
    worst = min(old_scores[positive])
    cache_key, cache = None, None

    def evaluate(values):
        nonlocal cache_key, cache
        key = values.tobytes()
        if key == cache_key:
            return cache
        x, goal = values[:-1], values[-1]
        state = model.evaluate(x)
        nominal, nominal_jac = agreement_components(brief.target_profile, state.nominal, state.nominal_jac, brief.avoided_dimensions)
        scores, gradients = [], []
        components, component_jacs = [], []
        for (target, _, avoided), vector, jac in zip(targets, state.temporal, state.temporal_jac):
            if target.sum() > 0:
                component, cj = agreement_components(target, vector, jac, avoided)
                active = int(np.argmin(component))
                scores.append(component[active])
                gradients.append(cj[active])
            else:
                component, cj = np.ones(3), np.zeros((3, len(x)))
                scores.append(0.)
                gradients.append(np.zeros(len(x)))
            components.append(component)
            component_jacs.append(cj)
        constraints = list(nominal - goal)
        jacobians = [np.r_[row, -1.] for row in nominal_jac]
        constraints.append(float(time_weights @ scores) - goal)
        jacobians.append(np.r_[time_weights @ np.asarray(gradients), -1.])
        constraints.append(nominal[2] - old_nominal[2])
        jacobians.append(np.r_[nominal_jac[2], 0.])
        for index in np.flatnonzero(positive):
            phase = simulator._phase_for_time(TIMEPOINTS_MINUTES[index])
            floor = old_scores[index] if phase in brief.phase_target_profiles else worst
            constraints.extend(components[index] - floor)
            jacobians.extend(np.c_[component_jacs[index], np.zeros(3)])
            _, _, avoided = targets[index]
            ids = [SCENT_DIMENSIONS.index(name) for name in avoided]
            constraints.append(baseline_state.temporal[index, ids].sum() - state.temporal[index, ids].sum())
            jacobians.append(np.r_[-state.temporal_jac[index, ids].sum(axis=0), 0.])
            signal = max(1e-12, baseline_state.signal[index])
            # A proposal margin reduces final-draw rejections; the public
            # simulator's original 0.90 preservation check is not relaxed.
            constraints.append(state.signal[index] / signal - max(proposal_signal_margin, policy["signal_retention_floor"]))
            jacobians.append(np.r_[state.signal_jac[index] / signal, 0.])
        if policy["long_lasting"]:
            signal = max(1e-12, baseline_state.signal[-1])
            constraints.append(state.signal[-1] / signal - 1.)
            jacobians.append(np.r_[state.signal_jac[-1] / signal, 0.])
            retention = baseline_state.signal[-1] / max(1e-12, baseline_state.signal[0])
            constraints.append(state.signal[-1] - retention * state.signal[0])
            jacobians.append(np.r_[state.signal_jac[-1] - retention * state.signal_jac[0], 0.])
        cache_key, cache = key, (np.asarray(constraints), np.asarray(jacobians), min(min(nominal), float(time_weights @ scores)))
        return cache

    initial = np.r_[weights, 0.]
    initial[-1] = max(0., min(1., evaluate(initial)[2]) - 1e-8)
    with warnings.catch_warnings():
        # SLSQP deliberately clips transient trial points. Only this specific
        # library notice is handled; unexpected warnings still propagate and
        # the returned primal constraints are checked independently below.
        warnings.filterwarnings("ignore", message="^Values in x were outside bounds during a minimize step, clipping to bounds$",
                                category=RuntimeWarning, module=r"scipy\.optimize\._slsqp_py")
        result = minimize(lambda values: -values[-1], initial, jac=lambda values: np.r_[np.zeros(len(ingredients)), -1.],
                          method="SLSQP", bounds=[*zip(lower, cap), (0., 1.)], constraints=[
                              {"type": "eq", "fun": lambda values: values[:-1].sum() - 1., "jac": lambda values: np.r_[np.ones(len(ingredients)), 0.]},
                              {"type": "ineq", "fun": lambda values: rhs - matrix @ values[:-1], "jac": lambda values: np.c_[-matrix, np.zeros(len(rhs))]},
                              {"type": "ineq", "fun": lambda values: evaluate(values)[0], "jac": lambda values: evaluate(values)[1]},
                          ], options={"maxiter": max_iterations, "ftol": 1e-8})
    values = np.asarray(getattr(result, "x", None), dtype=float)
    if values.shape != (len(ingredients) + 1,) or not np.isfinite(values).all():
        return None
    x = values[:-1]
    if (np.any(x < lower - 1e-8) or np.any(x > cap + 1e-8)
            or abs(x.sum() - 1) > 1e-7 or np.min(rhs - matrix @ x) < -1e-7):
        return None
    checks, _, score = evaluate(values)
    if checks.min() < -1e-6:
        return None
    return {"weights_percent": {item.ingredient_id: float(value * 100) for item, value in zip(ingredients, x)},
            "proposal_score": float(score * 100), "solver_converged": bool(result.success),
            "solver_iterations": int(result.nit)}


def actual_dose_replacement_seeds(candidates, brief, properties, original, minimums, maximum=3):
    """Rank same-mass substitutions using the central nonlinear mixture.

    Every eligible material participates; no requested-note-only slice is
    used for ranking. Central predictions only seed the sampled optimizer.
    """
    engine = TemporalMixtureSimulator()
    items = list(candidates)
    lookup = {item.ingredient_id: i for i, item in enumerate(items)}
    if not items or not set(original).issubset(lookup):
        return []
    inputs = engine.prepare_response_inputs(items, properties)
    vectors = np.asarray([item.vector() for item in items])
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    concentration = brief.constraints.product_concentration_percent
    mole_gain = concentration * inputs.strengths / 100 / inputs.molecular_weights
    oav_gain = mole_gain * inputs.activities * inputs.vapor_pressures / ATMOSPHERIC_PRESSURE_PA * 1e6 / inputs.thresholds
    decay = np.power(.5, np.asarray(TIMEPOINTS_MINUTES)[None, :] / inputs.half_lives[:, None])
    coefficient = oav_gain[:, None] * decay
    structural = np.asarray([item.odor_impact * item.active_strength_percent / 100 for item in items])
    prices = np.asarray([item.price_per_kg for item in items])
    cap = np.asarray([item.as_supplied_cap_percent() / 100 for item in items])
    has_props = np.array([properties.get(item.ingredient_id) is not None for item in items])
    logp = np.array([properties[item.ingredient_id].xlogp if has_props[i] and properties[item.ingredient_id].xlogp is not None else np.nan
                     for i, item in enumerate(items)])
    ids = np.array([lookup[key] for key in original])
    weights = np.array(list(original.values())) / 100
    targets = engine.targets_by_time(brief)
    time_weights = engine.time_weights(brief)
    nominal_target = profile_vector(brief.target_profile)

    def score_batch(target, predicted, avoided):
        if target.sum() <= 0:
            return np.zeros(len(predicted))
        cosine = (predicted @ target) / np.maximum(np.linalg.norm(predicted, axis=1) * np.linalg.norm(target), 1e-12)
        overlap = np.minimum(target, predicted).sum(axis=1)
        banned = [SCENT_DIMENSIONS.index(name) for name in set(avoided)]
        return np.minimum(np.minimum(cosine, overlap), 1 - predicted[:, banned].sum(axis=1))

    options = []
    base_moles = max(0., 100 - concentration) / ETHANOL_MOLECULAR_WEIGHT
    original_cost = weights @ prices[ids]
    for donor, weight in zip(ids, weights):
        if items[donor].ingredient_id in minimums:
            continue
        alternatives = [i for i, item in enumerate(items) if i not in ids and item.pyramid == items[donor].pyramid
                        and cap[i] + 1e-8 >= weight
                        and original_cost + weight * (prices[i] - prices[donor]) <= brief.constraints.max_formula_cost_per_kg + 1e-8]
        if not alternatives:
            continue
        choices = np.asarray([donor, *alternatives])
        keep = ids != donor
        rest, rest_weights = ids[keep], weights[keep]
        denominator = np.maximum(1e-12, base_moles + rest_weights @ mole_gain[rest] + weight * mole_gain[choices])
        rest_oav = (coefficient[rest] * rest_weights[:, None]).T
        current_rest = engine._odor_response(np.maximum(rest_oav[None, :, :] / denominator[:, None, None], 1e-12)) * inputs.transports[rest]
        current_new = engine._odor_response(np.maximum(weight * coefficient[choices] / denominator[:, None], 1e-12)) * inputs.transports[choices, None]
        interaction = engine._interaction_matrix([SimpleNamespace(ingredient=items[i], properties=properties.get(items[i].ingredient_id)) for i in rest]) if len(rest) else np.empty((0, 0))
        chemical = np.exp(-np.abs(inputs.molecular_weights[choices, None] - inputs.molecular_weights[rest][None, :]) / 300.)
        polarity = np.exp(-np.abs(logp[choices, None] - logp[rest][None, :]) / 3.)
        polarity = np.where(np.isfinite(polarity), polarity, .5)
        chemical = np.where(has_props[choices, None] & has_props[rest][None, :], chemical * polarity, .5)
        cross = np.clip(.70 * np.clip(normalized[choices] @ normalized[rest].T, 0., 1.) + .30 * chemical, 0., 1.)
        suppressed_rest = current_rest / (1 + .2 * (current_rest @ interaction.T + current_new[:, :, None] * cross[:, None, :]))
        suppressed_new = current_new / (1 + .2 * (current_rest * cross[:, None, :]).sum(axis=2))
        signal = suppressed_rest.sum(axis=2) + suppressed_new
        temporal = suppressed_rest @ vectors[rest] + suppressed_new[:, :, None] * vectors[choices, None, :]
        temporal /= np.maximum(temporal.sum(axis=2, keepdims=True), 1e-12)
        nominal = (rest_weights * structural[rest]) @ vectors[rest] + weight * structural[choices, None] * vectors[choices]
        nominal /= np.maximum(nominal.sum(axis=1, keepdims=True), 1e-12)
        scores = np.array([score_batch(target, temporal[:, t], avoided) for t, (target, _, avoided) in enumerate(targets)]).T
        quality = np.minimum(score_batch(nominal_target, nominal, brief.avoided_dimensions), scores @ time_weights)
        # Rank only: penalize central regressions, leaving the exact existing
        # per-time and signal guards to the final scientific evaluation.
        phase_loss = np.maximum(0., scores[0] - scores) @ time_weights
        signal_loss = np.maximum(0., .97 - signal / np.maximum(signal[0], 1e-12)) @ time_weights
        rank = quality - phase_loss - .25 * signal_loss
        best = 1 + int(np.argmax(rank[1:]))
        if rank[best] <= rank[0] + 1e-6:
            continue
        proposal = {key: value for key, value in original.items() if key != items[donor].ingredient_id}
        proposal[items[choices[best]].ingredient_id] = float(weight * 100)
        options.append((float(rank[best] - rank[0]), proposal))
    return [proposal for _, proposal in sorted(options, key=lambda pair: -pair[0])[:maximum]]


def _bounded_dose_refinement_proposals(candidates, brief, properties, baseline_lines, policy, seed_weights, minimums):
    """Rank the entire eligible pool, then refine bounded active supports.

    This does not change the caller's material-count limit or global pool.
    Grow support incrementally, retaining every existing component. The step
    size limits work per proposal, not eligible materials or final recipe size.
    """
    limit = min(max(32, len(baseline_lines) + 16), brief.constraints.max_ingredients)
    if not baseline_lines or len(baseline_lines) > brief.constraints.max_ingredients:
        return
    eligible = [item for item in candidates if item.vector().sum() > 0 and item.as_supplied_cap_percent() > 0
                and policy["bounds"][item.pyramid][1] > 0]
    by_id = {item.ingredient_id: item for item in eligible}
    original = {line.ingredient_id: line.concentrate_percent for line in baseline_lines}
    if not set(original).issubset(by_id):
        return
    baseline_model = DoseModel([by_id[key] for key in original], properties, brief.constraints.product_concentration_percent)
    baseline_state = baseline_model.evaluate(np.asarray(list(original.values())) / 100)
    vectors = np.asarray([item.vector() for item in eligible])
    goals = [profile_vector(brief.target_profile), *(profile_vector(p) for p in brief.phase_target_profiles.values() if sum(p.values()) > 0)]
    enrichment = np.max(np.asarray([np.minimum(vectors, goal).sum(axis=1) for goal in goals]), axis=0)
    ranked = sorted(range(len(eligible)), key=lambda i: (-enrichment[i], eligible[i].price_per_kg, eligible[i].ingredient_id))
    replacements = actual_dose_replacement_seeds(eligible, brief, properties, original, minimums)
    seeds = [(original, False), (original, True), *((seed, False) for seed in replacements), *((seed, True) for seed in seed_weights)]
    supports = set()
    attempted = 0
    for seed, enrich in seeds:
        selected = [by_id[key] for key in seed if key in by_id and seed[key] > 0]
        if len(selected) > limit or not set(minimums).issubset({item.ingredient_id for item in selected}):
            continue
        # First preserve the original support for a small continuous step;
        # later supports can recruit high-fit materials from the entire pool.
        if enrich:
            selected_ids = {item.ingredient_id for item in selected}
            for index in ranked:
                if len(selected) >= limit:
                    break
                item = eligible[index]
                if item.ingredient_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item.ingredient_id)
        selected.sort(key=lambda item: item.ingredient_id)
        identity = tuple(item.ingredient_id for item in selected)
        if identity in supports:
            continue
        supports.add(identity)
        attempted += 1
        proposal = optimize_dose_support(selected, brief, properties, seed, baseline_state, policy, minimums)
        if proposal is not None:
            proposal.update(support_index=attempted, full_pool_considered=len(eligible), proposal_samples=32)
            yield proposal
        if attempted >= 6:
            break

def full_inferred_note_policy(policy):
    """Remove only internal inferred note bands, not user-entered quantities."""
    fixed = policy["explicit_percentages"]
    return {**policy, "bounds": {group: (fixed[group], fixed[group]) if group in fixed else (0., 100.)
                                 for group in policy["bounds"]}}


def coordinated_replacement_seeds(original, replacements):
    """Compose disjoint substitutions without changing mass or dropping IDs."""
    for size in (2, 3):
        for group in combinations(replacements, size):
            keys = set(original).union(*(set(seed) for seed in group))
            weights = {key: original.get(key, 0.) + sum(seed.get(key, 0.) - original.get(key, 0.) for seed in group) for key in keys}
            if min(weights.values(), default=0.) < -1e-8:
                continue
            yield {key: value for key, value in sorted(weights.items()) if value > 1e-8}


def dose_refinement_proposals(candidates, brief, properties, baseline_lines, policy, seed_weights, minimums, current_lines=None, current_score=None):
    # Keep the verified bounded stage and its early-pass path exactly first.
    yield from _bounded_dose_refinement_proposals(candidates, brief, properties, baseline_lines, policy, seed_weights, minimums)
    opened = full_inferred_note_policy(policy)
    if opened["bounds"] == policy["bounds"]:
        return
    bound = profile_upper_bound(candidates, brief.target_profile)
    if bound["upper_score"] is None or bound["upper_score"] + 1e-8 < brief.constraints.target_similarity:
        return
    eligible = [item for item in candidates if item.vector().sum() > 0 and opened["bounds"][item.pyramid][1] > 0]
    extra = optimize_full_pool(eligible, brief, pyramid_bounds=opened["bounds"], minimum_percent=minimums,
                               intensity_range=opened["intensity_range"], diffusion_range=opened["diffusion_range"])
    seeds = ([extra.weights_percent] if extra.weights_percent else []) + list(seed_weights)
    visited = set()
    for iteration in range(2):
        lines = list(current_lines()) if current_lines is not None else baseline_lines
        identity = tuple(sorted((line.ingredient_id, line.concentrate_percent) for line in lines))
        if identity in visited:
            break
        visited.add(identity)
        for proposal in _bounded_dose_refinement_proposals(candidates, brief, properties, lines, opened, seeds, minimums):
            yield {**proposal, "allocation_mode": "inferred_full_range", "open_round": iteration + 1}
    # This is after the complete old trajectory, so no previous winner is lost.
    # Spend extra work only near the requested goal; final scoring is unchanged.
    score = current_score() if current_score is not None else None
    if score is None or not brief.constraints.target_similarity - 5 <= score < brief.constraints.target_similarity:
        return
    lines = list(current_lines()) if current_lines is not None else baseline_lines
    if not lines or len(lines) > brief.constraints.max_ingredients:
        return
    by_id = {item.ingredient_id: item for item in eligible}
    original = {line.ingredient_id: line.concentrate_percent for line in lines}
    model = DoseModel([by_id[key] for key in original], properties, brief.constraints.product_concentration_percent)
    state = model.evaluate(np.asarray(list(original.values())) / 100)
    replacements = actual_dose_replacement_seeds(eligible, brief, properties, original, minimums)
    for index, seed in enumerate(coordinated_replacement_seeds(original, replacements), 1):
        selected = sorted((by_id[key] for key in seed), key=lambda item: item.ingredient_id)
        if len(selected) > brief.constraints.max_ingredients:
            continue
        proposal = optimize_dose_support(selected, brief, properties, seed, state, opened, minimums)
        if proposal is not None:
            yield {**proposal, "allocation_mode": "inferred_full_range", "replacement_mode": "coordinated",
                   "support_index": index, "full_pool_considered": len(eligible), "proposal_samples": 32}
    # The original .97 proposal buffer is stricter than the unchanged .90
    # public signal gate. Near the goal, try the actual boundary on at most
    # two supports; the independent final-draw gate still decides acceptance.
    lines = list(current_lines()) if current_lines is not None else lines
    original = {line.ingredient_id: line.concentrate_percent for line in lines}
    model = DoseModel([by_id[key] for key in original], properties, brief.constraints.product_concentration_percent)
    state = model.evaluate(np.asarray(list(original.values())) / 100)
    replacements = actual_dose_replacement_seeds(eligible, brief, properties, original, minimums, maximum=1)
    for index, seed in enumerate([original, *replacements], 1):
        selected = sorted((by_id[key] for key in seed), key=lambda item: item.ingredient_id)
        if len(selected) > brief.constraints.max_ingredients:
            continue
        proposal = optimize_dose_support(selected, brief, properties, seed, state, opened, minimums,
                                         proposal_signal_margin=policy["signal_retention_floor"])
        if proposal is not None:
            yield {**proposal, "allocation_mode": "inferred_full_range", "replacement_mode": "final_gate_margin_polish",
                   "support_index": index, "full_pool_considered": len(eligible), "proposal_samples": 32}
