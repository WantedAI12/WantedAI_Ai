"""CPU-only, source-bound neural distillation of lotion transport, not odor truth.

Dimensionless variables preserve dose/phase-volume dependence. A learned
correction to an integrated-rate physical baseline is projected onto the mass
simplex. Torch is used only by the offline trainer, never by this runtime.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .lotion_transport import bidirectional_step

VERSION = 'lotion-transport-neural-v58/1'
FEATURE_NAMES = ('evaporation_time', 'uptake_time', 'hydrolysis_time',
                 'return_time', 'ventilation_time', 'drying_time',
                 'evaporable_water_capacity_fraction', 'lipid_capacity_fraction')
STATE_NAMES = ('remaining', 'headspace', 'skin_sink', 'ventilated', 'degraded')
MASS_FIELDS = ('remaining_mg_cm2', 'headspace_mg_cm2', 'skin_sink_mg_cm2',
               'ventilated_mg_cm2', 'degraded_parent_equivalent_mg_cm2')
EPS = 1e-14
# Selected training domain, not a declaration that these conditions are safe.
LOWER = np.array([1e-15, 0., 0., 1e-6, 1e-6, 0., 0., 0.])
UPPER = np.array([1e7, 1e7, 1e3, 1e6, 1e6, 100., .999, 1.])


def physical_baseline(raw):
    """Constant-generator approximation using exact mean inverse capacity."""
    x = np.asarray(raw, dtype=np.float64)
    e, u, h, r, v, k, w, lipid = x.T
    d = 1. - w
    integral = np.ones(len(x))
    small = k < 1e-4
    integral[small] += w[small]*k[small]/2 + (w[small]**2-w[small]/2)*k[small]**2/3
    z = ~small
    integral[z] = (k[z] + np.log1p(-w[z]*(-np.expm1(-k[z])))) / (k[z]*d[z])
    uptake = u * integral
    reaction = h * np.maximum(0., 1. - lipid * integral)
    sink = uptake + reaction
    f, a, loss, exhaust = bidirectional_step(np.ones(len(x)), np.zeros(len(x)),
        e*integral, sink, r, v, 1.)
    share = np.divide(uptake, sink, out=np.zeros_like(sink), where=sink > 0)
    result = np.column_stack((f, a, loss*share, exhaust, loss*(1-share)))
    result = np.maximum(result, 0.)
    return result / result.sum(axis=1, keepdims=True)


def features(raw):
    x = np.asarray(raw, dtype=np.float64)
    return np.column_stack((np.log10(np.maximum(x[:, :5], 1e-15)),
                            np.log1p(x[:, 5]), x[:, 6:]))


def state_mask(raw):
    mask = np.ones((len(raw), 5), dtype=bool)
    mask[:, 2] = raw[:, 1] > 0
    mask[:, 4] = raw[:, 2] > 0
    return mask


def phase_volumes(request):
    """Same additive phase-volume convention as the authoritative simulator."""
    context = request.application_context
    base = {c.name.casefold(): c for c in context.base_components}
    dose = context.application_mass_mg_cm2*(1-context.fragrance_concentration_percent/100)
    water = fixed = lipid = 0.
    for row in request.phase_components:
        component = base[row.name.casefold()]
        if row.portions is not None:
            for part in row.portions:
                amount = dose*component.mass_percent*part.mass_percent / (1e7*part.density_g_ml)
                if part.compartment == 'lipid':
                    lipid += amount
                elif part.compartment == 'water':
                    water += amount
                else:
                    fixed += amount
        else:
            amount = dose*component.mass_percent/(1e5*row.density_g_ml)
            if row.phase == 'lipid':
                lipid += amount
            elif component.role == 'water':
                water += amount
            else:
                fixed += amount
    return fixed, water, lipid


def request_features(request):
    fixed, water, lipid = phase_volumes(request)
    p = request.materials
    capacity = fixed + water + lipid*np.array([m.lipid_water_partition for m in p])
    e = np.array([m.gas_transfer_cm_min*m.air_water_partition for m in p])/capacity
    u = np.array([m.skin_permeability_cm_min for m in p])/capacity
    h = np.array([m.aqueous_hydrolysis_per_min for m in p])
    r = np.array([m.gas_transfer_cm_min for m in p])/request.headspace_height_cm
    v = np.full(len(p), request.air_exchange_per_min)
    k = np.full(len(p), request.water_loss_per_min)
    w = water*(1-request.retained_water_fraction)/capacity
    lipid_fraction = lipid*np.array([m.lipid_water_partition for m in p])/capacity
    rates = np.column_stack((e, u, h, r, v, k))
    return np.concatenate([np.column_stack((rates*t, w, lipid_fraction)) for t in request.times_minutes[1:]])


class LotionReleaseSurrogate:
    def __init__(self, path, *, sha256):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('lotion surrogate manifest hash or size mismatch')
        self.sha256 = sha256
        self.manifest = json.loads(raw)
        m = self.manifest
        if (m.get('schema') != VERSION or m.get('label_kind') != 'synthetic_transport'
                or m.get('measured_lotion_observations') != 0 or not m.get('training_executed')
                or not m.get('accepted_for_local_surrogate_inference')):
            raise ValueError('lotion checkpoint is not an accepted synthetic release model')
        if m.get('feature_names') != list(FEATURE_NAMES):
            raise ValueError('lotion checkpoint feature contract mismatch')
        item = m['weights']
        self.weights_path = (self.path.parent/item['path']).resolve()
        if not self.weights_path.is_relative_to(self.path.parent):
            raise ValueError('lotion weights escape checkpoint directory')
        raw = self.weights_path.read_bytes()
        if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('lotion weights hash or size mismatch')
        with np.load(self.weights_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        self.mean, self.scale = arrays['mean'], arrays['scale']
        dims = m['architecture']['dimensions']
        if (not isinstance(dims, list) or len(dims) != 5 or dims[0] != 8 or dims[-1] != 5
                or any(not isinstance(n, int) or not 1 <= n <= 512 for n in dims)):
            raise ValueError('invalid lotion network architecture')
        if self.mean.shape != (8,) or self.scale.shape != (8,) or np.any(self.scale <= 0):
            raise ValueError('invalid lotion feature normalization')
        self.layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            weight, bias = arrays[f'weight_{i}'], arrays[f'bias_{i}']
            if weight.shape != (a, b) or bias.shape != (b,):
                raise ValueError('invalid lotion network weights')
            self.layers.append((weight, bias))
        if any(not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('nonfinite lotion checkpoint')
        for x in arrays.values():
            x.setflags(write=False)
        self._metadata = self._stat()

    def _stat(self):
        return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (self.path, self.weights_path))

    def assert_current(self):
        if self._stat() != self._metadata:
            raise ValueError('lotion checkpoint changed after loading')

    def contract(self):
        self.assert_current()
        return {'schema': VERSION, 'checkpoint_sha256': self.sha256,
            'weights_sha256': self.manifest['weights']['sha256'],
            'training_executed': True, 'label_kind': 'synthetic_transport',
            'parent_v54_sha256': self.manifest['parent_models']['atlas']['sha256'],
            'inference_runtime': 'numpy_cpu', 'external_api_calls': 0,
            'human_similarity_percent': None, 'measured_lotion_observations': 0,
            'recipe_acceptance_score_modified': False}

    def predict(self, raw):
        self.assert_current()
        raw = np.asarray(raw, dtype=np.float64)
        if (raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all()
                or np.any(raw < LOWER) or np.any(raw > UPPER)
                or np.any(raw[:, 6]+raw[:, 7] > 1.+1e-10)):
            raise ValueError('outside lotion surrogate coefficient domain')
        baseline = physical_baseline(raw)
        x = ((features(raw)-self.mean)/self.scale).astype(np.float32)
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if i < len(self.layers)-1:
                x = np.maximum(x, 0.)
        logits = np.log(np.maximum(baseline, EPS)) + x.astype(np.float64)
        mask = state_mask(raw)
        logits[~mask] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)
        y = np.exp(logits)
        return y/y.sum(axis=1, keepdims=True)


@lru_cache(maxsize=2)
def _load(path, sha256, stat):
    return LotionReleaseSurrogate(path, sha256=sha256)


def configured_lotion_surrogate():
    from .local_runtime import local_profile
    profile = local_profile()
    if profile is not None and 'unified_product' in profile:
        from .unified_transport import configured_unified_transport, UnifiedLotionAdapter
        return UnifiedLotionAdapter(configured_unified_transport())
    if profile is None or 'lotion_release' not in profile:
        return None
    path, sha = profile['lotion_release']
    p = Path(path)
    model = _load(path, sha, (p.stat().st_size, p.stat().st_mtime_ns))
    model.assert_current()
    for role, binding in model.manifest['parent_models'].items():
        if role not in profile or profile[role][1] != binding['sha256']:
            raise ValueError('lotion surrogate parent binding does not match local V54')
    return model


def predict_release(request, catalog, *, model=None):
    """New trained output, separate from authoritative physics/acceptance scores."""
    model = model or configured_lotion_surrogate()
    if model is None:
        return None
    contract = model.contract()
    if request.transport_mode != 'bidirectional_air':
        return {**contract, 'status': 'abstained', 'reason': 'bidirectional_air_required'}
    context = request.application_context
    unified = hasattr(model, 'predict_trajectory')
    if not unified and (context.temperature_c != 25. or request.times_minutes[-1] > 480.
            or request.times_minutes[1] < 1.
            or not .05 <= request.retained_water_fraction <= .3
            or not .5 <= request.headspace_height_cm <= 2.
            or not .3 <= request.air_exchange_per_min <= 3.2
            or not (request.water_loss_per_min == 0. or .0075 <= request.water_loss_per_min <= .03)):
        return {**contract, 'status': 'abstained', 'reason': 'outside_trained_application_conditions'}
    try:
        if unified:
            y = model.predict_trajectory(request)
        else:
            raw = request_features(request)
            y = model.predict(raw).reshape(len(request.times_minutes)-1, len(request.materials), 5)
    except ValueError as exc:
        if str(exc) not in ('outside lotion surrogate coefficient domain', 'outside unified transition coefficient domain'):
            raise
        return {**contract, 'status': 'abstained', 'reason': str(exc)}
    known = {i.ingredient_id: i for i in catalog.ingredients}
    if any(m.ingredient_id not in known or known[m.ingredient_id].blocked
           or abs(known[m.ingredient_id].active_strength_percent-100) > 1e-8 for m in request.materials):
        raise ValueError('surrogate requires known admissible undiluted materials')
    initial = request.application_context.application_mass_mg_cm2 * request.application_context.fragrance_concentration_percent/100
    mass = initial*np.array([m.concentrate_percent/100 for m in request.materials])
    parent = np.array([m.initial_parent_fraction for m in request.materials])
    rows = y*(mass*parent)[None, :, None]
    rows[:, :, 4] += (mass*(1-parent))[None, :]
    start = np.zeros((1, len(mass), 5))
    start[0, :, 0], start[0, :, 4] = mass*parent, mass*(1-parent)
    rows = np.concatenate((start, rows))
    from .models import SCENT_DIMENSIONS
    profiles = np.array([known[m.ingredient_id].vector() for m in request.materials])
    use_oav = request.profile_weighting != 'air_mass' and all(m.odor_threshold_mg_m3 for m in request.materials)
    thresholds = np.array([m.odor_threshold_mg_m3 or 1. for m in request.materials])
    temporal = []
    for time, values in zip(request.times_minutes, rows):
        air = values[:, 1]/request.headspace_height_cm*1e6
        weights = air/thresholds if use_oav else air
        total = weights.sum()
        profile = dict(zip(SCENT_DIMENSIONS, (weights@profiles/total).tolist())) if total > 0 else None
        temporal.append({'minutes': time, 'total_air_concentration_mg_m3': float(air.sum()),
            'total_odor_activity_proxy': float(total) if use_oav else None,
            'scent_profile': profile,
            'profile_basis': 'linear_odor_activity_proxy' if use_oav else 'air_mass_weighted_catalog_proxy',
            'transport_basis': 'trained_unified_transition_operator' if unified else 'trained_neural_surrogate',
            'materials': [{'ingredient_id': m.ingredient_id, 'initial_mg_cm2': float(mass[i]),
                **{key: float(values[i, j]) for j, key in enumerate(MASS_FIELDS)},
                'air_concentration_mg_m3': float(air[i]),
                'odor_activity_proxy': float(air[i]/thresholds[i]) if m.odor_threshold_mg_m3 else None}
                for i, m in enumerate(request.materials)]})
    # Independent timepoint predictions can violate trajectory monotonicity.
    # Report this; do not quietly clip a curve into apparent accuracy.
    cumulative = rows[:, :, 2:]
    monotone = bool(np.all(np.diff(cumulative, axis=0) >= -np.maximum(mass[None, :, None]*1e-5, 1e-15)))
    return {**contract, 'status': 'research_prediction' if monotone else 'research_prediction_flagged', 'temporal_profile': temporal,
        'parameter_context_id': request.parameter_context_id,
        'diagnostics': {'mass_balance_max_abs_error_mg_cm2': float(np.max(np.abs(rows.sum(axis=2)-mass))),
            'nonnegative': bool(np.all(rows >= 0)), 'cumulative_sinks_monotone': monotone},
        'limitations': ['Synthetic teacher distillation, not empirical lotion calibration.',
            'No micellar kinetics, new chemical interactions or temperature extrapolation learned.',
            'Catalog scent weighting is a proxy; this checkpoint predicts physical transport.']}


def attach_release_prediction(result, request, catalog, *, provider=None, shape_predictor=None):
    """Keep authoritative solver outputs and scores; add the new trained path."""
    learned = predict_release(request, catalog)
    if learned is None:
        return result
    if provider is not None and 'temporal_profile' in learned:
        from .lotion_perception import attach_lotion_perception
        learned = attach_lotion_perception(learned, catalog, provider, _shape_predictor=shape_predictor)
    if 'temporal_profile' in learned:
        differences = []
        for exact, neural in zip(result['temporal_profile'], learned['temporal_profile']):
            if exact['minutes'] != neural['minutes']:
                raise ValueError('neural and exact lotion times disagree')
            for a, b in zip(exact['materials'], neural['materials']):
                if a['ingredient_id'] != b['ingredient_id']:
                    raise ValueError('neural and exact lotion material identities disagree')
                initial = a['initial_mg_cm2']
                differences.extend(abs(a[key]-b[key])/initial for key in MASS_FIELDS)
        learned['same_request_solver_comparison'] = {
            'state_fraction_mae': float(np.mean(differences)),
            'maximum_state_fraction_error': float(np.max(differences)),
            'scope': 'comparison_to_existing_uncalibrated_numerical_solver_not_measured_accuracy'}
    return {**result, 'learned_release': learned}
