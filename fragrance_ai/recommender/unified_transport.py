"""V60 shared, absorbing-state neural transport operator for product contexts.

One CPU checkpoint learns *transitions*, not independent timepoint masses.
Products share the same dimensionless conservation law. Carrier/phase capacity,
transfer coefficients and rinse events are explicit conditions, not hidden
product-name multipliers. Training labels are numerical, never human labels.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .lotion_transport import bidirectional_step

VERSION = 'unified-product-transport-v60/1'
PRODUCTS = ('perfume', 'body_lotion', 'body_wash')
FEATURE_NAMES = ('evaporation_time', 'uptake_time', 'hydrolysis_time',
                 'return_time', 'ventilation_time', 'capacity_decay_time',
                 'evaporating_capacity_fraction', 'nonreactive_capacity_fraction')
STATE_NAMES = ('remaining', 'headspace', 'skin_sink', 'ventilated', 'degraded', 'washed_off')
LOWER = np.zeros(8)
UPPER = np.array([1e8, 1e8, 1e4, 1e7, 1e7, 200., .999, 1.])
EPS = 1e-14


def validate_raw(raw):
    x = np.asarray(raw, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all()
            or np.any(x < LOWER) or np.any(x > UPPER)
            or np.any(x[:, 6]+x[:, 7] > 1.+1e-12)):
        raise ValueError('outside unified transition coefficient domain')
    return x


def features(raw):
    x = np.asarray(raw, float)
    return np.c_[np.log10(np.maximum(x[:, :5], EPS)), np.log1p(x[:, 5]), x[:, 6:]]


def correction_gate(raw):
    # No neural perturbation at constant capacity, including exact zero decay.
    return -np.expm1(-raw[:, 5])*raw[:, 6]


def _stochastic(values):
    """Repair roundoff only; a broken kernel cannot be disguised as conserved."""
    if (not np.isfinite(values).all() or np.min(values, initial=0.) < -1e-9
            or np.max(np.abs(values.sum(axis=-1)-1), initial=0.) > 1e-7):
        raise ValueError('transition kernel violates conservation')
    values = np.maximum(values, 0.)
    return values/values.sum(axis=-1, keepdims=True)


def constant_kernel(evaporation, uptake, reaction, return_rate, ventilation, dt=1.):
    """Rows are initial film/air; columns are film, air and three sinks."""
    e, u, h, r, v = np.broadcast_arrays(*[np.asarray(a, float) for a in
        (evaporation, uptake, reaction, return_rate, ventilation)])
    sink = u+h
    share = np.divide(u, sink, out=np.zeros_like(sink), where=sink > 0)
    rows = []
    for initial in (0, 1):
        f, a = np.full_like(e, 1.-initial), np.full_like(e, float(initial))
        f, a, loss, vent = bidirectional_step(f, a, e, sink, r, v, dt)
        rows.append(np.stack((f, a, loss*share, vent, loss*(1-share)), axis=-1))
    return _stochastic(np.stack(rows, axis=-2))


def baseline_kernel(raw):
    """Analytic constant generator at the exact mean inverse phase capacity."""
    e, u, h, r, v, k, w, lipid = np.asarray(raw, float).T
    fixed = 1.-w
    inverse_capacity = np.divide(np.log1p(fixed*np.expm1(k)), fixed*k,
                                 out=np.ones_like(k), where=k > 1e-10)
    return constant_kernel(e*inverse_capacity, u*inverse_capacity,
                           h*np.maximum(0., 1.-lipid*inverse_capacity), r, v)


def kernel_mask(raw):
    e, u, h, r, v = raw[:, :5].T
    # A zero physical path stays zero after neural correction.
    reaction = (h > 0) & (raw[:, 7] < 1)
    f = np.stack((np.ones(len(raw), bool), e > 0, u > 0, (e > 0)&(v > 0), reaction), axis=1)
    a = np.stack((r > 0, np.ones(len(raw), bool), (r > 0)&(u > 0), v > 0,
                  (r > 0)&reaction), axis=1)
    return np.stack((f, a), axis=1)


def interval_features(rates, fractions, start, end):
    """Rates/fractions refer to the start of a stage, not a plotting timepoint."""
    rates, fractions = np.asarray(rates, float), np.asarray(fractions, float)
    dt = float(end-start)
    if dt <= 0:
        raise ValueError('positive transition interval required')
    w, lipid = fractions.T
    evaporating = w*np.exp(-rates[:, 5]*start)
    # Keep the supplied nonreactive fraction an explicit part of capacity.
    # At w+lipid==1, computing 1-w can round BELOW lipid and produce an
    # impossible lipid/capacity > 1 after drying. No fraction is clipped.
    fixed = lipid + (1.-(w+lipid))
    capacity = fixed+evaporating
    current = rates.copy()
    current[:, :2] /= capacity[:, None]
    return np.c_[current*dt, evaporating/capacity, lipid/capacity]


def advance(state, kernel):
    flow = np.einsum('ni,nij->nj', state[:, :2], kernel)
    result = state.copy()
    result[:, :2] = flow[:, :2]
    result[:, 2:5] += flow[:, 2:]
    return result


def _domain_mesh(rates, fractions, edges):
    """Refine time, not the checkpoint's dimensionless applicability domain.

    Capacity is monotone decreasing. Rates scaled by capacity at the RIGHT
    endpoint therefore bound every subinterval's initial dimensionless rates.
    This refinement is independent of plotting times and leaves in-domain
    canonical cells unchanged. It is a domain guard, not an error estimator.
    """
    refined = [float(edges[0])]
    for left, right in zip(edges[:-1], edges[1:]):
        fixed = fractions[:, 1]+(1.-fractions.sum(axis=1))
        capacity = fixed+fractions[:, 0]*np.exp(-rates[:, 5]*right)
        bound = rates.copy()
        bound[:, :2] /= capacity[:, None]
        ratio = float(np.max(bound / UPPER[:6] * (right-left), initial=0.))
        if not np.isfinite(ratio) or ratio > 2048:
            raise ValueError('unified transition domain refinement work budget exceeded')
        count = max(1, int(np.ceil(ratio)))
        if len(refined)-1+count > 2048:
            raise ValueError('unified transition domain refinement work budget exceeded')
        refined.extend(np.linspace(left, right, count+1)[1:].tolist())
    return np.asarray(refined)


def trajectory(rates, fractions, times, *, operator=baseline_kernel, steps=32, initial=None, duration=None,
               work_budget=None):
    """Conservative rollout on a canonical mesh independent of display times.

    Partial observations are evaluated from the mesh state without changing it.
    Thus adding a requested time does not change already requested predictions.
    An optional request-local remaining budget is shared across product stages.
    """
    rates, fractions, times = np.asarray(rates, float), np.asarray(fractions, float), np.asarray(times, float)
    end = float(duration if duration is not None else times[-1])
    if (rates.ndim != 2 or rates.shape[1] != 6 or fractions.shape != (len(rates), 2)
            or not np.isfinite(rates).all() or not np.isfinite(fractions).all()
            or np.any(rates < 0) or np.any(fractions < 0) or np.any(fractions.sum(axis=1) > 1.)
            or np.any(fractions[:, 0] > .999) or not np.isfinite(times).all()
            or times.ndim != 1 or not len(times) or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.isfinite(end) or end <= 0 or times[-1] > end
            or type(steps) is not int or not 1 <= steps <= 2048):
        raise ValueError('invalid unified transport trajectory')
    state = np.zeros((len(rates), 6)) if initial is None else np.asarray(initial, float).copy()
    if initial is None:
        state[:, 0] = 1.
    if state.shape != (len(rates), 6) or not np.isfinite(state).all() or np.any(state < 0):
        raise ValueError('finite nonnegative six-state initial mass required')
    edges = _domain_mesh(rates, fractions, np.linspace(0., 1., steps+1)**2*end)
    work = len(rates)*(len(edges)-1+len(times))
    if work > 2_000_000:
        raise ValueError('unified transition domain refinement work budget exceeded')
    if work_budget is not None:
        remaining = work_budget.get('remaining_material_transitions')
        if type(remaining) is not int or not 0 <= remaining <= 2_000_000:
            raise ValueError('invalid unified request work budget')
        if work > remaining:
            raise ValueError('unified request multi-stage refinement work budget exceeded')
        work_budget['remaining_material_transitions'] -= work
    result, index = [], 0
    if times[0] == 0:
        result.append(state.copy())
        index = 1
    for left, right in zip(edges[:-1], edges[1:]):
        # Batch display and mesh transitions for this interval in one forward.
        stops = []
        while index < len(times) and times[index] <= right:
            stops.append(float(times[index]))
            index += 1
        include_mesh = not stops or stops[-1] != right
        all_stops = [*stops, right] if include_mesh else stops
        raw = np.concatenate([interval_features(rates, fractions, left, stop) for stop in all_stops])
        kernels = operator(raw).reshape(len(all_stops), len(rates), 2, 5)
        for kernel in kernels[:len(stops)]:
            result.append(advance(state, kernel))
        state = advance(state, kernels[-1])
    return np.stack(result), state


class UnifiedTransportModel:
    def __init__(self, path, *, sha256):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('unified model manifest hash mismatch')
        self.sha256, self.manifest = sha256, json.loads(raw)
        m = self.manifest
        if (m.get('schema') != VERSION or m.get('feature_names') != list(FEATURE_NAMES)
                or m.get('label_kind') != 'synthetic_transition_operators'
                or m.get('training_executed') is not True or m.get('accepted_for_local_inference') is not True
                or m.get('products') != list(PRODUCTS) or m.get('state_names') != list(STATE_NAMES)):
            raise ValueError('unaccepted unified checkpoint or incompatible contract')
        self.weights_path = (self.path.parent/m['weights']['path']).resolve()
        if not self.weights_path.is_relative_to(self.path.parent):
            raise ValueError('unified weights escape checkpoint directory')
        raw = self.weights_path.read_bytes()
        if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != m['weights']['sha256']:
            raise ValueError('unified weights hash mismatch')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        dims = m['architecture']['dimensions']
        if (not isinstance(dims, list) or len(dims) != 5 or dims[0] != 8 or dims[-1] != 10
                or any(type(n) is not int or not 1 <= n <= 256 for n in dims)):
            raise ValueError('invalid unified network architecture')
        self.mean, self.scale = arrays['mean'], arrays['scale']
        if self.mean.shape != (8,) or self.scale.shape != (8,) or np.any(self.scale <= 0):
            raise ValueError('invalid unified normalization')
        self.layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            weight, bias = arrays[f'weight_{i}'], arrays[f'bias_{i}']
            if weight.shape != (a, b) or bias.shape != (b,):
                raise ValueError('invalid unified weights')
            self.layers.append((weight, bias))
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('nonfinite unified checkpoint')
        self.blend = m['validation_selected_blend']
        if type(self.blend) not in (float, int) or not 0 < self.blend <= 1:
            raise ValueError('unified learned operator requires a validated positive blend')
        for a in arrays.values():
            a.setflags(write=False)
        self._metadata = self._stat()

    def _stat(self):
        return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (self.path, self.weights_path))

    def assert_current(self):
        if self._metadata != self._stat():
            raise ValueError('unified checkpoint changed; reload runtime')

    def kernel(self, raw):
        self.assert_current()
        raw = validate_raw(raw)
        base = baseline_kernel(raw)
        outputs = []
        for offset in range(0, len(raw), 4096):
            q = raw[offset:offset+4096]
            b = base[offset:offset+4096]
            x = ((features(q)-self.mean)/self.scale).astype(np.float32)
            for i, (weight, bias) in enumerate(self.layers):
                x = x@weight+bias
                if i < len(self.layers)-1:
                    x = np.maximum(x, 0.)
            correction = x.reshape(-1, 2, 5).astype(float)*correction_gate(q)[:, None, None]
            logits = np.log(np.maximum(b, EPS))+correction
            logits[~kernel_mask(q)] = -np.inf
            logits -= logits.max(axis=-1, keepdims=True)
            learned = np.exp(logits)
            learned /= learned.sum(axis=-1, keepdims=True)
            mixed = (1.-self.blend)*b+self.blend*learned
            constant = correction_gate(q) == 0
            mixed[constant] = b[constant]
            outputs.append(mixed)
        return np.concatenate(outputs) if outputs else base

    def contract(self):
        self.assert_current()
        return {'schema': VERSION, 'checkpoint_sha256': self.sha256,
            'weights_sha256': self.manifest['weights']['sha256'], 'products': list(PRODUCTS),
            'shared_transport_checkpoint': True, 'inference_runtime': 'numpy_cpu',
            'training_executed': True, 'label_kind': 'synthetic_transition_operators',
            'parent_v54_sha256': self.manifest['parent_models']['atlas']['sha256'],
            'trained_transition_blend': self.blend, 'external_api_calls': 0,
            'trajectory_constraints': ['mass_conservation', 'nonnegative', 'absorbing_cumulative_sinks'],
            'rollout_precision_policy': 'learned_proposal_with_two_half_step_analytic_defect_check',
            'transition_domain_policy': 'output_independent_time_refinement_no_coefficient_clipping',
            'human_similarity_percent': None, 'measured_product_release_observations': 0,
            'recipe_acceptance_score_modified': False}

    def stable_kernel(self, raw):
        """Check learned proposals against a two-half-step physical reference.

        For a second-order constant-generator rule, the fine/coarse difference
        estimates three times the fine local truncation error. Retain a neural
        proposal only inside that defect radius; otherwise use the fine rule.
        This is an embedded numerical guard, not an empirical accuracy bound.
        """
        raw = validate_raw(raw)
        learned, coarse = self.kernel(raw), baseline_kernel(raw)
        first = baseline_kernel(interval_features(raw[:, :6], raw[:, 6:], 0., .5))
        second = baseline_kernel(interval_features(raw[:, :6], raw[:, 6:], .5, 1.))
        fine = np.einsum('nbi,nij->nbj', first[:, :, :2], second)
        fine[:, :, 2:] += first[:, :, 2:]
        defect = np.max(np.abs(fine-coarse), axis=(1, 2))/3.
        use_learned = np.max(np.abs(learned-fine), axis=(1, 2)) <= defect
        result = np.where(use_learned[:, None, None], learned, fine)
        result = _stochastic(result)
        result[correction_gate(raw) == 0] = coarse[correction_gate(raw) == 0]
        return result


@lru_cache(maxsize=2)
def _load(path, digest, size, mtime):
    return UnifiedTransportModel(path, sha256=digest)


def configured_unified_transport():
    from .formulation_core import configured_formulation_core
    from .formulation_views import shared_views
    core = configured_formulation_core()
    if core is not None:
        return shared_views(core)[2]
    from .local_runtime import local_profile
    profile = local_profile()
    pair = profile.get('unified_product') if profile else None
    if pair is None:
        return None
    path, digest = pair
    stat = Path(path).stat()
    model = _load(path, digest, stat.st_size, stat.st_mtime_ns)
    model.assert_current()
    for role, binding in model.manifest['parent_models'].items():
        if role not in profile or profile[role][1] != binding['sha256']:
            raise ValueError('unified model parent does not match pinned runtime: '+role)
    return model


class UnifiedLotionAdapter:
    """Legacy lotion release interface, now using the shared trajectory model."""
    def __init__(self, model):
        self.model, self.manifest, self.sha256 = model, model.manifest, model.sha256

    def assert_current(self):
        self.model.assert_current()

    def contract(self):
        return {**self.model.contract(), 'adapter_product': 'body_lotion'}

    def predict_trajectory(self, request):
        from .lotion_surrogate import request_features
        raw = request_features(request)[:len(request.materials)]
        rates = raw[:, :6]/request.times_minutes[1]
        rows, _ = trajectory(rates, raw[:, 6:], request.times_minutes, operator=self.model.stable_kernel)
        return rows[1:, :, :5]
