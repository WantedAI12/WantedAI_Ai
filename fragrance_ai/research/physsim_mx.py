"""Isolated V54-MX hypotheses, not a replacement for the shipped predictor.

Atlas outputs are frozen semantic features, not gas intensity measurements.
The supervised head below has a nominal-stock domain. The separate release
solver emits mol/m3 and must NOT silently feed that head in different units.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm

VERSION = 'v54-mx-research/v1'
MODES = ('semantic', 'concentration', 'interaction')
CARRIERS = ('nt', 'pg', 'dep', 'paraffin oil', 'mineral oil', '90% ethanol', '99% ethanol', 'other')
DOMAIN = 'nominal_stock_dilution_not_measured_gas_concentration'


def mx_features(profiles, identities, dilutions, carriers, weights=None, *, mode='interaction', reference=.01):
    """Merge chemical identities BEFORE any nonlinear, order-free pooling.

    Dilutions retain the source's nominal scale (0..1), not an invented ppm
    conversion. Stock-volume fractions weight their dilution; total nominal
    concentration is retained. Atlas high/low references remain categorical
    feature blocks; they are never claimed to be measured stock dilutions.
    """
    p = np.asarray(profiles, float)
    d = np.asarray(dilutions, float)
    n = len(identities)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if (mode not in MODES or not math.isfinite(reference) or reference <= 0
            or p.ndim != 2 or p.shape[0] != n or p.shape[1] < 1 or n == 0
            or len(carriers) != n or d.shape != (n,) or w.shape != (n,)
            or any(not isinstance(k, str) or not k for k in identities)
            or any(not isinstance(k, str) or not k for k in carriers)
            or not all(np.isfinite(v).all() for v in (p, d, w))
            or np.any(p < 0) or np.any(d < 0) or np.any(d > 1)
            or np.any(w < 0) or w.max() <= 0):
        raise ValueError('invalid MX semantic/nominal-stock input')
    w = w / w.max(); w /= w.sum()
    grouped, values = defaultdict(lambda: [0., 0.]), {}
    carrier_weights = np.zeros(len(CARRIERS))
    for key, vector, dilution, carrier, weight in zip(identities, p, d, carriers, w):
        if weight == 0:
            continue
        carrier_weights[CARRIERS.index(carrier) if carrier in CARRIERS else -1] += weight
        if dilution == 0:
            continue
        if key in values and not np.array_equal(values[key], vector):
            raise ValueError('same molecular identity has conflicting semantic features')
        grouped[key][0] += weight
        grouped[key][1] += weight * dilution
        values[key] = vector
    width = p.shape[1]
    output_width = width if mode == 'semantic' else 2*width+2+len(CARRIERS)
    if mode == 'interaction':
        output_width += 2*width+1
    if not grouped:
        return np.zeros(output_width)
    order = sorted(grouped)
    p = np.asarray([values[k] for k in order])
    aliquot = np.asarray([grouped[k][0] for k in order]); aliquot /= aliquot.sum()
    u = np.asarray([grouped[k][1] for k in order])
    mean = aliquot @ p
    if mode == 'semantic':
        return mean
    total = float(u.sum())
    # Both amount and identity-specific dose are retained, not only a simplex.
    blocks = [mean, np.log1p(u/reference) @ p,
              np.array([total, math.log1p(total/reference)]), carrier_weights]
    if mode == 'interaction':
        q = u / (reference + total)
        pooled = q @ p
        # Bounded saturation and second-order cross-component associations.
        # These are statistical features, NOT fitted human receptor constants.
        cross = np.maximum(0., pooled*pooled - (q*q) @ (p*p))
        blocks += [pooled, cross, np.array([1.-float((u/total)@(u/total))])]
    value = np.concatenate(blocks)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite MX features')
    return value


def prepare_kernel(x, query, bandwidth):
    x, query = np.asarray(x, float), np.asarray(query, float)
    if (x.ndim != 2 or query.ndim != 2 or len(x) < 2 or x.shape[1] < 1
            or query.shape[1] != x.shape[1] or not np.isfinite(x).all()
            or not np.isfinite(query).all() or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('invalid MX kernel inputs')
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-10] = 1.
    z = (x-center)/scale
    q = (query-center)/scale
    return center, scale, z, _kernel(z, z, bandwidth), _kernel(q, z, bandwidth)


def _kernel(a, b, bandwidth):
    dot = a @ b.T / a.shape[1]
    distance = np.maximum(0., np.mean(a*a, axis=1)[:,None]+np.mean(b*b, axis=1)[None,:]-2*dot)
    return .75*np.exp(-distance/bandwidth)+.25*dot


def fit_mx(x, y, alpha, bandwidth, *, mode):
    y = np.asarray(y, float)
    if (mode not in MODES or y.ndim != 2 or len(y) != len(x) or y.shape[1] < 2
            or not np.isfinite(y).all() or np.any(y < 0) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('invalid MX supervised observations')
    center, scale, z, kernel, _ = prepare_kernel(x, x, bandwidth)
    intercept = y.mean(0)
    weights = np.linalg.solve(kernel + alpha*np.eye(len(x)), y-intercept)
    return {'schema': VERSION, 'mode': mode, 'input_domain': DOMAIN,
            'center': center.tolist(), 'scale': scale.tolist(), 'support': z.tolist(),
            'weights': weights.tolist(), 'intercept': intercept.tolist(),
            'alpha': float(alpha), 'bandwidth': float(bandwidth)}


def predict_mx(model, x, *, input_domain=DOMAIN):
    if (model.get('schema') != VERSION or model.get('mode') not in MODES
            or model.get('input_domain') != DOMAIN or input_domain != DOMAIN):
        raise ValueError('MX head requires its trained nominal-stock domain, not gas or lotion')
    x, center, scale, support, weights, intercept = [np.asarray(v, float) for v in
        (x, model['center'], model['scale'], model['support'], model['weights'], model['intercept'])]
    bandwidth = model['bandwidth']
    if (x.ndim != 2 or center.ndim != 1 or scale.shape != center.shape or support.ndim != 2
            or x.shape[1] != len(center) or support.shape[1] != len(center)
            or intercept.ndim != 1 or weights.shape != (len(support),len(intercept))
            or np.any(scale <= 0) or not all(np.isfinite(v).all() for v in (x, center, scale, support, weights, intercept))
            or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('invalid portable MX head')
    value = np.maximum(0., _kernel((x-center)/scale, support, bandwidth) @ weights+intercept)
    value[np.all(x == 0, axis=1)] = 0.
    if not np.isfinite(value).all():
        raise ValueError('nonfinite MX prediction')
    return value


@dataclass(frozen=True)
class ReleaseEnvironment:
    product_volume_m3: float
    air_volume_m3: float
    conductance_m3_per_minute: float
    ventilation_m3_per_minute: float
    basis: str = 'engineering_scenario_not_measured'


def release_trajectory(initial_moles, partition_air_product, times_minutes, environment,
                       *, log_rate_correction=None):
    """Dilute, finite-transfer reservoir <-> air -> exhaust balance.

    Partition coefficients must be per component and formulation. There is no
    universal lotion multiplier. A learned correction can alter transfer speed
    only, not create mass; without calibration it defaults explicitly to zero.
    """
    n, h, times = (np.asarray(v, float) for v in (initial_moles, partition_air_product, times_minutes))
    env = environment
    physical = [env.product_volume_m3, env.air_volume_m3, env.conductance_m3_per_minute,
                env.ventilation_m3_per_minute]
    delta = np.zeros_like(n) if log_rate_correction is None else np.asarray(log_rate_correction, float)
    if (n.ndim != 1 or len(n) == 0 or h.shape != n.shape or delta.shape != n.shape
            or times.ndim != 1 or len(times) == 0 or np.any(times < 0) or np.any(np.diff(times) < 0)
            or not all(np.isfinite(v).all() for v in (n,h,times,delta,physical))
            or np.any(n < 0) or np.any(h <= 0) or min(physical[:2]) <= 0 or min(physical[2:]) < 0
            or np.any(np.abs(delta) > 20) or not env.basis):
        raise ValueError('invalid physical release scenario')
    trajectory = np.zeros((len(times), len(n), 3))
    for i, (amount, partition, correction) in enumerate(zip(n, h, delta)):
        transfer = env.conductance_m3_per_minute * math.exp(float(correction))
        a, b = transfer*partition/env.product_volume_m3, transfer/env.air_volume_m3
        v = env.ventilation_m3_per_minute/env.air_volume_m3
        operator = np.array([[-a,b,0.], [a,-b-v,0.], [0.,v,0.]])
        for j, t in enumerate(times):
            # An exact matrix exponential preserves nonnegative linear transfer
            # without explicit-Euler clipping or unaccounted mass disappearance.
            trajectory[j,i] = expm(operator*t) @ np.array([amount,0.,0.])
    tolerance = max(float(n.max()), 1e-30)*1e-9
    if not np.isfinite(trajectory).all() or trajectory.min() < -tolerance:
        raise ValueError('unstable physical release solution')
    trajectory = np.maximum(0., trajectory)
    error = np.max(np.abs(trajectory.sum(2)-n[None,:]))/max(float(n.max()), 1e-30)
    if error > 1e-8:
        raise ValueError('physical release mass balance failed')
    return {'times_minutes': times.tolist(), 'compartments': ['product','air','exhaust'],
            'moles': trajectory.tolist(), 'air_mol_per_m3': (trajectory[:,:,1]/env.air_volume_m3).tolist(),
            'mass_balance_relative_error': float(error), 'nonnegative': True,
            'evidence_basis': env.basis, 'release_calibrated': False,
            'learned_rate_correction_applied': log_rate_correction is not None,
            'gas_to_human_perception_calibrated': False}
