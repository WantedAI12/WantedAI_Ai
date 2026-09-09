"""V54 molecular / V48 stock multi-view mixture learning on explicit aliquots.

Atlas high/low are ordinal source levels, never physical concentrations. The
physical inputs here remain the caller's stock dilutions and relative volumes.
All scalers and residual calibration are fitted on training compositions only.
"""
import numpy as np

from .mixture_profiles import _kernel
from .physsim_mx import mx_features

KIND = 'v54-stock-multiview-kernel/v1'
FEATURES = 'v54-four-head-concentration-with-coverage/v1'
TRANSFORMS = ('identity', 'sqrt', 'log1p')


def atlas_mixture_features(profiles, identities, dilutions, carriers, weights=None, *, supported=None):
    p = np.asarray(profiles, float)
    if p.ndim != 2 or p.shape[1] != 584:
        raise ValueError('four frozen 146-axis V54 heads required')
    ok = np.ones(len(p), bool) if supported is None else np.asarray(supported, bool)
    if ok.shape != (len(p),) or np.any(p[~ok] != 0):
        raise ValueError('unsupported Atlas graphs need an explicit zero feature and coverage mask')
    base = mx_features(p, identities, dilutions, carriers, weights, mode='concentration')
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    # mx_features validates the volumes before this normalization.
    w = w / w.max(); w /= w.sum()
    return np.r_[base, w @ ok]


def transform(values, name):
    v = np.asarray(values, float)
    if name not in TRANSFORMS or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('finite nonnegative responses and a supported transform required')
    return np.sqrt(v) if name == 'sqrt' else np.log1p(v) if name == 'log1p' else v


def inverse(values, name):
    v = np.maximum(0., values)
    if name == 'sqrt':
        result = v*v
    elif name == 'log1p':
        with np.errstate(over='ignore'):
            result = np.expm1(v)
    elif name == 'identity':
        result = v
    else:
        raise ValueError('unsupported response transform')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite transformed mixture response')
    return result


def prepare_views(x, atlas, scaling):
    x, atlas = np.asarray(x, float), np.asarray(atlas, float)
    d = (x.shape[1]-4)//3 if x.ndim == 2 else 0
    if (d < 1 or x.shape[1] != 3*d+4 or atlas.shape != (len(x), 1179)
            or len(x) < 2 or scaling not in ('per_feature', 'shared_sensory')
            or not np.isfinite(x).all() or not np.isfinite(atlas).all()
            or np.any(x[:, -4] < 1.-1e-8)):
        raise ValueError('invalid stock / Atlas training views')
    center, scale = x.mean(0), x.std(0)
    if scaling == 'shared_sensory':
        for offset in range(0, 3*d, d):
            scale[offset:offset+d] = np.sqrt(np.mean(scale[offset:offset+d]**2))
    scale[scale < 1e-10] = 1.
    acenter, ascale = atlas.mean(0), atlas.std(0)
    ascale[ascale < 1e-10] = 1.
    return center, scale, acenter, ascale


def view_kernel(x, atlas, query, query_atlas, prepared, bandwidth, atlas_weight):
    if (not np.isfinite(bandwidth) or bandwidth <= 0
            or not np.isfinite(atlas_weight) or not 0 <= atlas_weight <= 1):
        raise ValueError('invalid kernel parameters')
    center, scale, acenter, ascale = prepared
    stock = _kernel(query-center, x-center, scale, bandwidth)
    if atlas_weight == 0:
        return stock
    molecular = _kernel(query_atlas-acenter, atlas-acenter, ascale, bandwidth)
    return (1-atlas_weight)*stock + atlas_weight*molecular


def residual_targets(x, y, output_transform):
    y = np.asarray(y, float)
    d = y.shape[1] if y.ndim == 2 else 0
    if x.shape != (len(y), 3*d+4):
        raise ValueError('mixture target / feature shape mismatch')
    gate = np.clip(1.-1./x[:, -4], 0., 1.)[:, None]
    return np.divide(transform(y, output_transform)-transform(x[:, :d], output_transform), gate,
                     out=np.zeros_like(y), where=gate > 1e-10)


def restore_response(x, correction, output_transform):
    d = correction.shape[1]
    gate = np.clip(1.-1./x[:, -4], 0., 1.)[:, None]
    result = inverse(transform(x[:, :d], output_transform)+gate*correction, output_transform)
    # Exact identity for one distinct stock, including split equivalent rows.
    result[gate[:, 0] == 0] = x[gate[:, 0] == 0, :d]
    return result


def fit_unified(x, atlas, y, *, alpha, bandwidth, scaling, atlas_weight, output_transform):
    x, atlas, y = map(lambda v: np.asarray(v, float), (x, atlas, y))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive ridge regularization required')
    prepared = prepare_views(x, atlas, scaling)
    residual = residual_targets(x, y, output_transform)
    intercept = residual.mean(0)
    kernel = view_kernel(x, atlas, x, atlas, prepared, bandwidth, atlas_weight)
    coefficients = np.linalg.solve(kernel + alpha*np.eye(len(x)), residual-intercept)
    return {'kind': KIND, 'features': FEATURES, 'support': x.tolist(), 'atlas_support': atlas.tolist(),
            **{k:v.tolist() for k,v in zip(('center','scale','atlas_center','atlas_scale'), prepared)},
            'coefficients': coefficients.tolist(), 'intercept': intercept.tolist(), 'alpha': float(alpha),
            'bandwidth': float(bandwidth), 'scaling': scaling, 'atlas_weight': float(atlas_weight),
            'output_transform': output_transform}


def predict_unified(model, x, atlas):
    if model.get('kind') != KIND or model.get('features') != FEATURES:
        raise ValueError('unsupported unified model contract')
    keys = ('support','atlas_support','center','scale','atlas_center','atlas_scale','coefficients','intercept')
    support, asupport, center, scale, acenter, ascale, coeff, intercept = (
        np.asarray(model[k], float) for k in keys)
    x, atlas = np.asarray(x, float), np.asarray(atlas, float)
    d = len(intercept) if intercept.ndim == 1 else 0
    if (not d or support.ndim != 2 or len(support) < 2 or support.shape[1] != 3*d+4
            or asupport.shape != (len(support), 1179) or x.ndim != 2 or x.shape[1] != 3*d+4
            or atlas.shape != (len(x), 1179) or center.shape != (3*d+4,) or scale.shape != center.shape
            or acenter.shape != (1179,) or ascale.shape != acenter.shape
            or coeff.shape != (len(support), d) or np.any(scale <= 0) or np.any(ascale <= 0)
            or any(not np.isfinite(v).all() for v in (support,asupport,center,scale,acenter,ascale,coeff,intercept,x,atlas))
            or np.any(x[:, -4] < 1.-1e-8)):
        raise ValueError('invalid unified model or input arrays')
    kernel = view_kernel(support, asupport, x, atlas, (center,scale,acenter,ascale),
                         model['bandwidth'], model['atlas_weight'])
    return restore_response(x, kernel@coeff+intercept, model['output_transform'])
