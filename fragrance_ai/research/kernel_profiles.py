"""Small CPU nonlinear molecular predictor; no sensory accuracy claims.

The training support is retained explicitly, so this is a kernel model, not an
LLM or a change to the recipe acceptance metric.
"""
import numpy as np

from .perception_validation import predict_ridge


def _kernel(a, b, scale):
    af, bf = a[:, :1024], b[:, :1024]
    dot = af @ bf.T
    union = af.sum(1)[:, None] + bf.sum(1)[None, :] - dot
    similarity = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
    # Physical, dose, native-profile and carrier context. Scales are train-only.
    az, bz = a[:, 1024:] / scale, b[:, 1024:] / scale
    distance = np.maximum(0., (az*az).sum(1)[:, None] + (bz*bz).sum(1)[None, :] - 2*az@bz.T)
    return .75 * similarity + .25 * np.exp(-distance / max(1, az.shape[1]))


def fit_kernel(x, y, alpha):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or x.shape[1] <= 1024 or y.ndim != 2 or len(x) != len(y)
            or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(y < 0) or not np.isfinite(alpha) or alpha <= 0
            or np.any((x[:, :1024] != 0) & (x[:, :1024] != 1))):
        raise ValueError('invalid kernel training contract')
    scale = x[:, 1024:].std(0)
    scale[scale < 1e-10] = 1.
    center = y.mean(0)
    weights = np.linalg.solve(_kernel(x, x, scale) + alpha*np.eye(len(x)), y-center)
    return {'kind': 'molecular-kernel-v3', 'support': x.tolist(), 'scale': scale.tolist(),
            'weights': weights.tolist(), 'intercept': center.tolist(), 'alpha': float(alpha)}


def _kernel_v4(a, b, scale, parameters):
    """PSD sum of molecular kernels times a dose/carrier context kernel.

    Feature layout is the existing molecular conditional-profile contract:
    fingerprint(1024), physical(16), physical*dose(16), native(19),
    native*dose(19), native-presence(1), log-dose(1), log-dose-square(1), carrier(8).
    Interaction duplicates are deliberately not counted as independent axes.
    """
    if a.shape[1] != 1105 or b.shape[1] != 1105:
        raise ValueError('v4 requires the fixed 1105-feature molecular contract')
    fp_weight, native_weight = parameters['fingerprint_weight'], parameters['native_weight']
    bandwidth = parameters['dose_bandwidth']
    if (not np.isfinite([fp_weight, native_weight]).all() or min(fp_weight, native_weight) < 0
            or fp_weight + native_weight > 1 or
            (bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0))):
        raise ValueError('invalid v4 kernel parameters')
    dot = a[:, :1024] @ b[:, :1024].T
    union = a[:, :1024].sum(1)[:, None]+b[:, :1024].sum(1)[None, :]-dot
    similarity = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
    def radial(indices):
        az, bz = a[:, indices]/scale[np.asarray(indices)-1024], b[:, indices]/scale[np.asarray(indices)-1024]
        distance = np.maximum(0., (az*az).sum(1)[:, None]+(bz*bz).sum(1)[None, :]-2*az@bz.T)
        return np.exp(-distance/len(indices))
    molecular = (fp_weight*similarity + (1-fp_weight-native_weight)*radial(list(range(1024,1040)))
                 + native_weight*radial(list(range(1056,1075))+[1094]))
    if bandwidth is None:
        return molecular
    dose = np.exp(-.5*((a[:, 1095, None]-b[None, :, 1095])/bandwidth)**2)
    carrier = a[:, 1097:] @ b[:, 1097:].T
    return molecular*(.5+.5*dose)*(.5+.5*carrier)


def fit_kernel_v4(x, y, alpha, *, fingerprint_weight=.5, native_weight=.25, dose_bandwidth=2.):
    # Reuse strict training validation and train-only scaling from V3.
    model = fit_kernel(x, y, alpha)
    x, y = np.asarray(x, float), np.asarray(y, float)
    parameters = dict(fingerprint_weight=fingerprint_weight, native_weight=native_weight, dose_bandwidth=dose_bandwidth)
    center = np.asarray(model['intercept'])
    model.update(kind='molecular-kernel-v4', kernel_parameters=parameters,
        weights=np.linalg.solve(_kernel_v4(x, x, np.asarray(model['scale']), parameters)+alpha*np.eye(len(x)), y-center).tolist())
    return model


def _kernel_v5(a, b, scale, parameters):
    if a.shape[1] <= 1106 or b.shape[1] != a.shape[1]:
        raise ValueError('v5 requires source-bound fine odor features')
    weight = parameters['fine_weight']
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('invalid fine odor kernel weight')
    # Jaccard support similarity is a qualitative feature kernel, not a
    # sensory overlap score. Missing support has no pairwise similarity.
    af, bf = a[:,1105:-1], b[:,1105:-1]
    if any(np.any((v != 0) & (v != 1)) for v in (af,bf,a[:,-1],b[:,-1])):
        raise ValueError('binary fine odor features required')
    dot = af @ bf.T
    union = af.sum(1)[:,None]+bf.sum(1)[None,:]-dot
    fine = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
    base = _kernel_v4(a[:,:1105], b[:,:1105], scale[:81], parameters)
    dose = parameters['dose_bandwidth']
    context = np.ones_like(fine) if dose is None else (
        (.5+.5*np.exp(-.5*((a[:,1095,None]-b[None,:,1095])/dose)**2))
        *(.5+.5*(a[:,1097:1105] @ b[:,1097:1105].T)))
    return (1-weight)*base + weight*fine*context


def fit_kernel_v5(x, y, alpha, *, fine_weight=.5):
    model = fit_kernel(x,y,alpha)
    x, y = np.asarray(x,float), np.asarray(y,float)
    params = dict(fingerprint_weight=.5, native_weight=.25, dose_bandwidth=2., fine_weight=fine_weight)
    center = np.asarray(model['intercept'])
    model.update(kind='molecular-kernel-v5', kernel_parameters=params,
        weights=np.linalg.solve(_kernel_v5(x,x,np.asarray(model['scale']),params)+alpha*np.eye(len(x)),y-center).tolist())
    return model


def predict_component_regressor(model, x):
    if model.get('kind') is None:
        return predict_ridge(model, x)
    if model.get('kind') not in ('molecular-kernel-v3', 'molecular-kernel-v4', 'molecular-kernel-v5'):
        raise ValueError('unknown component regressor')
    x, support, scale, weights, intercept = [np.asarray(v, float) for v in
        (x, model['support'], model['scale'], model['weights'], model['intercept'])]
    if (x.ndim != 2 or support.ndim != 2 or x.shape[1] != support.shape[1]
            or x.shape[1] <= 1024 or scale.shape != (x.shape[1]-1024,)
            or intercept.ndim != 1 or weights.shape != (len(support), len(intercept))
            or any(not np.isfinite(v).all() for v in (x, support, scale, weights, intercept))
            or np.any(scale <= 0) or np.any((x[:, :1024] != 0) & (x[:, :1024] != 1))):
        raise ValueError('invalid kernel prediction contract')
    kernel = (_kernel_v5(x, support, scale, model['kernel_parameters']) if model['kind'] == 'molecular-kernel-v5' else
              _kernel_v4(x, support, scale, model['kernel_parameters'])
              if model['kind'] == 'molecular-kernel-v4' else _kernel(x, support, scale))
    return np.maximum(0., kernel @ weights + intercept)
