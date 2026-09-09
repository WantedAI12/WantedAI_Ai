"""Permutation/aliquot-invariant nonlinear stock-mixture research predictor.

The assay domain is explicit stock aliquots, not lotion headspace or a
finished perfume. Kernel interactions are learned associations, not fitted
receptor antagonism or a human similarity certificate.
"""
from collections import defaultdict
import math

import numpy as np


FEATURE_VERSION = 'relative-stock-power-pooling/v1'
CROSS_FEATURE_VERSION = 'relative-stock-cross-moments/v2'
MODEL_KIND = 'stock-mixture-kernel/v2'


def mixture_features(profiles, keys, weights=None, *, cross_moments=False):
    profiles = np.asarray(profiles, dtype=float)
    if (profiles.ndim != 2 or len(profiles) == 0 or len(keys) != len(profiles)
            or not np.isfinite(profiles).all() or np.any(profiles < 0)):
        raise ValueError('complete finite nonnegative stock profiles required')
    w = np.ones(len(keys)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(keys),) or not np.isfinite(w).all() or np.any(w < 0) or w.max() <= 0:
        raise ValueError('finite nonnegative relative aliquot weights required')
    w = w/w.max(); w = w/w.sum()
    grouped, values = defaultdict(float), {}
    for key, profile, weight in zip(keys, profiles, w):
        key = tuple(key)
        if len(key) != 3 or not math.isfinite(float(key[1])) or not 0 < float(key[1]) <= 1:
            raise ValueError('molecule, stock dilution and carrier required')
        if weight == 0:
            continue
        if key in values and not np.array_equal(values[key], profile):
            raise ValueError('same stock condition has conflicting profiles')
        grouped[key] += weight
        values[key] = profile
    order = sorted(grouped)
    w = np.array([grouped[k] for k in order]); w /= w.sum()
    p = np.array([values[k] for k in order])
    mean = w@p
    # Unlike an unweighted maximum, a trace aliquot cannot contribute its
    # entire stock intensity. All three moments approach the zero-weight limit.
    root_mean = (w@np.sqrt(p))**2
    rms = np.sqrt(w@(p*p))
    log_dose = np.log10([float(k[1]) for k in order])
    log_mean = float(w@log_dose)
    effective_count = float(1./(w@w))
    moments = [mean,root_mean,rms]
    if cross_moments:
        # Marginal moments cannot distinguish mixtures with different
        # within-component descriptor associations. Preserve their symmetric
        # covariance as well. sqrt(2) on off-diagonals preserves Frobenius
        # distances without storing the redundant half of the matrix.
        centered = p-mean
        covariance = (centered.T*w)@centered
        left,right = np.triu_indices(p.shape[1])
        moments.append(covariance[left,right]*np.where(left == right,1.,np.sqrt(2.)))
    return np.r_[*moments,effective_count,math.log(effective_count),
                 log_mean,np.sqrt(max(0.,float(w@(log_dose*log_dose))-log_mean**2))]


def _kernel(a,b,scale,bandwidth):
    az,bz = a/scale,b/scale
    dot = az@bz.T/az.shape[1]
    distance = np.maximum(0.,np.mean(az*az,axis=1)[:,None]+np.mean(bz*bz,axis=1)[None,:]-2*dot)
    return .75*np.exp(-distance/bandwidth)+.25*dot


def fit_mixture_model(x,y,alpha,bandwidth,scaling='per_feature'):
    x,y = np.asarray(x,float),np.asarray(y,float)
    d = y.shape[1] if y.ndim == 2 else 0
    base_width,cross_width = 3*d+4,3*d+d*(d+1)//2+4
    if (x.ndim != 2 or y.ndim != 2 or len(x) < 2 or len(x) != len(y)
            or d == 0 or x.shape[1] not in (base_width,cross_width) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(y < 0)
            or np.any(x[:,-4] < 1.-1e-8)
            or not np.isfinite([alpha,bandwidth]).all() or min(alpha,bandwidth) <= 0
            or scaling not in ('per_feature','shared_sensory','balanced_sensory')):
        raise ValueError('invalid mixture training contract')
    # Learn an interaction correction, not a replacement for a single stock.
    # The gate vanishes continuously as other aliquot fractions vanish.
    gate = np.clip(1.-1./x[:,-4],0.,1.)[:,None]
    residual = np.divide(y-x[:,:y.shape[1]],gate,out=np.zeros_like(y),where=gate > 1e-10)
    center,scale,intercept = x.mean(0),x.std(0),residual.mean(0)
    if scaling in ('shared_sensory','balanced_sensory'):
        # RATA endpoint intensities share units. A separate scale for every
        # rare endpoint can amplify near-zero noise into a major distance.
        # Share train-only RMS scales within each sensory-moment block.
        for offset in range(0,3*d,d):
            scale[offset:offset+d] = np.sqrt(np.mean(scale[offset:offset+d]**2))
        if x.shape[1] == cross_width:
            scale[3*d:-4] = np.sqrt(np.mean(scale[3*d:-4]**2))
        if scaling == 'balanced_sensory':
            # A covariance block has O(d^2) coordinates. Counting coordinates
            # as equal votes would let it overwhelm the O(d) sensory blocks.
            # Equal block weights retain 4/157 for the four dose/count terms;
            # this is fixed before outcomes, not tuned per held-out mixture.
            blocks = [(offset,offset+d) for offset in range(0,3*d,d)]
            if x.shape[1] == cross_width:
                blocks.append((3*d,x.shape[1]-4))
            sensory_weight = (1.-4./157)/len(blocks)
            for left,right in blocks:
                scale[left:right] *= np.sqrt((right-left)/(sensory_weight*x.shape[1]))
            scale[-4:] *= np.sqrt(157./x.shape[1])
    scale[scale < 1e-10] = 1.
    z = x-center
    coefficients = np.linalg.solve(_kernel(z,z,scale,bandwidth)+alpha*np.eye(len(z)),residual-intercept)
    feature_kind = CROSS_FEATURE_VERSION if x.shape[1] == cross_width else FEATURE_VERSION
    return {'kind':MODEL_KIND,'features':feature_kind,'support':z.tolist(),
        'center':center.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist(),
        'intercept':intercept.tolist(),'alpha':float(alpha),'bandwidth':float(bandwidth),'scaling':scaling}


def predict_mixture_model(model,x):
    if (model.get('kind') not in ('stock-mixture-kernel/v1',MODEL_KIND)
            or model.get('features') not in (FEATURE_VERSION,CROSS_FEATURE_VERSION)
            or (model.get('kind') == 'stock-mixture-kernel/v1' and model.get('features') != FEATURE_VERSION)):
        raise ValueError('unsupported mixture model contract')
    x,support,center,scale,coefficients,intercept = [np.asarray(v,float) for v in (
        x,model['support'],model['center'],model['scale'],model['coefficients'],model['intercept'])]
    d = len(intercept) if intercept.ndim == 1 else 0
    width = 3*d+4+(d*(d+1)//2 if model['features'] == CROSS_FEATURE_VERSION else 0)
    if (x.ndim != 2 or support.ndim != 2 or center.ndim != 1 or intercept.ndim != 1
            or d == 0 or len(support) < 2 or support.shape[1] != width
            or x.shape[1] != support.shape[1] or center.shape != (support.shape[1],)
            or scale.shape != center.shape or coefficients.shape != (len(support),len(intercept))
            or any(not np.isfinite(v).all() for v in (x,support,center,scale,coefficients,intercept))
            or np.any(scale <= 0) or not np.isfinite(model['bandwidth']) or model['bandwidth'] <= 0):
        raise ValueError('invalid portable mixture model')
    value = _kernel(x-center,support,scale,model['bandwidth'])@coefficients+intercept
    if model['kind'] == MODEL_KIND:
        if np.any(x[:,-4] < 1.-1e-8):
            raise ValueError('effective stock count must be at least one')
        value = x[:,:len(intercept)]+np.clip(1.-1./x[:,-4],0.,1.)[:,None]*value
    return np.maximum(0.,value)
