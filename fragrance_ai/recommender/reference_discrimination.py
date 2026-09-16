"""Source-validated, family-specific identity discrimination.

The clean prototype uses log likelihood contrast; other prototypes retain the
legacy cosine criterion. Selection is by source target identity, never by a
recipe score or benchmark ID. Context is request-local and final comparison
also selects the method explicitly.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import numpy as np

VERSION='source-family-discrimination/v89'
EPSILON=1e-9
MARGIN=1e-8
_CLEAN_REFERENCES=ContextVar('clean_reference_pairs',default=())


def _normalized(values):
    v=np.asarray(values,float)
    if not np.isfinite(v).all() or np.any(v<0) or np.any(v.sum(-1)<=0):
        raise ValueError('finite nonempty reference profile required')
    return v/v.sum(-1,keepdims=True)


@contextmanager
def reference_context(bank):
    clean=getattr(bank,'profiles',{}).get('clean')
    pairs=() if clean is None else tuple(zip(_normalized(clean),_normalized(bank.background)))
    token=_CLEAN_REFERENCES.set(pairs)
    try:
        yield
    finally:
        _CLEAN_REFERENCES.reset(token)


def using_reference_policy(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with reference_context(kwargs['bank']):
            return function(*args,**kwargs)
    return wrapped


def is_clean_reference(target,clean_reference):
    return clean_reference is not None and np.allclose(_normalized(target),_normalized(clean_reference),atol=1e-12,rtol=0.)


def contrast_direction(target,background,*,method=None):
    q,b=np.broadcast_arrays(_normalized(target),_normalized(background))
    old=q/np.linalg.norm(q,axis=-1,keepdims=True)-b/np.linalg.norm(b,axis=-1,keepdims=True)
    if method not in (None,'cosine','log_ratio'):
        raise ValueError('unknown source discrimination method')
    use=np.zeros(q.shape[:-1],bool)
    if method=='log_ratio':
        use[...] = True
    elif method is None:
        for clean,base in _CLEAN_REFERENCES.get():
            use |= np.all(np.abs(q-clean)<=1e-12,axis=-1)&np.all(np.abs(b-base)<=1e-12,axis=-1)
    delta=np.log(q+EPSILON)-np.log(b+EPSILON)
    norm=np.linalg.norm(delta,axis=-1,keepdims=True)
    ratio=np.divide(delta,norm,out=np.zeros_like(delta),where=norm>1e-12)
    return np.where(use[...,None],ratio,old)


def assess(target,predicted,background,*,clean_reference=None):
    q,p,b=map(lambda x:np.asarray(x,float),(target,predicted,background))
    method='log_ratio' if is_clean_reference(q,clean_reference) else 'cosine'
    direction=contrast_direction(q,b,method=method)
    norm=float(np.linalg.norm(p))
    if not np.isfinite(p).all() or np.any(p<0) or norm<=0:
        raise ValueError('positive finite predicted profile required')
    margin=float(p@direction/norm)
    old=float(p@(q/np.linalg.norm(q)-b/np.linalg.norm(b))/norm)
    return {'version':VERSION,'method':method,'margin':margin,'threshold':MARGIN,'passed':margin>MARGIN,
            'identifiable':bool(np.linalg.norm(direction)>0),
            'legacy_cosine_margin':old,'legacy_cosine_passed':old>MARGIN,
            'probability_calibrated':False,'source_smoothing':EPSILON if method=='log_ratio' else None}
