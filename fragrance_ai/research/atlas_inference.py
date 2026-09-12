"""Immutable CPU compilation of our molecular multi-output kernel heads.

The checkpoint equations do not change here. Identical feature-space kernels
are computed once per query batch, even when measurement heads have distinct
output coefficients. No model files or registries are read during this forward.
"""
from copy import deepcopy
import hashlib
from types import MappingProxyType

import numpy as np

from .atlas_profiles import _valid_features, atlas_kernel, predict_atlas


class CompiledAtlasHeads:
    def __init__(self, models):
        compiled, groups = {}, {}
        width = None
        for name, supplied in models.items():
            model = deepcopy(supplied)
            for key in ('support','weights','scale','intercept'):
                array = np.array(model[key], dtype=float, copy=True)
                array.setflags(write=False)
                model[key] = array
            # Reuse the portable reference implementation's complete contract
            # validation once. Query validation remains mandatory per call.
            predict_atlas(model, model['support'][:1])
            if width is not None and model['support'].shape[1] != width:
                raise ValueError('Atlas heads require the same feature layout')
            width = model['support'].shape[1]
            normalized = model['kind'] == 'atlas-quantitative-profiles/v3'
            digest = hashlib.sha256()
            for array in (model['support'],model['scale']):
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            digest.update(repr((float(model['fine_weight']),normalized)).encode())
            groups.setdefault(digest.hexdigest(), []).append((name,model))
            compiled[name] = MappingProxyType(model)
        if not compiled:
            raise ValueError('at least one Atlas measurement head required')
        self.models = MappingProxyType(compiled)
        self.width = width
        self.groups = []
        for heads in groups.values():
            example = heads[0][1]
            coefficients = np.concatenate([m['weights'] for _,m in heads],axis=1)
            intercept = np.concatenate([m['intercept'] for _,m in heads])
            coefficients.setflags(write=False)
            intercept.setflags(write=False)
            slices, start = [], 0
            for name, model in heads:
                stop = start+len(model['intercept'])
                slices.append((name,start,stop,model.get('target_transform','identity')))
                start = stop
            self.groups.append((example['support'],example['scale'],example['fine_weight'],
                example['kind']=='atlas-quantitative-profiles/v3',coefficients,intercept,tuple(slices)))
        self.groups = tuple(self.groups)

    def predict(self, features):
        x = np.asarray(features,float)
        if not _valid_features(x) or x.shape[1] != self.width:
            raise ValueError('invalid compiled Atlas query features')
        # Bound transient kernel memory by batching, not by dropping molecules.
        output = {name:np.empty((len(x),len(model['intercept']))) for name,model in self.models.items()}
        for offset in range(0,len(x),256):
            query = x[offset:offset+256]
            for support,scale,weight,normalized,coefficients,intercept,slices in self.groups:
                kernel = atlas_kernel(query,support,scale,weight,normalize=normalized)
                latent = np.maximum(0.,kernel@coefficients+intercept)
                for name,start,stop,transform in slices:
                    value = latent[:,start:stop]
                    with np.errstate(over='raise',invalid='raise'):
                        result = value**2 if transform=='sqrt' else np.expm1(value) if transform=='log1p' else value
                    if not np.isfinite(result).all():
                        raise ValueError('nonfinite compiled Atlas response')
                    output[name][offset:offset+len(query)] = result
        return output
