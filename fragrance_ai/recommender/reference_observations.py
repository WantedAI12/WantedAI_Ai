"""Pinned observed component reference shapes, separate from product claims."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

VERSION='observed-component-reference/v1'


class ComponentReferenceObservations:
    def __init__(self,path,digest):
        self.path,self.sha256=Path(path),digest
        raw=self.path.read_bytes()
        if len(raw)>20_000_000 or hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('observed component reference hash/size mismatch')
        value=json.loads(raw)
        if (value.get('schema')!=VERSION or value.get('source_ordinal_level')!='high'
                or value.get('recipe_outcomes_used') is not False
                or value.get('product_sensory_calibration') is not False):
            raise ValueError('invalid observed component reference scope')
        self.endpoints=tuple(value['endpoints'])
        if len(self.endpoints)!=146 or len(set(self.endpoints))!=146:
            raise ValueError('full observed endpoint identity required')
        entries={}
        for row in value['records']:
            graph=row['canonical_smiles']
            shape=np.asarray(row['profiles'],float)
            if (not isinstance(graph,str) or not graph or '.' in graph or graph in entries
                    or shape.shape!=(2,146) or not np.isfinite(shape).all() or np.any(shape<0)
                    or not np.allclose(shape.sum(-1),1.,atol=1e-12,rtol=1e-12)
                    or not row.get('source_stimulus_ids')):
                raise ValueError('invalid or duplicate observed component profile')
            shape.setflags(write=False)
            entries[graph]=(shape,tuple(row['source_stimulus_ids']))
        self.entries=MappingProxyType(entries)

    def assert_current(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.sha256:
            raise ValueError('observed component references changed during calculation')

    def lookup(self,graph):
        return self.entries.get(graph)


@lru_cache(maxsize=4)
def _load(path,digest):
    return ComponentReferenceObservations(path,digest)


def configured_component_references():
    from .local_runtime import local_profile
    profile=local_profile()
    binding=profile.get('component_reference_observations') if profile else None
    if binding is None:
        return None
    store=_load(*binding)
    store.assert_current()
    return store
