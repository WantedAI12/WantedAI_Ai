"""CPU-only fine annotation head, separate from quantitative Atlas intensities."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading
from types import MappingProxyType

import numpy as np

from .odor_expression import registry

VERSION = 'fine-odor-annotation-model/v1'
_LOCK = threading.RLock()


def structure_features(graphs):
    from fragrance_ai.research.conditional_profiles import molecule_features
    if any(not isinstance(g,str) or not g.strip() or '.' in g for g in graphs):
        raise ValueError('one explicit molecular graph is required')
    rows = [molecule_features(s) for s in graphs]
    result = np.zeros((len(rows),1040),np.float32)
    for i,row in enumerate(rows):
        result[i,row['fingerprint_bits']] = 1.
        result[i,1024:] = row['physical']
    return result


class FineOdorModel:
    def __init__(self, path, sha256, *, identity_provider=None, calibration=None):
        self.path, self.sha256 = Path(path), sha256
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('fine odor manifest hash mismatch')
        self.manifest = m = json.loads(raw)
        if (m.get('schema') != VERSION or m.get('registry_sha256') != registry()['sha256']
                or m.get('label_kind') != 'public_descriptor_annotations_not_measured_absence_or_intensity'
                or m.get('annotation_inputs_used') is not False):
            raise ValueError('fine odor provenance / feature contract mismatch')
        self.endpoints = tuple(m['endpoints'])
        if len(set(self.endpoints)) != len(self.endpoints) or set(self.endpoints)-set(registry()['rows']):
            raise ValueError('fine odor output identity mismatch')
        self.weights_path = (self.path.parent/m['weights']['path']).resolve()
        if not self.weights_path.is_relative_to(self.path.parent.resolve()):
            raise ValueError('fine odor weights escape model directory')
        raw_weights = self.weights_path.read_bytes()
        if hashlib.sha256(raw_weights).hexdigest() != m['weights']['sha256']:
            raise ValueError('fine odor weights hash mismatch')
        with np.load(self.weights_path, allow_pickle=False) as arrays:
            self.w = {k:arrays[k].copy() for k in arrays.files}
        if any(not np.isfinite(v).all() for v in self.w.values()):
            raise ValueError('nonfinite fine odor weights')
        required = {'mean','scale','prior','w0','b0','w1','b1','w2','b2'}
        if set(self.w)!=required:
            raise ValueError('fine odor weight keys mismatch')
        h0,h1 = len(self.w['b0']),len(self.w['b1'])
        expected={'mean':(16,),'scale':(16,),'prior':(len(self.endpoints),),
            'w0':(1040,h0),'b0':(h0,),'w1':(h0,h1),'b1':(h1,),
            'w2':(h1,len(self.endpoints)),'b2':(len(self.endpoints),)}
        if (any(self.w[k].shape!=shape for k,shape in expected.items()) or np.any(self.w['scale']<=0)
                or not 0<=m['neural_blend']<=1 or np.any(self.w['prior']<0) or np.any(self.w['prior']>1)):
            raise ValueError('fine odor weight dimensions/range mismatch')
        self.annotations = m['source_annotations']
        if any(not isinstance(g,str) or '.' in g or len(set(ids))!=len(ids)
               or any(type(i) is not int or not 0<=i<len(self.endpoints) for i in ids)
               for g,ids in self.annotations.items()):
            raise ValueError('invalid fine odor source annotation identity')
        self.identity_provider = identity_provider
        self.structures = MappingProxyType(dict(identity_provider.structures)) if identity_provider is not None else {}
        self._cache = {}
        self.calibration = None
        self.split_path = None
        if calibration is not None:
            from .odor_calibration import OdorCalibration
            split_sha=m['evaluation_summary'].get('split_sha256')
            training_graphs=None
            if split_sha is not None:
                self.split_path=self.path.parent/'split.json'
                split_raw=self.split_path.read_bytes()
                if hashlib.sha256(split_raw).hexdigest()!=split_sha:
                    raise ValueError('fine odor training split hash mismatch')
                records=json.loads(split_raw)['records']
                training_graphs=[r['graph'] for r in records if r['split']==0]
                train_groups={r['group'] for r in records if r['split']==0}
                if (train_groups & {r['group'] for r in records if r['split']!=0}
                        or set(training_graphs) & {r['graph'] for r in records if r['split']!=0}):
                    raise ValueError('fine odor correction split leaks heldout identities/groups')
            self.calibration=OdorCalibration(*calibration,parent_sha256=sha256,endpoints=self.endpoints,
                                             training_graphs=training_graphs,split_sha256=split_sha)

    def assert_current(self):
        if (registry()['sha256'] != self.manifest['registry_sha256']
                or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256
                or hashlib.sha256(self.weights_path.read_bytes()).hexdigest() != self.manifest['weights']['sha256']):
            raise ValueError('fine odor model changed during request')
        if self.identity_provider is not None:
            from .perception_runtime import assert_provider_current
            assert_provider_current(self.identity_provider)
        if self.calibration is not None:
            self.calibration.assert_current()
        if self.split_path is not None and hashlib.sha256(self.split_path.read_bytes()).hexdigest()!=self.manifest['evaluation_summary']['split_sha256']:
            raise ValueError('fine odor correction split changed during request')

    def predict_features(self, x):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 1040 or not np.isfinite(x).all():
            raise ValueError('fine odor input must be structure-only 1040 features')
        z = x.copy()
        z[:,1024:] = np.clip((z[:,1024:]-self.w['mean'])/self.w['scale'], -8., 8.)
        z = np.maximum(0., z@self.w['w0']+self.w['b0'])
        z = np.maximum(0., z@self.w['w1']+self.w['b1'])
        logits = z@self.w['w2']+self.w['b2']
        neural = 1/(1+np.exp(-np.clip(logits,-40,40)))
        alpha = self.manifest['neural_blend']
        values=alpha*neural+(1-alpha)*self.w['prior']
        return self.calibration.apply(values,x) if self.calibration is not None else values

    def predict(self, graphs):
        # No query-label lookup. Optional neighbors use the TRAIN split only.
        self.assert_current()
        result = self.predict_features(structure_features(graphs))
        self.assert_current()
        return result

    @property
    def prediction_correction_sha256(self):
        return self.calibration.sha256 if self.calibration is not None else None

    def materials(self, items):
        self.assert_current()
        graphs = []
        for item in items:
            graph = item.structure_smiles
            if not graph and item.ingredient_id in self.structures:
                graph,expected_cas = self.structures[item.ingredient_id]
                if expected_cas is not None and item.cas_number!=expected_cas:
                    raise ValueError('fine odor material identity/CAS binding mismatch')
            graphs.append(graph if graph and '.' not in graph else None)
        with _LOCK:
            missing = sorted({g for g in graphs if g and g not in self._cache})
            pending = {}
            for start in range(0,len(missing),256):
                batch = missing[start:start+256]
                # One validated operation, not one full artifact rehash per
                # 256 rows. Publish nothing if the ending snapshot has drifted.
                for graph, values in zip(batch,self.predict_features(structure_features(batch))):
                    pending[graph] = values
            self.assert_current()
            self._cache.update(pending)
            result, evidence = [], []
            for item, graph in zip(items,graphs):
                if not graph:
                    result.append(np.zeros(len(self.endpoints)))
                    evidence.append({'ingredient_id':item.ingredient_id,'status':'missing_or_multicomponent_structure'})
                    continue
                values = self._cache[graph].copy()
                linked = self.annotations.get(graph, [])
                # Positive public support overrides predicted annotation
                # propensity, not intensity. Unlisted values remain predicted.
                values[linked] = 1.
                result.append(values)
                evidence.append({'ingredient_id':item.ingredient_id,
                    'status':'source_support_plus_predicted_unlisted' if linked else 'structure_prediction_only',
                    'source_positive_labels':len(linked)})
        return np.asarray(result).reshape(len(items), len(self.endpoints)), evidence

    def contract(self):
        self.assert_current()
        evaluation=self.manifest['evaluation_summary']
        if self.calibration is not None:
            evaluation={**evaluation,'uncorrected_learned':evaluation.get('learned'),
                'learned':self.calibration.manifest['evaluation_summary']['after'],
                'correction_sha256':self.calibration.sha256}
        return {'version':VERSION,'model_sha256':self.sha256,'predicted_concepts':len(self.endpoints),
            'input_features':1040,'annotation_inputs_used':False,'cpu_only_runtime':True,
            'neural_blend':self.manifest['neural_blend'],
            'verified_material_identity_bindings':len(self.structures),
            'prediction_correction':self.calibration.contract() if self.calibration is not None else None,
            'evaluation':evaluation,
            'label_kind':self.manifest['label_kind'],'human_similarity_percent':None}


@lru_cache(maxsize=2)
def _load(path, digest, size, mtime, provider, calibration_key):
    return FineOdorModel(path,digest,identity_provider=provider,
                        calibration=calibration_key[:2] if calibration_key is not None else None)


def configured_fine_odor():
    from .formulation_core import configured_formulation_core
    from .formulation_views import shared_views
    core = configured_formulation_core()
    if core is not None:
        return shared_views(core)[1]
    from .local_runtime import local_profile
    profile = local_profile()
    if not profile or 'odor_expression' not in profile:
        return None
    path,digest = profile['odor_expression']
    from .perception_runtime import configured_perception
    provider = configured_perception()
    stat = Path(path).stat()
    calibration_key=None
    if 'odor_calibration' in profile:
        cp,ch=profile['odor_calibration']
        cs=Path(cp).stat()
        calibration_key=(cp,ch,cs.st_size,cs.st_mtime_ns)
    with _LOCK:
        return _load(path,digest,stat.st_size,stat.st_mtime_ns,provider,calibration_key)
