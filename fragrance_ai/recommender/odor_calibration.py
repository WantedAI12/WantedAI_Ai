"""Frozen, regularized calibration of public descriptor annotation predictions.

This calibrates annotation propensity, not olfactory intensity or recipe scores.
Only the explicitly configured local research model may use the correction.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = 'odor-annotation-calibration/v1'
LABEL_KIND = 'public_descriptor_annotations_not_measured_absence_or_intensity'


def neighbor_probabilities(features, fingerprints, targets, *, k, power=4., prior_fraction=.1):
    """Training-only Tanimoto kernel regression, never query-label retrieval."""
    x=np.asarray(features,dtype=np.float32)
    bank=np.asarray(fingerprints,dtype=np.float32)
    y=np.asarray(targets,dtype=np.float32)
    if (x.ndim!=2 or x.shape[1]!=1040 or not np.isfinite(x).all()
            or bank.ndim!=2 or bank.shape[1]!=1024 or y.ndim!=2 or len(y)!=len(bank)
            or type(k) is not int or not 1<=k<=len(bank)
            or not np.isfinite(power) or power<=0 or not 0<=prior_fraction<=1
            or not np.isfinite(bank).all() or np.any((bank!=0)&(bank!=1))
            or not np.isfinite(y).all() or np.any((y!=0)&(y!=1))
            or np.any((x[:,:1024]!=0)&(x[:,:1024]!=1))):
        raise ValueError('invalid structure correction bank or query')
    result=np.empty((len(x),y.shape[1]),float)
    count=bank.sum(axis=1)
    prior=(y.sum(axis=0)+.5)/(len(y)+1.)
    for start in range(0,len(x),128):
        query=x[start:start+128,:1024]
        intersection=query@bank.T
        union=query.sum(axis=1)[:,None]+count[None,:]-intersection
        similarity=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
        # Stable ordering gives deterministic handling of equal similarities.
        indices=np.argsort(-similarity,axis=1,kind='stable')[:,:k]
        weight=np.take_along_axis(similarity,indices,axis=1).astype(float)**power
        total=weight.sum(axis=1)
        estimate=np.einsum('ij,ijk->ik',weight,y[indices])/np.maximum(total[:,None],1e-30)
        estimate[total==0]=prior
        result[start:start+len(query)]=(1.-prior_fraction)*estimate+prior_fraction*prior
    return result


def probability_logits(values):
    x = np.asarray(values,dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
        raise ValueError('calibration requires a finite probability matrix in [0,1]')
    x = np.clip(x,1e-7,1.-1e-7)
    return np.log(x)-np.log1p(-x)


def apply_correction(values, slope, intercept, label_offsets, negative_log_slope=None):
    x = np.asarray(values,dtype=np.float64)
    offsets = np.asarray(label_offsets,dtype=np.float64)
    logits = probability_logits(x)
    b = slope if negative_log_slope is None else negative_log_slope
    if (offsets.shape!=(x.shape[1],) or not np.isfinite(offsets).all()
            or not np.isfinite(slope) or slope<=0 or not np.isfinite(intercept)
            or not np.isfinite(b) or b<=0):
        raise ValueError('invalid calibration coefficients')
    if slope==1. and b==1. and intercept==0. and np.all(offsets==0.):
        return x.copy()
    # Beta calibration includes the identity and can treat overconfidence in
    # the low/high probability tails differently while remaining monotonic.
    tail = -np.log1p(-np.clip(x,1e-7,1.-1e-7))
    corrected = np.clip(slope*logits+(b-slope)*tail+intercept+offsets,-40.,40.)
    return 1./(1.+np.exp(-corrected))


class OdorCalibration:
    def __init__(self,path,sha256,*,parent_sha256,endpoints,training_graphs=None,split_sha256=None):
        self.path,self.sha256 = Path(path),sha256
        raw = self.path.read_bytes()
        if not re.fullmatch('[0-9a-f]{64}',sha256) or hashlib.sha256(raw).hexdigest()!=sha256:
            raise ValueError('odor calibration content hash mismatch')
        m = json.loads(raw)
        if (m.get('schema')!=SCHEMA or m.get('scope')!='local_research'
                or m.get('parent_model_sha256')!=parent_sha256 or m.get('endpoints')!=list(endpoints)
                or m.get('label_kind')!=LABEL_KIND or m.get('accepted_for_local_research') is not True
                or m.get('recipe_score_offset')!=0 or m.get('recipe_threshold_changed') is not False
                or m.get('test_labels_used_for_fitting') is not False):
            raise ValueError('odor calibration binding, acceptance or evidence contract mismatch')
        self.slope = m['coefficients']['slope']
        self.intercept = m['coefficients']['intercept']
        self.negative_log_slope = m['coefficients'].get('negative_log_slope',self.slope)
        self.method = m['coefficients'].get('method','logit')
        if self.method not in ('logit','beta','beta_tail','neighbors'):
            raise ValueError('unsupported odor calibration method')
        if any(type(v) not in (int,float) for v in m['coefficients']['label_offsets']):
            raise ValueError('odor calibration offsets must be real numbers')
        self.offsets = np.asarray(m['coefficients']['label_offsets'],dtype=float)
        if (any(type(v) not in (int,float) for v in (self.slope,self.intercept,self.negative_log_slope))
                or not .25<=self.slope<=4. or not -5.<=self.intercept<=5.
                or not .25<=self.negative_log_slope<=4.
                or self.offsets.shape!=(len(endpoints),) or not np.isfinite(self.offsets).all()
                or np.any(np.abs(self.offsets)>3.)):
            raise ValueError('odor calibration coefficients exceed supported bounds')
        check = apply_correction(np.full((1,len(endpoints)),.5),self.slope,self.intercept,
                                 self.offsets,self.negative_log_slope)
        if not np.isfinite(check).all():
            raise ValueError('invalid odor calibration output')
        self.manifest = m
        self.bank_path = None
        self.bank = None
        if self.method=='neighbors':
            spec=m.get('training_bank',{})
            c=m['coefficients']
            self.neighbor_weight=c.get('neighbor_weight')
            self.neighbor_k=c.get('neighbor_k')
            self.neighbor_power=c.get('neighbor_power')
            self.neighbor_prior_fraction=c.get('neighbor_prior_fraction')
            if (not training_graphs or spec.get('split_sha256')!=split_sha256
                    or not isinstance(spec.get('path'),str) or not spec['path']
                    or type(self.neighbor_weight) not in (int,float) or not 0<self.neighbor_weight<=.5
                    or type(self.neighbor_k) is not int or not 1<=self.neighbor_k<=len(training_graphs)
                    or type(self.neighbor_power) not in (int,float) or not 1<=self.neighbor_power<=8
                    or type(self.neighbor_prior_fraction) not in (int,float) or not 0<=self.neighbor_prior_fraction<=1
                    or self.slope!=1. or self.negative_log_slope!=1. or self.intercept!=0.
                    or np.any(self.offsets!=0.)):
                raise ValueError('invalid training-only neighbor correction binding')
            self.bank_path=(self.path.parent/spec['path']).resolve()
            if not self.bank_path.is_relative_to(self.path.parent.resolve()):
                raise ValueError('odor correction bank escapes model directory')
            if hashlib.sha256(self.bank_path.read_bytes()).hexdigest()!=spec.get('sha256'):
                raise ValueError('odor correction bank hash mismatch')
            with np.load(self.bank_path,allow_pickle=False) as arrays:
                if set(arrays.files)!={'fingerprints','targets','graphs'}:
                    raise ValueError('odor correction bank keys mismatch')
                if arrays['graphs'].tolist()!=list(training_graphs) or len(set(training_graphs))!=len(training_graphs):
                    raise ValueError('odor correction bank is not the parent training split')
                fp=arrays['fingerprints'].astype(np.float32)
                y=arrays['targets'].astype(np.float32)
            if (fp.shape!=(len(training_graphs),1024) or y.shape!=(len(training_graphs),len(endpoints))
                    or spec.get('rows')!=len(training_graphs)):
                raise ValueError('odor correction bank dimensions mismatch')
            # Shared numeric validation also rejects nonbinary/NaN bank data.
            neighbor_probabilities(np.zeros((0,1040)),fp,y,k=self.neighbor_k)
            fp.setflags(write=False)
            y.setflags(write=False)
            self.bank=(fp,y)
        elif 'training_bank' in m or 'neighbor_weight' in m['coefficients']:
            raise ValueError('unused neighbor bank/coefficient is not allowed')
        self.offsets.setflags(write=False)

    def assert_current(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.sha256:
            raise ValueError('odor calibration changed during request')
        if self.bank_path is not None and hashlib.sha256(self.bank_path.read_bytes()).hexdigest()!=self.manifest['training_bank']['sha256']:
            raise ValueError('odor calibration bank changed during request')

    def apply(self,values,features=None):
        if self.bank is not None:
            if features is None or np.shape(features)!=(len(values),1040):
                raise ValueError('structure features are required for odor correction')
            neighbor=neighbor_probabilities(features,*self.bank,k=self.neighbor_k,power=self.neighbor_power,
                                              prior_fraction=self.neighbor_prior_fraction)
            # Never clip invalid upstream predictions into apparently valid outputs.
            probability_logits(values)
            values=(1.-self.neighbor_weight)*np.asarray(values)+self.neighbor_weight*neighbor
        return apply_correction(values,self.slope,self.intercept,self.offsets,self.negative_log_slope)

    def contract(self):
        return {'schema':SCHEMA,'sha256':self.sha256,'parent_model_sha256':self.manifest['parent_model_sha256'],
            'scope':'local_research','slope':self.slope,'intercept':self.intercept,
            'negative_log_slope':self.negative_log_slope,
            'per_label_corrected':int(np.count_nonzero(self.offsets)),
            'method':('training_only_tanimoto_neighbor_residual' if self.bank is not None else
                      'monotone_beta_with_support_shrunk_label_bias'),
            'training_neighbor_correction':({'training_molecules':len(self.bank[0]),'k':self.neighbor_k,
                'blend_weight':self.neighbor_weight,'similarity_power':self.neighbor_power,
                'prior_fraction':self.neighbor_prior_fraction,'bank_sha256':self.manifest['training_bank']['sha256'],
                'query_label_lookup':False,'validation_and_test_molecules_excluded':True} if self.bank is not None else None),
            'evaluation':self.manifest['evaluation_summary'],
            'label_kind':LABEL_KIND,'recipe_score_offset':0.,'recipe_threshold_changed':False,
            'calibrated_human_intensity':False,'human_similarity_percent':None}
