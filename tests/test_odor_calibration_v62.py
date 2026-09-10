"""Numerical, train-only provenance and runtime wiring; not human odor tests."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from fragrance_ai.recommender.odor_calibration import (
    SCHEMA,LABEL_KIND,OdorCalibration,apply_correction,neighbor_probabilities,
)
from fragrance_ai.recommender.fine_odor_model import FineOdorModel,VERSION,structure_features
from fragrance_ai.recommender.odor_expression import registry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return path,sha(path)


def manifest(parent='a'*64):
    return {'schema':SCHEMA,'scope':'local_research','parent_model_sha256':parent,
        'endpoints':['greenapple','peach'],'label_kind':LABEL_KIND,
        'accepted_for_local_research':True,'test_labels_used_for_fitting':False,
        'recipe_score_offset':0.,'recipe_threshold_changed':False,
        'coefficients':{'slope':1.,'negative_log_slope':1.5,'intercept':0.,'label_offsets':[0.,0.],'method':'beta'},
        'evaluation_summary':{'before':{},'after':{},'scope':'fixture_not_benchmark'}}


def test_identity_and_monotone_beta_bounds():
    x=np.linspace(0,1,1000).reshape(-1,2)
    np.testing.assert_array_equal(apply_correction(x,1.,0.,[0,0]),x)
    y=apply_correction(x,.9,-.2,[.1,-.1],1.4)
    assert np.isfinite(y).all() and np.all((y>=0)&(y<=1))
    assert np.all(np.diff(y,axis=0)>0)


@pytest.mark.parametrize('x',[[.1,.2],[[np.nan,.2]],[[np.inf,.2]],[[-.01,.2]],[[1.01,.2]]])
def test_invalid_base_probabilities_not_repaired_by_clipping(x):
    with pytest.raises(ValueError):
        apply_correction(x,1.,0.,[0,0])


def test_neighbor_kernel_is_train_only_and_no_overlap_uses_training_prior():
    fp=np.zeros((2,1024));fp[0,0]=1;fp[1,1]=1
    target=np.eye(2);x=np.zeros((3,1040));x[:2,:1024]=fp
    y=neighbor_probabilities(x,fp,target,k=1)
    np.testing.assert_allclose(y,[[.95,.05],[.05,.95],[.5,.5]])
    # Query physical features never stand in for binary fingerprint bits.
    x[:,1024:]=9
    np.testing.assert_array_equal(neighbor_probabilities(x,fp,target,k=1),y)


@pytest.mark.parametrize('change',['nonbinary_query','nonbinary_bank','nan_target','bad_k','wrong_shape'])
def test_invalid_neighbor_bank_rejected(change):
    fp=np.zeros((2,1024));y=np.eye(2);x=np.zeros((1,1040));k=1
    if change=='nonbinary_query':x[0,0]=.5
    elif change=='nonbinary_bank':fp[0,0]=.5
    elif change=='nan_target':y[0,0]=np.nan
    elif change=='bad_k':k=3
    else:x=np.zeros((1,1024))
    with pytest.raises(ValueError):neighbor_probabilities(x,fp,y,k=k)


@pytest.mark.parametrize('key,value',[
    ('parent_model_sha256','b'*64),('endpoints',['peach','greenapple']),
    ('accepted_for_local_research',False),('test_labels_used_for_fitting',True),
    ('recipe_score_offset',5),('recipe_threshold_changed',True),('scope','production'),
])
def test_calibration_requires_exact_parent_and_evidence_boundary(tmp_path,key,value):
    m=manifest();m[key]=value
    path,digest=dump(tmp_path/'correction.json',m)
    with pytest.raises(ValueError,match='contract'):
        OdorCalibration(path,digest,parent_sha256='a'*64,endpoints=['greenapple','peach'])


@pytest.mark.parametrize('key,value',[('slope',True),('slope','1'),('slope',0.),
    ('negative_log_slope',6.),('label_offsets',[True,0.]),('label_offsets',[4.,0.])])
def test_invalid_frozen_coefficients(tmp_path,key,value):
    m=manifest();m['coefficients'][key]=value
    path,digest=dump(tmp_path/'correction.json',m)
    with pytest.raises(ValueError):
        OdorCalibration(path,digest,parent_sha256='a'*64,endpoints=['greenapple','peach'])


@pytest.fixture
def bank_artifact(tmp_path):
    graphs=['CCO','CCC'];features=structure_features(graphs)
    path=tmp_path/'bank.npz'
    np.savez(path,fingerprints=features[:,:1024].astype(np.uint8),targets=np.eye(2,dtype=np.uint8),graphs=graphs)
    m=manifest()
    m['coefficients'].update(method='neighbors',negative_log_slope=1.,neighbor_weight=.25,
        neighbor_k=1,neighbor_power=4.,neighbor_prior_fraction=.1)
    m['training_bank']={'path':'bank.npz','sha256':sha(path),'split_sha256':'s'*64,'rows':2}
    return m,graphs


def test_neighbor_artifact_requires_exact_training_split(tmp_path,bank_artifact):
    m,graphs=bank_artifact;path,digest=dump(tmp_path/'correction.json',m)
    with pytest.raises(ValueError,match='training split'):
        OdorCalibration(path,digest,parent_sha256='a'*64,endpoints=m['endpoints'],
                        training_graphs=graphs[::-1],split_sha256='s'*64)
    with pytest.raises(ValueError,match='binding'):
        OdorCalibration(path,digest,parent_sha256='a'*64,endpoints=m['endpoints'])


def test_neighbor_runtime_parity_missing_features_and_drift(tmp_path,bank_artifact):
    m,graphs=bank_artifact;path,digest=dump(tmp_path/'correction.json',m)
    model=OdorCalibration(path,digest,parent_sha256='a'*64,endpoints=m['endpoints'],
                          training_graphs=graphs,split_sha256='s'*64)
    x=structure_features(['CCOC']);base=np.array([[.2,.4]])
    expected=.75*base+.25*neighbor_probabilities(x,*model.bank,k=1)
    np.testing.assert_array_equal(model.apply(base,x),expected)
    assert not model.contract()['training_neighbor_correction']['query_label_lookup']
    with pytest.raises(ValueError,match='structure features'):model.apply(base)
    model.bank_path.write_bytes(b'drift')
    with pytest.raises(ValueError,match='bank changed'):model.assert_current()


@pytest.fixture
def parent(tmp_path):
    weights={'mean':np.zeros(16),'scale':np.ones(16),'prior':np.array([.1,.2]),
        'w0':np.zeros((1040,4)),'b0':np.zeros(4),'w1':np.zeros((4,3)),'b1':np.zeros(3),
        'w2':np.zeros((3,2)),'b2':np.array([-2.,-1.])}
    np.savez(tmp_path/'weights.npz',**weights)
    split={'records':[{'graph':'CCO','split':0,'group':'g1'},{'graph':'CCC','split':0,'group':'g2'},
        {'graph':'CCOC','split':1,'group':'g3'},{'graph':'CCOCC','split':2,'group':'g4'}]}
    dump(tmp_path/'split.json',split)
    m={'schema':VERSION,'registry_sha256':registry()['sha256'],'label_kind':LABEL_KIND,
        'annotation_inputs_used':False,'endpoints':['greenapple','peach'],'neural_blend':.75,
        'weights':{'path':'weights.npz','sha256':sha(tmp_path/'weights.npz')},
        'source_annotations':{'CCO':[0],'CCC':[1],'CCOC':[1],'CCOCC':[0]},
        'evaluation_summary':{'learned':{},'split_sha256':sha(tmp_path/'split.json')}}
    return dump(tmp_path/'model.json',m)


def test_fine_model_correction_connected_before_positive_lookup_and_cache(tmp_path,parent,bank_artifact):
    m,graphs=bank_artifact;m['parent_model_sha256']=parent[1]
    m['training_bank']['split_sha256']=sha(tmp_path/'split.json')
    correction=dump(tmp_path/'correction.json',m)
    base=FineOdorModel(*parent);model=FineOdorModel(*parent,calibration=correction)
    p=model.predict(['CCOC']);original=base.predict(['CCOC'])
    assert not np.array_equal(p,original)
    model.annotations['CCOC']=[0]  # Learned-only predictions may not read query annotations.
    np.testing.assert_array_equal(model.predict(['CCOC']),p)
    from fragrance_ai.recommender.catalog import IngredientCatalog
    item=replace(IngredientCatalog.load_builtin().ingredients[0],structure_smiles='CCOC')
    material,_=model.materials([item]);repeat,_=model.materials([item])
    assert material[0,0]==1. and material[0,1]==p[0,1]
    np.testing.assert_array_equal(material,repeat)
    assert model.contract()['prediction_correction']['sha256']==correction[1]
    assert model.prediction_correction_sha256==correction[1]
    (tmp_path/'split.json').write_text('{}')
    with pytest.raises(ValueError,match='split changed'):model.assert_current()


def test_local_selection_keeps_base_and_corrected_instances_separate(tmp_path,parent,bank_artifact,monkeypatch):
    from fragrance_ai.recommender import fine_odor_model as fine
    m,_=bank_artifact;m['parent_model_sha256']=parent[1]
    m['training_bank']['split_sha256']=sha(tmp_path/'split.json')
    correction=dump(tmp_path/'correction.json',m)
    profile={'odor_expression':parent}
    monkeypatch.setattr('fragrance_ai.recommender.local_runtime.local_profile',lambda:dict(profile))
    monkeypatch.setattr('fragrance_ai.recommender.perception_runtime.configured_perception',lambda:None)
    before=fine.configured_fine_odor()
    profile['odor_calibration']=correction
    after=fine.configured_fine_odor()
    assert before is not after and before.calibration is None
    assert after.prediction_correction_sha256==correction[1]
    del profile['odor_calibration']
    assert fine.configured_fine_odor() is before


def test_validation_only_fitted_neighbor_weight_reduces_known_error():
    from scripts.calibrate_odor_expression_v62 import fit_correction,apply
    y=np.array([[1.,0.],[0.,1.]]*8);base=.2+.6*(1.-y);neighbor=.1+.8*y
    c=fit_correction(base,y,method='neighbors',neighbor=neighbor,neighbor_k=1)
    assert 0<c['neighbor_weight']<=.5
    assert np.mean((apply(base,c,neighbor)-y)**2)<np.mean((base-y)**2)
