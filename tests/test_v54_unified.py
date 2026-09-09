"""Synthetic invariants only; real 382-case results live in the experiment."""
import hashlib
import json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai import StockAliquot, StockMass, StockControlAliquot, StockMixturePredictor
from fragrance_ai.research.mixture_profiles import mixture_features,fit_mixture_model,predict_mixture_model
from fragrance_ai.research.unified_mixture import atlas_mixture_features,fit_unified,predict_unified
from fragrance_ai.recommender.perception_guidance import PerceptionSearchSession
from tests.test_perception_guidance import material,provider,brief
from tests.test_mixture_core_v48 import KEYS,sdk_fixture


def training(d=3):
    p = np.array([np.linspace(.1,1.,d),np.linspace(1.,.1,d)])
    a = np.array([np.linspace(.1,2.,584),np.linspace(2.,.1,584)])
    x,z,y = [],[],[]
    for f in np.linspace(.1,.9,12):
        x.append(mixture_features(p,KEYS,[f,1-f]))
        z.append(atlas_mixture_features(a,['CCO','CO'],[.1,.01],['pg','dep'],[f,1-f]))
        y.append(x[-1][:d]+.2*f*(1-f))
    return np.array(x),np.array(z),np.array(y),p,a


def fit(x,z,y,**kwargs):
    return fit_unified(x,z,y,**{'alpha':.1,'bandwidth':4.,'scaling':'shared_sensory',
                              'atlas_weight':.5,'output_transform':'log1p',**kwargs})


@pytest.mark.parametrize('transform',['identity','sqrt','log1p'])
def test_single_stock_identity_trace_and_serialization(transform):
    x,z,y,p,a = training()
    model = fit(x,z,y,output_transform=transform)
    pure = mixture_features(p[:1],KEYS[:1])
    apure = atlas_mixture_features(a[:1],['CCO'],[.1],['pg'])
    np.testing.assert_array_equal(predict_unified(model,pure[None],apure[None])[0],p[0])
    trace = mixture_features(p,KEYS,[1.,1e-13])
    atrace = atlas_mixture_features(a,['CCO','CO'],[.1,.01],['pg','dep'],[1.,1e-13])
    np.testing.assert_allclose(predict_unified(model,trace[None],atrace[None])[0],p[0],atol=1e-9,rtol=0)
    np.testing.assert_array_equal(predict_unified(json.loads(json.dumps(model)),x,z),predict_unified(model,x,z))


def test_zero_atlas_identity_transform_reproduces_v48_exactly():
    x,z,y,_,_ = training()
    old = fit_mixture_model(x,y,.1,4.,'shared_sensory')
    new = fit(x,z,y,atlas_weight=0.,output_transform='identity')
    np.testing.assert_allclose(predict_unified(new,x,z),predict_mixture_model(old,x),atol=1e-12,rtol=0)


def test_atlas_features_keep_split_permutation_scaling_and_partial_coverage():
    _,_,_,_,p = training()
    p[1] = 0
    expected = atlas_mixture_features(p,['CCO','[Na+].[Cl-]'],[.1,.01],['pg','dep'],[2.,1.],supported=[True,False])
    actual = atlas_mixture_features(p[[1,0,0]],['[Na+].[Cl-]','CCO','CCO'],[.01,.1,.1],
                                    ['dep','pg','pg'],[1e100,1e100,1e100],supported=[False,True,True])
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=0)
    assert actual[-1] == pytest.approx(2/3)


@pytest.mark.parametrize('change',['nan','scale','width','atlas_width','weight','bandwidth','transform','count'])
def test_invalid_model_and_input_fail_closed(change):
    x,z,y,_,_ = training(); model = fit(x,z,y)
    if change == 'nan': model['coefficients'][0][0] = float('nan')
    if change == 'scale': model['atlas_scale'][0] = 0
    if change == 'width': x = x[:,:-1]
    if change == 'atlas_width': z = z[:,:-1]
    if change == 'weight': model['atlas_weight'] = 1.1
    if change == 'bandwidth': model['bandwidth'] = -1
    if change == 'transform': model['output_transform'] = 'unknown'
    if change == 'count': x[:,-4] = 0
    with pytest.raises(ValueError): predict_unified(model,x,z)


def integrated_fixture(monkeypatch,tmp_path):
    legacy,p,_,_,calls,_,_ = sdk_fixture(monkeypatch,tmp_path)
    x,z,y,profiles,embeddings = training(len(p.endpoints))
    def stocks(self,ingredients,dilutions,solvents):
        return np.asarray([profiles[0 if i.ingredient_id == 'a' else 1] for i in ingredients]),[{'basis':'synthetic'}]*len(ingredients),[
            (i.ingredient_id,str(d),s) for i,d,s in zip(ingredients,dilutions,solvents)]
    monkeypatch.setattr(PerceptionSearchSession,'predict_stock_conditions',stocks)
    monkeypatch.setattr(PerceptionSearchSession,'prepare_stock_molecules',lambda self,items:[
        (None,{'canonical_smiles':'CCO' if i.ingredient_id == 'a' else 'CO'}) for i in items])
    atlas_calls = []
    def predict(graphs,reference_level):
        atlas_calls.append((tuple(graphs),reference_level))
        values = np.asarray([embeddings[0 if g == 'CCO' else 1] for g in graphs])
        offset = 0 if reference_level == 'high' else 292
        return {'applicability':values[:,offset:offset+146], 'use':values[:,offset+146:offset+292]}
    atlas = SimpleNamespace(sha256='a'*64,endpoints=list(range(146)),assert_current=lambda:None,predict=predict)
    artifact = {'schema':'v54-integrated-candidate/v1','model':fit(x,z,y),'endpoints':list(p.endpoints),
        'component_sha256':p.component_model_sha256,'atlas_sha256':atlas.sha256,'local_development_accepted':True,
        'runtime_promotion_allowed':False,'data_redistribution_authorized':False,'lotion_headspace_calibrated':False,
        'application_domain':'relative_aliquots_of_explicit_stock_conditions'}
    path = tmp_path/'integrated.json'; path.write_text(json.dumps(artifact),encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sdk = StockMixturePredictor(p,path,sha256=digest,experimental=True,atlas_predictor=atlas)
    return sdk,p,atlas,path,digest,atlas_calls


def test_sdk_features_match_training_and_mass_split_inputs(monkeypatch,tmp_path):
    sdk,*_ = integrated_fixture(monkeypatch,tmp_path)
    a,b = material('a'),material('b')
    result = sdk.predict([StockAliquot(a,.1,2.,'pg'),StockAliquot(b,.01,1.,'dep')])
    _,_,_,p,embeddings = training(len(sdk.provider.endpoints))
    x = mixture_features(p,[('a','0.1','pg'),('b','0.01','dep')],[2.,1.])
    z = atlas_mixture_features(embeddings,['CCO','CO'],[.1,.01],['pg','dep'],[2.,1.])
    expected = predict_unified(sdk.model,x[None],z[None])[0]
    np.testing.assert_allclose(list(result['predicted_rata_profile'].values()),expected,atol=1e-12,rtol=0)
    split = sdk.predict([StockAliquot(b,.01,1.,'dep'),StockAliquot(a,.1,1.,'pg'),StockAliquot(a,.1,1.,'pg')])
    mass = sdk.predict_masses([StockMass(a,.1,4.,2.,'pg'),StockMass(b,.01,1.,1.,'dep')])
    for other in (split,mass):
        np.testing.assert_allclose(list(other['predicted_rata_profile'].values()),expected,atol=1e-12,rtol=0)
    assert result['integrated_v54'] and result['atlas_supported_volume_fraction'] == 1.
    assert result['atlas_forward_batches'] == 2 and split['atlas_forward_batches'] == 0
    assert not result['recipe_acceptance_modified'] and result['human_similarity_percent'] is None


def test_concurrent_cache_and_original_v54_molecular_outputs(monkeypatch,tmp_path):
    sdk,_,atlas,_,_,calls = integrated_fixture(monkeypatch,tmp_path)
    stocks = [StockAliquot(material('a'),.1,1.,'pg'),StockAliquot(material('b'),.01,1.,'dep')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _:sdk.predict(stocks),range(8)))
    assert len(calls) == 2
    assert all(r['predicted_rata_profile'] == results[0]['predicted_rata_profile'] for r in results)
    for level in ('high','low'):
        expected = atlas.predict(['CCO'],reference_level=level)
        actual = sdk.predict_molecular(['CCO'],reference_level=level)
        for name in expected: np.testing.assert_array_equal(actual[name],expected[name])


def test_parent_mismatch_optin_and_checkpoint_drift(monkeypatch,tmp_path):
    sdk,p,atlas,path,digest,_ = integrated_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='parent'):
        StockMixturePredictor(p,path,sha256=digest,experimental=True)
    atlas.sha256 = 'b'*64
    with pytest.raises(ValueError,match='parent'):
        StockMixturePredictor(p,path,sha256=digest,experimental=True,atlas_predictor=atlas)
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'): sdk.contract()


def test_conflicting_materials_at_different_doses_are_not_cached_as_one(monkeypatch,tmp_path):
    sdk,*_ = integrated_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='conflicting'):
        sdk.predict([StockAliquot(material('a'),.1,1.,'pg'),StockAliquot(replace(material('a'),cas_number='wrong'),.01,1.,'pg')])


def test_known_multigraph_stock_is_not_silently_decomposed():
    p = provider({'a':('[Na+].[Cl-]',None),'b':('CC.CCCC',None)})
    p.by_structure = {'[Na+].[Cl-]':'123'}
    p.bank = {'123':{'canonical_smiles':'[Na+].[Cl-]'}}
    session = PerceptionSearchSession(p,brief())
    assert session.prepare_stock_molecules([material('a')]) == [('123',p.bank['123'])]
    assert not session.supports(material('a'))
    with pytest.raises(ValueError,match='unsupported'): session.prepare_stock_molecules([material('b')])


def test_exact_study_control_uses_anchor_without_inventing_a_graph(monkeypatch,tmp_path):
    sdk,p,_,_,_,calls = integrated_fixture(monkeypatch,tmp_path)
    p.model['anchors'] = [{'key':['-1','1','nt'],'profile':[.2]*len(p.endpoints)}]
    control = StockControlAliquot('-1',1.,1.,'nt')
    result = sdk.predict([],controls=[control])
    np.testing.assert_array_equal(list(result['predicted_rata_profile'].values()),[.2]*len(p.endpoints))
    assert not calls and result['atlas_supported_volume_fraction'] == 0
    assert result['study_control_basis'][0]['study_control_id'] == '-1'
    mix = sdk.predict([StockAliquot(material('a'),.1,1.,'pg')],controls=[control])
    split = sdk.predict([StockAliquot(material('a'),.1,1.,'pg')],
        controls=[replace(control,relative_volume=.5),replace(control,relative_volume=.5)])
    assert mix['predicted_rata_profile'] == split['predicted_rata_profile']
    assert mix['atlas_supported_volume_fraction'] == .5
    for bad in (replace(control,stock_dilution=.5),replace(control,study_control_id='-2'),replace(control,solvent='pg')):
        with pytest.raises(ValueError,match='exact'): sdk.predict([],controls=[bad])
