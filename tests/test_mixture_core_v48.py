import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from fragrance_ai import StockAliquot,StockMixturePredictor
from fragrance_ai.recommender.perception_guidance import PerceptionSearchSession
from fragrance_ai.research.mixture_profiles import mixture_features,fit_mixture_model,predict_mixture_model
from tests.test_perception_guidance import material,provider


KEYS = [('a','.1','pg'),('b','.01','dep')]
PROFILES = np.array([[1.,2.,.5],[3.,.1,1.]])


def test_features_preserve_permutation_volume_scaling_and_aliquot_splitting():
    expected = mixture_features(PROFILES,KEYS,[2.,1.])
    np.testing.assert_allclose(mixture_features(PROFILES[::-1],KEYS[::-1],[1.,2.]),expected,atol=1e-14)
    np.testing.assert_allclose(mixture_features(PROFILES,KEYS,[2e100,1e100]),expected,atol=1e-14)
    np.testing.assert_allclose(mixture_features(PROFILES[[0,0,1]],[KEYS[0],KEYS[0],KEYS[1]],[1.,1.,1.]),expected,atol=1e-14)


def test_extreme_relative_volumes_do_not_overflow():
    np.testing.assert_array_equal(mixture_features(PROFILES,KEYS,[1e308,1e308]),mixture_features(PROFILES,KEYS))


def test_trace_stock_and_zero_weight_do_not_receive_unweighted_maximum_credit():
    pure = mixture_features(PROFILES[:1],KEYS[:1])
    np.testing.assert_array_equal(mixture_features(PROFILES,KEYS,[1.,0.]),pure)
    np.testing.assert_allclose(mixture_features(PROFILES,KEYS,[1.,1e-14]),pure,atol=1e-6,rtol=0)


@pytest.mark.parametrize('weights',[[-1.,2.],[0.,0.],[1.,float('nan')],[1.]])
def test_invalid_relative_weights_fail(weights):
    with pytest.raises(ValueError): mixture_features(PROFILES,KEYS,weights)


def test_conflicting_duplicate_stock_profiles_fail():
    with pytest.raises(ValueError,match='conflicting'):
        mixture_features(PROFILES,[KEYS[0],KEYS[0]])


def example_model(width=3):
    a,b = np.linspace(.1,1.,width),np.linspace(1.,.1,width)
    features,targets = [],[]
    for fraction in np.linspace(.1,.9,12):
        feature = mixture_features(np.array([a,b]),KEYS,[fraction,1-fraction])
        features.append(feature)
        targets.append(feature[:width]+.2*fraction*(1-fraction))
    return fit_mixture_model(features,targets,.1,1.,'shared_sensory'),a,b


def test_single_stock_identity_and_trace_continuity_are_model_constraints():
    model,a,b = example_model()
    pure = mixture_features(a[None,:],KEYS[:1])
    np.testing.assert_array_equal(predict_mixture_model(model,pure[None,:])[0],a)
    trace = mixture_features(np.array([a,b]),KEYS,[1.,1e-12])
    np.testing.assert_allclose(predict_mixture_model(model,trace[None,:])[0],a,atol=1e-8,rtol=0)


def test_portable_model_and_feature_contracts_fail_closed():
    model,_,_ = example_model()
    x = np.asarray(model['center'])[None,:]
    restored = json.loads(json.dumps(model))
    np.testing.assert_array_equal(predict_mixture_model(model,x),predict_mixture_model(restored,x))
    restored['scale'][0] = 0.
    with pytest.raises(ValueError): predict_mixture_model(restored,x)
    with pytest.raises(ValueError): predict_mixture_model(model,x[:,:-1])


def sdk_fixture(monkeypatch,tmp_path):
    p = provider(); p.component_model_sha256 = '0'*64
    model,a,b = example_model(len(p.endpoints))
    artifact = {'schema':'stock-mixture-candidate/v1','model':model,'endpoints':list(p.endpoints),
        'component_sha256':p.component_model_sha256,'runtime_promotion_allowed':False,
        'data_redistribution_authorized':False,'lotion_headspace_calibrated':False,
        'application_domain':'relative_aliquots_of_explicit_stock_conditions'}
    path = tmp_path/'model.json'; path.write_text(json.dumps(artifact),encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    def paired(self,ingredients,dilutions,solvents):
        calls.append(len(ingredients)); self.stock_batch_forward_calls = getattr(self,'stock_batch_forward_calls',0)+1
        return np.array([a if i.ingredient_id == 'a' else b for i in ingredients]),[{'basis':'synthetic_test'}]*len(ingredients),[
            (i.ingredient_id,str(d),s) for i,d,s in zip(ingredients,dilutions,solvents)]
    monkeypatch.setattr(PerceptionSearchSession,'predict_stock_conditions',paired)
    sdk = StockMixturePredictor(p,path,sha256=digest,experimental=True)
    return sdk,p,path,digest,calls,a,b


def test_sdk_reuses_duplicates_without_changing_prediction(monkeypatch,tmp_path):
    sdk,_,_,_,calls,_,_ = sdk_fixture(monkeypatch,tmp_path)
    a,b = StockAliquot(material('a'),.1,1.,'pg'),StockAliquot(material('b'),.01,1.,'dep')
    expected = sdk.predict([a,b])
    changed = sdk.predict([b,replace(a,relative_volume=.5),replace(a,relative_volume=.5)])
    assert calls == [2,2]
    assert expected['effective_stock_count'] == changed['effective_stock_count'] == 2.
    np.testing.assert_allclose(list(expected['predicted_rata_profile'].values()),list(changed['predicted_rata_profile'].values()),atol=1e-12)
    assert expected['human_similarity_percent'] is None and not expected['recipe_acceptance_modified']


def test_sdk_batches_all_requested_stocks_without_cross_product_grid(monkeypatch,tmp_path):
    sdk,_,_,_,calls,_,_ = sdk_fixture(monkeypatch,tmp_path)
    result = sdk.predict([StockAliquot(material(str(i)),.1,1.,'pg') for i in range(257)])
    assert calls == [128,128,1]
    assert result['unique_input_stock_count'] == 257
    assert result['component_forward_batches'] == 3


def test_sdk_rejects_checkpoint_drift_and_wrong_physical_domain(monkeypatch,tmp_path):
    sdk,p,path,digest,_,_,_ = sdk_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='opt-in'): StockMixturePredictor(p,path,sha256=digest)
    with pytest.raises(ValueError,match='hash'): StockMixturePredictor(p,path,sha256='1'*64,experimental=True)
    with pytest.raises(ValueError,match='not lotion'): sdk.predict([],application='body_lotion')
    p.component_model_sha256 = '2'*64
    with pytest.raises(ValueError,match='changed'): sdk.predict([])


@pytest.mark.parametrize('dose,volume,solvent',[(True,1.,'pg'),(.1,True,'pg'),(.1,-1.,'pg'),(.1,1.,'other')])
def test_sdk_rejects_invalid_aliquots(monkeypatch,tmp_path,dose,volume,solvent):
    sdk,*_ = sdk_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError): sdk.predict([StockAliquot(material(),dose,volume,solvent)])


def test_paired_stock_batch_keeps_unknown_graphs_out_of_anchor_lookup(monkeypatch):
    from fragrance_ai.research import conditional_profiles,kernel_profiles
    from fragrance_ai.research.conditional_profiles import molecule_features
    from tests.test_perception_guidance import brief
    p = provider()
    p.model = {'regressor': {}}
    a = molecule_features('CCO'); b = molecule_features('CO')
    session = PerceptionSearchSession(p,brief())
    monkeypatch.setattr(session,'_prepare',lambda item: ('123',a) if item.ingredient_id == 'a' else (None,b))
    anchored_keys = []
    def anchored(model,x,keys):
        anchored_keys.extend(keys)
        return np.ones((len(keys),len(p.endpoints))),[{'basis':'synthetic_anchor'}]*len(keys)
    monkeypatch.setattr(conditional_profiles,'predict_conditional',anchored)
    monkeypatch.setattr(kernel_profiles,'predict_component_regressor',
        lambda model,x: np.repeat((10+x[:,1095])[:,None],len(p.endpoints),axis=1))
    # Unknown comes first to exercise output reassembly after grouped forwards.
    values,details,keys = session.predict_stock_conditions([material('b'),material('a')],[.01,.1],['dep','pg'])
    assert anchored_keys == [('123','0.1','pg')]
    assert keys == [('smiles:CO','0.01','dep'),('123','0.1','pg')]
    assert values[:,0].tolist() == [8.,1.]
    assert details[0]['basis'] == 'new_registry_graph_prediction'
    assert session.stock_batch_forward_calls == 2


def test_paired_stock_batch_rejects_missing_and_misaligned_inputs():
    from tests.test_perception_guidance import brief
    session = PerceptionSearchSession(provider(),brief())
    with pytest.raises(ValueError): session.predict_stock_conditions([material()],[.1],[])
    with pytest.raises(ValueError): session.predict_stock_conditions([material()],[.1],['pg'])
