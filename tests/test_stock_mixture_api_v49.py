from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from fragrance_ai import StockAliquot,StockMass
from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_ai_extensions import Formula
from tests.test_mixture_core_v48 import sdk_fixture
from tests.test_perception_guidance import material


def test_stock_mass_conversion_uses_density_once_and_does_not_redilute(monkeypatch,tmp_path):
    sdk,*_ = sdk_fixture(monkeypatch,tmp_path)
    a,b = material('a'),material('b')
    by_mass = sdk.predict_masses([StockMass(a,.1,2.,2.,'pg'),StockMass(b,.01,1.,1.,'dep')])
    by_volume = sdk.predict([StockAliquot(a,.1,1.,'pg'),StockAliquot(b,.01,1.,'dep')])
    np.testing.assert_array_equal(list(by_mass['predicted_rata_profile'].values()),list(by_volume['predicted_rata_profile'].values()))
    assert by_mass['mass_balance']['active_mass_fraction'] == pytest.approx(.07)
    assert [r['aliquot_volume_fraction'] for r in by_mass['mass_balance']['components']] == [.5,.5]
    assert not by_mass['mass_balance']['density_measured_verified']


def test_mass_conversion_is_scale_invariant_and_no_density_is_fabricated(monkeypatch,tmp_path):
    sdk,*_ = sdk_fixture(monkeypatch,tmp_path)
    row = StockMass(material('a'),.1,1e308,1.,'pg')
    big = sdk.predict_masses([row,replace(row,ingredient=material('b'))])
    small = sdk.predict_masses([replace(row,supplied_mass_g=1.),replace(row,ingredient=material('b'),supplied_mass_g=1.)])
    assert big['predicted_rata_profile'] == small['predicted_rata_profile']
    for density in (None,True,float('nan'),0.,'1.0'):
        with pytest.raises(ValueError): sdk.predict_masses([replace(row,stock_density_g_ml=density)])


def stock_body():
    return {'basis':'supplied_mass','components':[
        {'ingredient_id':'a','stock_dilution':.1,'supplied_mass_g':2.,'stock_density_g_ml':2.,'solvent':'pg'},
        {'ingredient_id':'b','stock_dilution':.01,'supplied_mass_g':1.,'stock_density_g_ml':1.,'solvent':'dep'}]}


def make_client(predictor,provider):
    app = FastAPI()
    catalog = IngredientCatalog([material('a'),material('b')])
    def no_generation(*args,**kwargs): pytest.fail('stock evaluation must not rerun recipe generation')
    register_ai_extensions(app,Formula,catalog,no_generation,lambda:None,
        perception_guidance=provider,stock_mixture_predictor=predictor)
    return TestClient(app)


def test_stock_api_cached_forward_and_existing_capabilities(monkeypatch,tmp_path):
    sdk,p,_,_,calls,*_ = sdk_fixture(monkeypatch,tmp_path)
    with make_client(sdk,p) as client:
        left = client.post('/v1/formulations/stock-mixture/predict',json=stock_body())
        assert left.status_code == 200,left.text
        right = client.post('/v1/formulations/stock-mixture/predict',json=stock_body())
        assert left.json() == right.json()
        assert right.headers['X-Perfumery-Stock-Cache'] == 'hit'
        assert calls == [2]
        capabilities = client.get('/v1/ai/capabilities').json()['features']
        assert capabilities['explicit_stock_mixture_prediction'] and capabilities['body_lotion_research_transport_simulation']
        assert not left.json()['recipe_acceptance_modified']


@pytest.mark.parametrize('change',['missing_density','extra_volume','unknown_material','lotion','model_path','bool_mass'])
def test_stock_api_rejects_ambiguous_basis_and_domain(monkeypatch,tmp_path,change):
    sdk,p,*_ = sdk_fixture(monkeypatch,tmp_path)
    body = stock_body()
    if change == 'missing_density': del body['components'][0]['stock_density_g_ml']
    if change == 'extra_volume': body['components'][0]['relative_volume'] = 1.
    if change == 'unknown_material': body['components'][0]['ingredient_id'] = 'absent'
    if change == 'lotion': body['application_domain'] = 'body_lotion'
    if change == 'model_path': body['model_path'] = 'untrusted.json'
    if change == 'bool_mass': body['components'][0]['supplied_mass_g'] = True
    with make_client(sdk,p) as client:
        result = client.post('/v1/formulations/stock-mixture/predict',json=body)
        assert result.status_code == 422,result.text


def test_unconfigured_stock_lane_does_not_pretend_to_predict():
    with make_client(None,None) as client:
        result = client.post('/v1/formulations/stock-mixture/predict',json=stock_body())
        assert result.status_code == 503
        assert not client.get('/v1/ai/capabilities').json()['features']['explicit_stock_mixture_prediction']
