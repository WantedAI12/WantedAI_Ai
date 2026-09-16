from fragrance_ai.platform.coefficient_contract import coefficient_contract
from tests.test_rd_contract import data, client_for


def test_coefficient_units_do_not_invent_measured_defaults():
    value=coefficient_contract()
    assert value['source_kind_values']==['measured','estimated','simulated']
    assert value['fields']['evaporation_per_min']['unit']=='1/min'
    assert value['fields']['evaporating_capacity_fraction']['unit']=='fraction_0_1'
    assert not value['coefficient_sources_verified']
    assert not value['measured_reference_table_available']
    assert all(row['default_value'] is None for row in value['fields'].values())


def test_version_listing_and_openapi_success_shape(data):
    catalog,store,bundle,assessment,_=data
    with client_for(catalog,store())[0] as client:
        versions=client.get('/v2/evidence/versions')
        assert versions.status_code==200
        assert versions.json()['previous_versions']==['previous']
        result=client.post('/v2/formulas/change-impact',json={**assessment,'previous_evidence_version':'previous'})
        assert result.status_code==200,result.text
        value=result.json()
        assert value['comparison_kind']=='operator_evidence' and not value['diagnostic_only']
        assert not value['manufacturing_approval']
        schema=client.get('/openapi.json').json()
        response=schema['paths']['/v2/formulas/change-impact']['post']['responses']['200']['content']['application/json']['schema']
        assert response['$ref'].endswith('ChangeImpactResponse')
