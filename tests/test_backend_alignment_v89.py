import pytest
from pydantic import ValidationError
from fragrance_ai.platform.backend_wire_contract import FormulaGenerationResponse,normalize_formula_response
from fragrance_ai.platform.rd_evidence import EvidenceStore
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_rd_contract import data as data,client_for
from tests.test_release_target_v87 import client90 as client90


@pytest.mark.parametrize('value',[None,.87,0.,1.,'heuristic_only','insufficient_nonhuman_evidence'])
def test_confidence_boundary_is_always_nullable_numeric(value):
    original={'confidence':value,'calculated_profile_similarity':87.,'recipe':[]}
    result=normalize_formula_response(original)
    assert result['confidence'] is None or type(result['confidence']) is float
    assert result['calculated_profile_similarity']==87.
    assert original['confidence']==value
    FormulaGenerationResponse.model_validate(result)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),1.1,-.1])
def test_invalid_confidence_is_not_repaired_to_a_fake_number(value):
    with pytest.raises(ValueError):
        normalize_formula_response({'confidence':value})


def test_output_schema_rejects_string_even_if_internal_boundary_is_bypassed():
    with pytest.raises(ValidationError):
        FormulaGenerationResponse(confidence='heuristic_only',confidence_kind='heuristic_only')
    assert FormulaGenerationResponse.model_json_schema()['properties']['confidence']['anyOf'][1]['type']=='null'


def test_omitted_pool_matches_extended_example_without_relaxing_risk():
    request=LotionEstimateRequest(brief='woody scent')
    assert request.registry_pool=='conditional_research'
    assert request.max_risk_tier==1 and request.max_formula_cost_per_kg==180.
    assert 'registry_pool' not in request.model_fields_set
    assert LotionEstimateRequest(brief='woody scent',registry_pool='core').registry_pool=='core'
    assert LotionOptimizationRequest.model_fields['registry_pool'].default=='conditional_research'


def test_lotion_reports_effective_pool_and_per_request_defaulting(client90):
    for body,defaulted,pool in (({'brief':'woody scent'},True,'conditional_research'),
            ({'brief':'woody scent','registry_pool':'conditional_research'},False,'conditional_research'),
            ({'brief':'woody scent','registry_pool':'core'},False,'core')):
        response=client90.post('/v1/applications/body-lotion/design',json=body)
        assert response.status_code==200,response.text
        assert response.json()['registry_pool']==pool
        assert response.json()['registry_selection']['defaulted']==defaulted
        assert response.headers['X-Perfumery-Registry-Pool']==pool


def test_no_evidence_change_impact_is_422_abstained_not_success(data):
    catalog,_,_,assessment,_=data
    client,calls=client_for(catalog,EvidenceStore())
    with client:
        response=client.post('/v2/formulas/change-impact',json={**assessment,'previous_evidence_version':'unregistered'})
    assert response.status_code==422
    assert response.json()['detail']['status']=='abstained'
    assert response.json()['detail']['code']=='EVIDENCE_SNAPSHOTS_MISSING'
    assert response.json()['detail']['public_sources_registered'] is False
    assert calls==[]
