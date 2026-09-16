"""Bounded API checks; synthetic wash coefficients remain explicitly labelled."""
import hashlib
import time


def run_checks(client, expected_wheel):
    calls=[]
    def call(name,method,path,request=None,expected=200,active_client=None):
        selected=client if active_client is None else active_client
        start=time.perf_counter()
        response=selected.request(method,path,json=request) if request is not None else selected.request(method,path)
        calls.append({'name':name,'method':method,'path':path,'request':request,
            'verification_scope':'deployed_image_app' if active_client is None else 'isolated_empty_evidence_app',
            'http_status':response.status_code,'seconds':time.perf_counter()-start,
            'response_headers':dict(response.headers),'response_body':response.text,
            'response_sha256':hashlib.sha256(response.content).hexdigest()})
        if response.status_code!=expected:
            raise ValueError(name+': HTTP '+str(response.status_code)+' '+response.text[:500])
        return response.json()
    health=call('health','GET','/health')
    assert health['wheel_sha256']==expected_wheel
    schema=call('openapi','GET','/openapi.json')
    assert 'confidence' in schema['components']['schemas']['FormulaGenerationResponse']['properties']
    versions=call('evidence_versions','GET','/v2/evidence/versions')
    coefficients=call('coefficient_contract','GET','/v1/applications/unified/coefficients/contract')
    assert not coefficients['measured_reference_table_available']
    language=call('odor_dictionary','GET','/v1/ai/odor-language?limit=1')
    assert language['version']=='compositional-odor-language/v89'
    call('poetic_intent','POST','/v1/ai/odor-language/interpret',{'text':'sun dried linen, without musk'})
    impact_request={
        'lines':[{'ingredient_id':'linalyl_acetate','concentrate_percent':100.}],
        'target_region':'EU','product_category':'eau_de_parfum','product_concentration_percent':15.,
        'max_formula_cost_per_kg':180.,'policy':{'finished_batch_mass_g':1000.,
            'maximum_lead_time_days':10,'maximum_purchase_cost_usd':100.},
        'previous_evidence_version':versions['previous_versions'][-1]}
    impact=call('change_impact','POST','/v2/formulas/change-impact',impact_request)
    assert impact['diagnostic_only'] and impact['comparison_kind']=='public_sources'
    # Exercise absence without removing or modifying the production store.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from fragrance_ai.platform.rd_api import register_rd_api
    from fragrance_ai.platform.rd_evidence import EvidenceStore
    from fragrance_ai.recommender.catalog import IngredientCatalog
    class Envelope(BaseModel):
        formula: dict
    empty=FastAPI()
    register_rd_api(empty,Envelope,lambda *_:{},lambda *_:{},IngredientCatalog.load_builtin(),
        lambda:None,lambda:{},evidence_store=EvidenceStore())
    with TestClient(empty) as empty_client:
        missing=call('change_impact_no_evidence','POST','/v2/formulas/change-impact',
                     impact_request,expected=422,active_client=empty_client)
    assert missing['detail']['status']=='abstained'
    assert missing['detail']['code']=='EVIDENCE_SNAPSHOTS_MISSING'
    context={'product_type':'body_wash','application_mass_mg_cm2':2.,
        'fragrance_concentration_percent':.5,'temperature_c':25.,'relative_humidity_percent':50.,
        'headspace_height_cm':1.,'stages':[{'stage_id':'initial','duration_minutes':3.,
            'formulation_reference':'synthetic software verification, not measured product'}]}
    prepared=call('body_wash_context','POST','/v1/applications/unified/context',context)
    ids=['registry_d9bd03cc379a5ac3d428846f','registry_05259f9860bf4dd95648459d']
    kinetics=[{'ingredient_id':key,'evaporation_per_min':rate,'uptake_per_min':.01,
        'hydrolysis_per_min':.02,'air_return_per_min':.1,'ventilation_per_min':.2,
        'capacity_decay_per_min':.03,'evaporating_capacity_fraction':.5,'nonreactive_capacity_fraction':.2,
        'source_kind':'simulated','source_reference':'synthetic software verification'}
        for key,rate in zip(ids,(.3,.02))]
    wash=call('body_wash_predict','POST','/v1/applications/unified/predict',{
        'context':context,'parameter_context_id':prepared['parameter_context_id'],
        'components':[{'ingredient_id':key,'concentrate_percent':50.} for key in ids],
        'stages':[{'stage_id':'initial','coefficients':kinetics,
            'rinse_retained_film_fractions':dict.fromkeys(ids,.1),'rinse_source_reference':'synthetic rinse'}],
        'times_minutes':[0.,.1,1.,2.,3.]})
    assert wash['diagnostics']['nonnegative']
    assert wash['diagnostics']['mass_balance_max_abs_error_mg_cm2']<1e-8
    lotion=call('body_lotion_design','POST','/v1/applications/body-lotion/design',
        {'brief':'woody scent','max_risk_tier':2})
    assert lotion['registry_pool']=='conditional_research' and lotion['registry_selection']['defaulted']
    assert lotion['profile_target_met'] and lotion['recipe']
    assert abs(sum(row['concentrate_percent'] for row in lotion['recipe'])-100)<.002
    explicit=call('body_lotion_explicit_pool','POST','/v1/applications/body-lotion/design',
        {'brief':'woody scent','max_risk_tier':2,'registry_pool':'conditional_research'})
    assert not explicit['registry_selection']['defaulted']
    assert explicit['recipe']==lotion['recipe'] and explicit['score']==lotion['score']
    request={'brief':'woody scent','max_risk_tier':2,'enable_registry_trace_candidates':True}
    perfume=call('perfume_design','POST','/v1/formulas',request)
    assert 'confidence' in perfume and (perfume['confidence'] is None or type(perfume['confidence']) in (int,float))
    assert perfume['score_contract']['effective_target']==90.
    assert calls[-1]['response_headers']['x-perfumery-confidence-contract']=='nullable-number-v1'
    response=client.post('/v1/formulas/stream',json=request)
    from deploy.verify_supported_api_v80 import verify_stream
    assert response.status_code==200
    stream=verify_stream(response.text,perfume)
    calls.append({'name':'formula_stream','method':'POST','path':'/v1/formulas/stream','request':request,
        'http_status':response.status_code,'response_headers':dict(response.headers),
        'response_body':response.text,'response_sha256':hashlib.sha256(response.content).hexdigest()})
    return {'passed':True,'calls':calls,'formula_stream':stream,
        'lotion_score':lotion['score'],'lotion_ingredient_count':len(lotion['recipe']),
        'perfume_score':perfume.get('calculated_profile_similarity'),
        'confidence_kind':perfume.get('confidence_kind'),
        'confidence_wire_type':'number_or_null','lotion_default_pool':lotion['registry_pool'],
        'lotion_omitted_equals_explicit_pool':True,'no_evidence_http_status':422,
        'no_evidence_test_scope':'isolated_empty_evidence_app_in_same_image',
        'body_wash_real_coefficients_verified':False,
        'scope':'image_API_contract_verification_not_public_backend_token_or_human_accuracy'}
