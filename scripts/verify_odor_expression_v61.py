"""Actual pinned local API + full-catalog preference checks; no deployment."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from fragrance_ai.recommender.fine_odor_model import configured_fine_odor
    from fragrance_ai.recommender.odor_expression import expression_contract, expression_utility
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.safety import CandidateSafetyScreen
    from fragrance_ai.recommender.global_profile_search import optimize_full_pool
    from fragrance_ai.recommender.models import RecipeConstraints
    import numpy as np
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output
    if out.exists():
        raise ValueError('new output directory required')
    out.mkdir(parents=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    profile=local_profile()
    if profile is None or 'odor_expression' not in profile:
        raise ValueError('real pinned expression model required')
    catalog,_=load_configured_catalog()
    model=configured_fine_odor()
    report={'scope':'actual_local_models_with_functional_process_fixtures_not_measured_product_quality',
        'profile_sha256':profile['profile_sha256'],'expression':expression_contract(),'model':model.contract(),
        'checks':{}}
    # Not a safe-material activation test. Only already eligible materials are
    # ranked, and every cost/cap/note constraint stays in the optimizer.
    parser=NaturalLanguageBriefParser(catalog)
    constraints=RecipeConstraints(enable_registry_trace_candidates=True,max_risk_tier=2)
    base=parser.parse('green apple scent',constraints)
    pool,rejected=CandidateSafetyScreen().screen(catalog,base)
    tick=time.perf_counter()
    values,evidence=model.materials(pool)
    report['material_coverage']={'eligible':len(pool),'predicted':sum(r['status']!='missing_or_multicomponent_structure' for r in evidence),
        'source_supported':sum(r.get('source_positive_labels',0)>0 for r in evidence),
        'cold_batch_seconds':time.perf_counter()-tick}
    candidates={}
    for text in ('green apple scent','peach scent'):
        brief=parser.parse(text,constraints)
        brief=replace(brief,target_profile={'fruity':1.},desired_dimensions=['fruity'])
        utility,diagnostic=expression_utility(pool,brief)
        solution=optimize_full_pool(pool,brief)
        candidates[text]=solution.weights_percent
        save(text.replace(' ','-')+'.json',{'weights':solution.weights_percent,'status':solution.status,
            'original_coarse_lp_score':solution.relaxed_overlap_score,'fine_preference':diagnostic,
            'top_ranked_ids':[pool[i].ingredient_id for i in np.argsort(-utility)[:12]],
            'scope':'full_catalog_search_connection_not_full_recipe_acceptance_or_human_test'})
    report['checks']['full_pool_fine_requests_change_composition']=all(candidates.values()) and candidates['green apple scent']!=candidates['peach scent']
    app=create_app(enable_language=False)
    with TestClient(app) as client:
        def post(path,payload):
            start=time.perf_counter()
            response=client.post(path,json=payload)
            elapsed=time.perf_counter()-start
            if response.status_code!=200:
                raise AssertionError((path,response.status_code,response.text[:1200]))
            return response.json(),elapsed
        cap=client.get('/v1/ai/capabilities')
        assert cap.status_code==200
        cap=cap.json();save('capabilities.json',cap)
        assert cap['odor_expression']['prediction_model']['model_sha256']==profile['odor_expression'][1]
        prepare,_=post('/v1/briefs/prepare',{'formula':{'brief':'피오니와 청사과, 코코넛은 제외'}})
        intent=prepare['intent']['representation']['fine_expression']
        assert set(intent['wanted'])=={'peony','greenapple'} and set(intent['avoided'])=={'coconut'}
        save('korean-prepare.json',prepare)
        report['checks']['existing_intent_api_preserves_fine_identity']=True
        ids=['dihydromyrcenol','linalyl_acetate','hexyl_acetate']
        direct,seconds=post('/v1/odor-expressions/predict',{'ingredient_ids':ids,'brief':'green apple and peach'})
        save('material-expression.json',direct)
        report['checks']['material_prediction_seconds']=seconds
        report['checks']['material_prediction_available']=all(x['status']=='annotation_profile_proxy' for x in direct['materials'])
        for product in ('perfume','body_lotion','body_wash'):
            payload=json.loads((ROOT/f'.benchmarks/unified_product_v60/api-01/{product}-request.json').read_text(encoding='utf-8'))
            value,seconds=post('/v1/applications/unified/predict',payload)
            save(product+'-response.json',value)
            assert len(value['odor_endpoints'])==146
            assert value['model']['fine_odor_expression']['model_sha256']==profile['odor_expression'][1]
            positive=[x for x in value['temporal_profile'] if x['total_air_concentration_mg_m3']>0]
            assert all(x['fine_odor_expression']['status']=='annotation_profile_proxy' for x in positive)
            assert all(x['fine_odor_expression']['human_similarity_percent'] is None for x in positive)
            repeat,_=post('/v1/applications/unified/predict',payload)
            assert repeat==value
            old=json.loads((ROOT/f'.benchmarks/unified_product_v60/api-01/{product}-response.json').read_text(encoding='utf-8'))
            for a,b in zip(old['temporal_profile'],value['temporal_profile']):
                np.testing.assert_allclose(a['total_air_concentration_mg_m3'],b['total_air_concentration_mg_m3'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(a['full_odor_reference_profiles']['applicability'] if a['full_odor_reference_profiles'] else [],
                    b['full_odor_reference_profiles']['applicability'] if b['full_odor_reference_profiles'] else [],atol=1e-12)
            report['checks'][product]={'fine_head_connected':True,'original_transport_and_146_axes_unchanged':True,'seconds':seconds}
        result,seconds=post('/v1/formulas',{'brief':'피오니와 청사과 향','enable_registry_trace_candidates':True,'max_risk_tier':2})
        save('recipe-response.json',result)
        report['checks']['actual_recipe_api']={'seconds':seconds,'status':result['status'],
            'fine_expression_status':result['score_contract']['fine_odor_expression']['status'],
            'coarse_score':result.get('calculated_profile_similarity'),'full_profile_target_met':result.get('full_profile_target_met'),
            'candidate_lines':len(result.get('recipe') or result.get('closest_candidate') or [])}
    assert report['checks']['material_prediction_available']
    assert report['checks']['full_pool_fine_requests_change_composition']
    report['status']='passed'
    save('report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
