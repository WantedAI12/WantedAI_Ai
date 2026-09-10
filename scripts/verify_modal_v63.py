"""Authenticated release check; temporary credentials never reach logs/files."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.runtime_release_v63 import WHEEL_SHA256,RELEASE_ID


def verify_assistant_result(value):
    """Require real model execution AND the correct exclusion/product contract."""
    assert value['source'] in ('quantized_language_model','deterministic_grounding_after_model_mismatch')
    proposal=value['intent_proposal']
    assert 'rose' in proposal['desired'] and 'musky' not in proposal['desired']
    assert 'musky' in proposal['avoided'] and proposal['product']=='body_lotion'
    assert value['requires_confirmation'] is True
    assert value['formula_generated'] is False and value['scientific_score_generated'] is False
    if value['source']=='deterministic_grounding_after_model_mismatch':
        grounding=value['grounding']
        assert grounding['status']=='explicit_parser_recovery'
        original=grounding['original_model_proposal']
        assert 'rose' in original['desired'] and 'musky' in original['avoided']
        assert original['product']=='body_lotion'
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assistant-only',action='store_true',help='after a worker-only change; retain prior core evidence')
    p.add_argument('--require-legacy-exact',action='store_true',help='also require identical V62 search traces and model floating-point diagnostics')
    args=p.parse_args();out=args.output
    if out.exists():raise ValueError('new verification directory required')
    out.mkdir(parents=True)
    base='https://junseong2im--perfumery-ai-core-web.modal.run'
    report={'release_id':RELEASE_ID,'base_url':base,'checks':{},'seconds':{},
        'temporary_token_revoked':False,'existing_backend_credentials_modified':False,
        'scope':'remote_authenticated_software_integration_not_human_accuracy','assistant_only':args.assistant_only}
    token_id=None;stage='create_verification_credential'
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    try:
        created=subprocess.run([sys.executable,'-m','modal','workspace','proxy-tokens','create','--json'],
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45)
        identifier=re.search(r'wk-[A-Za-z0-9]+',created.stdout)
        secret=re.search(r'ws-[A-Za-z0-9]+',created.stdout)
        token_id=identifier.group() if identifier else None
        if created.returncode or token_id is None or secret is None:
            raise RuntimeError('verification credential unavailable; raw output suppressed')
        headers={'Modal-Key':token_id,'Modal-Secret':secret.group()}
        with httpx.Client(base_url=base,timeout=httpx.Timeout(310,connect=30),follow_redirects=False) as client:
            stage='unauthenticated_health';response=client.get('/health')
            assert response.status_code in (401,403)
            report['checks']['unauthenticated_status']=response.status_code
            def request(label,method,path,**kwargs):
                nonlocal stage
                stage=label;start=time.perf_counter()
                response=client.request(method,path,headers=headers,**kwargs)
                report['seconds'][label]=round(time.perf_counter()-start,4)
                report['checks'][label+'_http_status']=response.status_code
                if response.status_code!=200:
                    raise RuntimeError('HTTP contract failed; response body suppressed')
                value=response.json();save(label+'.json',value)
                print(json.dumps({'phase':label,'http_status':response.status_code,'seconds':report['seconds'][label]}),flush=True)
                return value,response
            health,_=request('health','GET','/health')
            assert health['wheel_sha256']==WHEEL_SHA256 and health['gpu_required'] is False
            report['checks']['wheel_sha256']=health['wheel_sha256']
            catalog,_=request('catalog','GET','/v1/catalog')
            report['checks']['catalog_rows']=catalog['connected_catalog_rows']
            cap,_=request('capabilities','GET','/v1/ai/capabilities')
            model=cap['odor_expression']['prediction_model']
            assert model['model_sha256']=='418afec8b8363adb45ea78517b11c3a45de71404cb58b85ffad5c80605c6ce58'
            assert model['prediction_correction']['sha256']=='3feb654573423f98a9e0c4ec7fc079c2ea08195f3da29b91b399aed30e5f02a3'
            assert model['predicted_concepts']==450
            assert cap['stock_mixture_model']['integrated_v54']
            assert cap['unified_product_model']['fine_odor_expression']['prediction_correction']==model['prediction_correction']
            report['checks']['all_selected_models_connected']=True
            report['checks']['fine_output_concepts']=model['predicted_concepts']
            report['checks']['language_backend']=cap['language_model']['backend']
            if not args.assistant_only:
                prediction,_=request('material_prediction','POST','/v1/odor-expressions/predict',
                    json={'ingredient_ids':['dihydromyrcenol','linalyl_acetate','hexyl_acetate'],'brief':'green apple and peach'})
                assert len(prediction['materials'])==3
                for product in ('perfume','body_lotion','body_wash'):
                    payload=json.loads((ROOT/f'.benchmarks/unified_product_v60/api-01/{product}-request.json').read_text(encoding='utf-8'))
                    value,_=request(product,'POST','/v1/applications/unified/predict',json=payload)
                    assert len(value['odor_endpoints'])==146
                    assert value['model']['fine_odor_expression']['prediction_correction']['sha256']==model['prediction_correction']['sha256']
                    assert value['temporal_profile']
                save('before-formula.json',report)
                formula={'brief':'피오니와 청사과 향','max_risk_tier':2,'enable_registry_trace_candidates':True}
                value,_=request('formula','POST','/v1/formulas',json=formula)
                assert value.get('recipe') or value.get('closest_candidate')
                assert value['score_contract']['fine_odor_expression']['prediction_correction_sha256']==model['prediction_correction']['sha256']
                assert value['score_contract']['runtime_minimum_profile_target']==95
                if not value['full_profile_target_met']:assert value['recipe']==[]
                report['checks']['formula']={'status':value['status'],'score':value.get('calculated_profile_similarity'),
                    'target_met':value['full_profile_target_met'],'lines':len(value.get('recipe') or value.get('closest_candidate') or [])}
                repeated,response=request('formula_repeat','POST','/v1/formulas',json=formula)
                assert repeated==value
                report['checks']['formula_cache']=response.headers.get('X-Perfumery-Cache')
            assistant,response=request('assistant','POST','/v1/ai/assistant',json={'message':'머스크 없이 장미 향 바디로션으로 만들어줘'})
            report['checks']['assistant_source']=assistant.get('source')
            verify_assistant_result(assistant)
            report['checks']['assistant_model_executed_and_semantic_guards_passed']=True
            first_calls=response.headers.get('X-Perfumery-LLM-Calls')
            first_cache=response.headers.get('X-Perfumery-Language-Cache')
            assert first_calls in ('0','1') and first_cache in ('hit','miss')
            repeated,response=request('assistant_repeat','POST','/v1/ai/assistant',
                json={'message':'머스크 없이 장미 향 바디로션으로 만들어줘'})
            assert repeated==assistant
            assert response.headers.get('X-Perfumery-LLM-Calls')=='0'
            assert response.headers.get('X-Perfumery-Language-Cache')=='hit'
            report['checks']['assistant_cache']={'first_llm_calls':int(first_calls),
                'first_cache':first_cache,'repeat_llm_calls':0,'repeat_cache':'hit','body_exactly_equal':True}
            if not args.assistant_only:
                from scripts.assess_modal_v63_evidence import assess_formula_equivalence
                previous=ROOT/'.benchmarks/modal_v62_release/remote-01/formula.json'
                if previous.exists():
                    stage='legacy_formula_comparison'
                    comparison=assess_formula_equivalence(json.loads(previous.read_text(encoding='utf-8')),value)
                    save('v62-formula-differences.json',comparison['changes'])
                    report['checks']['v62_recipe_comparison']={k:v for k,v in comparison.items() if k!='changes'}
                    if args.require_legacy_exact:
                        assert comparison['full_response_exact_except_release_hashes']
            report['passed']=True
    except Exception as error:
        report.update(passed=False,error_stage=stage,error_type=type(error).__name__)
    finally:
        if token_id:
            deleted=subprocess.run([sys.executable,'-m','modal','workspace','proxy-tokens','delete',token_id,'-y'],
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45)
            report['temporary_token_revoked']=deleted.returncode==0
        save('report.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    if not report.get('passed') or not report['temporary_token_revoked']:
        raise SystemExit(1)


if __name__=='__main__':main()
